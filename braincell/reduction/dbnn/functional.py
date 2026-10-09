# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure JAX operations shared by DBNN training and inference."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp


Params = Mapping[str, Any]
StepState = Mapping[str, jax.Array]


def _param(params: Params, name: str) -> Any:
    try:
        return params[name]
    except KeyError as error:
        raise KeyError(f"Missing DBNN parameter {name!r}.") from error


def _resolve_dt(params: Params, dt_ms: Any | None) -> Any:
    if dt_ms is not None:
        return dt_ms
    if "dt_ms" not in params:
        raise KeyError("DBNN parameters must contain 'dt_ms' when dt_ms is not passed explicitly.")
    return params["dt_ms"]


def _validate_channel_signs(channel_signs: Any, n_channels: int, dtype: Any) -> jax.Array:
    signs = jnp.asarray(channel_signs, dtype=dtype)
    if signs.shape != (n_channels,):
        raise ValueError(f"channel_signs must have shape ({n_channels},), got {signs.shape}.")
    if not isinstance(signs, jax.core.Tracer) and not bool(jnp.all((signs == 1.0) | (signs == -1.0))):
        raise ValueError("channel_signs must contain only +1 for E and -1 for I channels.")
    return signs


def build_kernels(
    params: Params,
    n_time: int,
    *,
    channel_signs: Any,
    dt_ms: Any | None = None,
) -> jax.Array:
    """Build causal double-exponential kernels for all channels.

    Parameters
    ----------
    params : mapping of str to array-like
        DBNN parameters containing ``tau_rise``, ``tau_decay``, and ``omega``.
    n_time : int
        Number of time samples in each kernel.
    channel_signs : array-like
        Layout-derived vector containing ``+1`` for E and ``-1`` for I.
    dt_ms : array-like, optional
        Sampling interval in milliseconds. When omitted, use ``params["dt_ms"]``.

    Returns
    -------
    jax.Array
        Kernels with shape ``(channels, n_time)``.
    """
    if not isinstance(n_time, int) or isinstance(n_time, bool) or n_time <= 0:
        raise ValueError(f"n_time must be a positive integer, got {n_time!r}.")
    tau_rise = jnp.asarray(_param(params, "tau_rise"))
    tau_decay = jnp.asarray(_param(params, "tau_decay"))
    omega = jnp.asarray(_param(params, "omega"))
    signs = _validate_channel_signs(channel_signs, int(omega.shape[0]), omega.dtype)
    time_ms = jnp.arange(n_time, dtype=omega.dtype) * jnp.asarray(_resolve_dt(params, dt_ms), dtype=omega.dtype)
    return (
        signs[:, None]
        * omega[:, None]
        * (1.0 - jnp.exp(-time_ms[None, :] / tau_rise[:, None]))
        * jnp.exp(-time_ms[None, :] / tau_decay[:, None])
    )


def unpack_quadratic_weight(weight_upper: Any, n_channels: int) -> jax.Array:
    """Expand packed interactions into a strict upper-triangular matrix.

    Parameters
    ----------
    weight_upper : array-like
        Packed weights in ``jax.numpy.triu_indices(n_channels, k=1)`` order.
    n_channels : int
        Number of DBNN input channels.

    Returns
    -------
    jax.Array
        Matrix with a zero diagonal and zero strict lower triangle.

    Raises
    ------
    ValueError
        If the channel count or packed shape is invalid.
    """
    if not isinstance(n_channels, int) or isinstance(n_channels, bool) or n_channels <= 0:
        raise ValueError(f"n_channels must be a positive integer, got {n_channels!r}.")
    weight_upper = jnp.asarray(weight_upper)
    expected = n_channels * (n_channels - 1) // 2
    if weight_upper.shape != (expected,):
        raise ValueError(f"weight_upper must have shape ({expected},), got {weight_upper.shape}.")
    return (
        jnp.zeros((n_channels, n_channels), dtype=weight_upper.dtype)
        .at[jnp.triu_indices(n_channels, k=1)]
        .set(weight_upper)
    )


def causal_fft_convolve(inputs: Any, kernels: Any) -> jax.Array:
    """Convolve channel event sequences with causal kernels using FFTs.

    Parameters
    ----------
    inputs : array-like
        Event tensor with shape ``(batch, channels, time)``.
    kernels : array-like
        Kernel tensor with shape ``(channels, time)``.

    Returns
    -------
    jax.Array
        Linear convolution truncated to the input duration, with the same shape
        as ``inputs``.

    Raises
    ------
    ValueError
        If input and kernel shapes are incompatible.
    """
    inputs = jnp.asarray(inputs)
    kernels = jnp.asarray(kernels)
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be three-dimensional, got shape {inputs.shape}.")
    if kernels.shape != inputs.shape[1:]:
        raise ValueError(f"kernels must have shape {inputs.shape[1:]}, got {kernels.shape}.")
    n_time = inputs.shape[2]
    fft_size = 1 << (2 * n_time - 2).bit_length()
    input_fft = jnp.fft.rfft(inputs, n=fft_size, axis=2)
    kernel_fft = jnp.fft.rfft(kernels, n=fft_size, axis=1)
    return jnp.fft.irfft(input_fft * kernel_fft[None, :, :], n=fft_size, axis=2)[..., :n_time]


def compute_drive(filtered: Any, params: Params, bias: Any | None = None) -> jax.Array:
    """Compute the linear and strict pairwise DBNN voltage drive.

    Parameters
    ----------
    filtered : array-like
        Filtered channel responses with shape ``(batch, time, channels)``.
    params : mapping of str to array-like
        Parameters containing packed ``quadratic_weight_upper`` and optionally
        ``bias``.
    bias : array-like, optional
        Explicit voltage baseline. When omitted, use ``params["bias"]``.

    Returns
    -------
    jax.Array
        Open-loop voltage with shape ``(batch, time)``.
    """
    filtered = jnp.asarray(filtered)
    if filtered.ndim != 3:
        raise ValueError(f"filtered must be three-dimensional, got shape {filtered.shape}.")
    n_channels = filtered.shape[2]
    weight = unpack_quadratic_weight(_param(params, "quadratic_weight_upper"), n_channels)
    linear = jnp.sum(filtered, axis=2)
    quadratic = jnp.sum((filtered @ weight) * filtered, axis=2)
    baseline = _param(params, "bias") if bias is None else bias
    return linear + quadratic + jnp.asarray(baseline, dtype=filtered.dtype)


def dbnn_forward(
    params: Params,
    inputs: Any,
    *,
    channel_signs: Any,
    bias: Any | None = None,
    dt_ms: Any | None = None,
) -> jax.Array:
    """Evaluate the authoritative open-loop DBNN training forward pass.

    Parameters
    ----------
    params : mapping of str to array-like
        DBNN kernel and packed bilinear parameters.
    inputs : array-like
        Non-negative event tensor with shape ``(batch, channels, time)``.
    channel_signs : array-like
        Layout-derived vector containing ``+1`` for E and ``-1`` for I.
    bias : array-like, optional
        Explicit voltage baseline. When omitted, use ``params["bias"]``.
    dt_ms : array-like, optional
        Sampling interval in milliseconds. When omitted, use ``params["dt_ms"]``.

    Returns
    -------
    jax.Array
        Open-loop voltage with shape ``(batch, time)``.
    """
    inputs = jnp.asarray(inputs)
    kernels = build_kernels(params, inputs.shape[2], channel_signs=channel_signs, dt_ms=dt_ms)
    filtered = causal_fft_convolve(inputs, kernels)
    return compute_drive(jnp.transpose(filtered, (0, 2, 1)), params, bias)


def _dbnn_f_drive(params: Params, inputs: Any, channel_signs: Any) -> jax.Array:
    """Compute FFT-based open-loop voltage using ``params['dt_ms']``."""
    return dbnn_forward(params, inputs, channel_signs=channel_signs)


def _apply_gif_closed_loop(drive: Any, params: Params) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply threshold adaptation and an explicitly configured optional reset."""
    drive = jnp.asarray(drive)
    dt_ms = jnp.asarray(_resolve_dt(params, None), dtype=drive.dtype)
    threshold_tau = jnp.asarray(_param(params, "threshold_tau_ms"), dtype=drive.dtype)
    threshold_alpha = jnp.exp(-dt_ms / jnp.maximum(jnp.abs(threshold_tau), 1e-4))
    threshold_increment = jnp.asarray(_param(params, "threshold_increment_mv"), dtype=drive.dtype)
    resting_threshold = jnp.asarray(_param(params, "v_th"), dtype=drive.dtype)
    reset_enabled = "reset_amp" in params
    if reset_enabled:
        reset_amp = jnp.asarray(_param(params, "reset_amp"), dtype=drive.dtype)
        reset_tau = jnp.asarray(_param(params, "tau_reset"), dtype=drive.dtype)
        reset_alpha = jnp.exp(-dt_ms / jnp.maximum(jnp.abs(reset_tau), 1e-4))

    def step(state, drive_t):
        reset_state, threshold_state, previous_voltage, previous_threshold = state
        voltage_t = drive_t + reset_state
        threshold_t = resting_threshold + threshold_state
        spike_t = (voltage_t >= threshold_t) & (previous_voltage < previous_threshold)
        next_reset = (
            reset_alpha * reset_state - reset_amp * spike_t.astype(drive.dtype) if reset_enabled else reset_state
        )
        next_threshold = threshold_alpha * threshold_state + threshold_increment * spike_t.astype(drive.dtype)
        next_state = (next_reset, next_threshold, voltage_t, threshold_t)
        return next_state, (voltage_t, threshold_t, spike_t)

    initial_state = (
        jnp.zeros(drive.shape[0], dtype=drive.dtype),
        jnp.zeros(drive.shape[0], dtype=drive.dtype),
        drive[:, 0],
        jnp.broadcast_to(resting_threshold, drive.shape[:1]),
    )
    _, (voltage_tb, threshold_tb, spike_tb) = jax.lax.scan(step, initial_state, drive.T)
    return voltage_tb.T, threshold_tb.T, spike_tb.T


def threshold_crossings(
    voltage: Any,
    threshold: Any,
    *,
    previous_voltage: Any | None = None,
) -> jax.Array:
    """Read spikes from threshold up-crossings along the final axis.

    Parameters
    ----------
    voltage : array-like
        Voltage traces whose final axis is time.
    threshold : array-like
        Scalar threshold broadcastable to ``voltage`` without its time axis.

    Returns
    -------
    jax.Array
        Boolean spike traces with the same shape as ``voltage``.
    """
    voltage = jnp.asarray(voltage)
    threshold = jnp.asarray(threshold, dtype=voltage.dtype)
    above = voltage >= threshold
    if previous_voltage is None:
        previous_above = jnp.zeros_like(above[..., :1])
    else:
        previous_above = jnp.asarray(previous_voltage, dtype=voltage.dtype) >= threshold
        previous_above = jnp.broadcast_to(previous_above, above.shape[:-1])[..., None]
    previous = jnp.concatenate((previous_above, above[..., :-1]), axis=-1)
    return above & ~previous


def dbnn_f(params: Params, inputs: Any, *, channel_signs: Any) -> jax.Array:
    """Evaluate signed open-loop DBNN voltage with causal FFT convolution."""
    return _dbnn_f_drive(params, inputs, channel_signs)


def _recurrent_filter(params: Params, inputs: Any, channel_signs: Any) -> jax.Array:
    inputs = jnp.asarray(inputs)
    dt_ms = jnp.asarray(_resolve_dt(params, None), dtype=inputs.dtype)
    tau_decay = jnp.asarray(_param(params, "tau_decay"), dtype=inputs.dtype)
    tau_rise = jnp.asarray(_param(params, "tau_rise"), dtype=inputs.dtype)
    multipliers = jnp.stack(
        (
            jnp.exp(-dt_ms / tau_decay),
            jnp.exp(-dt_ms * (1.0 / tau_rise + 1.0 / tau_decay)),
        )
    )
    inputs_tbc = jnp.transpose(inputs, (2, 0, 1))
    omega = jnp.asarray(_param(params, "omega"), dtype=inputs.dtype)
    signs = _validate_channel_signs(channel_signs, int(omega.shape[0]), omega.dtype)

    def step(states, events):
        states = events[None, :, :] + multipliers[:, None, :] * states
        filtered = signs[None, :] * omega[None, :] * (states[0] - states[1])
        return states, filtered

    initial = jnp.zeros((2, inputs.shape[0], inputs.shape[1]), dtype=inputs.dtype)
    _, filtered_tbc = jax.lax.scan(step, initial, inputs_tbc)
    return jnp.transpose(filtered_tbc, (1, 0, 2))


def _dbnn_r_drive(params: Params, inputs: Any, channel_signs: Any) -> jax.Array:
    """Compute recurrent open-loop voltage using ``params['dt_ms']``."""
    filtered = _recurrent_filter(params, inputs, channel_signs)

    def step(_, filtered_t):
        drive_t = compute_drive(filtered_t[:, None, :], params)[:, 0]
        return None, drive_t

    _, drive_tb = jax.lax.scan(step, None, jnp.transpose(filtered, (1, 0, 2)))
    return drive_tb.T


def dbnn_r(params: Params, inputs: Any, *, channel_signs: Any) -> jax.Array:
    """Evaluate signed open-loop DBNN voltage with double-exponential recurrences."""
    return _dbnn_r_drive(params, inputs, channel_signs)


def _dbnn_gif_f_state(
    params: Params,
    inputs: Any,
    *,
    channel_signs: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate FFT DBNN-GIF voltage, dynamic threshold, and spikes."""
    return _apply_gif_closed_loop(_dbnn_f_drive(params, inputs, channel_signs), params)


def _dbnn_gif_r_state(
    params: Params,
    inputs: Any,
    *,
    channel_signs: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate recurrent DBNN-GIF voltage, dynamic threshold, and spikes."""
    return _apply_gif_closed_loop(_dbnn_r_drive(params, inputs, channel_signs), params)


def dbnn_step(
    params: Params,
    state: StepState,
    events: Any,
    *,
    channel_signs: Any,
    dt_ms: Any | None = None,
) -> tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array]]:
    """Advance one population step with the same recurrence as :func:`dbnn_r`.

    Parameters
    ----------
    params : mapping of str to array-like
        DBNN model parameters.
    state : mapping of str to jax.Array
        ``decay``, ``rise_decay``, and ``previous_voltage`` arrays from the
        previous step.
    events : array-like
        Current events with shape ``(population, channels)``.
    channel_signs : array-like
        Layout-derived vector containing ``+1`` for E and ``-1`` for I.
    dt_ms : array-like, optional
        Sampling interval in milliseconds. When omitted, use ``params["dt_ms"]``.

    Returns
    -------
    state : dict of str to jax.Array
        Updated recurrence states.
    output : tuple of jax.Array
        Current ``(voltage, spike)`` arrays, each with shape ``(population,)``.
    """
    events = jnp.asarray(events)
    if events.ndim != 2:
        raise ValueError(f"events must be two-dimensional, got shape {events.shape}.")
    interval = jnp.asarray(_resolve_dt(params, dt_ms), dtype=events.dtype)
    tau_decay = jnp.asarray(_param(params, "tau_decay"), dtype=events.dtype)
    tau_rise = jnp.asarray(_param(params, "tau_rise"), dtype=events.dtype)
    decay = events + jnp.exp(-interval / tau_decay) * state["decay"]
    rise_decay = events + jnp.exp(-interval * (1.0 / tau_rise + 1.0 / tau_decay)) * state["rise_decay"]
    omega = jnp.asarray(_param(params, "omega"), dtype=events.dtype)
    signs = _validate_channel_signs(channel_signs, int(omega.shape[0]), omega.dtype)
    filtered = signs * omega * (decay - rise_decay)
    drive = compute_drive(filtered[:, None, :], params)[:, 0]
    voltage = drive
    threshold = jnp.asarray(_param(params, "v_th"), dtype=events.dtype)
    previous_voltage = jnp.asarray(state["previous_voltage"], dtype=events.dtype)
    spike = (voltage >= threshold) & (previous_voltage < threshold)
    next_state = {"decay": decay, "rise_decay": rise_decay, "previous_voltage": voltage}
    return next_state, (voltage, spike)


def _dbnn_gif_step(
    params: Params,
    state: StepState,
    events: Any,
    *,
    channel_signs: Any,
    dt_ms: Any | None = None,
) -> tuple[dict[str, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Advance one DBNN-GIF population step."""
    events = jnp.asarray(events)
    if events.ndim != 2:
        raise ValueError(f"events must be two-dimensional, got shape {events.shape}.")
    interval = jnp.asarray(_resolve_dt(params, dt_ms), dtype=events.dtype)
    tau_decay = jnp.asarray(_param(params, "tau_decay"), dtype=events.dtype)
    tau_rise = jnp.asarray(_param(params, "tau_rise"), dtype=events.dtype)
    decay = events + jnp.exp(-interval / tau_decay) * state["decay"]
    rise_decay = events + jnp.exp(-interval * (1.0 / tau_rise + 1.0 / tau_decay)) * state["rise_decay"]
    omega = jnp.asarray(_param(params, "omega"), dtype=events.dtype)
    signs = _validate_channel_signs(channel_signs, int(omega.shape[0]), omega.dtype)
    filtered = signs * omega * (decay - rise_decay)
    drive = compute_drive(filtered[:, None, :], params)[:, 0]

    reset_enabled = "reset_amp" in params
    reset_state = jnp.asarray(state.get("reset", jnp.zeros_like(drive)), dtype=events.dtype)
    threshold_state = jnp.asarray(state["threshold"], dtype=events.dtype)
    voltage = drive + reset_state
    threshold = jnp.asarray(_param(params, "v_th"), dtype=events.dtype) + threshold_state
    threshold_increment = jnp.asarray(_param(params, "threshold_increment_mv"), dtype=events.dtype)
    previous_voltage = jnp.asarray(state["previous_voltage"], dtype=events.dtype)
    previous_threshold = jnp.asarray(state["previous_threshold"], dtype=events.dtype)
    spike = (voltage >= threshold) & (previous_voltage < previous_threshold)
    threshold_alpha = jnp.exp(-interval / jnp.maximum(jnp.abs(_param(params, "threshold_tau_ms")), 1e-4))
    next_state = {
        "decay": decay,
        "rise_decay": rise_decay,
        "threshold": threshold_alpha * threshold_state + threshold_increment * spike.astype(events.dtype),
        "previous_voltage": voltage,
        "previous_threshold": threshold,
    }
    if reset_enabled:
        reset_alpha = jnp.exp(-interval / jnp.maximum(jnp.abs(_param(params, "tau_reset")), 1e-4))
        next_state["reset"] = reset_alpha * reset_state - jnp.asarray(
            _param(params, "reset_amp"), dtype=events.dtype
        ) * spike.astype(events.dtype)
    return next_state, (voltage, threshold, spike)


def masked_mse(predictions: Any, targets: Any, mask: Any | None = None) -> jax.Array:
    """Compute mean squared error over selected samples."""
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must share a shape, got {predictions.shape} and {targets.shape}.")
    selected = jnp.ones(predictions.shape, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    if selected.shape != predictions.shape:
        raise ValueError(f"mask must have shape {predictions.shape}, got {selected.shape}.")
    count = jnp.sum(selected)
    total = jnp.sum(jnp.where(selected, jnp.square(predictions - targets), 0.0))
    return jnp.where(count > 0, total / count, jnp.asarray(jnp.nan, dtype=total.dtype))


def compute_metrics(predictions: Any, targets: Any, mask: Any | None = None) -> dict[str, jax.Array]:
    """Compute masked voltage-regression metrics, preserving undefined results."""
    predictions = jnp.asarray(predictions)
    targets = jnp.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError(f"predictions and targets must share a shape, got {predictions.shape} and {targets.shape}.")
    selected = jnp.ones(predictions.shape, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    if selected.shape != predictions.shape:
        raise ValueError(f"mask must have shape {predictions.shape}, got {selected.shape}.")
    count = jnp.sum(selected)
    denominator = jnp.maximum(count, 1)
    error = jnp.where(selected, predictions - targets, 0.0)
    mse = jnp.sum(jnp.square(error)) / denominator
    mae = jnp.sum(jnp.abs(error)) / denominator
    target_mean = jnp.sum(jnp.where(selected, targets, 0.0)) / denominator
    prediction_mean = jnp.sum(jnp.where(selected, predictions, 0.0)) / denominator
    target_centered = jnp.where(selected, targets - target_mean, 0.0)
    prediction_centered = jnp.where(selected, predictions - prediction_mean, 0.0)
    target_ss = jnp.sum(jnp.square(target_centered))
    prediction_ss = jnp.sum(jnp.square(prediction_centered))
    covariance = jnp.sum(target_centered * prediction_centered)
    nan = jnp.asarray(jnp.nan, dtype=mse.dtype)
    valid = count > 0
    return {
        "mse": jnp.where(valid, mse, nan),
        "rmse": jnp.where(valid, jnp.sqrt(mse), nan),
        "mae": jnp.where(valid, mae, nan),
        "pearson": jnp.where(
            valid & (target_ss > 0) & (prediction_ss > 0),
            covariance / jnp.sqrt(target_ss * prediction_ss),
            nan,
        ),
        "variance_explained": jnp.where(valid & (target_ss > 0), 1.0 - jnp.sum(jnp.square(error)) / target_ss, nan),
        "valid_count": count,
    }


__all__ = [
    "build_kernels",
    "causal_fft_convolve",
    "compute_drive",
    "compute_metrics",
    "dbnn_f",
    "dbnn_forward",
    "dbnn_r",
    "dbnn_step",
    "masked_mse",
    "threshold_crossings",
    "unpack_quadratic_weight",
]
