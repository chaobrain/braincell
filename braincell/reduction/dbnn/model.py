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

"""Deployable sequence and stateful DBNN models."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn.functional import (
    dbnn_f,
    _dbnn_gif_f_state,
    _dbnn_gif_r_state,
    _dbnn_gif_step,
    dbnn_r,
    dbnn_step,
    threshold_crossings,
)
from braincell.reduction.dbnn.layout import ChannelLayout


_PARAMETER_NAMES = (
    "tau_rise",
    "tau_decay",
    "omega",
    "quadratic_weight_upper",
    "bias",
    "v_th",
)

_GIF_PARAMETER_NAMES = (
    "threshold_increment_mv",
    "threshold_tau_ms",
)
_RESET_PARAMETER_NAMES = ("reset_amp", "tau_reset")


@dataclass(frozen=True)
class SpikeAlignmentEvidence:
    """Store retained validation traces for threshold-only spike realignment."""

    teacher_voltage_mv: np.ndarray
    raw_spike_times_ms: tuple[np.ndarray, ...]
    dt_ms: float
    match_window_ms: float
    layout_fingerprint: str
    dynamics_fingerprint: str
    validation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        voltage = np.array(self.teacher_voltage_mv, dtype=np.float32, copy=True)
        if voltage.ndim != 2 or voltage.shape[0] < 1 or voltage.shape[1] < 2:
            raise ValueError("teacher_voltage_mv must have shape (traces>=1, time>=2).")
        if not np.isfinite(voltage).all():
            raise ValueError("teacher_voltage_mv must contain only finite values.")
        spikes = tuple(np.array(row, dtype=np.float32, copy=True).reshape(-1) for row in self.raw_spike_times_ms)
        if len(spikes) != voltage.shape[0]:
            raise ValueError("raw_spike_times_ms must contain one row per teacher voltage trace.")
        duration_ms = (voltage.shape[1] - 1) * float(self.dt_ms)
        if any(
            not np.isfinite(row).all() or np.any(row < 0.0) or np.any(row > duration_ms)
            for row in spikes
        ):
            raise ValueError("Raw spike times must be finite and lie within the validation duration.")
        dt_ms = float(self.dt_ms)
        match_window_ms = float(self.match_window_ms)
        if not np.isfinite(dt_ms) or dt_ms <= 0:
            raise ValueError("dt_ms must be finite and positive.")
        if not np.isfinite(match_window_ms) or match_window_ms < 0:
            raise ValueError("match_window_ms must be finite and non-negative.")
        if not isinstance(self.layout_fingerprint, str) or not self.layout_fingerprint:
            raise ValueError("layout_fingerprint must be a non-empty string.")
        if not isinstance(self.dynamics_fingerprint, str) or not self.dynamics_fingerprint:
            raise ValueError("dynamics_fingerprint must be a non-empty string.")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in self.validation_seeds):
            raise TypeError("validation_seeds must contain integers.")
        seeds = tuple(int(seed) for seed in self.validation_seeds)
        if len(seeds) != voltage.shape[0] or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("validation_seeds must contain one unique non-negative seed per trace.")
        voltage.setflags(write=False)
        for row in spikes:
            row.setflags(write=False)
        object.__setattr__(self, "teacher_voltage_mv", voltage)
        object.__setattr__(self, "raw_spike_times_ms", spikes)
        object.__setattr__(self, "dt_ms", dt_ms)
        object.__setattr__(self, "match_window_ms", match_window_ms)
        object.__setattr__(self, "validation_seeds", seeds)


def _time_to_ms(dt: Any) -> float:
    if not hasattr(dt, "to_decimal"):
        raise TypeError("dt must be a brainunit time quantity, for example 0.1 * brainunit.ms.")
    try:
        value = float(np.asarray(dt.to_decimal(u.ms)).reshape(()))
    except Exception as error:
        raise TypeError("dt must be convertible to milliseconds.") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"dt must be finite and positive, got {dt!r}.")
    return value


def detect_spike_times(voltage_mv: Any, threshold_mv: float) -> np.ndarray:
    """Detect threshold up-crossings in a voltage trace."""
    voltage = np.asarray(voltage_mv)
    if voltage.ndim != 1 or voltage.size == 0:
        raise ValueError(f"voltage_mv must be a non-empty one-dimensional trace, got shape {voltage.shape}.")
    return np.flatnonzero(np.asarray(threshold_crossings(voltage, threshold_mv))).astype(np.int32)


def align_spike_times(
    spike_times_ms: Any,
    *,
    offset_ms: float,
    duration_ms: float,
) -> tuple[np.ndarray, ...]:
    """Subtract one calibrated offset and retain spikes in ``[0, duration]``."""
    offset = float(offset_ms)
    duration = float(duration_ms)
    if not np.isfinite(offset):
        raise ValueError("spike time offset must be finite.")
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("spike duration must be finite and non-negative.")
    rows = tuple(spike_times_ms)
    result = []
    for row in rows:
        values = np.asarray(row, dtype=float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("raw spike times must be finite.")
        aligned = values - offset
        scale = max(1.0, abs(offset), duration, float(np.max(np.abs(values), initial=0.0)))
        tolerance = 8.0 * np.finfo(np.float32).eps * scale
        aligned[(aligned < 0.0) & (aligned >= -tolerance)] = 0.0
        aligned[(aligned > duration) & (aligned <= duration + tolerance)] = duration
        result.append(aligned[(aligned >= 0.0) & (aligned <= duration)])
    return tuple(result)


class DBNN(brainstate.nn.Module):
    """Represent a trained two-layer dendritic bilinear neural network.

    Parameters
    ----------
    layout : ChannelLayout
        Fixed channel semantics derived from a detailed BrainCell teacher.
    mode : {"f", "r"}, optional
        FFT or recurrent complete-sequence backend.
    dt : brainunit.Quantity, optional
        Sampling interval. When omitted, use ``brainstate.environ.get_dt()``.
    name : str, optional
        BrainState module name.
    """

    _parameter_names = _PARAMETER_NAMES

    def __init__(
        self,
        layout: ChannelLayout,
        *,
        mode: str = "f",
        input_sign_mode: str = "channel_type",
        dt: Any | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if not isinstance(layout, ChannelLayout):
            raise TypeError(f"layout must be a ChannelLayout, got {type(layout).__name__!r}.")
        self.layout = layout
        self.dt = brainstate.environ.get_dt() if dt is None else dt
        self.dt_ms = _time_to_ms(self.dt)
        if input_sign_mode not in {"none", "channel_type"}:
            raise ValueError(f"input_sign_mode must be 'none' or 'channel_type', got {input_sign_mode!r}.")
        self.input_sign_mode = input_sign_mode
        self.mode = "f"
        self.set_mode(mode)
        pair_count = self.n_channels * (self.n_channels - 1) // 2
        self.tau_rise = brainstate.ParamState(jnp.full((self.n_channels,), 5.0, dtype=jnp.float32))
        self.tau_decay = brainstate.ParamState(jnp.full((self.n_channels,), 20.0, dtype=jnp.float32))
        self.omega = brainstate.ParamState(jnp.full((self.n_channels,), 2.0, dtype=jnp.float32))
        self.quadratic_weight_upper = brainstate.ParamState(jnp.zeros((pair_count,), dtype=jnp.float32))
        self.bias = brainstate.ParamState(jnp.asarray(-70.0, dtype=jnp.float32))
        self.v_th = brainstate.ParamState(jnp.asarray(-55.0, dtype=jnp.float32))

    @property
    def n_channels(self) -> int:
        """Return the fixed channel count from the authoritative layout."""
        return self.layout.n_channels

    @property
    def channel_signs(self) -> jax.Array:
        """Return the explicit input encoding selected for this model."""
        if self.input_sign_mode == "none":
            return jnp.ones((self.n_channels,), dtype=jnp.float32)
        return jnp.asarray(
            [1.0 if self.layout.polarity(channel_id) == "E" else -1.0 for channel_id in range(self.n_channels)],
            dtype=jnp.float32,
        )

    def set_mode(self, mode: str) -> None:
        """Select the FFT or recurrent complete-sequence backend."""
        if mode not in {"f", "r"}:
            raise ValueError(f"mode must be 'f' or 'r', got {mode!r}.")
        self.mode = mode

    def get_params(self) -> dict[str, jax.Array]:
        """Return the model parameters and numerical sampling interval."""
        params = {name: getattr(self, name).value for name in self._parameter_names}
        params["dt_ms"] = jnp.asarray(self.dt_ms, dtype=jnp.float32)
        return params

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Replace every model parameter after strict validation."""
        expected = set(self._parameter_names)
        supplied = set(params)
        extra = supplied - expected - {"dt_ms"}
        missing = expected - supplied
        if missing or extra:
            raise KeyError(f"DBNN parameter keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}.")
        expected_shapes = {
            "tau_rise": (self.n_channels,),
            "tau_decay": (self.n_channels,),
            "omega": (self.n_channels,),
            "quadratic_weight_upper": (self.n_channels * (self.n_channels - 1) // 2,),
            "bias": (),
            "v_th": (),
            "threshold_increment_mv": (),
            "threshold_tau_ms": (),
            "reset_amp": (),
            "tau_reset": (),
        }
        converted = {}
        for name in self._parameter_names:
            value = jnp.asarray(params[name], dtype=jnp.float32)
            if value.shape != expected_shapes[name]:
                raise ValueError(f"Parameter {name!r} must have shape {expected_shapes[name]}, got {value.shape}.")
            if not bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(f"Parameter {name!r} contains non-finite values.")
            converted[name] = value
        if bool(jnp.any(converted["tau_rise"] <= 0)) or bool(jnp.any(converted["tau_decay"] <= 0)):
            raise ValueError("DBNN time constants must be positive.")
        if bool(jnp.any(converted["tau_rise"] > 100.0)) or bool(jnp.any(converted["tau_decay"] > 200.0)):
            raise ValueError("DBNN tau_rise and tau_decay exceed the supported training ranges.")
        if bool(jnp.any(converted["omega"] < 0.0)) or bool(jnp.any(converted["omega"] > 10.0)):
            raise ValueError("DBNN omega must lie in [0, 10].")
        if "threshold_tau_ms" in converted and bool(converted["threshold_tau_ms"] <= 0):
            raise ValueError("DBNN GIF threshold time constant must be positive.")
        if "threshold_increment_mv" in converted and bool(converted["threshold_increment_mv"] < 0):
            raise ValueError("DBNN GIF threshold increment must be non-negative.")
        if "tau_reset" in converted and bool(converted["tau_reset"] <= 0):
            raise ValueError("DBNN GIF reset time constant must be positive.")
        if "reset_amp" in converted and bool(converted["reset_amp"] < 0):
            raise ValueError("DBNN GIF reset amplitude must be non-negative.")
        if "dt_ms" in params and not np.isclose(float(np.asarray(params["dt_ms"])), self.dt_ms):
            raise ValueError(f"Parameter dt_ms={params['dt_ms']} is incompatible with model dt_ms={self.dt_ms}.")
        for name, value in converted.items():
            getattr(self, name).value = value

    def validate_inputs(self, inputs: Any) -> jax.Array:
        """Validate complete non-negative event traces against the channel layout."""
        array = jnp.asarray(inputs)
        if array.ndim != 3:
            raise ValueError(f"inputs must have shape (batch, channels, time), got {array.shape}.")
        if array.shape[1] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {array.shape[1]}.")
        self._validate_event_values(array, channel_axis=1, field="inputs")
        return array

    def _validate_event_values(self, values: jax.Array, *, channel_axis: int, field: str) -> None:
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError(f"{field} must use a floating dtype, got {values.dtype}.")
        if isinstance(values, jax.core.Tracer):
            return
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError(f"{field} contain non-finite values.")
        if bool(jnp.any(values < 0)):
            raise ValueError(f"DBNN {field} must be non-negative event counts or amplitudes.")
        reduction_axes = tuple(axis for axis in range(values.ndim) if axis != channel_axis)
        maximum = np.asarray(jnp.max(values, axis=reduction_axes))
        upper = np.asarray([self.layout.training_range(channel_id)[1] for channel_id in range(self.n_channels)])
        outside = np.flatnonzero(maximum > upper)
        if len(outside):
            warnings.warn(
                f"DBNN input channels {outside.tolist()} exceed their declared training ranges.",
                RuntimeWarning,
                stacklevel=3,
            )

    def predict(self, inputs: Any, *, mode: str | None = None) -> dict[str, jax.Array]:
        """Predict open-loop voltage and threshold up-crossing spikes."""
        inputs = self.validate_inputs(inputs)
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in {"f", "r"}:
            raise ValueError(f"mode must be 'f' or 'r', got {selected_mode!r}.")
        voltage = (dbnn_f if selected_mode == "f" else dbnn_r)(
            self.get_params(),
            inputs,
            channel_signs=self.channel_signs,
        )
        spike = threshold_crossings(voltage, self.v_th.value, previous_voltage=self.bias.value)
        return {"voltage": voltage, "spike": spike}

    def init_state(self, batch_size: int | None = None) -> None:
        """Allocate independent recurrent state for each population member."""
        size = 1 if batch_size is None else batch_size
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"batch_size must be a positive integer or None, got {batch_size!r}.")
        channel_shape = (size, self.n_channels)
        self.decay = brainstate.HiddenState(jnp.zeros(channel_shape, dtype=jnp.float32))
        self.rise_decay = brainstate.HiddenState(jnp.zeros(channel_shape, dtype=jnp.float32))
        self.previous_voltage = brainstate.HiddenState(jnp.full((size,), self.bias.value, dtype=jnp.float32))
        self.voltage = brainstate.ShortTermState(jnp.full((size,), self.bias.value, dtype=jnp.float32))
        self.spike = brainstate.ShortTermState(jnp.zeros((size,), dtype=bool))

    def reset_state(self, batch_size: int | None = None) -> None:
        """Reset recurrent and output states, rebuilding them for a new size."""
        size = 1 if batch_size is None else batch_size
        expected = (size, self.n_channels)
        if not hasattr(self, "decay") or self.decay.value.shape != expected:
            self.init_state(batch_size)
            return
        self.decay.value = jnp.zeros(expected, dtype=jnp.float32)
        self.rise_decay.value = jnp.zeros(expected, dtype=jnp.float32)
        self.previous_voltage.value = jnp.full((size,), self.bias.value, dtype=jnp.float32)
        self.voltage.value = jnp.full((size,), self.bias.value, dtype=jnp.float32)
        self.spike.value = jnp.zeros((size,), dtype=bool)

    def update(self, events: Any) -> tuple[jax.Array, jax.Array]:
        """Advance one recurrent step for a population event matrix."""
        if self.mode != "r":
            raise RuntimeError("DBNN.update() requires mode='r'.")
        if not hasattr(self, "decay"):
            raise RuntimeError("DBNN state is not initialized; call init_state(batch_size) first.")
        events = jnp.asarray(events)
        if events.shape != self.decay.value.shape:
            raise ValueError(f"events must have shape {self.decay.value.shape}, got {events.shape}.")
        self._validate_event_values(events, channel_axis=1, field="events")
        state = {
            "decay": self.decay.value,
            "rise_decay": self.rise_decay.value,
            "previous_voltage": self.previous_voltage.value,
        }
        state, (voltage, spike) = dbnn_step(
            self.get_params(),
            state,
            events,
            channel_signs=self.channel_signs,
        )
        self.decay.value = state["decay"]
        self.rise_decay.value = state["rise_decay"]
        self.previous_voltage.value = state["previous_voltage"]
        self.voltage.value = voltage
        self.spike.value = spike
        return voltage, spike

    def save(
        self,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        spike_alignment_path: str | Path | None = None,
    ) -> None:
        """Save a DBNN deployment asset and optional spike-alignment sidecar."""
        from braincell.reduction.dbnn.asset import save_model

        save_model(path, self, metadata=metadata, spike_alignment_path=spike_alignment_path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        mode: str | None = None,
        expected_layout: ChannelLayout | None = None,
        expected_source_fingerprint: str | None = None,
    ) -> "DBNN":
        """Load a parameter-only DBNN asset."""
        from braincell.reduction.dbnn.asset import load_model

        model = load_model(
            path,
            mode=mode,
            expected_layout=expected_layout,
            expected_source_fingerprint=expected_source_fingerprint,
        )
        if cls is DBNNGIF and not isinstance(model, DBNNGIF):
            raise ValueError("The model asset does not contain a DBNNGIF model.")
        return model


class DBNNGIF(DBNN):
    """Extend DBNN with a spike-triggered dynamic threshold."""

    _parameter_names = _PARAMETER_NAMES + _GIF_PARAMETER_NAMES

    def __init__(
        self,
        layout: ChannelLayout,
        *,
        mode: str = "f",
        input_sign_mode: str = "channel_type",
        dt: Any | None = None,
        name: str | None = None,
    ):
        super().__init__(layout, mode=mode, input_sign_mode=input_sign_mode, dt=dt, name=name)
        self._parameter_names = _PARAMETER_NAMES + _GIF_PARAMETER_NAMES
        self.threshold_increment_mv = brainstate.ParamState(jnp.asarray(5.0, dtype=jnp.float32))
        self.threshold_tau_ms = brainstate.ParamState(jnp.asarray(50.0, dtype=jnp.float32))
        self.spike_time_offset_ms = 0.0
        self.spike_alignment_evidence: SpikeAlignmentEvidence | None = None

    def set_spike_time_offset(self, offset_ms: float) -> "DBNNGIF":
        """Set the validation-calibrated spike latency correction in milliseconds."""
        offset = float(offset_ms)
        if not np.isfinite(offset):
            raise ValueError("spike_time_offset_ms must be finite.")
        self.spike_time_offset_ms = offset
        return self

    @property
    def reset_enabled(self) -> bool:
        """Return whether optional exponential voltage reset is configured."""
        return "reset_amp" in self._parameter_names

    def enable_reset(self, amplitude_mv: float = 15.0, tau_ms: float = 10.0) -> "DBNNGIF":
        """Enable optional spike-triggered exponential voltage reset."""
        amplitude = float(amplitude_mv)
        tau = float(tau_ms)
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError("reset amplitude must be finite and non-negative.")
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("reset time constant must be finite and positive.")
        if not self.reset_enabled:
            self.reset_amp = brainstate.ParamState(jnp.asarray(amplitude, dtype=jnp.float32))
            self.tau_reset = brainstate.ParamState(jnp.asarray(tau, dtype=jnp.float32))
            self._parameter_names = _PARAMETER_NAMES + _GIF_PARAMETER_NAMES + _RESET_PARAMETER_NAMES
        else:
            self.reset_amp.value = jnp.asarray(amplitude, dtype=jnp.float32)
            self.tau_reset.value = jnp.asarray(tau, dtype=jnp.float32)
        if hasattr(self, "decay") and not hasattr(self, "reset"):
            self.reset = brainstate.HiddenState(jnp.zeros(self.voltage.value.shape, dtype=jnp.float32))
        return self

    def disable_reset(self) -> "DBNNGIF":
        """Disable optional voltage reset and remove its parameters and state."""
        if self.reset_enabled:
            del self.reset_amp
            del self.tau_reset
            if hasattr(self, "reset"):
                del self.reset
            self._parameter_names = _PARAMETER_NAMES + _GIF_PARAMETER_NAMES
        return self

    @classmethod
    def from_dbnn(cls, model: DBNN) -> "DBNNGIF":
        """Create an independently stateful GIF readout from a trained DBNN."""
        if not isinstance(model, DBNN) or isinstance(model, DBNNGIF):
            raise TypeError(f"model must be a DBNN, got {type(model).__name__}.")
        result = cls(model.layout, mode=model.mode, input_sign_mode=model.input_sign_mode, dt=model.dt)
        params = result.get_params()
        params.update(model.get_params())
        result.set_params(params)
        for attribute in (
            "source_fingerprint",
            "input_alignment",
            "dataset_trace_seeds",
            "consumed_trace_seeds",
            "data_seed_roots",
        ):
            if hasattr(model, attribute):
                setattr(result, attribute, getattr(model, attribute))
        return result

    def predict(self, inputs: Any, *, mode: str | None = None) -> dict[str, jax.Array]:
        """Predict closed-loop voltage, dynamic threshold, and spikes."""
        inputs = self.validate_inputs(inputs)
        selected_mode = self.mode if mode is None else mode
        if selected_mode not in {"f", "r"}:
            raise ValueError(f"mode must be 'f' or 'r', got {selected_mode!r}.")
        function = _dbnn_gif_f_state if selected_mode == "f" else _dbnn_gif_r_state
        voltage, threshold, spike = function(self.get_params(), inputs, channel_signs=self.channel_signs)
        return {"voltage": voltage, "threshold": threshold, "spike": spike}

    def init_state(self, batch_size: int | None = None) -> None:
        """Allocate DBNN recurrence, threshold, and output states."""
        super().init_state(batch_size)
        size = 1 if batch_size is None else batch_size
        self.threshold_adaptation = brainstate.HiddenState(jnp.zeros((size,), dtype=jnp.float32))
        if self.reset_enabled:
            self.reset = brainstate.HiddenState(jnp.zeros((size,), dtype=jnp.float32))
        self.previous_threshold = brainstate.HiddenState(jnp.full((size,), self.v_th.value, dtype=jnp.float32))
        self.threshold = brainstate.ShortTermState(jnp.full((size,), self.v_th.value, dtype=jnp.float32))

    def reset_state(self, batch_size: int | None = None) -> None:
        """Reset every DBNN-GIF recurrent and output state."""
        super().reset_state(batch_size)
        size = 1 if batch_size is None else batch_size
        self.threshold_adaptation.value = jnp.zeros((size,), dtype=jnp.float32)
        if self.reset_enabled:
            self.reset.value = jnp.zeros((size,), dtype=jnp.float32)
        self.previous_threshold.value = jnp.full((size,), self.v_th.value, dtype=jnp.float32)
        self.threshold.value = jnp.full((size,), self.v_th.value, dtype=jnp.float32)

    def update(self, events: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Advance one recurrent DBNN-GIF population step."""
        if self.mode != "r":
            raise RuntimeError("DBNNGIF.update() requires mode='r'.")
        if not hasattr(self, "decay"):
            raise RuntimeError("DBNN state is not initialized; call init_state(batch_size) first.")
        events = jnp.asarray(events)
        if events.shape != self.decay.value.shape:
            raise ValueError(f"events must have shape {self.decay.value.shape}, got {events.shape}.")
        self._validate_event_values(events, channel_axis=1, field="events")
        state = {
            "decay": self.decay.value,
            "rise_decay": self.rise_decay.value,
            "threshold": self.threshold_adaptation.value,
            "previous_voltage": self.previous_voltage.value,
            "previous_threshold": self.previous_threshold.value,
        }
        if self.reset_enabled:
            state["reset"] = self.reset.value
        state, (voltage, threshold, spike) = _dbnn_gif_step(
            self.get_params(), state, events, channel_signs=self.channel_signs
        )
        self.decay.value = state["decay"]
        self.rise_decay.value = state["rise_decay"]
        self.threshold_adaptation.value = state["threshold"]
        if self.reset_enabled:
            self.reset.value = state["reset"]
        self.previous_voltage.value = state["previous_voltage"]
        self.previous_threshold.value = state["previous_threshold"]
        self.voltage.value = voltage
        self.threshold.value = threshold
        self.spike.value = spike
        return voltage, threshold, spike


__all__ = [
    "DBNN",
    "DBNNGIF",
    "SpikeAlignmentEvidence",
    "align_spike_times",
]
