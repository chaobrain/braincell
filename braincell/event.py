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

"""Standalone event-source declarations."""

from dataclasses import dataclass, field
from typing import Any

import brainstate
import brainunit as u
import numpy as np

__all__ = ["NetStim"]


@dataclass(frozen=True)
class NetStim:
    """A population of deterministic or noisy artificial spike sources.

    Parameters
    ----------
    size : int, optional
        Number of independent sources. Defaults to one.
    start : Quantity[ms], optional
        Most likely first event time, scalar or one value per source. With
        nonzero ``noise``, the realized first event occurs after an
        exponential waiting time with mean ``noise * interval``.
    number : int or array-like of int, optional
        Event count, scalar or one value per source.
    interval : Quantity[ms], optional
        Mean inter-event interval, scalar or one value per source.
    noise : float or array-like, optional
        Fraction of each interval drawn from an exponential distribution.
        Zero is periodic and one is Poisson-like.
    seed : int or None, optional
        Source-local seed. ``None`` uses the reproducible standalone root 0.
    name : str or None, optional
        Optional display name.
    """

    size: int = 1
    start: Any = field(default_factory=lambda: 0.0 * u.ms)
    number: Any = 1
    interval: Any = field(default_factory=lambda: 10.0 * u.ms)
    noise: Any = 0.0
    seed: int | None = None
    name: str | None = None
    _event_times_ms: np.ndarray = field(init=False, repr=False, compare=False)
    _event_mask: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, (int, np.integer)):
            raise TypeError(f"NetStim.size must be an integer, got {self.size!r}.")
        size = int(self.size)
        if size < 1:
            raise ValueError(f"NetStim.size must be >= 1, got {size!r}.")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer))):
            raise TypeError(f"NetStim.seed must be an integer or None, got {self.seed!r}.")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError(f"NetStim.name must be a non-empty string or None, got {self.name!r}.")

        start = _quantity_vector(self.start, unit=u.ms, size=size, name="NetStim.start")
        interval = _quantity_vector(self.interval, unit=u.ms, size=size, name="NetStim.interval")
        number = _integer_vector(self.number, size=size, name="NetStim.number")
        noise = _real_vector(self.noise, size=size, name="NetStim.noise")
        start_ms = np.asarray(start.to_decimal(u.ms), dtype=np.float64)
        interval_ms = np.asarray(interval.to_decimal(u.ms), dtype=np.float64)
        if np.any(start_ms < 0.0):
            raise ValueError("NetStim.start must be >= 0 ms.")
        if np.any(interval_ms <= 0.0):
            raise ValueError("NetStim.interval must be > 0 ms.")
        if np.any(number < 0):
            raise ValueError("NetStim.number must be >= 0.")
        if np.any((noise < 0.0) | (noise > 1.0)):
            raise ValueError("NetStim.noise must be within [0, 1].")

        event_times, event_mask = _build_event_schedule(
            start_ms=start_ms,
            number=number,
            interval_ms=interval_ms,
            noise=noise,
            seed=0 if self.seed is None else int(self.seed),
        )
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "_event_times_ms", event_times)
        object.__setattr__(self, "_event_mask", event_mask)

    @property
    def instance_name(self) -> str:
        """Display label for this source population."""
        return self.name if self.name is not None else "NetStim"

    @property
    def event_times(self) -> u.Quantity:
        """Padded continuous event-time matrix with shape ``(size, max(number))``."""
        return u.Quantity(self._event_times_ms, u.ms)

    def event_count(self, source_index, *, t, delay, dt):
        """Count delayed events assigned to the nearest fixed-step boundary.

        An arrival is delivered at step ``t`` when it lies in the half-open
        interval ``[t - dt / 2, t + dt / 2)``. This matches NEURON's
        fixed-step event delivery rule and assigns an exact half-step tie to
        the later boundary.
        """
        source_index = np.asarray(source_index, dtype=np.int32)
        times = u.math.asarray(self._event_times_ms[source_index])
        mask = u.math.asarray(self._event_mask[source_index])
        delay_ms = u.math.asarray(delay.to_decimal(u.ms))
        if getattr(delay_ms, "ndim", 0) == 0:
            delay_ms = u.math.broadcast_to(delay_ms, source_index.shape)
        arrivals = times + delay_ms[..., None]
        t_ms = u.math.asarray(t.to_decimal(u.ms))
        dt_ms = u.math.asarray(dt.to_decimal(u.ms))
        arrival_steps = _round_half_up_steps(arrivals / dt_ms)
        current_step = _round_half_up_steps(t_ms / dt_ms)
        selected = mask & (arrival_steps == current_step)
        return u.math.sum(selected, axis=-1)


def _round_half_up_steps(values):
    """Round non-negative step ratios, snapping numerical half ties upward."""
    values = u.math.asarray(values)
    half = u.math.floor(values) + 0.5
    magnitude = u.math.abs(values)
    toward_positive = u.math.asarray(np.inf, dtype=values.dtype)
    ulp = u.math.nextafter(magnitude, toward_positive) - magnitude
    snapped = u.math.where(u.math.abs(values - half) <= 4.0 * ulp, half, values)
    return u.math.floor(snapped + 0.5)


def _quantity_vector(value, *, unit, size: int, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a quantity.")
    try:
        decimal = np.asarray(value.to_decimal(unit), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{name} has an incompatible unit.") from exc
    decimal = _broadcast_vector(decimal, size=size, name=name)
    if np.any(~np.isfinite(decimal)):
        raise ValueError(f"{name} must contain finite values.")
    return u.Quantity(decimal, unit)


def _integer_vector(value, *, size: int, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers.")
    return _broadcast_vector(values, size=size, name=name).astype(np.int64, copy=False)


def _real_vector(value, *, size: int, name: str) -> np.ndarray:
    values = np.asarray(value)
    if values.dtype.kind not in "iuf" or values.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numbers.")
    result = _broadcast_vector(values, size=size, name=name).astype(np.float64, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _broadcast_vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    if value.ndim == 0:
        return np.array(np.broadcast_to(value, (size,)), copy=True)
    if value.shape == (size,):
        return np.array(value, copy=True)
    raise ValueError(f"{name} must be scalar or have shape {(size,)!r}, got {value.shape!r}.")


def _build_event_schedule(*, start_ms, number, interval_ms, noise, seed):
    max_events = int(np.max(number, initial=0))
    mask = np.arange(max_events)[None, :] < number[:, None]
    if max_events == 0:
        return np.empty((len(number), 0), dtype=np.float64), mask

    rng = brainstate.random.RandomState(seed)
    exponential = np.asarray(
        rng.exponential(scale=1.0, size=(len(number), max_events)),
        dtype=np.float64,
    )
    first_wait = noise * interval_ms * exponential[:, 0]
    if max_events == 1:
        offsets = first_wait[:, None]
    else:
        deterministic = (1.0 - noise[:, None]) * interval_ms[:, None]
        stochastic = noise[:, None] * interval_ms[:, None] * exponential[:, 1:]
        gaps = deterministic + stochastic
        offsets = first_wait[:, None] + np.concatenate(
            [
                np.zeros((len(number), 1), dtype=np.float64),
                np.cumsum(gaps, axis=1),
            ],
            axis=1,
        )
    return start_ms[:, None] + offsets, mask
