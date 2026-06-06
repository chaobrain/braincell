"""Simulation time loop and trace helpers for :meth:`Cell.run`.

Propagates ``t`` through :mod:`brainstate.environ` inside the
``for_loop`` scan instead of mutating ``Cell._current_time``
per step; the final post-loop time is pinned once after the scan.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainstate
import brainunit as u
import jax
import numpy as np

if TYPE_CHECKING:
    from .cell import Cell

__all__ = ["RunResult", "run"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunResult:
    """Return value of :meth:`Cell.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        ``(n_steps,)`` time array spanning ``[start_t, start_t + duration)``.
    traces : dict[str, Any]
        Probe-name → trace array mapping; one entry per placed probe.
    """

    time: object
    traces: dict


def run(rcell: "Cell", *, dt, duration) -> RunResult:
    """Advance ``rcell`` for ``duration`` at ``dt`` and collect probe traces."""
    _validate_time_quantity(dt, name="dt")
    _validate_time_quantity(duration, name="duration")

    initial_samples = rcell.sample_probes()
    if len(initial_samples) == 0:
        raise ValueError("Cell.run(...) requires at least one placed probe.")
    ordered_names = tuple(sorted(initial_samples))

    with brainstate.environ.context(dt=dt):
        start_t = rcell.current_time
        relative_times = u.math.arange(0.0 * u.ms, duration, brainstate.environ.get_dt())
        if int(relative_times.shape[0]) == 0:
            raise ValueError(
                "Cell.run(...) produced no timesteps; "
                "ensure duration > 0 and dt > 0."
            )
        times = start_t + relative_times
        with brainstate.environ.context(t=start_t):
            rcell._prepare_next_synapse_inputs()

        def _step(t):
            with brainstate.environ.context(t=t):
                rcell._begin_step()
                rcell._update_dynamics()
                snapshot = rcell.sample_probes()
                rcell._prepare_next_synapse_inputs()
            return tuple(snapshot[name] for name in ordered_names)

        traces_over_time = brainstate.transform.for_loop(_step, times)
        rcell._set_current_time(
            start_t + int(times.shape[0]) * brainstate.environ.get_dt()
        )

    traces_tuple = _normalize_run_traces(traces_over_time, n_traces=len(ordered_names))
    traces = {name: trace for name, trace in zip(ordered_names, traces_tuple)}
    return RunResult(time=times, traces=traces)


def _validate_time_quantity(value, *, name: str) -> None:
    """Require ``value`` to be a positive scalar time :class:`Quantity`."""
    if not hasattr(value, "to_decimal"):
        raise TypeError(
            f"Cell.run(...) {name} must be a time quantity, got {value!r}."
        )
    decimal = np.asarray(value.to_decimal(u.ms), dtype=float)
    if decimal.shape not in ((), (1,)):
        raise ValueError(
            f"Cell.run(...) {name} must be scalar, got shape {decimal.shape!r}."
        )
    if float(decimal.reshape(())) <= 0.0:
        raise ValueError(
            f"Cell.run(...) {name} must be > 0, got {value!r}."
        )


def _normalize_run_traces(values, *, n_traces: int) -> tuple:
    """Wrap scalar ``for_loop`` output when a single trace is collected."""
    if n_traces == 1:
        return values if isinstance(values, tuple) else (values,)
    if not isinstance(values, tuple):
        raise TypeError(
            f"Cell.run(...) expected {n_traces} trace arrays, "
            f"got {type(values).__name__!s}."
        )
    if len(values) != n_traces:
        raise ValueError(
            f"Cell.run(...) expected {n_traces} trace arrays, got {len(values)!r}."
        )
    return values
