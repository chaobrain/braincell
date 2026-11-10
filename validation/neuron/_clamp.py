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

"""Voltage-clamp protocols shared by the NEURON comparison suites.

NEURON clamps the membrane with a near-ideal ``SEClamp``; BrainCell replays
the voltage NEURON produced through a Cell solver that replaces only the
voltage solve of the staggered step, so both sides advance the mechanism
equations under the same voltage.

Time convention, the same on both sides: step ``n`` takes the voltage from
``V_n`` to ``V_{n+1}`` and then advances the mechanism states with
``V_{n+1}``. NEURON's currents are re-evaluated with ``fcurrent()`` after
each step, so they are ``I(V_{n+1}, s_{n+1})``, like BrainCell's probes.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

SECLAMP_RS_MOHM = 1e-9
"""Series resistance of the NEURON clamp; ``|v - command| <= 1e-7 mV``."""


def _steps(duration_ms: float, dt_ms: float) -> int:
    count = float(duration_ms) / dt_ms
    if abs(count - round(count)) > 1e-9:
        raise ValueError(f"{duration_ms!r} ms is not a whole number of dt = {dt_ms!r} ms steps.")
    return int(round(count))


def protocol_commands(spec: Mapping[str, Any], dt_ms: float) -> list[np.ndarray]:
    """Expand the ``voltage_clamp`` block of a template into per-step commands.

    ``kind = "steps"`` holds ``hold_mV`` for ``hold_ms``, steps to each of
    ``test_mV`` for ``test_ms`` and returns to ``hold_mV`` for ``tail_ms``.
    ``kind = "ramp"`` holds ``start_mV`` for ``hold_ms``, rises at
    ``slope_mV_per_ms`` to ``stop_mV`` and holds it for ``final_hold_ms``.
    Element ``n`` of each returned command is held over the step that starts
    at ``t = n dt``.
    """
    kind = spec.get("kind")
    if kind == "steps":
        n_hold, n_test, n_tail = (_steps(spec[key], dt_ms) for key in ("hold_ms", "test_ms", "tail_ms"))
        commands = []
        for test in spec["test_mV"]:
            command = np.full(n_hold + n_test + n_tail, float(spec["hold_mV"]))
            command[n_hold : n_hold + n_test] = float(test)
            commands.append(command)
        return commands
    if kind == "ramp":
        start, stop, slope = (float(spec[key]) for key in ("start_mV", "stop_mV", "slope_mV_per_ms"))
        hold_ms = float(spec["hold_ms"])
        n_steps = _steps(hold_ms + (stop - start) / slope + float(spec["final_hold_ms"]), dt_ms)
        t_ms = np.arange(n_steps) * dt_ms
        return [np.clip(start + slope * (t_ms - hold_ms), min(start, stop), max(start, stop))]
    raise ValueError(f"voltage_clamp.kind must be 'steps' or 'ramp', got {kind!r}.")


def initialize_neuron(h, segment, *, v_init_mV, dt_ms, celsius, concentrations=None) -> None:
    """Initialize a fixed-step NEURON run, writing ion concentrations first.

    Concentrations (segment attributes such as ``cai``, in mM) are written
    *before* ``finitialize`` so that the mechanisms' ``INITIAL`` blocks and
    the Nernst potentials read them. Written afterwards, concentration-
    dependent states and ``eca`` keep NEURON's defaults until the next
    ``finitialize``.
    """
    h.CVode().active(0)
    h.secondorder = 0
    h.dt = float(dt_ms)
    h.celsius = float(celsius)
    for name, value in (concentrations or {}).items():
        setattr(segment, name, float(value))
    h.finitialize(float(v_init_mV))


def run_neuron_seclamp(
    h,
    segment,
    command_mV: Sequence[float],
    *,
    dt_ms: float,
    celsius: float,
    concentrations: Mapping[str, float] | None = None,
    observables: Mapping[str, Callable[[], float]] | None = None,
    initialize: Callable[..., None] | None = None,
) -> dict[str, np.ndarray]:
    """Clamp ``segment`` to ``command_mV`` from ``v_init = command_mV[0]``.

    The command is played into the ``SEClamp``'s ``amp1`` at the step starts.
    ``initialize`` replaces :func:`initialize_neuron` (tests use it to check
    the order). Returns ``"v"`` and each observable, read after
    initialization and after each step, ``len(command_mV) + 1`` samples.
    """
    command = np.asarray(command_mV, dtype=float)
    observables = {"v": lambda: segment.v, **(observables or {})}
    clamp = h.SEClamp(segment)
    clamp.rs, clamp.dur1, clamp.amp1 = SECLAMP_RS_MOHM, 1e9, float(command[0])
    # Both vectors must outlive the run: NEURON reads them while playing.
    values, times = h.Vector(command), h.Vector(np.arange(command.size) * float(dt_ms))
    values.play(clamp._ref_amp1, times)
    try:
        (initialize or initialize_neuron)(
            h, segment, v_init_mV=command[0], dt_ms=dt_ms, celsius=celsius, concentrations=concentrations
        )
        records = {name: [] for name in observables}
        for step in range(command.size + 1):
            if step:
                h.fadvance()
            h.fcurrent()
            for name, read in observables.items():
                records[name].append(read())
    finally:
        values.play_remove()
    return {name: np.asarray(trace, dtype=float) for name, trace in records.items()}


def voltage_replay_solver(voltages_mV: Sequence[float]):
    """Return a Cell solver that replays ``V_0, ..., V_N`` instead of solving for V.

    Step ``n`` (``t = n dt``, from a Cell starting at ``t = 0``) follows the
    staggered order with the voltage solve replaced: it caches the ion total
    currents at ``V_n``, sets every CV to ``voltages_mV[n + 1]``, then
    advances the runtime synapses and the ion and channel states with that
    voltage, honoring ``ion_channel_update_order``.
    """
    import brainunit as u
    import jax.numpy as jnp

    from braincell.quad._util import environ_time

    table = np.asarray(voltages_mV, dtype=float)

    def voltage_replay(cell):
        t, dt = environ_time(cell)
        step = jnp.round(u.get_magnitude(t / dt)).astype(jnp.int32)
        cell.cache_ion_total_currents(cell.V.value)
        mantissa = cell.V.value.to_decimal(u.mV)
        V = jnp.full_like(mantissa, jnp.asarray(table, dtype=mantissa.dtype)[step + 1]) * u.mV
        cell.V.value = V
        if cell.ion_channel_update_order == "family":
            cell._integrate_runtime_synapse_dynamics(cell._point_voltage_for_mechanisms(V))
            cell._update_ion_channel_families(V)
        else:
            cell._update_ion_channels_by_integration(V)

    return voltage_replay
