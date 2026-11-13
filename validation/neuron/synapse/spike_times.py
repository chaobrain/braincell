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

"""Spike times of an HH soma in BrainCell and in NEURON's ``NetCon.record`` (CELL-80).

Each simulator runs the same Hodgkin-Huxley soma (one 20 um x 20 um
compartment; NEURON's ``hh`` without its rate tables, BrainCell's
``Na_HH1952``, ``K_HH1952`` and ``IL``) under a constant current from
0.25 ms, fixed step 0.025 ms, 50 ms, float64. NEURON's ``NetCon.record``
stamps a threshold crossing during the step from ``t_m`` to ``t_{m+1}`` with
``t_{m+1}``; BrainCell's ``RunResult.spikes`` and ``NetworkResult.events``
carry the same time.
"""

from __future__ import annotations

import brainunit as u
import numpy as np

__all__ = ["AMPLITUDES_NA", "DT_MS", "DURATION_MS", "run_braincell", "run_neuron"]

AMPLITUDES_NA = (0.1, 0.2, 0.3)
DT_MS = 0.025
DURATION_MS = 50.0


def _cell(amplitude_na: float):
    import braincell as bc
    from braincell import mech
    from braincell.filter import AllRegion, at

    soma = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = bc.Cell(
        bc.Morphology.from_root(soma, name="soma"),
        cv_policy=bc.CVPerBranch(),
        solver="staggered",
        V_init=-65.0 * u.mV,
        V_th=0.0 * u.mV,
    )
    cell.paint(
        AllRegion(),
        mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * (u.uF / u.cm**2),
            axial_resistivity=100.0 * (u.ohm * u.cm),
        ),
        mech.Ion("SodiumFixed", E=50.0 * u.mV),
        mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        mech.Channel("Na_HH1952", g_max=120.0 * (u.mS / u.cm**2)),
        mech.Channel("K_HH1952", g_max=36.0 * (u.mS / u.cm**2)),
        mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.387 * u.mV),
    )
    cell.place(
        at("soma", 0.5), mech.CurrentClamp(delay=0.25 * u.ms, durations=1e3 * u.ms, amplitudes=amplitude_na * u.nA)
    )
    return cell


def run_braincell() -> dict[float, dict[str, np.ndarray]]:
    """``{amplitude: {"run": times, "network": times}}`` in ms, from ``Cell.run`` and ``Network.run``."""
    import braincell as bc

    dt = DT_MS * u.ms
    n_steps = round(DURATION_MS / DT_MS)
    out = {}
    for amplitude in AMPLITUDES_NA:
        cell = _cell(amplitude)
        cell.init_state()
        run = cell.run(dt=dt, duration=n_steps * dt).spikes
        net = bc.Network("spike_times")
        net.add_population("c", _cell(amplitude))
        net.init_state()
        network = net.run(dt=dt, duration=n_steps * dt).events["c"]["spike"]
        out[amplitude] = {
            "run": np.asarray(run.time.to_decimal(u.ms), dtype=np.float64),
            "network": np.asarray(network.time.to_decimal(u.ms), dtype=np.float64),
        }
    return out


def run_neuron() -> dict[float, np.ndarray]:
    """``{amplitude: times}`` in ms from NEURON's ``NetCon(v, None).record`` with threshold 0 mV."""
    from neuron import h

    h.load_file("stdrun.hoc")
    # hh interpolates its rates on a 1 mV table; BrainCell evaluates them exactly.
    usetable = h.usetable_hh
    h.usetable_hh = 0

    def run(amplitude):
        soma = h.Section(name="soma")
        soma.L = soma.diam = 20.0
        soma.nseg, soma.cm, soma.Ra = 1, 1.0, 100.0
        soma.insert("hh")
        soma.ena, soma.ek = 50.0, -77.0
        for segment in soma:
            segment.hh.gnabar, segment.hh.gkbar = 0.12, 0.036
            segment.hh.gl, segment.hh.el = 0.0003, -54.387
        clamp = h.IClamp(soma(0.5))
        clamp.delay, clamp.dur, clamp.amp = 0.25, 1e9, amplitude
        netcon = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
        netcon.threshold = 0.0
        times = h.Vector()
        netcon.record(times)
        h.cvode_active(0)
        h.secondorder = 0
        h.dt = DT_MS
        h.steps_per_ms = 1.0 / DT_MS
        try:
            h.finitialize(-65.0)
            h.continuerun(DURATION_MS)
            return np.asarray(times, dtype=np.float64).copy()
        finally:
            del netcon, clamp
            h.delete_section(sec=soma)

    try:
        return {amplitude: run(amplitude) for amplitude in AMPLITUDES_NA}
    finally:
        h.usetable_hh = usetable
