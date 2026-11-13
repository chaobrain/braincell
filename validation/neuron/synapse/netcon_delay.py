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

"""A delayed self-connection of an HH soma in BrainCell's Network and in NEURON's ``NetCon`` (CELL-75).

Each simulator runs its own Hodgkin-Huxley soma (NEURON's ``hh``; BrainCell's
``Na_HH1952``, ``K_HH1952`` and ``IL``), 0.1 nA from 0.25 ms, whose spike
drives its own ``ExpSyn`` (tau 2 ms, e 0 mV, 0.002 uS) ``k`` steps of
0.025 ms later, fixed step, float64. The two HH models cross 0 mV in
different steps, so the comparison is relative to each one's crossing step
``m`` (V before the step below 0 mV, V after it at or above): the first step
whose V differs from the same soma without the connection, and the
recorded conductance around the delivery boundary ``t_{m+1+k}``.
"""

from __future__ import annotations

import brainunit as u
import numpy as np

__all__ = ["DELAY_STEPS", "DT_MS", "TAU_MS", "WEIGHT_US", "run_braincell", "run_neuron"]

DT_MS = 0.025
DELAY_STEPS = (0, 1, 2, 3, 4, 12)
DURATION_MS = 4.0
WEIGHT_US = 0.002
TAU_MS = 2.0


def _crossing(v_after):
    v_before = np.concatenate([[-65.0], v_after[:-1]])
    return int(np.flatnonzero((v_before < 0.0) & (v_after >= 0.0))[0])


def _summary(v_after, reference, g):
    """Crossing step, first step seen after it, and g at the delivery boundary and the sample after it."""
    m = _crossing(reference)
    first = int(np.flatnonzero(v_after != reference)[0])
    return {"crossing": m, "first_seen": first, "g": g}


def run_braincell() -> dict[int, dict]:
    """``{k: summary}`` from a ``Network`` of one lumped HH Cell per delay."""
    import braincell as bc
    from braincell import mech
    from braincell.filter import AllRegion

    dt = DT_MS * u.ms
    n_steps = round(DURATION_MS / DT_MS)

    def run(k):
        cell = bc.Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV, solver="rk4")
        cell.paint(
            AllRegion(),
            mech.Ion("SodiumFixed", name="sodium", E=50.0 * u.mV),
            mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            mech.Channel("IL", name="leak"),
            mech.Channel("Na_HH1952", name="na"),
            mech.Channel("K_HH1952", name="k"),
        )
        cell.place(mech.CurrentClamp(delay=0.25 * u.ms, durations=1e3 * u.ms, amplitudes=0.1 * u.nA))
        cell.place(mech.Synapse("ExpSyn", name="syn", tau=TAU_MS * u.ms, e=0.0 * u.mV))
        cell.record("v", bc.observe.state("V"))
        cell.record("g", bc.observe.synapse(name="syn").state("g"))
        if k is not None:
            bc.connect(
                "self",
                source=cell.event_outputs["spike"],
                synapse=cell.synapses["syn"],
                weight=WEIGHT_US * u.uS,
                delay=k * dt,
            )
        net = bc.Network("netcon_delay")
        net.add_population("c", cell)
        net.init_state()
        samples = net.run(dt=dt, duration=n_steps * dt).samples["c"]
        v = np.asarray(samples["v"].values.to_decimal(u.mV), dtype=np.float64).reshape(n_steps)
        g = np.asarray(samples["g"].values.to_decimal(u.uS), dtype=np.float64).reshape(n_steps)
        last = float(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)[0])
        return np.concatenate([v[1:], [last]]), g

    reference, _ = run(None)
    return {k: _summary(*_with_reference(run(k), reference)) for k in DELAY_STEPS}


def _with_reference(run, reference):
    v_after, g = run
    return v_after, reference, g


def run_neuron() -> dict[int, dict]:
    """``{k: summary}`` from NEURON: one ``hh`` soma, a ``NetCon`` from its voltage to its ``ExpSyn``."""
    from neuron import h

    h.load_file("stdrun.hoc")

    def run(k):
        soma = h.Section(name="soma")
        soma.L = soma.diam = 20.0
        soma.insert("hh")
        clamp = h.IClamp(soma(0.5))
        clamp.delay, clamp.dur, clamp.amp = 0.25, 1e9, 0.1
        synapse = h.ExpSyn(soma(0.5))
        synapse.tau, synapse.e = TAU_MS, 0.0
        netcon = None
        if k is not None:
            netcon = h.NetCon(soma(0.5)._ref_v, synapse, sec=soma)
            netcon.threshold, netcon.delay, netcon.weight[0] = 0.0, k * DT_MS, WEIGHT_US
        v, g = h.Vector().record(soma(0.5)._ref_v), h.Vector().record(synapse._ref_g)
        h.cvode_active(0)
        h.secondorder = 0
        h.dt = DT_MS
        h.steps_per_ms = 1.0 / DT_MS
        try:
            h.finitialize(-65.0)
            h.continuerun(DURATION_MS)
            v, g = np.asarray(v), np.asarray(g)
        finally:
            del netcon, synapse, clamp
            h.delete_section(sec=soma)
        # Sample j is the state at t_j: V after step j - 1, and g before the events of t_j.
        n_steps = round(DURATION_MS / DT_MS)
        return v[1 : n_steps + 1], g[:n_steps]

    reference, _ = run(None)
    return {k: _summary(*_with_reference(run(k), reference)) for k in DELAY_STEPS}
