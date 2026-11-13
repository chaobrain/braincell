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

"""``ExpSyn`` and ``Exp2Syn`` in BrainCell and in NEURON's built-in mechanisms.

Both simulators run the same model: one cylinder, L = 20 um and diameter
20 um, one segment, cm 1 uF/cm^2, Ra 100 Ohm cm, a leak of 0.1 mS/cm^2
reversing at -65 mV (NEURON ``pas``, BrainCell ``IL``), V_init -65 mV, and
one synapse at x = 0.5. The events are an explicit list of times: in
BrainCell an ``EventSequence`` through ``Network.connect`` with zero delay,
in NEURON ``NetCon(None, syn).event(t)`` queued at ``finitialize``. NEURON
runs fixed-step (``cvode_active(0)``, ``secondorder = 0``); BrainCell in
float64.

Sample k of either trace is the state at t_k = k dt. In BrainCell it is
taken before that boundary's events, and an event rounded to step s shows
from sample s + 1; NEURON's recorded *states* (``ExpSyn.g``, ``Exp2Syn.A``
and ``B``) follow the same rule. NEURON's recorded point-process ``i`` and
``Exp2Syn.g`` are ``BREAKPOINT`` assignments: they include the boundary's
events and lag the states by one sample, and ``i`` is outward-positive. The
comparison therefore takes the conductance from NEURON's states and the
current as ``g_nrn (e - V_nrn)``, BrainCell's inward-positive convention.

An event time exactly half a step from two boundaries goes to the later step
in BrainCell; NEURON's choice depends on its accumulated ``t``. The
scenarios below avoid exact ties; the BrainCell package tests fix the tie
rule against the analytic solution.
"""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np

__all__ = ["NEAR_EQUAL_SCENARIOS", "SCENARIOS", "Scenario", "run_braincell", "run_neuron"]

DT_MS = 0.025
LENGTH_UM = 20.0
DIAMETER_UM = 20.0
LEAK_MS_PER_CM2 = 0.1
LEAK_E_MV = -65.0


@dataclass(frozen=True)
class Scenario:
    """One synapse driven by an explicit event list.

    Parameters
    ----------
    name : str
        Label used in reports and test ids.
    synapse_type : str
        ``"ExpSyn"`` or ``"Exp2Syn"``, the same name in both simulators.
    parameters : dict
        Synapse parameters as plain numbers: time constants in ms, ``e`` in mV.
    events_ms : tuple of float
        Event times; a repeated time is two events in one step.
    weight_uS : float
        Weight of every event.
    duration_ms : float
        Run length; ``duration / dt`` samples are compared.
    """

    name: str
    synapse_type: str
    parameters: dict
    events_ms: tuple
    weight_uS: float
    duration_ms: float

    @property
    def e_mV(self) -> float:
        return float(self.parameters["e"])


SCENARIOS = (
    Scenario("single, on grid", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0,), 0.001, 20.0),
    Scenario("5 events, 2 ms apart", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0, 3.0, 5.0, 7.0, 9.0), 0.001, 30.0),
    Scenario(
        "two in one step, near-ties", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0, 1.0, 2.0124, 2.0126, 4.0374), 0.001, 20.0
    ),
    Scenario("tau = 4 dt", "ExpSyn", {"tau": 0.1, "e": 0.0}, (1.0, 1.3, 1.6), 0.002, 10.0),
    Scenario(
        "tau 50 ms, 20 events", "ExpSyn", {"tau": 50.0, "e": 0.0}, tuple(1.0 + 2.5 * k for k in range(20)), 0.0005, 200.0
    ),
    Scenario("inhibitory, e below rest", "ExpSyn", {"tau": 5.0, "e": -80.0}, (1.0, 2.0, 3.0), 0.005, 40.0),
    Scenario("e = E_leak = V_init", "ExpSyn", {"tau": 5.0, "e": -65.0}, (1.0, 2.0, 3.0), 0.01, 40.0),
    Scenario(
        "strong drive, V towards e", "ExpSyn", {"tau": 20.0, "e": -55.0}, tuple(1.0 + k for k in range(10)), 0.05, 60.0
    ),
    Scenario("Exp2Syn 0.5/5", "Exp2Syn", {"tau1": 0.5, "tau2": 5.0, "e": 0.0}, (1.0, 3.0, 3.0, 8.0126), 0.001, 40.0),
    Scenario("Exp2Syn default 0.1/10", "Exp2Syn", {"tau1": 0.1, "tau2": 10.0, "e": 0.0}, (1.0, 6.0), 0.001, 60.0),
    Scenario("Exp2Syn near-equal 1/1.05", "Exp2Syn", {"tau1": 1.0, "tau2": 1.05, "e": 0.0}, (1.0, 2.0), 0.001, 30.0),
    Scenario("Exp2Syn inhibitory 1/20", "Exp2Syn", {"tau1": 1.0, "tau2": 20.0, "e": -80.0}, (1.0, 4.0), 0.003, 100.0),
)

# tau1 close to tau2 (CELL-62), delta = 1 - tau1 / tau2. NEURON's exp2syn.mod
# runs tau1 / tau2 > 0.9999 as tau1 = 0.9999 tau2 in INITIAL, and BrainCell
# does the same, so the 1e-5 scenario runs the delta = 1e-4 kernel on both
# sides. Its states are about 3e4 times the weight, so V and the current keep
# fewer digits than in SCENARIOS (see reference_test.py).
NEAR_EQUAL_SCENARIOS = tuple(
    Scenario(f"Exp2Syn delta {delta:g}", "Exp2Syn", {"tau1": 5.0 * (1.0 - delta), "tau2": 5.0, "e": 0.0}, (1.0, 11.0), 0.001, 40.0)
    for delta in (1e-2, 1e-5)
)


def _braincell_cell(scenario: Scenario):
    import braincell as bc
    from braincell.filter import at

    branch = bc.Branch.from_lengths(lengths=[LENGTH_UM] * u.um, radii=[DIAMETER_UM / 2, DIAMETER_UM / 2] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=LEAK_E_MV * u.mV)
    cell.paint(
        bc.mech.CableProperty(
            resting_potential=LEAK_E_MV * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        )
    )
    cell.paint(bc.mech.Channel("IL", g_max=LEAK_MS_PER_CM2 * u.mS / u.cm**2, E=LEAK_E_MV * u.mV))
    parameters = {key: value * (u.mV if key == "e" else u.ms) for key, value in scenario.parameters.items()}
    cell.place(at(0, 0.5), bc.mech.Synapse(scenario.synapse_type, name="syn", **parameters))
    observe = bc.observe.synapse(name="syn")
    cell.record("v", "V")
    cell.record("i", observe.current())
    for state in ("g",) if scenario.synapse_type == "ExpSyn" else ("A", "B"):
        cell.record(state, observe.state(state))
    return cell


def run_braincell(scenarios, dt_ms: float = DT_MS) -> dict[str, dict[str, np.ndarray]]:
    """Run every scenario as one population of a single Network, in float64.

    The scenarios share one compilation and run for the longest duration;
    each trace is cut to its own ``duration / dt`` samples.

    Returns
    -------
    dict
        Scenario name to ``time_ms``, ``v_mV``, ``g_uS`` and ``i_nA``
        (inward-positive), and ``A_uS``, ``B_uS`` for ``Exp2Syn``.
    """
    import braincell as bc

    scenarios = tuple(scenarios)
    with brainstate.environ.context(precision=64):
        network = bc.Network("exp_syn_reference")
        for index, scenario in enumerate(scenarios):
            post = network.add_population(f"post{index}", _braincell_cell(scenario))
            source = network.add_population(
                f"events{index}", bc.EventSequence.from_times([np.asarray(scenario.events_ms) * u.ms])
            )
            network.connect(
                f"drive{index}",
                source=source,
                synapse=post.synapses["syn"],
                weight=scenario.weight_uS * u.uS,
                delay=0.0 * u.ms,
            )
        duration = max(scenario.duration_ms for scenario in scenarios)
        result = network.run(dt=dt_ms * u.ms, duration=duration * u.ms)
    time_ms = np.asarray(result.time.to_decimal(u.ms), dtype=np.float64).reshape(-1)
    traces = {}
    for index, scenario in enumerate(scenarios):
        samples = result.samples[f"post{index}"]
        n = round(scenario.duration_ms / dt_ms)

        def trace(field, unit, samples=samples, n=n):
            return np.asarray(samples[field].values.to_decimal(unit), dtype=np.float64).reshape(-1)[:n]

        out = {"time_ms": time_ms[:n], "v_mV": trace("v", u.mV), "i_nA": trace("i", u.nA)}
        if scenario.synapse_type == "ExpSyn":
            out["g_uS"] = trace("g", u.uS)
        else:
            out["A_uS"], out["B_uS"] = trace("A", u.uS), trace("B", u.uS)
            out["g_uS"] = out["B_uS"] - out["A_uS"]
        traces[scenario.name] = out
    return traces


def run_neuron(scenario: Scenario, dt_ms: float = DT_MS) -> dict[str, np.ndarray]:
    """Run one scenario with NEURON's built-in mechanism, fixed step.

    Returns
    -------
    dict
        ``time_ms``, ``v_mV``, ``g_uS`` from the states (``g``, or ``B - A``),
        ``i_nA`` in BrainCell's convention, ``g_uS (e - v_mV)``, and
        ``i_recorded_nA``, NEURON's own outward-positive ``i`` (one sample
        behind, see the module notes); ``duration / dt`` samples each.
    """
    from neuron import h

    h.load_file("stdrun.hoc")
    soma = h.Section(name="soma")
    soma.L, soma.diam, soma.nseg, soma.cm, soma.Ra = LENGTH_UM, DIAMETER_UM, 1, 1.0, 100.0
    soma.insert("pas")
    soma(0.5).pas.g = LEAK_MS_PER_CM2 * 1e-3  # S/cm^2
    soma(0.5).pas.e = LEAK_E_MV
    synapse = getattr(h, scenario.synapse_type)(soma(0.5))
    for key, value in scenario.parameters.items():
        setattr(synapse, key, value)
    netcon = h.NetCon(None, synapse)
    netcon.weight[0] = scenario.weight_uS
    netcon.delay = 0.0
    handler = h.FInitializeHandler(lambda: [netcon.event(float(time)) for time in scenario.events_ms])
    h.cvode_active(0)
    h.secondorder = 0
    h.dt = dt_ms
    h.steps_per_ms = 1.0 / dt_ms
    vectors = {"time_ms": h._ref_t, "v_mV": soma(0.5)._ref_v, "i_recorded_nA": synapse._ref_i}
    states = ("g",) if scenario.synapse_type == "ExpSyn" else ("A", "B")
    vectors.update({state: getattr(synapse, f"_ref_{state}") for state in states})
    vectors = {key: h.Vector().record(ref) for key, ref in vectors.items()}
    try:
        h.finitialize(LEAK_E_MV)
        h.continuerun(scenario.duration_ms)
    finally:
        del handler, netcon, synapse
        h.delete_section(sec=soma)
    n = round(scenario.duration_ms / dt_ms)
    out = {key: np.asarray(vector, dtype=np.float64)[:n] for key, vector in vectors.items()}
    if scenario.synapse_type == "ExpSyn":
        out["g_uS"] = out.pop("g")
    else:
        out["A_uS"], out["B_uS"] = out.pop("A"), out.pop("B")
        out["g_uS"] = out["B_uS"] - out["A_uS"]
    out["i_nA"] = out["g_uS"] * (scenario.e_mV - out["v_mV"])
    return out
