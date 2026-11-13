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

"""``ExpSyn`` and ``Exp2Syn`` in a Cell against NEURON's built-in mechanisms (CELL-44).

The twelve scenarios of :mod:`validation.neuron.synapse.reference`. Both
sides discretize alike (implicit Euler for V, exact exponentials for the
synapse states, the same event rounding), so they differ by rounding only:
V within 1e-9 mV, the conductance within 1e-12 uS, and the current within
1e-10 nA of ``g_nrn (e - V_nrn)``, float64. NEURON 9.0.2 gave at most
1.5e-12 mV, 6.7e-15 uS and 3.6e-14 nA. Only built-in mechanisms are used,
so no ``nrnivmodl`` build is needed.
"""

import unittest

import numpy as np

from validation.neuron.synapse.reference import NEAR_EQUAL_SCENARIOS, SCENARIOS, run_braincell, run_neuron

try:
    import neuron
except ImportError:  # pragma: no cover - depends on the environment
    neuron = None

V_ATOL_MV = 1e-9
G_ATOL_US = 1e-12
I_ATOL_NA = 1e-10


def _first_change_samples(g):
    """Samples where g jumps (an event shows), from the sample-to-sample change."""
    previous = np.concatenate([[0.0], g[:-1]])
    return np.flatnonzero(np.abs(g - previous) > 1e-12 + 0.2 * np.abs(previous)).tolist()


class ExponentialSynapseNeuronTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if neuron is None:
            raise unittest.SkipTest("NEURON is not installed; the ExpSyn/Exp2Syn NEURON comparison needs it.")
        cls.braincell = run_braincell(SCENARIOS)
        cls.neuron = {scenario.name: run_neuron(scenario) for scenario in SCENARIOS}

    def _assert_each_scenario(self, check):
        for scenario in SCENARIOS:
            with self.subTest(scenario.name):
                bc_trace, nrn_trace = self.braincell[scenario.name], self.neuron[scenario.name]
                self.assertEqual(bc_trace["v_mV"].size, round(scenario.duration_ms / 0.025))
                self.assertEqual(nrn_trace["v_mV"].size, bc_trace["v_mV"].size)
                check(scenario, bc_trace, nrn_trace)

    def test_the_two_runs_sample_the_same_times(self):
        self._assert_each_scenario(
            lambda _, bc, nrn: np.testing.assert_allclose(bc["time_ms"], nrn["time_ms"], rtol=0.0, atol=1e-9)
        )

    def test_voltage_agrees(self):
        self._assert_each_scenario(
            lambda _, bc, nrn: np.testing.assert_allclose(bc["v_mV"], nrn["v_mV"], rtol=0.0, atol=V_ATOL_MV)
        )

    def test_conductance_agrees(self):
        self._assert_each_scenario(
            lambda _, bc, nrn: np.testing.assert_allclose(bc["g_uS"], nrn["g_uS"], rtol=0.0, atol=G_ATOL_US)
        )

    def test_current_agrees_with_the_neuron_states(self):
        self._assert_each_scenario(
            lambda _, bc, nrn: np.testing.assert_allclose(bc["i_nA"], nrn["i_nA"], rtol=0.0, atol=I_ATOL_NA)
        )

    def test_every_event_shows_at_the_same_sample(self):
        def check(scenario, bc, nrn):
            steps = np.floor(np.asarray(scenario.events_ms) / 0.025 + 0.5).astype(int)
            self.assertEqual(_first_change_samples(bc["g_uS"]), _first_change_samples(nrn["g_uS"]))
            self.assertEqual(_first_change_samples(bc["g_uS"])[0], steps[0] + 1)

        self._assert_each_scenario(check)

    def test_neurons_recorded_current_includes_the_boundary_events_one_sample_late(self):
        # The phase the module notes describe: NEURON's own `i` at sample k + 1
        # is outward-positive and uses g just after boundary k's events (ExpSyn
        # jumps there; Exp2Syn's B - A does not) with V at sample k.
        def check(scenario, _, nrn):
            g_after = nrn["g_uS"].copy()
            if scenario.synapse_type == "ExpSyn":
                steps = np.floor(np.asarray(scenario.events_ms) / 0.025 + 0.5).astype(int)
                np.add.at(g_after, steps, scenario.weight_uS)
            expected = g_after[:-1] * (scenario.e_mV - nrn["v_mV"][:-1])
            np.testing.assert_allclose(-nrn["i_recorded_nA"][1:], expected, rtol=0.0, atol=1e-12)

        self._assert_each_scenario(check)


class Exp2SynNearEqualNeuronTest(unittest.TestCase):
    """``Exp2Syn`` with tau1 close to tau2 against NEURON's ``exp2syn`` (CELL-62).

    NEURON sets tau1 = 0.9999 tau2 in INITIAL when tau1 / tau2 > 0.9999, and
    BrainCell runs the same effective tau1. NEURON 9.0.2 gave at most 1.3e-13
    uS, 1.7e-9 mV and 4.0e-12 nA (delta = 1e-5); before BrainCell clamped, the
    conductance differed by 6.0e-8 uS and V by 1.0e-3 mV (main e693336). The
    states are about 3e4 times the weight, hence the wider V and current limits
    than ``ExponentialSynapseNeuronTest``.
    """

    @classmethod
    def setUpClass(cls):
        if neuron is None:
            raise unittest.SkipTest("NEURON is not installed; the Exp2Syn NEURON comparison needs it.")
        cls.braincell = run_braincell(NEAR_EQUAL_SCENARIOS)
        cls.neuron = {scenario.name: run_neuron(scenario) for scenario in NEAR_EQUAL_SCENARIOS}

    def test_conductance_voltage_and_current_agree(self):
        for scenario in NEAR_EQUAL_SCENARIOS:
            with self.subTest(scenario.name):
                bc_trace, nrn_trace = self.braincell[scenario.name], self.neuron[scenario.name]
                self.assertEqual(bc_trace["g_uS"].size, round(scenario.duration_ms / 0.025))
                self.assertEqual(nrn_trace["g_uS"].size, bc_trace["g_uS"].size)
                self.assertTrue(np.all(np.isfinite(bc_trace["g_uS"])))
                self.assertGreater(np.ptp(nrn_trace["g_uS"]), 1e-4)
                np.testing.assert_allclose(bc_trace["g_uS"], nrn_trace["g_uS"], rtol=0.0, atol=G_ATOL_US)
                np.testing.assert_allclose(bc_trace["v_mV"], nrn_trace["v_mV"], rtol=0.0, atol=1e-8)
                np.testing.assert_allclose(bc_trace["i_nA"], nrn_trace["i_nA"], rtol=0.0, atol=I_ATOL_NA)


if __name__ == "__main__":
    unittest.main()
