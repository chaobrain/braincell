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

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import at
from braincell.mech import ScalarEventInput


class SynapseSchemaTest(unittest.TestCase):
    def test_expsyn_signature_is_the_only_parameter_source(self) -> None:
        import inspect

        model = braincell.synapse.ExpSyn
        self.assertIn("tau", inspect.signature(model.__init__).parameters)
        self.assertEqual(tuple(model.parameter_info()), ("tau", "e"))
        self.assertEqual(tuple(model.states), ("g",))
        self.assertEqual(model.event_input, ScalarEventInput(u.uS, aggregation="sum"))
        self.assertNotIn("weight", model.parameter_info())
        self.assertFalse(hasattr(model, "event_weight_unit"))
        self.assertFalse(hasattr(model, "current_sign"))
        self.assertFalse(hasattr(model, "current_units"))

    def test_physical_validation_is_not_a_learning_transform(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            braincell.synapse.ExpSyn(1, tau=0.0 * u.ms)
        field = braincell.synapse.ExpSyn.parameter_info()["tau"]
        self.assertFalse(hasattr(field, "trainable"))
        self.assertFalse(hasattr(field, "transform"))

    def test_retired_receptor_names_are_gone(self) -> None:
        for name in ("AMPA", "GABAa", "NMDA"):
            self.assertFalse(hasattr(braincell.synapse, name))


class ExpSynTest(unittest.TestCase):
    def test_event_and_current_are_vectorized(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=[1.0, 2.0] * u.ms)
        synapse.init_state()
        synapse.apply_events([0.01, 0.02] * u.uS)

        np.testing.assert_allclose(synapse.g.value.to_decimal(u.uS), [0.01, 0.02])
        np.testing.assert_allclose(
            synapse.current([-65.0, -50.0] * u.mV).to_decimal(u.nA),
            [0.65, 1.0],
        )

    def test_conductance_decays_with_its_own_tau(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=[1.0, 4.0] * u.ms)
        synapse.init_state()
        synapse.apply_events([0.02, 0.02] * u.uS)
        synapse.compute_derivative()

        np.testing.assert_allclose(
            synapse.g.derivative.to_decimal(u.uS / u.ms),
            [-0.02 / 1.0, -0.02 / 4.0],
        )

    def test_reset_state_returns_g_to_the_declared_initial(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=1.0 * u.ms)
        synapse.init_state()
        synapse.apply_events([0.01, 0.02] * u.uS)
        synapse.reset_state()

        np.testing.assert_allclose(synapse.g.value.to_decimal(u.uS), [0.0, 0.0])

    def test_a_wrong_unit_payload_is_rejected(self) -> None:
        synapse = braincell.synapse.ExpSyn(2, tau=1.0 * u.ms)
        synapse.init_state()
        with self.assertRaisesRegex(ValueError, "units incompatible"):
            synapse.apply_events([0.01, 0.02] * u.mV)
        with self.assertRaisesRegex(TypeError, "must be a quantity"):
            synapse.apply_events([0.01, 0.02])


class Exp2SynTest(unittest.TestCase):
    TAU1 = 0.5
    TAU2 = 5.0

    def _synapse(self, size: int = 1):
        synapse = braincell.synapse.Exp2Syn(
            size,
            tau1=self.TAU1 * u.ms,
            tau2=self.TAU2 * u.ms,
        )
        synapse.init_state()
        return synapse

    def test_canonical_time_order_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            braincell.synapse.Exp2Syn(1, tau1=2.0 * u.ms, tau2=1.0 * u.ms)

    def test_time_order_compares_physical_durations_not_magnitudes(self) -> None:
        # 2000 us > 1 ms, even though 2000 > 1 is the only numeric comparison.
        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            braincell.synapse.Exp2Syn(1, tau1=2000.0 * u.us, tau2=1.0 * u.ms)
        # ...and the same pair in the right order is accepted.
        braincell.synapse.Exp2Syn(1, tau1=200.0 * u.us, tau2=1.0 * u.ms)

    def test_construction_survives_an_open_jax_trace(self) -> None:
        # Regression: ``validate_parameter_values`` used ``u.math.any``, and
        # under an open trace every jax op returns a tracer even for concrete
        # inputs, so ``bool()`` on it raised ``TracerBoolConversionError``.
        def build():
            braincell.synapse.Exp2Syn(4, tau1=self.TAU1 * u.ms, tau2=self.TAU2 * u.ms)
            return jnp.zeros(())

        jaxpr = jax.make_jaxpr(build)()
        self.assertIsNotNone(jaxpr)

    def test_conductance_is_the_difference_of_the_two_states(self) -> None:
        synapse = self._synapse(2)
        synapse.A.value = np.asarray([0.3, 0.1]) * u.uS
        synapse.B.value = np.asarray([0.9, 0.4]) * u.uS

        np.testing.assert_allclose(synapse.g.to_decimal(u.uS), [0.6, 0.3])

    def test_one_event_moves_both_states_by_the_same_amount(self) -> None:
        synapse = self._synapse(2)
        synapse.apply_events([0.01, 0.02] * u.uS)

        a = synapse.A.value.to_decimal(u.uS)
        b = synapse.B.value.to_decimal(u.uS)
        np.testing.assert_allclose(a, b)
        # g is zero at the instant of the event and rises afterwards.
        np.testing.assert_allclose(synapse.g.to_decimal(u.uS), [0.0, 0.0], atol=1e-12)

    def test_each_state_decays_with_its_own_tau_and_they_are_not_swapped(self) -> None:
        synapse = self._synapse(1)
        synapse.A.value = np.asarray([0.4]) * u.uS
        synapse.B.value = np.asarray([0.4]) * u.uS
        synapse.compute_derivative()

        np.testing.assert_allclose(synapse.A.derivative.to_decimal(u.uS / u.ms), [-0.4 / self.TAU1])
        np.testing.assert_allclose(synapse.B.derivative.to_decimal(u.uS / u.ms), [-0.4 / self.TAU2])

    def test_factor_normalizes_the_peak_conductance_to_the_payload(self) -> None:
        # The defining property of exp2syn's normalisation constant: after a
        # unit event, max_t g(t) == payload exactly. Checked twice, against
        # the closed-form peak time and against a dense sampling of g(t).
        payload = 0.02
        synapse = self._synapse(1)
        synapse.apply_events([payload] * u.uS)
        delta = float(synapse.A.value.to_decimal(u.uS)[0])

        tau1, tau2 = self.TAU1, self.TAU2
        tp = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)

        def g_of(t):
            return delta * (np.exp(-t / tau2) - np.exp(-t / tau1))

        # rtol is float32-sized: ``delta`` comes back from the default-precision
        # runtime, so the normalisation cannot be tighter than the payload's
        # own representation.
        np.testing.assert_allclose(g_of(tp), payload, rtol=1e-6)

        t = np.linspace(0.0, 10.0 * tau2, 200_001)
        np.testing.assert_allclose(float(g_of(t).max()), payload, rtol=1e-6)

    def test_current_is_inward_positive_at_the_declared_reversal(self) -> None:
        synapse = self._synapse(2)
        synapse.A.value = np.asarray([0.0, 0.0]) * u.uS
        synapse.B.value = np.asarray([0.01, 0.02]) * u.uS

        np.testing.assert_allclose(
            synapse.current([-65.0, -50.0] * u.mV).to_decimal(u.nA),
            [0.65, 1.0],
        )

    def test_reset_state_returns_both_states_to_their_declared_initials(self) -> None:
        synapse = self._synapse(2)
        synapse.apply_events([0.01, 0.02] * u.uS)
        synapse.reset_state()

        np.testing.assert_allclose(synapse.A.value.to_decimal(u.uS), [0.0, 0.0])
        np.testing.assert_allclose(synapse.B.value.to_decimal(u.uS), [0.0, 0.0])


# The reference scenarios of CELL-44 (from the CELL-17 design): a passive
# single-CV cylinder, one synapse at x = 0.5, events from an EventSequence
# through Network.connect with zero delay, dt 0.025 ms, float64.
# name, synapse type, parameters (ms, mV), event times (ms), weight (uS), duration (ms)
_DT_MS = 0.025
_SCENARIOS = (
    ("single, on grid", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0,), 0.001, 20.0),
    ("5 events, 2 ms apart", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0, 3.0, 5.0, 7.0, 9.0), 0.001, 30.0),
    ("two in one step, near-ties", "ExpSyn", {"tau": 2.0, "e": 0.0}, (1.0, 1.0, 2.0124, 2.0126, 4.0374), 0.001, 20.0),
    ("tau = 4 dt", "ExpSyn", {"tau": 0.1, "e": 0.0}, (1.0, 1.3, 1.6), 0.002, 10.0),
    ("tau 50 ms, 20 events", "ExpSyn", {"tau": 50.0, "e": 0.0}, tuple(1.0 + 2.5 * k for k in range(20)), 0.0005, 200.0),
    ("inhibitory, e below rest", "ExpSyn", {"tau": 5.0, "e": -80.0}, (1.0, 2.0, 3.0), 0.005, 40.0),
    ("e = E_leak = V_init", "ExpSyn", {"tau": 5.0, "e": -65.0}, (1.0, 2.0, 3.0), 0.01, 40.0),
    ("strong drive, V towards e", "ExpSyn", {"tau": 20.0, "e": -55.0}, tuple(1.0 + k for k in range(10)), 0.05, 60.0),
    ("Exp2Syn 0.5/5", "Exp2Syn", {"tau1": 0.5, "tau2": 5.0, "e": 0.0}, (1.0, 3.0, 3.0, 8.0126), 0.001, 40.0),
    ("Exp2Syn default 0.1/10", "Exp2Syn", {"tau1": 0.1, "tau2": 10.0, "e": 0.0}, (1.0, 6.0), 0.001, 60.0),
    ("Exp2Syn near-equal 1/1.05", "Exp2Syn", {"tau1": 1.0, "tau2": 1.05, "e": 0.0}, (1.0, 2.0), 0.001, 30.0),
    ("Exp2Syn inhibitory 1/20", "Exp2Syn", {"tau1": 1.0, "tau2": 20.0, "e": -80.0}, (1.0, 4.0), 0.003, 100.0),
)
# Exact half-step ties (step ratios 0.5, 80.5, 201.5, 320.5, 480.5; float64
# puts the last three a few ulp below the half) act on the later step.
_TIES = (
    "exact half-step ties",
    "ExpSyn",
    {"tau": 2.0, "e": 0.0},
    (0.0125, 2.0125, 5.0375, 8.0125, 12.0125),
    0.001,
    15.0,
)
_TIE_STEPS = (1, 81, 202, 321, 481)


def _reference_cell(synapse_type, name="s", **parameters):
    """The scenarios' passive cylinder (L 20 um, diameter 20 um, one CV) with one synapse at x = 0.5."""
    branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um)
    cell = braincell.Cell(
        braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1), V_init=-65.0 * u.mV
    )
    cell.paint(
        braincell.mech.CableProperty(
            resting_potential=-65.0 * u.mV,
            membrane_capacitance=1.0 * u.uF / u.cm**2,
            axial_resistivity=100.0 * u.ohm * u.cm,
        )
    )
    cell.paint(braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at(0, 0.5), braincell.mech.Synapse(synapse_type, name=name, **parameters))
    return cell


def _event_steps(times_ms):
    """Round each event time to its step boundary, away from any half-step tie."""
    ratios = np.asarray(times_ms, dtype=float) / _DT_MS
    assert np.all(np.abs(ratios - np.floor(ratios) - 0.5) > 1e-3), "use _TIE_STEPS for half-step ties"
    return np.floor(ratios + 0.5).astype(int)


def _analytic_conductance(synapse_type, parameters, steps, weight, n):
    """Superpose each event's exact response on the step grid.

    Sample k of ``Network.run`` is the state at t_k = k dt before that
    boundary's events, so an event rounded to step s shows from sample s + 1.
    """
    k = np.arange(n)
    g = np.zeros(n)
    for step in steps:
        later = k > step
        elapsed = (k[later] - step) * _DT_MS
        if synapse_type == "ExpSyn":
            g[later] += weight * np.exp(-elapsed / parameters["tau"])
        else:
            tau1, tau2 = parameters["tau1"], parameters["tau2"]
            peak = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)
            factor = 1.0 / (np.exp(-peak / tau2) - np.exp(-peak / tau1))
            g[later] += factor * weight * (np.exp(-elapsed / tau2) - np.exp(-elapsed / tau1))
    return g


class ExponentialSynapseCellReferenceTest(unittest.TestCase):
    """ExpSyn and Exp2Syn integrated in a Cell, against the exact solution on the step grid.

    Every scenario is one population of one Network, so they share one
    compilation; each is compared on its own first ``duration / dt`` samples.
    The voltage has no closed form; the NEURON comparison of
    validation/neuron/synapse checks it.
    """

    @classmethod
    def setUpClass(cls):
        scenarios = _SCENARIOS + (_TIES,)
        cls.traces = {}
        with brainstate.environ.context(precision=64):
            network = braincell.Network("exp_syn_reference")
            for index, (name, synapse_type, parameters, times, weight, _) in enumerate(scenarios):
                quantities = {key: value * (u.mV if key == "e" else u.ms) for key, value in parameters.items()}
                cell = _reference_cell(synapse_type, **quantities)
                observe = braincell.observe.synapse(name="s")
                cell.record("v", "V")
                cell.record("i", observe.current())
                for state in ("g",) if synapse_type == "ExpSyn" else ("A", "B"):
                    cell.record(state, observe.state(state))
                post = network.add_population(f"post{index}", cell)
                source = network.add_population(
                    f"events{index}", braincell.EventSequence.from_times([np.asarray(times) * u.ms])
                )
                network.connect(
                    f"drive{index}", source=source, synapse=post.synapses["s"], weight=weight * u.uS, delay=0.0 * u.ms
                )
            duration = max(scenario[-1] for scenario in scenarios)
            result = network.run(dt=_DT_MS * u.ms, duration=duration * u.ms)
        for index, (name, synapse_type, *_, duration) in enumerate(scenarios):
            samples = result.samples[f"post{index}"]
            n = round(duration / _DT_MS)

            def trace(field, unit, samples=samples, n=n):
                return np.asarray(samples[field].values.to_decimal(unit), dtype=np.float64).reshape(-1)[:n]

            if synapse_type == "ExpSyn":
                g = trace("g", u.uS)
            else:
                g = trace("B", u.uS) - trace("A", u.uS)
            cls.traces[name] = {"v": trace("v", u.mV), "i": trace("i", u.nA), "g": g}

    def _assert_conductance(self, scenario, steps):
        name, synapse_type, parameters, _, weight, _ = scenario
        g = self.traces[name]["g"]
        expected = _analytic_conductance(synapse_type, parameters, steps, weight, g.size)
        large = expected > 1e-9
        np.testing.assert_allclose(g[large], expected[large], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(g[~large], expected[~large], rtol=0.0, atol=1e-15)

    def test_conductance_is_the_event_superposition(self) -> None:
        for scenario in _SCENARIOS:
            with self.subTest(scenario[0]):
                self._assert_conductance(scenario, _event_steps(scenario[3]))

    def _increments(self, scenario):
        """The ExpSyn increment applied at each step boundary s, read from samples s and s + 1."""
        name, _, parameters, *_ = scenario
        g = self.traces[name]["g"]
        return g[1:] * np.exp(_DT_MS / parameters["tau"]) - g[:-1]

    def test_two_events_in_one_step_add_twice_the_weight(self) -> None:
        weight = _SCENARIOS[2][4]
        increment = self._increments(_SCENARIOS[2])
        steps = [40, 80, 81, 161]  # 1.0 twice, then 2.0124, 2.0126 and 4.0374 ms
        np.testing.assert_allclose(increment[steps], [2 * weight, weight, weight, weight], rtol=1e-12)
        self.assertLess(np.max(np.abs(np.delete(increment, steps))), 1e-15)

    def test_exact_half_step_ties_act_on_the_later_step(self) -> None:
        increment = self._increments(_TIES)
        self.assertEqual(np.flatnonzero(increment > 1e-9).tolist(), list(_TIE_STEPS))
        self._assert_conductance(_TIES, _TIE_STEPS)

    def test_reversal_at_rest_keeps_voltage_at_rest_and_current_zero(self) -> None:
        trace = self.traces["e = E_leak = V_init"]
        self.assertGreater(trace["g"].max(), 0.01)
        np.testing.assert_allclose(trace["v"], -65.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(trace["i"], 0.0, rtol=0.0, atol=1e-12)

    def test_strong_drive_approaches_its_reversal_without_crossing_it(self) -> None:
        v = self.traces["strong drive, V towards e"]["v"]
        self.assertGreater(v.max(), -55.1)  # the prototype's peak is -55.031310 mV
        self.assertLess(v.max(), -55.0)

    def test_current_is_conductance_times_driving_force(self) -> None:
        for name, _, parameters, *_ in _SCENARIOS + (_TIES,):
            with self.subTest(name):
                trace = self.traces[name]
                np.testing.assert_allclose(
                    trace["i"], trace["g"] * (parameters["e"] - trace["v"]), rtol=1e-12, atol=0.0
                )


class ExponentialSynapseParameterErrorTest(unittest.TestCase):
    """Invalid parameters fail at the stage the Synapse API states, naming the synapse through a Cell."""

    def test_direct_construction_rejects_a_non_positive_or_unitless_tau(self) -> None:
        for tau in (-2.0 * u.ms, 0.0 * u.ms):
            with self.subTest(tau=tau), self.assertRaisesRegex(ValueError, "'tau' must be > 0"):
                braincell.synapse.ExpSyn(1, tau=tau)
        with self.assertRaisesRegex(TypeError, "'tau' requires a quantity"):
            braincell.synapse.ExpSyn(1, tau=2.0)

    def test_direct_construction_rejects_tau1_not_below_tau2(self) -> None:
        for tau1, tau2 in ((3.0, 1.0), (2.0, 2.0)):
            with self.subTest(tau1=tau1, tau2=tau2), self.assertRaisesRegex(ValueError, "tau1 < tau2"):
                braincell.synapse.Exp2Syn(1, tau1=tau1 * u.ms, tau2=tau2 * u.ms)

    def test_a_negative_tau_through_a_cell_fails_at_init_state_naming_the_synapse(self) -> None:
        cell = _reference_cell("ExpSyn", tau=-2.0 * u.ms)  # declaring and placing it is accepted
        with self.assertRaises(ValueError) as caught:
            cell.init_state()
        message = str(caught.exception)
        self.assertIn("Synapse 's' (ExpSyn)", message)
        self.assertIn("'tau'", message)
        self.assertIn("must be > 0", message)
        self.assertIn("placed on", message.lower())

    def test_tau1_not_below_tau2_through_a_cell_fails_at_init_state_naming_the_synapse(self) -> None:
        for tau1, tau2 in ((3.0, 1.0), (2.0, 2.0)):
            with self.subTest(tau1=tau1, tau2=tau2):
                cell = _reference_cell("Exp2Syn", name="t", tau1=tau1 * u.ms, tau2=tau2 * u.ms)
                with self.assertRaises(ValueError) as caught:
                    cell.init_state()
                self.assertIn("Synapse 't' (Exp2Syn)", str(caught.exception))
                self.assertIn("tau1 < tau2", str(caught.exception))

    def test_set_after_init_state_names_the_synapse_and_keeps_the_old_value(self) -> None:
        cell = _reference_cell("ExpSyn", tau=2.0 * u.ms)
        cell.place(at(0, 0.5), braincell.mech.Synapse("Exp2Syn", name="t", tau1=1.0 * u.ms, tau2=5.0 * u.ms))
        cell.init_state()
        with self.assertRaises(ValueError) as caught:
            cell.synapses["s"].set(tau=-1.0 * u.ms)
        self.assertIn("Synapse 's' (ExpSyn)", str(caught.exception))
        self.assertIn("'tau'", str(caught.exception))
        np.testing.assert_array_equal(cell.synapses["s"].tau.to_decimal(u.ms), [2.0])
        with self.assertRaises(ValueError) as caught:
            cell.synapses["t"].set(tau1=6.0 * u.ms)
        self.assertIn("Synapse 't' (Exp2Syn)", str(caught.exception))
        self.assertIn("tau1 < tau2", str(caught.exception))
        np.testing.assert_array_equal(cell.synapses["t"].tau1.to_decimal(u.ms), [1.0])


if __name__ == "__main__":
    unittest.main()
