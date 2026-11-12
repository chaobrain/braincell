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

"""Tests for the three reference reduction models.

Each model computes ``candidate = alpha * state + drive`` per population
member, emits one event when ``candidate > threshold`` (strictly), reports the
pre-reset ``candidate`` as ``values["value"]`` and resets the state to zero
after an event. The drive is the number of nonzero slots
(``EventAccumulatorReduction``), the sum of scalar payloads in the threshold's
unit (``PayloadAccumulatorReduction``), or the payload times the synapse's
analytic conductance-kernel area (``SynapticKernelAccumulatorReduction``). The
expected values below are worked out by hand from that rule.
"""

import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import at

_EVENT = braincell.EventAccumulatorReduction
_PAYLOAD = braincell.PayloadAccumulatorReduction
_KERNEL = braincell.SynapticKernelAccumulatorReduction
_MODELS = (_EVENT, _PAYLOAD, _KERNEL)

# Area of the peak-normalised Exp2Syn kernel for tau1 = 0.5 ms, tau2 = 5 ms,
# factor * (tau2 - tau1) in ms; see test_exp2syn_area_reference_matches_quadrature.
_EXP2SYN_AREA_MS = 6.4577483251


class _Owner:
    """Stand-in for the owning Cell: a context reads ``pop_size`` and keeps a weak reference."""

    def __init__(self, pop_size=(2,)) -> None:
        self.pop_size = pop_size


def _schema(
    event_input=None,
    *,
    layout_id=0,
    synapse_type="ExpSyn",
    synapse_id=(0, 1, 2),
    synapse_index=(0, 1, 0),
    population_index=(0, 0, 1),
):
    """One packed group; by default 3 slots, two on member 0 and one on member 1."""
    return braincell.ReductionInputGroupSchema(
        layout_id=layout_id,
        synapse_type=synapse_type,
        event_input=braincell.mech.ScalarEventInput(u.uS) if event_input is None else event_input,
        synapse_id=np.asarray(synapse_id),
        synapse_index=np.asarray(synapse_index),
        population_index=np.asarray(population_index),
    )


def _synapse(synapse_id, population_index, synapse_type, **parameters):
    return braincell.ReductionSynapse(
        id=synapse_id,
        synapse_index=0,
        population_index=population_index,
        placement_id=synapse_id,
        point_id=0,
        cv_id=0,
        branch_id=0,
        branch_x=0.5,
        name=f"syn{synapse_id}",
        synapse_type=synapse_type,
        parameters=parameters,
    )


def _context(owner, *groups, synapses=()):
    return braincell.ReductionContext.with_cell(owner, synapses=synapses, input_groups=groups, fingerprint="toy-test")


def _inputs(*pairs):
    return braincell.ReductionInputs(tuple(braincell.ReductionInputGroup(schema, payload) for schema, payload in pairs))


def _kernel_context(owner, **tau):
    """ExpSyn (``tau``) on member 0 and Exp2Syn (``tau1``, ``tau2``) on member 1, one group each."""
    synapses = (
        _synapse(0, 0, "ExpSyn", tau=tau.get("tau", 2.0 * u.ms)),
        _synapse(1, 1, "Exp2Syn", tau1=tau.get("tau1", 0.5 * u.ms), tau2=tau.get("tau2", 5.0 * u.ms)),
    )
    groups = (
        _schema(layout_id=0, synapse_type="ExpSyn", synapse_id=[0], synapse_index=[0], population_index=[0]),
        _schema(layout_id=1, synapse_type="Exp2Syn", synapse_id=[1], synapse_index=[0], population_index=[1]),
    )
    return _context(owner, *groups, synapses=synapses), groups


class _OutputAssertions(unittest.TestCase):
    def assert_output(self, output, value, event, *, unit=None) -> None:
        self.assertEqual(list(output.values), ["value"])
        raw = output.values["value"]
        if unit is None:
            self.assertNotIsInstance(raw, u.Quantity)
            magnitude = np.asarray(raw)
        else:
            self.assertEqual(u.get_unit(raw), unit)
            magnitude = np.asarray(raw.to_decimal(unit))
        np.testing.assert_allclose(magnitude, value, rtol=1e-6, atol=0.0)
        self.assertEqual(output.event.dtype, jnp.int32)
        np.testing.assert_array_equal(output.event, event)


class EventAccumulatorRuleTest(_OutputAssertions):
    # alpha = 0.5, threshold = 1.5; member 0 owns slots 0 and 1, member 1 owns slot 2.
    # (payload in uS, value, event)
    _STEPS = (
        ([0.2, 0.0, 0.0], [1.0, 0.0], [0, 0]),
        # Member 0 has two active slots: drive 2 whatever their magnitudes. 0.5 * 1 + 2 = 2.5 > 1.5 fires,
        # reports the pre-reset 2.5 and resets member 0 to 0.
        ([0.3, 0.7, 5.0], [2.5, 1.0], [1, 0]),
        # 0.5 * 1 + 1 = 1.5 is not strictly greater than 1.5: no event, no reset.
        ([0.0, 0.0, 5.0], [0.0, 1.5], [0, 0]),
        ([0.0, 0.0, 0.0], [0.0, 0.75], [0, 0]),
    )

    def _run_steps(self, model, schema) -> None:
        for step, (payload, value, event) in enumerate(self._STEPS, start=1):
            with self.subTest(step=step):
                self.assert_output(model.update(_inputs((schema, payload * u.uS))), value, event)

    def test_counts_active_slots_with_decay_strict_threshold_and_reset(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _EVENT(alpha=0.5, threshold=1.5)

        self.assert_output(model.init_state(_context(owner, schema)), [0.0, 0.0], [0, 0])
        self._run_steps(model, schema)
        self.assert_output(model.reset_state(), [0.0, 0.0], [0, 0])
        self._run_steps(model, schema)

    def test_batch_elements_do_not_share_state(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _EVENT(alpha=0.0, threshold=0.5)

        initial = model.init_state(_context(owner, schema), batch_size=2)
        self.assertEqual(initial.values["value"].shape, (2, 2))
        self.assertEqual(initial.event.shape, (2, 2))
        output = model.update(_inputs((schema, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] * u.uS)))
        self.assert_output(output, [[1.0, 0.0], [0.0, 1.0]], [[1, 0], [0, 1]])

    def test_batch_differing_from_members_and_slots_keeps_axes(self) -> None:
        # Batch 4, 2 members, 3 slots, every batch element different, with decay so a
        # batch-axis mix-up in the state would change step 2.
        owner = _Owner((2,))
        schema = _schema()
        model = _EVENT(alpha=0.5, threshold=1.5)

        initial = model.init_state(_context(owner, schema), batch_size=4)
        self.assert_output(initial, np.zeros((4, 2)), np.zeros((4, 2)))
        step1 = [[1, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 0]]
        self.assert_output(
            model.update(_inputs((schema, step1 * u.uS))),
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 0.0]],
            [[0, 0], [0, 0], [1, 0], [0, 0]],
        )
        step2 = [[1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0]]
        self.assert_output(
            model.update(_inputs((schema, step2 * u.uS))),
            [[1.5, 1.0], [0.0, 1.5], [0.0, 1.5], [2.5, 0.0]],
            [[0, 0], [0, 0], [0, 0], [1, 0]],
        )
        self.assertEqual(model.reset_state(batch_size=4).values["value"].shape, (4, 2))

    def test_counts_slots_of_any_input_contract(self) -> None:
        # A slot counts once whatever its payload: a trigger slot that merged two events,
        # or a scalar payload in any unit. Member 0: trigger slot 0 and the mV slot; member 1: trigger slot 1.
        owner = _Owner((2,))
        trigger = _schema(
            braincell.mech.TriggerEventInput(), synapse_id=[0, 1], synapse_index=[0, 0], population_index=[0, 1]
        )
        voltage = _schema(
            braincell.mech.ScalarEventInput(u.mV), layout_id=1, synapse_id=[2], synapse_index=[1], population_index=[0]
        )
        model = _EVENT(threshold=10.0)
        model.init_state(_context(owner, trigger, voltage))

        output = model.update(_inputs((trigger, jnp.asarray([2, 1])), (voltage, [0.3] * u.mV)))
        self.assert_output(output, [2.0, 1.0], [0, 0])

    def test_empty_inputs_only_decay(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _EVENT(alpha=0.5, threshold=10.0)
        model.init_state(_context(owner, schema))
        model.update(_inputs((schema, [1.0, 1.0, 1.0] * u.uS)))

        self.assert_output(model.update(braincell.ReductionInputs()), [1.0, 0.5], [0, 0])


class PayloadAccumulatorTest(_OutputAssertions):
    def test_sums_payloads_in_the_threshold_unit(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _PAYLOAD(alpha=0.5, threshold=0.4 * u.uS)

        self.assert_output(model.init_state(_context(owner, schema)), [0.0, 0.0], [0, 0], unit=u.uS)
        # 0.2 + 0.1 uS on member 0; nothing on member 1.
        self.assert_output(model.update(_inputs((schema, [200.0, 100.0, 0.0] * u.nS))), [0.3, 0.0], [0, 0], unit=u.uS)
        # Member 0: 0.5 * 0.3 + 0.1 = 0.25; member 1: 0.5 > 0.4 fires.
        self.assert_output(model.update(_inputs((schema, [100.0, 0.0, 500.0] * u.nS))), [0.25, 0.5], [0, 1], unit=u.uS)

    def test_value_takes_the_unit_of_the_threshold(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _PAYLOAD(threshold=400.0 * u.nS)
        model.init_state(_context(owner, schema))

        self.assert_output(model.update(_inputs((schema, [0.2, 0.1, 0.5] * u.uS))), [300.0, 500.0], [0, 1], unit=u.nS)

    def test_threshold_is_per_member(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _PAYLOAD(threshold=1.0 * u.uS)
        model.set((1,), threshold=0.1 * u.uS)
        model.init_state(_context(owner, schema))

        self.assert_output(model.update(_inputs((schema, [0.2, 0.0, 0.2] * u.uS))), [0.2, 0.2], [0, 1], unit=u.uS)
        np.testing.assert_allclose(model.get("threshold", (0, 1)).to_decimal(u.uS), [1.0, 0.1])


class SynapticKernelAccumulatorTest(_OutputAssertions):
    def test_exp2syn_area_reference_matches_quadrature(self) -> None:
        # Independent of the model's closed form: peak and integral of
        # exp(-t / tau2) - exp(-t / tau1) on a 1e-3 ms grid over 40 tau2.
        t = np.arange(0.0, 200.0 + 5e-4, 1e-3)
        kernel = np.exp(-t / 5.0) - np.exp(-t / 0.5)
        area = (kernel[:-1] + kernel[1:]).sum() * 1e-3 / 2.0 / kernel.max()

        np.testing.assert_allclose(area, _EXP2SYN_AREA_MS, rtol=1e-7)

    def test_payload_is_scaled_by_the_kernel_area_of_each_type(self) -> None:
        owner = _Owner((2,))
        context, (exp, exp2) = _kernel_context(owner)
        model = _KERNEL(threshold=100.0 * u.uS * u.ms)

        self.assert_output(model.init_state(context), [0.0, 0.0], [0, 0], unit=u.uS * u.ms)
        output = model.update(_inputs((exp, [1.0] * u.uS), (exp2, [1.0] * u.uS)))
        # ExpSyn: tau = 2 ms.
        self.assert_output(output, [2.0, _EXP2SYN_AREA_MS], [0, 0], unit=u.uS * u.ms)

    def test_reset_then_new_context_recompiles_the_areas(self) -> None:
        owner = _Owner((2,))
        context, (exp, exp2) = _kernel_context(owner)
        model = _KERNEL(threshold=100.0 * u.uS * u.ms)
        model.init_state(context)
        model.reset()

        context, (exp, exp2) = _kernel_context(owner, tau=4.0 * u.ms)
        model.init_state(context)
        output = model.update(_inputs((exp, [0.5] * u.uS), (exp2, [1.0] * u.uS)))
        self.assert_output(output, [2.0, _EXP2SYN_AREA_MS], [0, 0], unit=u.uS * u.ms)

    def test_area_matches_the_detailed_synapse_conductance_integral(self) -> None:
        # One event at t = 0 with weight 1 uS. The detailed Cell's sum(g) * dt is a
        # rectangle sum: for ExpSyn its offset from tau is about dt / (2 tau) = 6.25e-3
        # at dt = 0.025 ms; for Exp2Syn g(0) = 0 and the first-order term vanishes.
        dt = 0.025 * u.ms

        def cell():
            soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
            cell = braincell.Cell(
                braincell.Morphology.from_root(soma, name="soma"), pop_size=2, cv_policy=braincell.CVPerBranch(1)
            )
            cell[0].place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms))
            cell[1].place(
                at("soma", 0.5),
                braincell.mech.Synapse("Exp2Syn", name="exp2", tau1=0.5 * u.ms, tau2=5.0 * u.ms),
            )
            return cell

        def run(cell, duration):
            network = braincell.Network("kernel-area")
            source = network.add_population("source", braincell.NetStim(size=1, start=0.0 * u.ms, number=1))
            target = network.add_population("target", cell)
            network.connect("exp", source=source, synapse=target.synapses["exp"], weight=1.0 * u.uS)
            network.connect("exp2", source=source, synapse=target.synapses["exp2"], weight=1.0 * u.uS)
            return network.run(dt=dt, duration=duration).samples["target"]

        detailed = cell()
        detailed.record("g", braincell.observe.synapse(name="exp").state("g"))
        detailed.record("A", braincell.observe.synapse(name="exp2").state("A"))
        detailed.record("B", braincell.observe.synapse(name="exp2").state("B"))
        samples = run(detailed, 100.0 * u.ms)

        def integral(trace):
            return float(np.sum(np.asarray(u.Quantity(trace).to_decimal(u.uS), dtype=np.float64))) * 0.025

        detailed_exp = integral(samples["g"].values)
        detailed_exp2 = integral(samples["B"].values) - integral(samples["A"].values)

        reduced = cell()
        reduced.add_reduction("kernel", _KERNEL(threshold=100.0 * u.uS * u.ms))
        reduced.use_model("kernel")
        reduced.record("value", braincell.observe.output("value"))
        first = np.asarray(run(reduced, dt)["value"].values[0].to_decimal(u.uS * u.ms), dtype=np.float64)

        self.assertLess(abs(detailed_exp - first[0]) / first[0], 1e-2)
        self.assertLess(abs(detailed_exp2 - first[1]) / first[1], 1e-4)


class LifecycleTest(_OutputAssertions):
    def test_update_and_reset_state_require_init_state(self) -> None:
        for model_type in _MODELS:
            model = model_type()
            with self.subTest(model=model_type.__name__):
                with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
                    model.update(braincell.ReductionInputs())
                with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
                    model.reset_state()

    def test_reset_drops_the_context_and_keeps_parameters(self) -> None:
        owner = _Owner((2,))
        schema = _schema()
        model = _EVENT(alpha=0.5, threshold=10.0)
        model.set((1,), alpha=0.25)
        model.init_state(_context(owner, schema))
        model.update(_inputs((schema, [1.0, 0.0, 1.0] * u.uS)))

        self.assertIsNone(model.reset())
        with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
            model.update(braincell.ReductionInputs())
        with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
            model.reset_state()
        self.assertIsNone(model.reset())
        np.testing.assert_allclose(model.get("alpha", (0, 1)), [0.5, 0.25])

        # A new context with 3 members, one slot each: the retained per-member alpha applies.
        bigger = _Owner((3,))
        schema = _schema(synapse_id=[0, 1, 2], synapse_index=[0, 0, 0], population_index=[0, 1, 2])
        self.assert_output(model.init_state(_context(bigger, schema)), [0.0, 0.0, 0.0], [0, 0, 0])
        self.assert_output(model.update(_inputs((schema, [1.0, 1.0, 1.0] * u.uS))), [1.0, 1.0, 1.0], [0, 0, 0])
        self.assert_output(model.update(braincell.ReductionInputs()), [0.5, 0.25, 0.5], [0, 0, 0])


class ConstructionValidationTest(unittest.TestCase):
    def test_event_accumulator_rejects_invalid_parameters(self) -> None:
        cases = (
            ({"alpha": 1.5}, ValueError, "between 0 and 1 inclusive"),
            ({"alpha": -0.1}, ValueError, "between 0 and 1 inclusive"),
            ({"alpha": True}, TypeError, "finite scalar"),
            ({"alpha": float("nan")}, TypeError, "finite scalar"),
            ({"alpha": [0.5]}, TypeError, "finite scalar"),
            ({"threshold": -1.0}, ValueError, ">= 0"),
            ({"threshold": True}, TypeError, "finite scalar"),
            ({"threshold": float("inf")}, TypeError, "finite scalar"),
        )
        for kwargs, error, message in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(error, message):
                _EVENT(**kwargs)

    def test_alpha_bounds_and_zero_threshold_are_accepted(self) -> None:
        for alpha in (0.0, 1.0):
            np.testing.assert_allclose(_EVENT(alpha=alpha, threshold=0.0).get("alpha", (0,)), [alpha])

    def test_quantity_thresholds_reject_invalid_values(self) -> None:
        cases = (
            (_PAYLOAD, np.inf * u.uS, TypeError, "must be finite"),
            (_PAYLOAD, -1.0 * u.uS, ValueError, ">= 0"),
            (_PAYLOAD, [0.1, 0.2] * u.uS, TypeError, "scalar quantity"),
            (_KERNEL, -1.0 * u.uS * u.ms, ValueError, ">= 0"),
        )
        for model_type, threshold, error, message in cases:
            with self.subTest(model=model_type.__name__, threshold=threshold), self.assertRaisesRegex(error, message):
                model_type(threshold=threshold)


class ParameterSetGetTest(unittest.TestCase):
    def test_value_shape_must_match_the_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must be scalar or shape \(2,\), got \(3,\)"):
            _EVENT().set((0, 1), alpha=[0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, r"must be scalar or shape \(2,\), got \(3,\)"):
            _PAYLOAD().set((0, 1), threshold=[1.0, 2.0, 3.0] * u.uS)

    def test_unknown_parameter_names(self) -> None:
        model = _EVENT()

        with self.assertRaisesRegex(KeyError, "beta"):
            model.set((0,), beta=0.1)
        with self.assertRaisesRegex(KeyError, "beta"):
            model.get("beta", (0,))

    def test_an_invalid_value_writes_nothing(self) -> None:
        model = _EVENT()

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            model.set((0, 1), alpha=[0.5, 2.0])
        np.testing.assert_array_equal(model.get("alpha", (0, 1)), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, ">= 0"):
            model.set((0,), alpha=0.5, threshold=-1.0)
        with self.assertRaises(KeyError):
            model.set((0,), alpha=0.5, beta=1.0)
        np.testing.assert_array_equal(model.get("alpha", (0, 1)), [0.0, 0.0])

    def test_values_are_scattered_to_the_selected_members(self) -> None:
        model = _EVENT()
        model.set((1, 3), alpha=[0.25, 0.75])
        np.testing.assert_allclose(model.get("alpha", (0, 1, 2, 3)), [0.0, 0.25, 0.0, 0.75])

        model.set((0, 2), alpha=0.5)
        np.testing.assert_allclose(model.get("alpha", (0, 1, 2, 3)), [0.5, 0.25, 0.5, 0.75])

    def test_quantity_values_are_returned_in_the_default_unit(self) -> None:
        model = _PAYLOAD()
        model.set((1,), threshold=500.0 * u.nS)

        threshold = model.get("threshold", (0, 1))
        self.assertEqual(u.get_unit(threshold), u.uS)
        np.testing.assert_allclose(threshold.to_decimal(u.uS), [1.0, 0.5])
        model.set((0, 1), threshold=[0.5, 0.2] * u.uS)
        np.testing.assert_allclose(model.get("threshold", (0, 1)).to_decimal(u.uS), [0.5, 0.2])
        with self.assertRaisesRegex(ValueError, "compatible with uS"):
            model.set((0,), threshold=2.0 * u.mV)


class InputValidationTest(unittest.TestCase):
    def test_incompatible_payload_units_are_rejected(self) -> None:
        owner = _Owner((2,))
        context, _ = _kernel_context(owner)
        mv = _schema(braincell.mech.ScalarEventInput(u.mV), synapse_id=[0], synapse_index=[0], population_index=[0])
        context = _context(owner, mv, synapses=context.synapses)

        for model_type in (_PAYLOAD, _KERNEL):
            with self.subTest(model=model_type.__name__):
                with self.assertRaisesRegex(ValueError, "threshold units are incompatible"):
                    model_type().init_state(context)

    def test_kernel_requires_scalar_event_input(self) -> None:
        owner = _Owner((2,))
        context, _ = _kernel_context(owner)
        trigger = _schema(braincell.mech.TriggerEventInput(), synapse_id=[0], synapse_index=[0], population_index=[0])

        with self.assertRaisesRegex(TypeError, "requires ScalarEventInput"):
            _KERNEL().init_state(_context(owner, trigger, synapses=context.synapses))

    def test_kernel_validates_time_constants(self) -> None:
        owner = _Owner((1,))
        schema = _schema(synapse_id=[0], synapse_index=[0], population_index=[0])
        cases = (
            ({}, "missing required parameter 'tau'"),
            ({"tau": 2.0 * u.mV}, "must have time units"),
            ({"tau": 2.0}, "must have time units"),
            ({"tau": 0.0 * u.ms}, r"must be finite and > 0"),
            ({"tau": -1.0 * u.ms}, r"must be finite and > 0"),
            ({"tau": np.inf * u.ms}, r"must be finite and > 0"),
        )
        for parameters, message in cases:
            context = _context(owner, schema, synapses=(_synapse(0, 0, "ExpSyn", **parameters),))
            with self.subTest(parameters=parameters), self.assertRaisesRegex(ValueError, message):
                _KERNEL().init_state(context)

        exp2 = _schema(synapse_type="Exp2Syn", synapse_id=[0], synapse_index=[0], population_index=[0])
        context = _context(owner, exp2, synapses=(_synapse(0, 0, "Exp2Syn", tau1=0.5 * u.ms),))
        with self.assertRaisesRegex(ValueError, "missing required parameter 'tau2'"):
            _KERNEL().init_state(context)

    def test_multi_dimensional_pop_size_is_not_supported(self) -> None:
        owner = _Owner((2, 3))
        context = _context(owner)

        for model_type in _MODELS:
            with self.subTest(model=model_type.__name__):
                with self.assertRaisesRegex(NotImplementedError, "scalar or one-dimensional pop_size"):
                    model_type().init_state(context)


if __name__ == "__main__":
    unittest.main()
