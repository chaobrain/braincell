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
import jax.numpy as jnp
import numpy as np

import braincell
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import at


class _TickReduction(braincell.ReductionModel):
    def __init__(self, *, emit=False, features=1) -> None:
        self.emit = emit
        self.features = features
        self.state = None
        self.context = None

    def init_state(self, context, batch_size=None):
        self.context = context
        shape = ((batch_size,) if batch_size is not None else ()) + context.pop_size
        self.state = brainstate.ShortTermState(jnp.zeros(shape))
        return self._output()

    def update(self, inputs):
        del inputs
        self.state.value = self.state.value + 1
        return self._output()

    def reset_state(self, batch_size=None):
        shape = ((batch_size,) if batch_size is not None else ()) + self.context.pop_size
        self.state.value = jnp.zeros(shape)
        return self._output()

    def reset(self):
        self.state = None
        self.context = None

    def _output(self):
        value = self.state.value
        if self.features > 1:
            value = jnp.stack(tuple(value + index for index in range(self.features)), axis=-1)
        event = (
            jnp.ones_like(self.state.value, dtype=jnp.int32)
            if self.emit
            else jnp.zeros_like(self.state.value, dtype=jnp.int32)
        )
        return braincell.ReductionOutput({"value": value}, event)


class ReductionRuntimeTest(unittest.TestCase):
    def test_detailed_cell_exposes_full_voltage_output(self) -> None:
        cell = _cell(2)
        cell.init_state()

        self.assertEqual(tuple(cell.outputs), ("voltage",))
        self.assertEqual(cell.outputs["voltage"].shape, cell.V.value.shape)
        self.assertEqual(cell[1].outputs["voltage"].shape, (1, cell.n_cv))

    def test_registration_selection_and_population_parameter_views(self) -> None:
        cell = _cell(3)
        view = cell.add_reduction("acc", braincell.EventAccumulatorReduction(alpha=0.5, threshold=2.0))

        self.assertFalse(view.is_selected)
        cell[1:].reductions["acc"].set(threshold=[3.0, 4.0])
        np.testing.assert_allclose(cell.reductions["acc"].get("threshold"), [2.0, 3.0, 4.0])
        self.assertIs(cell.use_model("acc"), cell)
        self.assertTrue(view.is_selected)
        self.assertEqual(cell.outputs, {})

        with self.assertRaisesRegex(ValueError, "reserved"):
            cell.add_reduction("detailed", _TickReduction())

    def test_reduced_init_is_lightweight_and_outputs_support_features(self) -> None:
        cell = _cell(2)
        cell.add_reduction("tick", _TickReduction(features=2))
        cell.use_model("tick")
        cell.init_state()

        self.assertIsNone(cell._runtime)
        self.assertFalse(hasattr(cell, "V"))
        self.assertEqual(cell.outputs["value"].shape, (2, 2))
        self.assertEqual(cell[1].outputs["value"].shape, (1, 2))
        with self.assertRaisesRegex(RuntimeError, "Detailed Cell runtime"):
            _ = cell.runtime

    def test_batched_feature_outputs_preserve_batch_and_flatten_recording_rows(self) -> None:
        cell = _cell(2)
        cell.add_reduction("tick", _TickReduction(features=2))
        cell.use_model("tick")
        cell.record("features", braincell.observe.output("value"))
        cell.init_state(batch_size=3)

        self.assertEqual(cell.outputs["value"].shape, (3, 2, 2))
        self.assertEqual(cell[1].outputs["value"].shape, (3, 1, 2))
        result = cell.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(result.samples["features"].values.shape, (2, 3, 4))

    def test_output_recording_is_post_update_and_detailed_recording_is_omitted(self) -> None:
        cell = _cell(2)
        cell.soma.record("old_v", braincell.observe.state("v"))
        cell[1].record("reduced", braincell.observe.output("value"))
        cell.add_reduction("tick", _TickReduction(features=2))
        cell.use_model("tick")

        result = cell.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertEqual(tuple(result.samples), ("reduced",))
        np.testing.assert_allclose(result.samples["reduced"].values, [[1.0, 2.0], [2.0, 3.0]])
        rows = result.samples["reduced"].schema.rows
        self.assertEqual([(row.population_index, row.output_index) for row in rows], [(1, (0,)), (1, (1,))])

        cell.reset()
        cell.use_model()
        detailed = cell.run(dt=0.1 * u.ms, duration=0.1 * u.ms)
        self.assertEqual(tuple(detailed.samples), ("old_v",))
        self.assertIn("voltage", cell.outputs)

    def test_reset_state_keeps_mode_and_reset_allows_new_synapses(self) -> None:
        cell = _cell(1)
        model = _TickReduction()
        cell.add_reduction("tick", model)
        cell.use_model("tick")
        cell.run(dt=0.1 * u.ms, duration=0.1 * u.ms)
        cell.reset_state()

        self.assertEqual(cell._selected_model_name, "tick")
        np.testing.assert_allclose(cell.outputs["value"], [0.0])

        cell.reset()
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="later"))
        cell.init_state()
        self.assertEqual(len(model.context.synapses), 1)

    def test_context_preserves_heterogeneous_member_local_synapse_indices(self) -> None:
        cell = _cell(2)
        cell[0].place(at("soma", 0.25), braincell.mech.Synapse("ExpSyn", name="first"))
        cell[1].place(
            braincell.filter.LocsetMask.from_columns([0, 0], [0.25, 0.75]),
            braincell.mech.Synapse("ExpSyn", name="second"),
        )
        model = _TickReduction()
        cell.add_reduction("tick", model)
        cell.use_model("tick")
        cell.init_state()

        self.assertEqual(len(model.context.synapses), 3)
        self.assertEqual(
            [(item.population_index, item.synapse_index) for item in model.context.synapses],
            [(0, 0), (1, 0), (1, 1)],
        )

    def test_quantity_threshold_parameters_are_population_specific(self) -> None:
        cell = _cell(2)
        model = braincell.PayloadAccumulatorReduction(threshold=0.2 * u.uS)
        cell.add_reduction("payload", model)
        cell[1].reductions["payload"].set(threshold=500.0 * u.nS)

        np.testing.assert_allclose(
            cell.reductions["payload"].get("threshold").to_decimal(u.uS),
            [0.2, 0.5],
        )

    def test_payload_accumulator_preserves_converged_event_magnitude(self) -> None:
        network = braincell.Network("payload")
        source = network.add_population("source", braincell.NetStim(size=2, start=0.0 * u.ms, number=1))
        cell = _cell(1)
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="input"))
        cell.add_reduction("payload", braincell.PayloadAccumulatorReduction(threshold=0.4 * u.uS))
        cell.use_model("payload")
        cell.record("value", braincell.observe.output("value"))
        target = network.add_population("target", cell)
        network.connect(
            "drive",
            source=source,
            synapse=target.synapses["input"][[0, 0]],
            weight=[0.2, 0.3] * u.uS,
        )

        result = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        np.testing.assert_allclose(result.samples["target"]["value"].values.to_decimal(u.uS), [[0.5], [0.0]])
        np.testing.assert_array_equal(result.events["target"]["spike"].source_id, [0])

    def test_quantity_accumulators_validate_threshold_and_input_units(self) -> None:
        with self.assertRaisesRegex(TypeError, "scalar quantity"):
            braincell.PayloadAccumulatorReduction(threshold=0.5)
        with self.assertRaisesRegex(ValueError, "compatible with"):
            braincell.PayloadAccumulatorReduction(threshold=0.5 * u.mV)
        with self.assertRaisesRegex(ValueError, "compatible with"):
            braincell.SynapticKernelAccumulatorReduction(threshold=0.5 * u.uS)

        class Owner:
            pop_size = (1,)

        owner = Owner()
        schema = braincell.ReductionInputGroupSchema(
            layout_id=0,
            synapse_type="TriggerOnly",
            event_input=braincell.mech.TriggerEventInput(),
            synapse_id=np.asarray([0]),
            synapse_index=np.asarray([0]),
            population_index=np.asarray([0]),
        )
        context = braincell.ReductionContext.with_cell(
            owner,
            synapses=(),
            input_groups=(schema,),
            fingerprint="trigger",
        )
        with self.assertRaisesRegex(TypeError, "requires ScalarEventInput"):
            braincell.PayloadAccumulatorReduction().init_state(context)

    def test_synaptic_kernel_rejects_types_without_an_area_rule(self) -> None:
        class Owner:
            pop_size = (1,)

        owner = Owner()
        synapse = braincell.ReductionSynapse(
            id=0,
            synapse_index=0,
            population_index=0,
            placement_id=0,
            point_id=0,
            cv_id=0,
            branch_id=0,
            branch_x=0.5,
            name="custom",
            synapse_type="CustomSynapse",
        )
        schema = braincell.ReductionInputGroupSchema(
            layout_id=0,
            synapse_type="CustomSynapse",
            event_input=braincell.mech.ScalarEventInput(u.uS),
            synapse_id=np.asarray([0]),
            synapse_index=np.asarray([0]),
            population_index=np.asarray([0]),
        )
        context = braincell.ReductionContext.with_cell(
            owner,
            synapses=(synapse,),
            input_groups=(schema,),
            fingerprint="custom",
        )

        with self.assertRaisesRegex(TypeError, "supports only ExpSyn and Exp2Syn"):
            braincell.SynapticKernelAccumulatorReduction().init_state(context)

    def test_synaptic_kernel_uses_type_specific_analytic_area(self) -> None:
        network = braincell.Network("kernel")
        source = network.add_population("source", braincell.NetStim(size=2, start=0.0 * u.ms, number=1))
        cell = _cell(2)
        cell[0].place(
            at("soma", 0.5),
            braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms),
        )
        cell[1].place(
            at("soma", 0.5),
            braincell.mech.Synapse(
                "Exp2Syn",
                name="exp2",
                tau1=0.5 * u.ms,
                tau2=5.0 * u.ms,
            ),
        )
        model = braincell.SynapticKernelAccumulatorReduction(threshold=10.0 * u.uS * u.ms)
        cell.add_reduction("kernel", model)
        cell.use_model("kernel")
        cell.record("value", braincell.observe.output("value"))
        target = network.add_population("target", cell)
        network.connect(
            "drive_exp",
            source=source.event_outputs["spike"][0],
            synapse=target.synapses["exp"],
            weight=0.2 * u.uS,
        )
        network.connect(
            "drive_exp2",
            source=source.event_outputs["spike"][1],
            synapse=target.synapses["exp2"],
            weight=0.2 * u.uS,
        )

        result = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        tau1 = 0.5
        tau2 = 5.0
        t_peak = tau1 * tau2 / (tau2 - tau1) * np.log(tau2 / tau1)
        factor = 1.0 / (np.exp(-t_peak / tau2) - np.exp(-t_peak / tau1))
        expected = [0.2 * 2.0, 0.2 * factor * (tau2 - tau1)]
        np.testing.assert_allclose(
            result.samples["target"]["value"].values[0].to_decimal(u.uS * u.ms),
            expected,
            rtol=1e-6,
        )
        self.assertIsNone(cell._runtime)

    def test_scheduled_source_reaches_packed_inputs_and_emits_canonical_events(self) -> None:
        network = braincell.Network("scheduled")
        source = network.add_population("source", braincell.NetStim(size=2, start=0.0 * u.ms, number=1))
        cell = _cell(2)
        cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="input"))
        model = braincell.EventAccumulatorReduction(threshold=0.5)
        cell.add_reduction("acc", model)
        cell.use_model("acc")
        cell.record("value", braincell.observe.output("value"))
        target = network.add_population("target", cell)
        network.connect("drive", source=source, synapse=target.synapses["input"], weight=0.2 * u.uS)

        result = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        np.testing.assert_allclose(result.samples["target"]["value"].values, [[1.0, 1.0], [0.0, 0.0]])
        np.testing.assert_array_equal(result.events["target"]["spike"].source_id, [0, 1])
        self.assertEqual(len(model._context.input_groups), 1)
        np.testing.assert_array_equal(model._context.input_groups[0].population_index, [0, 1])

    def test_live_zero_delay_event_is_consumed_on_following_step(self) -> None:
        network = braincell.Network("live")
        pre_cell = _cell(1)
        pre_cell.add_reduction("tick", _TickReduction(emit=True))
        pre_cell.use_model("tick")
        pre = network.add_population("pre", pre_cell)

        post_cell = _cell(1)
        post_cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="input"))
        post_cell.add_reduction("acc", braincell.EventAccumulatorReduction(threshold=10.0))
        post_cell.use_model("acc")
        post_cell.record("value", braincell.observe.output("value"))
        post = network.add_population("post", post_cell)
        network.connect("drive", source=pre.event_outputs["spike"], synapse=post.synapses["input"])

        result = network.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(result.samples["post"]["value"].values[:, 0], [0.0, 1.0, 1.0])

    def test_voltage_crossing_source_is_rejected_in_reduced_mode(self) -> None:
        cell = _cell(1)
        source = braincell.VoltageCrossingSource(cell.soma)
        cell.add_reduction("tick", _TickReduction())
        cell.use_model("tick")
        cell.init_state()

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            source.current_event_count([0])


if __name__ == "__main__":
    unittest.main()
