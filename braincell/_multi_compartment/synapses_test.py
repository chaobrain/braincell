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

import brainunit as u
import brainstate
import numpy as np

import braincell
from braincell import Branch, Cell, CVPerBranch, Morphology, NetStim, SynapseView, connect
from braincell.filter import at
from braincell.filter import LocsetMask


@braincell.mech.register_synapse("_SynapseViewTrainableExpSyn")
class _SynapseViewTrainableExpSyn(braincell.synapse.ExpSyn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = brainstate.nn.Param(self.tau)


def _population(size=2):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    return Cell(morpho, cv_policy=CVPerBranch(), pop_size=(size,))


class SynapseViewTest(unittest.TestCase):
    def test_repr_aggregates_names_by_type_without_reading_values(self):
        cell = _population()
        fast = braincell.mech.SynapseSpec("ExpSyn", name="fast")
        slow = braincell.mech.SynapseSpec("ExpSyn", name="slow")
        nmda = braincell.mech.SynapseSpec("Exp2Syn", name="nmda")
        cell.place(at("soma", 0.2), fast)
        cell.place(at("soma", 0.4), slow)
        cell.place(at("soma", 0.6), nmda)

        display = repr(cell.synapses)
        self.assertIn("ExpSyn  instances=4  names={'fast': 2, 'slow': 2}", display)
        self.assertIn("Exp2Syn  instances=2  names={'nmda': 2}", display)
        selected = repr(cell.synapses["fast"])
        self.assertIn("synapse_type=ExpSyn", selected)
        self.assertIn("names={'fast': 2}", selected)
        single = repr(cell.synapses["fast"][0])
        self.assertIn("parameters={'e':", single)
        self.assertIn("'tau':", single)
        np.testing.assert_array_equal(cell.synapses.by_name("fast").id, cell.synapses["fast"].id)

    def test_root_name_type_and_numeric_selections_are_views(self):
        cell = _population()
        pf = braincell.mech.SynapseSpec("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.SynapseSpec("ExpSyn", name="aa", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.31), pf)
        cell.place(at("soma", 0.39), aa)

        self.assertIsInstance(cell.synapses, SynapseView)
        self.assertIsInstance(cell.synapses[0], SynapseView)
        self.assertIsInstance(cell.synapses["pf"], SynapseView)
        self.assertIsInstance(cell.synapses.by_type("ExpSyn"), SynapseView)
        self.assertEqual(len(cell.synapses), 4)
        self.assertEqual(len(cell.synapses[0]), 1)
        self.assertEqual(cell.synapses.name.tolist(), ["pf", "aa", "pf", "aa"])
        self.assertEqual(cell.synapses["pf"].population_index.tolist(), [0, 1])
        self.assertEqual(cell[1].synapses.id.tolist(), [1, 3])

    def test_same_name_cannot_identify_different_types(self):
        cell = _population()
        cell.place(at("soma", 0.31), braincell.mech.SynapseSpec("ExpSyn", name="shared"))

        with self.assertRaisesRegex(ValueError, "same name.*different synapse types"):
            cell.place(at("soma", 0.39), braincell.mech.SynapseSpec("Exp2Syn", name="shared"))

    def test_one_flat_runtime_per_type_preserves_logical_instances(self):
        cell = _population()
        pf = braincell.mech.SynapseSpec("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.SynapseSpec("ExpSyn", name="aa", tau=2.0 * u.ms, e=-5.0 * u.mV)
        cell.place(at("soma", 0.31), pf)
        cell.place(at("soma", 0.39), aa)

        cell.init_state()

        layouts = [layout for layout in cell.layouts if layout.kind == "synapse:ExpSyn"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual(layouts[0].n_active, 4)
        np.testing.assert_array_equal(layouts[0].population_index, [0, 0, 1, 1])
        np.testing.assert_allclose(cell.synapses.tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(cell.synapses.e.to_decimal(u.mV), [0.0, -5.0, 0.0, -5.0])

    def test_post_init_parameter_and_state_updates_are_separate(self):
        cell = _population()
        exp = braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=1.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)
        view = cell.synapses["exp"]
        cell.init_state()

        view.set(tau=np.asarray([2.0, 3.0]) * u.ms)
        view.set_state(g=np.asarray([0.1, 0.2]) * u.uS)

        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [2.0, 3.0])
        np.testing.assert_allclose(view.g.to_decimal(u.uS), [0.1, 0.2])
        with self.assertRaisesRegex(KeyError, "state"):
            view.set(g=0.0 * u.uS)
        with self.assertRaisesRegex(KeyError, "parameter"):
            view.set_state(tau=1.0 * u.ms)

    def test_post_init_parameter_validation_is_atomic(self):
        cell = _population()
        exp2 = braincell.mech.SynapseSpec(
            "Exp2Syn",
            name="exp2",
            tau1=1.0 * u.ms,
            tau2=2.0 * u.ms,
        )
        cell.place(at("soma", 0.5), exp2)
        view = cell.synapses["exp2"]
        cell.init_state()

        view.set(tau1=3.0 * u.ms, tau2=4.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])
        np.testing.assert_allclose(view.tau2.to_decimal(u.ms), [4.0, 4.0])

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            view.set(tau1=-1.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])

        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            view.set(tau1=5.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])

    def test_reset_state_clears_pending_event_payload(self):
        cell = _population()
        exp = braincell.mech.SynapseSpec("ExpSyn", name="exp")
        cell.place(at("soma", 0.5), exp)
        cell.init_state()
        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        cell.runtime.event_buffers[layout.id].value = np.asarray([0.1, 0.2]) * u.uS

        cell.reset_state()

        np.testing.assert_array_equal(
            cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS),
            [0.0, 0.0],
        )

    def test_connection_materializes_stable_synapse_ids(self):
        cell = _population()
        exp = braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=1.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)

        connection = connect(
            "exp_input",
            source=NetStim(size=2),
            synapse=cell.synapses["exp"],
            weight=0.1 * u.uS,
        )

        np.testing.assert_array_equal(connection.synapse_id, cell.synapses["exp"].id)

    def test_logical_ids_remain_stable_when_later_placements_change_view_order(self):
        cell = _population()
        first = braincell.mech.SynapseSpec("ExpSyn", name="first", tau=1.0 * u.ms, e=0.0 * u.mV)
        later = braincell.mech.SynapseSpec("ExpSyn", name="later", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.3), first)
        original_ids = cell.synapses[first].id
        connection = connect(
            "first_input",
            source=NetStim(size=2),
            synapse=cell.synapses[first],
            weight=0.1 * u.uS,
        )

        cell.place(at("soma", 0.7), later)

        np.testing.assert_array_equal(cell.synapses[first].id, original_ids)
        np.testing.assert_array_equal(connection.synapse_id, original_ids)
        self.assertEqual(cell.synapses.name.tolist(), ["first", "later", "first", "later"])
        self.assertEqual(cell.synapses.id.tolist(), [0, 2, 1, 3])

    def test_per_cell_ragged_locsets_accept_flat_and_per_cell_parameters(self):
        cell = _population()
        locations = (
            LocsetMask.from_columns([0, 0], [0.2, 0.4]),
            LocsetMask.from_columns([0], [0.7]),
        )
        exp = braincell.mech.SynapseSpec(
            "ExpSyn",
            name="ragged",
            tau=np.asarray([1.0, 2.0, 3.0]) * u.ms,
            e=np.asarray([[-70.0], [-60.0]]) * u.mV,
        )

        cell.place(locations, exp)
        view = cell.synapses[exp]
        cell.init_state()

        self.assertEqual(view.population_index.tolist(), [0, 0, 1])
        np.testing.assert_allclose(view.branch_x, [0.2, 0.4, 0.7])
        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(view.e.to_decimal(u.mV), [-70.0, -70.0, -60.0])

    def test_rectangular_per_cell_locsets_support_location_axis_broadcast(self):
        cell = _population()
        locations = (
            LocsetMask.from_columns([0, 0], [0.2, 0.4]),
            LocsetMask.from_columns([0, 0], [0.6, 0.8]),
        )
        exp = braincell.mech.SynapseSpec(
            "ExpSyn",
            name="rectangular",
            tau=np.asarray([1.0, 2.0]) * u.ms,
            e=0.0 * u.mV,
        )

        cell.place(locations, exp)
        cell.init_state()

        np.testing.assert_allclose(cell.synapses[exp].tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])

    def test_locset_batch_supports_location_and_population_parameter_axes(self):
        cell = _population()
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0]])]
        exp = braincell.mech.SynapseSpec(
            "ExpSyn",
            name="batch",
            tau=np.asarray([1.0, 2.0]) * u.ms,
            e=np.asarray([[-70.0], [-60.0]]) * u.mV,
        )

        cell.place(locations, exp)
        cell.init_state()

        view = cell.synapses[exp]
        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(view.e.to_decimal(u.mV), [-70.0, -70.0, -60.0, -60.0])

    def test_per_cell_locsets_allow_empty_rows(self):
        cell = _population()
        locations = (
            LocsetMask(),
            LocsetMask.from_columns([0], [0.5]),
        )
        exp = braincell.mech.SynapseSpec("ExpSyn", name="sparse", tau=2.0 * u.ms, e=0.0 * u.mV)

        cell.place(locations, exp)

        self.assertEqual(cell.synapses[exp].population_index.tolist(), [1])

    def test_bound_input_targets_one_name_inside_shared_type_runtime(self):
        cell = _population()
        pf = braincell.mech.SynapseSpec("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.SynapseSpec("ExpSyn", name="aa", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.3), pf)
        cell.place(at("soma", 0.7), aa)
        cell.bind_synapse_input("pf", np.asarray([1.0, 2.0]), weight=0.1 * u.uS)
        cell.init_state()

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        cell.runtime.get_runtime_node(layout.id)
        np.testing.assert_allclose(
            cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS),
            [0.1, 0.0, 0.2, 0.0],
        )

    def test_post_init_set_preserves_trainable_parameter_wrapper(self):
        cell = _population()
        spec = braincell.mech.SynapseSpec(
            "_SynapseViewTrainableExpSyn",
            name="trainable",
            tau=1.0 * u.ms,
            e=0.0 * u.mV,
        )
        cell.place(at("soma", 0.5), spec)
        cell.init_state()
        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        node = cell.runtime.get_runtime_node(layout.id)
        parameter = node.tau

        cell.synapses["trainable"].set(tau=np.asarray([2.0, 3.0]) * u.ms)

        self.assertIs(node.tau, parameter)
        self.assertIsInstance(node.tau, brainstate.nn.Param)
        np.testing.assert_allclose(node.tau.value().to_decimal(u.ms), [2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
