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

"""Unit tests for :class:`braincell.Cell`."""

import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import braincell
import braincell.mech as mech
from braincell import (
    Branch,
    CVPerBranch,
    Cell,
    CellView,
    Channel,
    CurrentClamp,
    Ion,
    IonChannel,
    Morphology,
    NetStim,
    connect,
)
from braincell._multi_compartment import cell as cell_module
from braincell.filter import AllRegion, BranchSlice, RootLocation, at
from braincell.mech import StateProbe
from braincell.quad import DiffEqSingleState, DiffEqState, get_integrator


def _soma_tree() -> Morphology:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def _simple_cell() -> Cell:
    return Cell(_soma_tree(), cv_policy=CVPerBranch())


def _cell_with_probe() -> Cell:
    cell = _simple_cell()
    cell.place(RootLocation(0.0), StateProbe(field="v", name="V_root"))
    return cell


class TestCellDeclaration(unittest.TestCase):
    def test_constructs_with_defaults(self):
        cell = _simple_cell()
        self.assertGreater(len(cell.paint_rules), 0)
        self.assertEqual(len(cell.place_rules), 0)
        self.assertFalse(cell._initialized)

    def test_subsolver_defaults_to_backward_euler_once(self):
        cell = _simple_cell()
        self.assertEqual(cell.subsolver_name, "backward_euler")
        self.assertIs(cell.subsolver, get_integrator("backward_euler"))
        self.assertEqual(cell.substeps, 1)

    def test_accepts_explicit_subsolver_schedule(self):
        cell = Cell(_simple_cell().morpho, subsolver="rk4", substeps=3)
        self.assertEqual(cell.subsolver_name, "rk4")
        self.assertIs(cell.subsolver, get_integrator("rk4"))
        self.assertEqual(cell.substeps, 3)

    def test_subsolver_schedule_must_be_an_atomic_pair(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            Cell(_simple_cell().morpho, subsolver="rk4")
        with self.assertRaisesRegex(ValueError, "provided together"):
            Cell(_simple_cell().morpho, substeps=2)

    def test_subsolver_schedule_validates_substeps(self):
        with self.assertRaises(TypeError):
            Cell(_simple_cell().morpho, subsolver="rk4", substeps=True)
        with self.assertRaises(ValueError):
            Cell(_simple_cell().morpho, subsolver="rk4", substeps=0)

    def test_rejects_non_morphology(self):
        with self.assertRaises(TypeError):
            Cell("not-a-morpho")  # type: ignore[arg-type]

    def test_rejects_unknown_ion_channel_update_order(self):
        with self.assertRaisesRegex(ValueError, "ion_channel_update_order"):
            Cell(
                Morphology.from_root(
                    Branch.from_lengths(
                        lengths=[20.0] * u.um,
                        radii=[10.0, 10.0] * u.um,
                        type="soma",
                    ),
                    name="soma",
                ),
                ion_channel_update_order="legacy",
            )

    def test_accepts_family_and_integration_update_orders(self):
        tree = Morphology.from_root(
            Branch.from_lengths(
                lengths=[20.0] * u.um,
                radii=[10.0, 10.0] * u.um,
                type="soma",
            ),
            name="soma",
        )
        self.assertEqual(
            Cell(tree, ion_channel_update_order="family").ion_channel_update_order,
            "family",
        )
        self.assertEqual(
            Cell(tree, ion_channel_update_order="integration").ion_channel_update_order,
            "integration",
        )

    def test_membrane_linearizer_defaults_to_point_and_validates(self):
        cell = _simple_cell()
        self.assertEqual(cell.membrane_linearizer, "point")
        cell.membrane_linearizer = "generic"
        self.assertEqual(cell.membrane_linearizer, "generic")
        with self.assertRaisesRegex(ValueError, "membrane_linearizer"):
            Cell(cell.morpho, membrane_linearizer="finite_difference")

    def test_membrane_linearizer_is_frozen_after_initialization(self):
        cell = _simple_cell()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "membrane_linearizer"):
            cell.membrane_linearizer = "generic"

    def test_cv_policy_mutation_invalidates_cache(self):
        cell = _simple_cell()
        _ = cell.cvs
        cell.cv_policy = CVPerBranch()
        _ = cell.cvs

    def test_paint_returns_self_for_chaining(self):
        cell = _simple_cell()
        from braincell import CableProperty

        result = cell.paint(
            cell.paint_rules[0].region,
            CableProperty(
                resting_potential=-70 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
        )
        self.assertIs(result, cell)

    def test_place_keeps_identical_rules_as_independent_instances(self):
        cell = _simple_cell()
        clamp = CurrentClamp(delay=1 * u.ms, durations=10 * u.ms, amplitudes=0.1 * u.nA)
        cell.place(RootLocation(0.0), clamp)
        cell.place(RootLocation(0.0), clamp)
        self.assertEqual(len(cell.place_rules), 2)
        placements = cell.point_placements
        self.assertEqual(len(placements), 2)
        self.assertEqual([item.id for item in placements], [0, 1])
        self.assertEqual([item.branch_x for item in placements], [0.0, 0.0])
        self.assertEqual(placements[0].point_id, placements[1].point_id)

    def test_point_placements_preserve_continuous_location_and_cv_owner(self):
        cell = _simple_cell()
        clamp = CurrentClamp(delay=1 * u.ms, durations=10 * u.ms, amplitudes=0.1 * u.nA)
        cell.place(at("soma", 0.31), clamp)
        cell.place(at("soma", 0.39), clamp)

        placements = cell.point_placements

        self.assertEqual([item.branch_name for item in placements], ["soma", "soma"])
        self.assertEqual([item.branch_type for item in placements], ["soma", "soma"])
        self.assertEqual([item.branch_x for item in placements], [0.31, 0.39])
        self.assertEqual([item.cv_id for item in placements], [0, 0])
        self.assertEqual(placements[0].point_id, placements[1].point_id)
        self.assertIs(cell.get_point_placement(1), placements[1])

    def test_population_selection_places_packed_instances(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        synapse = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms)

        cell[0].place(at("soma", 0.21), synapse)
        cell[[1, 3, 3]].place(at("soma", 0.45), synapse)
        cell[2:4].place(at("soma", 0.73), synapse)

        placements = cell.point_placements
        self.assertEqual(
            [item.population_index for item in placements],
            [0, 1, 3, 2, 3],
        )
        self.assertEqual(
            [item.branch_x for item in placements],
            [0.21, 0.45, 0.45, 0.73, 0.73],
        )

    def test_cell_view_is_public_compatible_and_composes_indices(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(5,))

        view = cell[[4, 1, 4, 2]]
        nested = view[1:]

        self.assertIsInstance(view, CellView)
        self.assertIs(view.root, cell)
        self.assertIs(view.cell, cell)
        self.assertEqual(view.population_indices, (4, 1, 2))
        np.testing.assert_array_equal(view.indices, [4, 1, 2])
        self.assertEqual(view.shape, (3,))
        self.assertEqual(view.size, 3)
        self.assertEqual(view.pop_size, (3,))
        self.assertEqual(nested.population_indices, (1, 2))
        self.assertEqual(view[-1].population_indices, (2,))

    def test_cell_view_reuses_shared_topology_objects(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        view = cell[1]

        self.assertIs(view.morpho, cell.morpho)
        self.assertIs(view.cv_policy, cell.cv_policy)
        self.assertIs(view.cvs, cell.cvs)
        self.assertIs(view.cv_tree, cell.cv_tree)
        self.assertIs(view.node_tree, cell.node_tree)
        self.assertIs(view.cv_contexts, cell.cv_contexts)
        self.assertEqual(view.varshape, (1, cell.n_cv))

    def test_cell_view_filters_effective_place_declarations(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms)
        cell.place(at("soma", 0.5), exp)
        selected = cell[[1, 3]]
        result = selected.place(at("soma", 0.25), exp)

        self.assertIs(result, selected)
        self.assertEqual(len(cell[0].place_rules), 1)
        self.assertEqual(len(cell[1].place_rules), 2)
        self.assertEqual(len(cell[0].point_placements), 1)
        self.assertEqual(len(cell[1].point_placements), 2)
        self.assertEqual(len(cell[2:2].point_placements), 0)
        self.assertEqual([item.population_index for item in cell[3].point_placements], [None, 3])

    def test_cell_view_synapses_filter_and_retain_parameter_set(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)
        cell[1].place(at("soma", 0.25), exp)

        view = cell[[1, 3]].synapses["exp"]
        view.set(tau=np.asarray([1.0, 2.0, 3.0]) * u.ms)

        self.assertEqual(view.population_index.tolist(), [1, 1, 3])
        cell.init_state()
        layouts = [item for item in cell.layouts if item.kind == "synapse:ExpSyn"]
        self.assertEqual(len(layouts), 1)
        tau = cell.get_state(layouts[0].id, "tau")
        self.assertEqual(tau.shape, (5,))
        np.testing.assert_allclose(tau.to_decimal(u.ms), [2.0, 1.0, 2.0, 2.0, 3.0])

    def test_cell_view_filters_connection_rows_by_synapse_population(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.SynapseSpec("ExpSyn", name="exp")
        cell.place(at("soma", 0.5), exp)
        connection = connect("cell_2", source=NetStim(), synapse=cell[2].synapses[exp])

        self.assertEqual(len(cell[0].connections), 0)
        np.testing.assert_array_equal(cell[2].connections.id, connection.id)

    def test_cell_view_sets_selected_initial_voltage_and_threshold(self):
        cell = Cell(
            _simple_cell().morpho,
            cv_policy=CVPerBranch(),
            pop_size=(4,),
            V_init=-65.0 * u.mV,
            V_th=-20.0 * u.mV,
        )

        result = cell[[1, 3]].set(
            V_init=np.asarray([-60.0, -55.0]) * u.mV,
            V_th=np.asarray([-10.0, 0.0]) * u.mV,
        )

        self.assertIs(result.root, cell)
        np.testing.assert_allclose(result.V_init.to_decimal(u.mV)[:, 0], [-60.0, -55.0])
        np.testing.assert_allclose(result.V_th.to_decimal(u.mV)[:, 0], [-10.0, 0.0])
        cell.init_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -60.0, -65.0, -55.0])
        np.testing.assert_allclose(cell.V_th.to_decimal(u.mV)[:, 0], [-20.0, -10.0, -20.0, 0.0])
        np.testing.assert_allclose(cell[[3, 0]].V.to_decimal(u.mV)[:, 0], [-55.0, -65.0])
        self.assertEqual(cell[[3, 0]].spike.shape, (2, cell.n_cv))

    def test_root_voltage_assignment_clears_selected_overrides(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,), V_init=-65.0 * u.mV)
        cell[1].V_init = -55.0 * u.mV
        cell.V_init = -70.0 * u.mV

        cell.init_state()

        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -70.0)

    def test_cell_view_voltage_overrides_validate_and_survive_reinitialization(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,), V_init=-65.0 * u.mV)
        view = cell[1]
        with self.assertRaisesRegex(TypeError, "voltage quantity"):
            view.set(V_init=-55.0)
        with self.assertRaisesRegex(ValueError, "voltage units"):
            view.set(V_th=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "cannot broadcast"):
            cell[[0, 2]].set(V_init=np.ones((3, 2)) * u.mV)
        with self.assertRaisesRegex(TypeError, "not None"):
            view.set(V_th=None)
        with self.assertRaisesRegex(KeyError, "does not support"):
            view.set(resting_potential=-70.0 * u.mV)

        view.set(V_init=-55.0 * u.mV)
        cell.init_state()
        cell.reset_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -55.0, -65.0])
        cell.reset()
        cell.init_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -55.0, -65.0])

    def test_cell_view_accepts_per_cv_values_without_changing_shape(self):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="basal_dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.attach(dend, name="dend")
        cell = Cell(morpho, cv_policy=CVPerBranch(), pop_size=(3,), V_init=-65.0 * u.mV)
        values = np.asarray([[-60.0, -61.0], [-50.0, -51.0]]) * u.mV

        cell[[0, 2]].V_init = values
        cell.init_state()

        self.assertEqual(cell.V.value.shape, (3, 2))
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[0], [-60.0, -61.0])
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[1], [-65.0, -65.0])
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[2], [-50.0, -51.0])

    def test_cell_view_defers_callable_initial_voltage_inspection(self):
        cell = Cell(
            _simple_cell().morpho,
            cv_policy=CVPerBranch(),
            pop_size=(2,),
            V_init=lambda shape: u.math.full(shape, -62.0) * u.mV,
        )

        with self.assertRaisesRegex(RuntimeError, "callable initializer"):
            _ = cell[0].V_init
        cell.init_state()

        np.testing.assert_allclose(cell[0].V_init.to_decimal(u.mV), -62.0)

    def test_cell_view_reads_dense_ion_population_rows(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        cell.init_state()

        ion = cell[[3, 1]].get_ion("na")

        self.assertIs(ion.root, cell.get_ion("na"))
        self.assertEqual(ion.length.shape, (2, cell.n_point))
        np.testing.assert_allclose(
            ion.length.to_decimal(u.um),
            cell.get_ion("na").length.to_decimal(u.um)[np.asarray([3, 1])],
        )

    def test_cell_view_reads_packed_runtime_rows_without_exposing_methods(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell[[1, 3]].place(at("soma", 0.5), exp)
        cell.init_state()
        layout = next(item for item in cell.layouts if item.kind == "synapse:ExpSyn")

        selected = cell[3].get_runtime_node(layout.id)
        empty = cell[0].get_runtime_node(layout.id)

        np.testing.assert_allclose(selected.tau.to_decimal(u.ms), [2.0])
        self.assertEqual(empty.tau.shape, (0,))
        with self.assertRaisesRegex(AttributeError, "read-only"):
            _ = selected.compute_derivative

    def test_cell_view_rejects_density_paint_and_lifecycle_ownership(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(2,))
        view = cell[0]
        channel = mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV)

        with self.assertRaisesRegex(NotImplementedError, r"root Cell\.paint"):
            view.paint(AllRegion(), channel)
        with self.assertRaisesRegex(RuntimeError, "root Cell"):
            view.init_state()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            view.reset_state()

    def test_unselected_place_retains_broadcast_semantics(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        cell.place(at("soma", 0.5), mech.SynapseSpec("ExpSyn", name="exp"))

        self.assertEqual(len(cell.point_placements), 1)
        self.assertIsNone(cell.point_placements[0].population_index)

    def test_cv_midpoints_are_resolved_locations_available_before_init(self):
        cell = _simple_cell()

        midpoints = cell.cv_midpoints

        self.assertEqual(len(midpoints), cell.n_cv)
        self.assertEqual(midpoints.branch_id.tolist(), [cv.branch_id for cv in cell.cvs])
        np.testing.assert_allclose(
            midpoints.branch_x,
            [(cv.prox + cv.dist) * 0.5 for cv in cell.cvs],
        )

    def test_locset_batch_place_aligns_rows_with_population_members(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        synapse = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms)
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0], [0, 0]])]

        cell.place(locations, synapse)

        placements = cell.point_placements
        self.assertEqual([item.population_index for item in placements], [0, 0, 1, 1, 2, 2])
        self.assertEqual([item.branch_x for item in placements], [0.5] * 6)

    def test_locset_batch_place_validates_population_alignment(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        synapse = mech.SynapseSpec("ExpSyn", name="exp")
        locations = cell.cv_midpoints[np.asarray([[0], [0]])]

        with self.assertRaisesRegex(ValueError, "batch rows"):
            cell.place(locations, synapse)
        with self.assertRaisesRegex(ValueError, "batch rows"):
            cell[[0, 2, 1]].place(locations, synapse)

    def test_synapse_view_expands_broadcast_and_selects_by_declaration_identity(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        other = mech.SynapseSpec("ExpSyn", name="other", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)
        cell[1].place(at("soma", 0.25), other)

        view = cell.synapses[exp]

        self.assertEqual(len(cell.synapses), 4)
        self.assertEqual(len(view), 3)
        self.assertEqual(view.population_index.tolist(), [0, 1, 2])
        self.assertEqual(view.placement_id.tolist(), [0, 0, 0])

    def test_synapse_view_sets_heterogeneous_runtime_parameters(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)

        cell.synapses[exp].set(
            tau=np.asarray([1.0, 2.0, 3.0]) * u.ms,
            e=np.asarray([-70.0, -60.0, -50.0]) * u.mV,
        )
        cell.init_state()

        synapse_layout = next(layout for layout in cell._runtime.layouts if layout.kind == "synapse:ExpSyn")
        tau = cell._runtime.state_buffers[(synapse_layout.id, "tau")]
        reversal = cell._runtime.state_buffers[(synapse_layout.id, "e")]
        np.testing.assert_allclose(tau.to_decimal(u.ms), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(reversal.to_decimal(u.mV), [-70.0, -60.0, -50.0])

    def test_synapse_view_sets_batch_placed_instances_in_row_order(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0], [0, 0]])]
        cell.place(locations, exp)

        view = cell.synapses[exp]
        view.set(tau=np.arange(1.0, 7.0) * u.ms)
        cell.init_state()

        synapse_layout = next(layout for layout in cell._runtime.layouts if layout.kind == "synapse:ExpSyn")
        tau = cell._runtime.state_buffers[(synapse_layout.id, "tau")]
        self.assertEqual(view.population_index.tolist(), [0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(tau.to_decimal(u.ms), np.arange(1.0, 7.0))

    def test_population_selection_validates_indices_and_phase(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        synapse = mech.SynapseSpec("ExpSyn", name="exp")
        self.assertEqual(cell[-1].population_indices, (3,))
        with self.assertRaises(IndexError):
            _ = cell[4]
        with self.assertRaises(TypeError):
            _ = cell[[0.5]]
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell[0].place(at("soma", 0.5), synapse)

    def test_junction_placements_share_point_but_keep_branch_ownership(self):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        dend = Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="basal_dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.attach(dend, name="dend", parent_x=1.0, child_x=0.0)
        cell = Cell(morpho, cv_policy=CVPerBranch())
        clamp = CurrentClamp(delay=1 * u.ms, durations=2 * u.ms, amplitudes=0.1 * u.nA)
        cell.place(at("soma", 1.0), clamp)
        cell.place(at("dend", 0.0), clamp)

        placements = cell.point_placements

        self.assertEqual([item.branch_name for item in placements], ["soma", "dend"])
        self.assertEqual([item.branch_x for item in placements], [1.0, 0.0])
        self.assertEqual([item.cv_id for item in placements], [0, 1])
        self.assertEqual(placements[0].point_id, placements[1].point_id)


class TestCellLifecycle(unittest.TestCase):
    def test_declaration_phase_flag(self):
        cell = _cell_with_probe()
        self.assertFalse(cell._initialized)

    def test_init_state_flips_flag_and_populates_runtime(self):
        cell = _cell_with_probe()
        cell.init_state()
        self.assertTrue(cell._initialized)
        self.assertIsNotNone(cell._runtime)
        self.assertGreater(len(cell.node_tree.nodes), 0)
        self.assertGreater(len(cell.runtime_nodes), 0)
        self.assertGreater(len(cell.runtime_cvs), 0)
        self.assertIsNone(cell._axial_jax)
        self.assertTrue(hasattr(cell, "V"))
        self.assertTrue(hasattr(cell, "spike"))

    def test_init_state_twice_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
            cell.init_state()

    def test_paint_after_init_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV))

    def test_place_after_init_raises(self):
        cell = _cell_with_probe()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.place(RootLocation(0.5), StateProbe(field="v", name="V_mid"))

    def test_config_setters_after_init_raise(self):
        cell = _cell_with_probe()
        cell.init_state()
        for name, value in (
            ("V_th", -60 * u.mV),
            ("V_init", -65 * u.mV),
            ("cv_policy", CVPerBranch()),
            ("solver", "staggered"),
        ):
            with self.subTest(attr=name):
                with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
                    setattr(cell, name, value)

    def test_reset_from_declaring_raises(self):
        cell = _cell_with_probe()
        with self.assertRaisesRegex(RuntimeError, r"reset\(\)"):
            cell.reset()

    def test_reset_round_trip(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.reset()
        self.assertFalse(cell._initialized)
        self.assertIsNone(cell._runtime)
        self.assertGreater(len(cell.node_tree.nodes), 0)
        self.assertIsNone(cell._axial_jax)
        self.assertFalse(hasattr(cell, "V"))
        self.assertFalse(hasattr(cell, "spike"))

        # Paint after reset works.
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV))
        cell.init_state()
        self.assertTrue(cell._initialized)

    def test_reset_restores_scalar_V_th(self):
        cell = _cell_with_probe()
        original_V_th = cell.V_th
        cell.init_state()
        # After init V_th has been vectorised by install_cell_runtime.
        self.assertNotEqual(cell.V_th.shape if hasattr(cell.V_th, "shape") else (), ())
        cell.reset()
        # Back to scalar declaration value.
        self.assertEqual(cell.V_th, original_V_th)

    def test_runtime_method_requires_init(self):
        cell = _cell_with_probe()
        for method_name in ("sample_probes", "mech_table"):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, method_name)()
        for property_name in ("runtime_cvs", "runtime_nodes"):
            with self.subTest(property=property_name):
                with self.assertRaisesRegex(RuntimeError, r"init_state\(\)"):
                    getattr(cell, property_name)

    def test_static_topology_is_available_before_init(self):
        cell = _cell_with_probe()
        self.assertGreater(len(cell.cvs), 0)
        self.assertGreater(len(cell.cv_tree.cvs), 0)
        self.assertGreater(len(cell.node_tree.nodes), 0)

    def test_runtime_views_bind_static_objects_after_init(self):
        cell = _cell_with_probe()
        cell.init_state()
        runtime_cv = cell.runtime_cvs[0]
        runtime_node = cell.runtime_nodes[0]
        self.assertIs(runtime_cv.declaration, cell.cvs[0])
        self.assertIs(runtime_node.declaration, cell.node_tree.nodes[0])
        self.assertIn("na", runtime_cv.ions)
        self.assertIn("na", runtime_node.ions)

    def test_nodes_query_api_is_restored_after_init(self):
        cell = _cell_with_probe()
        cell.init_state()
        nodes = cell.nodes(IonChannel, allowed_hierarchy=(1, 1))
        self.assertGreater(len(nodes), 0)

    def test_run_auto_inits_from_declaring(self):
        cell = _cell_with_probe()
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertTrue(cell._initialized)
        self.assertIn("V_root", result.traces)

    def test_run_does_not_reinit(self):
        cell = _cell_with_probe()
        cell.init_state()
        first_runtime = cell._runtime
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertIs(cell._runtime, first_runtime)

    def test_run_loop_cache_reuses_matching_shape(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 1)
        first_runner = next(iter(cell._run_loop_cache.values()))

        cell.reset_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)

        self.assertEqual(len(cell._run_loop_cache), 1)
        self.assertIs(next(iter(cell._run_loop_cache.values())), first_runner)

    def test_run_loop_cache_separates_step_count(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        cell.reset_state()
        cell.run(dt=0.1 * u.ms, duration=0.4 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 2)

    def test_reset_clears_run_loop_cache(self):
        cell = _cell_with_probe()
        cell.init_state()
        cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertEqual(len(cell._run_loop_cache), 1)

        cell.reset()

        self.assertEqual(cell._run_loop_cache, {})

    def test_axial_operator_cache_tracks_precision(self):
        cell = _simple_cell()
        with brainstate.environ.context(precision=32):
            cell.init_state()
            self.assertIsNone(cell.runtime.axial_operator_np)
            self.assertIsNone(cell.runtime.axial_operator_cache)
            self.assertIsNone(cell._axial_jax)

            operator32 = cell._get_axial_operator()
            cache32 = cell.runtime.axial_operator_cache
            self.assertEqual(operator32.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell._axial_jax.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell.runtime.axial_operator_np.dtype, np.float64)
            self.assertIsNotNone(cache32)

        with brainstate.environ.context(precision=64):
            operator64 = cell._get_axial_operator()
            cache64 = cell.runtime.axial_operator_cache
            self.assertEqual(operator64.dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)

    def test_staggered_run_does_not_build_dense_axial_operator(self):
        cell = _cell_with_probe()
        cell.init_state()

        cell.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertIsNone(cell.runtime.axial_operator_np)
        self.assertIsNone(cell.runtime.axial_operator_cache)
        self.assertIsNone(cell._axial_jax)
        self.assertIsNotNone(cell.runtime.dhs_static_source_np)

    def test_scalar_v_init_broadcasts_to_voltage_shape(self):
        cell = _simple_cell()
        cell.V_init = -60.0 * u.mV
        cell.init_state()
        self.assertEqual(cell.V.value.shape, cell.pop_size + (cell.n_cv,))
        self.assertTrue(
            u.math.allclose(
                cell.V.value,
                jnp.full(cell.pop_size + (cell.n_cv,), -60.0) * u.mV,
                atol=1e-9 * u.mV,
            )
        )

    def test_run_supports_scalar_v_init(self):
        cell = _cell_with_probe()
        cell.V_init = -60.0 * u.mV
        result = cell.run(dt=0.1 * u.ms, duration=0.5 * u.ms)
        self.assertIn("V_root", result.traces)

    def test_pop_size_extends_voltage_shape(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4, 4))
        cell.V_init = -60.0 * u.mV
        cell.init_state()
        self.assertEqual(cell.pop_size, (4, 4))
        self.assertEqual(cell.varshape, (4, 4, cell.n_cv))
        self.assertEqual(cell.V.value.shape, (4, 4, cell.n_cv))

    def test_pop_size_with_batch_size_keeps_batch_leading(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(2, 3))
        cell.V_init = -60.0 * u.mV
        cell.init_state(batch_size=5)
        self.assertEqual(cell.V.value.shape, (5, 2, 3, cell.n_cv))

    def test_bind_synapse_input_registers_source(self):
        cell = _simple_cell()
        source = lambda: jnp.asarray([1.0])

        returned = cell.bind_synapse_input("ampa", source, weight=2.0)

        self.assertIs(returned, cell)
        self.assertEqual(len(cell._synapse_input_bindings["ampa"]), 1)


class CellMembraneLinearizerTest(unittest.TestCase):
    @staticmethod
    def _leak_cell(mode: str) -> Cell:
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            V_init=-65.0 * u.mV,
            membrane_linearizer=mode,
        )
        cell.paint(
            AllRegion(),
            mech.Channel(
                "IL",
                name="leak",
                g_max=0.3 * (u.mS / u.cm**2),
                E=-54.3 * u.mV,
            ),
        )
        cell.init_state()
        return cell

    def test_point_matches_generic_values_and_higher_order_gradient(self):
        results = {}
        for mode in ("point", "generic"):
            cell = self._leak_cell(mode)
            voltage_unit = u.get_unit(cell.V.value)
            with brainstate.environ.context(t=0.0 * u.ms):
                linear, derivative = cell._voltage_linearizer()(cell.V.value)

                def objective(voltage_mantissa):
                    _, value = cell._voltage_linearizer()(u.Quantity(voltage_mantissa, voltage_unit))
                    return jnp.sum(u.get_mantissa(value))

                gradient = jax.grad(objective)(u.get_mantissa(cell.V.value))
            results[mode] = (linear, derivative, gradient)

        point = results["point"]
        generic = results["generic"]
        for point_value, generic_value in zip(point, generic):
            np.testing.assert_allclose(
                np.asarray(u.get_mantissa(point_value)),
                np.asarray(u.get_mantissa(generic_value)),
                rtol=1e-6,
                atol=1e-7,
            )


class CellIonChannelUpdateOrderTest(unittest.TestCase):
    def test_update_does_not_add_post_voltage_dispatch_for_custom_solver(self):
        calls = []

        def solver(target):
            calls.append("solver")

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
        )
        cell.init_state()

        def family(point_V):
            calls.append("family")

        def integration(point_V):
            calls.append("integration")

        cell._update_ion_channel_families = family
        cell._update_ion_channels_by_integration = integration
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver"])

    def test_staggered_solver_runs_family_dispatch_after_synapse_dynamics(self):
        calls = []

        def solver(target):
            calls.append("solver")
            point_V = target._cv_to_point(target.V.value)
            target._integrate_runtime_synapse_dynamics(point_V)
            target._update_ion_channel_families(point_V)

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
        )
        cell.init_state()
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._integrate_runtime_synapse_dynamics = lambda point_V: calls.append("synapse_dynamics")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver", "synapse_dynamics", "family"])

    def test_staggered_solver_can_use_integration_order(self):
        calls = []

        def solver(target):
            calls.append("solver")
            point_V = target._cv_to_point(target.V.value)
            target._update_ion_channels_by_integration(point_V)

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            solver=solver,
            ion_channel_update_order="integration",
        )
        cell.init_state()
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._prepare_runtime_synapse_inputs = lambda point_V: calls.append("prepare")

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell.update()

        self.assertEqual(calls, ["solver", "integration", "prepare"])

    def test_integration_phase_integrates_dependent_runtime_synapse(self):
        from unittest.mock import patch

        from braincell._base import Synapse as RuntimeSynapse

        calls = []

        class _RuntimeSynapse(RuntimeSynapse):
            def current(self, V_post):
                return 0.0

        cell = _simple_cell()
        cell.init_state()
        synapse = _RuntimeSynapse(size=1, name=None)

        cell.runtime_objects = lambda *args, **kwargs: {("layout_0",): synapse}
        cell._runtime_node_phase_args = lambda path, node, point_V: ("syn_arg",)
        cell._top_level_ion_channel_nodes = lambda: ((("layout_0",), synapse),)
        synapse.ind_update = lambda *args, **kwargs: calls.append(("ind", args))

        def _step(target, *args):
            calls.append(("dependent", target, args))

        with patch("braincell._multi_compartment.cell.ind_exp_euler_step", _step):
            point_V = cell._cv_to_point(cell.V.value)
            cell._update_ion_channels_by_integration(point_V)

        self.assertEqual(calls[0][0], "dependent")
        self.assertIs(calls[0][1], synapse)
        self.assertEqual(calls[0][2], ("syn_arg",))
        self.assertEqual(calls[1][0], "ind")
        self.assertEqual(len(calls[1][1]), 1)

    def test_family_phase_runs_ion_hooks_before_child_channel_hooks(self):
        calls = []

        class _Ion(Ion):
            def __init__(self):
                super().__init__(size=1, name=None)
                self.Ci = DiffEqSingleState(jnp.asarray([1.0]))
                self.Co = jnp.asarray([2.0])
                self.valence = 1

            @property
            def E(self):
                return jnp.asarray([0.0])

            def _ion_compute_derivative_hook(self, V):
                calls.append("ion")
                self.Ci.derivative = jnp.asarray([0.0]) / u.ms

        class _Channel(Channel):
            root_type = _Ion

            def __init__(self):
                super().__init__(size=1, name=None)
                self.x = DiffEqSingleState(jnp.asarray([1.0]))

            def compute_derivative(self, V, ion):
                calls.append("channel")
                self.x.derivative = jnp.asarray([0.0]) / u.ms

            def current(self, V, ion):
                return 0.0

        cell = _simple_cell()
        ion = _Ion()
        ion.add(ch=_Channel())
        cell.init_state()
        cell.ion_channels["test_ion"] = ion

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell._update_ion_channel_families(cell._cv_to_point(cell.V.value))

        self.assertEqual(calls, ["ion", "channel"])

    def test_family_phase_does_not_pass_recursive_child_to_channels(self):
        channel_args = []

        class _Ion(Ion):
            def __init__(self):
                super().__init__(size=1, name=None)
                self.Ci = DiffEqSingleState(jnp.asarray([1.0]))
                self.Co = jnp.asarray([2.0])
                self.valence = 1

            @property
            def E(self):
                return jnp.asarray([0.0])

            def _ion_compute_derivative_hook(self, V):
                self.Ci.derivative = jnp.asarray([0.0]) / u.ms

        class _Channel(Channel):
            root_type = _Ion

            def __init__(self):
                super().__init__(size=1, name=None)
                self.x = DiffEqSingleState(jnp.asarray([1.0]))

            def pre_integral(self, V, *ions):
                channel_args.append(("pre", ions))

            def compute_derivative(self, V, *ions):
                channel_args.append(("compute", ions))
                self.x.derivative = jnp.asarray([0.0]) / u.ms

            def post_integral(self, V, *ions):
                channel_args.append(("post", ions))

            def current(self, V, ion):
                return 0.0

        cell = _simple_cell()
        ion = _Ion()
        ion.add(ch=_Channel())
        cell.init_state()
        cell.ion_channels["test_ion"] = ion

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell._update_ion_channel_families(cell._cv_to_point(cell.V.value))

        self.assertGreaterEqual(len(channel_args), 3)
        for _, ions in channel_args:
            self.assertEqual(len(ions), 1)
            self.assertFalse(any(isinstance(arg, bool) for arg in ions))


class CellDoesNotAllocatePlaceholderIonsEagerlyTest(unittest.TestCase):
    """MED-09: Cell.__init__ must not allocate a throwaway ion container."""

    def test_build_placeholder_ions_not_called_in_init(self) -> None:
        from unittest.mock import patch

        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        tree = Morphology.from_root(soma, name="soma")

        with patch(
            "braincell._compute.ions.build_placeholder_ions",
            side_effect=AssertionError("placeholder must not be called at __init__"),
        ):
            _ = Cell(tree)


class CellPopulationAxisIsMandatoryTest(unittest.TestCase):
    """``Cell`` always carries a population axis, so state is rank >= 2."""

    def test_default_pop_size_is_one(self) -> None:
        self.assertEqual(_simple_cell().pop_size, (1,))

    def test_none_pop_size_means_unspecified_and_becomes_one(self) -> None:
        cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=None)
        self.assertEqual(cell.pop_size, (1,))

    def test_scalar_and_tuple_pop_size_agree(self) -> None:
        self.assertEqual(Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=4).pop_size, (4,))
        self.assertEqual(Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=(2, 3)).pop_size, (2, 3))

    def test_empty_pop_size_is_rejected(self) -> None:
        for empty in ((), []):
            with self.subTest(pop_size=empty):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=empty)

    def test_voltage_carries_the_population_axis(self) -> None:
        for pop_size, expected in ((1, (1,)), (4, (4,)), ((2, 3), (2, 3))):
            with self.subTest(pop_size=pop_size):
                cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=pop_size)
                cell.init_state()
                self.assertEqual(tuple(cell.V.value.shape), expected + (cell.n_cv,))


def _hh_cell(pop_size, *, calcium: bool = True) -> Cell:
    """A Cell exercising gate channels and a kinetic ion.

    ``calcium=False`` drops the bare ``CalciumDetailed`` ion, which has no
    current source and therefore cannot be stepped; it is only there so the
    kinetic-species allocation path is covered.
    """
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(dend, name="dend", parent_x=0.5)

    cell = Cell(tree, pop_size=pop_size, cv_policy=CVPerBranch(cv_per_branch=2))
    cell.paint(AllRegion(), mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)))
    cell.paint(AllRegion(), mech.Channel("K_HH1952", g_max=3.6 * (u.mS / u.cm**2)))
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.3 * (u.mS / u.cm**2), E=-54.0 * u.mV))
    if calcium:
        cell.paint(AllRegion(), mech.Ion("CalciumDetailed"))
    return cell


class CellHiddenStatesAreGroupStatesTest(unittest.TestCase):
    """Every ``Cell`` hidden state groups its trailing spatial axis.

    See ``docs/specs/2026-08-13-cell-hidden-group-state.md``.
    """

    def _hidden_states(self, cell: Cell):
        return [
            (".".join(map(str, path)), state)
            for path, state in brainstate.graph.states(cell).items()
            if isinstance(state, brainstate.HiddenState)
        ]

    @staticmethod
    def _expected_num_state(cell: Cell, name: str) -> int:
        """The group-axis length the named state must have.

        Each hidden state lives in exactly one space, so this pins the
        expectation rather than accepting any of several lengths:

        - ``V`` is CV space, so ``n_cv``.
        - A painted density channel's gates are point space, so ``n_point``.
        - A placed point mechanism (here, one synapse) spans only the sites
          it was placed on, which is its layout's ``n_active``.
        """
        if name == "V":
            return cell.n_cv
        head, _, tail = name.partition("ion_channels.")
        assert head == "", f"unexpected state path {name!r}"
        first = tail.split(".")[0]
        if first.startswith("layout_"):
            layout_id = int(first.removeprefix("layout_"))
            (layout,) = [item for item in cell.layouts if item.id == layout_id]
            return int(layout.n_active)
        return cell.n_point

    def _assert_all_grouped(self, cell: Cell) -> None:
        hidden = self._hidden_states(cell)
        self.assertGreater(len(hidden), 1, "expected channel/ion/synapse states beyond V")
        seen_spaces = set()
        for name, state in hidden:
            with self.subTest(state=name):
                self.assertIsInstance(state, brainstate.HiddenGroupState)
                # Everything but the trailing axis is population; the trailing
                # axis is the group axis this state is indexed by.
                self.assertEqual(state.varshape, cell.pop_size)
                self.assertEqual(state.num_state, self._expected_num_state(cell, name))
                seen_spaces.add(state.num_state)
        # Guard the guard: if every state happened to share one length, the
        # per-state expectation above would be much weaker than it looks.
        self.assertIn(cell.n_cv, seen_spaces)
        self.assertIn(cell.n_point, seen_spaces)

    def test_every_hidden_state_is_grouped(self) -> None:
        for pop_size in (1, 4):
            with self.subTest(pop_size=pop_size):
                cell = _hh_cell(pop_size)
                cell.init_state()
                self._assert_all_grouped(cell)

    def test_still_grouped_after_reset_state(self) -> None:
        cell = _hh_cell(4)
        cell.init_state()
        cell.reset_state()
        self._assert_all_grouped(cell)

    def test_voltage_is_a_diffeq_group_state(self) -> None:
        from braincell.quad import DiffEqGroupState

        cell = _hh_cell(1)
        cell.init_state()
        self.assertIsInstance(cell.V, DiffEqGroupState)
        # Solvers still select it, because DiffEqGroupState is a DiffEqState.
        self.assertIsInstance(cell.V, DiffEqState)
        self.assertEqual(cell.V.num_state, cell.n_cv)


class RuntimeInspectionCollapsesPopulationAxisTest(unittest.TestCase):
    """Node- and CV-local inspection answer for one morphology.

    Regression guard: the CV path collapsed the population axis but the
    point path did not, so ``runtime_nodes[i].ions[...]`` raised on a
    *default* ``Cell`` once the axis became mandatory. Reading a field —
    not merely checking the key is present — is what catches this.
    """

    @staticmethod
    def _cell(pop_size=1) -> Cell:
        cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=pop_size)
        cell.paint(AllRegion(), mech.Channel("Na_HH1952", g_max=12.0 * (u.mS / u.cm**2)))
        cell.init_state()
        return cell

    def test_node_and_cv_views_read_ion_fields_on_a_default_cell(self) -> None:
        cell = self._cell()
        self.assertEqual(cell.pop_size, (1,))
        for view in (cell.runtime_nodes[0], cell.runtime_cvs[0]):
            with self.subTest(view=type(view).__name__):
                length = view.ions["na"].length
                self.assertEqual(u.get_unit(length).dim, u.um.dim)
                self.assertEqual(np.ndim(u.get_mantissa(length)), 0)

    def test_multi_member_population_is_refused_with_a_useful_message(self) -> None:
        cell = self._cell(pop_size=4)
        for view in (cell.runtime_nodes[0], cell.runtime_cvs[0]):
            with self.subTest(view=type(view).__name__):
                with self.assertRaisesRegex(ValueError, r"population shape \(4,\).*pop_size=\(4,\)"):
                    _ = view.ions["na"].length


class CellPopulationWideningIsNumericallyNeutralTest(unittest.TestCase):
    """Widening the population axis must not change the simulated trace."""

    @staticmethod
    def _trace(pop_size, n_steps: int = 20, *, on_init=None):
        cell = _hh_cell(pop_size, calcium=False)
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(delay=0.1 * u.ms, durations=1.0 * u.ms, amplitudes=0.5 * u.nA),
        )
        cell.init_state()
        if on_init is not None:
            on_init(cell)

        dt = 0.01 * u.ms

        def step(i):
            with brainstate.environ.context(t=i * dt, dt=dt):
                cell.update()
            return cell.V.value

        with brainstate.environ.context(dt=dt):
            return brainstate.transform.for_loop(step, jnp.arange(n_steps))

    def test_grouped_state_classes_do_not_change_the_numbers(self) -> None:
        """The state-class change must be type-only.

        Runs the same model twice — once with the grouped classes a ``Cell``
        normally allocates, once with every hidden state forced back to the
        plain pre-change classes — and requires bit-identical traces. This
        is what makes ``DiffEqGroupState`` safe to adopt: it changes which
        type wraps the array, never the array.

        Each arm also asserts which classes it actually built. Without that
        the test would still pass if the monkeypatch stopped applying — it
        would then be comparing grouped against grouped, which proves
        nothing.
        """
        import contextlib

        @contextlib.contextmanager
        def _ungrouped(_enabled=True):
            from braincell.quad.protocol import state_grouping as real_scope

            with real_scope(False) as value:
                yield value

        def trace(*, grouped: bool):
            classes: list[type] = []

            def record(cell: Cell) -> None:
                classes.extend(
                    type(state)
                    for state in brainstate.graph.states(cell).values()
                    if isinstance(state, brainstate.HiddenState)
                )

            if grouped:
                values = self._trace(1, on_init=record)
            else:
                saved = (cell_module.DiffEqGroupState, cell_module.state_grouping)
                cell_module.DiffEqGroupState = DiffEqSingleState
                cell_module.state_grouping = _ungrouped
                try:
                    values = self._trace(1, on_init=record)
                finally:
                    cell_module.DiffEqGroupState, cell_module.state_grouping = saved
            return np.asarray(values.to_decimal(u.mV), dtype=float), classes

        plain, plain_classes = trace(grouped=False)
        grouped, grouped_classes = trace(grouped=True)

        # The two arms must have built the same *number* of hidden states
        # out of genuinely different classes, or the comparison below is
        # vacuous.
        self.assertGreater(len(grouped_classes), 1)
        self.assertEqual(len(plain_classes), len(grouped_classes))
        for cls in grouped_classes:
            self.assertTrue(issubclass(cls, brainstate.HiddenGroupState), cls)
        for cls in plain_classes:
            self.assertFalse(issubclass(cls, brainstate.HiddenGroupState), cls)

        self.assertTrue(np.all(np.isfinite(grouped)))
        np.testing.assert_array_equal(grouped, plain)

    def test_multi_axis_population_steps_and_jits(self) -> None:
        # ``pop_size`` may have more than one axis; only the trailing
        # compartment axis is the group axis.
        cell = _hh_cell((2, 3), calcium=False)
        cell.init_state()
        self.assertEqual(cell.V.varshape, (2, 3))
        self.assertEqual(cell.V.num_state, cell.n_cv)

        dt = 0.01 * u.ms

        @brainstate.transform.jit
        def step():
            with brainstate.environ.context(t=0.0 * u.ms, dt=dt):
                cell.update()
            return cell.V.value

        self.assertEqual(tuple(step().shape), (2, 3, cell.n_cv))

    def test_three_members_reproduce_the_single_member_trace(self) -> None:
        single = np.asarray(self._trace(1).to_decimal(u.mV), dtype=float)
        population = np.asarray(self._trace(3).to_decimal(u.mV), dtype=float)

        self.assertEqual(single.shape[1], 1)
        self.assertEqual(population.shape, single.shape[:1] + (3,) + single.shape[2:])
        self.assertTrue(np.all(np.isfinite(single)))
        for member in range(3):
            np.testing.assert_allclose(population[:, member], single[:, 0], rtol=1e-5, atol=1e-5)


class CoercersAgreeAcrossStorageFlavoursTest(unittest.TestCase):
    """United and unitless fields must coerce to the same numbers.

    Each coercer used to spell its length rules out twice, once for a
    ``Quantity`` and once for a bare array. They are now stated once and
    the unit is split off by ``_split_unit``, so this pins the property
    that made the duplication removable in the first place.
    """

    COERCERS = (
        "_coerce_vis_node_values_object",
        "_coerce_runtime_point_values_object",
        "_coerce_vis_cv_values_object",
        "_coerce_named_vis_node_values_object",
        "_coerce_named_vis_cv_values_object",
    )

    def setUp(self) -> None:
        self.cell = Cell(_soma_tree(), cv_policy=CVPerBranch(2))
        self.cell.init_state()

    def _cases(self):
        # Scalar, point-length, and CV-length: the three shapes every
        # coercer branches on.
        return {
            "scalar": 1.5,
            "point": np.linspace(-70.0, -60.0, self.cell.n_point),
            "cv": np.linspace(-70.0, -60.0, self.cell.n_cv),
        }

    def test_unit_carrying_and_bare_inputs_give_the_same_numbers(self) -> None:
        for name in self.COERCERS:
            coerce = getattr(self.cell, name)
            for label, plain in self._cases().items():
                with self.subTest(coercer=name, case=label):
                    bare = coerce(plain)
                    united = coerce(plain * u.mV)
                    self.assertTrue(u.get_unit(united).has_same_dim(u.mV))
                    np.testing.assert_allclose(
                        np.asarray(u.get_mantissa(bare), dtype=float),
                        np.asarray(united.to_decimal(u.mV), dtype=float),
                    )

    def test_a_length_that_is_neither_space_is_rejected_either_way(self) -> None:
        bad = np.zeros(self.cell.n_point + self.cell.n_cv + 1)
        for name in self.COERCERS:
            coerce = getattr(self.cell, name)
            with self.subTest(coercer=name, storage="bare"):
                with self.assertRaises(ValueError):
                    coerce(bad)
            with self.subTest(coercer=name, storage="united"):
                with self.assertRaises(ValueError):
                    coerce(bad * u.mV)


class MultiCompartmentAliasTest(unittest.TestCase):
    """``MultiCompartment`` is a second name for ``Cell``, not a subclass."""

    def test_alias_is_the_same_object_everywhere_it_is_exported(self) -> None:
        # Identity, not equality: a subclass or wrapper would break
        # ``isinstance`` checks written against the other name.
        self.assertIs(braincell.MultiCompartment, braincell.Cell)
        self.assertIs(cell_module.MultiCompartment, cell_module.Cell)
        self.assertIn("MultiCompartment", braincell.__all__)
        self.assertIn("MultiCompartment", cell_module.__all__)

    def test_instances_built_through_the_alias_are_cells(self) -> None:
        cell = braincell.MultiCompartment(_soma_tree())
        self.assertIsInstance(cell, braincell.Cell)
        self.assertEqual(cell.pop_size, (1,))


# =============================================================================
# Visualization entry points (Cell.vis_topology / vis_cv / vis_branch / vis_node)
# =============================================================================


def _soma_dend_tree() -> Morphology:
    """Soma plus one tapering basal dendrite — the tree the vis tests draw."""
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


class CellVisDiscreteTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_vis_cv_smoke_returns_axes_before_init(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        ax = cell.vis_cv(show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_vis_branch_smoke_returns_axes_before_init(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        ax = cell.vis_branch(show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)

    def test_vis_cv_fraction_passes_partial_coverage(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(region=BranchSlice(branch_index=1, prox=0.0, dist=0.5), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertAlmostEqual(coverage[1], 0.5)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_cv_any_mode_is_forwarded(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="any",
                show=False,
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "any")

    def test_vis_cv_locset_is_supported(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(locset=RootLocation(0.5), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertEqual(coverage[0], 1.0)

    def test_vis_cv_value_V_uses_cv_values(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value="V", show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_cv,))

    def test_vis_cv_raw_point_values_are_gathered_to_cv(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        point_values = np.asarray([np.nan, 1.0, np.nan, 2.0, np.nan], dtype=float)
        expected = cell._point_to_cv(point_values)

        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value=point_values, show=False)

        np.testing.assert_allclose(mocked.call_args.kwargs["values"], expected)

    def test_vis_cv_layout_id_lookup_works(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_cv(value=("layout_id", layout.id, "g_max"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_cv,))

    def test_vis_branch_fraction_passes_partial_branch_coverage(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_branch(region=BranchSlice(branch_index=1, prox=0.25, dist=0.75), show=False)

        coverage = mocked.call_args.kwargs["highlight_fractions"]
        self.assertAlmostEqual(coverage[1], 0.5)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_branch_all_mode_is_forwarded(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch("braincell.vis.point_topology._plot_discrete_topology_graph", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_branch(
                region=BranchSlice(branch_index=1, prox=0.0, dist=1.0),
                coverage_mode="all",
                show=False,
            )
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "all")

    def test_vis_topology_cv_dispatches(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch.object(cell, "vis_cv", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="cv", show=False)
        mocked.assert_called_once()

    def test_vis_topology_branch_dispatches(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with mock.patch.object(cell, "vis_branch", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="branch", show=False)
        mocked.assert_called_once()

    def test_vis_topology_branch_rejects_locset(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with self.assertRaisesRegex(ValueError, "does not support locset"):
            cell.vis_topology(level="branch", locset=RootLocation(0.5), show=False)

    def test_vis_topology_branch_rejects_value(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        with self.assertRaisesRegex(ValueError, "does not support value"):
            cell.vis_topology(level="branch", value="V", show=False)


if __name__ == "__main__":
    unittest.main()


class CellVisNodeTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_vis_node_region_maps_to_midpoint_point_ids(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        target_point_id = int(cell.node_tree.cv_to_mid_node_id[1])

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), show=False)

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_vis_node_locset_maps_to_midpoint_point_ids(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        target_point_id = int(cell.node_tree.cv_to_mid_node_id[0])

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(locset=RootLocation(0.5), show=False)

        self.assertEqual(mocked.call_args.kwargs["highlight_fractions"][target_point_id], 1.0)

    def test_vis_node_rejects_value_with_region(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            cell.vis_node(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), value="V", show=False)

    def test_vis_node_converts_raw_cv_values_to_point_values(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        cv_values = np.asarray([1.0, 2.0], dtype=float)
        expected = cell._discretization_to_node_values(cv_values)

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=cv_values, show=False)

        np.testing.assert_allclose(
            np.nan_to_num(mocked.call_args.kwargs["values"], nan=-1.0),
            np.nan_to_num(expected, nan=-1.0),
        )

    def test_vis_node_value_V_uses_point_voltage(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        expected = cell._discretization_to_node_values(cell.V.value)

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value="V", show=False)

        np.testing.assert_allclose(
            mocked.call_args.kwargs["values"].to_decimal(u.mV),
            expected.to_decimal(u.mV),
        )
        self.assertEqual(mocked.call_args.kwargs["value_label"], "V")

    def test_vis_node_channel_lookup_ambiguous_raises(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.paint(full, mech.Channel("IL", name="leak_b", g_max=5.0 * (u.mS / u.cm**2), E=-67.0 * u.mV))
        cell.init_state()

        with self.assertRaisesRegex(ValueError, "multiple runtime layouts"):
            cell.vis_node(value=("channel", "IL", "g_max"), show=False)

    def test_vis_node_layout_id_lookup_works(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        layout = cell.layouts[0]

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=("layout_id", layout.id, "g_max"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_point,))

    def test_vis_node_ion_lookup_works(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(value=("ion", "na", "E"), show=False)

        values = mocked.call_args.kwargs["values"]
        self.assertEqual(values.shape, (cell.n_point,))
        self.assertEqual(mocked.call_args.kwargs["value_label"], "na.E")

    def test_vis_node_region_coverage_is_forwarded(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()

        with mock.patch("braincell.vis.point_topology.plot_point_topology", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_node(
                region=BranchSlice(branch_index=1, prox=0.0, dist=0.5),
                coverage_mode="fraction",
                show=False,
            )

        self.assertIn("highlight_fractions", mocked.call_args.kwargs)
        self.assertEqual(mocked.call_args.kwargs["coverage_mode"], "fraction")

    def test_vis_topology_node_dispatches(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        with mock.patch.object(cell, "vis_node", autospec=True) as mocked:
            mocked.return_value = plt.subplots()[1]
            cell.vis_topology(level="node", show=False)
        mocked.assert_called_once()

    def test_vis_node_smoke_returns_axes(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        ax = cell.vis_node(value="V", show=False)
        self.assertIsInstance(ax, matplotlib.axes.Axes)


class CellVisPopulationAxisTest(unittest.TestCase):
    """Visualization draws one morphology, so it needs one population member."""

    def tearDown(self) -> None:
        plt.close("all")

    def test_default_single_member_population_still_draws(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch())
        cell.init_state()
        self.assertEqual(cell.pop_size, (1,))
        self.assertIsInstance(cell.vis_cv(value="V", show=False), matplotlib.axes.Axes)
        self.assertIsInstance(cell.vis_node(value="V", show=False), matplotlib.axes.Axes)

    def test_multi_member_population_is_refused_with_a_useful_message(self) -> None:
        cell = Cell(_soma_dend_tree(), cv_policy=CVPerBranch(), pop_size=4)
        cell.init_state()
        for method in ("vis_cv", "vis_node"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, r"population shape \(4,\).*pop_size=\(4,\)"):
                    getattr(cell, method)(value="V", show=False)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
