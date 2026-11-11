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

import dataclasses
import re
import sys
import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
import braincell.mech as mech
from braincell import (
    Branch,
    CompositeByTypePolicy,
    CVPerBranch,
    CVPerBranchList,
    CVPolicyByTypeRule,
    Cell,
    CellView,
    Channel,
    CurrentClamp,
    Ion,
    IonChannel,
    MaxCVLen,
    Morphology,
    NetStim,
    connect,
)
from braincell._multi_compartment import cell as cell_module
from braincell.filter import AllRegion, BranchSlice, LocsetBatch, LocsetMask, RootLocation, Terminals, at
from braincell.mech import StateProbe
from braincell.morph._testing import (
    EDIT_CASES,
    SABOTAGE_CASES,
    EditCase,
    assert_layer_fresh,
    cold_rebuild,
    make_tree_s,
    make_tree_t,
    make_tree_u,
    stretched,
)
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
        synapse = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms)

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
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms)
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
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)
        cell[1].place(at("soma", 0.25), exp)

        cell.init_state()
        view = cell[[1, 3]].synapses["exp"]
        view.set(tau=np.asarray([1.0, 2.0, 3.0]) * u.ms)

        self.assertEqual(view.population_index.tolist(), [1, 1, 3])
        cell.reset_state()
        layouts = [item for item in cell.layouts if item.kind == "synapse:ExpSyn"]
        self.assertEqual(len(layouts), 1)
        tau = cell.get_state(layouts[0].id, "tau")
        self.assertEqual(tau.shape, (5,))
        np.testing.assert_allclose(tau.to_decimal(u.ms), [2.0, 1.0, 2.0, 2.0, 3.0])

    def test_cell_view_filters_connection_rows_by_synapse_population(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.Synapse("ExpSyn", name="exp")
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

        cell.init_state()
        result = cell[[1, 3]].set(
            V_init=np.asarray([-60.0, -55.0]) * u.mV,
            V_th=np.asarray([-10.0, 0.0]) * u.mV,
        )

        self.assertIs(result.root, cell)
        np.testing.assert_allclose(result.V_init.to_decimal(u.mV)[:, 0], [-60.0, -55.0])
        np.testing.assert_allclose(result.V_th.to_decimal(u.mV)[:, 0], [-10.0, 0.0])
        cell.reset_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -60.0, -65.0, -55.0])
        np.testing.assert_allclose(cell.V_th.to_decimal(u.mV)[:, 0], [-20.0, -10.0, -20.0, 0.0])
        np.testing.assert_allclose(cell[[3, 0]].V.to_decimal(u.mV)[:, 0], [-55.0, -65.0])
        self.assertEqual(cell[[3, 0]].spike.shape, (2, cell.n_cv))

    def test_root_voltage_assignment_clears_selected_overrides(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,), V_init=-65.0 * u.mV)
        cell.init_state()
        cell[1].V_init = -55.0 * u.mV
        cell.reset()
        cell.V_init = -70.0 * u.mV

        cell.init_state()

        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), -70.0)

    def test_cell_view_voltage_overrides_validate_and_clear_on_deinit(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(3,), V_init=-65.0 * u.mV)
        cell.init_state()
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
        cell.reset_state()
        cell.reset_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -55.0, -65.0])
        cell.reset()
        cell.init_state()
        np.testing.assert_allclose(cell.V.value.to_decimal(u.mV)[:, 0], [-65.0, -65.0, -65.0])

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

        cell.init_state()
        cell[[0, 2]].V_init = values
        cell.reset_state()

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
        self.assertEqual(ion.length.shape, (2, cell.n_cv))
        np.testing.assert_allclose(
            ion.length.to_decimal(u.um),
            cell.get_ion("na").length.to_decimal(u.um)[np.asarray([3, 1])],
        )

    def test_cell_view_reads_packed_runtime_rows_without_exposing_methods(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell[[1, 3]].place(at("soma", 0.5), exp)
        cell.init_state()
        layout = next(item for item in cell.layouts if item.kind == "synapse:ExpSyn")

        selected = cell[3].get_runtime_node(layout.id)
        empty = cell[0].get_runtime_node(layout.id)

        np.testing.assert_allclose(selected.tau.to_decimal(u.ms), [2.0])
        self.assertEqual(empty.tau.shape, (0,))
        with self.assertRaisesRegex(AttributeError, "read-only"):
            _ = selected.compute_derivative

    def test_population_view_rejects_paint_and_lifecycle_ownership(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(2,))
        view = cell[0]
        channel = mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70 * u.mV)

        with self.assertRaisesRegex(ValueError, "every population member alike"):
            view.paint(channel)
        with self.assertRaisesRegex(ValueError, "every population member alike"):
            view.paint(AllRegion(), channel)
        self.assertEqual(len(cell.paint_rules), 1)
        with self.assertRaisesRegex(RuntimeError, "root Cell"):
            view.init_state()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            cell[0].reset_state()

    def test_unselected_place_retains_broadcast_semantics(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="exp"))

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
        synapse = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms)
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0], [0, 0]])]

        cell.place(locations, synapse)

        placements = cell.point_placements
        self.assertEqual([item.population_index for item in placements], [0, 0, 1, 1, 2, 2])
        self.assertEqual([item.branch_x for item in placements], [0.5] * 6)

    def test_locset_batch_place_validates_population_alignment(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        synapse = mech.Synapse("ExpSyn", name="exp")
        locations = cell.cv_midpoints[np.asarray([[0], [0]])]

        with self.assertRaisesRegex(ValueError, "batch rows"):
            cell.place(locations, synapse)
        with self.assertRaisesRegex(ValueError, "batch rows"):
            cell[[0, 2, 1]].place(locations, synapse)

    def test_synapse_view_expands_broadcast_and_selects_by_declaration_identity(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        other = mech.Synapse("ExpSyn", name="other", tau=2.0 * u.ms, e=0.0 * u.mV)
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
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)

        cell.init_state()
        cell.synapses[exp].set(
            tau=np.asarray([1.0, 2.0, 3.0]) * u.ms,
            e=np.asarray([-70.0, -60.0, -50.0]) * u.mV,
        )
        cell.reset_state()

        synapse_layout = next(layout for layout in cell._runtime.layouts if layout.kind == "synapse:ExpSyn")
        tau = cell._runtime.state_buffers[(synapse_layout.id, "tau")]
        reversal = cell._runtime.state_buffers[(synapse_layout.id, "e")]
        np.testing.assert_allclose(tau.to_decimal(u.ms), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(reversal.to_decimal(u.mV), [-70.0, -60.0, -50.0])

    def test_synapse_view_sets_batch_placed_instances_in_row_order(self):
        base = _simple_cell()
        cell = Cell(base.morpho, cv_policy=CVPerBranch(), pop_size=(3,))
        exp = mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0], [0, 0]])]
        cell.place(locations, exp)

        cell.init_state()
        view = cell.synapses[exp]
        view.set(tau=np.arange(1.0, 7.0) * u.ms)
        cell.reset_state()

        synapse_layout = next(layout for layout in cell._runtime.layouts if layout.kind == "synapse:ExpSyn")
        tau = cell._runtime.state_buffers[(synapse_layout.id, "tau")]
        self.assertEqual(view.population_index.tolist(), [0, 0, 1, 1, 2, 2])
        np.testing.assert_allclose(tau.to_decimal(u.ms), np.arange(1.0, 7.0))

    def test_population_selection_validates_indices_and_phase(self):
        cell = Cell(_simple_cell().morpho, cv_policy=CVPerBranch(), pop_size=(4,))
        synapse = mech.Synapse("ExpSyn", name="exp")
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


def _edit_probe_cell(morpho: Morphology) -> Cell:
    """The Cell the edit table declares on a tree: 40 um CVs and an ExpSyn on every terminal."""
    cell = Cell(morpho, cv_policy=MaxCVLen(40.0 * u.um))
    cell.place(Terminals(), mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV))
    return cell


def _declaration_fingerprint(cell: Cell) -> dict:
    return {
        "cvs": tuple((cv.branch_id, cv.prox, cv.dist, float(cv.length.to_decimal(u.um))) for cv in cell.cvs),
        "point mechanisms": tuple((p.branch_id, p.branch_x, p.cv_id) for p in cell.point_placements),
    }


def assert_cell_declaration_fresh(testcase: unittest.TestCase, case: EditCase) -> None:
    """Cell layer of the morphology edit table, before ``init_state``.

    A Cell declared and discretized on a tree, then read again after
    ``case.edit``, must report the CVs (branch, prox, dist, length) and
    point-mechanism placements (branch, x, CV) of a Cell declared from
    scratch on the cold rebuild of the edited tree.
    """
    assert_layer_fresh(
        testcase, case, layer="Cell declaration", warm=_edit_probe_cell, observe=_declaration_fingerprint
    )


class CellDeclarationEditTableTest(unittest.TestCase):
    """A declared Cell never serves the discretization of its tree from before an edit."""

    def test_every_edit_case_matches_a_cell_on_the_cold_rebuild(self):
        for case in EDIT_CASES:
            with self.subTest(case.name):
                assert_cell_declaration_fresh(self, case)

    def test_every_sabotage_case_is_reported(self):
        for case in SABOTAGE_CASES:
            with self.subTest(case.name):
                with self.assertRaisesRegex(AssertionError, r"Cell declaration: .*\n  cvs"):
                    assert_cell_declaration_fresh(self, case)

    def test_cv_lengths_follow_an_in_place_replacement(self):
        morpho = make_tree_s()
        cell = Cell(morpho)

        def total_cv_length() -> float:
            return sum(float(cv.length.to_decimal(u.um)) for cv in cell.cvs)

        np.testing.assert_allclose(total_cv_length(), 150.0, rtol=1e-12)
        morpho.replace("dend", stretched(morpho.dend.branch))
        np.testing.assert_allclose(total_cv_length(), 250.0, rtol=1e-12)


def _index_probe_synapse():
    return mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV)


def _extra_branch() -> Branch:
    return Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")


def _placed_branches(cell: Cell) -> list:
    return [(p.branch_id, cell.morpho.branch(index=p.branch_id).name) for p in cell.point_placements]


_MASK_PLACE_STALE = r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ .*LocsetMask.*revision \d+.*at\(\"axon\", 0\.5\)"
_BATCH_PLACE_STALE = r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ .*LocsetBatch"
_CV_LIST_STALE = r"cv_policy CVPerBranchList\(cv_per_branch=\(1, 2, 1, 1\)\) .*default branch index"


class IndexHeldDeclarationTest(unittest.TestCase):
    """A declaration that holds branch indices raises once a removal has shifted them.

    Tree U: soma 0, dend 1, axon 2, apical 3. Removing ``dend`` moves axon to
    1 and apical to 2, so an index 2 held from before now names apical.
    """

    def test_a_mask_placement_raises_at_the_first_grid_read_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.n_cv
        # The stale grid is never served instead: every later read raises too.
        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.point_placements

    def test_a_mask_placement_raises_at_init_state_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_an_expression_placement_follows_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5), _index_probe_synapse())

        morpho.remove("dend")

        self.assertEqual(_placed_branches(cell), [(1, "axon")])

    def test_an_expression_on_a_removed_branch_names_the_placement(self):
        # The expression follows the name; once no branch has it, the next
        # rebuild says which declaration lost its branch instead of a bare
        # KeyError: 'dend'.
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("dend", 0.5), _index_probe_synapse())

        morpho.remove("dend")

        pattern = r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ locates its mechanisms on branch 'dend'.*does not have"
        with self.assertRaisesRegex(KeyError, pattern):
            cell.n_cv
        with self.assertRaisesRegex(KeyError, pattern):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_an_expression_follows_a_new_branch_of_the_removed_name(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("dend", 0.5), _index_probe_synapse())

        morpho.remove("dend")
        morpho.axon.dend = _extra_branch()

        self.assertEqual(_placed_branches(cell), [(3, "dend")])
        self.assertEqual(cell.morpho.dend.parent.name, "axon")

    def test_an_unknown_branch_name_names_the_placement_and_rolls_back(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        rules = cell.place_rules

        with self.assertRaisesRegex(KeyError, r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ .*branch 'dned'"):
            cell.place(at("dned", 0.5), _index_probe_synapse())
        self.assertEqual(cell.place_rules, rules)

    def test_per_member_expressions_on_a_removed_branch_name_the_placement(self):
        morpho = make_tree_u()
        cell = Cell(morpho, pop_size=2)
        cell.place([at("dend", 0.5), at("apical", 0.5)], _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(KeyError, r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ .*branch 'dend'"):
            cell.n_cv

    def test_a_mask_placement_raises_after_a_move(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.reattach("dend", parent="apical")

        with self.assertRaisesRegex(
            RuntimeError, r"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ .*LocsetMask.*removed or moved"
        ):
            cell.n_cv

    def test_an_expression_placement_follows_a_move(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("dend", 1.0), _index_probe_synapse())

        morpho.reattach("dend", parent="apical")

        self.assertEqual(_placed_branches(cell), [(3, "dend")])

    def test_a_mask_placement_survives_an_append(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.soma.extra = _extra_branch()

        self.assertEqual(cell.n_cv, 5)
        self.assertEqual(_placed_branches(cell), [(2, "axon")])

    def test_a_mask_placement_survives_an_in_place_replacement(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.replace(
            "dend",
            Branch.from_lengths(lengths=[200.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"),
        )

        self.assertEqual(_placed_branches(cell), [(2, "axon")])

    def test_an_aligned_batch_raises_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho, pop_size=2)
        batch = LocsetBatch.from_columns(np.asarray([[2], [3]], dtype=np.int64), np.asarray([[0.5], [0.5]]))
        cell.place(batch, _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _BATCH_PLACE_STALE):
            cell.n_cv

    def test_per_member_masks_raise_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho, pop_size=2)
        mask = at("axon", 0.5).evaluate(morpho)
        cell.place([mask, at("apical", 0.5)], _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.n_cv

    def test_a_population_view_mask_raises_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho, pop_size=2)
        cell[[0]].place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.n_cv

    def test_a_location_view_mask_raises_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.loc(at("axon", 0.5).evaluate(morpho)).place(_index_probe_synapse())

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.n_cv

    def test_a_declaration_after_the_removal_is_rejected_and_rolled_back(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())
        rules = cell.place_rules

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell.place(at("apical", 0.5), _index_probe_synapse())
        self.assertEqual(cell.place_rules, rules)

    def test_a_population_view_after_the_removal_raises_and_adds_no_rule(self):
        # The view checks the grid before it reaches place(), so the stale
        # rule is reported there, even for a view taken before the removal.
        morpho = make_tree_u()
        cell = Cell(morpho, pop_size=2)
        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())
        rules = cell.place_rules
        early_view = cell[[0]]

        morpho.remove("dend")

        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            cell[[0]]
        with self.assertRaisesRegex(RuntimeError, _MASK_PLACE_STALE):
            early_view.place(at("apical", 0.5), _index_probe_synapse())
        self.assertEqual(cell.place_rules, rules)

    def test_a_mask_on_a_morphology_reassigned_away_is_not_checked(self):
        first = make_tree_u()
        cell = Cell(first)
        cell.place(at("axon", 0.5).evaluate(first), _index_probe_synapse())
        cell.morphology = make_tree_u()

        first.remove("dend")

        self.assertEqual(_placed_branches(cell), [(2, "axon")])

    def test_a_mask_placed_after_a_removal_is_valid(self):
        # The recorded revision is the one at place(): a mask resolved on
        # the tree as it now is holds current indices.
        morpho = make_tree_u()
        cell = Cell(morpho)
        morpho.remove("dend")

        cell.place(at("axon", 0.5).evaluate(morpho), _index_probe_synapse())

        self.assertEqual(_placed_branches(cell), [(1, "axon")])

    def test_cv_per_branch_list_raises_after_a_removal_and_an_append(self):
        # Same branch count as declared, so the length check cannot see it:
        # replaying (1, 2, 1, 1) would give dend's 2 CVs to the axon.
        morpho = make_tree_u()
        cell = Cell(morpho, cv_policy=CVPerBranchList((1, 2, 1, 1)))

        morpho.remove("dend")
        morpho.soma.extra = _extra_branch()

        with self.assertRaisesRegex(RuntimeError, _CV_LIST_STALE):
            cell.cvs

    def test_cv_per_branch_list_assigned_later_is_checked_too(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.cv_policy = CVPerBranchList((1, 2, 1, 1))

        morpho.remove("dend")
        morpho.soma.extra = _extra_branch()

        with self.assertRaisesRegex(RuntimeError, _CV_LIST_STALE):
            cell.n_cv

    def test_cv_per_branch_list_can_be_assigned_again_after_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho, cv_policy=CVPerBranchList((1, 2, 1, 1)))
        morpho.remove("dend")

        cell.cv_policy = CVPerBranchList((1, 2, 1))

        self.assertEqual([cv.branch_id for cv in cell.cvs], [0, 1, 1, 2])

    def test_a_failed_cv_policy_assignment_keeps_the_previous_basis(self):
        # A rejected assignment rolls back the policy, and with it the
        # revision its indices were declared at.
        morpho = make_tree_u()
        cell = Cell(morpho, cv_policy=CVPerBranchList((1, 2, 1, 1)))
        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, "cv_per_branch length must match"):
            cell.cv_policy = CVPerBranchList((1, 2, 1, 1))
        with self.assertRaisesRegex(RuntimeError, _CV_LIST_STALE):
            cell.n_cv

    def test_cv_per_branch_list_survives_an_in_place_replacement(self):
        morpho = make_tree_u()
        cell = Cell(morpho, cv_policy=CVPerBranchList((1, 2, 1, 1)))

        morpho.replace(
            "dend",
            Branch.from_lengths(lengths=[200.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite"),
        )

        self.assertEqual([cv.branch_id for cv in cell.cvs], [0, 1, 1, 2, 3])

    def test_cv_per_branch_list_inside_a_composite_raises_after_a_removal(self):
        morpho = make_tree_u()
        policy = CompositeByTypePolicy(
            rules=(CVPolicyByTypeRule(("apical_dendrite",), CVPerBranchList((1, 1, 1, 3))),),
            default_policy=CVPerBranch(),
        )
        cell = Cell(morpho, cv_policy=policy)

        morpho.remove("dend")
        morpho.soma.extra = _extra_branch()

        with self.assertRaisesRegex(RuntimeError, r"cv_policy CompositeByTypePolicy\(.*default branch index"):
            cell.n_cv

    def test_other_policies_follow_a_removal(self):
        morpho = make_tree_u()
        cell = Cell(morpho, cv_policy=MaxCVLen(40.0 * u.um))
        cell.place(at("axon", 0.5), _index_probe_synapse())

        morpho.remove("dend")

        fresh = Cell(cold_rebuild(morpho), cv_policy=MaxCVLen(40.0 * u.um))
        fresh.place(at("axon", 0.5), _index_probe_synapse())
        self.assertEqual(_declaration_fingerprint(cell), _declaration_fingerprint(fresh))

    def test_every_edit_case_keeps_or_rejects_a_mask_placement(self):
        # Rows that keep every default index keep the mask valid: the Cell
        # matches one declared with the same mask on the cold rebuild. Rows
        # that move indices raise at the next rebuild.
        def declare(morpho: Morphology, mask) -> Cell:
            cell = Cell(morpho, cv_policy=MaxCVLen(40.0 * u.um))
            cell.place(mask, _index_probe_synapse())
            return cell

        for case in EDIT_CASES:
            with self.subTest(case.name):
                morpho = make_tree_t()
                mask = (at("axon", 0.5) + at("d1", 1.0)).evaluate(morpho)
                cell = declare(morpho, mask)
                _declaration_fingerprint(cell)
                case.edit(morpho)
                if case.reindex:
                    with self.assertRaisesRegex(RuntimeError, r"LocsetMask.*no longer name the same branches"):
                        cell.n_cv
                else:
                    expected = _declaration_fingerprint(declare(cold_rebuild(morpho), mask))
                    self.assertEqual(_declaration_fingerprint(cell), expected)


def _first_member(locset):
    """Each way of calling place() for member 0 of a two-member Cell."""
    return (
        ("cell.place", lambda cell: cell.place(locset, _index_probe_synapse())),
        ("population view", lambda cell: cell[[0]].place(locset, _index_probe_synapse())),
        ("location view", lambda cell: cell[[0]].loc(locset).place(_index_probe_synapse())),
        ("per member", lambda cell: cell.place([locset, at("axon", 0.5)], _index_probe_synapse())),
    )


class FailedPlaceKeepsTheDeclarationTest(unittest.TestCase):
    """A place() that fails leaves the Cell's rules and grid as they were.

    The population-view path (``cell[[0]].place``) used to keep the new rule
    after a failed rebuild, so every later grid read raised the same error.
    """

    def _cell(self) -> Cell:
        cell = Cell(make_tree_u(), pop_size=2)
        cell.place(at("soma", 0.5), mech.Synapse("ExpSyn", name="kept", tau=2.0 * u.ms, e=0.0 * u.mV))
        return cell

    def test_a_failed_rebuild_rolls_back_on_every_path(self):
        for name, place in _first_member(at("no_such_branch", 0.5)):
            with self.subTest(name):
                cell = self._cell()
                rules, n_cv, placed = cell.place_rules, cell.n_cv, _placed_branches(cell)
                with self.assertRaisesRegex(KeyError, "no_such_branch"):
                    place(cell)
                self.assertEqual(cell.place_rules, rules)
                self.assertEqual(cell.n_cv, n_cv)
                self.assertEqual(_placed_branches(cell), placed)

    def test_a_mask_naming_a_missing_branch_is_rejected_on_every_path(self):
        missing = r"LocsetMask names branch 9, but the Cell's morphology has 4 branches \(indices 0 to 3\)"
        for name, place in _first_member(LocsetMask(points=((9, 0.5),))):
            with self.subTest(name):
                cell = self._cell()
                rules, n_cv = cell.place_rules, cell.n_cv
                with self.assertRaisesRegex(IndexError, missing):
                    place(cell)
                self.assertEqual(cell.place_rules, rules)
                self.assertEqual(cell.n_cv, n_cv)

    def test_a_named_mask_or_a_batch_naming_a_missing_branch_is_rejected(self):
        named = LocsetMask(points=((0, 0.5), (4, 0.5)), display_names=("soma", "gone"))
        batch = LocsetBatch.from_columns(np.asarray([[0], [7]], dtype=np.int64), np.asarray([[0.5], [0.5]]))
        cases = (
            ("named mask", lambda cell: cell[[0]].place(named, _index_probe_synapse()), "LocsetMask names branch 4"),
            ("batch", lambda cell: cell.place(batch, _index_probe_synapse()), "LocsetBatch names branch 7"),
            (
                "batch on a view",
                lambda cell: cell[[0, 1]].place(batch, _index_probe_synapse()),
                "LocsetBatch names branch 7",
            ),
            ("batch location view", lambda cell: cell.loc(batch).place(_index_probe_synapse()), "LocsetBatch"),
        )
        for name, place, message in cases:
            with self.subTest(name):
                cell = self._cell()
                rules = cell.place_rules
                with self.assertRaisesRegex(IndexError, message):
                    place(cell)
                self.assertEqual(cell.place_rules, rules)


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
        self.assertIsNone(cell.runtime.axial_operator_cache)
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

    def _run_leak_cell(self, **kwargs):
        cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), **kwargs)
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
        cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
        return cell

    def test_integer_valued_voltages_match_float_bitwise(self):
        reference = self._run_leak_cell(V_init=-65.0 * u.mV, V_th=0.0 * u.mV)
        dtype = brainstate.environ.dftype()
        ref_v = np.asarray(reference.V.value.to_decimal(u.mV))

        def int_callable(shape, **_):
            return u.Quantity(jnp.full(shape, -65), u.mV)

        for label, kwargs in {
            "constructor": dict(V_init=-65 * u.mV, V_th=0 * u.mV),
            "callable": dict(V_init=int_callable, V_th=0 * u.mV),
        }.items():
            with self.subTest(label):
                cell = self._run_leak_cell(**kwargs)
                np.testing.assert_array_equal(np.asarray(cell.V.value.to_decimal(u.mV)), ref_v)
                self.assertEqual(cell.V.value.mantissa.dtype, dtype)
                self.assertEqual(cell.V_th.mantissa.dtype, dtype)
                np.testing.assert_array_equal(
                    np.asarray(cell.V_th.to_decimal(u.mV)), np.asarray(reference.V_th.to_decimal(u.mV))
                )

    def test_integer_valued_voltage_setters_and_view_set(self):
        reference = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=(2,), V_init=-65.0 * u.mV)
        reference.V_th = 3.0 * u.mV
        reference.init_state()
        reference[1].set(V_init=-60.0 * u.mV, V_th=1.0 * u.mV)

        cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), pop_size=(2,))
        cell.V_init = -65 * u.mV
        cell.V_th = 3 * u.mV
        cell.init_state()
        cell[1].set(V_init=-60 * u.mV, V_th=1 * u.mV)

        dtype = brainstate.environ.dftype()
        self.assertEqual(cell.V_th.mantissa.dtype, dtype)
        np.testing.assert_array_equal(
            np.asarray(cell.V_th.to_decimal(u.mV)), np.asarray(reference.V_th.to_decimal(u.mV))
        )
        self.assertEqual(cell.V.value.mantissa.dtype, dtype)
        cell.reset_state()
        reference.reset_state()
        np.testing.assert_array_equal(
            np.asarray(cell.V.value.to_decimal(u.mV)), np.asarray(reference.V.value.to_decimal(u.mV))
        )

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
            self.assertIsNone(cell.runtime.axial_operator_source)
            self.assertIsNone(cell.runtime.axial_operator_cache)

            operator32 = cell._get_axial_operator()
            cache32 = cell.runtime.axial_operator_cache
            self.assertEqual(operator32.dtype, jnp.dtype(jnp.float32))
            self.assertIsNotNone(cache32)
            self.assertEqual(cache32.operator.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell.runtime.axial_operator_source.dtype, jnp.float32)

        with brainstate.environ.context(precision=64):
            operator64 = cell._get_axial_operator()
            cache64 = cell.runtime.axial_operator_cache
            self.assertEqual(operator64.dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)

    def test_staggered_run_does_not_build_dense_axial_operator(self):
        cell = _cell_with_probe()
        cell.init_state()

        cell.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertIsNone(cell.runtime.axial_operator_source)
        self.assertIsNone(cell.runtime.axial_operator_cache)
        # JIT-built numeric coefficients must not leak tracers into a host cache.
        if cell.runtime.dhs_source is not None:
            self.assertFalse(isinstance(cell.runtime.dhs_source.diag_ms_inv, jax.core.Tracer))

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

        with self.assertWarns(DeprecationWarning):
            returned = cell.bind_synapse_input("ampa", source, weight=2.0)

        self.assertIs(returned, cell)
        self.assertEqual(len(cell._synapse_input_bindings["ampa"]), 1)

    def test_bind_synapse_input_is_deprecated(self):
        # API-7: the warning names the removal release and the replacement, at the calling line.
        cell = _simple_cell()
        source = lambda: jnp.asarray([1.0])
        removed_in = re.escape(cell_module._BIND_SYNAPSE_INPUT.removed_in)
        pattern = (
            rf"^braincell\.Cell\.bind_synapse_input\(\) is deprecated since braincell 0\.3 and will be removed in "
            rf"braincell {removed_in}\. Use an EventSource with braincell\.connect\(\) instead\.$"
        )
        with self.assertWarnsRegex(DeprecationWarning, pattern) as caught:
            line, returned = sys._getframe().f_lineno, cell.bind_synapse_input("ampa", source, weight=2.0)

        self.assertEqual((caught.filename, caught.lineno), (__file__, line))
        self.assertIs(returned, cell)
        self.assertEqual(cell._synapse_input_bindings["ampa"], [(source, 2.0, None)])


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
            # The first update after init_state prepares its own boundary
            # first (CELL-35); the next, eager, finds it prepared.
            cell.update()

        self.assertEqual(calls, ["prepare", "solver", "integration", "prepare", "solver", "integration", "prepare"])

    def test_integration_phase_integrates_dependent_runtime_synapse(self):
        from unittest.mock import patch

        from braincell._base_channel import Synapse as RuntimeSynapse

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


class CellIonTotalCurrentSnapshotTest(unittest.TestCase):
    """``cache_ion_total_currents`` snapshots exactly what the ion hook would read."""

    def _calcium_cell(self):
        cell = Cell(_soma_tree(), cv_policy=CVPerBranch())
        cell.paint(AllRegion(), mech.Ion("CalciumDetailed", name="ca"))
        cell.paint(AllRegion(), mech.Channel("CaT_HM1992", name="cat", g_max=0.5 * u.mS / u.cm**2))
        cell.init_state()
        return cell, cell.runtime.get_ion("ca")

    def test_snapshot_sums_channel_and_external_currents(self):
        cell, ion = self._calcium_cell()
        external = 3.0 * u.uA / u.cm**2
        ion.register_external_current("external", lambda V, info: jnp.ones(V.shape) * external)

        cell.cache_ion_total_currents()
        expected = ion.current(cell.V.value) + external
        np.testing.assert_allclose(
            np.asarray(ion._cached_total_current.to_decimal(u.uA / u.cm**2)),
            np.asarray(expected.to_decimal(u.uA / u.cm**2)),
            rtol=1e-6,
        )
        cell.clear_ion_total_current_cache()
        self.assertFalse(hasattr(ion, "_cached_total_current"))

    def test_an_external_current_error_is_not_retried_without_it(self):
        cell, ion = self._calcium_cell()

        def broken(V, info):
            raise TypeError("broken external current")

        ion.register_external_current("broken", broken)
        with self.assertRaisesRegex(TypeError, "broken external current"):
            cell.cache_ion_total_currents()
        self.assertFalse(hasattr(ion, "_cached_total_current"))

    def test_disabled_cache_leaves_ions_reading_live_currents(self):
        cell, ion = self._calcium_cell()
        cell.cache_ion_total_current = False
        cell.cache_ion_total_currents()
        self.assertFalse(hasattr(ion, "_cached_total_current"))


# Cell-only steps: staggered schedules the mechanisms itself, dhs_voltage
# advances V only.
_CABLE_ONLY = ("staggered", "dhs_voltage")


def _generic_solver_names():
    """Every registered name a Cell accepts besides the cable-only steps, with two aliases."""
    names = [name for name in braincell.quad.list_integrators() if name not in _CABLE_ONLY]
    return names + ["heun3", "ssprk3"]


def _mechanism_cell(solver, mechanism, *, order="family"):
    """The one-CV soma of CELL-19 with ``IL`` and one more mechanism."""
    cell = Cell(_soma_tree(), cv_policy=CVPerBranch(), solver=solver, ion_channel_update_order=order)
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    if mechanism == "markov":
        cell.paint(AllRegion(), mech.Ion("SodiumFixed", name="na", E=55.0 * u.mV))
        cell.paint(AllRegion(), mech.Channel("Nav_MA2020_GrC", g_max=10.0 * u.mS / u.cm**2, ion_name="na"))
    elif mechanism == "kinetic":
        cell.paint(AllRegion(), mech.Ion("ToyCaBindingKinetic_SU2015_DCN", name="ca"))
    elif mechanism == "kinetic_child":
        # A kinetic pool with a dependent (HH) channel of its own.
        cell.paint(AllRegion(), mech.Ion("CdpStC_MA2020_GoC", name="ca"))
        cell.paint(AllRegion(), mech.Channel("CaHVA_MA2020_GoC", g_max=0.5 * u.mS / u.cm**2, ion_name="ca"))
    return cell


def _diffeq_values(cell):
    """Final ``V`` (mV) and every other DiffEqState value, keyed by graph path."""
    states = {
        ".".join(map(str, path)): np.asarray(u.get_mantissa(st.value), dtype=np.float64).ravel()
        for path, st in brainstate.graph.states(cell).items()
        if isinstance(st, DiffEqState) and path != ("V",)
    }
    return np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).ravel(), states


def _run_to_1ms(cell, dt_ms):
    cell.init_state()
    cell.run(dt=dt_ms * u.ms, duration=1.0 * u.ms)
    return _diffeq_values(cell)


class CellGenericSolverTest(unittest.TestCase):
    """CELL-19: every registered solver name runs on a cable Cell or is rejected at init_state."""

    def _assert_converges_to_staggered(self, build, names):
        # V at 1 ms within 1e-3 mV of staggered at dt = 0.005 ms, the gap
        # shrinking at least 3-fold to dt = 0.001 ms; mechanism states within 1e-3.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            reference = {dt: _run_to_1ms(build("staggered"), dt) for dt in (0.005, 0.001)}
            for name in names:
                with self.subTest(solver=name):
                    gap, state_gap = {}, {}
                    for dt in (0.005, 0.001):
                        V, states = _run_to_1ms(build(name), dt)
                        ref_V, ref_states = reference[dt]
                        self.assertEqual(set(states), set(ref_states))
                        gap[dt] = float(np.max(np.abs(V - ref_V)))
                        state_gap[dt] = max([float(np.max(np.abs(states[k] - ref_states[k]))) for k in states] or [0.0])
                    self.assertLessEqual(gap[0.005], 1e-3)
                    if gap[0.005] > 1e-12:
                        self.assertGreaterEqual(gap[0.005] / gap[0.001], 3.0)
                    self.assertLessEqual(state_gap[0.005], 1e-3)

    def test_every_generic_solver_converges_to_staggered(self):
        for mechanism in ("none", "markov", "kinetic"):
            with self.subTest(mechanism=mechanism):
                self._assert_converges_to_staggered(
                    lambda name, m=mechanism: _mechanism_cell(name, m), _generic_solver_names()
                )

    def test_dependent_channels_of_a_kinetic_ion_follow_the_solver(self):
        # The HH channel under a kinetic pool is integrated by the main solver,
        # the pool by its subsolver; it used to fail in run ("must be written",
        # or a unit-mismatch assertion under the block solvers).
        names = ("ralston2", "ind_exp_euler", "exp_euler", "backward_euler", "crank_nicolson")
        self._assert_converges_to_staggered(lambda name: _mechanism_cell(name, "kinetic_child"), names)

    def test_integration_order_advances_markov_channels_after_the_solver(self):
        self._assert_converges_to_staggered(
            lambda name: _mechanism_cell(name, "markov", order="integration"), ("ralston2", "exp_euler")
        )

    def test_independent_mechanisms_advance_after_a_generic_solver(self):
        for order, expected in (
            ("family", ["solver", "cache", "ions", "channels"]),
            ("integration", ["solver", "cache", "top_level"]),
        ):
            with self.subTest(order=order):
                calls = []
                cell = _mechanism_cell("ralston2", "markov", order=order)
                cell.init_state()
                cell._solver_fn = lambda target: calls.append("solver")
                cell._ind_update_after_solver = True
                cell.cache_ion_total_currents = lambda V=None: calls.append("cache")
                cell._update_independent_ions = lambda nodes, V: calls.append("ions")
                cell._update_independent_channels = lambda nodes, V: calls.append("channels")
                cell._ind_update_top_level_nodes = lambda V: calls.append("top_level")
                cell._update_ion_channel_families = lambda V: calls.append("family")
                with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                    cell.update()
                self.assertEqual(calls, expected)

    def test_which_solvers_advance_independent_mechanisms(self):
        def custom(target):
            pass

        for solver, expected in (
            ("ralston2", True),
            ("ind_exp_euler", True),
            ("backward_euler", True),
            (braincell.quad.rk4_step, True),
            ("staggered", False),
            ("dhs_voltage", False),
            (custom, False),
        ):
            with self.subTest(solver=getattr(solver, "__name__", solver)):
                cell = _mechanism_cell(solver, "markov")
                cell.init_state()
                self.assertIs(cell._ind_update_after_solver, expected)

    def test_a_registered_step_passed_as_a_callable_runs_markov_channels(self):
        cell = _mechanism_cell(braincell.quad.rk4_step, "markov")
        cell.init_state()
        before = _diffeq_values(cell)[1]
        cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
        after = _diffeq_values(cell)[1]
        self.assertTrue(any(not np.array_equal(before[k], after[k]) for k in before))

    def test_block_solvers_reject_point_synapses_at_init_state(self):
        for name in ("exp_euler", "backward_euler", "crank_nicolson"):
            with self.subTest(solver=name):
                cell = _mechanism_cell(name, "none")
                cell.place(at("soma", 0.5), mech.Synapse("Exp2Syn", name="syn"))
                with self.assertRaisesRegex(
                    ValueError, rf"solver '{name}' cannot advance a Cell with point synapses \(Exp2Syn\)"
                ):
                    cell.init_state()
                self.assertFalse(cell._initialized)
                cell.solver = "ralston2"
                cell.init_state()
                cell.run(dt=0.025 * u.ms, duration=0.05 * u.ms)

    def test_point_synapses_in_a_network_converge_under_generic_solvers(self):
        # Exp2Syn driven by a NetStim. The explicit solvers integrate the
        # synapse in their stages and agree with the Richardson limit of
        # staggered; ind_exp_euler is first order like staggered.
        def network(solver):
            cell = _mechanism_cell(solver, "none")
            cell.place(
                at("soma", 0.5),
                mech.Synapse("Exp2Syn", name="syn", tau1=0.5 * u.ms, tau2=5.0 * u.ms, e=0.0 * u.mV),
            )
            net = braincell.Network(name="exp2syn")
            source = net.add_population("stim", NetStim(size=1, start=0.1 * u.ms, interval=0.3 * u.ms, number=3))
            target = net.add_population("post", cell)
            net.connect("drive", source=source, synapse=target.synapses["syn"], weight=0.01 * u.uS)
            return net, cell

        def final_V(solver, dt):
            net, cell = network(solver)
            net.init_state()
            net.run(dt=dt * u.ms, duration=1.0 * u.ms)
            return float(cell.V.value.to_decimal(u.mV).reshape(-1)[0])

        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            limit = 2.0 * final_V("staggered", 0.00025) - final_V("staggered", 0.0005)
            # Measured: rk4 1.2e-7 mV, ralston2 1.25e-5 mV; staggered itself is
            # 0.21 mV off at dt = 0.005 ms here, so it is not the yardstick.
            self.assertLessEqual(abs(final_V("rk4", 0.005) - limit), 5e-7)
            self.assertLessEqual(abs(final_V("ralston2", 0.001) - limit), 5e-5)
            gap = {dt: abs(final_V("ind_exp_euler", dt) - limit) for dt in (0.005, 0.001)}
            self.assertGreaterEqual(gap[0.005] / gap[0.001], 3.0)
            net, _ = network("exp_euler")
            with self.assertRaisesRegex(ValueError, "solver 'exp_euler' cannot advance a Cell with point synapses"):
                net.init_state()


class DiscretizationTracksAMutatedMorphologyTest(unittest.TestCase):
    """A ``Cell`` shares its ``Morphology``, so it must notice edits to it.

    ``Cell.morpho`` is documented as returning the tree "without copying
    it", and ``Morphology`` is mutable -- it maintains a revision counter
    precisely so consumers can tell that it changed. The discretization
    cache has to consult that counter; the object's identity alone cannot
    distinguish a tree that has grown a branch from one that has not.
    """

    @staticmethod
    def _dendrite() -> Branch:
        return Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="basal_dendrite",
        )

    def test_attaching_a_branch_changes_the_cv_count(self) -> None:
        tree = _soma_tree()
        cell = Cell(tree, cv_policy=CVPerBranch(1))
        self.assertEqual(cell.n_cv, 1)

        tree.soma.dend = self._dendrite()

        self.assertEqual(tree.n_branches, 2)
        self.assertEqual(cell.n_cv, 2)

    def test_a_rebuild_does_not_leave_a_stale_root_scope(self) -> None:
        tree = _soma_tree()
        cell = Cell(tree, cv_policy=CVPerBranch(1))
        # Materialize every cache that hangs off the discretization.
        self.assertEqual(len(cell.soma.cv.ids), 1)

        tree.soma.dend = self._dendrite()

        # ``cell.soma`` selects by branch type, so the dendrite is excluded;
        # the unrestricted scope must still see both CVs.
        self.assertEqual(len(cell.cvs), 2)
        self.assertEqual(cell.cv.ids.tolist(), [0, 1])


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


class AlignedBatchPlacementLandsEachMemberOnItsOwnLocationTest(unittest.TestCase):
    """An aligned ``LocsetBatch`` must place member *n* at member *n*'s location.

    ``build_cv_mechanisms`` resolves the batch by indexing it inside a generator
    expression, so every ``LocsetBatch`` member is a temporary that dies as soon
    as it has been resolved. While ``_RegionCache`` keyed its results on
    ``id()``, CPython reused the freed address for the next member and the cache
    replayed the previous member's locations -- silently placing synapses on the
    wrong branch and the wrong CV for every member after the first. Nothing
    caught it: the only other ``LocsetBatch`` coverage drives ``.loc()``, which
    does not go through that cache.
    """

    def _two_branch_cell(self, pop_size: int, *, cv_per_branch: int = 1) -> Cell:
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[5.0, 0.5] * u.um, type="dendrite")
        tree = Morphology.from_root(soma, name="soma")
        tree.soma.dend = dend
        return Cell(tree, pop_size=pop_size, cv_policy=CVPerBranch(cv_per_branch=cv_per_branch))

    def test_each_population_member_keeps_its_own_branch_and_x(self) -> None:
        expected = ((0, 0.1), (1, 0.9), (0, 0.2), (1, 0.8))
        cell = self._two_branch_cell(len(expected))
        batch = LocsetBatch.from_columns(
            np.asarray([[branch] for branch, _ in expected], dtype=np.int64),
            np.asarray([[x] for _, x in expected], dtype=float),
        )
        cell[np.arange(len(expected))].place(
            batch,
            mech.Synapse("ExpSyn", name="syn", tau=5.0 * u.ms, e=0.0 * u.mV),
        )

        placed = {
            int(placement.population_index): (int(placement.branch_id), round(float(placement.branch_x), 6))
            for placement in cell._discretization.point_placements
        }
        self.assertEqual(placed, dict(enumerate(expected)))

    def test_members_on_the_same_branch_still_land_on_distinct_cvs(self) -> None:
        expected = ((1, 0.1), (1, 0.5), (1, 0.9))
        cell = self._two_branch_cell(len(expected), cv_per_branch=3)
        batch = LocsetBatch.from_columns(
            np.asarray([[branch] for branch, _ in expected], dtype=np.int64),
            np.asarray([[x] for _, x in expected], dtype=float),
        )
        cell[np.arange(len(expected))].place(
            batch,
            mech.Synapse("ExpSyn", name="syn", tau=5.0 * u.ms, e=0.0 * u.mV),
        )

        cv_by_member = {
            int(placement.population_index): int(placement.cv_id) for placement in cell._discretization.point_placements
        }
        self.assertEqual(len(set(cv_by_member.values())), len(expected), cv_by_member)


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
    """Every ``Cell`` hidden state groups its trailing spatial axis."""

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
        - A painted density channel's gates are CV space, so ``n_cv``.
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
        return cell.n_cv

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
        self.assertEqual(seen_spaces, {cell.n_cv})

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


def _topology_cell(**kwargs) -> Cell:
    from braincell._discretization import CVPerBranch
    from braincell.morph._testing import make_soma_dend_tree

    return Cell(make_soma_dend_tree(), cv_policy=CVPerBranch(), **kwargs)


class CellTopologyDataGuardTest(unittest.TestCase):
    """Argument validation of ``Cell.topology_data`` that happens before any level is built."""

    def test_an_unknown_level_names_the_valid_ones(self) -> None:
        with self.assertRaisesRegex(ValueError, r"level must be one of \{.*'node'.*'cv'.*'branch'.*\}"):
            _topology_cell().topology_data("layout")

    def test_node_level_requires_init_state(self) -> None:
        # The only level that reads runtime state, so the only one that
        # refuses a declaration-only cell.
        with self.assertRaisesRegex(RuntimeError, r"requires init_state\(\) first"):
            _topology_cell().topology_data("node")

    def test_node_is_the_default_level(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        self.assertEqual(cell.topology_data().level, "node")


class CellTopologyDataCvLevelTest(unittest.TestCase):
    def test_graph_before_init(self) -> None:
        cell = _topology_cell()
        graph = cell.topology_data("cv")
        self.assertEqual(graph.level, "cv")
        self.assertEqual(graph.node_ids, tuple(cv.id for cv in cell.cvs))
        self.assertEqual(graph.edges, ((0, 1),))
        self.assertEqual(graph.root_id, 0)
        self.assertIsNone(graph.values)
        self.assertIsNone(graph.highlight_fractions)

    def test_region_gives_area_coverage_fractions(self) -> None:
        graph = _topology_cell().topology_data("cv", region=BranchSlice(branch_index=1, prox=0.0, dist=0.5))
        # CV shading is the membrane-area fraction, the same one that scales
        # a painted conductance. The dendrite tapers, so its proximal half
        # is 0.5833... of the area and only 0.5 of the length. Branch-level
        # fractions below are a drawing extent and stay length-based.
        self.assertAlmostEqual(graph.highlight_fractions[1], 0.5833333333333334)

    def test_locset_is_full_intensity(self) -> None:
        graph = _topology_cell().topology_data("cv", locset=RootLocation(0.5))
        self.assertEqual(graph.highlight_fractions[0], 1.0)

    def test_value_V_uses_cv_values(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        graph = cell.topology_data("cv", value="V")
        self.assertEqual(graph.values.shape, (cell.n_cv,))
        self.assertEqual(graph.value_label, "V")

    def test_raw_point_values_are_gathered_to_cv(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        point_values = np.asarray([np.nan, 1.0, np.nan, 2.0, np.nan], dtype=float)
        graph = cell.topology_data("cv", value=point_values)
        np.testing.assert_allclose(graph.values, cell._point_to_cv(point_values))
        self.assertIsNone(graph.value_label)

    def test_a_two_dimensional_value_gives_one_frame_per_row(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        frames = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        graph = cell.topology_data("cv", value=frames)
        self.assertEqual(graph.n_frames, 3)
        np.testing.assert_allclose(graph.frame(2), [5.0, 6.0])

    def test_layout_id_lookup_works(self) -> None:
        cell = _topology_cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        graph = cell.topology_data("cv", value=("layout_id", cell.layouts[0].id, "g_max"))
        self.assertEqual(graph.values.shape, (cell.n_cv,))

    def test_an_unknown_layout_id_raises_key_error(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        with self.assertRaises(KeyError):
            cell.topology_data("cv", value=("layout_id", 999_999, "g_max"))

    def test_value_with_region_is_rejected(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            cell.topology_data("cv", region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), value="V")

    def test_a_value_of_the_wrong_length_is_rejected(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        with self.assertRaises(ValueError):
            cell.topology_data("cv", value=np.zeros(7))

    def test_more_than_one_root_cv_is_rejected(self) -> None:
        # A forest has no single node to lay the graph out from. Branch
        # level tolerates this; CV level does not. A Morphology always has
        # one root, so the second one has to be injected to reach the guard.
        cell = _topology_cell()
        cvs = tuple(cell.cvs)
        forest = cvs + (dataclasses.replace(cvs[-1], id=len(cvs), parent_cv=None),)
        with mock.patch.object(Cell, "cvs", new_callable=mock.PropertyMock) as mocked_cvs:
            mocked_cvs.return_value = forest
            with self.assertRaisesRegex(ValueError, "expects exactly one root CV"):
                cell.topology_data("cv")


class CellTopologyDataBranchLevelTest(unittest.TestCase):
    def test_graph_before_init(self) -> None:
        graph = _topology_cell().topology_data("branch")
        self.assertEqual((graph.level, graph.node_ids, graph.edges, graph.root_id), ("branch", (0, 1), ((0, 1),), 0))

    def test_region_gives_length_coverage_fractions(self) -> None:
        graph = _topology_cell().topology_data("branch", region=BranchSlice(branch_index=1, prox=0.25, dist=0.75))
        self.assertAlmostEqual(graph.highlight_fractions[1], 0.5)

    def test_locset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support: locset"):
            _topology_cell().topology_data("branch", locset=RootLocation(0.5))

    def test_value_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support: value"):
            _topology_cell().topology_data("branch", value="V")

    def test_both_unsupported_arguments_are_reported_together(self) -> None:
        with self.assertRaisesRegex(ValueError, r"does not support value together with region/locset"):
            _topology_cell().topology_data("branch", locset=RootLocation(0.5), value="V")

    def test_the_plot_rejects_colormap_arguments_for_a_branch_graph(self) -> None:
        # The rejection of value-colormap parameters moved to the plot,
        # which sees graph.level == "branch".
        import braintools.visualize.morphology as bvm

        graph = _topology_cell().topology_data("branch")
        for name, argument in (
            ("cmap", "viridis"),
            ("vmin", 0.0),
            ("vmax", 1.0),
            ("value_label", "V"),
            ("show_colorbar", False),
        ):
            with self.subTest(parameter=name):
                with self.assertRaisesRegex(ValueError, name):
                    bvm.plot_topology_graph(graph, **{name: argument})


class CellTopologyDataNodeLevelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = _topology_cell()
        self.cell.init_state()

    def test_graph_matches_the_node_tree(self) -> None:
        graph = self.cell.topology_data("node")
        expected = self.cell.node_tree.to_topology_data()
        self.assertEqual(
            (graph.node_ids, graph.edges, graph.root_id),
            (expected.node_ids, expected.edges, expected.root_id),
        )

    def test_region_maps_to_midpoint_point_ids(self) -> None:
        target = int(self.cell.node_tree.cv_to_mid_node_id[1])
        graph = self.cell.topology_data(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0))
        self.assertEqual(graph.highlight_fractions[target], 1.0)

    def test_locset_maps_to_midpoint_point_ids(self) -> None:
        target = int(self.cell.node_tree.cv_to_mid_node_id[0])
        graph = self.cell.topology_data(locset=RootLocation(0.5))
        self.assertEqual(graph.highlight_fractions[target], 1.0)

    def test_value_with_region_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support value together with region/locset"):
            self.cell.topology_data(region=BranchSlice(branch_index=1, prox=0.0, dist=1.0), value="V")

    def test_raw_cv_values_are_converted_to_point_values(self) -> None:
        from braincell._multi_compartment import field_resolution

        cv_values = np.asarray([1.0, 2.0], dtype=float)
        graph = self.cell.topology_data(value=cv_values)
        np.testing.assert_allclose(
            np.nan_to_num(graph.values, nan=-1.0),
            np.nan_to_num(field_resolution.cv_to_node_values(self.cell, cv_values), nan=-1.0),
        )

    def test_value_V_uses_point_voltage(self) -> None:
        from braincell._multi_compartment import field_resolution

        graph = self.cell.topology_data(value="V")
        expected = field_resolution.cv_to_node_values(self.cell, self.cell.V.value)
        np.testing.assert_allclose(graph.values.to_decimal(u.mV), expected.to_decimal(u.mV))
        self.assertEqual(graph.value_label, "V")

    def test_ion_lookup_works(self) -> None:
        graph = self.cell.topology_data(value=("ion", "na", "E"))
        self.assertEqual(graph.values.shape, (self.cell.n_point,))
        self.assertEqual(graph.value_label, "na.E")

    def test_layout_id_lookup_works(self) -> None:
        cell = _topology_cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.init_state()
        graph = cell.topology_data(value=("layout_id", cell.layouts[0].id, "g_max"))
        self.assertEqual(graph.values.shape, (cell.n_point,))

    def test_channel_lookup_ambiguous_raises(self) -> None:
        cell = _topology_cell()
        full = BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0)
        cell.paint(full, mech.Channel("IL", name="leak_a", g_max=4.0 * (u.mS / u.cm**2), E=-68.0 * u.mV))
        cell.paint(full, mech.Channel("IL", name="leak_b", g_max=5.0 * (u.mS / u.cm**2), E=-67.0 * u.mV))
        cell.init_state()
        with self.assertRaisesRegex(ValueError, "multiple runtime layouts"):
            cell.topology_data(value=("channel", "IL", "g_max"))

    def test_a_missing_mechanism_field_raises_attribute_error(self) -> None:
        with self.assertRaises(AttributeError):
            self.cell.topology_data(value=("ion", "na", "no_such_field"))

    def test_the_graph_draws(self) -> None:
        import braintools.visualize.morphology as bvm
        import matplotlib.axes
        import matplotlib.pyplot as plt

        self.addCleanup(plt.close, "all")
        for graph in (self.cell.topology_data(value="V"), self.cell.node_tree):
            with self.subTest(graph=type(graph).__name__):
                ax = bvm.plot_topology_graph(graph, layout="kamada_kawai")
                self.assertIsInstance(ax, matplotlib.axes.Axes)


class CellTopologyDataPopulationAxisTest(unittest.TestCase):
    """A topology graph is one morphology, so values need one population member."""

    def test_default_single_member_population(self) -> None:
        cell = _topology_cell()
        cell.init_state()
        self.assertEqual(cell.pop_size, (1,))
        for level in ("cv", "node"):
            with self.subTest(level=level):
                self.assertIsNotNone(cell.topology_data(level, value="V").values)

    def test_multi_member_population_is_refused_with_a_useful_message(self) -> None:
        cell = _topology_cell(pop_size=4)
        cell.init_state()
        for level in ("cv", "node"):
            with self.subTest(level=level):
                with self.assertRaisesRegex(ValueError, r"population shape \(4,\).*pop_size=\(4,\)"):
                    cell.topology_data(level, value="V")

    def test_the_refusal_names_the_level_that_was_asked_for(self) -> None:
        cell = _topology_cell(pop_size=4)
        cell.init_state()
        with self.assertRaisesRegex(ValueError, r"Cell\.topology_data\(level='cv'\)"):
            cell.topology_data("cv", value="V")


if __name__ == "__main__":
    unittest.main()
