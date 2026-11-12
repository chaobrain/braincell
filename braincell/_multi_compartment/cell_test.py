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
import functools
import json
import os
import re
import subprocess
import sys
import unittest
import warnings
from unittest import mock

import brainpy
import brainstate
import braintools
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
from braincell._compute.state import CellRuntimeState
from braincell._multi_compartment import cell as cell_module
from braincell.filter import (
    AllRegion,
    BranchSlice,
    LocsetBatch,
    LocsetMask,
    RegionAnchors,
    RootLocation,
    StepSamples,
    SubtreeRegion,
    Terminals,
    at,
    branch_in,
    branch_range,
)
from braincell.mech import StateProbe
from braincell.quad._util import split_diffeq_states
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

    def test_region_anchors_and_step_samples_place_in_row_order(self):
        """One placement per locset row, in order; junction rows share a point (MORPH-10a, MORPH-10b)."""
        soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
        dend = Branch.from_lengths(
            lengths=np.array([40.0, 60.0]) * u.um, radii=np.array([2.0, 1.5, 1.0]) * u.um, type="basal_dendrite"
        )
        dend_region = branch_in("name", "dend")

        def placed(locset):
            morpho = Morphology.from_root(soma, name="soma")
            morpho.attach(parent="soma", child_branch=dend, child_name="dend")
            cell = Cell(morpho, cv_policy=CVPerBranch())
            synapse = mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms, e=0.0 * u.mV)
            cell.place(locset, synapse)
            rows = [(p.branch_name, p.branch_x, p.cv_id, p.point_id) for p in cell.point_placements]
            cell.init_state()
            return rows, len(cell.synapses[synapse].id)

        self.assertEqual(
            placed(RegionAnchors(AllRegion(), 1.0) + RegionAnchors(AllRegion(), 0.0)),
            ([("soma", 1.0, 0, 2), ("dend", 1.0, 1, 4), ("soma", 0.0, 0, 0), ("dend", 0.0, 1, 2)], 4),
        )
        repeated = RegionAnchors(dend_region, 0.5) + RegionAnchors(dend_region, 0.5)
        self.assertEqual(placed(repeated), ([("dend", 0.5, 1, 3)] * 2, 2))
        self.assertEqual(placed(repeated.unique()), ([("dend", 0.5, 1, 3)], 1))
        # MORPH-10b: StepSamples counts its steps across the junction.
        self.assertEqual(
            placed(StepSamples(AllRegion(), 25 * u.um)),
            ([("soma", 0.625, 0, 1), ("dend", 0.175, 1, 3), ("dend", 0.425, 1, 3), ("dend", 0.675, 1, 3)], 4),
        )


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


def _index_probe_leak():
    return mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-70.0 * u.mV, name="leak")


def _painted_branches(cell: Cell) -> list:
    return sorted({cell.morpho.branch(index=cell.cvs[int(c)].branch_id).name for c in cell.channels["leak"].cv_id})


def _paint_stale(index: str) -> str:
    return rf"Cell\.paint\(\.\.\.\) at cell_test\.py:\d+ selects .*by default {index}, .*removed or moved"


def _place_stale(index: str) -> str:
    return rf"Cell\.place\(\.\.\.\) at cell_test\.py:\d+ selects .*by default {index}, .*removed or moved"


class IndexExpressionDeclarationTest(unittest.TestCase):
    """MORPH-17: an expression that names a branch by index raises once an edit shifts the indices.

    Tree T: soma 0, dend 1, d1 2, d2 3, axon 4; removing ``dend`` (and its
    children) leaves soma 0, axon 1. Tree U: soma 0, dend 1, axon 2,
    apical 3. The expression is evaluated again at every rebuild, but the
    integer is not: it used to name whichever branch had that index now.
    """

    def test_a_branch_slice_painted_before_a_removal_raises_at_init_state(self):
        morpho = make_tree_t()
        cell = Cell(morpho, cv_policy=CVPerBranch())
        cell.paint(BranchSlice(1, 0.0, 1.0), _index_probe_leak())
        self.assertEqual(_painted_branches(cell), ["dend"])

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, _paint_stale("index 1")):
            cell.init_state()
        self.assertFalse(cell._initialized)
        self.assertIs(cell.morpho, morpho)
        with self.assertRaisesRegex(ValueError, _paint_stale("index 1")):
            cell.n_cv
        # The recovery the message names: a new Cell, naming the branch.
        fresh = Cell(morpho, cv_policy=CVPerBranch())
        fresh.paint(branch_in("name", "axon"), _index_probe_leak())
        fresh.init_state()
        self.assertEqual(_painted_branches(fresh), ["axon"])

    def test_a_removal_after_init_raises_after_reset(self):
        morpho = make_tree_t()
        cell = Cell(morpho, cv_policy=CVPerBranch())
        cell.paint(BranchSlice(1, 0.0, 1.0), _index_probe_leak())
        cell.init_state()

        morpho.remove("dend")

        # The initialized Cell runs its snapshot, where index 1 is still dend.
        self.assertEqual(_painted_branches(cell), ["dend"])
        cell.reset()
        with self.assertRaisesRegex(ValueError, _paint_stale("index 1")):
            cell.n_cv
        with self.assertRaisesRegex(ValueError, _paint_stale("index 1")):
            cell.init_state()
        self.assertFalse(cell._initialized)

    def test_a_point_by_index_raises_after_a_move(self):
        # reattach("dend", parent="apical") gives soma 0, axon 1, apical 2,
        # dend 3: at(1, 0.5) would put the synapse on the axon.
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at(1, 0.5), _index_probe_synapse())
        self.assertEqual(_placed_branches(cell), [(1, "dend")])

        morpho.reattach("dend", parent="apical")

        with self.assertRaisesRegex(ValueError, _place_stale("index 1")):
            cell.n_cv
        with self.assertRaisesRegex(ValueError, _place_stale("index 1")):
            cell.init_state()

    def test_a_subtree_by_index_raises_after_a_removal(self):
        morpho = make_tree_t()
        cell = Cell(morpho, cv_policy=CVPerBranch())
        cell.paint(SubtreeRegion(1), _index_probe_leak())
        self.assertEqual(_painted_branches(cell), ["d1", "d2", "dend"])

        morpho.remove("d2")

        with self.assertRaisesRegex(ValueError, _paint_stale("index 1")):
            cell.n_cv

    def test_every_way_of_declaring_by_index_is_checked(self):
        # cell.branch[1] declares branch_in("index", [1]); composite
        # expressions report every index they hold.
        declarations = (
            ("branch view paint", lambda cell: cell.branch[1].paint(_index_probe_leak()), _paint_stale("index 1")),
            (
                "composite paint",
                lambda cell: cell.paint(branch_in("type", "soma") | BranchSlice([3, 2], 0.0, 0.5), _index_probe_leak()),
                _paint_stale(r"indices 3, 2"),
            ),
            (
                "index range paint",
                lambda cell: cell.paint(branch_range("index", (2, 3), closed="both"), _index_probe_leak()),
                _paint_stale(r"indices 2, 3"),
            ),
            (
                "anchors on a slice",
                lambda cell: cell.place(RegionAnchors(BranchSlice(2, 0.0, 1.0), 0.5), _index_probe_synapse()),
                _place_stale("index 2"),
            ),
            (
                "population view",
                lambda cell: cell[[0]].place(at(2, 0.5), _index_probe_synapse()),
                _place_stale("index 2"),
            ),
            ("location view", lambda cell: cell.loc(at(2, 0.5)).place(_index_probe_synapse()), _place_stale("index 2")),
            (
                "per member",
                lambda cell: cell.place([at("axon", 0.5), at(2, 0.5)], _index_probe_synapse()),
                _place_stale("index 2"),
            ),
        )
        for name, declare, pattern in declarations:
            with self.subTest(name):
                morpho = make_tree_u()
                cell = Cell(morpho, pop_size=2)
                declare(cell)
                cell.n_cv
                morpho.remove("dend")
                with self.assertRaisesRegex(ValueError, pattern):
                    cell.n_cv

    def test_declarations_by_index_survive_an_append(self):
        # Appending keeps every default index (counter-example: no false alarm).
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.paint(BranchSlice(2, 0.0, 1.0), _index_probe_leak())
        cell.place(at(3, 0.5), _index_probe_synapse())
        before = (_painted_branches(cell), _placed_branches(cell))

        morpho.soma.extra = _extra_branch()

        self.assertEqual(cell.n_cv, 5)
        self.assertEqual((_painted_branches(cell), _placed_branches(cell)), before)
        self.assertEqual(before, (["axon"], [(3, "apical")]))
        cell.init_state()

    def test_declarations_by_name_are_not_checked(self):
        # Names follow the edit, or raise KeyError (MORPH-2b), never ValueError.
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.paint(branch_in("name", "dend"), _index_probe_leak())
        cell.paint(SubtreeRegion("axon"), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV, name="ax"))
        cell.place(at("apical", 0.5), _index_probe_synapse())
        self.assertIsNone(cell.paint_rules[-1]._index_basis)
        self.assertIsNone(cell.place_rules[-1]._index_basis)

        morpho.remove("dend")

        self.assertEqual(_placed_branches(cell), [(2, "apical")])
        cell.init_state()
        self.assertEqual(cell.channels["ax"].cv_id.tolist(), [1])

    def test_a_declaration_after_the_removal_is_rejected_and_rolled_back(self):
        morpho = make_tree_u()
        cell = Cell(morpho)
        cell.place(at(2, 0.5), _index_probe_synapse())
        rules = cell.paint_rules, cell.place_rules

        morpho.remove("dend")

        with self.assertRaisesRegex(ValueError, _place_stale("index 2")):
            cell.paint(branch_in("name", "axon"), _index_probe_leak())
        with self.assertRaisesRegex(ValueError, _place_stale("index 2")):
            cell.place(at("axon", 0.5), _index_probe_synapse())
        self.assertEqual((cell.paint_rules, cell.place_rules), rules)

    def test_an_index_declared_after_the_removal_is_valid(self):
        # The recorded revision is the one at paint(): the index is current.
        morpho = make_tree_t()
        morpho.remove("dend")
        cell = Cell(morpho, cv_policy=CVPerBranch())

        cell.paint(BranchSlice(1, 0.0, 1.0), _index_probe_leak())

        self.assertEqual(_painted_branches(cell), ["axon"])

    def test_an_index_on_a_morphology_reassigned_away_is_not_checked(self):
        first = make_tree_u()
        cell = Cell(first)
        cell.place(at(2, 0.5), _index_probe_synapse())
        cell.morphology = make_tree_u()

        first.remove("dend")

        self.assertEqual(_placed_branches(cell), [(2, "axon")])

    def test_every_edit_case_keeps_or_rejects_an_index_declaration(self):
        # Rows that keep every default index keep the declaration where it
        # was (by name) and match a Cell declared with it on the cold
        # rebuild; rows that move indices raise ValueError. Nothing else.
        def declare(morpho: Morphology) -> Cell:
            cell = Cell(morpho, cv_policy=MaxCVLen(40.0 * u.um))
            cell.paint(BranchSlice(4, 0.0, 1.0), _index_probe_leak())
            cell.place(at(2, 1.0) + at(4, 0.5), _index_probe_synapse())
            return cell

        def landing(cell: Cell) -> tuple:
            return _painted_branches(cell), [name for _, name in _placed_branches(cell)]

        for case in EDIT_CASES:
            with self.subTest(case.name):
                morpho = make_tree_t()
                cell = declare(morpho)
                self.assertEqual(landing(cell), (["axon"], ["d1", "axon"]))
                case.edit(morpho)
                if case.reindex:
                    with self.assertRaisesRegex(ValueError, _paint_stale("index 4")):
                        cell.n_cv
                else:
                    self.assertEqual(landing(cell), (["axon"], ["d1", "axon"]))
                    fresh = declare(cold_rebuild(morpho))
                    self.assertEqual(_declaration_fingerprint(cell), _declaration_fingerprint(fresh))


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
        cell._runtime_node_phase_args = lambda path, node, cv_V, stage_point_V=None: ("syn_arg",)
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

    def test_integration_order_hands_a_self_integrating_ions_dependent_channels_on(self):
        # CELL-32: the ion's own update leaves its channels out, so the
        # dependent pass advances its dependent channel (with the ion's info,
        # before the ion's update) and leaves the independent one to it.
        from braincell.quad.protocol import IndependentIntegration

        calls = []

        class _SelfIon(Ion, IndependentIntegration):
            def __init__(self):
                super().__init__(size=1, name=None)
                self.Ci = DiffEqSingleState(jnp.asarray([1.0]))
                self.Co = jnp.asarray([2.0])
                self.valence = 1

            @property
            def E(self):
                return jnp.asarray([0.0])

            def ind_update(self, V, *args, **kwargs):
                calls.append("ion")

        class _Dependent(Channel):
            root_type = _SelfIon

            def __init__(self):
                super().__init__(size=1, name=None)
                self.x = DiffEqSingleState(jnp.asarray([1.0]))

            def compute_derivative(self, V, *ions):
                calls.append(("dependent", len(ions)))
                self.x.derivative = -jnp.asarray([1.0]) / u.ms

            def current(self, V, ion):
                return 0.0

        class _Independent(Channel, IndependentIntegration):
            root_type = _SelfIon

            def __init__(self):
                super().__init__(size=1, name=None)
                self.y = DiffEqSingleState(jnp.asarray([1.0]))

            def compute_derivative(self, V, *ions):
                calls.append("independent")
                self.y.derivative = -jnp.asarray([1.0]) / u.ms

            def current(self, V, ion):
                return 0.0

        cell = _simple_cell()
        cell.ion_channel_update_order = "integration"
        ion = _SelfIon()
        ion.add(dep=_Dependent(), ind=_Independent())
        cell.init_state()
        cell.ion_channels["test_ion"] = ion

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            cell._update_ion_channels_by_integration(cell._cv_to_point(cell.V.value))

        self.assertIn(("dependent", 1), calls)
        self.assertNotIn("independent", calls)
        self.assertEqual(calls[-1], "ion")
        np.testing.assert_allclose(np.asarray(ion.channels["dep"].x.value), [0.9], rtol=1e-6)
        np.testing.assert_array_equal(np.asarray(ion.channels["ind"].y.value), [1.0])


def _kinetic_calcium_cell(order, solver="staggered"):
    """Soma 20 um x 10 um: leak 0.1 mS/cm^2 at -70 mV, kinetic Ca pool, CaHVA 0.5 mS/cm^2 under it."""
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = Cell(
        Morphology.from_root(soma, name="soma"),
        cv_policy=CVPerBranch(),
        solver=solver,
        ion_channel_update_order=order,
    )
    cell.paint(mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.paint(mech.Ion("CdpStC_MA2020_GoC", name="ca"))
    cell.paint(mech.Channel("CaHVA_MA2020_GoC", g_max=0.5 * u.mS / u.cm**2, ion_name="ca"))
    return cell


def _cahva_gates(cell):
    gates = {}
    for path, state in cell.states(brainstate.HiddenState).items():
        names = [str(part) for part in path]
        if any("CaHVA" in name for name in names) and names[-1] in ("s", "u"):
            gates[names[-1]] = float(np.asarray(u.get_mantissa(state.value)).reshape(-1)[0])
    return gates


def _kinetic_integration_report() -> dict:
    """The CELL-32 measurements; run in a float64 subprocess by :func:`_kinetic_integration_numbers`."""

    def run(order, solver, dt_ms):
        cell = _kinetic_calcium_cell(order, solver)
        cell.init_state()
        cell.run(dt=dt_ms * u.ms, duration=1.0 * u.ms)
        return {"V": float(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)[0]), **_cahva_gates(cell)}

    initial = _kinetic_calcium_cell("integration")
    initial.init_state()
    report = {"initial": _cahva_gates(initial)}
    for order, solver in (("integration", "staggered"), ("family", "staggered"), ("integration", "ralston2")):
        for dt_ms in (0.005, 0.001):
            report[f"{order}/{solver}/{dt_ms}"] = run(order, solver, dt_ms)
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    sodium = Cell(
        Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch(), ion_channel_update_order="integration"
    )
    sodium.paint(mech.Ion("SodiumFixed", name="na", E=50.0 * u.mV))
    sodium.paint(mech.Channel("Nav_MA2020_GrC", g_max=20.0 * u.mS / u.cm**2, ion_name="na"))
    for name, cell in (("sodium", sodium), ("kinetic", _kinetic_calcium_cell("integration"))):
        cell.init_state()
        hand_on = getattr(cell, "_integrate_dependent_child_channels", None)
        if hand_on is None:
            report[f"calls/{name}"] = None
            continue
        with mock.patch.object(cell, "_integrate_dependent_child_channels", wraps=hand_on) as spy:
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.005 * u.ms):
                cell.update()
        report[f"calls/{name}"] = spy.call_count
    return report


@functools.cache
def _kinetic_integration_numbers() -> dict:
    # In a subprocess: in one process, a kinetic-ion Cell stepped under
    # "integration" at one precision makes the next one fail at the other
    # (JaxRuntimeError at init_state; on main too), which would break the
    # float32 tests that follow.
    code = (
        "import json; from braincell._multi_compartment.cell_test import _kinetic_integration_report; "
        "print(json.dumps(_kinetic_integration_report()))"
    )
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class IntegrationOrderAdvancesKineticIonChannelsTest(unittest.TestCase):
    """CELL-32: under ``"integration"``, the dependent channels of a kinetic ion advance."""

    def test_channels_under_a_kinetic_ion_advance_as_in_family_order(self) -> None:
        # Acceptance 1. On main, staggered + "integration" left CaHVA's s and
        # u at their initial 0.01005754343517934 and 0.7574863259922417, and
        # V stayed 1.31e-4 mV from "family" at dt 0.005 and 0.001 ms.
        report = _kinetic_integration_numbers()
        start = report["initial"]
        self.assertEqual(set(start), {"s", "u"})
        differences = []
        for dt_ms in (0.005, 0.001):
            integration = report[f"integration/staggered/{dt_ms}"]
            family = report[f"family/staggered/{dt_ms}"]
            for name in ("s", "u"):
                self.assertNotEqual(integration[name], start[name])
                self.assertAlmostEqual(integration[name], family[name], delta=1e-12)
            differences.append(abs(integration["V"] - family["V"]))
        self.assertLessEqual(differences[0], 1e-3)
        # The CaHVA gates read V only, so the two orders agree here.
        self.assertLessEqual(max(differences), 1e-12)

    def test_generic_solver_converges_to_staggered_integration_order(self) -> None:
        # Acceptance 1. On main the gap shrank 1.58x (2.42e-4 -> 1.53e-4 mV).
        report = _kinetic_integration_numbers()
        gaps = [
            abs(report[f"integration/ralston2/{dt_ms}"]["V"] - report[f"integration/staggered/{dt_ms}"]["V"])
            for dt_ms in (0.005, 0.001)
        ]
        self.assertLessEqual(gaps[0], 1e-3)
        self.assertGreaterEqual(gaps[0] / gaps[1], 3.0)

    def test_dependent_ion_keeps_its_one_integration_step(self) -> None:
        # Acceptance 2: an ion integrated with its channels (SodiumFixed) is
        # untouched; only a self-integrating ion hands its channels on.
        report = _kinetic_integration_numbers()
        self.assertEqual(report["calls/sodium"], 0)
        self.assertEqual(report["calls/kinetic"], 1)


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


def _two_cv_soma(solver):
    """CELL-31a A1: the CELL-19 soma in 2 CVs, a 0.05 nA clamp at x = 0.25 (linear, stiff axial coupling)."""
    cell = Cell(_soma_tree(), cv_policy=CVPerBranch(2), solver=solver, V_init=-65.0 * u.mV)
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
    cell.place(at("soma", 0.25), CurrentClamp(delay=0.0 * u.ms, durations=100.0 * u.ms, amplitudes=0.05 * u.nA))
    return cell


def _dendrite_600um(solver, n_cv, *, hh):
    """A 600 um x 0.5 um dendrite with a midpoint clamp: CELL-31a A2 (``hh``) or CELL-2a M1."""
    branch = Branch.from_lengths(lengths=[600.0] * u.um, radii=[0.5, 0.5] * u.um, type="dendrite")
    cell = Cell(Morphology.from_root(branch, name="d"), cv_policy=CVPerBranch(n_cv), solver=solver, V_init=-65.0 * u.mV)
    cable = mech.CableProperty(
        resting_potential=-65.0 * u.mV,
        membrane_capacitance=1.0 * u.uF / u.cm**2,
        axial_resistivity=100.0 * u.ohm * u.cm,
    )
    if not hh:
        cell.paint(AllRegion(), cable, mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
        amplitude = 0.05 * u.nA
    else:
        cell.paint(
            AllRegion(),
            cable,
            mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.4 * u.mV),
            mech.Ion("SodiumFixed", name="na", E=50.0 * u.mV),
            mech.Ion("PotassiumFixed", name="k", E=-77.0 * u.mV),
        )
        cell.paint(
            AllRegion(),
            mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2, ion_name="na"),
            mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2, ion_name="k"),
        )
        amplitude = 0.02 * u.nA
    cell.place(at("d", 0.5), CurrentClamp(delay=0.0 * u.ms, durations=100.0 * u.ms, amplitudes=amplitude))
    return cell


def _final_V(cell, dt_ms, duration_ms):
    cell.init_state()
    cell.run(dt=dt_ms * u.ms, duration=duration_ms * u.ms)
    return np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).ravel()


#: The merging the implicit block solvers used before CELL-31a, whatever the host declared.
_PER_CV_MERGING = mock.patch.object(Cell, "diffeq_state_merging", "stack")


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

    def test_implicit_block_solvers_couple_the_cvs_of_a_cell(self):
        # CELL-31a A1. backward_euler and crank_nicolson merged a cable Cell per
        # CV ('stack'): the Jacobian kept only column sums of the axial terms
        # and V reached 1e92 mV or NaN without an error. The model is linear,
        # so exp_euler is exact; the bounds are about 1.5 times the measured
        # errors (backward_euler 4.49e-3, 9.00e-4; crank_nicolson 2.12e-4, 7.5e-8).
        bounds = {
            0.025: {"backward_euler": 6.7e-3, "crank_nicolson": 3.2e-4},
            0.005: {"backward_euler": 1.35e-3, "crank_nicolson": 1.2e-7},
        }
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for dt, per_solver in bounds.items():
                exact = _final_V(_two_cv_soma("exp_euler"), dt, 1.0)
                # rk4 at dt = 1e-5 ms.
                np.testing.assert_allclose(exact, [-61.213203287434446, -61.213999054192236], rtol=0, atol=1e-9)
                for name, bound in per_solver.items():
                    with self.subTest(solver=name, dt=dt):
                        V = _final_V(_two_cv_soma(name), dt, 1.0)
                        self.assertTrue(np.all(np.isfinite(V)), V)
                        self.assertLessEqual(float(np.max(np.abs(V - exact))), bound)

    def test_implicit_block_solvers_stay_finite_on_a_30_cv_hh_dendrite(self):
        # CELL-31a A2: NaN at dt = 0.025 ms when merged per CV. Reference: the
        # Richardson limit of staggered (dt 0.0005, 0.00025 ms); measured
        # backward_euler 1.23e-2 mV, crank_nicolson 2.06e-5 mV.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            coarse = _final_V(_dendrite_600um("staggered", 30, hh=True), 0.0005, 5.0)
            fine = _final_V(_dendrite_600um("staggered", 30, hh=True), 0.00025, 5.0)
            limit = 2.0 * fine - coarse
            for name, bound in (("backward_euler", 1.85e-2), ("crank_nicolson", 3.1e-5)):
                with self.subTest(solver=name):
                    V = _final_V(_dendrite_600um(name, 30, hh=True), 0.025, 5.0)
                    self.assertTrue(np.all(np.isfinite(V)), V)
                    self.assertLessEqual(float(np.max(np.abs(V - limit))), bound)

    def test_per_cell_blocks_change_a_non_stiff_cable_cell_within_truncation_error(self):
        # CELL-2a M1 (3 CVs). Measured change from per-CV blocks: backward_euler
        # 6.5e-5 mV at dt = 0.025 ms against its own error of 1.0e-2 mV;
        # crank_nicolson 1.6e-12 mV (the Newton fixed point does not depend on
        # the Jacobian). exp_euler is exact on this linear model.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            error = {}
            for dt in (0.025, 0.005):
                exact = _final_V(_dendrite_600um("exp_euler", 3, hh=False), dt, 5.0)
                for name, bound in (("backward_euler", 1e-4), ("crank_nicolson", 1e-10)):
                    with self.subTest(solver=name, dt=dt):
                        V = _final_V(_dendrite_600um(name, 3, hh=False), dt, 5.0)
                        with _PER_CV_MERGING:
                            before = _final_V(_dendrite_600um(name, 3, hh=False), dt, 5.0)
                        self.assertLessEqual(float(np.max(np.abs(V - before))), bound)
                        error[name, dt] = float(np.max(np.abs(V - exact)))
            ratio = error["backward_euler", 0.025] / error["backward_euler", 0.005]
            self.assertTrue(4.0 <= ratio <= 6.0, error)

    def test_one_cv_cells_are_unchanged_by_per_cell_blocks(self):
        # A one-CV cell is one block either way.
        names = ("exp_euler", "backward_euler", "crank_nicolson")
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for mechanism in ("none", "markov", "kinetic_child"):
                for name in names:
                    with self.subTest(mechanism=mechanism, solver=name):
                        V, states = _run_to_1ms(_mechanism_cell(name, mechanism), 0.025)
                        with _PER_CV_MERGING:
                            ref_V, ref_states = _run_to_1ms(_mechanism_cell(name, mechanism), 0.025)
                        np.testing.assert_allclose(V, ref_V, rtol=1e-12, atol=0)
                        self.assertEqual(set(states), set(ref_states))
                        for key in states:
                            np.testing.assert_allclose(states[key], ref_states[key], rtol=1e-12, atol=1e-300)

    def test_block_solvers_advance_a_cable_cell_with_point_synapses(self):
        # CELL-19 rejected these at init_state; CELL-31b puts each synapse row
        # in its cell's block (convergence: CellBlockSolverPointSynapseTest).
        for name in ("exp_euler", "backward_euler", "crank_nicolson"):
            with self.subTest(solver=name):
                cell = _mechanism_cell(name, "none")
                cell.place(at("soma", 0.5), mech.Synapse("Exp2Syn", name="syn"))
                cell.init_state()
                self.assertTrue(cell._initialized)
                cell.run(dt=0.025 * u.ms, duration=0.05 * u.ms)
                self.assertTrue(np.all(np.isfinite(np.asarray(cell.V.value.to_decimal(u.mV)))))

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
            # CELL-31b: exp_euler is second order with the synapse in the
            # block; measured 1.64e-5 mV at dt = 0.001 ms.
            self.assertLessEqual(abs(final_V("exp_euler", 0.001) - limit), 2.5e-5)


_EXP2SYN = dict(tau1=0.5 * u.ms, tau2=5.0 * u.ms, e=0.0 * u.mV)
_BLOCK_SOLVER_NAMES = ("exp_euler", "crank_nicolson", "backward_euler")


def _synapse_cell(solver, model, **kwargs):
    """The CELL-31b models.

    - ``"S"``: the CELL-19 soma (``IL`` 0.1 mS/cm^2 at -70 mV) with an Exp2Syn
      at its midpoint.
    - ``"P"``: S with ``pop_size=2`` and an ExpSyn at the x = 0 end (an
      algebraic boundary point) of cell 0 only: two synapse rows in cell 0, one
      in cell 1.
    - ``"G"``: the CELL-19 GABA model, ``GabaA_JE2011`` writing ``Cldif_JE2011``
      (the ion-current snapshot path).
    """
    if model == "G":
        branch = Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um)
        cell = Cell(
            Morphology.from_root(branch), cv_policy=CVPerBranch(1), V_init=-70.0 * u.mV, solver=solver, **kwargs
        )
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.0274578 * u.mS / u.cm**2, E=-70.0 * u.mV))
        cell.paint(AllRegion(), mech.Ion("Cldif_JE2011", name="cl"))
        cell.place(RootLocation(0.5), mech.Synapse("GabaA_JE2011", name="s"))
        return cell
    cell = Cell(_soma_tree(), pop_size=2 if model == "P" else 1, cv_policy=CVPerBranch(), solver=solver, **kwargs)
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.place(at("soma", 0.5), mech.Synapse("Exp2Syn", name="a", **_EXP2SYN))
    if model == "P":
        cell[0:1].place(at("soma", 0.0), mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
    return cell


def _synapse_network(solver, model):
    """A model of :func:`_synapse_cell` driven by NetStims: events at 0.1, 0.4, 0.7 ms (and 0.25, 0.45, 0.65 ms)."""
    cell = _synapse_cell(solver, model)
    net = braincell.Network(name=f"cell31b_{model.lower()}")
    stim = net.add_population("stim", NetStim(size=1, start=0.1 * u.ms, interval=0.3 * u.ms, number=3))
    post = net.add_population("post", cell)
    if model == "G":
        net.connect("s", source=stim, synapse=post.synapses["s"], weight=0.0005 * u.uS)
        return net, cell
    net.connect("a", source=stim, synapse=post.synapses["a"], weight=0.01 * u.uS)
    if model == "P":
        stim2 = net.add_population("stim2", NetStim(size=1, start=0.25 * u.ms, interval=0.2 * u.ms, number=3))
        net.connect("b", source=stim2, synapse=post.synapses["b"], weight=0.02 * u.uS)
    return net, cell


@functools.lru_cache(maxsize=None)
def _synapse_run(model, solver, dt_ms):
    """``(V per cell, other DiffEqStates)`` at 1 ms; call under float64."""
    net, cell = _synapse_network(solver, model)
    net.init_state()
    net.run(dt=dt_ms * u.ms, duration=1.0 * u.ms)
    return _diffeq_values(cell)


def _gaba_block_solver_report() -> dict:
    """CELL-31b on G; run in a float64 subprocess by :func:`_gaba_block_solver_numbers`.

    ``{"<solver>/<dt>": (max |V - reference| in mV, max |Ci - reference| in mM)}``
    against rk4 at dt = 0.00025 ms.
    """
    with brainstate.environ.context(precision=64):
        ref_V, ref_states = _synapse_run("G", "rk4", 0.00025)
        report = {}
        for name in _BLOCK_SOLVER_NAMES:
            for dt_ms in (0.005, 0.001):
                V, states = _synapse_run("G", name, dt_ms)
                ci = "ion_channels.cl.Ci"
                report[f"{name}/{dt_ms}"] = (
                    float(np.max(np.abs(V - ref_V))),
                    float(np.max(np.abs(states[ci] - ref_states[ci]))),
                )
    return report


@functools.cache
def _gaba_block_solver_numbers() -> dict:
    # In a subprocess, as _kinetic_integration_numbers: in one process a
    # Cldif_JE2011 Cell run at one precision makes the next one fail at the
    # other (JaxRuntimeError at init_state; on main too).
    code = (
        "import json; from braincell._multi_compartment.cell_test import _gaba_block_solver_report; "
        "print(json.dumps(_gaba_block_solver_report()))"
    )
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class CellBlockSolverPointSynapseTest(unittest.TestCase):
    """CELL-31b: the block solvers advance point synapses, each row in a slot of its cell's block."""

    #: Per solver: the bound on |V - reference| at dt 0.005 and 0.001 ms (mV), about
    #: 1.5 times the measured worst cell of S and P, and the window of their ratio.
    BOUNDS = {
        "exp_euler": (6e-4, 2.5e-5, (20.0, 30.0)),
        "crank_nicolson": (6e-4, 2.4e-5, (20.0, 30.0)),
        "backward_euler": (1.6e-1, 3.2e-2, (4.0, 6.0)),
    }

    def test_block_solvers_keep_their_order_with_point_synapses(self):
        # Measured (S; P cell 0) at dt 0.005 -> 0.001 ms against rk4 at 0.001 ms:
        # exp_euler 4.14e-4 -> 1.65e-5; 4.36e-5 -> 1.73e-6. crank_nicolson
        # 3.63e-4 -> 1.45e-5; 3.90e-4 -> 1.56e-5. backward_euler 1.84e-2 ->
        # 3.86e-3; 1.06e-1 -> 2.12e-2. The window rules out a first-order
        # splitting of the synapses (exp_euler's ratio would be 5).
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for model in ("S", "P"):
                reference = _synapse_run(model, "rk4", 0.001)[0]
                for name, (coarse, fine, (low, high)) in self.BOUNDS.items():
                    with self.subTest(model=model, solver=name):
                        error = {dt: np.abs(_synapse_run(model, name, dt)[0] - reference) for dt in (0.005, 0.001)}
                        self.assertLessEqual(float(error[0.005].max()), coarse)
                        self.assertLessEqual(float(error[0.001].max()), fine)
                        ratio = error[0.005] / error[0.001]
                        self.assertTrue(np.all((low <= ratio) & (ratio <= high)), error)

    def test_gaba_writing_a_chloride_pool_converges(self):
        # The snapshot path: Cl- current read after the block from the new V and
        # synapse states. Reference: rk4 at dt 0.00025 ms (rk4 itself is only
        # about 1.3rd order here: Cldif_JE2011 advances after the main step).
        # Measured V: exp_euler 2.62e-5 -> 1.15e-6, crank_nicolson 2.06e-5 ->
        # 7.25e-7, backward_euler 6.42e-4 -> 1.39e-4 mV; Ci at most 2.26e-5 ->
        # 3.58e-6 mM.
        bounds = {
            "exp_euler": (4e-5, 1.8e-6),
            "crank_nicolson": (3.1e-5, 1.1e-6),
            "backward_euler": (9.6e-4, 2.1e-4),
        }
        report = _gaba_block_solver_numbers()
        for name, (coarse, fine) in bounds.items():
            with self.subTest(solver=name):
                for dt, bound, ci_bound in ((0.005, coarse, 3.4e-5), (0.001, fine, 5.4e-6)):
                    gap_V, gap_ci = report[f"{name}/{dt}"]
                    self.assertLessEqual(gap_V, bound)
                    self.assertLessEqual(gap_ci, ci_bound)

    def test_an_unused_slot_leaves_its_cell_as_if_alone(self):
        # P's cell 1 has one synapse row and one zero slot; it is S.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in _BLOCK_SOLVER_NAMES:
                with self.subTest(solver=name):
                    alone = _synapse_run("S", name, 0.005)[0]
                    np.testing.assert_allclose(_synapse_run("P", name, 0.005)[0][1:], alone, rtol=1e-12, atol=0)

    def test_segments_and_reset_state_repeat_one_run(self):
        # Ten 0.1 ms segments (the events fall on segment ends), then
        # reset_state and one 1 ms run: both equal one 1 ms run.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for model in ("S", "P"):
                for name in _BLOCK_SOLVER_NAMES:
                    with self.subTest(model=model, solver=name):
                        once = _synapse_run(model, name, 0.005)[0]
                        net, cell = _synapse_network(name, model)
                        net.init_state()
                        for _ in range(10):
                            net.run(dt=0.005 * u.ms, duration=0.1 * u.ms)
                        np.testing.assert_allclose(_diffeq_values(cell)[0], once, rtol=1e-12, atol=0)
                        net.reset_state()
                        net.run(dt=0.005 * u.ms, duration=1.0 * u.ms)
                        np.testing.assert_allclose(_diffeq_values(cell)[0], once, rtol=1e-12, atol=0)

    def test_batch_members_match_the_unbatched_cell(self):
        # A Network has no batch axis (CELL-15): drive the Cell alone through
        # a written conductance, B = 0.01 uS and A = 0.
        def final_V(name, batch_size):
            cell = _synapse_cell(name, "S")
            cell.init_state(batch_size=batch_size)
            for _, node in cell._runtime.iter_synapse_layouts():
                node.A.value = u.math.zeros_like(node.A.value)
                node.B.value = u.math.zeros_like(node.B.value) + 0.01 * u.uS
            cell.run(dt=0.005 * u.ms, duration=1.0 * u.ms)
            return np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).reshape(batch_size or 1, -1)

        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for name in _BLOCK_SOLVER_NAMES:
                with self.subTest(solver=name):
                    alone = final_V(name, None)
                    batched = final_V(name, 2)
                    self.assertGreater(float(alone[0, 0]), -69.0)  # the conductance moved V
                    for member in batched:
                        np.testing.assert_allclose(member, alone[0], rtol=1e-12, atol=0)

    def test_cells_without_synapses_keep_the_merged_layout(self):
        # No synapse state: the hook returns None and the step merges as before
        # (BlockLayoutTest checks the merged layout against the former merging).
        returned = []
        declare = Cell._diffeq_block_layout

        def spy(self, diffeq_states, merging):
            layout = declare(self, diffeq_states, merging)
            returned.append(layout)
            return layout

        with mock.patch.object(Cell, "_diffeq_block_layout", spy):
            for mechanism in ("none", "markov", "kinetic_child"):
                for name in _BLOCK_SOLVER_NAMES:
                    _run_to_1ms(_mechanism_cell(name, mechanism), 0.025)
            for name in _BLOCK_SOLVER_NAMES:
                _final_V(_dendrite_600um(name, 30, hh=True), 0.025, 0.05)
        self.assertGreaterEqual(len(returned), 12)
        self.assertTrue(all(layout is None for layout in returned))
        cell = _synapse_cell("exp_euler", "P")
        cell.init_state()
        cell.run(dt=0.025 * u.ms, duration=0.025 * u.ms)
        diffeq = split_diffeq_states(cell)[1]
        layout = cell._diffeq_block_layout(diffeq, "concat")
        self.assertEqual(layout.pop_shape, (2,))
        # Exp2Syn A, B and ExpSyn g: rows over the population.
        self.assertEqual(len(layout.slots), 3)

    def test_a_disabled_ion_current_cache_still_needs_the_snapshot(self):
        # The block solvers' rejection used to come first; the snapshot check
        # now names the synapse.
        for name in _BLOCK_SOLVER_NAMES:
            with self.subTest(solver=name):
                cell = _synapse_cell(name, "G", cache_ion_total_current=False)
                with self.assertRaisesRegex(ValueError, "GabaA_JE2011.*cache_ion_total_current=True"):
                    cell.init_state()

    def test_slots_follow_the_placement_after_reset(self):
        # The slot table belongs to the runtime's layouts, so reset() and a new
        # placement rebuild it: cell 1 gains an ExpSyn, which lands in its own
        # slot (k stays 1), and with equal conductances both cells agree.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            cell = _synapse_cell("exp_euler", "P")
            cell.init_state()
            cell.run(dt=0.005 * u.ms, duration=0.1 * u.ms)
            cell.reset()
            cell[1:2].place(at("soma", 0.0), mech.Synapse("ExpSyn", name="b", tau=2.0 * u.ms, e=0.0 * u.mV))
            cell.init_state()
            slots = {
                type(node).__name__: (layout.population_slots, node)
                for layout, node in cell._runtime.iter_synapse_layouts()
            }
            (owner, slot, width), expsyn = slots["ExpSyn"]
            np.testing.assert_array_equal(owner, [0, 1])
            np.testing.assert_array_equal(slot, [0, 0])
            self.assertEqual(width, 1)
            expsyn.g.value = u.math.zeros_like(expsyn.g.value) + 0.005 * u.uS
            cell.run(dt=0.005 * u.ms, duration=1.0 * u.ms)
            V = np.asarray(cell.V.value.to_decimal(u.mV), dtype=np.float64).ravel()
            self.assertGreater(V[0], -69.0)
            np.testing.assert_allclose(V[1], V[0], rtol=1e-12, atol=0)

    def test_a_lumped_cell_advances_point_synapses_too(self):
        # CELL-31c lifted the last rejection (convergence: lumped_test).
        for name in _BLOCK_SOLVER_NAMES:
            with self.subTest(solver=name):
                cell = Cell.point(1, length=20.0 * u.um, radius=10.0 * u.um, solver=name)
                cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
                cell.place(at("soma", 0.5), mech.Synapse("Exp2Syn", name="a", **_EXP2SYN))
                cell.init_state()
                cell.run(dt=0.025 * u.ms, duration=0.05 * u.ms)
                self.assertTrue(np.all(np.isfinite(np.asarray(cell.V.value.to_decimal(u.mV)))))


class DiscretizationTracksAMutatedMorphologyTest(unittest.TestCase):
    """A declaring ``Cell`` shares its ``Morphology``, so it must notice edits to it.

    Before ``init_state`` ``Cell.morpho`` is the declaration tree itself, not
    a copy, and ``Morphology`` is mutable -- it maintains a revision counter
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


def _snapshot_tree() -> Morphology:
    """Soma 20 um (r 5 um) and a 100 um dendrite (r 1 um): two CVs under CVPerBranch."""
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")
    return tree


def _extra_dendrite() -> Branch:
    return Branch.from_lengths(lengths=[30.0] * u.um, radii=[1.0, 1.0] * u.um, type="basal_dendrite")


def _run_briefly(cell: Cell):
    return cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)


def _record_clamped_voltage(cell: Cell) -> Cell:
    """Drive the soma so the recorded voltage depends on the whole tree's geometry."""
    cell.place(at("soma", 0.5), CurrentClamp(delay=0.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.5 * u.nA))
    cell.record("v", braincell.observe.state("V"))
    return cell


def _recorded_mv(result) -> np.ndarray:
    return np.asarray(result.samples["v"].values.to_decimal(u.mV))


def _snapshot_warnings(caught) -> list:
    return [w for w in caught if issubclass(w.category, RuntimeWarning) and "snapshot" in str(w.message)]


def _run_counting_snapshot_warnings(cell: Cell):
    """Run ``cell`` briefly; return the result and its snapshot warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_briefly(cell)
    return result, _snapshot_warnings(caught)


class InitSnapshotTest(unittest.TestCase):
    """MORPH-14: an initialized Cell runs on a frozen copy of its morphology.

    Editing the tree a Cell was declared with after ``init_state()`` used to
    make every grid read, ``run`` and ``reset()`` itself raise, so the Cell
    could not be recovered. The Cell now copies the tree at initialization:
    the declaration stays editable, the edit applies at the next
    ``reset()`` and ``init_state()``, and ``run`` says so once per edit.
    """

    def test_init_takes_a_frozen_copy_without_rediscretizing(self):
        tree = _snapshot_tree()
        cell = Cell(tree)
        generation = cell._view_generation
        with mock.patch.object(cell_module, "build_discretization", wraps=cell_module.build_discretization) as build:
            cell.init_state()
        self.assertEqual(build.call_count, 0)
        self.assertEqual(cell._view_generation - generation, 1)
        self.assertIsNot(cell.morpho, tree)
        self.assertEqual(cell.morpho, tree)
        self.assertIs(cell.morphology, cell.morpho)
        self.assertIs(cell[0].morpho, cell.morpho)
        self.assertEqual(cell.n_cv, 2)

    def test_editing_the_declaration_after_init_does_not_change_the_run(self):
        edited_tree = _snapshot_tree()
        edited = _record_clamped_voltage(Cell(edited_tree))
        unedited = _record_clamped_voltage(Cell(_snapshot_tree()))
        for cell in (edited, unedited):
            cell.init_state()

        edited_tree.soma.extra = _extra_dendrite()

        got, warned = _run_counting_snapshot_warnings(edited)
        expected, _ = _run_counting_snapshot_warnings(unedited)
        np.testing.assert_array_equal(_recorded_mv(got), _recorded_mv(expected))
        self.assertGreater(np.ptp(_recorded_mv(got)), 0.0)
        self.assertEqual(len(warned), 1)
        self.assertRegex(str(warned[0].message), r"snapshot.*reset\(\).*init_state\(\)")
        self.assertEqual(edited.n_cv, 2)

        again, warned = _run_counting_snapshot_warnings(edited)
        expected, _ = _run_counting_snapshot_warnings(unedited)
        np.testing.assert_array_equal(_recorded_mv(again), _recorded_mv(expected))
        self.assertEqual(warned, [])

    def test_each_new_edit_warns_once(self):
        tree = _snapshot_tree()
        cell = Cell(tree)
        cell.init_state()
        self.assertEqual(_run_counting_snapshot_warnings(cell)[1], [])
        tree.soma.extra = _extra_dendrite()
        self.assertEqual(len(_run_counting_snapshot_warnings(cell)[1]), 1)
        self.assertEqual(_run_counting_snapshot_warnings(cell)[1], [])
        tree.extra.tip = _extra_dendrite()
        self.assertEqual(len(_run_counting_snapshot_warnings(cell)[1]), 1)

    def test_reset_after_an_edit_applies_it_at_the_next_init(self):
        tree = _snapshot_tree()
        cell = Cell(tree)
        cell.init_state()
        tree.soma.extra = _extra_dendrite()

        cell.reset()

        self.assertIs(cell.morpho, tree)
        self.assertEqual(cell.n_cv, 3)
        cell.init_state()
        self.assertEqual(cell.n_cv, 3)
        self.assertIsNot(cell.morpho, tree)
        self.assertEqual(cell.morpho, tree)
        self.assertEqual(_run_counting_snapshot_warnings(cell)[1], [])

    def test_editing_the_snapshot_raises_before_writing(self):
        cell = Cell(_snapshot_tree())
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, r"read-only snapshot.*reset\(\)"):
            cell.morpho.soma.more = _extra_dendrite()
        self.assertEqual(cell.morpho.n_branches, 2)
        self.assertEqual(_run_counting_snapshot_warnings(cell)[1], [])
        cell.reset()
        self.assertEqual(cell.n_cv, 2)

    def test_lumped_cell_runs_on_the_snapshot(self):
        tree = Morphology.from_root(_snapshot_tree().soma.branch, name="soma")
        cell = Cell(tree, cv_policy=braincell.Lumped())
        cell.init_state()
        self.assertIsNot(cell.morpho, tree)
        tree.soma.extra = _extra_dendrite()

        self.assertEqual(len(_run_counting_snapshot_warnings(cell)[1]), 1)
        cell.reset()

        self.assertIs(cell.morpho, tree)
        # The edit applies: the tree now has two branches, which Lumped() rejects.
        with self.assertRaisesRegex(ValueError, "Lumped"):
            cell.init_state()

    def test_reduction_cell_runs_on_the_snapshot(self):
        tree = _snapshot_tree()
        cell = Cell(tree)
        cell.add_reduction("acc", braincell.reduction.EventAccumulatorReduction(threshold=2.0))
        cell.use_model("acc")
        cell.init_state()
        self.assertIsNot(cell.morpho, tree)
        tree.soma.extra = _extra_dendrite()

        self.assertEqual(len(_run_counting_snapshot_warnings(cell)[1]), 1)
        cell.reset()

        self.assertIs(cell.morpho, tree)
        cell.init_state()
        self.assertEqual(cell.n_cv, 3)

    def test_a_failed_init_leaves_the_declaration_in_place(self):
        tree = _snapshot_tree()
        cell = Cell(tree)
        generation = cell._view_generation
        with mock.patch.object(CellRuntimeState, "from_cell", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                cell.init_state()
        self.assertFalse(cell._initialized)
        self.assertIs(cell.morpho, tree)
        self.assertEqual(cell._view_generation, generation)
        cell.init_state()
        self.assertEqual(cell.n_cv, 2)
        self.assertIsNot(cell.morpho, tree)


def _snapshot_probe_cell(morpho: Morphology) -> Cell:
    return _record_clamped_voltage(_edit_probe_cell(morpho))


class InitSnapshotEditTableTest(unittest.TestCase):
    """Cell layer of the edit table after ``init_state``.

    Every edit of the declaration tree leaves the running Cell as it was;
    after ``reset()`` and ``init_state()`` the Cell matches one declared from
    scratch on the cold rebuild of the edited tree.
    """

    def test_every_edit_case_waits_for_reset(self):
        reference = _snapshot_probe_cell(make_tree_t())
        expected = _recorded_mv(_run_briefly(reference))
        self.assertGreater(np.ptp(expected), 0.0)
        for case in EDIT_CASES:
            with self.subTest(case.name):
                morpho = make_tree_t()
                cell = _snapshot_probe_cell(morpho)
                cell.init_state()
                before = _declaration_fingerprint(cell)
                case.edit(morpho)
                result, warned = _run_counting_snapshot_warnings(cell)
                np.testing.assert_array_equal(_recorded_mv(result), expected)
                self.assertEqual(len(warned), 1)
                self.assertEqual(_declaration_fingerprint(cell), before)
                cell.reset()
                cell.init_state()
                fresh = _snapshot_probe_cell(cold_rebuild(morpho))
                self.assertEqual(_declaration_fingerprint(cell), _declaration_fingerprint(fresh))


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


class LumpedCellAlignPreTest(unittest.TestCase):
    """brainpy's ``align_pre`` with a descriptor builds the module for the spike a Cell returns.

    On main a lumped Cell built it with ``varshape`` (``pop_size + (1,)``)
    while ``V`` and the spike have shape ``pop_size``, so the first step of
    an ``align_pre`` chain into a projection failed on the shapes.
    """

    DT = 0.025 * u.ms
    DRIVE = np.asarray([0.2, 0.4, 0.6, 0.8]) * u.nA

    @staticmethod
    def _leak():
        return mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV)

    def _pair(self):
        pre = Cell.point(4, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
        pre.paint(
            AllRegion(),
            mech.Ion("SodiumFixed", E=50.0 * u.mV),
            mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            mech.Channel("Na_HH1952", g_max=120.0 * u.mS / u.cm**2),
            mech.Channel("K_HH1952", g_max=36.0 * u.mS / u.cm**2),
            self._leak(),
        )
        post = Cell.point(3, length=20.0 * u.um, radius=10.0 * u.um, V_init=-65.0 * u.mV)
        post.paint(AllRegion(), self._leak())
        return pre, post

    def _run(self, *size):
        """Post voltages and pre spikes of an STD -> AMPA ``CurrentProj`` and an ``AlignPostProj``."""
        pre, post = self._pair()
        weight = np.linspace(1.0, 2.0, 12).reshape(4, 3) * u.nS
        std = pre.align_pre(brainpy.state.STD.desc(*size, tau=700.0 * u.ms, U=0.07))
        ampa = std.align_pre(
            brainpy.state.AMPA.desc(
                *size,
                alpha=0.94 / (u.ms * u.mM),
                beta=0.18 / u.ms,
                g_initializer=braintools.init.Constant(0.0),
            )
        )
        self.assertEqual((std.varshape, ampa.varshape), ((4,), (4,)))
        current = brainpy.state.CurrentProj(
            ampa.prefetch("g"),
            comm=_MatMul(weight),
            out=brainpy.state.COBA(E=0.0 * u.mV),
            post=post,
        )
        aligned = brainpy.state.AlignPostProj(
            comm=_MatMul(weight),
            syn=brainpy.state.Expon.desc(post.pop_size, tau=5.0 * u.ms),
            out=brainpy.state.COBA.desc(E=0.0 * u.mV),
            post=post,
        )
        brainstate.nn.init_all_states([pre, post, current, aligned])

        def step(index):
            with brainstate.environ.context(t=index * self.DT, i=index, dt=self.DT):
                current()
                spike = pre(self.DRIVE)
                aligned(spike)
                post()
                return post.V.value, spike

        return brainstate.transform.for_loop(step, np.arange(400))

    def test_a_descriptor_without_a_size_runs_as_one_sized_by_pop_size(self) -> None:
        voltage, spike = self._run()
        sized_voltage, sized_spike = self._run((4,))
        self.assertEqual((voltage.shape, spike.shape), ((400, 3), (400, 4)))
        self.assertGreater(float(np.sum(np.asarray(spike))), 0.0)
        self.assertGreater(float(u.math.max(voltage).to_decimal(u.mV)), -64.0)  # the synapses reach post
        np.testing.assert_array_equal(np.asarray(spike), np.asarray(sized_spike))
        np.testing.assert_array_equal(voltage.to_decimal(u.mV), sized_voltage.to_decimal(u.mV))

    def test_a_descriptor_is_built_once_and_other_inputs_follow_brainpy(self) -> None:
        cell = Cell.point(4)
        descriptor = brainpy.state.STD.desc(tau=700.0 * u.ms, U=0.07)
        self.assertIs(cell.align_pre(descriptor), cell.align_pre(descriptor))
        self.assertEqual(cell.align_pre(brainpy.state.STD.desc(in_size=(2, 2), tau=700.0 * u.ms)).varshape, (2, 2))
        module = brainpy.state.STD(4, tau=700.0 * u.ms)
        self.assertIs(cell.align_pre(module), module)
        with self.assertRaisesRegex(TypeError, "should be an instance of"):
            cell.align_pre(brainstate.mixin.ParamDescriber(dict))
        with self.assertRaisesRegex(TypeError, "should be an instance of"):
            cell.align_pre(object())

    def test_a_cable_cell_sizes_a_descriptor_by_varshape_the_shape_of_its_spike(self) -> None:
        cable = Cell(Morphology.from_soma(), pop_size=4, cv_policy=CVPerBranch(3))
        std = cable.align_pre(brainpy.state.STD.desc(tau=700.0 * u.ms, U=0.07))
        self.assertEqual(std.varshape, (4, 3))
        cable.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=self.DT):
            self.assertEqual(cable.update().shape, std.varshape)


class _MatMul(brainstate.nn.Module):
    """``spikes @ weight``, the communication of :class:`LumpedCellAlignPreTest`."""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def update(self, spikes):
        return u.math.matmul(spikes, self.weight)


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
