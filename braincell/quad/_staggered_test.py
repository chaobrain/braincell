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

"""Tests for :mod:`braincell.quad._staggered`.

The full ``staggered_step`` integrator chains a ``dhs_voltage_step``
implicit voltage solve with an ``ind_exp_euler_step`` gating update and
therefore requires a point-tree-aware Cell target. End-to-end behaviour
is exercised by the cell-level test suite. The tests in this file
guard against misuse and verify the registry metadata.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell import (
    Branch,
    CVPerBranch,
    Cell,
    DiffEqModule,
    Morphology,
)
from braincell.quad import  get_registry, staggered_step
from braincell.quad._staggered import (
    _build_backsub_indices,
    _to_jax_quantity,
    comp_backsub_raw,
    comp_triang_raw,
    dhs_voltage_step,
)


class DhsVoltageGuardTest(unittest.TestCase):

    def test_requires_point_tree_attribute(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaisesRegex(TypeError, "point-tree aware"):
            dhs_voltage_step(Plain(), 0. * u.ms, 0.1 * u.ms)

    def test_requires_both_point_tree_and_scheduling(self):
        # An object with only ``point_tree`` is still rejected because the
        # scheduling helper is missing.
        class HalfTarget(brainstate.nn.Module):
            def point_tree(self):  # pragma: no cover - never called
                return None

        with self.assertRaisesRegex(TypeError, "point-tree aware"):
            dhs_voltage_step(HalfTarget(), 0. * u.ms, 0.1 * u.ms)


class CompTriangRawTest(unittest.TestCase):

    def test_no_levels_is_identity(self):
        diags = jnp.array([[2.0, 3.0]])
        solves = jnp.array([[5.0, 7.0]])
        lowers = jnp.array([0.0, 0.0])
        uppers = jnp.array([0.0, 0.0])
        edges = jnp.empty((0, 2), dtype=jnp.int32)
        level_offsets = np.array([0], dtype=np.int32)
        new_diags, new_solves = comp_triang_raw(
            diags, solves, lowers, uppers, edges, level_offsets
        )
        np.testing.assert_array_equal(new_diags, diags)
        np.testing.assert_array_equal(new_solves, solves)

    def test_kernel_contract_violation_on_wrong_rank(self):
        # ``diags`` must be 2D — passing a 1D array trips the contract check.
        with self.assertRaises(AssertionError):
            comp_triang_raw(
                jnp.array([1.0]),
                jnp.array([[1.0]]),
                jnp.array([0.0]),
                jnp.array([0.0]),
                jnp.empty((0, 2), dtype=jnp.int32),
                np.array([0], dtype=np.int32),
            )

    def test_accepts_unitless_quantity_factors_and_quantity_solves(self):
        diags = u.Quantity(jnp.array([[2.0]]), u.UNITLESS)
        solves = jnp.array([[1.0]]) * u.mV
        lowers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        uppers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        edges = np.empty((0, 2), dtype=np.int32)
        level_offsets = np.array([0], dtype=np.int32)
        new_diags, new_solves = comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets)
        self.assertIsInstance(new_diags, u.Quantity)
        self.assertTrue(u.get_unit(new_diags).is_unitless)
        self.assertIsInstance(new_solves, u.Quantity)
        self.assertTrue(u.get_unit(new_solves).has_same_dim(u.mV))

    def test_kernel_contract_violation_on_dimful_diags(self):
        diags = jnp.array([[2.0]]) * u.mV
        solves = jnp.array([[1.0]]) * u.mV
        lowers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        uppers = u.Quantity(jnp.array([0.0]), u.UNITLESS)
        edges = jnp.empty((0, 2), dtype=jnp.int32)
        level_offsets = np.array([0], dtype=np.int32)
        with self.assertRaises(AssertionError):
            comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets)


class CompBacksubRawTest(unittest.TestCase):

    def test_kernel_contract_violation_on_shape_mismatch(self):
        diags = jnp.array([[2.0, 3.0]])
        solves = jnp.array([[1.0, 1.0, 1.0]])  # mismatched second dim
        lowers = jnp.array([0.0, 0.0])
        backsub_indices = jnp.zeros((1, 2), dtype=jnp.int32)
        with self.assertRaises(AssertionError):
            comp_backsub_raw(diags, solves, lowers, backsub_indices)

    def test_accepts_quantity_solves(self):
        diags = u.Quantity(jnp.array([[2.0, 3.0]]), u.UNITLESS)
        solves = jnp.array([[1.0, 2.0]]) * u.mV
        lowers = u.Quantity(jnp.array([0.0, 0.0]), u.UNITLESS)
        backsub_indices = np.zeros((1, 2), dtype=np.int32)
        out = comp_backsub_raw(diags, solves, lowers, backsub_indices)
        self.assertIsInstance(out, u.Quantity)
        self.assertTrue(u.get_unit(out).has_same_dim(u.mV))


class BuildBacksubIndicesTest(unittest.TestCase):

    def test_root_only_tree(self):
        # A single node with sentinel parent: ancestor at any step is
        # the sentinel itself (index 1).
        parent_lookup = np.array([1, 1], dtype=np.int32)
        idx = _build_backsub_indices(parent_lookup, n_nodes=1)
        # The first row jumps one step → parent of node 0 is sentinel (1).
        self.assertEqual(idx.shape[1], 2)
        self.assertEqual(int(idx[0, 0]), 1)

    def test_chain_tree(self):
        # 0 -> 1 -> 2 -> sentinel(3)
        parent_lookup = np.array([1, 2, 3, 3], dtype=np.int32)
        idx = _build_backsub_indices(parent_lookup, n_nodes=3)
        # First row (1 step): each node points to its direct parent.
        np.testing.assert_array_equal(idx[0], [1, 2, 3, 3])
        # Second row (2 steps): node 0 → node 1 → node 2.
        self.assertEqual(int(idx[1, 0]), 2)


class StaggeredStepGuardTest(unittest.TestCase):

    def test_rejects_plain_module(self):
        class Plain(brainstate.nn.Module):
            pass

        # HIGH-03: TypeError (not AssertionError) so ``python -O`` preserves
        # the contract.
        with self.assertRaises(TypeError):
            staggered_step(Plain())

    def test_error_message_mentions_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaises(TypeError) as ctx:
            staggered_step(Plain())
        self.assertIn(DiffEqModule.__name__, str(ctx.exception))


class StaggeredRegistryMetadataTest(unittest.TestCase):

    def test_canonical_name_and_alias(self):
        registry = get_registry()
        self.assertIn("staggered", registry)
        self.assertIn("stagger", registry)
        # ``stagger`` is an alias of the canonical ``staggered`` entry.
        self.assertIs(registry["stagger"], registry["staggered"])
        self.assertIs(registry["staggered"], staggered_step)

    def test_category_and_description(self):
        entry = get_registry().entry("staggered")
        self.assertEqual(entry.category, "staggered")
        self.assertEqual(entry.aliases, ("stagger",))
        self.assertTrue(entry.description)


class DhsRuntimeCacheTest(unittest.TestCase):

    def _simple_cell(self):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.init_state()
        return cell

    def test_cache_rebuilds_when_precision_changes(self):
        cell = self._simple_cell()

        with brainstate.environ.context(dt=0.1 * u.ms, precision=32):
            dhs_voltage_step(cell, 0.0 * u.ms, 0.1 * u.ms)
            source32 = cell.runtime.dhs_static_source_np
            cache32 = cell.runtime.dhs_static_cache
            self.assertEqual(source32.diag_ms_inv_np.dtype, np.float64)
            self.assertEqual(source32.dynamic_rows_np.dtype, np.int32)
            self.assertEqual(cache32.float_dtype, jnp.dtype(jnp.float32))

            dhs_voltage_step(cell, 0.0 * u.ms, 0.1 * u.ms)
            self.assertIs(source32, cell.runtime.dhs_static_source_np)
            self.assertIs(cache32, cell.runtime.dhs_static_cache)

        with brainstate.environ.context(dt=0.1 * u.ms, precision=64):
            dhs_voltage_step(cell, 0.0 * u.ms, 0.1 * u.ms)
            source64 = cell.runtime.dhs_static_source_np
            cache64 = cell.runtime.dhs_static_cache
            self.assertIs(source32, source64)
            self.assertEqual(cache64.float_dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)

    def test_to_jax_quantity_preserves_existing_dtype(self):
        with brainstate.environ.context(precision=32):
            voltage32 = jnp.asarray([1.0, 2.0]) * u.mV

        with brainstate.environ.context(precision=64):
            preserved = _to_jax_quantity(voltage32, u.mV)
            self.assertEqual(preserved.dtype, jnp.dtype(jnp.float32))


if __name__ == "__main__":
    unittest.main()
