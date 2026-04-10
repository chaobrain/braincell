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

"""Tests for :mod:`braincell.quad._voltage_solver`.

The full ``dhs_voltage_step`` driver requires a point-tree-aware Cell
target with a compiled runtime cache. The end-to-end behaviour is
exercised by the cell-level test suite. Here we focus on:

* the type guard that protects ``dhs_voltage_step`` from being called on
  a target that does not own a point tree, and
* the raw DHS kernels (``comp_triang_raw``, ``comp_backsub_raw``,
  ``_build_backsub_indices``) which are pure functions over jnp/np
  arrays and can be tested in isolation.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.quad import dhs_voltage_step
from braincell.quad._voltage_solver import (
    _build_backsub_indices,
    comp_backsub_raw,
    comp_triang_raw,
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

    def test_kernel_contract_violation_on_quantity_input(self):
        # Quantities are not allowed in the hot path.
        diags = jnp.array([[2.0]]) * u.mV
        solves = jnp.array([[1.0]])
        lowers = jnp.array([0.0])
        uppers = jnp.array([0.0])
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


if __name__ == "__main__":
    unittest.main()
