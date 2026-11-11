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

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell import (
    Branch,
    CVPerBranch,
    Cell,
    CurrentClamp,
    DiffEqModule,
    Morphology,
    NetStim,
    connect,
    mech,
    observe,
)
from braincell.filter import AllRegion, RootLocation
from braincell.quad import get_integrator, integrator_info, staggered_step
from braincell.quad._staggered import (
    _build_backsub_indices,
    _build_dhs_ordinary_backsub_order,
    _linear_and_const_term,
    comp_backsub_hines_raw,
    comp_backsub_jvp,
    comp_backsub_raw,
    comp_triang_jvp,
    comp_triang_raw,
    dhs_voltage_step,
    _dhs_custom_jvp_enabled,
)


class StaggeredReadsRuntimeAttrDirectlyTest(unittest.TestCase):
    """MED-07: _get_dhs_static_source / _get_dhs_static_cache must not probe _compiled_runtime."""

    def test_compiled_runtime_fallback_is_not_consulted(self) -> None:
        from braincell.quad._staggered import _get_dhs_static_source

        class _TrapRuntime:
            # If the fallback branch runs, this getattr trap fires.
            dhs_source = None

            def __getattribute__(self, name):
                raise AssertionError(f"_compiled_runtime must not be read (got getattr {name!r})")

        class _Target:
            _compiled_runtime = _TrapRuntime()

        # Missing ``_runtime`` must raise AttributeError outright — not quietly
        # fall back to ``_compiled_runtime``.
        with self.assertRaises(AttributeError):
            _get_dhs_static_source(_Target(), node_tree=None, scheduling=None)


class DhsVoltageGuardTest(unittest.TestCase):
    def test_requires_node_tree_attribute(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaisesRegex(TypeError, "node-tree aware"):
            dhs_voltage_step(Plain(), t=0.0 * u.ms, dt=0.1 * u.ms)

    def test_requires_both_node_tree_and_scheduling(self):
        # An object with only ``node_tree`` is still rejected because the
        # scheduling helper is missing.
        class HalfTarget(brainstate.nn.Module):
            def node_tree(self):  # pragma: no cover - never called
                return None

        with self.assertRaisesRegex(TypeError, "node-tree aware"):
            dhs_voltage_step(HalfTarget(), t=0.0 * u.ms, dt=0.1 * u.ms)


class DhsLinearizationUnitTest(unittest.TestCase):
    def test_linear_unit_uses_derivative_over_voltage_when_grad_is_unitless(self):
        class Target:
            def _voltage_linearizer(self):
                def linearizer(V):
                    linear = jnp.full(V.shape, -0.1)
                    derivative = jnp.full(V.shape, 2.0) * (u.nA / u.uF)
                    return linear, derivative

                return linearizer

        V_n = jnp.array([-65.0, -64.0]) * u.mV
        linear, const = _linear_and_const_term(Target(), V_n)

        expected_linear_unit = (u.nA / u.uF) / u.mV
        np.testing.assert_allclose(np.asarray(linear.to_decimal(expected_linear_unit)), [-0.1, -0.1])
        self.assertEqual(u.get_unit(const), u.get_unit(2.0 * u.nA / u.uF))
        np.testing.assert_allclose(np.asarray((V_n * linear + const).to_decimal(u.nA / u.uF)), [2.0, 2.0])


class StaggeredAutodiffTest(unittest.TestCase):
    @staticmethod
    def _hh_cell():
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            V_init=-65.0 * u.mV,
            solver="staggered",
        )
        cell.paint(
            AllRegion(),
            mech.Channel(
                "Na_HH1952",
                name="na_hh",
                g_max=12.0 * (u.mS / u.cm**2),
            ),
        )
        cell.paint(
            AllRegion(),
            mech.Channel(
                "K_HH1952",
                name="k_hh",
                g_max=3.6 * (u.mS / u.cm**2),
            ),
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

    def test_multistep_voltage_and_conductance_gradients_match_finite_difference(self):
        cell = self._hh_cell()
        sodium = cell.ion_channels["na"].channels["na_hh"]
        original_g_max = sodium.g_max
        conductance_unit = u.mS / u.cm**2
        active_mask = u.get_mantissa(original_g_max) != 0

        def objective(v0_mantissa, g_max_mantissa):
            sodium.g_max = active_mask * g_max_mantissa * conductance_unit
            cell.reset_state()
            cell.V.value = jnp.broadcast_to(v0_mantissa, cell.V.value.shape) * u.mV

            def step(_):
                cell.update()
                return cell.V.value

            trace = brainstate.transform.for_loop(step, jnp.arange(6))
            return jnp.sum(trace[-1].to_decimal(u.mV))

        try:
            with brainstate.environ.context(dt=0.025 * u.ms, precision=64):
                v0 = jnp.asarray(-65.0, dtype=jnp.float64)
                g_max = jnp.asarray(12.0, dtype=jnp.float64)
                grad_v0, grad_g_max = brainstate.transform.grad(
                    objective,
                    argnums=(0, 1),
                )(v0, g_max)

                epsilon = 1e-3
                finite_diff_v0 = (objective(v0 + epsilon, g_max) - objective(v0 - epsilon, g_max)) / (2.0 * epsilon)
                finite_diff_g_max = (objective(v0, g_max + epsilon) - objective(v0, g_max - epsilon)) / (2.0 * epsilon)

            for gradient in (grad_v0, grad_g_max):
                self.assertTrue(bool(jnp.isfinite(gradient)))
                self.assertNotEqual(float(gradient), 0.0)
            np.testing.assert_allclose(grad_v0, finite_diff_v0, rtol=1e-4, atol=1e-8)
            np.testing.assert_allclose(grad_g_max, finite_diff_g_max, rtol=1e-4, atol=1e-8)
        finally:
            sodium.g_max = original_g_max
            cell.reset_state()


class CompTriangRawTest(unittest.TestCase):
    def test_custom_jvp_is_opt_in(self):
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("BRAINCELL_DHS_CUSTOM_JVP", None)
            self.assertFalse(_dhs_custom_jvp_enabled())
        with patch.dict("os.environ", {"BRAINCELL_DHS_CUSTOM_JVP": "1"}):
            self.assertTrue(_dhs_custom_jvp_enabled())

    def test_explicit_jvp_matches_jax(self):
        diags = jnp.asarray([[2.0, 3.0, 4.0, 1.0]])
        solves = jnp.asarray([[4.0, 3.0, 8.0, 0.0]])
        lowers = jnp.asarray([0.0, -1.0, -2.0, 0.0])
        uppers = jnp.asarray([0.0, -0.5, -1.0, 0.0])
        edges = np.asarray([[1, 0], [2, 1]], dtype=np.int32)
        level_offsets = np.asarray([0, 1, 2], dtype=np.int32)
        primal_tangents = (
            jnp.asarray([[0.1, -0.2, 0.3, 0.0]]),
            jnp.asarray([[0.2, 0.1, -0.1, 0.0]]),
            jnp.asarray([0.0, 0.1, -0.2, 0.0]),
            jnp.asarray([0.0, -0.2, 0.1, 0.0]),
        )
        tangent_kernels = (
            primal_tangents[0][None],
            primal_tangents[1][None],
            primal_tangents[2][None],
            primal_tangents[3][None],
        )

        def primal(d, s, l, u):
            return comp_triang_raw(d, s, l, u, edges, level_offsets)

        _, expected = jax.jvp(primal, (diags, solves, lowers, uppers), primal_tangents)
        actual = comp_triang_jvp(
            diags,
            solves,
            lowers,
            uppers,
            edges,
            level_offsets,
            *tangent_kernels,
        )
        np.testing.assert_allclose(actual[2], expected[0][None], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual[3], expected[1][None], rtol=1e-6, atol=1e-7)

    def test_no_levels_is_identity(self):
        diags = jnp.array([[2.0, 3.0]])
        solves = jnp.array([[5.0, 7.0]])
        lowers = jnp.array([0.0, 0.0])
        uppers = jnp.array([0.0, 0.0])
        edges = jnp.empty((0, 2), dtype=jnp.int32)
        level_offsets = np.array([0], dtype=np.int32)
        new_diags, new_solves = comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets)
        np.testing.assert_array_equal(new_diags, diags)
        np.testing.assert_array_equal(new_solves, solves)

    def test_kernel_contract_violation_on_wrong_rank(self):
        # ``diags`` must be 2D — passing a 1D array trips the contract check.
        # HIGH-04: raises ValueError (not AssertionError) under ``python -O``.
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
            comp_triang_raw(diags, solves, lowers, uppers, edges, level_offsets)


class CompBacksubRawTest(unittest.TestCase):
    def test_complete_dhs_jvp_chain_matches_jax(self):
        diags = jnp.asarray([[2.0, 3.0, 4.0, 1.0]])
        solves = jnp.asarray([[4.0, 3.0, 8.0, 0.0]])
        lowers = jnp.asarray([0.0, -1.0, -2.0, 0.0])
        uppers = jnp.asarray([0.0, -0.5, -1.0, 0.0])
        edges = np.asarray([[1, 0], [2, 1]], dtype=np.int32)
        level_offsets = np.asarray([0, 1, 2], dtype=np.int32)
        parent_lookup = np.asarray([3, 0, 1, 3], dtype=np.int32)
        indices = _build_backsub_indices(parent_lookup, n_nodes=3)
        primal_tangents = (
            jnp.asarray([[0.1, -0.2, 0.3, 0.0]]),
            jnp.asarray([[0.2, 0.1, -0.1, 0.0]]),
            jnp.asarray([0.0, 0.1, -0.2, 0.0]),
            jnp.asarray([0.0, -0.2, 0.1, 0.0]),
        )

        def primal(d, s, l, u):
            td, ts = comp_triang_raw(d, s, l, u, edges, level_offsets)
            return comp_backsub_raw(td, ts, l, indices)

        _, expected = jax.jvp(primal, (diags, solves, lowers, uppers), primal_tangents)
        td, ts, dtd, dts = comp_triang_jvp(
            diags,
            solves,
            lowers,
            uppers,
            edges,
            level_offsets,
            primal_tangents[0][None],
            primal_tangents[1][None],
            primal_tangents[2][None],
            primal_tangents[3][None],
        )
        actual, d_actual = comp_backsub_jvp(
            td,
            ts,
            lowers,
            indices,
            dtd,
            dts,
            primal_tangents[2][None],
        )
        primal_expected = comp_backsub_raw(td, ts, lowers, indices)
        np.testing.assert_allclose(actual[0], primal_expected[0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(d_actual[0], expected, rtol=1e-6, atol=1e-7)

        jit_jvp = jax.jit(lambda d, s, l, u, dd, ds, dl, du: jax.jvp(primal, (d, s, l, u), (dd, ds, dl, du)))
        _, jit_expected = jit_jvp(diags, solves, lowers, uppers, *primal_tangents)
        np.testing.assert_allclose(jit_expected, expected, rtol=1e-6, atol=1e-7)

    def test_explicit_jvp_matches_jax(self):
        diags = jnp.asarray([[2.0, 3.0, 4.0, 1.0]])
        solves = jnp.asarray([[4.0, 3.0, 8.0, 0.0]])
        lowers = jnp.asarray([0.0, -1.0, -2.0, 0.0])
        parent_lookup = np.asarray([3, 0, 1, 3], dtype=np.int32)
        indices = _build_backsub_indices(parent_lookup, n_nodes=3)
        d_diags = jnp.asarray([[[0.1, -0.2, 0.3, 0.0]]])
        d_solves = jnp.asarray([[[0.2, 0.1, -0.1, 0.0]]])
        d_lowers = jnp.asarray([[0.0, 0.1, -0.2, 0.0]])

        def primal(d, s, l):
            return comp_backsub_raw(d, s, l, indices)

        _, expected = jax.jvp(primal, (diags, solves, lowers), (d_diags[0], d_solves[0], d_lowers[0]))
        actual, d_actual = comp_backsub_jvp(diags, solves, lowers, indices, d_diags, d_solves, d_lowers)
        np.testing.assert_allclose(actual, primal(diags, solves, lowers), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(d_actual[0], expected, rtol=1e-6, atol=1e-7)

    def test_kernel_contract_violation_on_shape_mismatch(self):
        diags = jnp.array([[2.0, 3.0]])
        solves = jnp.array([[1.0, 1.0, 1.0]])  # mismatched second dim
        lowers = jnp.array([0.0, 0.0])
        backsub_indices = jnp.zeros((1, 2), dtype=jnp.int32)
        with self.assertRaises(ValueError):
            comp_backsub_raw(diags, solves, lowers, backsub_indices)

    def test_accepts_quantity_solves(self):
        diags = u.Quantity(jnp.array([[2.0, 3.0]]), u.UNITLESS)
        solves = jnp.array([[1.0, 2.0]]) * u.mV
        lowers = u.Quantity(jnp.array([0.0, 0.0]), u.UNITLESS)
        backsub_indices = np.zeros((1, 2), dtype=np.int32)
        out = comp_backsub_raw(diags, solves, lowers, backsub_indices)
        self.assertIsInstance(out, u.Quantity)
        self.assertTrue(u.get_unit(out).has_same_dim(u.mV))


class CompBacksubHinesRawTest(unittest.TestCase):
    def test_chain_matches_recursive_doubling(self):
        diags = jnp.asarray([[2.0, 3.0, 4.0, 1.0]])
        solves = jnp.asarray([[4.0, 3.0, 8.0, 0.0]]) * u.mV
        lowers = jnp.asarray([0.0, -1.0, -2.0, 0.0])
        ordinary_edges = np.asarray([[1, 0], [2, 1]], dtype=np.int32)
        ordinary_offsets = np.asarray([0, 1, 2], dtype=np.int32)
        parent_lookup = np.asarray([3, 0, 1, 3], dtype=np.int32)
        recursive_indices = _build_backsub_indices(parent_lookup, n_nodes=3)

        ordinary = comp_backsub_hines_raw(diags, solves, lowers, ordinary_edges, ordinary_offsets)
        recursive = comp_backsub_raw(diags, solves, lowers, recursive_indices)

        self.assertIsInstance(ordinary, u.Quantity)
        np.testing.assert_allclose(ordinary.to_decimal(u.mV), recursive.to_decimal(u.mV), rtol=1e-6, atol=1e-7)

    def test_schedule_is_root_to_leaf_and_covers_each_edge(self):
        soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        morphology = Morphology.from_root(soma, name="soma")
        parent = "soma"
        for index in range(3):
            branch = Branch.from_lengths(lengths=[20.0] * u.um, radii=[2.0, 2.0] * u.um, type="dendrite")
            name = f"dend_{index}"
            morphology.attach(parent=parent, child_branch=branch, child_name=name, parent_x=1.0)
            parent = name
        cell = Cell(morphology, cv_policy=CVPerBranch())
        cell.init_state()
        scheduling = cell.node_scheduling(algorithm="dhs")
        edges, level_size = _build_dhs_ordinary_backsub_order(scheduling)

        self.assertEqual(edges.shape[0], len(cell.node_tree.nodes) - 1)
        self.assertEqual(int(level_size.sum()), edges.shape[0])
        roots = np.flatnonzero(scheduling.parent_rows < 0)
        self.assertEqual(len(roots), 1)
        seen = {int(roots[0])}
        for child, parent_row in edges.tolist():
            self.assertIn(parent_row, seen)
            seen.add(child)


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

    def test_family_schedule_integrates_synapses_then_family_after_voltage(self):
        calls = []
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.init_state()
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._integrate_runtime_synapse_dynamics = lambda point_V: calls.append("synapse_dynamics")
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")

        with patch("braincell.quad._staggered._dhs_voltage_solve", lambda *args, **kwargs: calls.append("voltage")):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                staggered_step(cell)

        self.assertEqual(calls, ["voltage", "synapse_dynamics", "family"])

    def test_integration_schedule_integrates_after_voltage_without_preparing_synapses(self):
        calls = []
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            ion_channel_update_order="integration",
        )
        cell.init_state()
        cell._update_runtime_synapses = lambda point_V: calls.append("synapse")
        cell._prepare_runtime_synapse_inputs = lambda point_V: calls.append("prepare")
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._update_ion_channels_by_integration = lambda point_V: calls.append("integration")

        with patch("braincell.quad._staggered._dhs_voltage_solve", lambda *args, **kwargs: calls.append("voltage")):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                staggered_step(cell)

        self.assertEqual(calls, ["voltage", "integration"])


class StaggeredRegistryMetadataTest(unittest.TestCase):
    def test_canonical_name_and_alias(self):
        # ``stagger`` is an alias of the canonical ``staggered`` entry.
        self.assertIs(get_integrator("stagger"), get_integrator("staggered"))
        self.assertIs(get_integrator("staggered"), staggered_step)
        self.assertEqual(integrator_info("stagger").name, "staggered")

    def test_category_and_description(self):
        entry = integrator_info("staggered")
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
            dhs_voltage_step(cell, t=0.0 * u.ms, dt=0.1 * u.ms)
            source32 = cell.runtime.dhs_source
            cache32 = cell.runtime.dhs_static_cache
            self.assertEqual(source32.diag_ms_inv.dtype, jnp.float32)
            self.assertEqual(source32.dynamic_rows_np.dtype, np.int32)
            self.assertEqual(cache32.float_dtype, jnp.dtype(jnp.float32))

            dhs_voltage_step(cell, t=0.0 * u.ms, dt=0.1 * u.ms)
            self.assertIs(source32, cell.runtime.dhs_source)
            self.assertIs(cache32, cell.runtime.dhs_static_cache)

        with brainstate.environ.context(dt=0.1 * u.ms, precision=64):
            dhs_voltage_step(cell, t=0.0 * u.ms, dt=0.1 * u.ms)
            source64 = cell.runtime.dhs_source
            cache64 = cell.runtime.dhs_static_cache
            self.assertIs(source32, source64)
            self.assertEqual(cache64.float_dtype, jnp.dtype(jnp.float64))
            self.assertIsNot(cache32, cache64)
            self.assertEqual(cell.V.value.dtype, jnp.dtype(jnp.float32))
            self.assertEqual(cell._point_V.value.dtype, jnp.dtype(jnp.float32))

    def test_numeric_operands_preserve_an_existing_dtype(self):
        # ``_build_dhs_numeric_state`` materializes its operands with
        # ``u.math.asarray(value, unit=...)``, which replaced a hand-rolled
        # ``_to_jax_quantity``. The DHS step depends on that call converting
        # units *without* promoting to the ambient precision, so the
        # third-party contract is pinned here rather than left implicit.
        with brainstate.environ.context(precision=32):
            voltage32 = jnp.asarray([1.0, 2.0]) * u.mV

        with brainstate.environ.context(precision=64):
            preserved = u.math.asarray(voltage32, unit=u.mV)
            self.assertEqual(preserved.dtype, jnp.dtype(jnp.float32))


class DhsEndpointClampTest(unittest.TestCase):
    def test_root_endpoint_current_changes_midpoint_voltage(self):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.place(RootLocation(x=0.0), CurrentClamp(durations=1.0 * u.ms, amplitudes=1.0 * u.nA))
        cell.init_state()

        before = float(cell.V.value[0, 0].to_decimal(u.mV))
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            dhs_voltage_step(cell, t=0.0 * u.ms, dt=0.05 * u.ms)
        after = float(cell.V.value[0, 0].to_decimal(u.mV))

        self.assertGreater(after, before)


class DhsEndpointSynapseNeuronReferenceTest(unittest.TestCase):
    @staticmethod
    def _run_expsyn_at(x):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            V_init=-65.0 * u.mV,
            solver="staggered",
        )
        cell.paint(
            AllRegion(),
            mech.CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
            mech.Channel(
                "IL",
                name="leak",
                g_max=0.1 * u.mS / u.cm**2,
                E=-65.0 * u.mV,
            ),
        )
        synapse = mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(RootLocation(x=x), synapse)
        connect(
            f"drive_{x}",
            source=NetStim(start=1.0 * u.ms, number=1),
            synapse=cell.synapses[synapse],
            weight=0.01 * u.uS,
        )
        cell.soma.record("v", observe.state("v"))

        result = cell.run(dt=0.025 * u.ms, duration=5.0 * u.ms)
        trace = np.asarray(result.samples["v"].values.to_decimal(u.mV), dtype=float)[:, 0]
        final = float(np.asarray(cell.V.value.to_decimal(u.mV)).reshape(-1)[0])
        return float(np.max(trace)), final

    def test_endpoint_and_midpoint_match_fixed_neuron_reference(self):
        # Generated once with NEURON 8.2.6, fixed-step secondorder=0:
        # L=diam=20 um, nseg=1, Ra=100 ohm cm, cm=1 uF/cm2,
        # pas.g=0.0001 S/cm2, pas.e=-65 mV, ExpSyn(tau=2 ms, e=0),
        # NetCon event at 1 ms with weight=0.01 uS, dt=0.025 ms.
        neuron_reference = {
            0.0: (-25.36324335207355, -25.809036761905066),
            0.5: (-25.359568290945035, -25.805809720317054),
        }

        with brainstate.environ.context(precision=64):
            endpoint = self._run_expsyn_at(0.0)
            midpoint = self._run_expsyn_at(0.5)

        np.testing.assert_allclose(endpoint, neuron_reference[0.0], rtol=0.0, atol=1e-8)
        np.testing.assert_allclose(midpoint, neuron_reference[0.5], rtol=0.0, atol=1e-8)
        self.assertLess(endpoint[0], midpoint[0])
        self.assertLess(endpoint[1], midpoint[1])


class DhsMidpointClampPopulationTest(unittest.TestCase):
    def _build_clamped_cell(self, *, pop_size=None):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        kwargs = {} if pop_size is None else {"pop_size": pop_size}
        cell = Cell(
            Morphology.from_root(soma, name="soma"),
            cv_policy=CVPerBranch(),
            V_init=-65.0 * u.mV,
            **kwargs,
        )
        cell.place(
            RootLocation(x=0.5),
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=0.1 * u.ms,
                amplitudes=1.0 * u.nA,
            ),
        )
        cell.init_state()
        cell.reset_state()
        return cell

    def test_midpoint_clamp_is_not_double_counted_with_population_axis(self):
        # A Cell always carries a population axis, so the invariant is that
        # widening it changes nothing: a homogeneous population of N members
        # must reproduce the single-member trace N times. A midpoint clamp
        # applied once per population member instead of once per CV would
        # break this.
        single = self._build_clamped_cell()
        population = self._build_clamped_cell(pop_size=(3,))
        self.assertEqual(single.pop_size, (1,))

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            dhs_voltage_step(single, t=0.0 * u.ms, dt=0.025 * u.ms)
            dhs_voltage_step(population, t=0.0 * u.ms, dt=0.025 * u.ms)

        single_v = np.asarray(single.V.value.to_decimal(u.mV), dtype=float)
        population_v = np.asarray(population.V.value.to_decimal(u.mV), dtype=float)
        self.assertEqual(population_v.shape, (3,) + single_v.shape[1:])
        for member in range(3):
            np.testing.assert_allclose(population_v[member], single_v[0], rtol=1e-12, atol=1e-12)


class DhsMultistepClampTest(unittest.TestCase):
    def test_multistep_current_clamp_voltage_step_handles_zero_linearization(self):
        soma = Branch.from_lengths(
            lengths=[20.0] * u.um,
            radii=[10.0, 10.0] * u.um,
            type="soma",
        )
        cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch())
        cell.place(
            RootLocation(x=0.0),
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(0.5 * u.ms, 0.5 * u.ms),
                amplitudes=(1.0 * u.nA, 0.0 * u.nA),
            ),
        )
        cell.init_state()

        before = float(cell.V.value[0, 0].to_decimal(u.mV))
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            dhs_voltage_step(cell, t=0.0 * u.ms, dt=0.05 * u.ms)
        after = float(cell.V.value[0, 0].to_decimal(u.mV))

        self.assertGreater(after, before)


# Run in a child interpreter because JAX fixes its float precision at start-up.
_PASSIVE_SOMA_SCRIPT = """
import json
import brainunit as u
import numpy as np
from braincell import Branch, CVPerBranch, Cell, Morphology, mech
from braincell.filter import AllRegion

out = {}
for dt in (0.025, 0.005):
    soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = Cell(Morphology.from_root(soma, name="soma"), cv_policy=CVPerBranch(), solver="staggered")
    cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
    cell.run(dt=dt * u.ms, duration=1.0 * u.ms)
    V = cell.V.value.to_decimal(u.mV)
    out[str(dt)] = {"dtype": str(V.dtype), "V": float(np.asarray(V).ravel()[0])}
print(json.dumps(out))
"""


def _run_passive_soma(enable_x64: bool) -> dict:
    env = dict(os.environ, JAX_ENABLE_X64="1" if enable_x64 else "0", JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [sys.executable, "-c", _PASSIVE_SOMA_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class StaggeredFloat32AccumulationTest(unittest.TestCase):
    """QUAD-5: float32 rounding must not accumulate with |V| in the DHS solve.

    Before the increment-form solve, the passive soma below drifted about
    1e-2 mV from float64 within 1 ms, at every dt: eliminating onto an
    algebraic boundary row cancels nearly equal terms, and that relative error
    multiplied the full voltage each step.
    """

    @classmethod
    def setUpClass(cls):
        cls.f32 = _run_passive_soma(enable_x64=False)
        cls.f64 = _run_passive_soma(enable_x64=True)

    def test_float32_matches_float64_after_one_millisecond(self):
        for dt in ("0.025", "0.005"):
            with self.subTest(dt=dt):
                self.assertEqual(self.f32[dt]["dtype"], "float32")
                self.assertEqual(self.f64[dt]["dtype"], "float64")
                self.assertLessEqual(abs(self.f32[dt]["V"] - self.f64[dt]["V"]), 1e-4)

    def test_float64_matches_the_exact_implicit_euler_recursion(self):
        # The axial terms cancel on an isopotential soma, so implicit Euler
        # reduces to V <- (V + dt k E) / (1 + dt k) with k = g / c_m = 0.1 / ms.
        for dt in (0.025, 0.005):
            V = -65.0
            for _ in range(round(1.0 / dt)):
                V = (V + dt * 0.1 * -70.0) / (1.0 + dt * 0.1)
            with self.subTest(dt=dt):
                self.assertAlmostEqual(self.f64[str(dt)]["V"], V, delta=1e-10)


class StaggeredDeltaInputTest(unittest.TestCase):
    """CELL-33: the point-tree steps add delta inputs last, as ``post_integral`` does."""

    @staticmethod
    def _dendrite(solver):
        """60 um x 2 um, 3 CVs, leak 0.1 mS/cm^2 at -70 mV: isopotential under a uniform kick."""
        branch = Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 2.0] * u.um, type="dendrite")
        cell = Cell(Morphology.from_root(branch), cv_policy=CVPerBranch(3), V_init=-65.0 * u.mV, solver=solver)
        cell.paint(AllRegion(), mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV))
        cell.init_state()
        return cell

    def test_kick_each_step_follows_the_implicit_euler_recursion(self):
        # The axial terms cancel, so every CV follows
        # V <- (V + dt k E) / (1 + dt k) + kick with k = 0.1 / ms. Every point
        # of the stored tree moves with it: before CELL-33 the kick was
        # ignored, and without shifting the tree the boundary points would
        # lag the last kick.
        dt = 0.025
        for solver in ("staggered", "dhs_voltage"):
            with self.subTest(solver=solver), jax.enable_x64(True), brainstate.environ.context(precision=64):
                cell = self._dendrite(solver)
                cell.add_delta_input("kick", lambda: 0.5 * u.mV)

                def step(index):
                    with brainstate.environ.context(t=index * dt * u.ms):
                        cell.update()

                with brainstate.environ.context(dt=dt * u.ms):
                    brainstate.transform.for_loop(step, np.arange(40))
                V = -65.0
                for _ in range(40):
                    V = (V + dt * 0.1 * -70.0) / (1.0 + dt * 0.1) + 0.5
                np.testing.assert_allclose(cell.V.value.to_decimal(u.mV), V, rtol=0.0, atol=1e-10)
                point_V = cell._point_voltage_for_mechanisms().to_decimal(u.mV)
                self.assertEqual(point_V.shape[-1], cell.runtime.n_point)
                np.testing.assert_allclose(point_V, V, rtol=0.0, atol=1e-10)

    def test_delta_inputs_end_the_step(self):
        calls = []
        cell = self._dendrite("staggered")
        cell._integrate_runtime_synapse_dynamics = lambda point_V: calls.append("synapse_dynamics")
        cell._update_ion_channel_families = lambda point_V: calls.append("family")
        cell._apply_delta_inputs = lambda **kwargs: calls.append(("delta", kwargs))
        with patch("braincell.quad._staggered._dhs_voltage_solve", lambda *args, **kwargs: calls.append("voltage")):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                staggered_step(cell)
                dhs_voltage_step(cell)
        delta = ("delta", {"point_tree": True})
        self.assertEqual(calls, ["voltage", "synapse_dynamics", "family", delta, "voltage", delta])


if __name__ == "__main__":
    unittest.main()
