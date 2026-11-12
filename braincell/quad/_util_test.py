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

"""Tests for :mod:`braincell.quad._util`.

Covers ``split_diffeq_states``, ``apply_standard_solver_step``, and
``jacrev_last_dim`` — the plumbing every step function relies on.
"""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell import (
    DiffEqModule,
    DiffEqSingleState,
    IndependentIntegration,
)

from braincell.quad._testing import FLOAT_DTYPE, LinearDecay
from braincell.quad._util import (
    _MergedLayout,
    _PopulationSlotLayout,
    apply_standard_solver_step,
    jacrev_last_dim,
    population_slot_layout,
    split_diffeq_states,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class _IndepSub(brainstate.nn.Module, DiffEqModule, IndependentIntegration):
    def __init__(self):
        IndependentIntegration.__init__(self, "euler")
        brainstate.nn.Module.__init__(self)
        self.y = DiffEqSingleState(jnp.ones(2, dtype=FLOAT_DTYPE) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.y.derivative = -self.y.value / (5.0 * u.ms)


class _DependentChild(brainstate.nn.Module, DiffEqModule):
    def __init__(self):
        super().__init__()
        self.z = DiffEqSingleState(jnp.ones(2, dtype=FLOAT_DTYPE) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.z.derivative = -self.z.value / (7.0 * u.ms)


class _IndepSubWithDependentChild(_IndepSub):
    def __init__(self):
        super().__init__()
        self.child = _DependentChild()


class _OuterWithIndep(brainstate.nn.Module, DiffEqModule):
    def __init__(self, sub=None):
        super().__init__()
        self.x = DiffEqSingleState(jnp.ones(2, dtype=FLOAT_DTYPE) * u.mV)
        self.sub = _IndepSub() if sub is None else sub

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / (10.0 * u.ms)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class SplitDiffEqStatesTest(unittest.TestCase):
    def test_separates_diffeq_and_other_states(self):
        m = LinearDecay(shape=(2,))
        all_st, diffeq_st, other_st = split_diffeq_states(m)
        # Both states show up in the full set.
        self.assertEqual(len(all_st), 2)
        # Only the DiffEqState is integrated.
        self.assertEqual(set(diffeq_st.keys()), {("x",)})
        self.assertEqual(set(other_st.keys()), {("aux",)})

    def test_excludes_states_directly_owned_by_independent_submodules(self):
        m = _OuterWithIndep()
        _, diffeq_st, _ = split_diffeq_states(m)
        # The outer ``x`` is integrated, but ``sub.y`` lives under an
        # IndependentIntegration sub-module and must be excluded.
        diffeq_keys = set(diffeq_st.keys())
        self.assertIn(("x",), diffeq_keys)
        self.assertNotIn(("sub", "y"), diffeq_keys)

    def test_keeps_dependent_child_state_under_independent_parent(self):
        m = _OuterWithIndep(sub=_IndepSubWithDependentChild())
        _, diffeq_st, _ = split_diffeq_states(m)
        diffeq_keys = set(diffeq_st.keys())
        self.assertIn(("x",), diffeq_keys)
        self.assertNotIn(("sub", "y"), diffeq_keys)
        self.assertIn(("sub", "child", "z"), diffeq_keys)

    def test_excluded_paths_still_exclude_subtrees(self):
        m = _OuterWithIndep(sub=_IndepSubWithDependentChild())
        _, diffeq_st, _ = split_diffeq_states(m, excluded_paths=(("sub", "child"),))
        diffeq_keys = set(diffeq_st.keys())
        self.assertIn(("x",), diffeq_keys)
        self.assertNotIn(("sub", "y"), diffeq_keys)
        self.assertNotIn(("sub", "child", "z"), diffeq_keys)


class ApplyStandardSolverStepTest(unittest.TestCase):
    def test_passes_array_state_and_invokes_hooks(self):
        m = LinearDecay(shape=(2,))

        observed = {}

        def fake_step(fn, y0, t, dt, args):
            observed["y0_shape"] = y0.shape
            observed["t"] = t
            observed["dt"] = dt
            observed["fn"] = fn
            return y0, {}  # no-op solver: return inputs unchanged

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            apply_standard_solver_step(fake_step, m, 0.0 * u.ms, 0.1 * u.ms)

        # The merged y0 has shape ``(*pop_shape, n_diffeq_state)``; with one
        # scalar state of shape (2,) and 'concat' merging it stays (2,).
        self.assertEqual(observed["y0_shape"], (2,))
        # pre/post hooks are invoked exactly once each.
        self.assertEqual(m.pre_calls, 1)
        self.assertEqual(m.post_calls, 1)

    def test_rejects_non_diffeq_module(self):
        # TypeError rather than AssertionError so ``python -O`` keeps the guard.
        with self.assertRaises(TypeError):
            apply_standard_solver_step(lambda *a, **k: None, object(), 0.0, 0.1)

    def test_rejects_unknown_merging(self):
        m = LinearDecay(shape=(2,))
        with self.assertRaises(ValueError):
            apply_standard_solver_step(
                lambda *a, **k: None,
                m,
                0.0 * u.ms,
                0.1 * u.ms,
                merging="bogus",
            )

    def test_rejects_non_callable_solver_step(self):
        m = LinearDecay(shape=(2,))
        with self.assertRaises(TypeError):
            apply_standard_solver_step(
                "not callable",  # type: ignore[arg-type]
                m,
                0.0 * u.ms,
                0.1 * u.ms,
            )


class _RowsAndCells(brainstate.nn.Module, DiffEqModule):
    """A host with per-cell states ``(n_cell, 2)`` and packed rows ``(n_rows,)`` (CELL-31b).

    ``dv/dt = -v + sum of its cell's rows`` and ``dr/dt = -2 r``, so each cell
    is coupled to its own rows only. Declares the rows through
    ``_diffeq_block_layout``.
    """

    def __init__(self, owner=(0, 0, 2), merging="concat"):
        super().__init__()
        self.owner = np.asarray(owner)
        n_cell = 3
        self.v = DiffEqSingleState(jnp.ones((n_cell, 2), dtype=FLOAT_DTYPE) * u.mV)
        self.rows = _Rows(len(owner))
        self.merging = merging
        self.layouts = []

    def compute_derivative(self, *args):
        drive = (
            jnp.zeros(self.v.value.shape[0], dtype=FLOAT_DTYPE).at[self.owner].add(self.rows.r.value.to_decimal(u.mV))
        )
        self.v.derivative = (-self.v.value + drive[:, None] * u.mV) / u.ms
        self.rows.r.derivative = -2.0 * self.rows.r.value / u.ms

    def _diffeq_block_layout(self, diffeq_states, merging):
        slots = (self.owner, np.asarray([0, 1, 0]), 2)
        layout = population_slot_layout(diffeq_states, merging, (3,), [(self.rows, slots)])
        self.layouts.append(layout)
        return layout


class _Rows(brainstate.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.r = DiffEqSingleState(jnp.arange(1.0, n + 1.0, dtype=FLOAT_DTYPE) * u.mV)


class BlockLayoutTest(unittest.TestCase):
    """CELL-31b: the default merged layout and the per-cell slot layout."""

    def test_merged_layout_is_the_former_merging(self):
        # The default layout must pack exactly as the step always did:
        # jnp.concatenate / jnp.stack of the leaves along the last axis.
        values = {("a",): jnp.arange(6.0).reshape(2, 3) * u.mV, ("b",): jnp.arange(6.0, 12.0).reshape(2, 3)}
        leaves = jax.tree.leaves(values)
        for method, merge in (("concat", jnp.concatenate), ("stack", jnp.stack)):
            with self.subTest(method=method):
                layout = _MergedLayout(method)
                packed = layout.pack(values)
                np.testing.assert_array_equal(packed, merge(leaves, axis=-1))
                back = layout.unpack(packed, values)
                self.assertEqual(set(back), set(values))
                for key in values:
                    np.testing.assert_array_equal(u.get_mantissa(back[key]), u.get_mantissa(values[key]))
                    self.assertEqual(u.get_unit(back[key]), u.get_unit(values[key]))

    def test_slot_layout_puts_each_row_in_its_cells_block(self):
        cells = jnp.arange(12.0).reshape(2, 3, 2)  # batch 2, 3 cells, 2 columns
        rows = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # batch 2, 3 rows
        values = {("v",): cells * u.mV, ("r",): rows * u.mV}
        layout = _PopulationSlotLayout("concat", (3,), {("r",): (np.asarray([0, 0, 2]), np.asarray([0, 1, 0]), 2)})
        packed = layout.pack(values)
        self.assertEqual(packed.shape, (2, 3, 4))
        np.testing.assert_array_equal(packed[..., :2], cells)
        # Cell 0 holds rows 0 and 1, cell 1 nothing (zeros), cell 2 row 2 in slot 0.
        np.testing.assert_array_equal(packed[:, 0, 2:], rows[:, :2])
        np.testing.assert_array_equal(packed[:, 1, 2:], 0.0)
        np.testing.assert_array_equal(packed[:, 2, 2:], np.stack([rows[:, 2], np.zeros(2)], axis=-1))
        back = layout.unpack(packed, values)
        np.testing.assert_array_equal(u.get_mantissa(back[("v",)]), cells)
        np.testing.assert_array_equal(u.get_mantissa(back[("r",)]), rows)

    def test_slot_layout_under_stack_gives_each_leaf_one_column(self):
        # A lumped runtime declares 'stack': per-member leaves have no trailing axis.
        values = {("V",): jnp.asarray([1.0, 2.0]), ("r",): jnp.asarray([5.0])}
        layout = _PopulationSlotLayout("stack", (2,), {("r",): (np.asarray([1]), np.asarray([0]), 1)})
        packed = layout.pack(values)
        np.testing.assert_array_equal(packed, [[1.0, 0.0], [2.0, 5.0]])
        back = layout.unpack(packed, values)
        np.testing.assert_array_equal(back[("V",)], values[("V",)])
        np.testing.assert_array_equal(back[("r",)], values[("r",)])

    def test_without_packed_states_the_host_gets_the_merged_default(self):
        m = LinearDecay(shape=(2,))
        diffeq = split_diffeq_states(m)[1]
        self.assertIsNone(population_slot_layout(diffeq, "concat", (2,), [(_Rows(1), (np.zeros(1), np.zeros(1), 1))]))

    def test_the_step_solves_each_cell_with_its_own_rows(self):
        # One linear step through apply_standard_solver_step with the slot
        # layout: f(y) is evaluated per cell, coupled to that cell's rows.
        m = _RowsAndCells()
        observed = {}

        def euler(fn, y0, t, dt, args):
            dy, aux = fn(t, y0, *args)
            observed["shape"] = y0.shape
            return y0 + 0.1 * dy, aux

        apply_standard_solver_step(euler, m, 0.0 * u.ms, 0.1 * u.ms, merging="concat")
        self.assertIsInstance(m.layouts[0], _PopulationSlotLayout)
        self.assertEqual(observed["shape"], (3, 4))  # 3 cells x (2 v columns + 2 slots)
        # Cell 0 is driven by rows 0 and 1 (1 + 2 mV), cell 1 by none, cell 2 by row 2 (3 mV).
        expected_v = 1.0 + 0.1 * (-1.0 + np.asarray([3.0, 0.0, 3.0]))
        np.testing.assert_allclose(m.v.value.to_decimal(u.mV), np.repeat(expected_v[:, None], 2, axis=1), rtol=1e-6)
        np.testing.assert_allclose(m.rows.r.value.to_decimal(u.mV), np.asarray([1.0, 2.0, 3.0]) * 0.8, rtol=1e-6)


class _Holder:
    """A plain object: graph traversal does not see the state it holds."""

    def __init__(self, value):
        self.state = brainstate.ShortTermState(value)


class _ReadsUnlisted(brainstate.nn.Module, DiffEqModule):
    """dx/dt = -x / tau, with tau read from a state outside the module graph."""

    def __init__(self, write=False):
        super().__init__()
        self.x = DiffEqSingleState(jnp.ones(2, dtype=FLOAT_DTYPE) * u.mV)
        self.holder = _Holder(jnp.full(2, 10.0, dtype=FLOAT_DTYPE) * u.ms)
        self.write = write

    def compute_derivative(self, *args, **kwargs):
        tau = self.holder.state.value
        if self.write:
            self.holder.state.value = tau
        self.x.derivative = -self.x.value / tau


class UnlistedStateTest(unittest.TestCase):
    """A Cell keeps its geometry outside the graph; reading it is allowed."""

    def test_block_and_independent_steps_read_an_unlisted_state(self):
        from braincell.quad import backward_euler_step, crank_nicolson_step, exp_euler_step, ind_exp_euler_step

        for step in (exp_euler_step, backward_euler_step, crank_nicolson_step, ind_exp_euler_step):
            with self.subTest(step=step.__name__):
                m = _ReadsUnlisted()
                self.assertNotIn(id(m.holder.state), {id(st) for st in split_diffeq_states(m)[0].values()})
                with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                    step(m)
                np.testing.assert_allclose(m.x.value.to_decimal(u.mV), np.exp(-0.01), rtol=1e-4)

    def test_writing_an_unlisted_state_raises(self):
        from braincell.quad import exp_euler_step, ind_exp_euler_step

        for step in (exp_euler_step, ind_exp_euler_step):
            with self.subTest(step=step.__name__):
                with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                    with self.assertRaisesRegex(ValueError, "not in the state list, but compute_derivative"):
                        step(_ReadsUnlisted(write=True))


class JacrevLastDimTest(unittest.TestCase):
    def test_diagonal_linear_function(self):
        def f(y):
            return -2.0 * y

        y0 = jnp.array([1.0, 2.0, 3.0])
        A, val = jacrev_last_dim(f, y0)
        np.testing.assert_allclose(A, -2.0 * np.eye(3))
        np.testing.assert_allclose(val, -2.0 * np.array([1, 2, 3]))

    def test_with_aux(self):
        def f(y):
            return -y, ("aux",)

        y0 = jnp.array([1.0, 2.0])
        A, val, aux = jacrev_last_dim(f, y0, has_aux=True)
        np.testing.assert_allclose(A, -np.eye(2))
        np.testing.assert_allclose(val, -np.array([1.0, 2.0]))
        self.assertEqual(aux, ("aux",))


if __name__ == "__main__":
    unittest.main()
