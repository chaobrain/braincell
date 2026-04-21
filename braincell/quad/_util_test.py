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
import jax.numpy as jnp
import numpy as np

from braincell import (
    DiffEqModule,
    DiffEqState,
    IndependentIntegration,
)
from braincell.quad._util import (
    apply_standard_solver_step,
    jacrev_last_dim,
    split_diffeq_states,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
class _LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Minimal module: dx/dt = -x/tau, with one DiffEqState and one auxiliary."""

    def __init__(self, shape=(2,)):
        super().__init__()
        self.tau = 10.0 * u.ms
        self.x = DiffEqState(jnp.ones(shape, dtype=jnp.float32) * u.mV)
        # A non-DiffEqState should be ignored by the integrator.
        self.aux = brainstate.ShortTermState(jnp.zeros(shape, dtype=jnp.float32))
        self.pre_calls = 0
        self.post_calls = 0

    def pre_integral(self, *args, **kwargs):
        self.pre_calls += 1

    def post_integral(self, *args, **kwargs):
        self.post_calls += 1

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


class _IndepSub(brainstate.nn.Module, DiffEqModule, IndependentIntegration):
    def __init__(self):
        IndependentIntegration.__init__(self, "euler")
        brainstate.nn.Module.__init__(self)
        self.y = DiffEqState(jnp.ones(2, dtype=jnp.float32) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.y.derivative = -self.y.value / (5. * u.ms)


class _OuterWithIndep(brainstate.nn.Module, DiffEqModule):
    def __init__(self):
        super().__init__()
        self.x = DiffEqState(jnp.ones(2, dtype=jnp.float32) * u.mV)
        self.sub = _IndepSub()

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / (10. * u.ms)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class SplitDiffEqStatesTest(unittest.TestCase):

    def test_separates_diffeq_and_other_states(self):
        m = _LinearDecay()
        all_st, diffeq_st, other_st = split_diffeq_states(m)
        # Both states show up in the full set.
        self.assertEqual(len(all_st), 2)
        # Only the DiffEqState is integrated.
        self.assertEqual(set(diffeq_st.keys()), {("x",)})
        self.assertEqual(set(other_st.keys()), {("aux",)})

    def test_excludes_independent_submodules(self):
        m = _OuterWithIndep()
        _, diffeq_st, _ = split_diffeq_states(m)
        # The outer ``x`` is integrated, but ``sub.y`` lives under an
        # IndependentIntegration sub-module and must be excluded.
        diffeq_keys = set(diffeq_st.keys())
        self.assertIn(("x",), diffeq_keys)
        self.assertNotIn(("sub", "y"), diffeq_keys)


class ApplyStandardSolverStepTest(unittest.TestCase):

    def test_passes_array_state_and_invokes_hooks(self):
        m = _LinearDecay()

        observed = {}

        def fake_step(fn, y0, t, dt, args):
            observed["y0_shape"] = y0.shape
            observed["t"] = t
            observed["dt"] = dt
            observed["fn"] = fn
            return y0, {}  # no-op solver: return inputs unchanged

        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            apply_standard_solver_step(fake_step, m, 0. * u.ms, 0.1 * u.ms)

        # The merged y0 has shape ``(*pop_shape, n_diffeq_state)``; with one
        # scalar state of shape (2,) and 'concat' merging it stays (2,).
        self.assertEqual(observed["y0_shape"], (2,))
        # pre/post hooks are invoked exactly once each.
        self.assertEqual(m.pre_calls, 1)
        self.assertEqual(m.post_calls, 1)

    def test_rejects_non_diffeq_module(self):
        with self.assertRaises(AssertionError):
            apply_standard_solver_step(lambda *a, **k: None, object(), 0., 0.1)

    def test_rejects_unknown_merging(self):
        m = _LinearDecay()
        with self.assertRaises(AssertionError):
            apply_standard_solver_step(
                lambda *a, **k: None,
                m,
                0. * u.ms,
                0.1 * u.ms,
                merging="bogus",
            )

    def test_rejects_non_callable_solver_step(self):
        m = _LinearDecay()
        with self.assertRaises(AssertionError):
            apply_standard_solver_step(
                "not callable",  # type: ignore[arg-type]
                m,
                0. * u.ms,
                0.1 * u.ms,
            )


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
