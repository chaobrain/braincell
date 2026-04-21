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

"""Tests for :mod:`braincell.quad._protocol`.

These tests cover the public mixin/state classes that every BrainCell
model uses to declare itself integrable: :class:`DiffEqState`,
:class:`DiffEqModule`, and :class:`IndependentIntegration`.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.quad import (
    get_integrator,
)
from braincell.quad._protocol import (
    DiffEqModule,
    DiffEqState,
    IndependentIntegration,
)


class DiffEqStateTest(unittest.TestCase):

    def test_initial_derivative_and_diffusion_are_none(self):
        st = DiffEqState(jnp.zeros(3))
        self.assertIsNone(st.derivative)
        self.assertIsNone(st.diffusion)

    def test_set_derivative_and_diffusion(self):
        st = DiffEqState(jnp.zeros(3))
        d = jnp.ones(3)
        st.derivative = d
        st.diffusion = 2 * d
        self.assertIs(st.derivative, d)
        np.testing.assert_array_equal(st.diffusion, 2 * d)

    def test_state_value_roundtrip(self):
        v = jnp.arange(4, dtype=jnp.float32) * u.mV
        st = DiffEqState(v)
        np.testing.assert_array_equal(
            st.value.to_decimal(u.mV), np.arange(4, dtype=np.float32)
        )


class DiffEqModuleTest(unittest.TestCase):

    def test_compute_derivative_must_be_overridden(self):
        class Bare(brainstate.nn.Module, DiffEqModule):
            pass

        with self.assertRaises(NotImplementedError):
            Bare().compute_derivative()

    def test_default_pre_and_post_integral_are_noops(self):
        class Bare(brainstate.nn.Module, DiffEqModule):
            def compute_derivative(self):
                pass

        b = Bare()
        # Both methods exist on the mixin and accept arbitrary args.
        self.assertIsNone(b.pre_integral(1, 2, k=3))
        self.assertIsNone(b.post_integral(1, 2, k=3))


class IndependentIntegrationTest(unittest.TestCase):

    def _make_sub(self, solver):
        class Sub(brainstate.nn.Module, DiffEqModule, IndependentIntegration):
            def __init__(self):
                IndependentIntegration.__init__(self, solver)
                brainstate.nn.Module.__init__(self)
                self.y = DiffEqState(jnp.ones(2, dtype=jnp.float32) * u.mV)

            def compute_derivative(self, *args, **kwargs):
                self.y.derivative = -self.y.value / (5. * u.ms)

        return Sub()

    def test_constructor_resolves_solver_string(self):
        sub = self._make_sub("euler")
        self.assertIs(sub.solver, get_integrator("euler"))

    def test_constructor_accepts_callable(self):
        def my_solver(target, *args):
            return "sentinel"

        sub = self._make_sub(my_solver)
        self.assertIs(sub.solver, my_solver)

    def test_constructor_resolves_alias_string(self):
        # Aliases should work the same as canonical names.
        sub = self._make_sub("explicit")
        self.assertIs(sub.solver, get_integrator("euler"))

    def test_make_integration_invokes_solver(self):
        observed = []

        def my_solver(target, *args):
            observed.append((target, args))

        sub = self._make_sub(my_solver)
        sub.make_integration("extra-arg")
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0][0], sub)
        self.assertEqual(observed[0][1], ("extra-arg",))


if __name__ == "__main__":
    unittest.main()
