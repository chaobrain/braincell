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

"""Tests for :mod:`braincell.quad._implicit`.

The cell-only step functions in this module (``splitting_step``,
``cn_rk4_step``, ``cn_exp_euler_step``, ``implicit_rk4_step``,
``implicit_exp_euler_step``, ``exp_exp_euler_step``) require a multi-
compartment :class:`braincell.Cell` target with a full conductance
matrix and are exercised by their dedicated cell tests.

What we test here is the small subset that runs on a minimal
:class:`DiffEqModule` target — namely ``implicit_euler_step`` — plus
basic registry-level metadata for the remaining cell-only methods.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell.quad import (
    DiffEqModule,
    DiffEqState,
    cn_exp_euler_step,
    cn_rk4_step,
    exp_exp_euler_step,
    get_registry,
    implicit_euler_step,
    implicit_exp_euler_step,
    implicit_rk4_step,
    splitting_step,
)


class _LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Scalar linear ODE ``dx/dt = -x/tau``."""

    def __init__(self, x0=1.0, tau_ms=10.0, shape=(3,)):
        super().__init__()
        self.tau = tau_ms * u.ms
        self.x = DiffEqState(jnp.full(shape, x0, dtype=jnp.float32) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


class ImplicitEulerLinearTest(unittest.TestCase):
    """``implicit_euler_step`` defaults to a Crank-Nicolson Newton solver."""

    def test_one_step_lies_in_cn_bracket(self):
        m = _LinearDecay()
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            implicit_euler_step(m, 0. * u.ms, 0.1 * u.ms)
        v = float(m.x.value.to_decimal(u.mV)[0])
        # The 1-step Crank-Nicolson value lies between the implicit-Euler
        # value 1/(1 + dt/tau) ≈ 0.99010 and the exact decay exp(-dt/tau)
        # ≈ 0.99005.
        self.assertGreater(v, 0.989)
        self.assertLess(v, 0.991)


class ImplicitMethodRegistrationTest(unittest.TestCase):
    """Sanity-check that every cell-only implicit step is registered."""

    EXPECTED = {
        "implicit_euler": implicit_euler_step,
        "splitting": splitting_step,
        "implicit_rk4": implicit_rk4_step,
        "implicit_exp_euler": implicit_exp_euler_step,
        "cn_rk4": cn_rk4_step,
        "cn_exp_euler": cn_exp_euler_step,
        "exp_exp_euler": exp_exp_euler_step,
    }

    def test_all_implicit_steps_registered(self):
        registry = get_registry()
        for name, func in self.EXPECTED.items():
            with self.subTest(name=name):
                self.assertIn(name, registry)
                self.assertIs(registry[name], func)

    def test_implicit_methods_have_expected_categories(self):
        registry = get_registry()
        # implicit_euler / splitting / implicit_rk4 / implicit_exp_euler /
        # cn_rk4 / cn_exp_euler are all in the "implicit" group.
        for name in [
            "implicit_euler",
            "splitting",
            "implicit_rk4",
            "implicit_exp_euler",
            "cn_rk4",
            "cn_exp_euler",
        ]:
            with self.subTest(name=name):
                self.assertEqual(registry.entry(name).category, "implicit")
        # exp_exp_euler is grouped with the exponential family.
        self.assertEqual(
            registry.entry("exp_exp_euler").category, "exponential"
        )


if __name__ == "__main__":
    unittest.main()
