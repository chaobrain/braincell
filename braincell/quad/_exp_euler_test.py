# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-


import brainstate
import brainunit as u
import matplotlib.pyplot as plt

import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.K_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))


def integrate(method: str, dt=0.01 * u.ms):
    brainstate.random.seed(1)
    hh = HH(1, solver=method)
    hh.init_state()

    def step_fun(t):
        with brainstate.environ.context(t=t):
            spike = hh.update(10 * u.nA / u.cm ** 2)
        return hh.V.value

    with brainstate.environ.context(dt=dt):
        times = u.math.arange(0. * u.ms, 10 * u.ms, brainstate.environ.get_dt())
        vs = brainstate.transform.for_loop(step_fun, times)
    return vs


def compare(method):
    norm = []
    dts = [1e-3 * u.ms, 2e-3 * u.ms, 4e-3 * u.ms, 8e-3 * u.ms, 1e-2 * u.ms, 2e-2 * u.ms]
    for dt in dts:
        gold_vs = integrate('exp_euler', dt=dt)
        vs = integrate(method, dt=dt)
        norm.append(u.linalg.norm(gold_vs - vs))
    return u.math.asarray(dts), u.math.asarray(norm)


class TestRungeKutta:
    def test_euler_step(self):
        dts, norms = compare('ind_exp_euler')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()


# --------------------------------------------------------------------------- #
# Unit tests for exp_euler / ind_exp_euler that do not require the full HH
# stack. ind_exp_euler is exact for a locally linear ODE so we can compare
# against ``exp(-dt/tau)`` directly.
# --------------------------------------------------------------------------- #
import math
import unittest

import jax.numpy as jnp
import numpy as np

from braincell.quad.protocol import (
    DiffEqModule,
    DiffEqState,
)
from braincell.quad import (
    exp_euler_step,
    ind_exp_euler_step,
)

_FLOAT_DTYPE = jnp.asarray(0.0).dtype


class _LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Scalar linear ODE ``dx/dt = -x/tau``."""

    def __init__(self, x0=1.0, tau_ms=10.0, shape=(3,)):
        super().__init__()
        self.tau = tau_ms * u.ms
        self.x = DiffEqState(jnp.full(shape, x0, dtype=_FLOAT_DTYPE) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


class IndExpEulerLinearTest(unittest.TestCase):

    def test_one_step_matches_exponential(self):
        # For dx/dt = lambda * x with constant lambda, ind_exp_euler should
        # produce y_{n+1} = y_n * exp(lambda * dt) up to float precision.
        m = _LinearDecay()
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            ind_exp_euler_step(m)
        expected = math.exp(-0.01)  # dt/tau = 0.1/10 = 0.01
        self.assertAlmostEqual(
            float(m.x.value.to_decimal(u.mV)[0]), expected, places=5
        )

    def test_excluded_paths_are_skipped(self):
        m = _LinearDecay()
        original = np.array(m.x.value.to_decimal(u.mV))
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            ind_exp_euler_step(m, excluded_paths=[("x",)])
        np.testing.assert_array_equal(
            np.array(m.x.value.to_decimal(u.mV)), original
        )

    def test_rejects_non_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        # HIGH-03: TypeError (not AssertionError) so ``python -O`` preserves
        # the contract.
        with self.assertRaises(TypeError):
            with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
                ind_exp_euler_step(Plain())


class ExpEulerTypeGuardTest(unittest.TestCase):
    """``exp_euler_step`` requires an ``HHTypedNeuron`` target."""

    def test_rejects_minimal_diffeq_module(self):
        # HIGH-03: TypeError (not AssertionError) so ``python -O`` preserves
        # the contract.
        with self.assertRaises(TypeError):
            with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
                exp_euler_step(_LinearDecay())

    def test_rejects_plain_object(self):
        from braincell.quad import exp_euler_step

        with brainstate.environ.context(t=0. * u.ms, dt=0.025 * u.ms):
            with self.assertRaises(TypeError) as ctx:
                exp_euler_step(object())
        self.assertIn("HHTypedNeuron", str(ctx.exception))


class ExponentialEulerHandlesSingularJacobianTest(unittest.TestCase):
    """MED-06: update must remain finite when A is singular."""

    def test_update_finite_for_singular_A(self) -> None:
        from braincell.quad._exp_euler import _exponential_euler

        def f(t, y, *args):
            # df = 0 and A = df/dy = 0 → singular.  Update must be zero,
            # not NaN from solve(zeros, …).
            return jnp.zeros_like(y), None

        y0 = jnp.array([1.0, 2.0])
        y1, _ = _exponential_euler(f, y0, t=jnp.asarray(0.0), dt=0.1 * u.ms)
        self.assertTrue(bool(jnp.isfinite(y1).all()))
        np.testing.assert_allclose(np.asarray(y1), np.asarray(y0), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
