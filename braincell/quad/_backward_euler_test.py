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

"""Tests for :mod:`braincell.quad._backward_euler`.

Combines:

* an analytical-solution test on a minimal linear ODE, and
* a smoke test that runs ``backward_euler`` through the standard
  Hodgkin-Huxley single-compartment fixture used by other quad tests.
"""

import math
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt

import braincell
from braincell.quad import (
    backward_euler_step,
)
from braincell.quad.protocol import (
    DiffEqModule,
    DiffEqState,
)

_FLOAT_DTYPE = jnp.asarray(0.0).dtype


# --------------------------------------------------------------------------- #
# Analytical-solution test on a linear ODE
# --------------------------------------------------------------------------- #
class _LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Scalar linear ODE ``dx/dt = -x/tau``."""

    def __init__(self, x0=1.0, tau_ms=10.0, shape=(3,)):
        super().__init__()
        self.tau = tau_ms * u.ms
        self.x = DiffEqState(jnp.full(shape, x0, dtype=_FLOAT_DTYPE) * u.mV)

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


def _drive(method, dt_ms=0.1, n_steps=100, x0=1.0, tau_ms=10.0):
    m = _LinearDecay(x0=x0, tau_ms=tau_ms)
    dt = dt_ms * u.ms
    with brainstate.environ.context(dt=dt):
        for i in range(n_steps):
            with brainstate.environ.context(t=i * dt):
                method(m)
    return float(m.x.value.to_decimal(u.mV)[0])


class BackwardEulerLinearTest(unittest.TestCase):

    def test_one_step_matches_analytical(self):
        # For dx/dt = -x/tau, backward Euler gives
        #     x_{n+1} = x_n / (1 + dt/tau).
        # With dt=0.1 ms, tau=10 ms → x_1 = 1/1.01 ≈ 0.990099.
        m = _LinearDecay()
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            backward_euler_step(m)
        result = float(m.x.value.to_decimal(u.mV)[0])
        self.assertAlmostEqual(result, 1.0 / 1.01, places=5)

    def test_long_run_converges_to_exact_solution(self):
        # Backward Euler is L-stable; on a linear decay it converges to the
        # exact solution as dt → 0.
        target = math.exp(-1.0)
        result = _drive(backward_euler_step, dt_ms=0.01, n_steps=1000)
        self.assertAlmostEqual(result, target, delta=5e-4)

    def test_rejects_non_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaises(AssertionError):
            with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
                backward_euler_step(Plain())


# --------------------------------------------------------------------------- #
# Hodgkin-Huxley single-compartment smoke test (mirrors the existing
# convergence-comparison harness used by ``_runge_kutta_test`` and
# ``_exp_euler_test``).
# --------------------------------------------------------------------------- #
class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='backward_euler'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.IK_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))


def integrate(method: str, dt=0.01 * u.ms):
    brainstate.random.seed(1)
    hh = HH(1, solver=method)
    hh.init_state()

    def step_fun(t):
        with brainstate.environ.context(t=t):
            hh.update(10 * u.nA / u.cm ** 2)
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


class TestBackwardEulerHH(unittest.TestCase):
    def test_backward_euler_step(self):
        dts, norms = compare('backward_euler')
        plt.loglog(dts, norms)
        plt.close()


if __name__ == "__main__":
    unittest.main()
