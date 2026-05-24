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
    # Strip units before returning so matplotlib can convert via np.asarray.
    # Newer saiunit rejects np.asarray(dimensional_quantity).
    dts_q = u.math.asarray(dts)
    norm_q = u.math.asarray(norm)
    return dts_q.mantissa, norm_q.mantissa


class TestRungeKutta:
    def test_euler_step(self):
        dts, norms = compare('euler')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_midpoint_step(self):
        dts, norms = compare('midpoint')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_rk2_step(self):
        dts, norms = compare('rk2')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_heun2_step(self):
        dts, norms = compare('heun2')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_ralston2_step(self):
        dts, norms = compare('ralston2')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_rk3_step(self):
        dts, norms = compare('rk3')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_heun3_step(self):
        dts, norms = compare('heun3')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_ssprk3_step(self):
        dts, norms = compare('ssprk3')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_ralston3_step(self):
        dts, norms = compare('ralston3')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_rk4_step(self):
        dts, norms = compare('rk4')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()

    def test_ralston4_step(self):
        dts, norms = compare('ralston4')
        plt.loglog(dts, norms)
        # plt.show()
        plt.close()


# --------------------------------------------------------------------------- #
# Convergence on a linear ODE with a known analytical solution.
#
# These tests do not require the HH machinery: they exercise each step
# function on a minimal :class:`DiffEqModule` whose exact solution is
# ``x(t) = x0 * exp(-t/tau)``.
# --------------------------------------------------------------------------- #
import math
import unittest

import jax.numpy as jnp
import numpy as np

from braincell import (
    DiffEqModule,
    DiffEqState,
)
from braincell.quad import (

    euler_step,
    heun2_step,
    heun3_step,
    midpoint_step,
    ralston2_step,
    ralston3_step,
    ralston4_step,
    rk2_step,
    rk3_step,
    rk4_step,
    ssprk3_step,
)

_FLOAT_DTYPE = jnp.asarray(0.0).dtype


class _LinearDecay(brainstate.nn.Module, DiffEqModule):
    """Scalar linear ODE ``dx/dt = -x/tau`` for analytical comparisons."""

    def __init__(self, x0=1.0, tau_ms=10.0, shape=(3,)):
        super().__init__()
        self.tau = tau_ms * u.ms
        self.x = DiffEqState(jnp.full(shape, x0, dtype=_FLOAT_DTYPE) * u.mV)
        self.aux = brainstate.ShortTermState(jnp.zeros(shape, dtype=_FLOAT_DTYPE))
        self.pre_calls = 0
        self.post_calls = 0

    def pre_integral(self, *args, **kwargs):
        self.pre_calls += 1

    def post_integral(self, *args, **kwargs):
        self.post_calls += 1

    def compute_derivative(self, *args, **kwargs):
        self.x.derivative = -self.x.value / self.tau


def _drive(method, dt_ms=0.1, n_steps=100, x0=1.0, tau_ms=10.0):
    """Drive ``method`` ``n_steps`` times on a fresh :class:`_LinearDecay`."""
    m = _LinearDecay(x0=x0, tau_ms=tau_ms)
    dt = dt_ms * u.ms
    with brainstate.environ.context(dt=dt):
        for i in range(n_steps):
            with brainstate.environ.context(t=i * dt):
                method(m)
    return float(m.x.value.to_decimal(u.mV)[0]), m


class RungeKuttaConvergenceTest(unittest.TestCase):
    """Verifies each RK step on a linear ODE with known analytical solution."""

    # ``set_module_as`` rewrites ``__name__`` to ``"braincell"``, so we keep
    # the canonical name alongside the function for nicer test labels.
    METHODS_AND_ORDERS = [
        ('euler', euler_step, 1),
        ('midpoint', midpoint_step, 2),
        ('rk2', rk2_step, 2),
        ('heun2', heun2_step, 2),
        ('ralston2', ralston2_step, 2),
        ('rk3', rk3_step, 3),
        ('heun3', heun3_step, 3),
        ('ssprk3', ssprk3_step, 3),
        ('ralston3', ralston3_step, 3),
        ('rk4', rk4_step, 4),
        ('ralston4', ralston4_step, 4),
    ]

    def _final_value(self, method, dt_ms, n_steps, tau_ms=10.0):
        return _drive(method, dt_ms=dt_ms, n_steps=n_steps, tau_ms=tau_ms)[0]

    def test_each_method_matches_analytical_solution(self):
        # 100 steps of dt=0.1 ms over a tau=10 ms decay → final value
        # should be close to exp(-1) ≈ 0.367879.
        target = math.exp(-1.0)
        for name, method, order in self.METHODS_AND_ORDERS:
            with self.subTest(method=name, order=order):
                final = self._final_value(method, dt_ms=0.1, n_steps=100)
                # Allow looser tolerance for low-order methods.
                tol = {1: 5e-3, 2: 1e-4, 3: 1e-5, 4: 1e-6}[order]
                self.assertAlmostEqual(final, target, delta=tol)

    def test_global_error_decreases_with_step_size(self):
        # Each integrator should be consistent: shrinking the step size
        # from 0.4 ms down to 0.1 ms should not *increase* the global error.
        # (Going below 0.05 ms makes float32 noise dominate the higher-order
        # methods, so we stop there.)
        target = math.exp(-1.0)
        for name, method, order in self.METHODS_AND_ORDERS:
            with self.subTest(method=name):
                err_coarse = abs(
                    self._final_value(method, dt_ms=0.4, n_steps=25) - target
                )
                err_fine = abs(
                    self._final_value(method, dt_ms=0.1, n_steps=100) - target
                )
                # Small noise margin so float32 round-off near the noise
                # floor doesn't trip an otherwise consistent method.
                self.assertLessEqual(err_fine, err_coarse + 1e-5)

    def test_convergence_order_estimate(self):
        # Empirically estimate convergence order on the linear decay using
        # only first- and second-order methods, where the error stays well
        # above the float32 noise floor for the step sizes considered.
        target = math.exp(-1.0)
        order_methods = [
            (name, m, o) for name, m, o in self.METHODS_AND_ORDERS if o <= 2
        ]
        for name, method, order in order_methods:
            with self.subTest(method=name, order=order):
                err1 = abs(self._final_value(method, dt_ms=0.4, n_steps=25) - target)
                err2 = abs(self._final_value(method, dt_ms=0.2, n_steps=50) - target)
                if err1 < 1e-6 or err2 == 0:
                    continue
                empirical = math.log(err1 / err2, 2)
                # Require at least half the theoretical order to allow for
                # higher-order error terms on this small problem.
                self.assertGreater(empirical, 0.5 * order)

    def test_pre_and_post_integral_called_once_per_step(self):
        m = _LinearDecay()
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        self.assertEqual(m.pre_calls, 1)
        self.assertEqual(m.post_calls, 1)

    def test_aux_state_unchanged_after_step(self):
        m = _LinearDecay()
        before = np.array(m.aux.value)
        with brainstate.environ.context(t=0. * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        np.testing.assert_array_equal(m.aux.value, before)


if __name__ == "__main__":
    unittest.main()
