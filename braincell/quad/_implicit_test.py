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

The module exposes one step function, ``crank_nicolson_step`` (the
trapezoidal rule solved by Newton iteration per neuron). The six cell-only
splitting variants that used to live here (``splitting_step``,
``cn_rk4_step``, ``cn_exp_euler_step``, ``implicit_rk4_step``,
``implicit_exp_euler_step``, ``exp_exp_euler_step``) were removed earlier:
they depended on a ``Cell`` interface that no longer exists.
"""

import time
import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell.quad as quad
from braincell.quad import (
    backward_euler_step,
    crank_nicolson_step,
    get_integrator,
    integrator_info,
    list_integrators,
    module_step_from_ode,
)
from braincell.quad._testing import HH, CoupledCompartments, LinearDecay, integrate
from braincell.quad.protocol import DiffEqModule, DiffEqSingleState

#: One trapezoidal step of dx/dt = -x**3 / (tau mV**2), x0 = 1 mV,
#: tau = 1 ms, dt = 0.1 ms: the real root of 0.05 x**3 + x - 0.95 = 0.
_CUBIC_TRAPEZOID = 0.9120644341272194
#: The same step with the former one-iteration Newton loop.
_CUBIC_ONE_ITERATION = 0.9130434782608695
#: One step of LinearDecay (tau = 10 ms, dt = 0.1 ms).
_DECAY_TRAPEZOID = 0.9900497512437811  # (1 - 0.005) / (1 + 0.005)
_DECAY_BACKWARD_EULER = 0.9900990099009901  # 1 / (1 + 0.01)


class _Cubic(brainstate.nn.Module, DiffEqModule):
    """dx/dt = -x**3 / (tau mV**2) in float64."""

    def __init__(self):
        super().__init__()
        self.tau = 1.0 * u.ms
        self.x = DiffEqSingleState(jnp.ones(2, dtype=jnp.float64) * u.mV)

    def compute_derivative(self, *args):
        self.x.derivative = -(self.x.value**3) / (self.tau * u.mV**2)


class _Float64Decay(LinearDecay):
    def __init__(self):
        super().__init__()
        self.x = DiffEqSingleState(jnp.ones(3, dtype=jnp.float64) * u.mV)


def _one_step(module, step=crank_nicolson_step, dt=0.1 * u.ms):
    with brainstate.environ.context(t=0.0 * u.ms, dt=dt):
        step(module)
    return np.asarray(module.x.value.to_decimal(u.mV))


def _maxrel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _hh_trace(method, dt_ms):
    return np.asarray(integrate(method, dt=dt_ms * u.ms).to_decimal(u.mV))[:, 0]


class CrankNicolsonAccuracyTest(unittest.TestCase):
    """The Newton loop converges to the trapezoidal solution."""

    def test_cubic_reaches_the_trapezoid_root(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            x = _one_step(_Cubic())
        np.testing.assert_allclose(x, _CUBIC_TRAPEZOID, rtol=0, atol=1e-10)
        # The former loop stopped after one Newton iteration.
        self.assertGreater(np.min(np.abs(x - _CUBIC_ONE_ITERATION)), 1e-4)

    def test_linear_decay_is_trapezoid_not_backward_euler(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            x = _one_step(_Float64Decay())
            be = _one_step(_Float64Decay(), step=backward_euler_step)
        np.testing.assert_allclose(x, _DECAY_TRAPEZOID, rtol=0, atol=1e-12)
        self.assertGreater(np.min(np.abs(x - _DECAY_BACKWARD_EULER)), 1e-5)
        np.testing.assert_allclose(be, _DECAY_BACKWARD_EULER, rtol=0, atol=1e-12)

    def test_explicit_time_arguments_override_the_environ(self):
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=5.0 * u.ms):
            crank_nicolson_step(m, dt=0.1 * u.ms)
        np.testing.assert_allclose(np.asarray(m.x.value.to_decimal(u.mV)), _DECAY_TRAPEZOID, rtol=1e-6)
        self.assertEqual((m.pre_calls, m.post_calls), (1, 1))

    def test_hh_matches_braintools_cnab(self):
        # imex_cnab_step with no explicit part is the trapezoidal rule solved by
        # braintools' own Newton loop; run it through the module adapter.
        def cnab(f, y, t):
            def zero(y_, t_):
                # Zero derivatives with the units of ``f``.
                return jax.tree.map(lambda v: 0 * v, f(y_, t_))

            return braintools.quad.imex_cnab_step(zero, f, y, y, t, max_iter=8)

        cnab.__name__ = 'cnab_trapezoid'
        braintools_cn = module_step_from_ode(cnab)
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            for dt_ms in (0.1, 0.025):
                with self.subTest(dt_ms=dt_ms):
                    ours = _hh_trace('crank_nicolson', dt_ms)
                    theirs = _hh_trace(braintools_cn, dt_ms)
                    self.assertTrue(np.all(np.isfinite(ours)))
                    self.assertLessEqual(_maxrel(ours, theirs), 1e-12)

    def test_hh_second_order(self):
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            ref = _hh_trace('crank_nicolson', 0.001)
            errors = []
            for dt_ms in (0.02, 0.01):
                trace = _hh_trace('crank_nicolson', dt_ms)
                stride = round(dt_ms / 0.001)
                errors.append(np.max(np.abs(trace - ref[stride - 1 :: stride])))
        ratio = errors[0] / errors[1]
        self.assertGreater(ratio, 3.5, errors)
        self.assertLess(ratio, 4.5, errors)

    def test_large_population_uses_per_neuron_blocks(self):
        # HH(1000) has 4000 states; a dense Jacobian would be 4000 x 4000.
        n_steps = 200
        timings = {}
        for name in ('crank_nicolson', 'backward_euler'):
            hh = HH(1000, solver=name)
            hh.init_state()

            def run(hh=hh):
                def step_fun(t):
                    with brainstate.environ.context(t=t):
                        hh.update(10 * u.nA / u.cm**2)
                    return hh.V.value[0]

                times = u.math.arange(0.0 * u.ms, n_steps * 0.025 * u.ms, 0.025 * u.ms)
                return brainstate.transform.for_loop(step_fun, times)

            with brainstate.environ.context(dt=0.025 * u.ms):
                jit_run = brainstate.transform.jit(run)
                jax.block_until_ready(jit_run())
                t0 = time.perf_counter()
                v = jax.block_until_ready(jit_run())
                timings[name] = time.perf_counter() - t0
            self.assertTrue(np.all(np.isfinite(np.asarray(v.to_decimal(u.mV)))))
        # Same order of magnitude as the single-linear-solve backward Euler.
        self.assertLess(timings['crank_nicolson'], 30 * timings['backward_euler'] + 0.5, timings)


class CrankNicolsonHostMergingTest(unittest.TestCase):
    """CELL-31a: the Newton system is one block per cell on a ``'concat'`` host."""

    def test_concat_host_reaches_the_coupled_trapezoid_solution(self):
        # Linear system: (I - dt A / 2) x_1 = (I + dt A / 2) x_0. Merged with
        # 'stack' the Newton matrix lacked the stiff coupling and the iteration
        # did not converge at this dt.
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            m = CoupledCompartments()
            x0 = np.asarray(m.x.value.to_decimal(u.mV))
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                crank_nicolson_step(m)
            x1 = np.asarray(m.x.value.to_decimal(u.mV))
            A = np.asarray(m.jacobian())
        half = 0.05 * A
        expected = np.linalg.solve(np.eye(2) - half, (np.eye(2) + half) @ x0.T).T
        np.testing.assert_allclose(x1, expected, rtol=0, atol=1e-12)

    def test_rejects_non_diffeq_module(self):
        # Checked before the host's merging is read: TypeError, not AttributeError.
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaisesRegex(TypeError, "DiffEqModule"):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                crank_nicolson_step(Plain())


class CrankNicolsonRegistrationTest(unittest.TestCase):
    """Registry metadata and the removed ``implicit_euler`` name."""

    #: Names deleted in an earlier cleanup; they must not come back.
    REMOVED = (
        "splitting",
        "implicit_rk4",
        "implicit_exp_euler",
        "cn_rk4",
        "cn_exp_euler",
        "exp_exp_euler",
    )

    def test_registered(self):
        self.assertIn("crank_nicolson", list_integrators())
        self.assertIs(get_integrator("crank_nicolson"), crank_nicolson_step)
        info = integrator_info("crank_nicolson")
        self.assertEqual((info.kind, info.category, info.order), ("module", "implicit", 2))
        self.assertEqual(info.fn.__module__, "braincell.quad")

    def test_implicit_euler_is_removed(self):
        with self.assertRaises(ValueError) as ctx:
            get_integrator("implicit_euler")
        self.assertIn("crank_nicolson", str(ctx.exception))
        self.assertIn("backward_euler", str(ctx.exception))
        self.assertNotIn("implicit_euler", list_integrators())
        with self.assertRaisesRegex(AttributeError, "crank_nicolson_step"):
            quad.implicit_euler_step

    def test_removed_cell_only_steps_are_gone(self):
        for name in self.REMOVED:
            with self.subTest(name=name):
                self.assertNotIn(name, list_integrators())
                self.assertFalse(hasattr(quad, f"{name}_step"))


if __name__ == "__main__":
    unittest.main()
