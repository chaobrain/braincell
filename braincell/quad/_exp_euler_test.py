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

import math
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.quad import (
    exp_euler_step,
    ind_exp_euler_step,
)
from braincell.quad._testing import LinearDecay, integrate


class IndExpEulerLinearTest(unittest.TestCase):
    def test_one_step_matches_exponential(self):
        # For dx/dt = lambda * x with constant lambda, ind_exp_euler should
        # produce y_{n+1} = y_n * exp(lambda * dt) up to float precision.
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            ind_exp_euler_step(m)
        expected = math.exp(-0.01)  # dt/tau = 0.1/10 = 0.01
        self.assertAlmostEqual(float(m.x.value.to_decimal(u.mV)[0]), expected, places=5)

    def test_excluded_paths_are_skipped(self):
        m = LinearDecay()
        original = np.array(m.x.value.to_decimal(u.mV))
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            ind_exp_euler_step(m, excluded_paths=[("x",)])
        np.testing.assert_array_equal(np.array(m.x.value.to_decimal(u.mV)), original)

    def test_rejects_non_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        # HIGH-03: TypeError (not AssertionError) so ``python -O`` preserves
        # the contract.
        with self.assertRaises(TypeError):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                ind_exp_euler_step(Plain())


class ExpEulerTargetContractTest(unittest.TestCase):
    """``exp_euler_step`` accepts any ``DiffEqModule`` and reads its layout."""

    def test_rejects_plain_object(self):
        # TypeError (not AssertionError) so ``python -O`` preserves the contract.
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            with self.assertRaises(TypeError) as ctx:
                exp_euler_step(object())
        self.assertIn("DiffEqModule", str(ctx.exception))

    def test_accepts_minimal_diffeq_module(self):
        # The target no longer has to be an HHTypedNeuron: the state layout is
        # declared by the host through ``diffeq_state_merging`` rather than
        # inferred from the concrete class.
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            exp_euler_step(m)
        # dx/dt = -x/tau is linear, so one step is exact: x * exp(-dt/tau).
        expected = math.exp(-0.01)  # dt/tau = 0.1/10 = 0.01
        self.assertAlmostEqual(float(m.x.value.to_decimal(u.mV)[0]), expected, places=5)

    def test_default_merging_is_stack(self):
        # A plain DiffEqModule inherits 'stack'; Cell overrides it to 'concat'.
        self.assertEqual(LinearDecay.diffeq_state_merging, "stack")


class IndExpEulerHHTest(unittest.TestCase):
    def test_drives_hodgkin_huxley_to_a_spiking_trace(self):
        # Exercises the full HH stack through ``solver='ind_exp_euler'``:
        # a 10 nA/cm^2 drive over 10 ms must produce a finite, spiking trace
        # rather than diverging or going NaN.
        vs = np.asarray(integrate('ind_exp_euler').to_decimal(u.mV))
        self.assertTrue(np.all(np.isfinite(vs)))
        self.assertGreater(vs.min(), -100.0)
        self.assertLess(vs.max(), 80.0)
        self.assertGreater(vs.max() - vs.min(), 50.0)


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
