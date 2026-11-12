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
import jax
import numpy as np

from braincell.quad import (
    backward_euler_step,
)
from braincell.quad._testing import CoupledCompartments, LinearDecay, drive, integrate


# --------------------------------------------------------------------------- #
# Analytical-solution test on a linear ODE
# --------------------------------------------------------------------------- #
class BackwardEulerLinearTest(unittest.TestCase):
    def test_one_step_matches_analytical(self):
        # For dx/dt = -x/tau, backward Euler gives
        #     x_{n+1} = x_n / (1 + dt/tau).
        # With dt=0.1 ms, tau=10 ms → x_1 = 1/1.01 ≈ 0.990099.
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            backward_euler_step(m)
        result = float(m.x.value.to_decimal(u.mV)[0])
        self.assertAlmostEqual(result, 1.0 / 1.01, places=5)

    def test_long_run_converges_to_exact_solution(self):
        # Backward Euler is L-stable; on a linear decay it converges to the
        # exact solution as dt → 0.
        target = math.exp(-1.0)
        result, _ = drive(backward_euler_step, dt_ms=0.01, n_steps=1000)
        self.assertAlmostEqual(result, target, delta=5e-4)

    def test_rejects_non_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        # TypeError (not AssertionError) so ``python -O`` preserves the contract.
        with self.assertRaises(TypeError):
            with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
                backward_euler_step(Plain())


# --------------------------------------------------------------------------- #
# Hodgkin-Huxley single-compartment end-to-end test.
# --------------------------------------------------------------------------- #
class _StackedCompartments(CoupledCompartments):
    diffeq_state_merging = 'stack'


def _one_coupled_step(cls, dt_ms=0.1):
    with jax.enable_x64(True), brainstate.environ.context(precision=64):
        module = cls()
        x0 = np.asarray(module.x.value.to_decimal(u.mV))
        with brainstate.environ.context(t=0.0 * u.ms, dt=dt_ms * u.ms):
            backward_euler_step(module)
        return x0, np.asarray(module.x.value.to_decimal(u.mV)), np.asarray(module.jacobian())


class BackwardEulerHostMergingTest(unittest.TestCase):
    """CELL-31a: the step merges states as the host declares (``diffeq_state_merging``)."""

    def test_concat_host_gets_one_block_per_cell(self):
        # On a linear system one step is (I - dt A) x_1 = x_0, coupling included.
        # The step used to merge with 'stack' whatever the host declared, which
        # made one block per compartment: on a cable Cell the axial coupling
        # left the Jacobian and V went non-finite on stiff discretizations.
        x0, x1, A = _one_coupled_step(CoupledCompartments)
        expected = np.linalg.solve(np.eye(2) - 0.1 * A, x0.T).T
        np.testing.assert_allclose(x1, expected, rtol=0, atol=1e-12)

    def test_stack_host_gets_one_block_per_state_row(self):
        # 'stack' (the lumped Cell's engine, a plain DiffEqModule) solves each row on
        # its own; jacrev over the last axis then sees the column sum of A,
        # here -1/tau, so x_1 = x_0 + dt f(x_0) / (1 + dt / tau).
        x0, x1, A = _one_coupled_step(_StackedCompartments)
        expected = x0 + 0.1 * (x0 @ A.T) / (1.0 + 0.1 / 10.0)
        np.testing.assert_allclose(x1, expected, rtol=0, atol=1e-12)


class BackwardEulerHHTest(unittest.TestCase):
    def test_drives_hodgkin_huxley_to_a_spiking_trace(self):
        # Exercises the full HH stack through ``solver='backward_euler'``:
        # a 10 nA/cm^2 drive over 10 ms must produce a finite, spiking trace
        # rather than diverging or going NaN.
        vs = np.asarray(integrate('backward_euler').to_decimal(u.mV))
        self.assertTrue(np.all(np.isfinite(vs)))
        # Physiological envelope: the trace must stay bounded ...
        self.assertGreater(vs.min(), -100.0)
        self.assertLess(vs.max(), 80.0)
        # ... and must actually spike rather than sit at rest.
        self.assertGreater(vs.max() - vs.min(), 50.0)


if __name__ == "__main__":
    unittest.main()
