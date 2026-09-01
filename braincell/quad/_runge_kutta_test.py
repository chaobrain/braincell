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
import numpy as np

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
from braincell.quad._testing import LinearDecay, drive


class RungeKuttaConvergenceTest(unittest.TestCase):
    """Verifies each RK step on a linear ODE with known analytical solution."""

    # The short registry name, not ``func.__name__``, so a subTest label
    # reads ``rk2`` rather than ``rk2_step``.
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
        return drive(method, dt_ms=dt_ms, n_steps=n_steps, tau_ms=tau_ms)[0]

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
                err_coarse = abs(self._final_value(method, dt_ms=0.4, n_steps=25) - target)
                err_fine = abs(self._final_value(method, dt_ms=0.1, n_steps=100) - target)
                # Small noise margin so float32 round-off near the noise
                # floor doesn't trip an otherwise consistent method.
                self.assertLessEqual(err_fine, err_coarse + 1e-5)

    def test_convergence_order_estimate(self):
        # Empirically estimate convergence order on the linear decay using
        # only first- and second-order methods, where the error stays well
        # above the float32 noise floor for the step sizes considered.
        target = math.exp(-1.0)
        order_methods = [(name, m, o) for name, m, o in self.METHODS_AND_ORDERS if o <= 2]
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
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        self.assertEqual(m.pre_calls, 1)
        self.assertEqual(m.post_calls, 1)

    def test_aux_state_unchanged_after_step(self):
        m = LinearDecay()
        before = np.array(m.aux.value)
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            rk4_step(m)
        np.testing.assert_array_equal(m.aux.value, before)


if __name__ == "__main__":
    unittest.main()
