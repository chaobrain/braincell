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

"""Unit tests for ``PairSTDP`` (CELL-12a); the Cell-level acceptance is in the network plasticity tests."""

import unittest

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.plasticity import PairSTDP

DT = 0.1 * u.ms


def _rule(size=2, **kwargs):
    params = dict(A_plus=0.01 * u.uS, A_minus=0.0105 * u.uS, tau_plus=20.0 * u.ms, tau_minus=10.0 * u.ms)
    params.update(kwargs)
    return PairSTDP(size, u.uS, **params)


class PairSTDPTest(unittest.TestCase):
    def test_init_state_has_the_weight_and_zero_traces(self) -> None:
        weight = [0.05, 0.03] * u.uS
        state = _rule().init_state(weight)
        self.assertEqual(sorted(state), ["w", "x_pre", "y_post"])
        self.assertIs(state["w"], weight)
        np.testing.assert_array_equal(state["x_pre"], [0.0, 0.0])
        np.testing.assert_array_equal(state["y_post"], [0.0, 0.0])

    def test_update_decays_reads_before_and_then_adds(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(precision=64):
            rule = _rule()
            state = {
                "w": jnp.asarray([0.05, 0.05]) * u.uS,
                "x_pre": jnp.asarray([1.0, 0.0]),
                "y_post": jnp.asarray([0.0, 2.0]),
            }
            new = rule.update(state, pre=jnp.asarray([1.0, 1.0]), post=jnp.asarray([1.0, 0.0]), dt=DT)
            x = np.exp(-0.1 / 20.0)
            y = 2.0 * np.exp(-0.1 / 10.0)
            np.testing.assert_allclose(new["w"].to_decimal(u.uS), [0.05 + 0.01 * x, 0.05 - 0.0105 * y], rtol=1e-12)
            np.testing.assert_allclose(new["x_pre"], [x + 1.0, 1.0], rtol=1e-12)
            np.testing.assert_allclose(new["y_post"], [1.0, y], rtol=1e-12)

    def test_bounds_clip_after_the_update(self) -> None:
        rule = _rule(w_min=-0.02 * u.uS, w_max=-0.01 * u.uS, A_plus=1.0 * u.uS, A_minus=1.0 * u.uS)
        state = {"w": [-0.015, -0.015] * u.uS, "x_pre": jnp.asarray([1.0, 0.0]), "y_post": jnp.asarray([0.0, 1.0])}
        new = rule.update(state, pre=jnp.asarray([0.0, 1.0]), post=jnp.asarray([1.0, 0.0]), dt=DT)
        np.testing.assert_allclose(new["w"].to_decimal(u.uS), [-0.01, -0.02], rtol=1e-6)

    def test_per_row_parameters(self) -> None:
        rule = _rule(A_plus=[0.01, 0.02] * u.uS, tau_plus=[10.0, 20.0] * u.ms)
        state = {"w": [0.0, 0.0] * u.uS, "x_pre": jnp.ones(2), "y_post": jnp.zeros(2)}
        new = rule.update(state, pre=jnp.zeros(2), post=jnp.ones(2), dt=DT)
        expected = [0.01 * np.exp(-0.01), 0.02 * np.exp(-0.005)]
        np.testing.assert_allclose(new["w"].to_decimal(u.uS), expected, rtol=1e-6)

    def test_parameter_units_convert_to_the_weight_unit(self) -> None:
        rule = _rule(A_plus=10.0 * u.nS, w_max=1.0 * u.nS)
        self.assertEqual(rule.A_plus.unit, u.uS)
        np.testing.assert_allclose(rule.A_plus.mantissa, 0.01)
        np.testing.assert_allclose(rule.w_max.mantissa, 0.001)

    def test_rejects_bad_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "'tau_plus' must be > 0 ms"):
            _rule(tau_plus=0.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "'tau_minus' must be > 0 ms"):
            _rule(tau_minus=[1.0, -1.0] * u.ms)
        with self.assertRaisesRegex(ValueError, "w_min <= w_max"):
            _rule(w_min=0.02 * u.uS, w_max=0.01 * u.uS)
        with self.assertRaisesRegex(TypeError, "'A_minus'"):
            _rule(A_minus=0.01)
        with self.assertRaisesRegex(ValueError, "'A_plus'"):
            _rule(A_plus=0.01 * u.mV)

    def test_check_weight_rejects_a_weight_outside_the_bounds(self) -> None:
        rule = _rule(w_min=0.0 * u.uS, w_max=0.04 * u.uS)
        rule.check_weight([0.0, 0.04] * u.uS)
        with self.assertRaisesRegex(ValueError, r"1 row\(s\) \(first: row 1.*w_max"):
            rule.check_weight([0.01, 0.05] * u.uS)
        with self.assertRaisesRegex(ValueError, "w_min"):
            rule.check_weight([-0.01, 0.01] * u.uS)
        _rule().check_weight([1e3, -1e3] * u.uS)  # unbounded


if __name__ == "__main__":
    unittest.main()
