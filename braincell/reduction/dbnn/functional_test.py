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

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn.functional import (
    build_kernels,
    causal_fft_convolve,
    compute_metrics,
    dbnn_f,
    dbnn_forward,
    _dbnn_gif_f_state,
    dbnn_r,
    _dbnn_gif_step,
    dbnn_step,
    masked_mse,
    unpack_quadratic_weight,
)


def _params():
    return {
        "tau_rise": jnp.asarray([2.0, 4.0], dtype=jnp.float32),
        "tau_decay": jnp.asarray([8.0, 12.0], dtype=jnp.float32),
        "omega": jnp.asarray([1.5, 0.75], dtype=jnp.float32),
        "quadratic_weight_upper": jnp.asarray([0.2], dtype=jnp.float32),
        "bias": jnp.asarray(-70.0, dtype=jnp.float32),
        "v_th": jnp.asarray(1e6, dtype=jnp.float32),
        "dt_ms": jnp.asarray(0.25, dtype=jnp.float32),
    }


def _channel_signs():
    return jnp.asarray([1.0, -1.0], dtype=jnp.float32)


class FunctionalTest(unittest.TestCase):
    def test_unpack_quadratic_weight_is_strict_upper_triangle(self):
        actual = unpack_quadratic_weight(jnp.asarray([1.0, 2.0, 3.0]), 3)
        expected = np.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(actual, expected)

    def test_causal_fft_convolve_does_not_wrap(self):
        inputs = jnp.asarray([[[0.0, 0.0, 0.0, 1.0]]])
        kernels = jnp.asarray([[0.0, 2.0, 3.0, 4.0]])
        actual = causal_fft_convolve(inputs, kernels)
        np.testing.assert_allclose(actual, [[[0.0, 0.0, 0.0, 0.0]]], atol=1e-6)

    def test_fft_and_recurrent_paths_are_equivalent(self):
        params = _params()
        inputs = jnp.asarray(
            [
                [[1.0, 0.0, 0.0, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.25, 0.0]],
                [[0.0, 0.5, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.5, 0.0, 0.0]],
            ],
            dtype=jnp.float32,
        )
        np.testing.assert_allclose(
            dbnn_f(params, inputs, channel_signs=_channel_signs()),
            dbnn_r(params, inputs, channel_signs=_channel_signs()),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sequence_and_single_step_recurrences_are_equivalent(self):
        params = _params()
        inputs = jnp.asarray([[[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.25]]], dtype=jnp.float32)
        initial = {
            "decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "rise_decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "previous_voltage": jnp.full((1,), params["bias"], dtype=jnp.float32),
        }

        def step(state, events):
            state, (voltage, _) = dbnn_step(params, state, events, channel_signs=_channel_signs())
            return state, voltage

        _, voltage_tb = jax.lax.scan(step, initial, jnp.transpose(inputs, (2, 0, 1)))
        np.testing.assert_allclose(
            voltage_tb.T,
            dbnn_r(params, inputs, channel_signs=_channel_signs()),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_long_recurrent_sequence_matches_single_step_exactly(self):
        params = _params()
        pattern = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.25], [0.0, 0.0]], dtype=jnp.float32)
        events_tb = jnp.tile(pattern, (32, 1))[:, None, :]
        inputs = jnp.transpose(events_tb, (1, 2, 0))
        initial = {
            "decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "rise_decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "previous_voltage": jnp.full((1,), params["bias"], dtype=jnp.float32),
        }

        def step(state, events):
            state, (voltage, _) = dbnn_step(params, state, events, channel_signs=_channel_signs())
            return state, voltage

        _, voltage_tb = jax.lax.scan(step, initial, events_tb)
        np.testing.assert_array_equal(voltage_tb.T, dbnn_r(params, inputs, channel_signs=_channel_signs()))

    def test_forward_is_jittable_and_differentiable(self):
        params = _params()
        inputs = jnp.ones((1, 2, 5), dtype=jnp.float32)
        expected = dbnn_forward(params, inputs, channel_signs=_channel_signs())
        actual = jax.jit(dbnn_forward)(params, inputs, channel_signs=_channel_signs())
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        gradient = jax.grad(
            lambda omega: jnp.sum(dbnn_forward({**params, "omega": omega}, inputs, channel_signs=_channel_signs()))
        )(params["omega"])
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))
        self.assertTrue(bool(jnp.any(gradient != 0)))

    def test_metrics_preserve_undefined_variance_explained(self):
        predictions = jnp.asarray([1.0, 2.0, 3.0])
        targets = jnp.asarray([2.0, 2.0, 2.0])
        metrics = compute_metrics(predictions, targets)
        self.assertTrue(bool(jnp.isnan(metrics["variance_explained"])))
        self.assertEqual(int(metrics["valid_count"]), 3)
        self.assertAlmostEqual(float(masked_mse(predictions, targets)), 2.0 / 3.0, places=6)

    def test_build_kernels_respects_explicit_time_step(self):
        params = _params()
        coarse = build_kernels(params, 3, channel_signs=_channel_signs(), dt_ms=1.0)
        fine = build_kernels(params, 3, channel_signs=_channel_signs(), dt_ms=0.5)
        self.assertFalse(np.allclose(coarse, fine))

    def test_channel_sign_is_applied_once_to_first_layer_kernels(self):
        params = _params()
        params["omega"] = jnp.asarray([1.5, 0.75], dtype=jnp.float32)
        kernels = build_kernels(params, 3, channel_signs=jnp.asarray([1.0, -1.0]))
        self.assertGreater(float(kernels[0, 1]), 0.0)
        self.assertLess(float(kernels[1, 1]), 0.0)

    def test_dbnn_gif_adapts_threshold_without_changing_voltage(self):
        params = _params()
        params.update(
            bias=jnp.asarray(-70.0, dtype=jnp.float32),
            v_th=jnp.asarray(-69.0, dtype=jnp.float32),
            omega=jnp.asarray([10.0, 0.0], dtype=jnp.float32),
            threshold_increment_mv=jnp.asarray(5.0, dtype=jnp.float32),
            threshold_tau_ms=jnp.asarray(50.0, dtype=jnp.float32),
        )
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)

        voltage, threshold, spike = _dbnn_gif_f_state(params, inputs, channel_signs=_channel_signs())
        drive = dbnn_f(params, inputs, channel_signs=_channel_signs())

        np.testing.assert_allclose(voltage, drive)
        self.assertFalse(bool(spike[0, 0]))
        self.assertTrue(bool(spike[0, 1]))
        self.assertGreater(float(threshold[0, 2]), -69.0)
        self.assertFalse(bool(spike[0, 2]))

    def test_dbnn_gif_applies_reset_only_when_explicitly_configured(self):
        params = _params()
        params.update(
            v_th=jnp.asarray(-69.0, dtype=jnp.float32),
            omega=jnp.asarray([10.0, 0.0], dtype=jnp.float32),
            threshold_increment_mv=jnp.asarray(0.0, dtype=jnp.float32),
            threshold_tau_ms=jnp.asarray(50.0, dtype=jnp.float32),
            reset_amp=jnp.asarray(4.0, dtype=jnp.float32),
            tau_reset=jnp.asarray(5.0, dtype=jnp.float32),
        )
        inputs = jnp.asarray([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=jnp.float32)
        voltage, _, spike = _dbnn_gif_f_state(params, inputs, channel_signs=_channel_signs())
        drive = dbnn_f(params, inputs, channel_signs=_channel_signs())
        self.assertTrue(bool(spike[0, 1]))
        self.assertLess(float(voltage[0, 2]), float(drive[0, 2]))

    def test_dbnn_gif_sequence_and_step_are_equivalent(self):
        params = _params()
        params.update(
            v_th=jnp.asarray(-69.0, dtype=jnp.float32),
            threshold_increment_mv=jnp.asarray(4.0, dtype=jnp.float32),
            threshold_tau_ms=jnp.asarray(30.0, dtype=jnp.float32),
        )
        inputs = jnp.asarray([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]], dtype=jnp.float32)
        initial = {
            "decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "rise_decay": jnp.zeros((1, 2), dtype=jnp.float32),
            "threshold": jnp.zeros((1,), dtype=jnp.float32),
            "previous_voltage": jnp.full((1,), params["bias"], dtype=jnp.float32),
            "previous_threshold": jnp.full((1,), params["v_th"], dtype=jnp.float32),
        }

        def step(state, events):
            return _dbnn_gif_step(params, state, events, channel_signs=_channel_signs())

        _, (voltage_tb, threshold_tb, spike_tb) = jax.lax.scan(step, initial, jnp.transpose(inputs, (2, 0, 1)))
        voltage, threshold, spike = _dbnn_gif_f_state(params, inputs, channel_signs=_channel_signs())
        np.testing.assert_allclose(voltage_tb.T, voltage, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(threshold_tb.T, threshold, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(spike_tb.T, spike)

    def test_dbnn_gif_zero_increment_keeps_threshold_static(self):
        params = _params()
        params.update(
            bias=jnp.asarray(-70.0, dtype=jnp.float32),
            v_th=jnp.asarray(-69.0, dtype=jnp.float32),
            omega=jnp.asarray([10.0, 0.0], dtype=jnp.float32),
            threshold_increment_mv=jnp.asarray(0.0, dtype=jnp.float32),
            threshold_tau_ms=jnp.asarray(50.0, dtype=jnp.float32),
        )
        inputs = jnp.asarray([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32)

        voltage, threshold, spike = _dbnn_gif_f_state(params, inputs, channel_signs=_channel_signs())

        np.testing.assert_array_equal(threshold, np.full((1, 4), -69.0))
        np.testing.assert_array_equal(spike, [[False, True, False, False]])
        self.assertTrue(bool(np.all(voltage[0, 1:] >= -69.0)))


if __name__ == "__main__":
    unittest.main()
