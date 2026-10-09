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

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell.reduction.dbnn import model
from braincell.reduction.dbnn.layout import ChannelLayout
from braincell.mech import Synapse


def _layout(n_channels=2, *, polarity="E"):
    e = 0.0 * u.mV if polarity == "E" else -80.0 * u.mV
    indices = tuple(range(n_channels))
    return ChannelLayout(
        (Synapse("ExpSyn", name=polarity, e=e),),
        indices,
        indices,
        indices,
        (0,) * n_channels,
    )


class ModelBoundaryTest(unittest.TestCase):
    def test_exposes_sequence_and_step_implementations(self):
        self.assertIn("DBNN", model.__all__)
        self.assertNotIn("dbnn_step", model.__all__)
        self.assertNotIn("dbnn_gif_f_state", model.__all__)
        self.assertNotIn("dbnn_gif_step", model.__all__)

    def test_package_does_not_export_numerical_kernels(self):
        import braincell.reduction.dbnn as dbnn_package

        self.assertNotIn("dbnn_forward", dbnn_package.__all__)
        self.assertNotIn("dbnn_step", dbnn_package.__all__)

    def test_detect_spike_times_uses_up_crossings(self):
        actual = model.detect_spike_times([-40.0, -60.0, -40.0, -30.0, -60.0], -50.0)
        np.testing.assert_array_equal(actual, [0, 2])

    def test_detect_spike_times_rejects_empty_trace(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            model.detect_spike_times([], -50.0)

    def test_model_predict_and_update_are_equivalent(self):
        dbnn = model.DBNN(_layout(2), mode="r", dt=0.25 * u.ms)
        inputs = jnp.asarray([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]], dtype=jnp.float32)
        expected = dbnn.predict(inputs)
        dbnn.init_state(batch_size=1)

        def step(events):
            return dbnn.update(events)

        voltage_tb, spike_tb = brainstate.transform.for_loop(step, jnp.transpose(inputs, (2, 0, 1)))
        np.testing.assert_allclose(voltage_tb.T, expected["voltage"], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(spike_tb.T, expected["spike"])

    def test_parameter_validation_is_strict(self):
        dbnn = model.DBNN(_layout(2), dt=0.1 * u.ms)
        params = dbnn.get_params()
        params.pop("dt_ms")
        params["tau_rise"] = jnp.asarray([-1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "time constants"):
            dbnn.set_params(params)

    def test_dt_requires_units(self):
        with self.assertRaisesRegex(TypeError, "brainunit"):
            model.DBNN(_layout(1), dt=0.1)

    def test_quadratic_parameters_follow_actual_sparse_channels(self):
        channel_layout = ChannelLayout(
            (Synapse("ExpSyn", name="E"),),
            (0, 1, 2),
            (0, 1, 2),
            (2, 40, 99),
            (0, 0, 0),
        )
        dbnn = model.DBNN(channel_layout, dt=0.1 * u.ms)
        self.assertEqual(dbnn.n_channels, 3)
        self.assertEqual(dbnn.quadratic_weight_upper.value.shape, (3,))

    def test_default_sign_mode_applies_layout_inhibitory_polarity(self):
        dbnn = model.DBNN(_layout(1, polarity="I"), dt=1.0 * u.ms)
        inputs = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32)
        voltage = np.asarray(dbnn.predict(inputs)["voltage"])
        self.assertEqual(float(inputs[0, 0, 0]), 1.0)
        self.assertLess(voltage[0, 1], float(dbnn.bias.value))

    def test_none_sign_mode_disables_layout_polarity_encoding(self):
        dbnn = model.DBNN(_layout(1, polarity="I"), input_sign_mode="none", dt=1.0 * u.ms)
        voltage = dbnn.predict(jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float32))["voltage"]
        self.assertGreater(float(voltage[0, 1]), float(dbnn.bias.value))

    def test_omega_is_non_negative_and_bounded(self):
        dbnn = model.DBNN(_layout(1), dt=1.0 * u.ms)
        params = dbnn.get_params()
        params["omega"] = jnp.asarray([-0.1], dtype=jnp.float32)
        with self.assertRaisesRegex(ValueError, "omega"):
            dbnn.set_params(params)

    def test_negative_events_are_rejected_for_all_polarities(self):
        dbnn = model.DBNN(_layout(1, polarity="I"), dt=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dbnn.predict(jnp.asarray([[[-1.0, 0.0]]], dtype=jnp.float32))

    def test_gif_model_predict_and_update_are_equivalent(self):
        dbnn = model.DBNNGIF(_layout(2), mode="r", dt=0.25 * u.ms)
        dbnn.v_th.value = jnp.asarray(-69.0)
        inputs = jnp.asarray([[[1.0, 0.0, 0.5], [0.0, 1.0, 0.0]]], dtype=jnp.float32)
        expected = dbnn.predict(inputs)
        dbnn.init_state(batch_size=1)

        def step(events):
            return dbnn.update(events)

        (voltage_tb, threshold_tb, spike_tb) = brainstate.transform.for_loop(step, jnp.transpose(inputs, (2, 0, 1)))
        np.testing.assert_allclose(voltage_tb.T, expected["voltage"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(threshold_tb.T, expected["threshold"], rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(spike_tb.T, expected["spike"])

    def test_gif_complete_sequence_prediction_is_jittable_in_both_modes(self):
        inputs = jnp.asarray([[[0.0, 1.0, 0.0]]], dtype=jnp.float32)
        for mode_name in ("f", "r"):
            with self.subTest(mode=mode_name):
                dbnn = model.DBNNGIF(_layout(1), mode=mode_name, dt=1.0 * u.ms)
                compiled = jax.jit(lambda values: dbnn.predict(values)["voltage"])
                np.testing.assert_allclose(compiled(inputs), dbnn.predict(inputs)["voltage"])

    def test_gif_parameters_are_strictly_validated(self):
        dbnn = model.DBNNGIF(_layout(1), dt=1.0 * u.ms)
        params = dbnn.get_params()
        params["threshold_tau_ms"] = jnp.asarray(0.0)
        with self.assertRaisesRegex(ValueError, "threshold time constant"):
            dbnn.set_params(params)
        params = dbnn.get_params()
        params["threshold_increment_mv"] = jnp.asarray(-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            dbnn.set_params(params)

    def test_gif_spike_time_offset_must_be_finite(self):
        dbnn = model.DBNNGIF(_layout(1), dt=1.0 * u.ms)
        self.assertEqual(dbnn.spike_time_offset_ms, 0.0)
        dbnn.set_spike_time_offset(6.28)
        self.assertAlmostEqual(dbnn.spike_time_offset_ms, 6.28)
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                dbnn.set_spike_time_offset(invalid)

    def test_align_spike_times_shifts_only_times_and_keeps_closed_interval(self):
        aligned = model.align_spike_times(
            (np.asarray([1.0, 6.28, 11.28]),),
            offset_ms=6.28,
            duration_ms=5.0,
        )
        np.testing.assert_allclose(aligned[0], [0.0, 5.0])

    def test_align_spike_times_clamps_roundoff_at_closed_endpoints(self):
        aligned = model.align_spike_times(
            (np.asarray([6.28 - 1e-7, 11.28 + 1e-7]),),
            offset_ms=6.28,
            duration_ms=5.0,
        )
        np.testing.assert_allclose(aligned[0], [0.0, 5.0])

    def test_gif_adds_only_dynamic_threshold_parameters(self):
        dbnn = model.DBNNGIF(_layout(1), dt=1.0 * u.ms)
        self.assertEqual(
            set(dbnn.get_params()) - set(model.DBNN(_layout(1), dt=1.0 * u.ms).get_params()),
            {"threshold_increment_mv", "threshold_tau_ms"},
        )
        self.assertFalse(hasattr(dbnn, "reset_amp"))
        self.assertFalse(hasattr(dbnn, "tau_reset"))

    def test_gif_reset_is_explicitly_enabled_and_disabled(self):
        dbnn = model.DBNNGIF(_layout(1), dt=1.0 * u.ms)
        dbnn.enable_reset(amplitude_mv=4.0, tau_ms=5.0)
        self.assertTrue(dbnn.reset_enabled)
        self.assertAlmostEqual(float(dbnn.reset_amp.value), 4.0)
        self.assertEqual(
            set(dbnn.get_params()) - set(model.DBNN(_layout(1), dt=1.0 * u.ms).get_params()),
            {"threshold_increment_mv", "threshold_tau_ms", "reset_amp", "tau_reset"},
        )
        dbnn.disable_reset()
        self.assertFalse(dbnn.reset_enabled)
        self.assertFalse(hasattr(dbnn, "reset_amp"))

    def test_update_requires_recurrent_mode(self):
        dbnn = model.DBNN(_layout(1), mode="f", dt=1.0 * u.ms)
        dbnn.init_state()
        with self.assertRaisesRegex(RuntimeError, "mode='r'"):
            dbnn.update(jnp.zeros((1, 1), dtype=jnp.float32))
