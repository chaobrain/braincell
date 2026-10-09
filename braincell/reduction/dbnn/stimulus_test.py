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

import brainunit as u
import numpy as np

from braincell.reduction.dbnn import layout, stimulus
from braincell.mech import Synapse


def _layout(*, reference_weight_us=1.0):
    specs = (
        Synapse("ExpSyn", name="E"),
        Synapse("ExpSyn", name="I", e=-80.0 * u.mV),
    )
    return layout.ChannelLayout(specs, (0, 1), (0, 1), (0, 0), (0, 1), (reference_weight_us, 1.0))


def _large_layout(n_channels):
    indices = tuple(range(n_channels))
    return layout.ChannelLayout((Synapse("ExpSyn", name="E"),), indices, indices, indices, (0,) * n_channels)


class StimulusBoundaryTest(unittest.TestCase):
    def test_multichannel_protocol_can_preserve_zero_rate_channels(self):
        channel_layout = _layout()
        plan = stimulus.generate_multichannel_protocol(
            channel_layout,
            n_traces=2,
            duration_ms=4.0,
            dt_ms=1.0,
            rate_hz=0.0,
            ensure_channel_coverage=False,
        )
        self.assertEqual(len(plan.events.trace_id), 0)

        with self.assertRaises(TypeError):
            stimulus.generate_multichannel_protocol(
                channel_layout,
                n_traces=2,
                duration_ms=4.0,
                dt_ms=1.0,
                rate_hz=0.0,
                ensure_channel_coverage=1,
            )

    def test_multichannel_protocol_accepts_per_channel_amplitudes(self):
        plan = stimulus.generate_multichannel_protocol(
            _layout(),
            n_traces=2,
            duration_ms=4.0,
            dt_ms=1.0,
            rate_hz=1000.0,
            amplitude=np.asarray([0.25, 0.75]),
            ensure_channel_coverage=False,
        )
        np.testing.assert_allclose(plan.channel_weights, [[0.25, 0.75], [0.25, 0.75]])

    def test_multichannel_trace_seeds_stay_inside_supported_range(self):
        plan = stimulus.generate_multichannel_protocol(
            _layout(),
            n_traces=3,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=0.0,
            seed=2**31 - 2,
        )
        self.assertEqual(len(set(plan.seeds)), 3)
        self.assertTrue(all(0 <= seed < 2**31 - 1 for seed in plan.seeds))

    def test_module_is_importable(self):
        self.assertIn("stimulus", stimulus.__doc__)

    def test_pair_protocol_reports_ei_coverage(self):
        plan = stimulus.generate_pair_protocol(_layout(), repetitions=2)
        report = stimulus.measure_coverage(plan, _layout(), target=2)
        np.testing.assert_array_equal(report.channel_count, [2, 2])
        np.testing.assert_array_equal(report.pair_count, [2])
        self.assertEqual(report.pair_class_count, {"EE": 0, "EI": 2, "II": 0})
        self.assertEqual(report.undercovered_pairs, ())
        self.assertEqual(report.channel_coverage_fraction, 1.0)
        self.assertEqual(report.pair_coverage_fraction, 1.0)
        self.assertEqual(report.protocol_channel_coverage, {"pair": 1.0})
        self.assertTrue(report.meets_target)

    def test_multichannel_protocol_is_reproducible(self):
        first = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=2, duration_ms=10.0, dt_ms=1.0, rate_hz=100.0, seed=4
        )
        second = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=2, duration_ms=10.0, dt_ms=1.0, rate_hz=100.0, seed=4
        )
        np.testing.assert_array_equal(first.events.trace_id, second.events.trace_id)
        np.testing.assert_array_equal(first.events.channel_id, second.events.channel_id)
        np.testing.assert_array_equal(first.events.time_ms, second.events.time_ms)

    def test_same_seed_generates_independent_dataset_splits(self):
        train_plan = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=2, duration_ms=20.0, dt_ms=1.0, rate_hz=500.0, seed=4, split="train"
        )
        validation_plan = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=2, duration_ms=20.0, dt_ms=1.0, rate_hz=500.0, seed=4, split="validation"
        )
        self.assertFalse(np.array_equal(train_plan.events.time_ms, validation_plan.events.time_ms))

    def test_pair_protocol_uses_requested_training_size_and_reports_undercoverage(self):
        channel_layout = _large_layout(50)
        plan = stimulus.generate_pair_protocol(channel_layout, n_traces=1000)
        report = stimulus.measure_coverage(plan, channel_layout)
        self.assertEqual(plan.n_traces, 1000)
        self.assertAlmostEqual(report.pair_coverage_fraction, 1000.0 / 1225.0)
        self.assertEqual(len(report.undercovered_pairs), 225)
        self.assertEqual(report.channel_coverage_fraction, 1.0)
        self.assertTrue(np.all(report.channel_count >= 1))
        self.assertFalse(report.meets_target)

    def test_default_protocol_combines_modes_without_exceeding_limit(self):
        channel_layout = _large_layout(50)
        plan = stimulus.generate_default_protocol(channel_layout)
        self.assertEqual(plan.n_traces, stimulus.DEFAULT_TRAINING_TRACES)
        self.assertIn("pair", plan.protocol_labels)
        self.assertIn("multichannel", plan.protocol_labels)
        self.assertEqual(plan.protocol_labels.count("pair"), 200)
        self.assertEqual(plan.protocol_labels.count("multichannel"), 800)
        report = stimulus.measure_coverage(plan, channel_layout)
        self.assertEqual(report.channel_coverage_fraction, 1.0)
        self.assertEqual(report.protocol_channel_coverage["pair"], 1.0)
        self.assertEqual(report.protocol_channel_coverage["multichannel"], 1.0)

    def test_multichannel_protocol_deterministically_fills_missing_channels(self):
        channel_layout = _large_layout(50)
        plan = stimulus.generate_multichannel_protocol(
            channel_layout,
            n_traces=4,
            duration_ms=10.0,
            dt_ms=1.0,
            rate_hz=0.0,
            seed=8,
        )
        report = stimulus.measure_coverage(plan, channel_layout)
        self.assertEqual(report.protocol_channel_coverage["multichannel"], 1.0)
        self.assertTrue(np.all(report.channel_count >= 1))

    def test_pair_protocol_rejects_budget_that_cannot_cover_every_channel(self):
        with self.assertRaisesRegex(ValueError, "requires at least 25 traces"):
            stimulus.generate_pair_protocol(_large_layout(50), n_traces=24)

    def test_default_rejects_budget_without_room_for_both_rules(self):
        with self.assertRaisesRegex(ValueError, "additional multichannel trace"):
            stimulus.generate_default_protocol(_large_layout(50), n_traces=25)

    def test_explicit_protocols_allow_caller_selected_size_above_default(self):
        plan = stimulus.generate_multichannel_protocol(
            _layout(),
            n_traces=1001,
            duration_ms=10.0,
            dt_ms=1.0,
            rate_hz=10.0,
        )
        self.assertEqual(plan.n_traces, 1001)

    def test_combination_preserves_caller_selected_size_above_default(self):
        first = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=600, duration_ms=2.0, dt_ms=1.0, rate_hz=10.0
        )
        second = stimulus.generate_multichannel_protocol(
            _layout(), n_traces=500, duration_ms=2.0, dt_ms=1.0, rate_hz=10.0, seed=1000
        )
        combined = stimulus.combine_protocols(first, second)
        self.assertEqual(combined.n_traces, 1100)

    def test_same_channel_count_does_not_allow_different_layout_semantics(self):
        first_layout = _layout(reference_weight_us=1.0)
        other_layout = _layout(reference_weight_us=2.0)
        plan = stimulus.generate_multichannel_protocol(
            first_layout,
            n_traces=1,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=10.0,
        )
        self.assertEqual(plan.n_channels, other_layout.n_channels)
        with self.assertRaisesRegex(ValueError, "different channel layout"):
            stimulus.measure_coverage(plan, other_layout)

    def test_combine_rejects_plans_from_different_layouts(self):
        first = stimulus.generate_multichannel_protocol(
            _layout(reference_weight_us=1.0),
            n_traces=1,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=10.0,
        )
        second = stimulus.generate_multichannel_protocol(
            _layout(reference_weight_us=2.0),
            n_traces=1,
            duration_ms=2.0,
            dt_ms=1.0,
            rate_hz=10.0,
        )
        with self.assertRaisesRegex(ValueError, "same channel layout"):
            stimulus.combine_protocols(first, second)

    def test_larger_training_size_makes_large_channel_layout_feasible(self):
        channel_layout = _large_layout(3000)
        with self.assertRaisesRegex(ValueError, "requires 1500 pair traces"):
            stimulus.generate_default_protocol(channel_layout, n_traces=1000)
        plan = stimulus.generate_default_protocol(channel_layout, n_traces=1502)
        self.assertEqual(plan.n_traces, 1502)
        self.assertEqual(plan.protocol_labels.count("pair"), 1500)
        self.assertEqual(plan.protocol_labels.count("multichannel"), 2)
        pair_channels = np.unique(plan.events.channel_id[plan.events.trace_id < 1500])
        multichannel_channels = np.unique(plan.events.channel_id[plan.events.trace_id >= 1500])
        np.testing.assert_array_equal(pair_channels, np.arange(3000))
        np.testing.assert_array_equal(multichannel_channels, np.arange(3000))
