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

"""Known waveform errors, strict time alignment and spike comparisons."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .analysis import align_traces, analyze, save_analysis, spike_times
from .results import ComparisonResult, Trace


class AnalysisTest(unittest.TestCase):
    def test_initial_point_alignment_and_known_errors(self):
        result = ComparisonResult(Trace([0, 1, 2, 3], [-9, 1, 2, 3]), Trace([1, 2, 3], [2, 0, 5]), {"model": "toy"})
        analysis = analyze(result)
        np.testing.assert_array_equal(analysis.error_mV, [1, -2, 2])
        self.assertAlmostEqual(analysis.summary["voltage"]["rmse_mV"], np.sqrt(3))
        self.assertAlmostEqual(analysis.summary["voltage"]["mae_mV"], 5 / 3)
        self.assertEqual(analysis.summary["voltage"]["max_abs_mV"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            save_analysis(analysis, tmp)
            self.assertEqual(json.loads((Path(tmp) / "metrics.json").read_text()), analysis.summary)
            with np.load(Path(tmp) / "aligned.npz") as data:
                np.testing.assert_array_equal(data["error_mV"], [1, -2, 2])

    def test_equal_lengths_do_not_hide_time_shift(self):
        for times in ([0.5, 1.5], [1, 3], [0, 1, 2, 3]):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "Sample times"):
                align_traces(Trace(times, np.zeros(len(times))), Trace([1, 2], [0, 0]))

    def test_extra_noninitial_sample_is_not_discarded(self):
        with self.assertRaises(ValueError):
            align_traces(Trace([0.5, 1, 2], [0, 0, 0]), Trace([1, 2], [0, 0]))

    def test_fixed_step_roundoff_is_tolerated(self):
        time, _, _ = align_traces(Trace([0, 1 + 1e-12, 2], [0, 0, 0]), Trace([1, 2], [0, 0]))
        np.testing.assert_array_equal(time, [1, 2])

    def test_spike_interpolation_plateau_and_falling_crossing(self):
        trace = Trace([0, 1, 2, 3, 4, 5], [-2, 2, 0, -2, 0, 0])
        np.testing.assert_array_equal(spike_times(trace), [0.5, 4])
        self.assertEqual(len(spike_times(trace, threshold_mV=3)), 0)

    def test_equal_spike_counts_pair_in_temporal_order(self):
        result = ComparisonResult(Trace([0, 1, 2], [-2, 2, -2]), Trace([0, 1, 2], [-3, 1, -2]), {})
        spikes = analyze(result).summary["spikes"]
        self.assertTrue(spikes["paired_by_order"])
        np.testing.assert_allclose(spikes["time_error_ms"], [0.25])

    def test_unequal_spike_counts_remain_unpaired(self):
        result = ComparisonResult(Trace([0, 1, 2], [-2, 2, -2]), Trace([0, 1, 2], [-2, -1, -2]), {})
        spikes = analyze(result).summary["spikes"]
        self.assertEqual((spikes["neuron_count"], spikes["braincell_count"]), (1, 0))
        self.assertFalse(spikes["paired_by_order"])
        self.assertIsNone(spikes["time_error_ms"])

    def test_no_spikes_and_common_window(self):
        result = ComparisonResult(Trace([0, 1, 2], [-2, 2, 1]), Trace([1, 2], [2, 1]), {})
        spikes = analyze(result).summary["spikes"]
        self.assertEqual(spikes["neuron_count"], 0)
        self.assertEqual(spikes["braincell_count"], 0)
        self.assertEqual(spikes["time_error_ms"], [])

    def test_nonfinite_voltages_cannot_produce_accuracy_metrics(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            analyze(ComparisonResult(Trace([1], [0]), Trace([1], [np.nan]), {}))
