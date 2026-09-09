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

"""Portable trace validation and lossless result round trips."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from .results import ComparisonResult, Trace, load_result, save_result


class ResultsTest(unittest.TestCase):
    def test_round_trip_keeps_raw_initial_point_and_nonfinite_values(self):
        result = ComparisonResult(
            Trace([0, 1, 2], [-65, -64, -63]),
            Trace([1, 2], [-64, np.nan]),
            {"format_version": 1, "protocol": {"precision": 32}},
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_result(result, tmp)
            loaded = load_result(tmp)
            np.testing.assert_array_equal(loaded.neuron.time_ms, [0, 1, 2])
            np.testing.assert_array_equal(loaded.braincell.voltage_mV, [-64, np.nan])
            self.assertEqual(loaded.metadata, result.metadata)
            self.assertEqual(loaded.output_dir, Path(tmp))
            with self.assertRaises(FileExistsError):
                save_result(result, tmp)

    def test_trace_rejects_mismatched_or_invalid_times(self):
        for times, voltage in (
            ([], []),
            ([0, 1], [0]),
            ([1, 1], [0, 0]),
            ([1, 0], [0, 0]),
            ([0, np.nan], [0, 0]),
            ([[0]], [[1]]),
        ):
            with self.subTest(times=times), self.assertRaises(ValueError):
                Trace(times, voltage)

    def test_trace_owns_immutable_arrays(self):
        source = np.array([0.0, 1.0])
        trace = Trace(source, [1, 2])
        source[0] = -5
        np.testing.assert_array_equal(trace.time_ms, [0, 1])
        with self.assertRaises(ValueError):
            trace.voltage_mV[0] = 9

    def test_unknown_result_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "metadata.json").write_text('{"format_version": 99}')
            with self.assertRaisesRegex(ValueError, "format_version"):
                load_result(tmp)
