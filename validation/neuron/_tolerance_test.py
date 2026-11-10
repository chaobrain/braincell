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

"""Comparison metrics and the tiered tolerance table."""

import unittest

import numpy as np

from validation.neuron._tolerance import (
    T0_LIMITS,
    ToleranceEntry,
    ToleranceTable,
    max_abs_error,
    relative_current_error,
)


class MetricTest(unittest.TestCase):
    def test_max_abs_error_spans_every_pair(self):
        pairs = [([0.0, 1.0], [0.0, 1.5]), ([2.0], [-1.0])]
        self.assertEqual(max_abs_error(pairs), 3.0)

    def test_current_error_is_normalized_by_the_protocol_maximum(self):
        # A barely activated command must not dominate through its own small scale.
        quiet = (np.array([1e-9 + 1e-12]), np.array([1e-9]))
        active = (np.array([2.0, -4.0 + 4e-6]), np.array([2.0, -4.0]))
        self.assertAlmostEqual(relative_current_error([quiet, active]), 1e-6, places=15)

    def test_metrics_reject_mismatched_or_empty_input(self):
        with self.assertRaises(ValueError):
            max_abs_error([([1.0, 2.0], [1.0])])
        with self.assertRaises(ValueError):
            max_abs_error([])
        with self.assertRaises(ValueError):
            relative_current_error([([0.0], [0.0])])


class ToleranceTableTest(unittest.TestCase):
    def test_unlisted_configurations_and_metrics_are_t0(self):
        table = ToleranceTable([ToleranceEntry("a/x", "vc_steps", {"e_I": 2e-4}, "d", "Section")])
        self.assertEqual(table.tier("a/x", "vc_steps"), "T1")
        self.assertEqual(table.limit("a/x", "vc_steps", "e_I"), 2e-4)
        self.assertEqual(table.limit("a/x", "vc_steps", "e_s"), T0_LIMITS["e_s"])
        self.assertEqual(table.tier("a/x", "vc_ramp"), "T0")
        self.assertEqual(table.limit("a/y", "dc", "e_V"), 1e-6)
        self.assertEqual(T0_LIMITS, {"e_V": 1e-6, "e_I": 1e-10, "e_s": 1e-10})
        with self.assertRaises(ValueError):
            table.limit("a/x", "dc", "e_I")

    def test_invalid_entries_are_rejected(self):
        invalid = (
            dict(protocol="unknown"),
            dict(deviation_class="a"),
            dict(deviation_class="b"),
            dict(section=" "),
            dict(limits={}),
            dict(limits={"e_V": 1.0}),
            dict(limits={"e_I": 1e-11}),
            dict(tier="T0"),
        )
        valid = dict(config="a/x", protocol="vc_steps", limits={"e_I": 1e-4}, deviation_class="c", section="S")
        ToleranceEntry(**valid)
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ToleranceEntry(**{**valid, **change})
        entry = ToleranceEntry(**valid)
        with self.assertRaises(ValueError):
            ToleranceTable([entry, entry])

    def test_missing_sections(self):
        table = ToleranceTable(
            [
                ToleranceEntry("a/x", "dc", {"e_V": 1e-3}, "c", "First cause"),
                ToleranceEntry("a/x", "ac", {"e_V": 1e-3}, "d", "Second cause"),
            ]
        )
        document = "# Title\n\n### First cause\n\nText naming Second cause.\n"
        self.assertEqual(table.missing_sections(document), ["Second cause"])


if __name__ == "__main__":
    unittest.main()
