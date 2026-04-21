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

"""Tests for :mod:`braincell.io.neuromorpho.urls`."""

import unittest

from braincell.io.neuromorpho import (
    NeuroMorphoNeuron,
    build_original_file_url,
    build_standard_swc_url,
    plan_neuron_files,
)
from braincell.io.neuromorpho._testing import sample_neuron_payload
from braincell.io.neuromorpho.urls import (
    build_measurement_url,
    coerce_https,
    infer_original_extension,
    safe_filename,
)


class SafeFilenameTest(unittest.TestCase):
    def test_strips_unsafe_chars_to_underscore(self) -> None:
        self.assertEqual(safe_filename("Type A/10 (mouse)"), "Type_A_10_mouse")

    def test_collapses_runs(self) -> None:
        self.assertEqual(safe_filename("a   b"), "a_b")

    def test_strips_leading_and_trailing_punct(self) -> None:
        self.assertEqual(safe_filename("__abc__"), "abc")

    def test_falls_back_to_default_when_empty(self) -> None:
        self.assertEqual(safe_filename("..."), "neuromorpho_neuron")


class CoerceHttpsTest(unittest.TestCase):
    def test_upgrades_http(self) -> None:
        self.assertEqual(coerce_https("http://example.org/x"), "https://example.org/x")

    def test_passes_https_through(self) -> None:
        self.assertEqual(coerce_https("https://example.org/x"), "https://example.org/x")


class BuildStandardSwcUrlTest(unittest.TestCase):
    def test_uses_archive_and_neuron_name(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        url = build_standard_swc_url(neuron)
        self.assertIn("CNG%20version", url)
        self.assertTrue(url.endswith("/TypeA-10.CNG.swc"))
        self.assertIn("/scanziani/", url)

    def test_raises_when_archive_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(archive=None))
        with self.assertRaises(ValueError):
            build_standard_swc_url(neuron)


class BuildOriginalFileUrlTest(unittest.TestCase):
    def test_returns_url_when_metadata_complete(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        url = build_original_file_url(neuron)
        self.assertIsNotNone(url)
        self.assertIn("Source-Version", url)
        self.assertTrue(url.endswith(".asc"))

    def test_returns_none_when_format_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(original_format=None))
        self.assertIsNone(build_original_file_url(neuron))

    def test_returns_none_when_archive_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(archive=None))
        self.assertIsNone(build_original_file_url(neuron))


class InferOriginalExtensionTest(unittest.TestCase):
    def test_returns_extension(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        self.assertEqual(infer_original_extension(neuron), ".asc")

    def test_raises_when_format_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(original_format=None))
        with self.assertRaises(ValueError):
            infer_original_extension(neuron)

    def test_raises_when_format_lacks_extension(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(original_format="naked"))
        with self.assertRaises(ValueError):
            infer_original_extension(neuron)


class BuildMeasurementUrlTest(unittest.TestCase):
    def test_uses_link_when_present(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        url = build_measurement_url(neuron)
        # http link in the payload is upgraded to https.
        self.assertEqual(url, "https://neuromorpho.org/api/morphometry/id/10047")

    def test_falls_back_to_canonical_route(self) -> None:
        payload = sample_neuron_payload()
        payload["_links"] = {}
        neuron = NeuroMorphoNeuron.from_payload(payload)
        url = build_measurement_url(neuron)
        self.assertEqual(url, "https://neuromorpho.org/api/morphometry/id/10047")


class PlanNeuronFilesTest(unittest.TestCase):
    def test_standard_only(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        plans = plan_neuron_files(neuron, mode="standard")
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].kind, "standard")
        self.assertFalse(plans[0].skip)

    def test_both_returns_two_plans(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        plans = plan_neuron_files(neuron, mode="both")
        self.assertEqual([p.kind for p in plans], ["standard", "original"])
        self.assertFalse(plans[0].skip)
        self.assertFalse(plans[1].skip)

    def test_skips_original_when_format_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(original_format=None))
        plans = plan_neuron_files(neuron, mode="both")
        self.assertEqual(len(plans), 2)
        self.assertFalse(plans[0].skip)
        self.assertTrue(plans[1].skip)
        self.assertIn("original_format", plans[1].reason)

    def test_rejects_unknown_mode(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        with self.assertRaises(ValueError):
            plan_neuron_files(neuron, mode="weird")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
