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

"""Tests for :mod:`braincell.io.neuromorpho.models`."""

import unittest

from braincell.io.neuromorpho import (
    NeuroMorphoMeasurement,
    NeuroMorphoNeuron,
)
from braincell.io.neuromorpho._testing import sample_neuron_payload


class NeuroMorphoNeuronTest(unittest.TestCase):
    def test_from_payload_promotes_common_fields(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        self.assertEqual(neuron.neuron_id, 10047)
        self.assertEqual(neuron.neuron_name, "TypeA-10")
        self.assertEqual(neuron.archive, "Scanziani")
        self.assertEqual(neuron.species, "mouse")
        self.assertEqual(neuron.original_format, "TypeA-10.asc")
        self.assertEqual(neuron.brain_region, ["neocortex", "occipital", "layer 6"])
        self.assertEqual(neuron.cell_type, ["principal cell"])

    def test_from_payload_normalizes_string_to_list(self) -> None:
        payload = sample_neuron_payload(brain_region="hippocampus", cell_type="basket")
        neuron = NeuroMorphoNeuron.from_payload(payload)
        self.assertEqual(neuron.brain_region, ["hippocampus"])
        self.assertEqual(neuron.cell_type, ["basket"])

    def test_payload_field_preserves_full_record(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        # The raw HAL JSON is still accessible for callers who need
        # fields that the dataclass does not promote.
        self.assertIn("_links", neuron.payload)


class NeuroMorphoMeasurementTest(unittest.TestCase):
    def test_promotes_known_fields(self) -> None:
        payload = {
            "neuron_id": 10047,
            "n_stems": 2.0,
            "n_branch": 5.0,
            "n_bifs": 3.0,
            "length": 123.4,
            "surface": 567.89,
            "volume": 42.0,
            "pathDistance": 88.0,
            "eucDistance": 77.0,
            "soma_Surface": 12.5,
            "fancy_extra": "kept",
        }
        meas = NeuroMorphoMeasurement.from_payload(payload)
        self.assertEqual(meas.neuron_id, 10047)
        self.assertEqual(meas.n_stems, 2)
        self.assertEqual(meas.n_branch, 5)
        self.assertEqual(meas.n_bifs, 3)
        self.assertEqual(meas.length, 123.4)
        self.assertEqual(meas.surface, 567.89)
        self.assertEqual(meas.volume, 42.0)
        self.assertEqual(meas.path_distance, 88.0)
        self.assertEqual(meas.euclidean_distance, 77.0)
        self.assertEqual(meas.soma_surface, 12.5)
        self.assertEqual(meas.extras["fancy_extra"], "kept")

    def test_defaults_missing_fields_to_none(self) -> None:
        meas = NeuroMorphoMeasurement.from_payload({"neuron_id": 1})
        self.assertIsNone(meas.length)
        self.assertIsNone(meas.n_stems)

    def test_requires_neuron_id(self) -> None:
        with self.assertRaises(ValueError):
            NeuroMorphoMeasurement.from_payload({"n_stems": 1.0})

    def test_get_falls_through_to_extras(self) -> None:
        meas = NeuroMorphoMeasurement.from_payload({"neuron_id": 1, "weird_key": 9})
        self.assertEqual(meas.get("weird_key"), 9)
        self.assertEqual(meas.get("nope", "default"), "default")
        # Promoted attribute lookup also works through .get().
        self.assertIsNone(meas.get("length"))

    def test_as_dict_includes_promoted_and_extras(self) -> None:
        meas = NeuroMorphoMeasurement.from_payload({
            "neuron_id": 1,
            "length": 10.0,
            "rare": "value",
        })
        flat = meas.as_dict()
        self.assertEqual(flat["length"], 10.0)
        self.assertEqual(flat["rare"], "value")
        self.assertEqual(flat["neuron_id"], 1)

    def test_raw_preserves_original_key_spellings(self) -> None:
        payload = {
            "neuron_id": 1,
            "pathDistance": 99.0,
            "soma_Surface": 12.5,
        }
        meas = NeuroMorphoMeasurement.from_payload(payload)
        # The promoted snake_case attribute is set...
        self.assertEqual(meas.path_distance, 99.0)
        self.assertEqual(meas.soma_surface, 12.5)
        # ...but the raw view keeps the upstream camelCase keys verbatim.
        self.assertEqual(meas.raw["pathDistance"], 99.0)
        self.assertEqual(meas.raw["soma_Surface"], 12.5)

    def test_int_fields_round_floats(self) -> None:
        meas = NeuroMorphoMeasurement.from_payload({
            "neuron_id": 1,
            "n_stems": 2.7,
            "n_branch": "5",
        })
        self.assertEqual(meas.n_stems, 3)
        self.assertEqual(meas.n_branch, 5)


if __name__ == "__main__":
    unittest.main()
