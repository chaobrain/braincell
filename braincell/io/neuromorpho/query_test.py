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

"""Tests for :mod:`braincell.io.neuromorpho.query`."""

import unittest

from braincell.io.neuromorpho import NeuroMorphoQuery


class NeuroMorphoQueryTest(unittest.TestCase):
    def test_default_is_match_all(self) -> None:
        self.assertEqual(NeuroMorphoQuery().to_q(), "*:*")
        self.assertEqual(NeuroMorphoQuery().to_fq(), [])

    def test_combines_typed_fields_with_and(self) -> None:
        q = NeuroMorphoQuery(species="mouse", brain_region="cerebellum")
        self.assertEqual(q.to_q(), "species:mouse AND brain_region:cerebellum")

    def test_multivalue_field_uses_or(self) -> None:
        self.assertEqual(
            NeuroMorphoQuery(species=("mouse", "rat")).to_q(),
            "(species:mouse OR species:rat)",
        )

    def test_raw_q_appended_verbatim(self) -> None:
        q = NeuroMorphoQuery(species="mouse", raw_q=("foo:bar",))
        self.assertEqual(q.to_q(), "species:mouse AND foo:bar")

    def test_raw_fq_passes_through_to_fq(self) -> None:
        q = NeuroMorphoQuery(raw_fq=("custom:thing", "another:value"))
        self.assertEqual(q.to_fq(), ["custom:thing", "another:value"])

    def test_to_params_packages_q_and_fq(self) -> None:
        q = NeuroMorphoQuery(
            species="mouse",
            brain_region=("cerebellum", "neocortex"),
            raw_fq=("custom:thing",),
        )
        params = q.to_params()
        self.assertIn("species:mouse", params["q"])
        self.assertIn("brain_region:cerebellum OR brain_region:neocortex", params["q"])
        self.assertEqual(params["fq"], ["custom:thing"])

    def test_handles_all_typed_fields(self) -> None:
        q = NeuroMorphoQuery(
            species="mouse",
            brain_region="cerebellum",
            cell_type="purkinje",
            archive="someone",
            original_format="asc",
            stain="biocytin",
            age_classification="adult",
            gender="male",
        )
        rendered = q.to_q()
        for clause in (
            "species:mouse",
            "brain_region:cerebellum",
            "cell_type:purkinje",
            "archive:someone",
            "original_format:asc",
            "stain:biocytin",
            "age_classification:adult",
            "gender:male",
        ):
            self.assertIn(clause, rendered)


if __name__ == "__main__":
    unittest.main()
