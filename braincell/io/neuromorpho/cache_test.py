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

"""Tests for :mod:`braincell.io.neuromorpho.cache`."""

import json
import tempfile
import unittest
from pathlib import Path

from braincell import Morphology
from braincell.io.neuromorpho import (
    NeuroMorphoCache,
    NeuroMorphoCacheLayout,
)
from braincell.io.neuromorpho._testing import FIXTURE_SWC


class NeuroMorphoCacheLayoutTest(unittest.TestCase):
    def test_neuron_dir(self) -> None:
        layout = NeuroMorphoCacheLayout(root=Path("/tmp/nm"))
        self.assertEqual(layout.neuron_dir(10047), Path("/tmp/nm/10047"))

    def test_metadata_path(self) -> None:
        layout = NeuroMorphoCacheLayout(root=Path("/tmp/nm"))
        self.assertEqual(
            layout.metadata_path(10047),
            Path("/tmp/nm/10047/metadata.json"),
        )

    def test_standard_swc_path(self) -> None:
        layout = NeuroMorphoCacheLayout(root=Path("/tmp/nm"))
        self.assertEqual(
            layout.standard_swc_path(10047, "TypeA-10"),
            Path("/tmp/nm/10047/TypeA-10.CNG.swc"),
        )

    def test_original_file_path(self) -> None:
        layout = NeuroMorphoCacheLayout(root=Path("/tmp/nm"))
        self.assertEqual(
            layout.original_file_path(10047, "TypeA-10", ".asc"),
            Path("/tmp/nm/10047/TypeA-10.asc"),
        )


class NeuroMorphoCacheDiscoveryTest(unittest.TestCase):
    def test_list_neurons_round_trips_with_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            (cache.root / "10047").mkdir()
            (cache.root / "10048").mkdir()
            (cache.root / "not-an-id").mkdir()  # ignored
            self.assertEqual(cache.list_neurons(), (10047, 10048))
            self.assertTrue(cache.contains(10047))
            self.assertTrue(cache.remove(10047))
            self.assertFalse(cache.remove(10047))
            self.assertEqual(cache.list_neurons(), (10048,))

    def test_clear_removes_only_neuron_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            (cache.root / "1").mkdir()
            (cache.root / "2").mkdir()
            (cache.root / "keepme").mkdir()
            self.assertEqual(cache.clear(), 2)
            self.assertEqual(cache.list_neurons(), ())
            self.assertTrue((cache.root / "keepme").exists())

    def test_status_uses_metadata_for_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            folder = cache.root / "10047"
            folder.mkdir(parents=True)
            (folder / "TypeA-10.CNG.swc").write_text("swc", encoding="utf-8")
            (folder / "metadata.json").write_text(
                json.dumps({
                    "neuron_name": "TypeA-10",
                    "original_format": "TypeA-10.asc",
                }),
                encoding="utf-8",
            )
            status = cache.status(10047)
            self.assertTrue(status.exists)
            self.assertTrue(status.metadata_exists)
            self.assertTrue(status.standard_exists)
            self.assertFalse(status.original_exists)

    def test_status_when_folder_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            status = cache.status(10047)
            self.assertFalse(status.exists)
            self.assertFalse(status.metadata_exists)
            self.assertEqual(status.neuron_id, 10047)


class NeuroMorphoCacheReadTest(unittest.TestCase):
    def test_metadata_raises_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            with self.assertRaises(FileNotFoundError):
                cache.metadata(99999)

    def test_measurement_returns_typed_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            folder = cache.root / "10047"
            folder.mkdir(parents=True)
            (folder / "metadata.json").write_text(
                json.dumps({
                    "neuron_id": 10047,
                    "neuron_name": "TypeA-10",
                    "measurement": {"neuron_id": 10047, "n_stems": 1.0, "length": 10.0},
                }),
                encoding="utf-8",
            )
            meas = cache.measurement(10047)
            self.assertIsNotNone(meas)
            self.assertEqual(meas.n_stems, 1)
            self.assertEqual(meas.length, 10.0)

    def test_measurement_returns_none_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            self.assertIsNone(cache.measurement(99999))

    def test_standard_swc_path_uses_download_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            folder = cache.root / "10047"
            folder.mkdir(parents=True)
            swc = folder / "TypeA-10.CNG.swc"
            swc.write_text("swc", encoding="utf-8")
            (folder / "metadata.json").write_text(
                json.dumps({
                    "neuron_id": 10047,
                    "neuron_name": "TypeA-10",
                    "download_items": [
                        {
                            "kind": "standard",
                            "filename": "TypeA-10.CNG.swc",
                            "path": str(swc),
                        }
                    ],
                }),
                encoding="utf-8",
            )
            self.assertEqual(cache.standard_swc_path(10047), swc)

    def test_standard_swc_path_falls_back_to_glob(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            folder = cache.root / "10047"
            folder.mkdir(parents=True)
            swc = folder / "Random.CNG.swc"
            swc.write_text("swc", encoding="utf-8")
            self.assertEqual(cache.standard_swc_path(10047), swc)

    def test_load_returns_morphology(self) -> None:
        if not FIXTURE_SWC.exists():
            self.skipTest(f"missing fixture: {FIXTURE_SWC}")
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            folder = cache.root / "10047"
            folder.mkdir(parents=True)
            swc = folder / "TypeA-10.CNG.swc"
            swc.write_text(FIXTURE_SWC.read_text(encoding="utf-8"), encoding="utf-8")
            (folder / "metadata.json").write_text(
                json.dumps({"neuron_id": 10047, "neuron_name": "TypeA-10"}),
                encoding="utf-8",
            )
            morph = cache.load(10047)
            self.assertIsInstance(morph, Morphology)
            self.assertGreaterEqual(len(morph.branches), 1)

    def test_load_raises_when_no_swc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            with self.assertRaises(FileNotFoundError):
                cache.load(99999)


if __name__ == "__main__":
    unittest.main()
