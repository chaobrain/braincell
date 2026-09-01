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

"""Tests for :mod:`braincell.io.neuromorpho.entry`."""

import tempfile
import unittest
from pathlib import Path

from braincell import Morphology
from braincell.io.neuromorpho import (
    DEFAULT_USER_CACHE_DIR,
    NeuroMorphoClient,
    fetch_neuromorpho,
    load_neuromorpho,
)
from braincell.io.neuromorpho._testing import (
    FIXTURE_SWC,
    FakeResponse,
    FakeSession,
    sample_neuron_payload,
)


class LoadNeuromorphoTest(unittest.TestCase):
    def test_returns_morphology(self) -> None:
        session = FakeSession(
            [
                FakeResponse(json_data=sample_neuron_payload()),  # get_neuron
                FakeResponse(json_data={"n_stems": 1.0}),  # measurement
                FakeResponse(content=[FIXTURE_SWC.read_bytes()]),  # standard download
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NeuroMorphoClient(session=session, cache_dir=tmpdir)
            morph = load_neuromorpho(10047, cache_dir=tmpdir, client=client)
            self.assertIsInstance(morph, Morphology)
            self.assertGreaterEqual(len(morph.branches), 1)
            # The cache folder now contains the SWC file.
            self.assertTrue((Path(tmpdir) / "10047" / "TypeA-10.CNG.swc").exists())

    def _warm_cache(self, cache_dir: Path) -> Path:
        folder = cache_dir / "10047"
        folder.mkdir()
        (folder / "TypeA-10.CNG.swc").write_text(
            FIXTURE_SWC.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return folder

    def test_warm_cache_issues_no_requests(self) -> None:
        # The docstring promises already-cached neurons are not
        # re-downloaded. download() used to call get_neuron() and
        # get_measurement() first regardless, so a warm cache still cost two
        # requests against a rate-limited public API on every single call.
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            self._warm_cache(cache_dir)
            session = FakeSession([])  # any request at all raises
            client = NeuroMorphoClient(session=session, cache_dir=cache_dir)

            morph = load_neuromorpho(10047, cache_dir=cache_dir, client=client)

            self.assertIsInstance(morph, Morphology)
            self.assertEqual(session.calls, [])

    def test_warm_cache_does_not_rewrite_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            folder = self._warm_cache(cache_dir)
            metadata = folder / "metadata.json"
            metadata.write_text('{"sentinel": true}', encoding="utf-8")
            client = NeuroMorphoClient(session=FakeSession([]), cache_dir=cache_dir)

            load_neuromorpho(10047, cache_dir=cache_dir, client=client)

            self.assertEqual(metadata.read_text(encoding="utf-8"), '{"sentinel": true}')

    def test_overwrite_still_redownloads_a_cached_neuron(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            self._warm_cache(cache_dir)
            session = FakeSession(
                [
                    FakeResponse(json_data=sample_neuron_payload()),  # get_neuron
                    FakeResponse(json_data={"n_stems": 1.0}),  # measurement
                    FakeResponse(content=[FIXTURE_SWC.read_bytes()]),  # standard download
                ]
            )
            client = NeuroMorphoClient(session=session, cache_dir=cache_dir)

            morph = load_neuromorpho(10047, cache_dir=cache_dir, client=client, overwrite=True)

            self.assertIsInstance(morph, Morphology)
            self.assertEqual(len(session.calls), 3)


class MorphologyClassmethodTest(unittest.TestCase):
    def test_from_neuromorpho_matches_function(self) -> None:
        session = FakeSession(
            [
                FakeResponse(json_data=sample_neuron_payload()),
                FakeResponse(json_data={"n_stems": 1.0}),
                FakeResponse(content=[FIXTURE_SWC.read_bytes()]),
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NeuroMorphoClient(session=session, cache_dir=tmpdir)
            morph = Morphology.from_neuromorpho(10047, cache_dir=tmpdir, client=client)
            self.assertIsInstance(morph, Morphology)


class FetchNeuromorphoTest(unittest.TestCase):
    def test_writes_to_dest(self) -> None:
        session = FakeSession(
            [
                FakeResponse(json_data=sample_neuron_payload()),  # get_neuron
                FakeResponse(json_data={"n_stems": 1.0}),  # measurement
                FakeResponse(content=[b"standard-swc"]),  # download
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NeuroMorphoClient(session=session, cache_dir=tmpdir)
            record = fetch_neuromorpho(10047, dest=tmpdir, mode="standard", client=client)
            self.assertTrue((record.folder / "TypeA-10.CNG.swc").exists())


class DefaultCacheDirTest(unittest.TestCase):
    def test_resolves_under_home(self) -> None:
        self.assertTrue(str(DEFAULT_USER_CACHE_DIR).startswith(str(Path.home())))
        self.assertIn("braincell", str(DEFAULT_USER_CACHE_DIR))


if __name__ == "__main__":
    unittest.main()
