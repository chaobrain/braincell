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

"""Tests for :mod:`braincell.io.neuromorpho.client`."""

import json
import tempfile
import unittest
from pathlib import Path

from braincell.io.neuromorpho import (
    NeuroMorphoClient,
    NeuroMorphoFilePlan,
    NeuroMorphoMeasurement,
    NeuroMorphoNeuron,
    NeuroMorphoNotFoundError,
    NeuroMorphoQuery,
    NeuroMorphoUrls,
)
from braincell.io.neuromorpho._testing import (
    FakeResponse,
    FakeSession,
    sample_neuron_payload,
)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class NeuroMorphoClientSearchTest(unittest.TestCase):
    def test_search_passes_fq_inside_params(self) -> None:
        session = FakeSession([
            FakeResponse(
                json_data={
                    "_embedded": {"neuronResources": [sample_neuron_payload()]},
                    "page": {"number": 2, "size": 5, "totalPages": 9, "totalElements": 42},
                },
                url="https://neuromorpho.org/api/neuron/select/?q=species:mouse&page=2",
            )
        ])
        client = NeuroMorphoClient(session=session, timeout=12)

        page = client.search(
            "species:mouse",
            fq=["brain_region:cerebellum"],
            size=5,
            page=2,
        )
        self.assertEqual(page.page, 2)
        self.assertEqual(page.total_pages, 9)
        self.assertEqual(page.items[0].neuron_id, 10047)

        url, kwargs = session.calls[0]
        self.assertEqual(url, "https://neuromorpho.org/api/neuron/select/")
        self.assertEqual(kwargs["params"]["fq"], ["brain_region:cerebellum"])
        self.assertEqual(kwargs["params"]["q"], "species:mouse")
        self.assertEqual(kwargs["timeout"], 12)

    def test_search_accepts_query_object(self) -> None:
        session = FakeSession([
            FakeResponse(
                json_data={
                    "_embedded": {"neuronResources": []},
                    "page": {"number": 0, "size": 20, "totalPages": 0, "totalElements": 0},
                },
            )
        ])
        client = NeuroMorphoClient(session=session)
        client.search(NeuroMorphoQuery(species="mouse", brain_region="cerebellum"))
        params = session.calls[0][1]["params"]
        self.assertEqual(params["q"], "species:mouse AND brain_region:cerebellum")

    def test_iter_search_paginates_until_exhausted(self) -> None:
        session = FakeSession([
            FakeResponse(
                json_data={
                    "_embedded": {"neuronResources": [sample_neuron_payload()]},
                    "page": {"number": 0, "size": 1, "totalPages": 2, "totalElements": 2},
                },
            ),
            FakeResponse(
                json_data={
                    "_embedded": {
                        "neuronResources": [
                            sample_neuron_payload(),  # duplicate, should be skipped
                            sample_neuron_payload(neuron_id=10048, neuron_name="TypeA-11"),
                        ]
                    },
                    "page": {"number": 1, "size": 2, "totalPages": 2, "totalElements": 2},
                },
            ),
        ])
        client = NeuroMorphoClient(session=session)
        ids = [n.neuron_id for n in client.iter_search("species:mouse", size=2)]
        self.assertEqual(ids, [10047, 10048])

    def test_iter_search_respects_limit(self) -> None:
        session = FakeSession([
            FakeResponse(
                json_data={
                    "_embedded": {
                        "neuronResources": [
                            sample_neuron_payload(),
                            sample_neuron_payload(neuron_id=10048, neuron_name="TypeA-11"),
                            sample_neuron_payload(neuron_id=10049, neuron_name="TypeA-12"),
                        ]
                    },
                    "page": {"number": 0, "size": 3, "totalPages": 5, "totalElements": 15},
                },
            ),
        ])
        client = NeuroMorphoClient(session=session)
        result = list(client.iter_search("species:mouse", size=3, limit=2))
        self.assertEqual([n.neuron_id for n in result], [10047, 10048])
        # Stops after the first page since limit is reached.
        self.assertEqual(len(session.calls), 1)

    def test_iter_search_stops_on_empty_page(self) -> None:
        session = FakeSession([
            FakeResponse(
                json_data={
                    "_embedded": {"neuronResources": []},
                    "page": {"number": 0, "size": 1, "totalPages": 0, "totalElements": 0},
                },
            ),
        ])
        client = NeuroMorphoClient(session=session)
        self.assertEqual(list(client.iter_search("species:mouse")), [])


# ---------------------------------------------------------------------------
# Single neuron / metadata
# ---------------------------------------------------------------------------


class NeuroMorphoClientNeuronTest(unittest.TestCase):
    def test_get_neuron_returns_typed_record(self) -> None:
        session = FakeSession([FakeResponse(json_data=sample_neuron_payload())])
        client = NeuroMorphoClient(session=session)
        neuron = client.get_neuron(10047)
        self.assertEqual(neuron.neuron_id, 10047)
        self.assertEqual(neuron.archive, "Scanziani")

    def test_get_neuron_propagates_not_found(self) -> None:
        session = FakeSession([FakeResponse(status_code=404)])
        client = NeuroMorphoClient(session=session, retries=1)
        with self.assertRaises(NeuroMorphoNotFoundError):
            client.get_neuron(999999)

    def test_get_measurement_returns_typed_object(self) -> None:
        session = FakeSession([
            FakeResponse(json_data={"n_stems": 1.0, "length": 10.0}),
        ])
        client = NeuroMorphoClient(session=session)
        meas = client.get_measurement(10047)
        self.assertIsInstance(meas, NeuroMorphoMeasurement)
        self.assertEqual(meas.neuron_id, 10047)
        self.assertEqual(meas.n_stems, 1)
        self.assertEqual(meas.length, 10.0)

    def test_describe_returns_typed_measurement_and_cache_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            folder = cache_dir / "10047"
            folder.mkdir()
            (folder / "TypeA-10.CNG.swc").write_text("swc", encoding="utf-8")
            (folder / "metadata.json").write_text("{}", encoding="utf-8")
            session = FakeSession([FakeResponse(json_data={"n_stems": 1.0})])
            client = NeuroMorphoClient(session=session, cache_dir=cache_dir)

            detail = client.describe(NeuroMorphoNeuron.from_payload(sample_neuron_payload()))

            self.assertIsInstance(detail.measurement, NeuroMorphoMeasurement)
            self.assertEqual(detail.measurement.n_stems, 1)
            self.assertEqual(detail.thumbnail_url, "https://neuromorpho.org/images/typea-10.png")
            self.assertIn("CNG%20version", detail.standard_swc_url)
            self.assertIn("Source-Version", detail.original_file_url)
            self.assertTrue(detail.cache_status.metadata_exists)
            self.assertTrue(detail.cache_status.standard_exists)
            self.assertFalse(detail.cache_status.original_exists)

    def test_describe_skips_measurement_when_requested(self) -> None:
        session = FakeSession([])
        client = NeuroMorphoClient(session=session)
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        detail = client.describe(neuron, include_measurement=False)
        self.assertIsNone(detail.measurement)
        self.assertEqual(len(session.calls), 0)

    def test_get_urls_returns_bundle(self) -> None:
        client = NeuroMorphoClient(session=FakeSession([]))
        urls = client.get_urls(NeuroMorphoNeuron.from_payload(sample_neuron_payload()))
        self.assertIsInstance(urls, NeuroMorphoUrls)
        self.assertIn("CNG%20version", urls.standard_swc)
        self.assertTrue(urls.thumbnail.startswith("https://"))
        self.assertIsNotNone(urls.original_file)

    def test_get_cache_status_unconfigured(self) -> None:
        client = NeuroMorphoClient(session=FakeSession([]))
        status = client.get_cache_status(10047)
        self.assertFalse(status.configured)
        self.assertEqual(status.neuron_id, 10047)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


class NeuroMorphoClientDownloadTest(unittest.TestCase):
    def test_download_writes_files_and_metadata(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        session = FakeSession([
            FakeResponse(json_data={"n_stems": 1.0, "n_branch": 2.0}),  # measurement
            FakeResponse(content=[b"standard-swc"]),                    # standard download
            FakeResponse(content=[b"original-data"]),                   # original download
        ])
        client = NeuroMorphoClient(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = client.download(neuron, output_dir=tmpdir, mode="both")
            folder = Path(tmpdir) / "10047"
            self.assertTrue((folder / "TypeA-10.CNG.swc").exists())
            self.assertTrue((folder / "TypeA-10.asc").exists())
            self.assertEqual(record.download_mode, "both")
            self.assertFalse(record.dry_run)
            self.assertIsInstance(record.measurement, NeuroMorphoMeasurement)
            self.assertEqual(record.measurement.n_stems, 1)
            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["measurement"]["n_stems"], 1.0)
            self.assertEqual(metadata["download_mode"], "both")
            self.assertEqual(len(metadata["download_items"]), 2)

    def test_download_skips_original_when_format_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload(original_format=None))
        session = FakeSession([
            FakeResponse(json_data={"n_stems": 1.0}),     # measurement
            FakeResponse(content=[b"standard-swc"]),      # standard
        ])
        client = NeuroMorphoClient(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = client.download(neuron, output_dir=tmpdir, mode="both")
            self.assertEqual(len(record.download_items), 2)
            original = record.download_items[1]
            self.assertEqual(original.kind, "original")
            self.assertFalse(original.downloaded_now)
            self.assertIn("original_format", original.reason or "")

    def test_download_dry_run_does_not_touch_disk(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        session = FakeSession([])  # no calls expected
        client = NeuroMorphoClient(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = client.download(neuron, output_dir=tmpdir, mode="both", dry_run=True)
            folder = Path(tmpdir) / "10047"
            self.assertFalse(folder.exists())
            self.assertTrue(record.dry_run)
            self.assertIsNone(record.measurement)
            for item in record.download_items:
                self.assertFalse(item.downloaded_now)
                self.assertEqual(item.reason, "dry_run")
            self.assertEqual(len(session.calls), 0)

    def test_download_uses_client_cache_when_output_dir_omitted(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        session = FakeSession([
            FakeResponse(json_data={"n_stems": 1.0}),
            FakeResponse(content=[b"standard"]),
        ])
        with tempfile.TemporaryDirectory() as tmpdir:
            client = NeuroMorphoClient(session=session, cache_dir=tmpdir)
            record = client.download(neuron, mode="standard")
            self.assertTrue(record.folder.is_relative_to(Path(tmpdir)))

    def test_download_requires_destination(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        client = NeuroMorphoClient(session=FakeSession([]))
        with self.assertRaises(ValueError):
            client.download(neuron, mode="standard")

    def test_file_plan_returns_typed_records(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        client = NeuroMorphoClient(session=FakeSession([]))
        plans = client.file_plan(neuron, mode="both")
        self.assertIsInstance(plans, tuple)
        self.assertIsInstance(plans[0], NeuroMorphoFilePlan)
        self.assertEqual({p.kind for p in plans}, {"standard", "original"})


if __name__ == "__main__":
    unittest.main()
