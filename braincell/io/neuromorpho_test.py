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


import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from braincell.io.neuromorpho import (
    find_standard_swc,
    load_cached_metadata,
    NeuroMorphoClient,
    NeuroMorphoDetail,
    NeuroMorphoDownloadItem,
    NeuroMorphoDownloadRecord,
    NeuroMorphoNeuron,
    NeuroMorphoSearchPage,
    main,
)


class FakeResponse:
    def __init__(self, *, json_data=None, content=None, url="https://example.test", status_code=200):
        self._json_data = json_data
        self._content = content or []
        self.url = url
        self.status_code = status_code

    def json(self):
        if self._json_data is None:
            raise AssertionError("json() was not expected for this response")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=8192):
        del chunk_size
        for chunk in self._content:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected GET {url!r}")
        return self.responses.pop(0)


def _sample_neuron_payload(**overrides):
    payload = {
        "neuron_id": 10047,
        "neuron_name": "TypeA-10",
        "archive": "Scanziani",
        "species": "mouse",
        "brain_region": ["neocortex", "occipital", "layer 6"],
        "cell_type": ["principal cell"],
        "original_format": "TypeA-10.asc",
        "png_url": "http://neuromorpho.org/images/typea-10.png",
        "_links": {
            "measurements": {"href": "http://neuromorpho.org/api/morphometry/id/10047"},
        },
    }
    payload.update(overrides)
    return payload


class NeuroMorphoClientTest(unittest.TestCase):
    def test_search_parses_results_and_paging(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    json_data={
                        "_embedded": {"neuronResources": [_sample_neuron_payload()]},
                        "page": {"number": 2, "size": 5, "totalPages": 9, "totalElements": 42},
                    },
                    url="https://neuromorpho.org/api/neuron/select/?q=species%3Amouse&page=2",
                )
            ]
        )
        client = NeuroMorphoClient(session=session, timeout=12)

        page = client.search(q="species:mouse", fq=["brain_region:cerebellum"], size=5, page=2)

        self.assertEqual(page.page, 2)
        self.assertEqual(page.size, 5)
        self.assertEqual(page.total_pages, 9)
        self.assertEqual(page.total_elements, 42)
        self.assertEqual(len(page.items), 1)
        self.assertEqual(page.items[0].neuron_id, 10047)
        self.assertEqual(
            session.calls[0][0],
            "https://neuromorpho.org/api/neuron/select/",
        )
        self.assertEqual(session.calls[0][1]["params"]["fq"], ["brain_region:cerebellum"])
        self.assertEqual(session.calls[0][1]["timeout"], 12)

    def test_search_batch_deduplicates_and_collects_query_urls(self) -> None:
        session = FakeSession(
            [
                FakeResponse(
                    json_data={
                        "_embedded": {"neuronResources": [_sample_neuron_payload()]},
                        "page": {"number": 0, "size": 1, "totalPages": 2, "totalElements": 2},
                    },
                    url="https://neuromorpho.org/api/neuron/select/?page=0",
                ),
                FakeResponse(
                    json_data={
                        "_embedded": {
                            "neuronResources": [
                                _sample_neuron_payload(),
                                _sample_neuron_payload(neuron_id=10048, neuron_name="TypeA-11"),
                            ]
                        },
                        "page": {"number": 1, "size": 2, "totalPages": 2, "totalElements": 2},
                    },
                    url="https://neuromorpho.org/api/neuron/select/?page=1",
                ),
            ]
        )
        client = NeuroMorphoClient(session=session)

        neurons, query_urls = client.search_batch(q="species:mouse", size=2, page_start=0, max_pages=3)

        self.assertEqual([neuron.neuron_id for neuron in neurons], [10047, 10048])
        self.assertEqual(
            query_urls,
            (
                "https://neuromorpho.org/api/neuron/select/?page=0",
                "https://neuromorpho.org/api/neuron/select/?page=1",
            ),
        )

    def test_describe_returns_measurement_urls_thumbnail_and_cache_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            folder = cache_dir / "10047"
            folder.mkdir()
            (folder / "TypeA-10.CNG.swc").write_text("swc", encoding="utf-8")
            (folder / "metadata.json").write_text("{}", encoding="utf-8")
            session = FakeSession([FakeResponse(json_data={"n_stems": 1.0})])
            client = NeuroMorphoClient(session=session, cache_dir=cache_dir)

            detail = client.describe(NeuroMorphoNeuron.from_payload(_sample_neuron_payload()))

            self.assertEqual(detail.measurement, {"n_stems": 1.0})
            self.assertEqual(detail.thumbnail_url, "https://neuromorpho.org/images/typea-10.png")
            self.assertIn("CNG%20version", detail.standard_swc_url)
            self.assertIn("Source-Version", detail.original_file_url)
            self.assertTrue(detail.cache_status["metadata_exists"])
            self.assertTrue(detail.cache_status["standard_exists"])
            self.assertFalse(detail.cache_status["original_exists"])

    def test_download_writes_files_and_metadata_for_both_modes(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(_sample_neuron_payload())
        session = FakeSession(
            [
                FakeResponse(json_data={"n_stems": 1.0, "n_branch": 2.0}),
                FakeResponse(content=[b"standard-swc"]),
                FakeResponse(content=[b"original-data"]),
            ]
        )
        client = NeuroMorphoClient(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = client.download(neuron, output_dir=tmpdir, mode="both")
            folder = Path(tmpdir) / "10047"

            self.assertTrue((folder / "TypeA-10.CNG.swc").exists())
            self.assertTrue((folder / "TypeA-10.asc").exists())
            self.assertEqual(record.download_mode, "both")
            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["measurement"]["n_stems"], 1.0)
            self.assertEqual(metadata["download_mode"], "both")
            self.assertEqual(len(metadata["download_items"]), 2)
            self.assertTrue(record.download_items[0].downloaded_now)
            self.assertTrue(record.download_items[1].downloaded_now)

    def test_download_skips_original_when_original_format_missing(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(_sample_neuron_payload(original_format=None))
        session = FakeSession(
            [
                FakeResponse(json_data={"n_stems": 1.0}),
                FakeResponse(content=[b"standard-swc"]),
            ]
        )
        client = NeuroMorphoClient(session=session)

        with tempfile.TemporaryDirectory() as tmpdir:
            record = client.download(neuron, output_dir=tmpdir, mode="both")

            self.assertEqual(len(record.download_items), 2)
            self.assertEqual(record.download_items[1].kind, "original")
            self.assertFalse(record.download_items[1].downloaded_now)
            self.assertIn("original_format", record.download_items[1].reason)

    def test_load_cached_metadata_and_find_standard_swc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "10047"
            folder.mkdir()
            swc_path = folder / "TypeA-10.CNG.swc"
            swc_path.write_text("swc", encoding="utf-8")
            metadata_path = folder / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "download_items": [
                            {
                                "kind": "standard",
                                "filename": swc_path.name,
                                "path": str(swc_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            metadata = load_cached_metadata(folder)
            resolved = find_standard_swc(folder, metadata)

            self.assertEqual(metadata["download_items"][0]["filename"], "TypeA-10.CNG.swc")
            self.assertEqual(resolved, swc_path)


class NeuroMorphoCliTest(unittest.TestCase):
    def test_main_search_prints_page_summary(self) -> None:
        fake_page = NeuroMorphoSearchPage(
            items=(NeuroMorphoNeuron.from_payload(_sample_neuron_payload()),),
            page=1,
            size=10,
            total_pages=5,
            total_elements=42,
            query_url="https://neuromorpho.org/api/neuron/select/?q=species%3Amouse&page=1",
        )
        with mock.patch("braincell.io.neuromorpho.NeuroMorphoClient") as client_cls:
            client_cls.return_value.search.return_value = fake_page
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["search", "--q", "species:mouse", "--page", "1"])

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("page=1 size=10 total_pages=5 total_elements=42", output)
        self.assertIn("id=10047", output)

    def test_main_show_prints_detail(self) -> None:
        fake_detail = NeuroMorphoDetail(
            neuron=NeuroMorphoNeuron.from_payload(_sample_neuron_payload()),
            measurement={"n_stems": 1.0},
            thumbnail_url="https://neuromorpho.org/images/typea-10.png",
            standard_swc_url="https://neuromorpho.org/dableFiles/scanziani/CNG%20version/TypeA-10.CNG.swc",
            original_file_url="https://neuromorpho.org/dableFiles/scanziani/Source-Version/TypeA-10.asc",
            cache_status={"configured": False},
        )
        with mock.patch("braincell.io.neuromorpho.NeuroMorphoClient") as client_cls:
            client_cls.return_value.describe.return_value = fake_detail
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["show", "--id", "10047"])

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("thumbnail_url=https://neuromorpho.org/images/typea-10.png", output)
        self.assertIn('"n_stems": 1.0', output)

    def test_main_download_prints_result(self) -> None:
        fake_record = NeuroMorphoDownloadRecord(
            folder=Path("/tmp/out/10047"),
            metadata_path=Path("/tmp/out/10047/metadata.json"),
            download_items=(
                NeuroMorphoDownloadItem(
                    kind="standard",
                    url="https://neuromorpho.org/standard",
                    filename="TypeA-10.CNG.swc",
                    path=Path("/tmp/out/10047/TypeA-10.CNG.swc"),
                    downloaded_now=True,
                ),
            ),
            measurement={"n_stems": 1.0},
            download_mode="both",
        )
        with mock.patch("braincell.io.neuromorpho.NeuroMorphoClient") as client_cls:
            client_cls.return_value.download.return_value = fake_record
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(
                    ["download", "--id", "10047", "--output-dir", "/tmp/out", "--mode", "both"]
                )

        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("metadata_path=/tmp/out/10047/metadata.json", output)
        self.assertIn("downloaded_now=True", output)


class NeuroMorphoNotebookStructureTest(unittest.TestCase):
    def test_notebook_uses_io_client_for_search_and_download_cells(self) -> None:
        notebook_path = Path(__file__).with_name("neuromorpho_diff.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        cell0 = "".join(notebook["cells"][0]["source"])
        cell2 = "".join(notebook["cells"][2]["source"])
        cell4 = "".join(notebook["cells"][4]["source"])
        cell5 = "".join(notebook["cells"][5]["source"])
        cell7 = "".join(notebook["cells"][7]["source"])

        self.assertIn("from braincell.io import", cell0)
        self.assertIn("NeuroMorphoClient", cell0)
        self.assertNotIn("import requests", cell0)
        self.assertNotIn("def fetch_neurons", cell2)
        self.assertNotIn("def cache_neuron_bundle", cell2)
        self.assertIn("client = NeuroMorphoClient", cell4)
        self.assertIn("client.search_batch", cell4)
        self.assertIn("client.download", cell5)
        self.assertIn("load_cached_metadata", cell5)
        self.assertIn("find_standard_swc", cell7)


if __name__ == "__main__":
    unittest.main()
