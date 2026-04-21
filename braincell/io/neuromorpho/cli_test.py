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

"""Tests for :mod:`braincell.io.neuromorpho.cli`."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from braincell.io.neuromorpho import (
    NeuroMorphoCache,
    NeuroMorphoCacheStatus,
    NeuroMorphoDetail,
    NeuroMorphoDownloadItem,
    NeuroMorphoDownloadRecord,
    NeuroMorphoMeasurement,
    NeuroMorphoNeuron,
    NeuroMorphoSearchPage,
    NeuroMorphoUrls,
)
from braincell.io.neuromorpho._testing import sample_neuron_payload
from braincell.io.neuromorpho.cli import build_arg_parser, main


def _parse_key_value_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key] = value
    return parsed


class CliSearchTest(unittest.TestCase):
    def test_search_subcommand_prints_page(self) -> None:
        fake_page = NeuroMorphoSearchPage(
            items=(NeuroMorphoNeuron.from_payload(sample_neuron_payload()),),
            page=1,
            size=10,
            total_pages=5,
            total_elements=42,
            query_url="https://neuromorpho.org/api/neuron/select/?q=species:mouse&page=1",
        )
        with mock.patch("braincell.io.neuromorpho.cli.NeuroMorphoClient") as client_cls:
            client_cls.return_value.search.return_value = fake_page
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["search", "--q", "species:mouse", "--page", "1"])
        self.assertEqual(exit_code, 0)
        output = stream.getvalue()
        self.assertIn("page=1 size=10 total_pages=5 total_elements=42", output)
        self.assertIn("id=10047", output)


class CliShowTest(unittest.TestCase):
    def test_show_subcommand_prints_detail(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        urls = NeuroMorphoUrls(
            standard_swc="https://neuromorpho.org/dableFiles/scanziani/CNG%20version/TypeA-10.CNG.swc",
            original_file="https://neuromorpho.org/dableFiles/scanziani/Source-Version/TypeA-10.asc",
            measurement="https://neuromorpho.org/api/morphometry/id/10047",
            thumbnail="https://neuromorpho.org/images/typea-10.png",
        )
        cache_status = NeuroMorphoCacheStatus(
            configured=False,
            folder=None,
            exists=False,
            metadata_exists=False,
            standard_exists=False,
            original_exists=False,
            neuron_id=10047,
        )
        meas = NeuroMorphoMeasurement.from_payload({"neuron_id": 10047, "n_stems": 1.0})
        detail = NeuroMorphoDetail(
            neuron=neuron, measurement=meas, urls=urls, cache_status=cache_status
        )
        with mock.patch("braincell.io.neuromorpho.cli.NeuroMorphoClient") as client_cls:
            client_cls.return_value.describe.return_value = detail
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["show", "--id", "10047"])
        self.assertEqual(exit_code, 0)
        out = stream.getvalue()
        self.assertIn("thumbnail_url=https://neuromorpho.org/images/typea-10.png", out)
        self.assertIn("\"n_stems\": 1", out)


class CliDownloadTest(unittest.TestCase):
    def test_download_subcommand_prints_record(self) -> None:
        record = NeuroMorphoDownloadRecord(
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
            measurement=None,
            download_mode="both",
        )
        with mock.patch("braincell.io.neuromorpho.cli.NeuroMorphoClient") as client_cls:
            client_cls.return_value.download.return_value = record
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main([
                    "download", "--id", "10047", "--output-dir", "/tmp/out", "--mode", "both"
                ])
        self.assertEqual(exit_code, 0)
        out = stream.getvalue()
        parsed = _parse_key_value_output(out)
        self.assertEqual(Path(parsed["folder"]), record.folder)
        self.assertEqual(Path(parsed["metadata_path"]), record.metadata_path)
        self.assertIn("downloaded_now=True", out)


class CliUrlsTest(unittest.TestCase):
    def test_urls_subcommand(self) -> None:
        neuron = NeuroMorphoNeuron.from_payload(sample_neuron_payload())
        urls = NeuroMorphoUrls(
            standard_swc="https://example/standard.swc",
            original_file="https://example/orig.asc",
            measurement="https://example/meas",
            thumbnail="https://example/thumb.png",
        )
        with mock.patch("braincell.io.neuromorpho.cli.NeuroMorphoClient") as client_cls:
            client_cls.return_value.get_neuron.return_value = neuron
            client_cls.return_value.get_urls.return_value = urls
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["urls", "10047"])
        self.assertEqual(exit_code, 0)
        out = stream.getvalue()
        self.assertIn("standard_swc_url=https://example/standard.swc", out)


class CliCacheTest(unittest.TestCase):
    def test_list_subcommand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = NeuroMorphoCache(tmpdir)
            (cache.root / "10047").mkdir()
            (cache.root / "10048").mkdir()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["--cache-dir", tmpdir, "cache", "list"])
            self.assertEqual(exit_code, 0)
            out = stream.getvalue()
            self.assertIn("count=2", out)
            self.assertIn("10047", out)
            self.assertIn("10048", out)

    def test_clear_requires_yes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "1").mkdir()
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = main(["--cache-dir", tmpdir, "cache", "clear"])
            self.assertEqual(exit_code, 2)
            self.assertIn("--yes", stream.getvalue())
            # Folder still present.
            self.assertTrue((Path(tmpdir) / "1").exists())


class CliArgParserTest(unittest.TestCase):
    def test_round_trips_all_subcommands(self) -> None:
        parser = build_arg_parser()
        for argv in (
            ["search", "--q", "species:mouse"],
            ["search", "--species", "mouse", "--brain-region", "cerebellum"],
            ["show", "--id", "10047"],
            ["download", "--id", "10047", "--output-dir", "/tmp/out"],
            ["fetch", "10047", "--load"],
            ["urls", "10047"],
            ["cache", "list"],
            ["cache", "info", "10047"],
            ["cache", "rm", "10047"],
            ["cache", "clear", "--yes"],
        ):
            try:
                parser.parse_args(argv)
            except SystemExit as exc:  # pragma: no cover
                self.fail(f"failed to parse {argv}: {exc}")


if __name__ == "__main__":
    unittest.main()
