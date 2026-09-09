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

"""Verify reports retain failures, reject mismatched devices and expose costs."""

import csv
import json
import re
from pathlib import Path
import tempfile
import unittest

from benchmarks.performance.synapse_events.query_benchmark import QueryCase
from benchmarks.performance.synapse_events import query_report as r


def fixture(directory):
    rows = []
    for device in ("cpu", "gpu"):
        data = dict(
            status="ok",
            case=vars(QueryCase(n=10, m=10, k=10000)),
            rows=[],
            build_s=0.1,
            arrivals=1000,
            source_host_bytes=9000,
            schedule_sha256="same",
            count_sha256="same",
            environment=dict(device_kind=device, versions=dict(jax="0.10.1", brainstate="0.5.4", brainunit="0.3.0")),
        )
        for i, method in enumerate(r.METHODS):
            metric = dict(median_s=0.004 / (i + 1), first_call_s=0.1)
            data["rows"].append(
                dict(
                    method=method,
                    prepare_cold_s=0.01 * (i + 1),
                    reset_cold_s=0.001,
                    query_array_bytes=100,
                    cursor_bytes=4,
                    preparation={"bucket_build_s": 0.0},
                    timing={"query": metric, "synapse": metric},
                )
            )
        name = f"{device}.json"
        (directory / name).write_text(json.dumps(data))
        rows.append(dict(file=name, device=device, key="same", groups=["schedule", "scaling"]))
    manifest = dict(
        rows=rows,
        created_utc="2026-09-08",
        protocol=dict(duration_ms=100.0, dt_ms=0.025, precision=32, warmup=2, repeat=5, suite="smoke", devices="both"),
        cpu_info="model name : Example CPU",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class ReportTest(unittest.TestCase):
    def test_generated_reports_stay_local_and_resolve_links_from_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="report paths ") as tmp:
            root = Path(tmp)
            fixture(root)
            maintained = [r.HERE / "RESULTS.md", *sorted((r.HERE / "results").glob("*.md"))]
            before = {p: p.read_bytes() for p in maintained}
            self.assertEqual(r.main([str(root)]), 0)
            default_output = root / r.REPORT_FILENAME
            self.assertTrue(default_output.is_file())
            output = root / "separate reports" / "nested" / "custom.md"
            self.assertEqual(r.main([str(root), "--report", str(output)]), 0)
            self.assertTrue(output.is_file())
            for document in (default_output, output):
                links = re.findall(r"\]\(<([^>]+)>\)", document.read_text())
                self.assertTrue(links)
                for target in links:
                    self.assertTrue((document.parent / target).exists(), target)
                resolved = {(document.parent / target).resolve() for target in links}
                self.assertIn((r.HERE / "README.md").resolve(), resolved)
            self.assertEqual(before, {p: p.read_bytes() for p in maintained})

    def test_complete_report_csv_costs_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture(directory)
            result = r.export_report(directory, directory / "results.md")
            self.assertEqual(result, dict(cases=2, device_pairs=1, failures=[], rows=16))
            with (directory / "query_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(rows[0]["one_use_ms"]), 211.0)
            self.assertAlmostEqual(float(rows[0]["reuse_100_ms"]), 607.0)
            self.assertEqual(float(rows[2]["speedup"]), 2.0)
            self.assertTrue((directory / "query_schedule.svg").exists())
            self.assertTrue((directory / "query_scaling.png").exists())
            self.assertIn("未包含完整 Cell", (directory / "results.md").read_text())
            self.assertEqual(r.main([tmp, "--report", str(directory / "again.md")]), 0)
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["revisions"] = [{"description": "fixture revision"}]
            (directory / "manifest.json").write_text(json.dumps(manifest))
            r.export_report(directory, directory / "revised.md")
            self.assertIn("测量版本与边界修正", (directory / "revised.md").read_text())

    def test_failed_missing_incomplete_and_mismatched_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = fixture(directory)
            gpu = json.loads((directory / "gpu.json").read_text())
            gpu["count_sha256"] = "different"
            (directory / "gpu.json").write_text(json.dumps(gpu))
            self.assertEqual(r.main([tmp, "--report", str(directory / "bad.md")]), 1)
            (directory / "failed.json").write_text(json.dumps(dict(status="failed", error="timeout")))
            gpu["rows"] = []
            (directory / "gpu.json").write_text(json.dumps(gpu))
            manifest["rows"] += [dict(file=f, device="cpu", key=f, groups=[]) for f in ("missing.json", "failed.json")]
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = r.export_report(directory, directory / "partial.md")
            self.assertEqual(len(result["failures"]), 3)
            self.assertIn("未通过记录", (directory / "partial.md").read_text())
            (directory / "cpu.json").write_text(json.dumps(dict(status="failed")))
            result = r.export_report(directory, directory / "empty.md")
            self.assertEqual(result["cases"], 0)
            manifest["rows"] = []
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = r.export_report(directory, directory / "pending.md")
            self.assertIn("incomplete run", result["failures"][0])
