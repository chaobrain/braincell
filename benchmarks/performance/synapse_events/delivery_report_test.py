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

"""Verify paired comparisons and honest reporting of partial experiments."""

import csv
import json
import re
from pathlib import Path
import tempfile
import unittest

from benchmarks.performance.synapse_events import delivery_report as r


def _fixture(root, phase="micro"):
    directory = root / phase
    directory.mkdir()
    methods = ["current", "direct", "padded"]
    manifest = dict(
        protocol=dict(phase=phase, cases=["small"], devices=["cpu", "gpu"], rounds=3, precision=32, methods=methods),
        rows=[],
    )
    for rep in range(3):
        for device in ("cpu", "gpu"):
            rows = []
            for method in (["production"] if phase == "full" else []) + methods:
                if method == "padded":
                    rows.append(dict(method=method, status="not_applicable", reason="padding limit"))
                    continue
                seconds = 0.005 if method == "direct" else 0.01
                metric = dict(median_s=seconds, min_s=seconds * 0.9, max_s=seconds * 1.1, first_call_s=0.1)
                rows.append(
                    dict(
                        method=method,
                        status="ok",
                        prepare_cold_s=0.02,
                        array_bytes=100,
                        profile_status="ok",
                        timing={"full" if phase == "full" else "synapse": metric},
                    )
                )
            name = f"{device}_{rep}.json"
            data = dict(status="ok", rows=rows, build_s=0.3, schedule_sha256="same", count_sha256="same", arrivals=100)
            (directory / name).write_text(json.dumps(data))
            manifest["rows"].append(dict(file=name, device=device, case="small", round=rep))
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


class DeliveryReportTest(unittest.TestCase):
    def test_generated_reports_stay_local_and_resolve_links_from_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="report paths ") as tmp:
            root = Path(tmp)
            _fixture(root)
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

    def test_paired_evidence_costs_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fixture(root)
            _fixture(root, "full")
            _fixture(root, "profile")
            result = r.export_report(root, root / "report.md")
            self.assertEqual(result["failures"], [])
            self.assertEqual(len(result["profiles"]), 12)
            with (root / "delivery_summary.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            direct = next(x for x in rows if x["method"] == "direct" and x["phase"] == "micro")
            self.assertEqual(direct["stable_gain"], "True")
            self.assertEqual(float(direct["speedup"]), 2.0)
            self.assertAlmostEqual(float(direct["one_use_ms"]), 420.0)
            self.assertAlmostEqual(float(direct["reuse_100_ms"]), 915.0)
            self.assertTrue((root / "delivery_comparison.svg").exists())
            self.assertEqual(r.main([tmp, "--report", str(root / "again.md")]), 0)
            self.assertIn("padding limit", (root / "report.md").read_text())
            flat = r.collect(root)[0]
            one = [x for x in flat if x["round"] == 0]
            self.assertFalse(any(x["stable_gain"] for x in r.aggregate(one)))
            unpaired = [x for x in flat if x["method"] == "direct"]
            self.assertTrue(all(x["speedup"] is None for x in r.aggregate(unpaired)))

    def test_failed_missing_incomplete_and_mismatched_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest = _fixture(root)
            path = directory / "gpu_0.json"
            data = json.loads(path.read_text())
            data["count_sha256"] = "different"
            data["rows"].pop()
            path.write_text(json.dumps(data))
            (directory / "cpu_1.json").unlink()
            (directory / "cpu_2.json").write_text(json.dumps(dict(status="timeout", reason="120 seconds")))
            manifest["rows"].append(manifest["rows"][0])
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = r.export_report(root, root / "partial.md")
            self.assertTrue(any("duplicate" in x for x in result["failures"]))
            self.assertTrue(any("incomplete workers" in x for x in result["failures"]))
            self.assertTrue(any("incomplete methods" in x for x in result["failures"]))
            self.assertTrue(any("missing" in x for x in result["failures"]))
            self.assertTrue(any("mismatch" in x for x in result["failures"]))
            self.assertTrue(any("timeout" in x for x in result["failures"]))
            self.assertEqual(r.main([tmp, "--report", str(root / "failed.md")]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            result = r.export_report(Path(tmp), Path(tmp) / "empty.md")
            self.assertEqual(result["rows"], 0)
            self.assertEqual(result["failures"], ["no manifests found"])

    def test_full_profiles_do_not_enter_timing_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest = _fixture(root, "full")
            manifest["protocol"]["trace_full"] = True
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = r.export_report(root, root / "profile.md")
            self.assertEqual(result["rows"], 0)
            self.assertEqual(len(result["profiles"]), 18)
            self.assertEqual(result["failures"], [])
