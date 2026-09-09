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

"""Exercise report completeness and matched numerical/performance comparisons."""

from copy import deepcopy
from dataclasses import asdict, replace
import csv
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.performance.synapse_events.benchmark import Case, cases_for
from benchmarks.performance.synapse_events import report as r


def _row(case, device, groups):
    # Fixtures are synthetic timing records, never presented as measured results.
    timing = dict(
        median_s=0.01 if device == "cpu" else 0.005,
        stdev_s=0.001,
        first_call_s=0.3,
        us_per_step=2.5,
        events_per_second=1000,
        samples_s=[0.009, 0.011],
    )
    return dict(
        key=case.key,
        case=asdict(case),
        device=device,
        groups=groups,
        status="ok",
        arrival_count=10,
        arrivals_per_source=[10],
        final_voltage_mv=[[-60.0]],
        final_conductance_us=[0.01],
        timing={m: dict(timing) for m in ("full", "query", "aggregate")},
        synapse_count=10,
        schedule_shape=[10, case.number or 11],
        build_s=0.1,
        source_build_s=0.01,
        prepare_s=0.2,
        environment=dict(device_kind=device, python="3.11.0", versions={"jax": "0.8.0"}, x64=False),
    )


class ReportTest(unittest.TestCase):
    def test_generated_reports_stay_local_and_resolve_links_from_custom_output(self):
        with tempfile.TemporaryDirectory(prefix="report paths ") as tmp:
            root = Path(tmp)
            case = Case(n=1, m=1)
            row = _row(case, "cpu", ["smoke"])
            (root / "cpu.json").write_text(json.dumps(row))
            (root / "manifest.json").write_text(
                json.dumps(
                    dict(
                        rows=[dict(key=case.key, device="cpu", groups=["smoke"], file="cpu.json", status="ok")],
                        created_utc="synthetic",
                        git_head="fixture",
                        warmup=0,
                        repeat=1,
                        gpu_status="not used",
                    )
                )
            )
            maintained = [r.HERE / "RESULTS.md", *sorted((r.HERE / "results").glob("*.md"))]
            before = {p: p.read_bytes() for p in maintained}
            self.assertEqual(r.main(["--input", str(root)]), 0)
            default_output = root / r.REPORT_FILENAME
            self.assertTrue(default_output.is_file())
            output = root / "separate reports" / "nested" / "custom.md"
            self.assertEqual(r.main(["--input", str(root), "--markdown", str(output)]), 0)
            self.assertTrue(output.is_file())
            for document in (default_output, output):
                links = re.findall(r"\]\(<([^>]+)>\)", document.read_text())
                self.assertTrue(links)
                for target in links:
                    self.assertTrue((document.parent / target).exists(), target)
                resolved = {(document.parent / target).resolve() for target in links}
                self.assertIn((r.HERE / "README.md").resolve(), resolved)
            self.assertEqual(before, {p: p.read_bytes() for p in maintained})

    def test_pairing_detects_counts_and_state_mismatches(self):
        case = Case(n=10, number=10)
        rows = {}
        for c in (
            case,
            replace(case, layout="shared"),
            replace(case, number=100),
            replace(case, declaration="per_cell"),
        ):
            for device in ("cpu", "gpu"):
                rows[c.key, device] = _row(c, device, ["schedule"])
        checks = r.validate_results(rows)
        self.assertEqual(
            checks,
            dict(
                device_pairs=4,
                layout_pairs=2,
                schedule_pairs=2,
                declaration_pairs=2,
                max_voltage_error_mv=0.0,
                max_conductance_error_us=0.0,
                failures=[],
            ),
        )
        bad = deepcopy(rows)
        bad[case.key, "gpu"]["arrivals_per_source"] = [9]
        bad[case.key, "gpu"]["final_voltage_mv"] = [[-50.0]]
        failures = r.validate_results(bad)["failures"]
        self.assertTrue(any("event counts" in f for f in failures))
        self.assertTrue(any("voltage" in f for f in failures))
        self.assertEqual(r.validate_results(bad)["max_voltage_error_mv"], 10.0)

    def test_load_excludes_failed_missing_and_invalid_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            entries = [
                dict(key=str(i), device="cpu", groups=[], file=f"{i}.json", status=status)
                for i, status in enumerate(("ok", "failed", "ok", "ok"))
            ]
            (path / "manifest.json").write_text(json.dumps({"rows": entries}))
            (path / "0.json").write_text(json.dumps({"status": "ok"}))
            (path / "3.json").write_text(json.dumps({"status": "failed"}))
            _, rows, failures = r.load_results(path)
            self.assertEqual(list(rows), [("0", "cpu")])
            self.assertEqual(len(failures), 3)

    def test_complete_report_all_groups_csv_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            entries = []
            for case, groups in cases_for("all").items():
                for device in ("cpu", "gpu"):
                    row = _row(case, device, groups)
                    filename = f"{device}_{case.key}.json"
                    (path / filename).write_text(json.dumps(row))
                    entries.append(dict(key=case.key, device=device, groups=groups, file=filename, status="ok"))
            manifest = dict(
                rows=entries,
                created_utc="synthetic test",
                git_head="fixture",
                warmup=2,
                repeat=2,
                gpu_status="test device",
            )
            (path / "manifest.json").write_text(json.dumps(manifest))
            trace_dir = path / "diagnostics" / "trace"
            trace_dir.mkdir(parents=True)
            (trace_dir / "independent_summary.json").write_text(
                json.dumps(
                    dict(
                        case=asdict(Case()),
                        command_buffers="default",
                        total_gpu_events=52000,
                        total_gpu_time_ms=80.0,
                        matched_gpu_events=0,
                        kernels=[dict(name="fusion", calls=4000, total_ms=8.0, mean_us=2.0)],
                    )
                )
            )
            diagnostic_case = Case(n=1, m=1, pattern="sync")
            for precision in (32, 64):
                diagnostic_path = path / "diagnostics" / f"precision{precision}"
                diagnostic_path.mkdir(parents=True)
                diagnostic_entries = []
                for device in ("cpu", "gpu"):
                    row = _row(diagnostic_case, device, ["scaling"])
                    if precision == 32 and device == "gpu":
                        row["final_voltage_mv"] = [[-59.98]]
                    (diagnostic_path / f"{device}.json").write_text(json.dumps(row))
                    diagnostic_entries.append(
                        dict(
                            key=diagnostic_case.key,
                            device=device,
                            groups=["scaling"],
                            file=f"{device}.json",
                            status="ok",
                        )
                    )
                diagnostic_entries.append(dict(key="missing", device="gpu", file="missing.json", status="failed"))
                (diagnostic_path / "manifest.json").write_text(json.dumps(dict(manifest, rows=diagnostic_entries)))
            markdown = path / "RESULTS.md"
            self.assertEqual(r.main(["--input", tmp, "--markdown", str(markdown)]), 0)
            text = markdown.read_text()
            self.assertIn("全部配对通过", text)
            self.assertIn("连接声明", text)
            self.assertIn("时间表从 10 增至 10000", text)
            self.assertIn("精度补充诊断", text)
            self.assertIn("0.02", text)
            self.assertIn("诊断有 1 个未完成配置", text)
            self.assertIn("未匹配到 BrainCell named scope", text)
            self.assertIn("GPU trace", text)
            for group in ("scaling", "schedule"):
                for suffix in ("svg", "png"):
                    self.assertGreater((path / f"{group}.{suffix}").stat().st_size, 100)
            with (path / "summary.csv").open() as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(len(csv_rows), len(entries) * 3)
            self.assertEqual(json.loads(csv_rows[0]["samples_s"]), [0.009, 0.011])

    def test_partial_report_and_empty_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            case = Case()
            row = _row(case, "cpu", ["smoke"])
            row["timing"] = {"full": row["timing"]["full"]}
            manifest = dict(rows=[], created_utc="test", git_head="test", warmup=0, repeat=1)
            failures = [dict(device="gpu", key=case.key, status="failed")]
            checks = dict(
                device_pairs=0,
                layout_pairs=0,
                schedule_pairs=0,
                declaration_pairs=0,
                max_voltage_error_mv=0.0,
                max_conductance_error_us=0.0,
                failures=["injected mismatch"],
            )
            with (
                patch.object(r, "load_results", return_value=(manifest, {(case.key, "cpu"): row}, failures)),
                patch.object(r, "validate_results", return_value=checks),
            ):
                self.assertEqual(r.main(["--input", tmp, "--markdown", str(path / "partial.md")]), 1)
            text = (path / "partial.md").read_text()
            self.assertIn("未完成配置", text)
            self.assertIn("不匹配", text)
            self.assertIn("—", text)
            r.export_csv({}, path / "empty.csv")
            self.assertEqual((path / "empty.csv").read_text().strip(), "key,device,mode")
            self.assertEqual(r.plot_results({}, path), [])
            self.assertEqual(r.findings({}), [])


if __name__ == "__main__":
    unittest.main()
