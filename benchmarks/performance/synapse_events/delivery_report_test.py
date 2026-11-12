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

import pytest

from benchmarks.performance.synapse_events import delivery_report as r


# Slow tier: runs nightly, not on every pull request (see the CI workflow).
pytestmark = pytest.mark.slow


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


def _regression_fixture(root, *, spikes_differ=False):
    """Two models, one entry, three methods, three rounds; ``padded`` of ``small`` not applicable."""
    import numpy as np

    directory = root / "regression"
    directory.mkdir()
    protocol = dict(
        phase="regression",
        cases=["small", "active"],
        entries=["run"],
        methods=["default", "direct", "padded"],
        devices=["cpu"],
        rounds=3,
        steps=[4000, 2000],
        git_sha="abc1234",
    )
    manifest = dict(protocol=protocol, rows=[])
    # (T(4000), T(2000)) in seconds; first call; plan preparation (inside the first call of the run entry).
    times = {"default": (0.40, 0.21, 2.0, 0.0), "direct": (0.20, 0.11, 2.5, 0.1), "padded": (0.30, 0.16, 1.5, 0.1)}
    for round_id in range(3):
        for case in ("small", "active"):
            for method in ("default", "direct", "padded"):
                name = f"cpu_{case}_run_{method}_r{round_id}_a0.json"
                if case == "small" and method == "padded":
                    data = dict(status="not_applicable", reason="padding limit", padding_ratio=40.0, attempt_s=0.5)
                else:
                    hi, lo, first, prep = times[method]
                    scale = 1.0 + 0.01 * round_id
                    timing = {
                        "4000": dict(
                            median_s=hi * scale,
                            first_call_s=first,
                            iqr_over_median=0.02,
                            plan=dict(plan_prepare_s=prep, array_bytes=2048),
                        ),
                        "2000": dict(median_s=lo * scale, first_call_s=1.0, iqr_over_median=0.05),
                    }
                    if method == "default":
                        del timing["4000"]["plan"]
                    data = dict(
                        status="ok",
                        entry="run",
                        timing=timing,
                        load_start=2.0,
                        load_end=3.0,
                        cpu_busy_share={"1": 0.1},
                        peak_rss_mb=900.0,
                        build_s=0.1,
                        prepare_s=0.2,
                    )
                    spike = np.array([5, 9]) + (1 if spikes_differ and method == "direct" and round_id == 1 else 0)
                    np.savez(
                        directory / name.replace(".json", ".npz"),
                        V=np.full(3, -65.0),
                        g=np.zeros(4),
                        spike_step=spike,
                        spike_source=np.array([0, 1]),
                    )
                (directory / name).write_text(json.dumps(data))
                manifest["rows"].append(
                    dict(
                        device="cpu",
                        case=case,
                        entry="run",
                        method=method,
                        round=round_id,
                        file=name,
                        status=data["status"],
                        flagged=None,
                        attempts=[{}],
                    )
                )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


class RegressionReportTest(unittest.TestCase):
    def test_ratios_slopes_break_even_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _regression_fixture(root)
            result = r.export_regression(root)
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["rows"], 18)
            summary, selections, _ = r.aggregate_regression(r.collect_regression(root)[0])
            pick = {(s["case"], s["method"]): s for s in summary}
            direct = pick["active", "direct"]
            # Same scale per round in both methods: every paired ratio is 0.40 / 0.20.
            self.assertEqual([round(x, 12) for x in direct["ratios"]], [2.0, 2.0, 2.0])
            # Slope (T(4000) - T(2000)) / 2000 and intercept 2 T(2000) - T(4000), round 1 (scale 1.01).
            self.assertAlmostEqual(direct["per_step_s"], 0.09 * 1.01 / 2000)
            self.assertAlmostEqual(direct["per_run_s"], 0.02 * 1.01)
            # Run entry: the plan is built in the first call, so extra cold = 2.5 - 2.0.
            self.assertAlmostEqual(direct["extra_cold_s"], 0.5)
            self.assertAlmostEqual(direct["break_even"], 0.5 / (0.40 * 1.01 - 0.20 * 1.01))
            self.assertTrue(direct["stable_gain"])
            # Faster first call than the default: pays back at once.
            self.assertEqual(pick["active", "padded"]["break_even"], 0.0)
            self.assertEqual(pick["small", "padded"]["status"], "not_applicable")
            self.assertEqual(pick["small", "padded"]["padding_ratio"], 40.0)
            chosen = {(s["case"], s["entry"]): s["recommendation"] for s in selections}
            self.assertEqual(chosen, {("active", "run"): "direct", ("small", "run"): "direct"})
            text = (root / r.REGRESSION_FILENAME).read_text()
            self.assertIn("not_applicable (padding ratio 40)", text)
            self.assertIn("| cpu | active | run | direct | 2.00, 2.00, 2.00 | 2.00 |", text)
            self.assertTrue((root / "regression_summary.csv").exists())
            self.assertEqual(r.main([tmp, "--regression", "--report", str(root / "again.md")]), 0)
            # The delivery report ignores regression manifests.
            self.assertEqual(r.collect(root)[:3], ([], [], []))

    def test_update_entry_adds_plan_preparation_and_slower_methods_never_pay_back(self):
        rows = []
        for round_id in range(3):
            for method, run_s, first, prep in (
                ("default", 0.2, 1.0, 0.0),
                ("direct", 0.25, 1.0, 0.3),
                ("padded", 0.1, 0.9, 0.3),
            ):
                rows.append(
                    dict(
                        device="gpu",
                        case="large",
                        entry="update",
                        method=method,
                        round=round_id,
                        status="ok",
                        run_s=run_s,
                        per_step_s=run_s / 4000,
                        per_run_s=0.0,
                        first_call_s=first,
                        plan_prepare_s=prep,
                        array_bytes=1,
                        peak_rss_mb=1.0,
                        iqr_over_median=0.01,
                        load=1.0,
                        flagged="IQR" if round_id == 2 else None,
                        sha="abc",
                        check=None,
                    )
                )
        summary, selections, failures = r.aggregate_regression(rows)
        pick = {s["method"]: s for s in summary}
        self.assertEqual(pick["direct"]["break_even"], "never")
        self.assertFalse(pick["direct"]["stable_gain"])
        self.assertAlmostEqual(pick["padded"]["extra_cold_s"], 0.2)
        self.assertAlmostEqual(pick["padded"]["break_even"], 0.2 / 0.1)
        self.assertEqual(pick["padded"]["flagged"], [2])
        self.assertEqual(selections[0]["recommendation"], "padded")
        self.assertEqual(failures, [])

    def test_check_mismatch_missing_and_failed_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory, manifest = _regression_fixture(root, spikes_differ=True)
            (directory / "cpu_active_run_padded_r1_a0.json").unlink()
            (directory / "cpu_active_run_direct_r0_a0.json").write_text(json.dumps(dict(status="failed", error="x")))
            manifest["rows"].pop()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            result = r.export_regression(root, root / "partial.md")
            failures = "\n".join(result["failures"])
            self.assertIn("incomplete workers 17/18", failures)
            self.assertIn("missing cpu_active_run_padded_r1_a0.json", failures)
            self.assertIn("cpu_active_run_direct_r0_a0.json: failed", failures)
            self.assertIn("cpu/active/run/direct/r1", failures)
            self.assertEqual(r.main([tmp, "--regression", "--report", str(root / "failed.md")]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            # A run of selected groups expects their workers only.
            directory, manifest = _regression_fixture(Path(tmp))
            manifest["protocol"]["groups"] = [[r, c, "run"] for r in range(3) for c in ("small", "active")]
            (directory / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(r.collect_regression(Path(tmp))[1], [])
            manifest["protocol"]["groups"] = manifest["protocol"]["groups"][:2]
            (directory / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(r.collect_regression(Path(tmp))[1], ["regression: incomplete workers 18/6"])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(r.export_regression(Path(tmp))["failures"], ["no regression manifests found"])
            directory, manifest = _regression_fixture(Path(tmp))
            for row in manifest["rows"]:
                if row["method"] == "default" and row["case"] == "active":
                    (directory / row["file"]).write_text(json.dumps(dict(status="timeout", reason="deadline")))
            summary, selections, _ = r.aggregate_regression(r.collect_regression(Path(tmp))[0])
            active = {s["method"]: s for s in summary if s["case"] == "active"}
            self.assertEqual(active["default"]["status"], "timeout")
            self.assertIsNone(active["direct"]["ratio"])
            self.assertNotIn("active", {s["case"] for s in selections})
