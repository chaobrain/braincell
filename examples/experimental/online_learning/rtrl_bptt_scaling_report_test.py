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

"""Tests for the local scaling-result report generator."""

import csv
from pathlib import Path
import tempfile
import unittest

from examples.experimental.online_learning.rtrl_bptt_scaling_report import (
    generate_report,
    load_result_rows,
    write_report,
)


class ScalingReportTest(unittest.TestCase):
    def test_missing_results_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                generate_report(Path(directory))

    def test_report_loads_recursive_and_ordinary_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, "full_block_exact", backsub=None)
            self._write_run(root, "backsub_ordinary_block_exact", backsub="ordinary")

            rows = load_result_rows(root)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["backsub"] for row in rows}, {"recursive", "ordinary"})
            report = generate_report(root)
            self.assertIn("# RTRL/BPTT Scaling Results", report)
            self.assertIn("Ordinary Hines Backsub A/B", report)
            self.assertIn("Successful trials", report)
            output = write_report(root)
            self.assertEqual(output, root / "RESULTS.md")
            self.assertIn("# RTRL/BPTT Scaling Results", output.read_text())

    @staticmethod
    def _write_run(root: Path, name: str, *, backsub: str | None) -> None:
        directory = root / name
        directory.mkdir()
        fields = [
            "config_id",
            "status",
            "method",
            "n_cv",
            "duration_ms",
            "batch_size",
            "n_seed",
            "compile_seconds",
            "steady_median_seconds",
            "temporary_bytes",
            "gpu_peak_steady_bytes",
            "rtrl_carry_bytes",
            "gpu_power_steady_median_watts",
            "gradient_max_rel_error",
            "loss_max_abs_error",
        ]
        if backsub is not None:
            fields.append("backsub")
        with (directory / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method in ("bptt", "rtrl"):
                row = {
                    "config_id": "c9_t40_b16_s16",
                    "status": "ok",
                    "method": method,
                    "n_cv": 9,
                    "duration_ms": 40,
                    "batch_size": 16,
                    "n_seed": 16,
                    "compile_seconds": 2,
                    "steady_median_seconds": 1,
                    "temporary_bytes": 1000,
                    "gpu_peak_steady_bytes": 2000,
                    "rtrl_carry_bytes": "" if method == "bptt" else 3000,
                    "gpu_power_steady_median_watts": 100,
                    "gradient_max_rel_error": 1e-10,
                    "loss_max_abs_error": 1e-12,
                }
                if backsub is not None:
                    row["backsub"] = backsub
                writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
