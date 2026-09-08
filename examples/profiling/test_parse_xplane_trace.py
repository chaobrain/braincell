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

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from examples.profiling.parse_xplane_trace import (
    _latest_xplane,
    summarize_dhs_level_scopes,
    summarize_trace_viewer_json,
)


class TestParseXPlaneTrace(unittest.TestCase):
    """Tests for BrainCell scope attribution from profiler traces."""

    def test_summarize_trace_leaf_uses_deepest_braincell_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_trace(Path(tmpdir))

            summary = summarize_trace_viewer_json(path, mode="leaf")

        self.assertEqual(summary.device_planes, ("/device:GPU:0",))
        self.assertEqual(summary.total_gpu_events, 2)
        self.assertEqual(summary.total_gpu_time_ps, 30_000)
        self.assertEqual(summary.matched_gpu_events, 2)
        self.assertEqual(summary.matched_gpu_time_ps, 30_000)
        self.assertEqual(
            [(row.scope, row.calls, row.total_ps) for row in summary.scopes],
            [
                ("braincell:dhs:linearize_membrane_current", 1, 20_000),
                ("braincell:cell_update:solver", 1, 10_000),
            ],
        )
        self.assertEqual(
            [(row.name, row.total_ps) for row in summary.kernels], [("fusion_b", 20_000), ("fusion_a", 10_000)]
        )

    def test_summarize_trace_inclusive_charges_all_braincell_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_trace(Path(tmpdir))

            summary = summarize_trace_viewer_json(path, mode="inclusive")

        rows = {row.scope: (row.calls, row.total_ps) for row in summary.scopes}
        self.assertEqual(
            rows,
            {
                "braincell:cell_run:update_dynamics": (2, 30_000),
                "braincell:cell_update:solver": (2, 30_000),
                "braincell:dhs:linearize_membrane_current": (1, 20_000),
            },
        )

    def test_summarize_xplane_trace_dir_finds_nested_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "plugins" / "profile" / "run"
            nested.mkdir(parents=True)
            path = _write_trace(nested)

            summary = summarize_trace_viewer_json(
                _latest_xplane(Path(tmpdir)),
                mode="leaf",
            )

        self.assertEqual(summary.xplane_path, path)
        self.assertEqual(
            summary.scopes[0].scope,
            "braincell:dhs:linearize_membrane_current",
        )

    def test_summarize_dhs_level_scopes_extracts_work_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_dhs_level_trace(Path(tmpdir))

            summary = summarize_trace_viewer_json(
                path,
                mode="leaf",
                scope_prefix="braincell:dhs_toy",
            )
            rows = summarize_dhs_level_scopes(summary)

        self.assertEqual(
            [(row.phase, row.level, row.width, row.popsize, row.work_items) for row in rows],
            [
                ("forward", 1, 2, 32, 64),
                ("forward", 2, 4, 32, 128),
                ("root", 0, 1, 32, 32),
                ("backward", 1, 2, 32, 64),
            ],
        )
        totals = {(row.phase, row.level): row.total_ps for row in rows}
        self.assertEqual(totals[("forward", 2)], 40_000)

    def test_summarize_real_dhs_level_scopes_extracts_kernel_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_real_dhs_level_trace(Path(tmpdir))

            summary = summarize_trace_viewer_json(
                path,
                mode="leaf",
                scope_prefix="braincell:dhs",
            )
            rows = summarize_dhs_level_scopes(summary)

        self.assertEqual(
            [(row.phase, row.level, row.width, row.popsize, row.work_items) for row in rows],
            [
                ("forward_level", 0, 256, 32, 8192),
                ("forward_level", 1, 64, 32, 2048),
            ],
        )
        self.assertEqual(rows[0].grid, "4,1,1")
        self.assertEqual(rows[0].block, "256,1,1")
        self.assertAlmostEqual(rows[0].occ_pct, 75.0)
        self.assertEqual(summary.kernels[0].grid, "4,1,1")
        self.assertEqual(summary.kernels[0].block, "256,1,1")
        self.assertAlmostEqual(summary.kernels[0].mean_occ_pct, 75.0)


def _write_trace(tmp_path: Path) -> Path:
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "fusion_a",
                "dur": 0.01,
                "args": {
                    "kernel_details": "regs:1",
                    "name": (
                        "jit(scan)/jit(main)/while/body/braincell:cell_run:update_dynamics/braincell:cell_update:solver"
                    ),
                },
            },
            {
                "ph": "X",
                "pid": 1,
                "name": "fusion_b",
                "dur": 0.02,
                "args": {
                    "kernel_details": "regs:1",
                    "name": (
                        "jit(scan)/jit(main)/while/body/"
                        "braincell:cell_run:update_dynamics/"
                        "braincell:cell_update:solver/"
                        "braincell:dhs:linearize_membrane_current"
                    ),
                },
            },
        ]
    }
    path = tmp_path / "test.xplane.pb"
    path.write_text(json.dumps(trace))
    return path


def _write_dhs_level_trace(tmp_path: Path) -> Path:
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            _trace_event("braincell:dhs_toy:forward:level=02:width=000004:pop=32", 0.04),
            _trace_event("braincell:dhs_toy:forward:level=01:width=000002:pop=32", 0.02),
            _trace_event("braincell:dhs_toy:root:level=00:width=000001:pop=32", 0.01),
            _trace_event("braincell:dhs_toy:backward:level=01:width=000002:pop=32", 0.03),
        ]
    }
    path = tmp_path / "dhs.xplane.pb"
    path.write_text(json.dumps(trace))
    return path


def _write_real_dhs_level_trace(tmp_path: Path) -> Path:
    trace = {
        "traceEvents": [
            {
                "ph": "M",
                "pid": 1,
                "name": "process_name",
                "args": {"name": "/device:GPU:0"},
            },
            _trace_event(
                "braincell:dhs:forward_level:i=000:edges=000256:batch=32",
                0.08,
                kernel_details=("regs:16 static_shared:0 dynamic_shared:0 grid:4,1,1 block:256,1,1 occ_pct:75"),
            ),
            _trace_event(
                "braincell:dhs:forward_level:i=001:edges=000064:batch=32",
                0.06,
                kernel_details=("regs:16 static_shared:0 dynamic_shared:0 grid:1,1,1 block:64,1,1 occ_pct:25"),
            ),
        ]
    }
    path = tmp_path / "real_dhs.xplane.pb"
    path.write_text(json.dumps(trace))
    return path


def _trace_event(scope: str, duration_us: float, *, kernel_details: str = "regs:1") -> dict:
    return {
        "ph": "X",
        "pid": 1,
        "name": "fusion",
        "dur": duration_us,
        "args": {
            "kernel_details": kernel_details,
            "name": f"jit(main)/{scope}",
        },
    }


if __name__ == "__main__":
    unittest.main()
