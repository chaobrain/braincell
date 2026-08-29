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

"""Generate a local Markdown report from stored RTRL/BPTT benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_scaling"
KNOWN_RUNS = (
    "pilot_block_exact",
    "full_block_exact",
    "large_cv_block_exact",
    "backsub_ordinary_block_exact",
)


def load_result_rows(artifact_root: Path) -> list[dict[str, object]]:
    """Load successful rows from every known run directory."""
    rows = []
    for name in KNOWN_RUNS:
        result_path = artifact_root / name / "results.csv"
        if not result_path.exists():
            continue
        with result_path.open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = {key: _coerce(value) for key, value in raw.items()}
                row["run"] = name
                row["backsub"] = row.get("backsub") or ("ordinary" if "ordinary" in name else "recursive")
                if row.get("status") == "ok":
                    rows.append(row)
    return rows


def generate_report(artifact_root: Path) -> str:
    """Return the complete scaling report as Markdown."""
    rows = load_result_rows(artifact_root)
    if not rows:
        raise FileNotFoundError(f"No successful scaling results found under {artifact_root}.")
    recursive = _deduplicate([row for row in rows if row["backsub"] == "recursive"])
    ordinary = _deduplicate([row for row in rows if row["backsub"] == "ordinary"])
    lines = [
        "# RTRL/BPTT Scaling Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This file is generated from local ignored artifacts. CSV/NPZ files are the authoritative measurements.",
        "",
        "## Stored Runs",
        "",
    ]
    run_counts = []
    for name in KNOWN_RUNS:
        selected = [row for row in rows if row["run"] == name]
        if selected:
            run_counts.append((name, len(selected), len({row["config_id"] for row in selected})))
    lines.extend(_table(("Run", "Successful trials", "Configurations"), run_counts))
    lines.extend(
        [
            "",
            "## Model And Complexity",
            "",
            "Each seed owns `3C` Leak/Na/K per-CV parameter coordinates. A batch shares those parameters.",
            "Seeds are differentiated as independent blocks, so no cross-seed zero sensitivity is stored.",
            "",
            "```text",
            "Nx_active = 4 * B * C",
            "Ntheta = 3 * C",
            "minimal x64 carry = 96 * S * B * C^2 bytes",
            "Nz_full = (6B + 6) * C + 10",
            "measured full carry = S * Ntheta * Nz_full * 8 bytes",
            "```",
            "",
            "The logical carry is independent of rollout length `T`. BPTT temporary memory contains a temporal tape and grows with `T`.",
            "",
            "## Environment",
            "",
            "Stored GPU runs use an NVIDIA A100-SXM4-80GB, JAX 0.10.1, x64, `dt=0.025 ms`,",
            "`XLA_PYTHON_CLIENT_PREALLOCATE=false`, independent worker processes, and ten synchronized steady executions.",
            "Adam and target generation are excluded from gradient-kernel timings.",
            "",
            "## Full-Suite Reference Points",
            "",
        ]
    )
    reference_ids = ("c1_t40_b16_s16", "c5_t40_b16_s16", "c9_t40_b16_s16", "c9_t80_b32_s32")
    reference_rows = []
    for config_id in reference_ids:
        for method in ("bptt", "rtrl"):
            row = _find(recursive, config_id=config_id, method=method)
            if row:
                reference_rows.append(_summary_row(row))
    lines.extend(
        _table(
            ("Configuration", "Method", "Compile", "Steady", "XLA temporary", "Process peak", "RTRL carry"),
            reference_rows,
        )
    )
    lines.extend(["", "## Large-CV Axis", ""])
    cv_rows = []
    for n_cv in (1, 3, 5, 7, 9, 13, 17, 25, 33):
        for method in ("bptt", "rtrl"):
            row = _find(
                recursive,
                n_cv=n_cv,
                duration_ms=40.0,
                batch_size=16,
                n_seed=16,
                method=method,
            )
            if row:
                cv_rows.append(_cv_row(row))
    lines.extend(
        _table(
            ("CV", "Method", "Compile", "Steady", "XLA temporary", "Process peak", "RTRL carry", "Power median"),
            cv_rows,
        )
    )
    lines.extend(
        [
            "",
            "At `C=25` reverse BPTT crosses a measured execution regime boundary: runtime rises more sharply than temporary memory.",
            "RTRL carry follows the expected near-quadratic CV scaling; GPU wall time grows more slowly while parameter directions remain parallel.",
            "",
            "## Ordinary Hines Backsub A/B",
            "",
        ]
    )
    ab_rows = []
    for n_cv in (9, 17, 25, 33):
        for method in ("bptt", "rtrl"):
            rec = _find(recursive, n_cv=n_cv, duration_ms=40.0, batch_size=16, n_seed=16, method=method)
            ordinary_row = _find(ordinary, n_cv=n_cv, method=method)
            if rec and ordinary_row:
                ab_rows.append(
                    (
                        n_cv,
                        method.upper(),
                        _seconds(rec["steady_median_seconds"]),
                        _seconds(ordinary_row["steady_median_seconds"]),
                        f"{ordinary_row['steady_median_seconds'] / rec['steady_median_seconds']:.2f}x",
                        _bytes(rec["temporary_bytes"]),
                        _bytes(ordinary_row["temporary_bytes"]),
                    )
                )
    lines.extend(
        _table(
            (
                "CV",
                "Method",
                "Recursive",
                "Ordinary",
                "Ordinary/recursive",
                "Recursive temporary",
                "Ordinary temporary",
            ),
            ab_rows,
        )
    )
    lines.extend(
        [
            "",
            "Ordinary Hines backsub reduces BPTT temporary memory by roughly 28--35% but is 4--10% slower.",
            "For RTRL it provides almost no temporary-memory reduction and is 37--67% slower.",
            "The C17--C25 BPTT runtime transition remains under ordinary backsub, so recursive doubling is not its cause.",
            "",
            "## Numerical Agreement",
            "",
        ]
    )
    worst_gradient = max(float(row.get("gradient_max_rel_error") or 0.0) for row in rows)
    worst_loss = max(float(row.get("loss_max_abs_error") or 0.0) for row in rows)
    lines.extend(
        [
            f"Worst paired BPTT/RTRL relative gradient error: `{worst_gradient:.3e}`.",
            "",
            f"Worst paired BPTT/RTRL absolute loss error: `{worst_loss:.3e}`.",
            "",
            "The analysis notebook also compares ordinary and recursive NPZ outputs; the stored worst relative gradient difference is approximately `1.27e-8`.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python examples/experimental/online_learning/rtrl_bptt_scaling_benchmark.py run \\",
            "  --suite full --gpu 7 --repeats 10 \\",
            "  --python /home/swl/anaconda3/envs/braincell_311/bin/python \\",
            "  --output-dir examples/experimental/online_learning/artifacts/rtrl_bptt_scaling/full_block_exact \\",
            "  --resume",
            "",
            "python examples/experimental/online_learning/rtrl_bptt_scaling_report.py",
            "```",
            "",
            "Plots are stored in `../../rtrl_bptt_scaling_analysis.ipynb` relative to the artifact directories.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(artifact_root: Path, output: Path | None = None) -> Path:
    """Generate and write the report, returning its path."""
    output = artifact_root / "RESULTS.md" if output is None else output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_report(artifact_root), encoding="utf-8")
    return output


def _deduplicate(rows):
    by_key = {}
    for row in rows:
        by_key[(row["config_id"], row["method"], row["backsub"])] = row
    return tuple(by_key.values())


def _find(rows, **criteria):
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    return matches[-1] if matches else None


def _summary_row(row):
    return (
        row["config_id"],
        str(row["method"]).upper(),
        _seconds(row["compile_seconds"]),
        _seconds(row["steady_median_seconds"]),
        _bytes(row["temporary_bytes"]),
        _bytes(row["gpu_peak_steady_bytes"]),
        "-" if row.get("rtrl_carry_bytes") is None else _bytes(row["rtrl_carry_bytes"]),
    )


def _cv_row(row):
    return (
        int(row["n_cv"]),
        str(row["method"]).upper(),
        _seconds(row["compile_seconds"]),
        _seconds(row["steady_median_seconds"]),
        _bytes(row["temporary_bytes"]),
        _bytes(row["gpu_peak_steady_bytes"]),
        "-" if row.get("rtrl_carry_bytes") is None else _bytes(row["rtrl_carry_bytes"]),
        f"{float(row['gpu_power_steady_median_watts']):.1f} W",
    )


def _table(headers, rows):
    rows = tuple(tuple(str(value) for value in row) for row in rows)
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _seconds(value) -> str:
    return f"{float(value):.3f} s"


def _bytes(value) -> str:
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f} GB"
    if value >= 1e6:
        return f"{value / 1e6:.2f} MB"
    return f"{value / 1e3:.2f} KB"


def _coerce(value: str):
    if value == "" or value.lower() == "nan":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    print(write_report(args.artifact_root, args.output))


if __name__ == "__main__":
    main()
