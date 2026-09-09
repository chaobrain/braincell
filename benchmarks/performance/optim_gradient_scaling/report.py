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
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_scaling"
KNOWN_RUNS = (
    "pilot_block_exact",
    "full_block_exact",
    "large_cv_block_exact",
    "backsub_ordinary_block_exact",
    "mechanism_factorial_block_exact",
    "controlled_complexity_a100",
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
            "In the legacy per-CV full-HH suites, each seed owns `3C` Leak/Na/K parameter coordinates. A batch shares those parameters.",
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
            *_environment_table(artifact_root, rows),
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
            "",
            "## Controlled Complexity",
            "",
        ]
    )
    controlled_rows = [row for row in rows if row.get("run") == "controlled_complexity_a100"]
    controlled_groups = {}
    for row in controlled_rows:
        key = (row["config_id"], row["workload"], row["method"], int(row["n_x"]), int(row["n_theta"]))
        controlled_groups.setdefault(key, []).append(row)
    controlled_summary = []
    for (config_id, workload, method, n_x, n_theta), group in sorted(controlled_groups.items()):
        times = [float(row["steady_median_seconds"]) for row in group]
        controlled_summary.append(
            (
                config_id,
                workload,
                str(method).upper(),
                n_x,
                n_theta,
                len(group),
                _seconds(statistics.median(times)),
                f"{min(times):.4f}--{max(times):.4f}",
                _bytes(statistics.median(float(row["temporary_bytes"]) for row in group)),
            )
        )
    if controlled_summary:
        lines.extend(
            _table(
                (
                    "Configuration",
                    "Workload",
                    "Method",
                    "Nx",
                    "Ntheta",
                    "Workers",
                    "Median",
                    "Worker range",
                    "XLA temporary",
                ),
                controlled_summary,
            )
        )
    else:
        lines.append("No stored controlled-complexity run was found.")
    lines.extend(
        [
            "",
            "Synthetic rows independently control `Nx` and `Ntheta`; full-HH rows are application-level scaling and are not pure complexity proofs.",
            "",
            "## Mechanism Factorial Ablation",
            "",
        ]
    )
    factorial_rows = []
    for row in recursive:
        if row.get("run") != "mechanism_factorial_block_exact":
            continue
        factorial_rows.append(
            (
                int(row["n_cv"]),
                row["mechanism_case"],
                str(row["method"]).upper(),
                int(row["n_x"]),
                int(row["n_theta"]),
                _seconds(row["steady_median_seconds"]),
                _bytes(row["temporary_bytes"]),
                "-" if row.get("rtrl_carry_bytes") is None else _bytes(row["rtrl_carry_bytes"]),
            )
        )
    if factorial_rows:
        lines.extend(
            _table(
                ("CV", "Case", "Method", "Nx", "Ntheta", "Steady", "XLA temporary", "RTRL carry"),
                sorted(factorial_rows),
            )
        )
    else:
        lines.append("No stored mechanism-factorial ablation was found.")
    lines.extend(
        [
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
            "",
            "## Numerical Agreement",
            "",
        ]
    )
    for label, key in (
        ("relative gradient error", "gradient_max_rel_error"),
        ("relative L2 gradient error", "gradient_relative_l2_error"),
        ("absolute loss error", "loss_max_abs_error"),
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        value = f"{max(values):.3e}" if values else "not recorded"
        lines.extend([f"Worst recorded paired BPTT/RTRL {label}: `{value}`.", ""])
    lines.extend([
        "## Reproduction", "",
        "Use the experiment README for supported scripts and parameterized commands. "
        "Select devices and confirm all execution counts before starting a new measurement.", "",
        "Environment fields below describe stored records, not the reporting machine. "
        "A device ordinal alone does not identify its hardware model. "
        "Unavailable metadata is not reconstructed from historical run names.", "",
    ])
    return "\n".join(lines)


def _environment_table(artifact_root: Path, rows: list[dict]) -> list[str]:
    records = []
    for name in sorted({row["run"] for row in rows}):
        directory = artifact_root / name
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        trials = [json.loads(path.read_text()) for path in sorted((directory / "trials").glob("*.json"))]
        trials = [trial for trial in trials if trial.get("status") == "ok"]
        selected = [row for row in rows if row["run"] == name]
        def recorded(key):
            values = {str(record[key]) for record in [manifest, *trials, *selected]
                      if record.get(key) is not None and record[key] != ""}
            return "; ".join(sorted(values)).replace("|", "\\|").replace("\n", " ") or "not recorded"
        records.append((name, recorded("device"), recorded("jax_version"),
                        recorded("created_utc"), recorded("repeats"), recorded("replicates"),
                        recorded("dt_ms"), recorded("python_executable")))
    return _table(("Run", "Device record", "JAX", "Created UTC", "Timing repeats",
                   "Process replicates", "dt (ms)", "Interpreter"), records)


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
