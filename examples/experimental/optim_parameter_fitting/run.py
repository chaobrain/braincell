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

# ruff: noqa: E402

"""Run one trusted Python-composed parameter-learning experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] in {"run", "compare", "report", "archive", "list-runs"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import brainunit as u
import jax

from examples.experimental.optim_parameter_fitting.config import load_config
from examples.experimental.optim_parameter_fitting.reporting import (
    archive_completed_run,
    compare_completed_runs,
    save_run,
)
from examples.experimental.optim_parameter_fitting.training import run_pipeline

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "basic_1cv_bounded_direct_adam.py"
DEFAULT_RESULT_ROOT = Path(__file__).resolve().parent / "artifacts" / "parameter_experiments"


def run_worker(config_path: Path, output_dir: Path, *, resume: bool) -> dict[str, object]:
    """Generate target data, execute the pipeline, and write artifacts."""
    config, _module = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    resolved_path = output_dir / "resolved_config.json"
    if resume and summary_path.exists() and resolved_path.exists():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if resolved.get("config_digest") != config.digest():
            raise ValueError("Existing result config does not match the requested Python preset.")
        return json.loads(summary_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    with jax.enable_x64(True), brainstate.environ.context(dt=config.dataset.dt_ms * u.ms, precision=64):
        dataset = config.dataset.generate(config.model)
        result = run_pipeline(config, dataset)
        summary = save_run(
            output_dir,
            config,
            config_path,
            result,
            run_started_at=started,
        )
    return summary


def launch_worker(
    config_path: Path,
    output_dir: Path,
    *,
    gpu: int,
    python_executable: Path,
    resume: bool,
) -> None:
    """Launch an isolated CUDA worker and monitor physical GPU memory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    command = [
        str(python_executable),
        str(Path(__file__).resolve()),
        "worker",
        "--config",
        str(config_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment.pop("JAX_PLATFORMS", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    samples = []
    with (logs / "worker.log").open("a" if resume else "w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=_REPO_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            sample = _gpu_sample(gpu)
            if sample is not None:
                samples.append(sample)
            time.sleep(0.5)
        return_code = process.wait()
    if samples:
        with (logs / "gpu_monitor.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        monitor = {
            "gpu": gpu,
            "samples": len(samples),
            "peak_memory_used_mib": max(item["memory_used_mib"] for item in samples),
            "peak_utilization_percent": max(item["utilization_percent"] for item in samples),
        }
        (logs / "gpu_monitor.json").write_text(json.dumps(monitor, indent=2) + "\n", encoding="utf-8")
        if (output_dir / "summary.json").exists():
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            summary["gpu_monitor"] = monitor
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        report_path = output_dir / "REPORT.md"
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8").split("\n## GPU Monitor\n", 1)[0].rstrip()
            report += (
                "\n\n## GPU Monitor\n\n"
                f"- Physical GPU index: `{gpu}`\n"
                f"- Peak memory used: `{monitor['peak_memory_used_mib']} MiB`\n"
                f"- Peak utilization: `{monitor['peak_utilization_percent']}%`\n"
            )
            report_path.write_text(report, encoding="utf-8")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    rebuild_runs_index(DEFAULT_RESULT_ROOT)


def resolve_output_dir(
    config,
    config_path: Path,
    *,
    requested: Path | None,
    resume: bool,
    now: datetime | None = None,
) -> Path:
    """Resolve one readable timestamped run directory or an exact resume target."""
    if requested is not None:
        return Path(requested)
    matches = []
    if resume and DEFAULT_RESULT_ROOT.exists():
        for resolved in DEFAULT_RESULT_ROOT.rglob("resolved_config.json"):
            try:
                values = json.loads(resolved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if values.get("config_digest") == config.digest():
                matches.append(resolved.parent)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Multiple completed runs match this config; pass --output-dir explicitly.")
    timestamp = (datetime.now() if now is None else now).strftime("%Y%m%d-%H%M%S")
    label = config.artifact_label or Path(config_path).stem
    return DEFAULT_RESULT_ROOT / f"{timestamp}_{label}_{config.digest()[:8]}"


def rebuild_runs_index(root: Path) -> Path:
    """Rebuild a compact index for all complete result directories."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for resolved in sorted(root.rglob("resolved_config.json")):
        summary_path = resolved.parent / "summary.json"
        if not summary_path.exists():
            continue
        config = json.loads(resolved.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stage = config["stages"][-1]
        rows.append(
            {
                "run": resolved.parent.name,
                "path": str(resolved.parent.relative_to(root)),
                "digest": config.get("config_digest", ""),
                "model": config["model"]["name"],
                "loss": config["loss"]["name"],
                "optimizer": stage.get("optimizer", stage["name"]),
                "epochs": stage["epochs"],
                "trace_success": summary["trace_success"]["count"],
                "parameter_success": summary["parameter_success"]["count"],
                "joint_success": summary["joint_success"]["count"],
            }
        )
    path = root / "runs.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


def _gpu_sample(gpu: int) -> dict[str, object] | None:
    result = subprocess.run(
        (
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=timestamp,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        return None
    timestamp, memory, utilization = (item.strip() for item in result.stdout.strip().split(","))
    return {
        "timestamp": timestamp,
        "memory_used_mib": int(memory),
        "utilization_percent": int(utilization),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--gpu", type=int, default=1)
    run.add_argument("--python", type=Path, default=Path("/home/swl/anaconda3/envs/braincell_311/bin/python"))
    run.add_argument("--resume", action="store_true")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.add_argument("result_dir", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("baseline_dir", type=Path)
    compare.add_argument("extended_dir", type=Path)
    compare.add_argument("--output-dir", type=Path)
    compare.add_argument("--kind", choices=("epoch", "lr", "bounds", "lr_bounds", "optimizer", "loss"), default="epoch")
    archive = subparsers.add_parser("archive")
    archive.add_argument("--config", type=Path, required=True)
    archive.add_argument("result_dir", type=Path)
    subparsers.add_parser("list-runs")
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "worker":
        summary = run_worker(args.config, args.output_dir, resume=args.resume)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "report":
        print((args.result_dir / "REPORT.md").read_text(encoding="utf-8"))
        return
    if args.command == "compare":
        output_dir = args.output_dir or args.extended_dir / "comparisons" / args.baseline_dir.name
        comparison = compare_completed_runs(
            args.baseline_dir,
            args.extended_dir,
            output_dir,
            comparison_kind=args.kind,
        )
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return
    if args.command == "archive":
        config, _module = load_config(args.config)
        with jax.enable_x64(True), brainstate.environ.context(dt=config.dataset.dt_ms * u.ms, precision=64):
            summary = archive_completed_run(config, args.result_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.command == "list-runs":
        path = rebuild_runs_index(DEFAULT_RESULT_ROOT)
        print(path.read_text(encoding="utf-8") if path.exists() else "")
        return
    config, _module = load_config(args.config)
    output_dir = resolve_output_dir(
        config,
        args.config,
        requested=args.output_dir,
        resume=args.resume,
    )
    launch_worker(
        args.config,
        output_dir,
        gpu=args.gpu,
        python_executable=args.python,
        resume=args.resume,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
