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

"""Plot benchmark diagnostics from an existing aggregate JSON result."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from common import read_json


GPU_STYLES = {
    "braincell": {"color": "#0072B2", "label": "BrainCell GPU"},
    "jaxley": {"color": "#009E73", "label": "Jaxley GPU"},
}


def diagnostic_rows(payload: dict) -> list[dict]:
    """Derive directly comparable GPU metrics without modifying raw results."""
    rows = []
    for run in payload["runs"]:
        if run["backend"] not in GPU_STYLES or not run["timing"]["measured"]:
            continue
        timing = run["timing"]
        transfer = run.get("host_transfer") or {}
        output_bytes = run.get("output_validation", {}).get("trace_output_bytes")
        transfer_seconds = transfer.get("median_seconds")
        rows.append(
            {
                "backend": run["backend"],
                "batch_size": int(run["batch_size"]),
                "build_seconds": float(run["build_seconds"]),
                "compilation_seconds": float(run["compilation_seconds"]),
                "cold_start_seconds": float(run["build_seconds"])
                + float(run["compilation_seconds"]),
                "steady_median_seconds": float(timing["median_seconds"]),
                "steady_q1_seconds": float(timing["q1_seconds"]),
                "steady_q3_seconds": float(timing["q3_seconds"]),
                "steady_relative_iqr_percent": 100.0
                * (float(timing["q3_seconds"]) - float(timing["q1_seconds"]))
                / float(timing["median_seconds"]),
                "cell_steps_per_second": float(timing["cell_steps_per_second"]),
                "peak_gpu_memory_mib": float(run["device_memory"]["peak_mib_in_use"]),
                "host_transfer_median_seconds": transfer_seconds,
                "host_transfer_q1_seconds": transfer.get("q1_seconds"),
                "host_transfer_q3_seconds": transfer.get("q3_seconds"),
                "trace_output_bytes": output_bytes,
                "effective_transfer_gb_per_second": (
                    float(output_bytes) / float(transfer_seconds) / 1e9
                    if output_bytes is not None and transfer_seconds
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["backend"], row["batch_size"]))


def _backend_rows(rows: list[dict], backend: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["backend"] == backend),
        key=lambda row: row["batch_size"],
    )


def _style_axis(ax, *, log_x: bool = True, log_y: bool = False) -> None:
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.25)


def draw_cold_start(ax, rows: list[dict]) -> None:
    batch_sizes = sorted({row["batch_size"] for row in rows})
    positions = np.arange(len(batch_sizes), dtype=float)
    width = 0.36
    available = [backend for backend in GPU_STYLES if _backend_rows(rows, backend)]
    offsets = [0.0] if len(available) == 1 else [-width / 2, width / 2]
    for offset, backend in zip(offsets, available):
        backend_rows = _backend_rows(rows, backend)
        by_size = {row["batch_size"]: row for row in backend_rows}
        builds = [by_size[size]["build_seconds"] for size in batch_sizes]
        compiles = [by_size[size]["compilation_seconds"] for size in batch_sizes]
        color = GPU_STYLES[backend]["color"]
        ax.bar(positions + offset, builds, width, color=color, alpha=0.9)
        ax.bar(
            positions + offset,
            compiles,
            width,
            bottom=builds,
            color=color,
            alpha=0.42,
            hatch="//",
        )
    ax.set_xticks(positions, [str(size) for size in batch_sizes])
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Seconds")
    ax.set_title("Cold start: build + first compiled call")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.legend(
        handles=[
            *[
                Patch(color=GPU_STYLES[backend]["color"], label=GPU_STYLES[backend]["label"])
                for backend in available
            ],
            Patch(facecolor="#777777", label="Build"),
            Patch(facecolor="#bbbbbb", hatch="//", label="Compile + first call"),
        ],
        frameon=False,
        fontsize=8,
        ncol=2,
    )


def draw_memory(ax, rows: list[dict]) -> None:
    for backend, style in GPU_STYLES.items():
        values = _backend_rows(rows, backend)
        ax.plot(
            [row["batch_size"] for row in values],
            [row["peak_gpu_memory_mib"] for row in values],
            marker="o",
            linewidth=1.8,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Peak allocator memory (MiB)")
    ax.set_title("Peak GPU memory (one high-water mark per run)")
    _style_axis(ax, log_y=True)
    ax.legend(frameon=False, fontsize=8)


def draw_throughput(ax, rows: list[dict]) -> None:
    for backend, style in GPU_STYLES.items():
        values = _backend_rows(rows, backend)
        ax.plot(
            [row["batch_size"] for row in values],
            [row["cell_steps_per_second"] for row in values],
            marker="o",
            linewidth=1.8,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Cell-steps / second")
    ax.set_title("Steady-state throughput")
    _style_axis(ax, log_y=True)
    ax.legend(frameon=False, fontsize=8)


def draw_transfer(ax, rows: list[dict]) -> None:
    for backend, style in GPU_STYLES.items():
        values = _backend_rows(rows, backend)
        medians = [1000.0 * row["host_transfer_median_seconds"] for row in values]
        lower = [
            1000.0
            * (row["host_transfer_median_seconds"] - row["host_transfer_q1_seconds"])
            for row in values
        ]
        upper = [
            1000.0
            * (row["host_transfer_q3_seconds"] - row["host_transfer_median_seconds"])
            for row in values
        ]
        ax.errorbar(
            [row["batch_size"] for row in values],
            medians,
            yerr=[lower, upper],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Device-to-host time (ms)")
    ax.set_title("Full recorded-output transfer (median and IQR, n=3)")
    _style_axis(ax, log_y=True)
    ax.legend(frameon=False, fontsize=8)


def draw_variability(ax, rows: list[dict]) -> None:
    for backend, style in GPU_STYLES.items():
        values = _backend_rows(rows, backend)
        ax.plot(
            [row["batch_size"] for row in values],
            [row["steady_relative_iqr_percent"] for row in values],
            marker="o",
            linewidth=1.8,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("IQR / median (%)")
    ax.set_title("Steady-state timing variability (n=7)")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8)


def draw_bandwidth(ax, rows: list[dict]) -> None:
    for backend, style in GPU_STYLES.items():
        values = _backend_rows(rows, backend)
        ax.plot(
            [row["batch_size"] for row in values],
            [row["effective_transfer_gb_per_second"] for row in values],
            marker="o",
            linewidth=1.8,
            color=style["color"],
            label=style["label"],
        )
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Effective GB/s")
    ax.set_title("Effective output-transfer rate")
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=8)


PANELS = {
    "cold_start": draw_cold_start,
    "memory": draw_memory,
    "throughput": draw_throughput,
    "transfer": draw_transfer,
    "variability": draw_variability,
    "bandwidth": draw_bandwidth,
}


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(payload: dict, path: Path) -> Path | None:
    accuracy = payload.get("accuracy") or {}
    comparisons = accuracy.get("comparisons") or {}
    available = [backend for backend in GPU_STYLES if backend in comparisons]
    if not available:
        return None
    probes = [probe["branch"] for probe in comparisons[available[0]]["probes"]]
    positions = np.arange(len(probes), dtype=float)
    width = 0.36
    offsets = [0.0] if len(available) == 1 else [-width / 2, width / 2]
    metrics = (
        ("rmse_mv", "Voltage RMSE (mV)", "max_rmse_mv"),
        (
            "spike_amplitude_delta_mv",
            "Spike peak difference (mV)",
            "max_spike_amplitude_delta_mv",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, (metric, ylabel, limit_key) in zip(axes, metrics):
        for offset, backend in zip(offsets, available):
            values = [probe[metric] for probe in comparisons[backend]["probes"]]
            ax.bar(
                positions + offset,
                values,
                width,
                color=GPU_STYLES[backend]["color"],
                label=GPU_STYLES[backend]["label"],
            )
        limit = accuracy.get("limits", {}).get(limit_key)
        if limit is not None:
            ax.axhline(limit, color="#555555", linestyle="--", linewidth=1.2, label="Gate limit")
        ax.set_yscale("log")
        ax.set_xticks(positions, [str(probe) for probe in probes])
        ax.set_xlabel("Recorded branch")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", which="both", linewidth=0.5, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Numerical agreement with NEURON (spike-time delta is 0 ms)")
    _save_figure(fig, path)
    return path


def plot_diagnostics(result_path: Path, output_prefix: Path) -> list[Path]:
    payload = read_json(result_path)
    if payload.get("accuracy") and payload["accuracy"].get("passed") is False:
        raise RuntimeError("refusing to plot diagnostics because the accuracy gate failed")
    rows = diagnostic_rows(payload)
    if not rows:
        raise ValueError("no measured GPU runs found")

    outputs = []
    for name, draw in PANELS.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        draw(ax, rows)
        path = output_prefix.parent / f"{output_prefix.name}_{name}.png"
        _save_figure(fig, path)
        outputs.append(path)

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.0), constrained_layout=True)
    for ax, draw in zip(axes.flat, PANELS.values()):
        draw(ax, rows)
    dashboard = output_prefix.parent / f"{output_prefix.name}_diagnostics.png"
    _save_figure(fig, dashboard)
    outputs.append(dashboard)

    accuracy_path = output_prefix.parent / f"{output_prefix.name}_accuracy.png"
    if plot_accuracy(payload, accuracy_path) is not None:
        outputs.append(accuracy_path)

    csv_path = output_prefix.parent / f"{output_prefix.name}_diagnostics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(csv_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    for output in plot_diagnostics(args.result, args.output_prefix):
        print(output)


if __name__ == "__main__":
    main()
