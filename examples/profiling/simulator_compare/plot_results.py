"""Plot measured and extrapolated simulator timing results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from common import read_json

STYLES = {
    "braincell": {"color": "#0072B2", "label": "BrainCell GPU"},
    "jaxley": {"color": "#009E73", "label": "Jaxley GPU"},
    "neuron": {"color": "#D55E00", "label": "NEURON CPU, serial extrapolated"},
}


def speedup_values(run: dict, neuron_run: dict) -> dict[str, float]:
    """Return NEURON-relative median and denominator-derived IQR bounds."""
    if int(run["batch_size"]) != int(neuron_run["batch_size"]):
        raise ValueError("speedup runs must have the same batch size")
    timing = run["timing"]
    reference = float(neuron_run["timing"]["median_seconds"])
    if run["backend"] == "neuron":
        return {"median": 1.0, "q1": 1.0, "q3": 1.0}
    median = float(timing["median_seconds"])
    q1 = float(timing["q1_seconds"])
    q3 = float(timing["q3_seconds"])
    if min(reference, median, q1, q3) <= 0.0:
        raise ValueError("speedup timing values must be positive")
    return {
        "median": reference / median,
        "q1": reference / q3,
        "q3": reference / q1,
    }


def plot_results(result_path: Path, output_path: Path) -> Path:
    payload = read_json(result_path)
    if payload.get("accuracy") and payload["accuracy"].get("passed") is False:
        raise RuntimeError("refusing to plot performance results because the accuracy gate failed")
    neuron_by_size = {
        int(run["batch_size"]): run
        for run in payload["runs"]
        if run["backend"] == "neuron"
    }
    batch_sizes = {int(run["batch_size"]) for run in payload["runs"]}
    missing = sorted(batch_sizes - neuron_by_size.keys())
    if missing:
        raise ValueError(f"missing NEURON timing reference for batch sizes {missing!r}")
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for backend, style in STYLES.items():
        runs = sorted((run for run in payload["runs"] if run["backend"] == backend), key=lambda run: run["batch_size"])
        if not runs:
            continue
        measured = [run for run in runs if run["timing"]["measured"]]
        extrapolated = [run for run in runs if not run["timing"]["measured"]]
        values = {
            int(run["batch_size"]): speedup_values(
                run, neuron_by_size[int(run["batch_size"])]
            )
            for run in runs
        }
        ax.plot(
            [run["batch_size"] for run in runs],
            [values[int(run["batch_size"])]["median"] for run in runs],
            color=style["color"],
            linestyle="--" if backend == "neuron" else "-",
            linewidth=1.6,
            label=style["label"],
        )
        for group, filled in ((measured, True), (extrapolated, False)):
            if not group:
                continue
            x = [run["batch_size"] for run in group]
            speedups = [values[int(run["batch_size"])] for run in group]
            y = [value["median"] for value in speedups]
            lower = [value["median"] - value["q1"] for value in speedups]
            upper = [value["q3"] - value["median"] for value in speedups]
            ax.errorbar(
                x,
                y,
                yerr=[lower, upper],
                fmt="o",
                markersize=5,
                markerfacecolor=style["color"] if filled else "white",
                markeredgecolor=style["color"],
                color=style["color"],
                capsize=2,
            )
        label_runs = runs[-1:] if backend == "neuron" else runs
        for run in label_runs:
            batch_size = int(run["batch_size"])
            speedup = values[batch_size]["median"]
            peer_speedups = [
                speedup_values(peer, neuron_by_size[batch_size])["median"]
                for peer in payload["runs"]
                if int(peer["batch_size"]) == batch_size
                and peer["backend"] in {"braincell", "jaxley"}
            ]
            y_offset = (
                8
                if backend == "neuron"
                or speedup < 2.0
                or speedup == max(peer_speedups)
                else -14
            )
            ax.annotate(
                f"{speedup:.1f}x",
                xy=(run["batch_size"], speedup),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                color=style["color"],
                fontsize=8,
                fontweight="semibold",
                bbox={
                    "boxstyle": "square,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
        rows.extend(
            {
                "backend": backend,
                "batch_size": run["batch_size"],
                "median_seconds": run["timing"]["median_seconds"],
                "q1_seconds": run["timing"]["q1_seconds"],
                "q3_seconds": run["timing"]["q3_seconds"],
                "neuron_reference_median_seconds": neuron_by_size[
                    int(run["batch_size"])
                ]["timing"]["median_seconds"],
                "speedup_vs_neuron": values[int(run["batch_size"])]["median"],
                "speedup_q1": values[int(run["batch_size"])]["q1"],
                "speedup_q3": values[int(run["batch_size"])]["q3"],
                "measured": run["timing"]["measured"],
                "extrapolated_from": run["timing"].get("extrapolated_from", ""),
                "cell_steps_per_second": run["timing"].get("cell_steps_per_second", ""),
                "host_transfer_median_seconds": (
                    run.get("host_transfer") or {}
                ).get("median_seconds", ""),
                "peak_gpu_memory_mib": (
                    run.get("device_memory") or {}
                ).get("peak_mib_in_use", ""),
                "trace_output_bytes": run.get("output_validation", {}).get(
                    "trace_output_bytes", ""
                ),
            }
            for run in runs
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Independent cells")
    ax.set_ylabel("Speedup vs. NEURON (x)")
    ax.grid(True, which="both", linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)
    with output_path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.result, args.output)


if __name__ == "__main__":
    main()
