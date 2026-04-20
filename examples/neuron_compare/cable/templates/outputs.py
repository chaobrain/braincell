"""Output helpers shared by cable sweep dispatch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(out_path: str | Path, payload: Any) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_sweep_output_dir(
    *,
    cable_root: str | Path,
    config_id: str,
) -> Path:
    return Path(cable_root) / "artifacts" / "sweeps" / config_id


def iter_metric_rows(metrics: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    return [("voltage", metrics["overall"])]


def write_case_metrics_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    fieldnames = [
        "case_id",
        "group_id",
        "status",
        "stimulus_kind",
        "morphology_kind",
        "morphology_path",
        "dt_ms",
        "cv_per_branch",
        "observable",
        "n_samples",
        "mae",
        "rmse",
        "max_abs",
        "rel_mae_pct",
        "error_message",
    ]
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_case_metrics(
    config,
    *,
    success_results: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    total_cases: int,
) -> dict[str, Any]:
    observable_rows: dict[str, list[dict[str, float]]] = {}
    for result in success_results:
        for observable, metric in iter_metric_rows(result["metrics"]):
            observable_rows.setdefault(observable, []).append(metric)

    observables = {
        observable: {
            "n_cases": len(rows),
            "mae_mean": float(sum(row["mae"] for row in rows) / len(rows)),
            "rmse_mean": float(sum(row["rmse"] for row in rows) / len(rows)),
            "max_abs_max": float(max(row["max_abs"] for row in rows)),
            "rel_mae_pct_mean": float(sum(row["rel_mae_pct"] for row in rows) / len(rows)),
        }
        for observable, rows in observable_rows.items()
    }

    return {
        "config_id": config.config_id,
        "template_family": "multi_compartment_cable",
        "n_total_cases": total_cases,
        "n_success_cases": len(success_results),
        "n_failed_cases": len(failed_cases),
        "observables": observables,
        "failed_cases": failed_cases,
    }


def save_case_plot(out_path: str | Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(result["time_ms"], dtype=float)
    braincell_voltage = np.asarray(result["braincell"]["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(result["neuron"]["voltage_mV"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for index in range(braincell_voltage.shape[1]):
        axes[0].plot(time_ms, neuron_voltage[:, index], linewidth=1.0, alpha=0.7)
        axes[0].plot(time_ms, braincell_voltage[:, index], linewidth=1.0, linestyle="--", alpha=0.7)
        axes[1].plot(
            time_ms,
            np.abs(braincell_voltage[:, index] - neuron_voltage[:, index]),
            linewidth=1.0,
            alpha=0.7,
        )
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("NEURON (solid) vs braincell (dashed)")
    axes[1].set_ylabel("|delta V| (mV)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_title("Per-compartment absolute error")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
