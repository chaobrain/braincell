"""Metric helpers shared by cable compare and reporting."""

from typing import Any

import numpy as np


def ensure_2d(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={array.shape!r}.")
    return array


def metric_record(abs_diff: np.ndarray, *, reference: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    max_abs = float(np.max(abs_diff))
    ref_scale = float(np.mean(np.abs(reference)))
    if ref_scale <= 1e-12:
        rel_mae_pct = 0.0 if mae <= 1e-12 else float("inf")
    else:
        rel_mae_pct = float((mae / ref_scale) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "rel_mae_pct": rel_mae_pct,
    }


def metric_record_for_pair(braincell_trace: np.ndarray, neuron_trace: np.ndarray) -> dict[str, float]:
    if braincell_trace.shape != neuron_trace.shape:
        raise ValueError(
            "braincell and NEURON trace shapes do not match: "
            f"{braincell_trace.shape!r} vs {neuron_trace.shape!r}."
        )
    abs_diff = np.abs(braincell_trace - neuron_trace)
    return metric_record(abs_diff, reference=neuron_trace)
