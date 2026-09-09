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

"""Sample-time alignment, voltage errors and threshold-crossing comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .results import ComparisonResult, Trace


@dataclass(frozen=True)
class AnalysisResult:
    """Aligned voltage samples and a JSON-compatible metric summary."""

    time_ms: np.ndarray
    neuron_voltage_mV: np.ndarray
    braincell_voltage_mV: np.ndarray
    error_mV: np.ndarray
    summary: dict


def align_traces(neuron: Trace, braincell: Trace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match actual sample times, allowing only NEURON's extra initial t=0 point.

    Time tolerance is 1e-8 ms (absolute), accommodating accumulated fixed-step
    roundoff. Voltage values are never interpolated or shifted by array length.
    """
    if not np.isfinite(neuron.voltage_mV).all() or not np.isfinite(braincell.voltage_mV).all():
        raise ValueError("Nonfinite voltage samples; inspect the raw result before computing errors.")
    nt = neuron.time_ms
    bt = braincell.time_ms
    nv = neuron.voltage_mV
    if len(nt) == len(bt) + 1 and np.isclose(nt[0], 0.0, atol=1e-8, rtol=0) and bt[0] > 1e-8:
        nt, nv = nt[1:], nv[1:]
    if nt.shape != bt.shape or not np.allclose(nt, bt, atol=1e-8, rtol=0):
        raise ValueError("Sample times do not match; use the same fixed-step protocol on both simulators.")
    return bt.copy(), nv.copy(), braincell.voltage_mV.copy()


def spike_times(trace: Trace, threshold_mV: float = 0.0) -> np.ndarray:
    """Interpolate upward threshold crossings between consecutive recorded samples.

    Parameters
    ----------
    trace : Trace
        Recorded voltage and sample times.
    threshold_mV : float, optional
        Crossing threshold in mV, default 0.

    Returns
    -------
    numpy.ndarray
        Increasing event times in ms; empty when no upward crossing occurs.
    """
    if not np.isfinite(threshold_mV) or not np.isfinite(trace.voltage_mV).all():
        raise ValueError("Spike detection requires finite voltages and threshold.")
    voltage = trace.voltage_mV
    indices = np.flatnonzero((voltage[:-1] < threshold_mV) & (voltage[1:] >= threshold_mV))
    fraction = (threshold_mV - voltage[indices]) / (voltage[indices + 1] - voltage[indices])
    return trace.time_ms[indices] + fraction * np.diff(trace.time_ms)[indices]


def analyze(result: ComparisonResult, *, threshold_mV: float = 0.0) -> AnalysisResult:
    """Compute voltage errors and spike counts over the common recorded time window.

    Errors are BrainCell minus NEURON. Equal spike counts are paired in temporal
    order; unequal counts report separate event times without forcing a pairing.
    """
    time, nv, bv = align_traces(result.neuron, result.braincell)
    error = bv - nv
    ns = spike_times(Trace(time, nv), threshold_mV)
    bs = spike_times(Trace(time, bv), threshold_mV)
    paired = len(ns) == len(bs)
    summary = {
        "model": result.metadata.get("model"),
        "window_ms": [float(time[0]), float(time[-1])],
        "n_samples": len(time),
        "voltage": {
            "rmse_mV": float(np.sqrt(np.mean(error**2))),
            "mae_mV": float(np.mean(np.abs(error))),
            "max_abs_mV": float(np.max(np.abs(error))),
        },
        "spikes": {
            "threshold_mV": float(threshold_mV),
            "neuron_count": len(ns),
            "braincell_count": len(bs),
            "neuron_times_ms": ns.tolist(),
            "braincell_times_ms": bs.tolist(),
            "paired_by_order": paired,
            "time_error_ms": (bs - ns).tolist() if paired else None,
        },
    }
    return AnalysisResult(time, nv, bv, error, summary)


def save_analysis(analysis: AnalysisResult, directory: str | Path) -> Path:
    """Save metrics and aligned differences; repeated calls refresh derived files.

    Parameters
    ----------
    analysis : AnalysisResult
        Computed error arrays and metric summary.
    directory : str or pathlib.Path
        Destination for metrics.json and aligned.npz.

    Returns
    -------
    pathlib.Path
        Directory containing the derived files.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(analysis.summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(
        directory / "aligned.npz",
        time_ms=analysis.time_ms,
        neuron_voltage_mV=analysis.neuron_voltage_mV,
        braincell_voltage_mV=analysis.braincell_voltage_mV,
        error_mV=analysis.error_mV,
    )
    return directory
