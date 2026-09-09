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

"""Matplotlib views of saved whole-cell comparisons; simulation is not required."""

from __future__ import annotations

from pathlib import Path

from .analysis import AnalysisResult, analyze
from .results import ComparisonResult


def plot_comparison(result: ComparisonResult, *, threshold_mV: float = 0.0):
    """Plot voltage overlays, spikes and voltage error.

    Parameters
    ----------
    result : ComparisonResult
        Simulation result to analyze.
    threshold_mV : float, optional
        Spike threshold in mV, default 0.

    Returns
    -------
    tuple
        Matplotlib Figure and two Axes, ready for customization.
    """
    return plot_analysis(analyze(result, threshold_mV=threshold_mV))


def plot_analysis(analysis: AnalysisResult):
    """Plot an already computed analysis.

    Parameters
    ----------
    analysis : AnalysisResult
        Aligned traces, errors and spike events.

    Returns
    -------
    tuple
        Matplotlib Figure and two Axes; no analysis or simulation is repeated.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    axes[0].plot(analysis.time_ms, analysis.neuron_voltage_mV, label="NEURON")
    axes[0].plot(analysis.time_ms, analysis.braincell_voltage_mV, "--", label="BrainCell")
    spikes = analysis.summary["spikes"]
    for backend, color, marker in (("neuron", "C0", "|"), ("braincell", "C1", "x")):
        times = spikes[f"{backend}_times_ms"]
        axes[0].scatter(times, [spikes["threshold_mV"]] * len(times), color=color, marker=marker)
    axes[0].set_ylabel("Soma voltage (mV)")
    axes[0].set_title(analysis.summary.get("model") or "Soma voltage comparison")
    axes[0].legend()
    axes[1].plot(analysis.time_ms, analysis.error_mV, color="C3")
    axes[1].axhline(0, color="0.6", linewidth=0.7)
    axes[1].set_ylabel("BrainCell − NEURON (mV)")
    axes[1].set_xlabel("Time (ms)")
    return figure, axes


def save_plot(figure, path: str | Path) -> Path:
    """Save a customized figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to export.
    path : str or pathlib.Path
        Destination; its suffix selects the format, such as PNG or PDF.

    Returns
    -------
    pathlib.Path
        Saved figure path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path
