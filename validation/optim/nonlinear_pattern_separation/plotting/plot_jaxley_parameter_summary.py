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
# ===============================================================================

"""Summarize Jaxley parameter changes across all saved restarts."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JAXLEY_DIR = (
    WORKFLOW_ROOT
    / "artifacts"
    / "nonlinear_pattern_separation_2026-09-15"
    / "raw"
    / "jaxley"
)
DEFAULT_OUTPUT_DIR = (
    WORKFLOW_ROOT
    / "artifacts"
    / "nonlinear_pattern_separation_2026-09-15"
    / "analysis"
    / "figures"
)

_PARAMETERS = {
    "radius": ("radius", "um"),
    "length": ("length", "um"),
    "axial_resistivity": ("Ra", "ohm cm"),
    "HH_gNa": ("gNa", "S/cm^2"),
    "HH_gK": ("gK", "S/cm^2"),
    "HH_gLeak": ("gLeak", "S/cm^2"),
}


def _load_parameter_arrays(jaxley_dir):
    with (jaxley_dir / "parameters.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    initial = saved["initial"]
    final = saved["final"]
    if len(initial) != 10 or len(final) != 10:
        raise ValueError(
            f"Expected 10 Jaxley restarts, found initial={len(initial)}, final={len(final)}."
        )

    arrays = {}
    for name in _PARAMETERS:
        before = np.asarray([
            next(group[name] for group in restart if name in group)
            for restart in initial
        ], dtype=float)
        after = np.asarray([
            next(group[name] for group in restart if name in group)
            for restart in final
        ], dtype=float)
        if before.shape != (10, 12) or after.shape != (10, 12):
            raise ValueError(
                f"Unexpected shape for {name}: before={before.shape}, after={after.shape}."
            )
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError(f"Non-finite values found for parameter {name}.")
        arrays[name] = (before, after)
    return arrays


def plot_parameter_summary(jaxley_dir, output_dir):
    """Generate the Jaxley before/after parameter summary figure."""
    arrays = _load_parameter_arrays(jaxley_dir)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.arange(12)
    width = 0.34
    for axis, name in zip(axes.flat, _PARAMETERS):
        before, after = arrays[name]
        before_mean = before.mean(axis=0)
        after_mean = after.mean(axis=0)
        before_error = np.vstack((
            np.maximum(before_mean - before.min(axis=0), 0.0),
            np.maximum(before.max(axis=0) - before_mean, 0.0),
        ))
        after_error = np.vstack((
            np.maximum(after_mean - after.min(axis=0), 0.0),
            np.maximum(after.max(axis=0) - after_mean, 0.0),
        ))
        axis.bar(
            x - width / 2,
            before_mean,
            width,
            yerr=before_error,
            capsize=2,
            color="#777777",
            alpha=0.85,
            label="before training",
        )
        axis.bar(
            x + width / 2,
            after_mean,
            width,
            yerr=after_error,
            capsize=2,
            color="#b33b32",
            alpha=0.85,
            label="after training",
        )
        axis.axvline(3.5, color="#bbbbbb", linewidth=0.8)
        axis.axvline(7.5, color="#bbbbbb", linewidth=0.8)
        axis.set_title(_PARAMETERS[name][0])
        axis.set_ylabel(_PARAMETERS[name][1])
        axis.set_xlabel("CV position")
        axis.set_xticks(x)
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Jaxley parameter summary across 10 restarts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "jaxley_parameter_before_after_summary.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaxley-dir", type=Path, default=DEFAULT_JAXLEY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(plot_parameter_summary(args.jaxley_dir, args.output_dir))


if __name__ == "__main__":
    main()
