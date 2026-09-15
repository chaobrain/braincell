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

"""Plot the three full-HH slices from saved paired results without running models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
LABELS = (
    r"Leak: $N_\theta=C=N_x/4$",
    r"Leak+K: $N_\theta=2C=N_x/2$",
    r"Leak+K+Na: $N_\theta=3C=3N_x/4$",
)


def _load(input_dir):
    with (input_dir / "raw" / "paired_results.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    groups = []
    seen = set()
    for multiplier in (1, 2, 3):
        group = sorted(
            [r for r in pairs if int(r["parameter_multiplier"]) == multiplier],
            key=lambda r: int(r["n_x"]),
        )
        if not group:
            raise ValueError(f"Missing HH slice {multiplier}C")
        for row in group:
            key = (int(row["n_x"]), multiplier)
            if key in seen:
                raise ValueError(f"Duplicate slice point: {key}")
            seen.add(key)
            if (row["usable_for_speed"].lower() != "true"
                    or row["bptt_status"] != "ok" or row["rtrl_status"] != "ok"):
                raise ValueError(f"Pair is incomplete or invalid: {row['config_id']}")
            if int(row["n_x"]) != 4 * int(row["n_cv"]) or int(row["n_theta"]) != multiplier * int(row["n_cv"]):
                raise ValueError(f"Point is outside its declared HH slice: {key}")
        groups.append(group)
    if len(seen) != len(pairs):
        raise ValueError("Unexpected parameter multiplier")
    return groups


def _values(group, field, scale=1):
    values = np.array([float(row[field]) / scale for row in group])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"Expected finite positive {field}")
    return values


def _legend():
    return [Line2D([], [], color=c, marker=m, label=l)
            for c, m, l in zip(COLORS, MARKERS, LABELS)]


def _style_axis(ax, ticks):
    ax.set_xlabel(r"$N_x$ (active states per trajectory)")
    ax.set_xticks(ticks)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, output, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _geometry(groups, output, context):
    fig, ax = plt.subplots(figsize=(10, 6.8))
    fig.subplots_adjust(top=.84, bottom=.21)
    fig.suptitle("HH scan: three parameter slices", fontsize=17, y=.98)
    fig.text(.5, .922, context, ha="center", fontsize=10, color="#555555")
    ticks = sorted({int(r["n_x"]) for g in groups for r in g})
    for group, color, marker, label in zip(groups, COLORS, MARKERS, LABELS):
        ax.plot(_values(group, "n_x"), _values(group, "n_theta"),
                color=color, marker=marker, ms=8, lw=1.7, label=label)
        for row in group:
            if int(row["n_cv"]) == 1:
                continue
            ax.annotate(f"r={float(row['rtrl_over_bptt_time']):.3f}",
                        (int(row["n_x"]), int(row["n_theta"])),
                        xytext=(5, 8), textcoords="offset points", fontsize=9, color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1})
    _style_axis(ax, ticks)
    ax.set_ylabel(r"$N_\theta$ (trainable parameters per seed)")
    ax.set_xlim(0, max(ticks) * 1.12)
    ax.set_ylim(0, max(float(r["n_theta"]) for g in groups for r in g) * 1.12)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    if all(int(g[0]["n_cv"]) == 1 for g in groups):
        inset = ax.inset_axes([.10, .46, .22, .25])
        for group, color, marker in zip(groups, COLORS, MARKERS):
            row = group[0]
            inset.plot(4, int(row["n_theta"]), marker=marker, color=color, ms=7)
            inset.annotate(f"r={float(row['rtrl_over_bptt_time']):.3f}",
                           (4, int(row["n_theta"])), xytext=(10, 0),
                           textcoords="offset points", va="center", fontsize=9, color=color)
        inset.set(xlim=(3.85, 4.9), ylim=(.5, 3.5), xticks=[4], yticks=[1, 2, 3])
        inset.set_title("C=1 detail", fontsize=10)
        inset.grid(alpha=.15)
    fig.text(.125, .055,
             r"Labels: $r=t_{RTRL}/t_{BPTT}$; r < 1 favors RTRL. Lines connect sampled slice locations only."
             "\nOnly markers are measured; no crossover boundary has been inferred.", fontsize=9, color="#555555")
    _save(fig, output, "hh_slices_nx_ntheta")


def _panels(groups, input_dir, output, context, costs=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7))
    fig.subplots_adjust(top=.73, bottom=.21, wspace=.27)
    title = "HH memory and compilation" if costs else "HH steady loss + gradient performance"
    fig.suptitle(title, fontsize=17, y=.99)
    fig.text(.5, .93, context, ha="center", fontsize=10, color="#555555")
    handles = _legend() + [
        Line2D([], [], color="#333333", lw=2, label="BPTT"),
        Line2D([], [], color="#333333", lw=2, ls="--", label="RTRL"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .90),
               ncol=3, frameon=False, fontsize=9)
    ticks = sorted({int(r["n_x"]) for g in groups for r in g})
    for group, color, marker in zip(groups, COLORS, MARKERS):
        x = _values(group, "n_x")
        for method, ls in (("bptt", "-"), ("rtrl", "--")):
            if costs:
                for ax, field, scale in zip(axes, ("temporary_bytes", "compile_seconds"), (2**20, 1)):
                    ax.plot(x, _values(group, f"{method}_{field}", scale),
                            color=color, ls=ls, marker=marker, ms=6)
            else:
                median = _values(group, f"{method}_steady_median_seconds")
                quartiles = []
                for row in group:
                    trial = json.loads((input_dir / "raw" / "trials" / f"{row['config_id']}__{method}.json").read_text())
                    samples = np.array(trial["steady_seconds"], dtype=float)
                    if len(samples) < 2 or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
                        raise ValueError("Invalid saved timing samples")
                    if not np.isclose(np.median(samples), float(row[f"{method}_steady_median_seconds"]), rtol=1e-10):
                        raise ValueError("Paired median differs from saved samples")
                    quartiles.append(np.percentile(samples, [25, 75]))
                quartiles = np.array(quartiles)
                axes[0].errorbar(x, median, yerr=[median-quartiles[:, 0], quartiles[:, 1]-median],
                                 color=color, ls=ls, marker=marker, ms=6, capsize=3, lw=1.6)
        if not costs:
            axes[1].plot(x, _values(group, "rtrl_over_bptt_time"),
                         color=color, marker=marker, ms=7, lw=1.6)
    for ax in axes:
        _style_axis(ax, ticks)
    axes[0].set_yscale("log")
    if costs:
        axes[0].set_ylabel("XLA temporary memory (MiB; log scale)")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Gradient kernel compile (s; log scale)")
        note = "XLA analysis, not process peak GPU memory. Compile includes tracing/lowering; target preparation is excluded."
    else:
        axes[0].set_ylabel("Steady median (s; log scale)")
        axes[1].set_ylabel(r"$r=t_{RTRL}/t_{BPTT}$")
        axes[1].axhspan(.9, 1.1, color="#cccccc", alpha=.35)
        axes[1].axhline(1, color="#666666", lw=1, ls=":")
        axes[1].text(.03, .90, "Near candidate: 0.9–1.1", transform=axes[1].transAxes, fontsize=9)
        axes[1].set_ylim(0, max(1.16, max(float(r['rtrl_over_bptt_time']) for g in groups for r in g)*1.1))
        note = "Error bars: within-worker Q25–Q75 (often smaller than markers). r < 1 favors RTRL."
    fig.text(.09, .08, note + "\nLines are guides between measured points; no unmeasured crossover boundary is asserted.",
             fontsize=9, color="#555555")
    _save(fig, output, "hh_memory_compile" if costs else "hh_runtime_ratio")


def main(argv=None):
    """Render PNG, SVG and PDF figures from an existing paired-results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to INPUT_DIR/analysis/figures")
    parser.add_argument("--context", default="", help="Optional measured environment/protocol subtitle")
    args = parser.parse_args(argv)
    output = args.output_dir or args.input_dir / "analysis" / "figures"
    groups = _load(args.input_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "svg.fonttype": "none", "pdf.fonttype": 42})
    _geometry(groups, output, args.context)
    _panels(groups, args.input_dir, output, args.context)
    _panels(groups, args.input_dir, output, args.context, costs=True)
    sources = [args.input_dir / "raw" / "paired_results.csv"] + sorted((args.input_dir / "raw" / "trials").glob("*.json"))
    (output / "plot_sources.json").write_text(json.dumps({
        "input_dir": str(args.input_dir.resolve()), "new_model_executions": 0,
        "context": args.context,
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
