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

"""Plot saved Jaxley and BrainCell nonlinear-pattern outcomes."""

from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]

# The nonlinear-pattern dataset has three input regions.  These categorical
# colors are deliberately outside the blue/red ``coolwarm`` voltage map so
# that the sampled inputs remain visible on both sides of the decision edge.
_CLUSTER_CENTERS = np.asarray(((1.5, 3.5), (2.5, 2.5), (3.5, 1.5)), dtype=float)
_CLUSTER_COLORS = ("#00ff66", "#ffea00", "#ff00cc")

def _load_braincell(braincell_dir):
    rows = []
    for path in sorted(braincell_dir.glob("seed_*/surface.npz")):
        rows.append((int(path.parent.name.split("_")[1]), np.load(path), np.load(path.parent / "history.npz")))
    return rows

def _cluster_ids(inputs):
    inputs = np.asarray(inputs)
    distances = ((inputs[:, None, :] - _CLUSTER_CENTERS[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1)

def _plot_input_samples(axis, train_inputs, test_inputs):
    for inputs, filled in ((train_inputs, True), (test_inputs, False)):
        inputs = np.asarray(inputs)
        cluster_ids = _cluster_ids(inputs)
        for cluster_id, color in enumerate(_CLUSTER_COLORS):
            selected = cluster_ids == cluster_id
            axis.scatter(
                inputs[selected, 0],
                inputs[selected, 1],
                s=34,
                marker="o",
                facecolors=color if filled else "none",
                edgecolors=color if not filled else "black",
                linewidths=1.0,
                zorder=3,
            )

def _input_legend_handles():
    cluster_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor=color, markeredgecolor="black", label=f"region {idx + 1}")
        for idx, color in enumerate(_CLUSTER_COLORS)
    ]
    split_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor="#777777", markeredgecolor="black", label="train (filled)"),
        Line2D([0], [0], marker="o", linestyle="", markersize=7,
               markerfacecolor="none", markeredgecolor="#777777", label="test (open)"),
    ]
    return cluster_handles + split_handles

def _plot_surfaces(rows, jaxley_dir, output):
    jaxley = np.load(jaxley_dir / "history.npz")
    jv = np.asarray(jaxley["sweep_voltage_at_3ms_mv"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    extent = (0, 5, 0, 5)
    im = axes[0, 0].imshow(jv, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
    _plot_input_samples(axes[0, 0], np.asarray(jaxley["train_inputs"])[0], np.asarray(jaxley["test_inputs"])[0])
    axes[0, 0].set_title("Jaxley saved sweep\nrestart 0 parameters")
    fig.colorbar(im, ax=axes[0, 0], label="V(3 ms), mV")
    axes[1, 0].imshow(np.ma.masked_where(~np.isfinite(jv), jv > -17.5), origin="lower", extent=extent, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
    axes[1, 0].set_title("Jaxley 3 ms voltage class (> -17.5 mV)")
    for col, (seed, surface, _history) in enumerate(rows[:2], start=1):
        voltage = np.asarray(surface["voltage"])
        im = axes[0, col].imshow(voltage, origin="lower", extent=extent, aspect="equal", cmap="coolwarm")
        _plot_input_samples(axes[0, col], _history["train_inputs"], _history["test_inputs"])
        axes[0, col].set_title(f"BrainCell seed {seed}\nV(3 ms)")
        fig.colorbar(im, ax=axes[0, col], label="mV")
        presence = np.ma.masked_where(~np.isfinite(voltage), np.asarray(surface["spike_count"]) > 0)
        axes[1, col].imshow(presence, origin="lower", extent=extent, aspect="equal", cmap="gray_r", vmin=0, vmax=1)
        axes[1, col].set_title(f"BrainCell seed {seed}\nspike presence")
    for axis in axes.flat:
        axis.set_xlim(0, 5); axis.set_ylim(0, 5); axis.set_xlabel("x1"); axis.set_ylabel("x2")
    fig.legend(
        handles=_input_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        title="Input regions and samples",
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "surface_comparison_jaxley_braincell.png", dpi=180)
    plt.close(fig)

def _plot_outcomes(rows, jaxley_dir, output):
    # The Jaxley pickle contains JAX 0.4 arrays and is intentionally not
    # unpickled under the BrainCell/JAX 0.8 environment.  Its summary already
    # records the final loss for every restart.
    jaxley_summary = __import__("json").load((jaxley_dir / "summary.json").open())
    jaxley_losses = [float(item["final_epoch_loss_sum"]) / 32.0 for item in jaxley_summary["restart_metrics_on_matched_test_data"]]
    all_losses = [float(np.asarray(history["epoch_loss"])[-1]) for _, _, history in rows]
    losses = [loss for loss in all_losses if np.isfinite(loss)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].boxplot(jaxley_losses, tick_labels=["Jaxley\n10 restarts"]); axes[0].set_title("Jaxley final loss spread"); axes[0].set_ylabel("MAE at 3 ms (mV)")
    axes[1].boxplot(losses, tick_labels=[f"BrainCell\n{len(losses)}/{len(rows)} finite seeds"]); axes[1].set_title("BrainCell final loss spread"); axes[1].set_ylabel("MAE at 3 ms (mV)")
    fig.savefig(output / "parameter_outcome_comparison.png", dpi=180); plt.close(fig)

def _plot_braincell_overview(rows, output):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for axis, (seed, surface, _history) in zip(axes.flat, rows):
        voltage = np.asarray(surface["voltage"])
        image = axis.imshow(voltage, origin="lower", extent=(0, 5, 0, 5), aspect="equal", cmap="coolwarm", vmin=-75, vmax=45)
        _plot_input_samples(axis, _history["train_inputs"], _history["test_inputs"])
        axis.set_title(f"BrainCell seed {seed}")
        axis.set_xlabel("x1"); axis.set_ylabel("x2")
    fig.legend(
        handles=_input_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        title="Input regions and samples",
    )
    fig.colorbar(image, ax=axes, label="V(3 ms), mV", shrink=0.8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "braincell_all_seed_voltage_surfaces.png", dpi=180)
    plt.close(fig)

def _plot_spikes(rows, output):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for axis, (seed, surface, _history) in zip(axes.flat, rows):
        values = np.ma.masked_where(~np.isfinite(surface["voltage"]), np.asarray(surface["spike_count"]) > 0)
        image = axis.imshow(values, origin="lower", extent=(0, 5, 0, 5), aspect="equal", cmap="gray_r", vmin=0, vmax=1)
        _plot_input_samples(axis, _history["train_inputs"], _history["test_inputs"])
        axis.set_title(f"BrainCell seed {seed}")
        axis.set_xlabel("x1"); axis.set_ylabel("x2")
    fig.legend(
        handles=_input_legend_handles(),
        loc="outside lower center",
        ncol=5,
        frameon=False,
        title="Input regions and samples",
    )
    fig.colorbar(image, ax=axes, ticks=[0, 1], label="spike presence", shrink=0.8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "braincell_all_seed_spike_surfaces.png", dpi=180)
    plt.close(fig)

def _plot_parameters(rows, jaxley_dir, output):
    labels = {"radius": ("radius_mid", "um"), "length": ("length", "um"),
              "Ra": ("Ra", "ohm·cm"), "gNa": ("gNa", "mS/cm²"),
              "gK": ("gK", "mS/cm²"), "gLeak": ("gLeak", "mS/cm²")}
    for seed, _surface, history in rows:
        if not all(f"initial_parameter/{name}" in history.files and
                   f"physical_parameter/{name}" in history.files for name in labels):
            continue
        figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        x = np.arange(12)
        for axis, name in zip(axes.flat, labels):
            initial = np.asarray(history[f"initial_parameter/{name}"])
            final = np.asarray(history[f"physical_parameter/{name}"])
            axis.plot(x, initial, "--", color="#666666", linewidth=1.4, label="before training")
            axis.plot(x, final, "-o", color="#b33b32", linewidth=1.4, markersize=3.5, label="after training")
            axis.axvline(3.5, color="#bbbbbb", linewidth=0.8)
            axis.axvline(7.5, color="#bbbbbb", linewidth=0.8)
            axis.set_title(labels[name][0])
            axis.set_ylabel(labels[name][1])
            axis.set_xticks(x)
            axis.grid(alpha=0.2)
        axes[1, 0].set_xlabel("CV position")
        axes[1, 1].set_xlabel("CV position")
        axes[1, 2].set_xlabel("CV position")
        axes[0, 0].legend(frameon=False)
        figure.suptitle(f"BrainCell physical parameters by CV — seed {seed}")
        figure.savefig(output / f"parameter_before_after_seed_{seed}.png", dpi=180)
        plt.close(figure)

def _plot_parameter_summary(rows, output):
    labels = {"radius": ("radius_mid", "um"), "length": ("length", "um"),
              "Ra": ("Ra", "ohm·cm"), "gNa": ("gNa", "mS/cm²"),
              "gK": ("gK", "mS/cm²"), "gLeak": ("gLeak", "mS/cm²")}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.arange(12)
    width = 0.34
    for axis, name in zip(axes.flat, labels):
        before = np.asarray([
            history[f"initial_parameter/{name}"] for _, _, history in rows
        ], dtype=float)
        after = np.asarray([
            history[f"physical_parameter/{name}"] for _, _, history in rows
        ], dtype=float)
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise ValueError(f"Non-finite values found for parameter {name}")

        before_mean = before.mean(axis=0)
        after_mean = after.mean(axis=0)
        before_min = before.min(axis=0)
        before_max = before.max(axis=0)
        after_min = after.min(axis=0)
        after_max = after.max(axis=0)
        before_error = np.vstack((
            np.maximum(before_mean - before_min, 0.0),
            np.maximum(before_max - before_mean, 0.0),
        ))
        after_error = np.vstack((
            np.maximum(after_mean - after_min, 0.0),
            np.maximum(after_max - after_mean, 0.0),
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
        axis.set_title(labels[name][0])
        axis.set_ylabel(labels[name][1])
        axis.set_xticks(x)
        axis.set_xlabel("CV position")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(f"BrainCell parameter summary across {len(rows)} seeds")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "braincell_parameter_before_after_summary.png", dpi=180)
    plt.close(figure)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jaxley-dir", type=Path, required=True,
                        help="Saved Jaxley export with history.npz and summary.json")
    parser.add_argument("--braincell-dir", type=Path,
                        default=WORKFLOW_ROOT / "artifacts" / "nonlinear_pattern_separation_2026-09-15" / "raw" / "braincell")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    rows = _load_braincell(args.braincell_dir)
    if not rows:
        parser.error(f"No saved seed surfaces in {args.braincell_dir}")
    output = args.output_dir or args.braincell_dir / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    try:
        _plot_surfaces(rows, args.jaxley_dir, output)
        _plot_outcomes(rows, args.jaxley_dir, output)
        _plot_braincell_overview(rows, output)
        _plot_spikes(rows, output)
        _plot_parameters(rows, args.jaxley_dir, output)
        _plot_parameter_summary(rows, output)
    finally:
        for _, surface, history in rows:
            surface.close()
            history.close()
    print(output)

if __name__ == "__main__":
    main()
