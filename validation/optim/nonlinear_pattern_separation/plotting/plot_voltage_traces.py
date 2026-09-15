"""Forward-simulate saved BrainCell parameters and plot 6 ms test traces."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import brainstate
import brainunit as u
import jax

from validation.optim.nonlinear_pattern_separation.runners.braincell.model import build_strict_experiment
from validation.optim.nonlinear_pattern_separation.runners.braincell.task import generate_dataset, NonlinearPatternExperiment, DT_MS, NUM_STEPS, TARGET_VOLTAGES_MV


def run_seed(seed, raw_root):
    history = np.load(raw_root / f"seed_{seed}" / "history.npz")
    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        ref, cell, parameters = build_strict_experiment(seed)
        values = {
            "length": np.asarray(history["physical_parameter/length"]) * u.um,
            "Ra": np.asarray(history["physical_parameter/Ra"]) * u.ohm * u.cm,
            "na.g_max": np.asarray(history["physical_parameter/gNa"]) * u.mS / u.cm**2,
            "k.g_max": np.asarray(history["physical_parameter/gK"]) * u.mS / u.cm**2,
            "leak.g_max": np.asarray(history["physical_parameter/gLeak"]) * u.mS / u.cm**2,
            "radius_scale": np.asarray(history["physical_parameter/radius"]) / 2.55,
        }
        parameters.set_physical_values(values)
        cell.trainables.materialize(); cell.update()
        dataset = generate_dataset(seed=seed, n_train=32, n_test=16)
        NonlinearPatternExperiment._build_cell = staticmethod(lambda *, seed: build_strict_experiment(seed)[1])
        experiment = NonlinearPatternExperiment(seed=seed, learning_rate=0.01, loss_kind="voltage_at_3ms")
        experiment.parameters.set_physical_values(values)
        experiment.cell.trainables.materialize(); experiment.cell.update()
        traces = np.asarray(experiment.simulate_dataset(dataset.test_inputs, dataset.test_targets_mv))
    return traces, np.asarray(dataset.test_targets_mv), np.asarray(dataset.test_classes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    time_ms = np.arange(NUM_STEPS) * DT_MS
    saved = {}
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True, constrained_layout=True)
    for axis, seed in zip(axes.flat, args.seeds):
        traces, targets, classes = run_seed(seed, args.raw_dir)
        saved[f"seed_{seed}/traces_mv"] = traces
        saved[f"seed_{seed}/targets_mv"] = targets
        saved[f"seed_{seed}/classes"] = classes
        for trace, cls in zip(traces, classes):
            axis.plot(time_ms, trace, color="#b33b32" if cls else "#3568a8", alpha=0.35, linewidth=0.8)
        axis.axvline(3.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.axhline(TARGET_VOLTAGES_MV[0], color="#3568a8", linestyle=":", linewidth=0.8)
        axis.axhline(TARGET_VOLTAGES_MV[1], color="#b33b32", linestyle=":", linewidth=0.8)
        axis.scatter([3.0, 3.0], TARGET_VOLTAGES_MV, c=["#3568a8", "#b33b32"], s=28, zorder=4)
        axis.set_title(f"BrainCell seed {seed}")
        axis.grid(alpha=0.2)
    for axis in axes[-1]: axis.set_xlabel("time (ms)")
    for axis in axes[:, 0]: axis.set_ylabel("voltage (mV)")
    axes[0, 0].legend(["class 0 trace", "class 1 trace", "3 ms", "target -70 mV", "target 35 mV"], frameon=False, fontsize=8)
    fig.suptitle("BrainCell final test voltage traces (0–6 ms)")
    fig.savefig(args.output_dir / "braincell_test_voltage_traces_6ms.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output_dir.parent / "data" / "braincell_test_voltage_traces.npz", time_ms=time_ms, **saved)


if __name__ == "__main__":
    main()
