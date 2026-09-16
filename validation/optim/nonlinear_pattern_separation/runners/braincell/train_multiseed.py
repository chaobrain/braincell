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

"""Multi-seed strict geometry experiment and nonlinear decision surfaces."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validation.optim.nonlinear_pattern_separation.runners.braincell.model import ARTIFACT_ROOT, load_reference, build_strict_experiment, export_physical_parameters


def _parameter_array(value):
    if hasattr(value, "to_decimal"):
        for unit in (u.mS / u.cm ** 2, u.um, u.ohm * u.cm):
            try:
                return np.asarray(value.to_decimal(unit))
            except Exception:
                pass
        try:
            return np.asarray(value.to_decimal())
        except Exception:
            pass
    return np.asarray(value)


def run_seed(seed: int, *, epochs: int, learning_rate: float, output_dir: Path):
    ref = load_reference()
    ref.NonlinearPatternExperiment._build_cell = staticmethod(lambda *, seed: build_strict_experiment(seed)[1])
    with jax.enable_x64(True), brainstate.environ.context(dt=ref.DT_MS * ref.u.ms, precision=64):
        dataset = ref.generate_dataset(seed=seed, n_train=32, n_test=16)
        experiment = ref.NonlinearPatternExperiment(
            seed=seed, learning_rate=learning_rate, loss_kind="voltage_at_3ms"
        )
        initial_parameters = export_physical_parameters(experiment)
        losses, sample_losses = experiment.train(dataset, epochs=epochs, batch_size=1)
        final_train = experiment.simulate_dataset(dataset.train_inputs, dataset.train_targets_mv)
        final_test = experiment.simulate_dataset(dataset.test_inputs, dataset.test_targets_mv)
        final_parameters = export_physical_parameters(experiment)
        axis = jnp.linspace(0.0, 5.0, 51)
        mesh_x, mesh_y = jnp.meshgrid(axis, axis, indexing="xy")
        sweep_inputs = jnp.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)
        voltage, count, first_spike = experiment.simulate_sweep(sweep_inputs, chunk_size=256)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / "history.npz",
            loss=np.asarray(losses),
            sample_loss=np.asarray(sample_losses),
            epoch_loss=np.asarray(sample_losses).reshape((epochs, 32)).mean(axis=1),
            train_inputs=np.asarray(dataset.train_inputs),
            test_inputs=np.asarray(dataset.test_inputs),
            train_classes=np.asarray(dataset.train_classes),
            test_classes=np.asarray(dataset.test_classes),
            train_readout=np.asarray(final_train)[:, ref.READOUT_INDEX],
            test_readout=np.asarray(final_test)[:, ref.READOUT_INDEX],
            **{f"parameter/{name}": _parameter_array(value)
               for name, value in experiment.parameters.physical_values().items()},
            **{f"initial_parameter/{name}": value for name, value in initial_parameters.items()},
            **{f"physical_parameter/{name}": value for name, value in final_parameters.items()},
        )
        np.savez_compressed(
            output_dir / "surface.npz",
            x=np.asarray(axis), y=np.asarray(axis),
            voltage=np.asarray(voltage).reshape(51, 51),
            spike_count=np.asarray(count).reshape(51, 51),
            first_spike_ms=np.asarray(first_spike).reshape(51, 51),
        )
        train_readout = np.asarray(final_train)[:, ref.READOUT_INDEX]
        test_readout = np.asarray(final_test)[:, ref.READOUT_INDEX]
        summary = {
            "seed": seed, "epochs": epochs, "updates": epochs * 32,
            "parameter_count": 72, "batch_size": 1,
            "learning_rate": learning_rate, "loss_kind": "voltage_at_3ms",
            "backend": jax.default_backend(), "precision": 64,
            "final_epoch_loss": float(np.asarray(sample_losses).reshape((epochs, 32))[-1].mean()),
            "train_mae_mv": float(np.mean(np.abs(train_readout - np.asarray(dataset.train_targets_mv)))),
            "test_mae_mv": float(np.mean(np.abs(test_readout - np.asarray(dataset.test_targets_mv)))),
            "train_accuracy": float(ref.classification_accuracy(train_readout, dataset.train_classes)),
            "test_accuracy": float(ref.classification_accuracy(test_readout, dataset.test_classes)),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.learning_rate <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("epochs and learning-rate must be positive and seeds nonnegative")
    root = args.output_dir
    epochs = args.epochs
    summaries = []
    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        summary_file = seed_dir / "summary.json"
        history_file = seed_dir / "history.npz"
        has_parameters = False
        if history_file.exists():
            with np.load(history_file) as existing:
                has_parameters = any(name.startswith("parameter/") for name in existing.files)
        has_snapshots = False
        if history_file.exists():
            with np.load(history_file) as existing:
                has_snapshots = all(
                    f"initial_parameter/{name}" in existing.files and
                    f"physical_parameter/{name}" in existing.files
                    for name in ("radius", "length", "Ra", "gNa", "gK", "gLeak")
                )
        if summary_file.exists() and has_parameters and has_snapshots and (seed_dir / "surface.npz").exists():
            existing_summary = json.loads(summary_file.read_text())
            if existing_summary["epochs"] != epochs or existing_summary.get("learning_rate") != args.learning_rate:
                raise ValueError(f"Existing run at {seed_dir} has different protocol; choose another --output-dir")
            summaries.append(json.loads(summary_file.read_text()))
        else:
            summaries.append(run_seed(seed, epochs=epochs, learning_rate=args.learning_rate, output_dir=seed_dir))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for summary in summaries:
        data = np.load(root / f"seed_{summary['seed']}" / "history.npz")
        axes[0].plot(data["epoch_loss"], label=f"seed {summary['seed']}")
        surface = np.load(root / f"seed_{summary['seed']}" / "surface.npz")
        axes[1].contourf(surface["x"], surface["y"], surface["voltage"], levels=30, alpha=0.28)
    axes[0].set(xlabel="epoch", ylabel="MAE at 3 ms (mV)", title="Multi-seed convergence")
    axes[0].legend(); axes[0].grid(alpha=0.2)
    axes[1].set(xlabel="x1", ylabel="x2", title="Final 3 ms voltage surfaces")
    axes[1].set_xlim(0, 5); axes[1].set_ylim(0, 5)
    figure.tight_layout(); figure.savefig(root / "multi_seed_summary.png", dpi=150); plt.close(figure)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
