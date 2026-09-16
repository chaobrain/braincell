"""Single-seed runner for the bounded 72-parameter model."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
from .model import ARTIFACT_ROOT, REFERENCE, load_reference, build_strict_experiment, parameter_count, export_physical_parameters

def run_smoke(*, seed: int = 0, epochs: int = 1, n_train: int = 4, n_test: int = 2,
              learning_rate: float = 0.01):
    """Run the reference RTRL loop on the strict 72-parameter cell."""
    ref = load_reference()
    ref.NonlinearPatternExperiment._build_cell = staticmethod(lambda *, seed: build_strict_experiment(seed)[1])
    with jax.enable_x64(True), brainstate.environ.context(dt=ref.DT_MS * u.ms, precision=64):
        dataset = ref.generate_dataset(seed=seed, n_train=n_train, n_test=n_test)
        experiment = ref.NonlinearPatternExperiment(
            seed=seed, learning_rate=learning_rate, loss_kind="voltage_at_3ms"
        )
        losses, sample_losses = experiment.train(dataset, epochs=epochs, batch_size=1)
        return experiment, dataset, losses, sample_losses


def run_formal(*, seed: int = 0, epochs: int = 200, learning_rate: float = 0.01,
               output_dir: str | Path = ARTIFACT_ROOT):
    """Run and save the CPU reference comparison without unit assumptions."""
    ref = load_reference()
    ref.NonlinearPatternExperiment._build_cell = staticmethod(lambda *, seed: build_strict_experiment(seed)[1])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with jax.enable_x64(True), brainstate.environ.context(dt=ref.DT_MS * u.ms, precision=64):
        dataset = ref.generate_dataset(seed=seed, n_train=32, n_test=16)
        experiment = ref.NonlinearPatternExperiment(
            seed=seed, learning_rate=learning_rate, loss_kind="voltage_at_3ms"
        )
        initial = experiment.simulate_dataset(dataset.train_inputs, dataset.train_targets_mv)
        initial_parameters = export_physical_parameters(experiment)
        started = time.perf_counter()
        losses, sample_losses = experiment.train(dataset, epochs=epochs, batch_size=1)
        jax.block_until_ready((losses, sample_losses))
        elapsed = time.perf_counter() - started
        final_train = experiment.simulate_dataset(dataset.train_inputs, dataset.train_targets_mv)
        final_test = experiment.simulate_dataset(dataset.test_inputs, dataset.test_targets_mv)
        final_parameters = export_physical_parameters(experiment)
        initial_readout = np.asarray(initial)[:, ref.READOUT_INDEX]
        final_train_readout = np.asarray(final_train)[:, ref.READOUT_INDEX]
        final_test_readout = np.asarray(final_test)[:, ref.READOUT_INDEX]
        targets_train = np.asarray(dataset.train_targets_mv)
        targets_test = np.asarray(dataset.test_targets_mv)
        physical = {}
        for name, value in experiment.parameters.physical_values().items():
            try:
                physical[name] = np.asarray(value.to_decimal(ref.CONDUCTANCE_UNIT))
            except (AttributeError, ValueError):
                physical[name] = np.asarray(value)
        arrays = {
            "loss": np.asarray(losses),
            "sample_loss": np.asarray(sample_losses),
            "epoch_loss": np.asarray(sample_losses).reshape((epochs, 32)).mean(axis=1),
            "initial_train_traces_mv": np.asarray(initial),
            "final_train_traces_mv": np.asarray(final_train),
            "final_test_traces_mv": np.asarray(final_test),
            "train_inputs": np.asarray(dataset.train_inputs),
            "test_inputs": np.asarray(dataset.test_inputs),
            "train_targets_mv": targets_train,
            "test_targets_mv": targets_test,
            "train_classes": np.asarray(dataset.train_classes),
            "test_classes": np.asarray(dataset.test_classes),
            **{f"parameter/{name}": value for name, value in physical.items()},
            **{f"initial_parameter/{name}": value for name, value in initial_parameters.items()},
            **{f"physical_parameter/{name}": value for name, value in final_parameters.items()},
        }
        np.savez_compressed(output / "history.npz", **arrays)
        train_accuracy = float(ref.classification_accuracy(final_train_readout, dataset.train_classes))
        test_accuracy = float(ref.classification_accuracy(final_test_readout, dataset.test_classes))
        summary = {
            "seed": seed, "epochs": epochs, "updates": epochs * 32,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count(experiment.parameters),
            "physical_parameter_fields": list(final_parameters),
            "loss_kind": "voltage_at_3ms", "gradient_method": "exact_full_rtrl",
            "backend": jax.default_backend(), "device": str(jax.devices()[0]),
            "precision": 64, "batch_size": 1, "train_size": 32, "test_size": 16,
            "final_epoch_loss": float(np.asarray(arrays["epoch_loss"])[-1]),
            "initial_train_mae_mv": float(np.mean(np.abs(initial_readout - targets_train))),
            "final_train_mae_mv": float(np.mean(np.abs(final_train_readout - targets_train))),
            "final_test_mae_mv": float(np.mean(np.abs(final_test_readout - targets_test))),
            "train_voltage_accuracy": train_accuracy, "test_voltage_accuracy": test_accuracy,
            "compile_and_train_seconds": elapsed, "history_file": "history.npz",
        }
        finite_loss = np.isfinite(np.asarray(arrays["loss"], dtype=float))
        summary["first_nonfinite_update"] = (
            int(np.flatnonzero(~finite_loss)[0]) if np.any(~finite_loss) else None
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(np.asarray(arrays["loss"]), alpha=0.25, label="batch loss")
        axis.plot(np.asarray(arrays["epoch_loss"]), label="epoch MAE")
        axis.set(xlabel="update / epoch", ylabel="loss (mV)", title="Strict 72-parameter RTRL")
        axis.grid(alpha=0.2); axis.legend(); figure.tight_layout()
        figure.savefig(output / "loss_convergence.png", dpi=140); plt.close(figure)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT,
                        help="Formal-mode artifact directory; smoke mode prints its result")
    args = parser.parse_args(argv)
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.mode == "formal":
        print(json.dumps(run_formal(seed=args.seed, epochs=args.epochs or 200,
                                   learning_rate=args.learning_rate,
                                   output_dir=args.output_dir), indent=2))
        return
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        _ref, cell, parameters = build_strict_experiment(args.seed)
        print("n_cv", cell.n_cv)
        print("parameter_fields", sorted(parameters.physical_values()))
        print("parameter_count", parameter_count(parameters))
        _experiment, _dataset, losses, _sample_losses = run_smoke(
            seed=args.seed, epochs=args.epochs or 1, learning_rate=args.learning_rate)
        print("smoke_loss", float(losses[-1]))


if __name__ == "__main__":
    main()
