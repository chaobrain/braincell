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

# ruff: noqa: E402

"""End-to-end Adam comparison for block-exact BPTT and full RTRL gradients."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import brainstate
import brainunit as u
import braintools
import jax
import jax.numpy as jnp
import numpy as np

from examples.experimental.optim_gradient_scaling.benchmark import (
    BACKSUBS,
    DT_MS,
    METHODS,
    BenchmarkConfig,
    prepare_benchmark,
)

DEFAULT_CONFIG = BenchmarkConfig(n_cv=5, duration_ms=40.0, batch_size=16, n_seed=32)
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.01
PARAMETER_NAMES = ("leak.scale", "na.scale", "k.scale")
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts" / "rtrl_bptt_training"


def run_training_worker(
    config: BenchmarkConfig,
    method: str,
    *,
    epochs: int,
    learning_rate: float,
    output_json: Path,
    backsub: str = "recursive",
    execution_seed_count: int | None = None,
) -> dict[str, object]:
    """Train one method and persist its complete seed/epoch history."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS!r}.")
    if backsub not in BACKSUBS:
        raise ValueError(f"backsub must be one of {BACKSUBS!r}.")
    if epochs < 1 or learning_rate <= 0.0:
        raise ValueError("epochs and learning_rate must be positive.")
    execution_seed_count = config.n_seed if execution_seed_count is None else int(execution_seed_count)
    if execution_seed_count < config.n_seed:
        raise ValueError("execution_seed_count must be at least config.n_seed.")
    os.environ["BRAINCELL_DHS_BACKSUB"] = backsub
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with jax.enable_x64(True), brainstate.environ.context(dt=DT_MS * u.ms, precision=64):
        prepared = prepare_benchmark(config, method)
        parameter_states = {
            name: brainstate.ParamState(value) for name, value in zip(PARAMETER_NAMES, prepared.seed_roots)
        }
        optimizer = braintools.optim.Adam(lr=learning_rate)
        optimizer.register_trainable_weights(parameter_states)

        example_roots = tuple(state.value for state in parameter_states.values())
        execution_function = _seed_padded_function(
            prepared.function,
            requested_seed_count=config.n_seed,
            execution_seed_count=execution_seed_count,
        )
        started = time.perf_counter()
        compiled = jax.jit(execution_function).lower(example_roots).compile()
        compile_seconds = time.perf_counter() - started
        memory = compiled.memory_analysis()

        losses = []
        gradients = []
        parameters_before = []
        parameters_after = []
        gradient_seconds = []
        optimizer_seconds = []
        for _ in range(epochs):
            roots = tuple(state.value for state in parameter_states.values())
            parameters_before.append(_stack_roots(roots))
            started = time.perf_counter()
            seed_loss, _step_losses, flat_gradient = compiled(roots)
            _block_until_ready((seed_loss, flat_gradient))
            gradient_seconds.append(time.perf_counter() - started)
            gradient_mapping = _unflatten_gradient(flat_gradient, roots)

            started = time.perf_counter()
            optimizer.update(gradient_mapping)
            _block_until_ready(tuple(state.value for state in parameter_states.values()))
            optimizer_seconds.append(time.perf_counter() - started)

            losses.append(seed_loss)
            gradients.append(_stack_roots(tuple(gradient_mapping[name] for name in PARAMETER_NAMES)))
            parameters_after.append(_stack_roots(tuple(state.value for state in parameter_states.values())))

    arrays = {
        "loss": np.asarray(jnp.stack(losses)),
        "gradient": np.asarray(jnp.stack(gradients)),
        "parameters_before": np.asarray(jnp.stack(parameters_before)),
        "parameters_after": np.asarray(jnp.stack(parameters_after)),
        "gradient_seconds": np.asarray(gradient_seconds, dtype=np.float64),
        "optimizer_seconds": np.asarray(optimizer_seconds, dtype=np.float64),
    }
    output_npz = output_json.with_suffix(".npz")
    np.savez_compressed(output_npz, **arrays)
    result = {
        **asdict(config),
        "config_id": config.id,
        "method": method,
        "backsub": backsub,
        "execution_seed_count": execution_seed_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": "ok",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "compile_seconds": compile_seconds,
        "argument_bytes": int(memory.argument_size_in_bytes),
        "output_bytes": int(memory.output_size_in_bytes),
        "temporary_bytes": int(memory.temp_size_in_bytes),
        "alias_bytes": int(memory.alias_size_in_bytes),
        "gradient_seconds": arrays["gradient_seconds"].tolist(),
        "gradient_median_seconds": float(np.median(arrays["gradient_seconds"])),
        "optimizer_median_seconds": float(np.median(arrays["optimizer_seconds"])),
        "initial_loss_mean": float(np.mean(arrays["loss"][0])),
        "final_loss_mean": float(np.mean(arrays["loss"][-1])),
        "initial_loss_min": float(np.min(arrays["loss"][0])),
        "final_loss_min": float(np.min(arrays["loss"][-1])),
        "history_file": output_npz.name,
        "history_shapes": {name: list(value.shape) for name, value in arrays.items()},
    }
    _write_json(output_json, result)
    return result


def compare_training_runs(output_dir: Path) -> dict[str, object]:
    """Compare completed BPTT/RTRL histories and generate JSON/PNG output."""
    metadata = {method: _read_json(output_dir / f"{method}.json") for method in METHODS}
    histories = {method: np.load(output_dir / str(metadata[method]["history_file"])) for method in METHODS}
    bptt = histories["bptt"]
    rtrl = histories["rtrl"]
    loss_difference = np.abs(bptt["loss"] - rtrl["loss"])
    parameter_difference = np.abs(bptt["parameters_after"] - rtrl["parameters_after"])
    gradient_difference = np.abs(bptt["gradient"] - rtrl["gradient"])
    gradient_scale = np.maximum(np.abs(bptt["gradient"]), np.abs(rtrl["gradient"]))
    comparison = {
        "status": "ok",
        "config_id": metadata["bptt"]["config_id"],
        "epochs": metadata["bptt"]["epochs"],
        "loss_max_abs_difference": float(np.max(loss_difference)),
        "loss_final_max_abs_difference": float(np.max(loss_difference[-1])),
        "parameter_max_abs_difference": float(np.max(parameter_difference)),
        "parameter_final_max_abs_difference": float(np.max(parameter_difference[-1])),
        "gradient_epoch0_max_abs_difference": float(np.max(gradient_difference[0])),
        "gradient_final_max_abs_difference": float(np.max(gradient_difference[-1])),
        "gradient_epoch0_max_relative_difference": _max_relative_difference(gradient_difference[0], gradient_scale[0]),
        "gradient_final_max_relative_difference": _max_relative_difference(gradient_difference[-1], gradient_scale[-1]),
        "bptt_gradient_median_seconds": metadata["bptt"]["gradient_median_seconds"],
        "rtrl_gradient_median_seconds": metadata["rtrl"]["gradient_median_seconds"],
        "bptt_over_rtrl_gradient_time": (
            metadata["bptt"]["gradient_median_seconds"] / metadata["rtrl"]["gradient_median_seconds"]
        ),
        "bptt_compile_seconds": metadata["bptt"]["compile_seconds"],
        "rtrl_compile_seconds": metadata["rtrl"]["compile_seconds"],
        "bptt_temporary_bytes": metadata["bptt"]["temporary_bytes"],
        "rtrl_temporary_bytes": metadata["rtrl"]["temporary_bytes"],
        "bptt_over_rtrl_temporary_bytes": (metadata["bptt"]["temporary_bytes"] / metadata["rtrl"]["temporary_bytes"]),
    }
    _write_json(output_dir / "comparison.json", comparison)
    _plot_comparison(output_dir / "comparison.png", histories)
    return comparison


def run_comparison(
    config: BenchmarkConfig,
    *,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    gpu: int,
    python_executable: Path,
    resume: bool,
    execution_seed_count: int | None = None,
) -> dict[str, object]:
    """Launch isolated method workers and combine their histories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    _write_json(
        output_dir / "manifest.json",
        {
            **asdict(config),
            "config_id": config.id,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gpu": gpu,
            "python": str(python_executable),
            "methods": METHODS,
            "execution_seed_count": execution_seed_count,
        },
    )
    expected_execution_seed_count = config.n_seed if execution_seed_count is None else int(execution_seed_count)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    for method in METHODS:
        output_json = output_dir / f"{method}.json"
        if resume and output_json.exists():
            existing = _read_json(output_json)
            existing_execution_seed_count = int(existing.get("execution_seed_count", existing.get("n_seed", 0)))
            if existing.get("status") == "ok" and existing_execution_seed_count == expected_execution_seed_count:
                continue
        command = [
            str(python_executable),
            str(Path(__file__).resolve()),
            "worker",
            "--config",
            json.dumps(asdict(config)),
            "--method",
            method,
            "--epochs",
            str(epochs),
            "--learning-rate",
            str(learning_rate),
            "--output",
            str(output_json),
        ]
        if execution_seed_count is not None:
            command.extend(["--execution-seed-count", str(execution_seed_count)])
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        (log_dir / f"{method}.log").write_text(
            completed.stdout + ("\nSTDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{method} worker failed; see {log_dir / f'{method}.log'}.")
    return compare_training_runs(output_dir)


def _stack_roots(roots) -> object:
    arrays = []
    for value in roots:
        value = jnp.asarray(value)
        arrays.append(value[:, None] if value.ndim == 1 else value)
    return jnp.stack(arrays, axis=1)


def _seed_padded_function(function, *, requested_seed_count: int, execution_seed_count: int):
    """Return a function with a larger static seed extent and requested-size outputs."""
    if execution_seed_count < requested_seed_count:
        raise ValueError("execution_seed_count must preserve every requested seed.")
    indices = jnp.arange(execution_seed_count, dtype=jnp.int32) % requested_seed_count

    def execute(roots):
        execution_roots = tuple(root[indices] for root in roots)
        loss, losses, gradient = function(execution_roots)
        return (
            loss[:requested_seed_count],
            losses[:requested_seed_count],
            gradient[:requested_seed_count],
        )

    return execute


def _unflatten_gradient(flat_gradient, roots) -> dict[str, object]:
    gradients = {}
    offset = 0
    for name, root in zip(PARAMETER_NAMES, roots):
        coordinate_count = int(np.prod(root.shape[1:], dtype=np.int64)) if root.ndim > 1 else 1
        value = flat_gradient[:, offset : offset + coordinate_count]
        gradients[name] = value.reshape(root.shape)
        offset += coordinate_count
    if offset != flat_gradient.shape[1]:
        raise ValueError("Flat gradient does not match parameter root shapes.")
    return gradients


def _plot_comparison(path: Path, histories) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, histories["bptt"]["loss"].shape[0] + 1)
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    for method, color in (("bptt", "tab:orange"), ("rtrl", "tab:blue")):
        loss = histories[method]["loss"]
        axes[0, 0].plot(epochs, loss.mean(axis=1), label=method.upper(), color=color)
        axes[0, 1].plot(epochs, loss.min(axis=1), label=method.upper(), color=color)
        axes[1, 1].plot(epochs, histories[method]["gradient_seconds"], label=method.upper(), color=color)
    parameter_difference = np.max(
        np.abs(histories["bptt"]["parameters_after"] - histories["rtrl"]["parameters_after"]),
        axis=(1, 2, 3),
    )
    axes[1, 0].plot(epochs, parameter_difference, color="tab:green")
    axes[0, 0].set(xlabel="epoch", ylabel="mean seed loss", title="Mean loss")
    axes[0, 1].set(xlabel="epoch", ylabel="minimum seed loss", title="Best seed loss")
    axes[1, 0].set(xlabel="epoch", ylabel="max absolute difference", title="BPTT/RTRL parameter difference")
    axes[1, 1].set(xlabel="epoch", ylabel="seconds", title="Gradient kernel time")
    for axis in axes.flat:
        axis.grid(True)
        if axis is not axes[1, 0]:
            axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _max_relative_difference(difference, scale) -> float:
    return float(np.max(difference / np.maximum(scale, 1e-30)))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_output_dir(config: BenchmarkConfig, epochs: int) -> Path:
    return ARTIFACT_ROOT / f"{config.id}_e{epochs}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    run.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    run.add_argument("--gpu", type=int, default=7)
    run.add_argument("--python", type=Path, default=Path("/home/swl/anaconda3/envs/braincell_311/bin/python"))
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--config", default=json.dumps(asdict(DEFAULT_CONFIG)))
    run.add_argument("--execution-seed-count", type=int)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--epochs", type=int, required=True)
    worker.add_argument("--learning-rate", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--execution-seed-count", type=int)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig(**json.loads(args.config))
    if args.command == "worker":
        run_training_worker(
            config,
            args.method,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_json=args.output,
            execution_seed_count=args.execution_seed_count,
        )
        return
    output_dir = args.output_dir or _default_output_dir(config, args.epochs)
    comparison = run_comparison(
        config,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
        gpu=args.gpu,
        python_executable=args.python,
        resume=args.resume,
        execution_seed_count=args.execution_seed_count,
    )
    print(output_dir)
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
