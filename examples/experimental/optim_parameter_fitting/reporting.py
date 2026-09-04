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

"""Persist and visualize modular parameter-learning experiment results."""

from __future__ import annotations

import csv
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.config import ExperimentConfig
from examples.experimental.optim_parameter_fitting.datasets import DatasetBundle, Protocol, SPLITS, write_bundle
from examples.experimental.optim_parameter_fitting.training import (
    ForwardEvaluator,
    GradientStageResult,
    PipelineResult,
)


def save_run(
    directory: Path,
    config: ExperimentConfig,
    config_source: Path,
    result: PipelineResult,
    *,
    run_started_at: float,
) -> dict[str, object]:
    """Write the complete result contract and return the aggregate summary."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    stages_dir = root / "stages"
    stages_dir.mkdir(exist_ok=True)
    shutil.copyfile(config_source, root / "config.py")
    resolved = config.describe()
    resolved["config_digest"] = config.digest()
    _write_json(root / "resolved_config.json", resolved)
    _write_json(root / "parameter_space.json", config.model.parameter_space.describe())
    _write_json(root / "environment.json", _environment_metadata())
    write_bundle(root, result.dataset)

    parameter_space = config.model.parameter_space
    initial = result.initial_candidates.physical
    np.savez_compressed(
        root / "initial_candidates.npz",
        physical=initial,
        normalized=np.asarray(parameter_space.physical_to_normalized(initial)),
        optimizer_z=np.asarray(parameter_space.physical_to_z(initial)),
        candidate_id=result.initial_candidates.candidate_id,
        parameter_name=np.asarray(parameter_space.names),
    )
    gradient_stages = []
    for index, stage in enumerate(result.stages):
        stage_root = stages_dir / f"{index:02d}_{getattr(stage, 'name', 'stage')}"
        stage_root.mkdir(exist_ok=True)
        if isinstance(stage, GradientStageResult):
            _save_gradient_stage(stage_root, stage)
            gradient_stages.append(stage)
        else:
            _write_json(
                stage_root / "summary.json",
                {
                    "kind": "derivative_free",
                    "forward_evaluations": stage.forward_evaluations,
                    "metadata": stage.metadata,
                    "output_candidates": stage.candidates.size,
                },
            )
    if not gradient_stages:
        raise ValueError("Reporting currently requires at least one gradient stage.")
    final_stage = gradient_stages[-1]
    rows, summary = _summarize(config, result.dataset, final_stage, end_to_end_seconds=0.0)
    _write_rows(root / "per_start.csv", rows)
    _write_metrics(root / "metrics.csv", result.dataset, final_stage)
    _plot_dataset(figures / "dataset.png", result.dataset)
    _plot_losses(figures / "loss_curves.png", final_stage)
    _plot_success(figures / "success.png", config, final_stage, summary)
    _plot_parameters(figures / "parameters.png", config, final_stage)
    _plot_traces(figures / "traces.png", config, result.dataset, final_stage)
    summary["end_to_end_seconds"] = time.perf_counter() - run_started_at
    _write_json(root / "summary.json", summary)
    (root / "REPORT.md").write_text(_format_report(config, result.dataset, summary), encoding="utf-8")
    return summary


def compare_completed_runs(
    baseline_dir: Path,
    extended_dir: Path,
    output_dir: Path,
    *,
    comparison_kind: str = "epoch",
) -> dict[str, object]:
    """Compare a longer run against an otherwise identical shorter run."""
    baseline_dir = Path(baseline_dir)
    extended_dir = Path(extended_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_config = json.loads((baseline_dir / "resolved_config.json").read_text(encoding="utf-8"))
    extended_config = json.loads((extended_dir / "resolved_config.json").read_text(encoding="utf-8"))
    baseline_epochs = int(baseline_config["stages"][0]["epochs"])
    extended_epochs = int(extended_config["stages"][0]["epochs"])
    normalized = copy.deepcopy(extended_config)
    if comparison_kind == "epoch":
        normalized["stages"][0]["epochs"] = baseline_epochs
        only_config_change = "stages[0].epochs"
    elif comparison_kind == "lr_bounds":
        normalized["stages"][0]["learning_rate"] = baseline_config["stages"][0]["learning_rate"]
        normalized["model"]["parameter_space"]["lower"] = baseline_config["model"]["parameter_space"]["lower"]
        normalized["model"]["parameter_space"]["upper"] = baseline_config["model"]["parameter_space"]["upper"]
        normalized["initialization"].pop("target_relative_range", None)
        only_config_change = "stages[0].learning_rate + model.parameter_space bounds"
    elif comparison_kind == "bounds":
        normalized["model"]["parameter_space"]["lower"] = baseline_config["model"]["parameter_space"]["lower"]
        normalized["model"]["parameter_space"]["upper"] = baseline_config["model"]["parameter_space"]["upper"]
        normalized["initialization"].pop("target_relative_range", None)
        only_config_change = "model.parameter_space bounds"
    elif comparison_kind == "lr":
        normalized["stages"][0]["learning_rate"] = baseline_config["stages"][0]["learning_rate"]
        only_config_change = "stages[0].learning_rate"
    elif comparison_kind == "optimizer":
        normalized["stages"] = baseline_config["stages"]
        only_config_change = "stages[0].optimizer"
    elif comparison_kind == "loss":
        normalized["loss"] = baseline_config["loss"]
        only_config_change = "loss"
    else:
        raise ValueError("comparison_kind must be 'epoch', 'lr', 'bounds', 'lr_bounds', 'optimizer', or 'loss'.")
    normalized["config_digest"] = baseline_config["config_digest"]
    if normalized != baseline_config:
        raise ValueError("Run configurations differ by more than the first stage's epoch count.")
    if comparison_kind == "epoch" and extended_epochs <= baseline_epochs:
        raise ValueError("The extended run must have more epochs than the baseline.")
    if comparison_kind in {"lr", "bounds", "lr_bounds", "optimizer", "loss"} and extended_epochs != baseline_epochs:
        raise ValueError("Non-epoch comparisons require equal epoch budgets.")

    with (
        np.load(baseline_dir / "initial_candidates.npz") as baseline_initial,
        np.load(extended_dir / "initial_candidates.npz") as extended_initial,
    ):
        all_initial_exact = all(
            np.array_equal(baseline_initial[name], extended_initial[name]) for name in baseline_initial.files
        )
        physical_initial_exact = all(
            np.array_equal(baseline_initial[name], extended_initial[name])
            for name in ("physical", "candidate_id", "parameter_name")
        )
        initial_coordinate_difference = {
            name: float(np.max(np.abs(baseline_initial[name] - extended_initial[name])))
            for name in ("normalized", "optimizer_z")
        }
    if comparison_kind in {"epoch", "lr", "optimizer", "loss"} and not all_initial_exact:
        raise ValueError("Epoch/LR comparison requires identical initial candidate artifacts.")
    if comparison_kind in {"bounds", "lr_bounds"} and not physical_initial_exact:
        raise ValueError("Bounds comparisons require identical physical initial candidates.")

    baseline_history_path = _single_gradient_history(baseline_dir)
    extended_history_path = _single_gradient_history(extended_dir)
    with np.load(baseline_history_path) as baseline_history, np.load(extended_history_path) as extended_history:
        prefix = {}
        for name in ("optimizer_z", "physical_parameters", "train_loss", "gradient"):
            baseline_value = np.asarray(baseline_history[name])
            extended_value = np.asarray(extended_history[name])[: baseline_value.shape[0]]
            prefix[name] = {
                "exact": bool(np.array_equal(baseline_value, extended_value)),
                "max_abs_difference": float(np.max(np.abs(baseline_value - extended_value))),
            }
        location = int(np.flatnonzero(extended_history["validation_epoch"] == baseline_epochs)[0])
        validation_prefix_difference = float(
            np.max(
                np.abs(
                    baseline_history["validation_total_loss"][-1] - extended_history["validation_total_loss"][location]
                )
            )
        )

    baseline_rows = _read_rows(baseline_dir / "per_start.csv")
    extended_rows = _read_rows(extended_dir / "per_start.csv")
    if [row["candidate_id"] for row in baseline_rows] != [row["candidate_id"] for row in extended_rows]:
        raise ValueError("Per-start candidate IDs do not align.")
    success_fields = ("trace_success", "parameter_success", "joint_success")
    metric_fields = ("final_train_mse", "final_validation_mse", "final_test_mse", "parameter_relative_rms")
    transitions = {}
    output_rows = []
    for start, (baseline, extended) in enumerate(zip(baseline_rows, extended_rows)):
        row: dict[str, object] = {"start": start, "candidate_id": int(baseline["candidate_id"])}
        for field in success_fields:
            before = baseline[field] == "True"
            after = extended[field] == "True"
            row[f"{field}_e{baseline_epochs}"] = before
            row[f"{field}_e{extended_epochs}"] = after
            row[f"{field}_transition"] = _transition(before, after)
        for field in metric_fields:
            before = float(baseline[field])
            after = float(extended[field])
            row[f"{field}_e{baseline_epochs}"] = before
            row[f"{field}_e{extended_epochs}"] = after
            row[f"{field}_delta"] = after - before
        output_rows.append(row)
    for field in success_fields:
        labels = [row[f"{field}_transition"] for row in output_rows]
        transitions[field] = {label: labels.count(label) for label in ("both", "gained", "lost", "neither")}
    metric_summary = {
        field: {
            "median_delta": float(np.median([row[f"{field}_delta"] for row in output_rows])),
            "improved": int(sum(row[f"{field}_delta"] < 0.0 for row in output_rows)),
            "worsened": int(sum(row[f"{field}_delta"] > 0.0 for row in output_rows)),
        }
        for field in metric_fields
    }
    baseline_summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    extended_summary = json.loads((extended_dir / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "status": "ok",
        "baseline_epochs": baseline_epochs,
        "extended_epochs": extended_epochs,
        "only_config_change": only_config_change,
        "initial_candidates_exact": all_initial_exact,
        "initial_physical_candidates_exact": physical_initial_exact,
        "initial_coordinate_max_abs_difference": initial_coordinate_difference,
        "prefix": prefix,
        "validation_epoch_prefix_max_abs_difference": validation_prefix_difference,
        "success": {
            field: {
                "baseline_count": baseline_summary[field]["count"],
                "extended_count": extended_summary[field]["count"],
                "transitions": transitions[field],
            }
            for field in success_fields
        },
        "metrics": metric_summary,
        "timing": {
            "baseline_end_to_end_seconds": baseline_summary["end_to_end_seconds"],
            "extended_end_to_end_seconds": extended_summary["end_to_end_seconds"],
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_rows(output_dir / "per_start.csv", output_rows)
    _plot_epoch_comparison(output_dir / "comparison.png", summary, output_rows)
    return summary


def archive_completed_run(config: ExperimentConfig, result_dir: Path) -> dict[str, object]:
    """Select validation-feasible checkpoints and evaluate test afterwards."""
    result_dir = Path(result_dir)
    with np.load(_single_gradient_history(result_dir)) as history:
        epochs = np.asarray(history["validation_epoch"])
        validation_key = (
            "validation_raw_total_mse" if "validation_raw_total_mse" in history.files else "validation_total_loss"
        )
        validation_loss = np.asarray(history[validation_key])
        validation_counts = np.asarray(history["validation_spike_counts"])
        physical_history = np.asarray(history["physical_parameters"])
        optimizer_history = np.asarray(history["optimizer_z"])
    dataset = _load_result_dataset(result_dir)
    validation_protocols = dataset.subset("validation")[2]
    target_counts = np.asarray([item.target_spike_count for item in validation_protocols])
    feasible = np.all(validation_counts == target_counts[None, None, :], axis=2)
    valid = np.any(feasible, axis=0)
    selected_checkpoint = np.argmin(np.where(feasible, validation_loss, np.inf), axis=0)
    selected_checkpoint = np.where(valid, selected_checkpoint, np.argmin(validation_loss, axis=0))
    starts = np.arange(validation_loss.shape[1])
    selected_epoch = epochs[selected_checkpoint]
    selected_validation_loss = validation_loss[selected_checkpoint, starts]
    selected_physical = physical_history[selected_epoch, starts]
    selected_z = optimizer_history[selected_epoch, starts]
    current, target, test_protocols = dataset.subset("test")
    with jax.enable_x64(True), brainstate.environ.context(dt=dataset.dt_ms * u.ms, precision=64):
        test = ForwardEvaluator(config, current, target).evaluate(selected_physical)
    target_parameter = np.asarray(config.model.parameter_space.target)
    parameter_rms = np.sqrt(np.mean(((selected_physical - target_parameter) / target_parameter) ** 2, axis=1))
    trace_success = valid & (np.sqrt(selected_validation_loss) <= config.reporting.validation_rmse_threshold_mv)
    parameter_success = parameter_rms <= config.reporting.parameter_relative_rms_threshold
    test_target_counts = np.asarray([item.target_spike_count for item in test_protocols])
    test_count_exact = np.all(test.spike_counts == test_target_counts[None, :], axis=1)
    test_rmse = np.sqrt(test.raw_total_mse)
    test_trace_success = test_count_exact & (test_rmse <= config.reporting.validation_rmse_threshold_mv)
    summary = {
        "valid_validation_feasible": int(np.sum(valid)),
        "validation_trace_success": int(np.sum(trace_success)),
        "parameter_success": int(np.sum(parameter_success)),
        "joint_success": int(np.sum(trace_success & parameter_success)),
        "test_count_exact": int(np.sum(test_count_exact)),
        "test_trace_success": int(np.sum(test_trace_success)),
        "median_test_rmse_mv": float(np.median(test_rmse)),
        "selected_epoch_median": float(np.median(selected_epoch)),
        "selected_epoch_min": int(np.min(selected_epoch)),
        "selected_epoch_max": int(np.max(selected_epoch)),
        "selection_split": "validation",
        "test_used_for_selection": False,
    }
    np.savez_compressed(
        result_dir / "validation_archive.npz",
        valid=valid,
        selected_checkpoint=selected_checkpoint,
        selected_epoch=selected_epoch,
        optimizer_z=selected_z,
        physical_parameters=selected_physical,
        validation_loss=selected_validation_loss,
        parameter_relative_rms=parameter_rms,
        test_loss=test.raw_total_mse,
        test_spike_counts=test.spike_counts,
    )
    _write_json(result_dir / "validation_archive_summary.json", summary)
    return summary


def _save_gradient_stage(directory: Path, stage: GradientStageResult) -> None:
    np.savez_compressed(
        directory / "history.npz",
        state_epoch=stage.state_epoch,
        optimizer_z=stage.optimizer_z,
        physical_parameters=stage.physical_parameters,
        train_loss=stage.train_loss,
        validation_epoch=stage.validation_epoch,
        validation_total_loss=stage.validation.total_loss,
        validation_raw_total_mse=stage.validation.raw_total_mse,
        validation_protocol_loss=stage.validation.protocol_loss,
        validation_spike_counts=stage.validation.spike_counts,
        test_epoch=stage.test_epoch,
        test_total_loss=stage.test.total_loss,
        test_raw_total_mse=stage.test.raw_total_mse,
        test_protocol_loss=stage.test.protocol_loss,
        test_spike_counts=stage.test.spike_counts,
        gradient=stage.gradient,
        gradient_norm=stage.gradient_norm,
        gradient_seconds=stage.gradient_seconds,
        optimizer_seconds=stage.update_seconds,
    )
    _write_json(
        directory / "timing.json",
        {
            "compile_seconds": stage.compile_seconds,
            "stage_seconds": stage.stage_seconds,
            "gradient_median_seconds": float(np.median(stage.gradient_seconds)),
            "optimizer_median_seconds": float(np.median(stage.update_seconds)),
            "memory": stage.memory,
            "forward_evaluations": stage.forward_evaluations,
        },
    )


def _summarize(config, dataset, stage, *, end_to_end_seconds):
    final_physical = stage.physical_parameters[-1]
    validation_loss = stage.validation.raw_total_mse[-1]
    validation_rmse = np.sqrt(validation_loss)
    test_loss = stage.test.raw_total_mse[-1]
    test_rmse = np.sqrt(test_loss)
    target = np.asarray(config.model.parameter_space.target)
    parameter_rms = np.sqrt(np.mean(((final_physical - target) / target) ** 2, axis=1))
    validation_protocols = dataset.subset("validation")[2]
    validation_target_counts = np.asarray([item.target_spike_count for item in validation_protocols])
    count_exact = np.all(stage.validation.spike_counts[-1] == validation_target_counts[None], axis=1)
    finite = (
        np.isfinite(validation_rmse)
        & np.isfinite(test_rmse)
        & np.isfinite(parameter_rms)
        & np.all(np.isfinite(final_physical), axis=1)
    )
    trace_success = finite & count_exact & (validation_rmse <= config.reporting.validation_rmse_threshold_mv)
    parameter_success = finite & (parameter_rms <= config.reporting.parameter_relative_rms_threshold)
    joint_success = trace_success & parameter_success
    rows = []
    for start in range(final_physical.shape[0]):
        row = {
            "start": start,
            "candidate_id": int(stage.output_candidates.candidate_id[start]),
            "initial_train_objective": float(stage.train_loss[0, start]),
            "final_train_objective": float(stage.train_loss[-1, start]),
            "final_train_mse": float(stage.final_train.raw_total_mse[start]),
            "final_validation_mse": float(validation_loss[start]),
            "final_test_mse": float(test_loss[start]),
            "validation_rmse_mv": float(validation_rmse[start]),
            "test_rmse_mv": float(test_rmse[start]),
            "validation_count_exact": bool(count_exact[start]),
            "parameter_relative_rms": float(parameter_rms[start]),
            "finite": bool(finite[start]),
            "trace_success": bool(trace_success[start]),
            "parameter_success": bool(parameter_success[start]),
            "joint_success": bool(joint_success[start]),
        }
        for index, name in enumerate(config.model.parameter_space.names):
            row[f"initial/{name}"] = float(stage.physical_parameters[0, start, index])
            row[f"final/{name}"] = float(final_physical[start, index])
        rows.append(row)
    total = len(rows)
    summary = {
        "status": "ok",
        "config_digest": config.digest(),
        "model": config.model.name,
        "n_cv": 1,
        "num_parameters": config.model.parameter_space.size,
        "num_starts": total,
        "epochs": int(stage.state_epoch[-1]),
        "split_counts": {split: int(dataset.indices(split).size) for split in SPLITS},
        "median_initial_train_objective": float(np.median(stage.train_loss[0])),
        "median_final_train_objective": float(np.median(stage.train_loss[-1])),
        "median_final_train_mse": float(np.median(stage.final_train.raw_total_mse)),
        "median_final_validation_mse": float(np.median(validation_loss)),
        "median_final_validation_rmse_mv": float(np.median(validation_rmse)),
        "median_final_test_mse": float(np.median(test_loss)),
        "median_final_test_rmse_mv": float(np.median(test_rmse)),
        "median_parameter_relative_rms": float(np.median(parameter_rms)),
        "finite_endpoints": int(np.sum(finite)),
        "validation_rmse_pass": int(np.sum(validation_rmse <= config.reporting.validation_rmse_threshold_mv)),
        "validation_count_exact": int(np.sum(count_exact)),
        "compile_seconds": stage.compile_seconds,
        "stage_seconds": stage.stage_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "gradient_median_seconds": float(np.median(stage.gradient_seconds)),
        "optimizer_median_seconds": float(np.median(stage.update_seconds)),
        "memory": stage.memory,
    }
    for name, mask in (
        ("trace_success", trace_success),
        ("parameter_success", parameter_success),
        ("joint_success", joint_success),
    ):
        count = int(np.sum(mask))
        summary[name] = {
            "count": count,
            "fraction": count / total,
            "wilson_95": list(wilson_interval(count, total)),
        }
    return rows, summary


def _write_metrics(path: Path, dataset: DatasetBundle, stage: GradientStageResult) -> None:
    rows = []
    for epoch_index, epoch in enumerate(stage.state_epoch):
        for start, value in enumerate(stage.train_loss[epoch_index]):
            rows.append(
                {
                    "stage": stage.name,
                    "epoch": int(epoch),
                    "start": start,
                    "split": "train",
                    "metric": "mse",
                    "value": float(value),
                }
            )
    for epoch_index, epoch in enumerate(stage.validation_epoch):
        for start, value in enumerate(stage.validation.raw_total_mse[epoch_index]):
            rows.append(
                {
                    "stage": stage.name,
                    "epoch": int(epoch),
                    "start": start,
                    "split": "validation",
                    "metric": "mse",
                    "value": float(value),
                }
            )
    for start, value in enumerate(stage.test.raw_total_mse[-1]):
        rows.append(
            {
                "stage": stage.name,
                "epoch": int(stage.test_epoch[-1]),
                "start": start,
                "split": "test",
                "metric": "mse",
                "value": float(value),
            }
        )
    _write_rows(path, rows)


def _format_report(config: ExperimentConfig, dataset: DatasetBundle, summary) -> str:
    parameter = config.model.parameter_space
    optimizer_name = config.stages[-1].describe().get("optimizer", config.stages[-1].name)
    run_label = config.artifact_label or config.name
    loss_name = config.loss.describe()["name"]
    protocols = "\n".join(
        f"| {item.split} | `{item.protocol_id}` | {item.amplitude_na:.6g} | {item.target_spike_count} |"
        for item in dataset.protocols
    )
    parameter_rows = "\n".join(
        f"| `{name}` | {target:g} | {lower:g} | {upper:g} | {parameter.unit} |"
        for name, target, lower, upper in zip(parameter.names, parameter.target, parameter.lower, parameter.upper)
    )
    trace = summary["trace_success"]
    return f"""# {run_label} 实验结果

## 配置

- 模型：`{config.model.name}`，1 CV，3个bounded direct conductance parameters。
- 参数坐标：optimizer更新`z`，runtime使用`lower + (upper-lower) * sigmoid(z)`。
- 数据：Step-only，train/validation/test=`5/2/1`。
- Loss：`{loss_name}`；下表RMSE仍使用protocol等权raw voltage MSE。
- 优化：64个start一次并行，exact RTRL + {optimizer_name}，{summary['epochs']} updates。
- Test：只在最终状态评价。
- Validation：每{config.stages[-1].validation_every}轮评价一次，不影响optimizer trajectory。

| 参数 | Target | Lower | Upper | Unit |
| --- | ---: | ---: | ---: | --- |
{parameter_rows}

## 协议

| Split | Protocol | Current nA | Target spikes |
| --- | --- | ---: | ---: |
{protocols}

## 结果

| 指标 | 结果 |
| --- | ---: |
| Trace success | {trace['count']}/{summary['num_starts']} = {trace['fraction']:.3%} |
| Trace Wilson 95% | [{trace['wilson_95'][0]:.3%}, {trace['wilson_95'][1]:.3%}] |
| Parameter success | {summary['parameter_success']['count']}/{summary['num_starts']} |
| Joint success | {summary['joint_success']['count']}/{summary['num_starts']} |
| Median train MSE | {summary['median_final_train_mse']:.6g} mV^2 |
| Median validation RMSE | {summary['median_final_validation_rmse_mv']:.6g} mV |
| Median test RMSE | {summary['median_final_test_rmse_mv']:.6g} mV |
| Median parameter relative RMS | {summary['median_parameter_relative_rms']:.6g} |
| Compile time | {summary['compile_seconds']:.3f} s |
| Stage time | {summary['stage_seconds']:.3f} s |
| End-to-end time | {summary['end_to_end_seconds']:.3f} s |
| XLA temporary | {summary['memory']['temporary_bytes'] / 2**20:.3f} MiB |
"""


def _plot_dataset(path: Path, dataset: DatasetBundle) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True, constrained_layout=True)
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    seen = set()
    for index, protocol in enumerate(dataset.protocols):
        label = protocol.split if protocol.split not in seen else None
        seen.add(protocol.split)
        axes[0].plot(dataset.time_ms, dataset.current_na[index], color=colors[protocol.split], alpha=0.75, label=label)
        axes[1].plot(dataset.time_ms, dataset.target_voltage_mv[index, :, 0], color=colors[protocol.split], alpha=0.75)
    axes[0].set(title="Step currents", ylabel="current (nA)")
    axes[1].set(title="Target voltage", xlabel="time (ms)", ylabel="V (mV)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_losses(path: Path, stage: GradientStageResult) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    _plot_band(axes[0], stage.state_epoch, stage.train_loss, "train objective")
    _plot_band(axes[0], stage.validation_epoch, stage.validation.total_loss, "validation objective")
    if not np.allclose(stage.validation.total_loss, stage.validation.raw_total_mse):
        _plot_band(
            axes[0],
            stage.validation_epoch,
            stage.validation.raw_total_mse,
            "validation raw MSE",
            linestyle="--",
        )
    axes[0].set(
        xlabel="epoch",
        ylabel="configured objective (mV squared)",
        yscale="log",
        title="State-aligned objective",
    )
    axes[0].grid(True)
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        [stage.final_train.raw_total_mse, stage.validation.raw_total_mse[-1], stage.test.raw_total_mse[-1]],
        tick_labels=("train", "validation", "test"),
        showfliers=False,
    )
    axes[1].set(yscale="log", ylabel="final MSE (mV squared)", title="Final-only test comparison")
    axes[1].grid(True, axis="y")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_success(path: Path, config: ExperimentConfig, stage: GradientStageResult, summary) -> None:
    import matplotlib.pyplot as plt

    names = ("trace_success", "parameter_success", "joint_success")
    fractions = [summary[name]["fraction"] for name in names]
    intervals = [summary[name]["wilson_95"] for name in names]
    error = np.asarray(
        [
            [value - bounds[0] for value, bounds in zip(fractions, intervals)],
            [bounds[1] - value for value, bounds in zip(fractions, intervals)],
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    axes[0].bar(np.arange(3), fractions, yerr=error, capsize=4)
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=("trace", "parameter", "joint"),
        ylim=(0, 1),
        ylabel="fraction",
        title="Final success",
    )
    physical = np.asarray(stage.output_candidates.physical)
    target = np.asarray(config.model.parameter_space.target)
    parameter_rms = np.sqrt(np.mean(((physical - target) / target) ** 2, axis=1))
    axes[1].scatter(np.sqrt(stage.validation.raw_total_mse[-1]), parameter_rms, alpha=0.75)
    axes[1].set(xlabel="validation RMSE (mV)", ylabel="parameter relative RMS", title="Endpoint distribution")
    for axis in axes:
        axis.grid(True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_parameters(path: Path, config: ExperimentConfig, stage: GradientStageResult) -> None:
    import matplotlib.pyplot as plt

    count = config.model.parameter_space.size
    figure, axes = plt.subplots(count, 2, figsize=(11.0, 3.2 * count), constrained_layout=True)
    for index, name in enumerate(config.model.parameter_space.names):
        axes[index, 0].plot(stage.state_epoch, stage.optimizer_z[:, :, index], alpha=0.18)
        axes[index, 0].set(ylabel="z", title=f"{name}: optimizer coordinate")
        axes[index, 1].plot(stage.state_epoch, stage.physical_parameters[:, :, index], alpha=0.18)
        axes[index, 1].axhline(config.model.parameter_space.target[index], color="black", linestyle="--")
        axes[index, 1].set(ylabel=str(config.model.parameter_space.unit), title=f"{name}: physical")
        for axis in axes[index]:
            axis.grid(True)
            if index == count - 1:
                axis.set_xlabel("epoch")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_traces(path: Path, config: ExperimentConfig, dataset: DatasetBundle, stage: GradientStageResult) -> None:
    import matplotlib.pyplot as plt

    validation_rmse = np.sqrt(stage.validation.raw_total_mse[-1])
    order = np.argsort(validation_rmse)
    selected = np.asarray([order[0], order[len(order) // 2], order[-1]], dtype=np.int32)
    labels = ("best", "median", "worst")
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 9.0), constrained_layout=True)
    for column, split in enumerate(SPLITS):
        current, target, protocols = dataset.subset(split)
        evaluator = ForwardEvaluator(config, current, target)
        prediction = evaluator.traces(stage.physical_parameters[-1, selected])
        protocol_index = 0
        for row, label in enumerate(labels):
            axis = axes[row, column]
            axis.plot(dataset.time_ms, target[protocol_index, :, 0], color="black", label="target")
            axis.plot(dataset.time_ms, prediction[row, protocol_index, :, 0], label=label)
            if row == 0:
                axis.set_title(f"{split}: {protocols[protocol_index].protocol_id}")
            if column == 0:
                axis.set_ylabel(f"{label}\nV (mV)")
            if row == 2:
                axis.set_xlabel("time (ms)")
            axis.grid(True)
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_band(axis, epochs, values, label, *, linestyle="-") -> None:
    array = np.asarray(values)
    median = np.median(array, axis=1)
    lower = np.percentile(array, 25, axis=1)
    upper = np.percentile(array, 75, axis=1)
    line = axis.plot(epochs, median, label=label, linestyle=linestyle)[0]
    axis.fill_between(epochs, lower, upper, color=line.get_color(), alpha=0.2)


def _plot_epoch_comparison(path: Path, summary, rows) -> None:
    import matplotlib.pyplot as plt

    baseline = summary["baseline_epochs"]
    extended = summary["extended_epochs"]
    names = ("trace_success", "parameter_success", "joint_success")
    same_epoch = baseline == extended
    baseline_label = "baseline" if same_epoch else f"epoch {baseline}"
    if not same_epoch:
        extended_label = f"epoch {extended}"
    elif summary["only_config_change"] == "model.parameter_space bounds":
        extended_label = "wide bounds"
    elif summary["only_config_change"] == "stages[0].learning_rate":
        extended_label = "LR 0.02"
    elif summary["only_config_change"] == "stages[0].optimizer":
        extended_label = "Rprop"
    elif summary["only_config_change"] == "loss":
        extended_label = "balanced loss"
    else:
        extended_label = "LR 0.02 + wide bounds"
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    axes[0].bar(
        x - 0.18,
        [summary["success"][name]["baseline_count"] for name in names],
        width=0.36,
        label=baseline_label,
    )
    axes[0].bar(
        x + 0.18,
        [summary["success"][name]["extended_count"] for name in names],
        width=0.36,
        label=extended_label,
    )
    axes[0].set(xticks=x, xticklabels=("trace", "parameter", "joint"), ylabel="successful starts")
    axes[0].set_title("Endpoint success")
    axes[0].legend(frameon=False)
    transitions = summary["success"]["trace_success"]["transitions"]
    axes[1].bar(tuple(transitions), tuple(transitions.values()))
    axes[1].set(title="Trace-success transitions", ylabel="starts")
    fields = ("final_train_mse", "final_validation_mse", "final_test_mse")
    axes[2].boxplot(
        [[row[f"{field}_delta"] for row in rows] for field in fields],
        tick_labels=("train", "validation", "test"),
        showfliers=False,
    )
    axes[2].axhline(0.0, color="black", linestyle="--")
    delta_title = "changed - baseline" if same_epoch else f"epoch {extended} - {baseline}"
    axes[2].set(title=f"Metric change: {delta_title}", ylabel="MSE delta")
    for axis in axes:
        axis.grid(True, axis="y")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "both"
    if not before and after:
        return "gained"
    if before and not after:
        return "lost"
    return "neither"


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson binomial proportion interval."""
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2)) / denominator
    return center - radius, center + radius


def _environment_metadata() -> dict[str, object]:
    def git(*arguments):
        return subprocess.run(("git", *arguments), capture_output=True, text=True, check=False).stdout.strip()

    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "git_revision": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": bool(git("status", "--short")),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _load_result_dataset(directory: Path) -> DatasetBundle:
    directory = Path(directory)
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    with np.load(directory / "dataset.npz") as values:
        protocols = tuple(
            Protocol(str(protocol_id), str(split), str(feature), float(amplitude), int(count))
            for protocol_id, split, feature, amplitude, count in zip(
                values["protocol_id"],
                values["split"],
                values["feature"],
                values["amplitude_nA"],
                values["target_spike_count"],
            )
        )
        return DatasetBundle(
            protocols,
            np.asarray(values["time_ms"]),
            np.asarray(values["current_nA"]),
            np.asarray(values["target_voltage_mV"]),
            float(manifest["dt_ms"]),
            float(manifest["baseline_stop_ms"]),
            float(manifest["stimulus_stop_ms"]),
        )


def _single_gradient_history(directory: Path) -> Path:
    matches = tuple((Path(directory) / "stages").glob("00_*/history.npz"))
    if len(matches) != 1:
        raise ValueError(f"Expected one stage-0 gradient history in {directory}, found {len(matches)}.")
    return matches[0]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
