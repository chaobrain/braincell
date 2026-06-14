"""Notebook-facing workflow helpers for channel_no_conc comparisons."""



import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_ENGINE_ROOT = _ROOT / "engine"
_REPO_ROOT = next(
    (
        candidate
        for candidate in (_ROOT, *_ROOT.parents)
        if (candidate / "braincell").exists() and (candidate / "examples").exists()
    ),
    _ROOT,
)
for candidate in (_REPO_ROOT, _HERE, _ROOT, _ENGINE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import experiment_schema
    import run as channel_run
    from outputs import default_config_run_output_dir, default_sweep_output_dir
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import channel_no_conc template modules for notebook workflow helpers.") from exc


_BATCH_CONFIG_COLUMNS = (
    "config_name",
    "config_path",
    "out_dir",
    "batch_status",
    "n_templates",
    "n_total_cases",
    "n_success_cases",
    "n_failed_cases",
    "error_message",
)
_BATCH_OBSERVABLE_COLUMNS = (
    "config_name",
    "config_path",
    "batch_status",
    "observable",
    "n_cases",
    "mae_mean",
    "rmse_mean",
    "max_abs_max",
    "rel_mae_pct_mean",
)
_BATCH_FAILURE_COLUMNS = (
    "config_name",
    "config_path",
    "batch_status",
    "template_name",
    "template_path",
    "out_dir",
    "n_failed_cases",
    "error_message",
)


def find_repo_root(start: str | Path | None = None) -> Path:
    cwd = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "braincell").exists() and (candidate / "examples").exists():
            return candidate
    raise RuntimeError("Run from the repository root or provide a path inside the repository.")


def load_workflow_inputs(config_path: str | Path, template_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_template_path = _resolve_template_path(config_path=resolved_config_path, template_path=template_path)
    config = experiment_schema.load_sweep_config(resolved_config_path, resolved_template_path)
    normalized_config = experiment_schema.config_to_payload(config)
    expanded_cases = experiment_schema.expand_cases(config)

    mod_dir = Path(str(normalized_config["identity"]["mod_dir"])).expanduser().resolve()
    if not mod_dir.exists():
        raise FileNotFoundError(f"identity.mod_dir does not exist: {mod_dir!s}.")
    if not mod_dir.is_dir():
        raise NotADirectoryError(f"identity.mod_dir must be a directory: {mod_dir!s}.")

    default_out_dir = default_sweep_output_dir(
        channel_no_conc_root=_ROOT,
        config_id=config.config_id,
    ).resolve()

    return {
        "config_path": resolved_config_path,
        "template_path": resolved_template_path,
        "config_name": config.config_name,
        "template_name": config.template_name,
        "run_id": config.config_id,
        "group_id": normalized_config["template"]["group"]["group_id"],
        "mod_dir": mod_dir,
        "mapping": normalized_config["mapping"],
        "normalized_config": normalized_config,
        "n_expanded_cases": len(expanded_cases),
        "default_out_dir": default_out_dir,
    }


def load_config_workflow_inputs(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    model_config = experiment_schema.load_model_config(resolved_config_path)
    template_paths = tuple(Path(path).expanduser().resolve() for path in model_config.template_paths)
    mod_dir = Path(str(model_config.mod_dir)).expanduser().resolve()
    if not mod_dir.exists():
        raise FileNotFoundError(f"identity.mod_dir does not exist: {mod_dir!s}.")
    if not mod_dir.is_dir():
        raise NotADirectoryError(f"identity.mod_dir must be a directory: {mod_dir!s}.")

    default_out_dir = default_config_run_output_dir(
        channel_no_conc_root=_ROOT,
        config_name=model_config.config_name,
    ).resolve()
    return {
        "config_path": resolved_config_path,
        "config_name": model_config.config_name,
        "mod_dir": mod_dir,
        "template_paths": template_paths,
        "template_names": tuple(path.stem for path in template_paths),
        "n_templates": len(template_paths),
        "default_out_dir": default_out_dir,
    }


def discover_batch_configs(config_dir: str | Path) -> list[dict[str, Any]]:
    resolved_config_dir = Path(config_dir).expanduser().resolve()
    if not resolved_config_dir.exists():
        raise FileNotFoundError(
            f"channel_no_conc batch config directory does not exist: {resolved_config_dir!s}."
        )
    if not resolved_config_dir.is_dir():
        raise NotADirectoryError(
            f"channel_no_conc batch config path must be a directory: {resolved_config_dir!s}."
        )

    config_paths = sorted(
        path
        for path in resolved_config_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    )
    if len(config_paths) == 0:
        raise ValueError(f"No channel_no_conc config JSON files were found in {resolved_config_dir!s}.")

    records: list[dict[str, Any]] = []
    for config_path in config_paths:
        try:
            info = load_config_workflow_inputs(config_path)
        except Exception as exc:
            raise RuntimeError(
                f"channel_no_conc batch discovery failed for {config_path!s}: {exc}"
            ) from exc
        records.append(
            {
                "config_path": info["config_path"],
                "config_name": info["config_name"],
                "run_id": info["config_name"],
                "mod_dir": info["mod_dir"],
                "template_paths": list(info["template_paths"]),
                "template_names": list(info["template_names"]),
                "n_templates": info["n_templates"],
                "default_out_dir": info["default_out_dir"],
            }
        )
    return records


def run_notebook_workflow(
    *,
    config_path: str | Path,
    template_path: str | Path,
    out_dir: str | Path | None = None,
    plot: bool | None = None,
    expand_only: bool = False,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    inputs = load_workflow_inputs(config_path, template_path)
    config = experiment_schema.load_sweep_config(inputs["config_path"], inputs["template_path"])
    resolved_out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else inputs["default_out_dir"]
    )

    status = channel_run.run_sweep_config(
        config,
        out_dir=resolved_out_dir,
        expand_only=bool(expand_only),
        plot=plot,
    )

    run_info = {
        "status": int(status),
        "config_path": inputs["config_path"],
        "template_path": inputs["template_path"],
        "run_id": inputs["run_id"],
        "out_dir": resolved_out_dir,
        "normalized_config_path": resolved_out_dir / "normalized_config.json",
        "expanded_cases_path": resolved_out_dir / "expanded_cases.json",
        "case_metrics_path": resolved_out_dir / "case_metrics.csv",
        "aggregate_path": resolved_out_dir / "aggregate.json",
        "case_results_dir": resolved_out_dir / "case_results",
        "plots_dir": resolved_out_dir / "plots",
    }

    if raise_on_failure and status != 0:
        aggregate = json.loads(run_info["aggregate_path"].read_text())
        raise RuntimeError(
            "channel_no_conc workflow completed with failed cases: "
            f"{aggregate.get('n_failed_cases', 'unknown')}."
        )
    return run_info


def run_notebook_config_workflow(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    plot: bool | None = None,
    expand_only: bool = False,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    inputs = load_config_workflow_inputs(config_path)
    resolved_out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else inputs["default_out_dir"]
    )
    result = channel_run.run_config_file(
        inputs["config_path"],
        out_dir=resolved_out_dir,
        expand_only=bool(expand_only),
        plot=plot,
    )
    run_info = {
        "status": int(result["status"]),
        "config_path": inputs["config_path"],
        "config_name": inputs["config_name"],
        "out_dir": resolved_out_dir,
        "manifest_path": Path(result["manifest_path"]).resolve(),
        "config_runs_path": Path(result["config_runs_path"]).resolve(),
        "observable_summary_path": Path(result["observable_summary_path"]).resolve(),
        "observable_summary_json_path": Path(result["observable_summary_json_path"]).resolve(),
        "failures_path": Path(result["failures_path"]).resolve(),
        "template_runs": list(result["template_runs"]),
        "observables": list(result["observables"]),
        "failures": list(result["failures"]),
        "status_counts": dict(result["status_counts"]),
        "n_templates": int(result["n_templates"]),
        "n_total_cases": sum(
            int(row["n_total_cases"])
            for row in result["template_runs"]
            if str(row.get("n_total_cases", "")) not in {"", "None"}
        ),
        "n_success_cases": sum(
            int(row["n_success_cases"])
            for row in result["template_runs"]
            if str(row.get("n_success_cases", "")) not in {"", "None"}
        ),
        "n_failed_cases": sum(
            int(row["n_failed_cases"])
            for row in result["template_runs"]
            if str(row.get("n_failed_cases", "")) not in {"", "None"}
        ),
    }
    if raise_on_failure and run_info["status"] != 0:
        raise RuntimeError(
            "channel_no_conc config workflow completed with failed templates: "
            f"{run_info['status_counts'].get('failed', 'unknown')}."
        )
    return run_info


def make_batch_run_id(*, now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%y%m%d_%H%M%S")


def default_batch_run_output_dir(*, batch_run_id: str, summary_dir: str | Path | None = None) -> Path:
    if summary_dir is not None:
        return Path(summary_dir).expanduser().resolve()
    return (_ROOT / "results" / "batch_runs" / batch_run_id).resolve()


def build_batch_config_rows(config_run_infos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_info in config_run_infos:
        status_counts = dict(run_info.get("status_counts", {}))
        if int(run_info.get("status", 1)) != 0:
            batch_status = "failed"
            error_message = "Config run exited with non-zero status."
        elif int(status_counts.get("failed", 0)) > 0 or int(status_counts.get("partial", 0)) > 0:
            batch_status = "partial"
            error_message = (
                f"{int(status_counts.get('failed', 0))} failed template(s), "
                f"{int(status_counts.get('partial', 0))} partial template(s)."
            )
        else:
            batch_status = "ok"
            error_message = ""

        rows.append(
            {
                "config_name": str(run_info["config_name"]),
                "config_path": str(Path(str(run_info["config_path"])).expanduser().resolve()),
                "out_dir": str(Path(str(run_info["out_dir"])).expanduser().resolve()),
                "batch_status": batch_status,
                "n_templates": int(run_info.get("n_templates", 0)),
                "n_total_cases": int(run_info.get("n_total_cases", 0)),
                "n_success_cases": int(run_info.get("n_success_cases", 0)),
                "n_failed_cases": int(run_info.get("n_failed_cases", 0)),
                "error_message": error_message,
            }
        )
    return rows


def build_batch_observable_rows(config_run_infos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_info in config_run_infos:
        summary_path = Path(str(run_info["observable_summary_json_path"])).expanduser().resolve()
        payload = json.loads(summary_path.read_text())
        config_rows = payload.get("all_templates", {}).get("observables", {})
        status_counts = dict(run_info.get("status_counts", {}))
        if int(run_info.get("status", 1)) != 0:
            batch_status = "failed"
        elif int(status_counts.get("failed", 0)) > 0 or int(status_counts.get("partial", 0)) > 0:
            batch_status = "partial"
        else:
            batch_status = "ok"

        for observable, metrics in sorted(config_rows.items()):
            rows.append(
                {
                    "config_name": str(run_info["config_name"]),
                    "config_path": str(Path(str(run_info["config_path"])).expanduser().resolve()),
                    "batch_status": batch_status,
                    "observable": str(observable),
                    "n_cases": int(metrics.get("n_cases", 0)),
                    "mae_mean": float(metrics.get("mae_mean", 0.0)),
                    "rmse_mean": float(metrics.get("rmse_mean", 0.0)),
                    "max_abs_max": float(metrics.get("max_abs_max", 0.0)),
                    "rel_mae_pct_mean": float(metrics.get("rel_mae_pct_mean", 0.0)),
                }
            )
    return rows


def build_batch_failure_rows(config_run_infos: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_info in config_run_infos:
        status_counts = dict(run_info.get("status_counts", {}))
        if int(run_info.get("status", 1)) != 0:
            batch_status = "failed"
        elif int(status_counts.get("failed", 0)) > 0 or int(status_counts.get("partial", 0)) > 0:
            batch_status = "partial"
        else:
            batch_status = "ok"
        for failure in run_info.get("failures", []):
            rows.append(
                {
                    "config_name": str(run_info["config_name"]),
                    "config_path": str(Path(str(run_info["config_path"])).expanduser().resolve()),
                    "batch_status": batch_status,
                    "template_name": str(failure.get("template_name", "")),
                    "template_path": str(failure.get("template_path", "")),
                    "out_dir": str(failure.get("out_dir", "")),
                    "n_failed_cases": failure.get("n_failed_cases", ""),
                    "error_message": str(failure.get("error_message", "")),
                }
            )
    return rows


def write_batch_summary_artifacts(
    *,
    config_dir: str | Path,
    summary_dir: str | Path,
    batch_run_id: str,
    config_records: Sequence[Mapping[str, Any]],
    config_run_infos: Sequence[Mapping[str, Any]],
    plot_cases: bool,
) -> dict[str, Any]:
    resolved_summary_dir = Path(summary_dir).expanduser().resolve()
    resolved_summary_dir.mkdir(parents=True, exist_ok=True)

    config_rows = build_batch_config_rows(config_run_infos)
    observable_rows = build_batch_observable_rows(config_run_infos)
    failure_rows = build_batch_failure_rows(config_run_infos)

    manifest = {
        "batch_run_id": batch_run_id,
        "summary_dir": str(resolved_summary_dir),
        "config_dir": str(Path(config_dir).expanduser().resolve()),
        "n_configs": len(config_records),
        "plot_cases": bool(plot_cases),
        "status_counts": {
            status: sum(1 for row in config_rows if row["batch_status"] == status)
            for status in ("ok", "partial", "failed")
        },
        "config_names": [str(record["config_name"]) for record in config_records],
        "summary_files": {
            "config_runs_csv": str(resolved_summary_dir / "config_runs.csv"),
            "batch_observable_summary_csv": str(resolved_summary_dir / "batch_observable_summary.csv"),
            "batch_observable_summary_json": str(resolved_summary_dir / "batch_observable_summary.json"),
            "batch_failures_csv": str(resolved_summary_dir / "batch_failures.csv"),
        },
    }

    batch_observable_summary_payload = {
        "batch_run_id": batch_run_id,
        "config_dir": str(Path(config_dir).expanduser().resolve()),
        "n_configs": len(config_records),
        "rows": observable_rows,
    }

    manifest_path = resolved_summary_dir / "batch_manifest.json"
    config_runs_path = resolved_summary_dir / "config_runs.csv"
    batch_observable_summary_path = resolved_summary_dir / "batch_observable_summary.csv"
    batch_observable_summary_json_path = resolved_summary_dir / "batch_observable_summary.json"
    batch_failures_path = resolved_summary_dir / "batch_failures.csv"

    _write_json(manifest_path, manifest)
    _write_csv_rows(config_runs_path, _BATCH_CONFIG_COLUMNS, config_rows)
    _write_csv_rows(batch_observable_summary_path, _BATCH_OBSERVABLE_COLUMNS, observable_rows)
    _write_json(batch_observable_summary_json_path, batch_observable_summary_payload)
    _write_csv_rows(batch_failures_path, _BATCH_FAILURE_COLUMNS, failure_rows)

    return {
        "batch_run_id": batch_run_id,
        "summary_dir": resolved_summary_dir,
        "manifest_path": manifest_path,
        "config_runs_path": config_runs_path,
        "batch_observable_summary_path": batch_observable_summary_path,
        "batch_observable_summary_json_path": batch_observable_summary_json_path,
        "batch_failures_path": batch_failures_path,
        "config_rows": config_rows,
        "observable_rows": observable_rows,
        "failure_rows": failure_rows,
    }


def load_run_artifacts(out_dir: str | Path) -> dict[str, Any]:
    import pandas as pd

    resolved_out_dir = Path(out_dir).expanduser().resolve()
    metrics_df = pd.read_csv(resolved_out_dir / "case_metrics.csv")
    for column in ("temperature_celsius", "v_init_mV", "n_samples", "mae", "rmse", "max_abs", "rel_mae_pct"):
        if column in metrics_df.columns:
            metrics_df[column] = pd.to_numeric(metrics_df[column], errors="coerce")

    cases_payload = json.loads((resolved_out_dir / "expanded_cases.json").read_text())
    cases_df = pd.json_normalize(cases_payload, sep=".")
    return {
        "out_dir": resolved_out_dir,
        "normalized_config": json.loads((resolved_out_dir / "normalized_config.json").read_text()),
        "expanded_cases": cases_payload,
        "aggregate": json.loads((resolved_out_dir / "aggregate.json").read_text()),
        "metrics_df": metrics_df,
        "cases_df": cases_df,
        "case_results_dir": resolved_out_dir / "case_results",
        "plots_dir": resolved_out_dir / "plots",
    }


def build_summary_tables(out_dir: str | Path) -> dict[str, Any]:
    artifacts = load_run_artifacts(out_dir)
    metrics_df = artifacts["metrics_df"].copy()
    cases_df = artifacts["cases_df"].copy()
    merge_keys = ["case_id", "group_id"]
    merged_df = metrics_df.merge(cases_df, on=merge_keys, how="left")

    ok_df = merged_df[merged_df["status"] == "ok"].copy()
    failed_df = merged_df[merged_df["status"] != "ok"].copy()
    worst_cases_df = ok_df.sort_values(["max_abs", "rmse", "mae"], ascending=False).reset_index(drop=True)

    if len(ok_df) == 0:
        by_observable_df = metrics_df.iloc[0:0].copy()
        by_group_df = metrics_df.iloc[0:0].copy()
    else:
        by_observable_df = (
            ok_df.groupby("observable", dropna=False)
            .agg(
                n_rows=("case_id", "count"),
                n_cases=("case_id", "nunique"),
                mae_mean=("mae", "mean"),
                rmse_mean=("rmse", "mean"),
                max_abs_max=("max_abs", "max"),
                rel_mae_pct_mean=("rel_mae_pct", "mean"),
            )
            .reset_index()
            .sort_values(["max_abs_max", "mae_mean"], ascending=False)
            .reset_index(drop=True)
        )
        by_group_df = (
            ok_df.groupby(["group_id", "observable"], dropna=False)
            .agg(
                n_rows=("case_id", "count"),
                n_cases=("case_id", "nunique"),
                mae_mean=("mae", "mean"),
                rmse_mean=("rmse", "mean"),
                max_abs_max=("max_abs", "max"),
                rel_mae_pct_mean=("rel_mae_pct", "mean"),
            )
            .reset_index()
            .sort_values(["group_id", "max_abs_max", "mae_mean"], ascending=[True, False, False])
            .reset_index(drop=True)
        )

    return {
        "aggregate": artifacts["aggregate"],
        "normalized_config": artifacts["normalized_config"],
        "metrics_df": metrics_df,
        "cases_df": cases_df,
        "merged_df": merged_df,
        "ok_df": ok_df,
        "failed_df": failed_df,
        "worst_cases_df": worst_cases_df,
        "by_observable_df": by_observable_df,
        "by_group_df": by_group_df,
    }


def load_case_result(out_dir: str | Path, case_id: str) -> dict[str, Any]:
    resolved_out_dir = Path(out_dir).expanduser().resolve()
    return json.loads((resolved_out_dir / "case_results" / f"{case_id}.json").read_text())


def plot_sweep_summary(
    summary_tables: Mapping[str, Any],
    *,
    metric: str = "max_abs",
    top_k: int = 12,
):
    import matplotlib.pyplot as plt
    import numpy as np

    ok_df = summary_tables["ok_df"]
    failed_df = summary_tables["failed_df"]
    worst_cases_df = summary_tables["worst_cases_df"]
    aggregate = summary_tables["aggregate"]

    aggregate_field = {
        "mae": "mae_mean",
        "rmse": "rmse_mean",
        "max_abs": "max_abs_max",
        "rel_mae_pct": "rel_mae_pct_mean",
    }.get(metric, "max_abs_max")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    observable_names = list(aggregate.get("observables", {}).keys())
    observable_values = [
        float(aggregate["observables"][name][aggregate_field])
        for name in observable_names
    ]
    axes[0, 0].bar(observable_names, observable_values, color="#bfd7ea", edgecolor="#2f3e46")
    axes[0, 0].set_title(f"Aggregate observable summary ({aggregate_field})")
    axes[0, 0].set_ylabel(aggregate_field)
    axes[0, 0].grid(axis="y", alpha=0.25)

    if len(ok_df) > 0:
        observables = list(dict.fromkeys(ok_df["observable"].tolist()))
        data = [ok_df.loc[ok_df["observable"] == name, metric].dropna().to_numpy() for name in observables]
        axes[0, 1].boxplot(data, tick_labels=observables, orientation="vertical")
        axes[0, 1].set_title(f"Per-case distribution of {metric}")
        axes[0, 1].set_ylabel(metric)
        axes[0, 1].grid(axis="y", alpha=0.25)
    else:
        axes[0, 1].text(0.5, 0.5, "No successful cases.", ha="center", va="center")
        axes[0, 1].set_axis_off()

    if len(worst_cases_df) > 0:
        top_df = worst_cases_df.head(top_k).copy().iloc[::-1]
        labels = [
            f"{row.case_id} / {row.observable}"
            for row in top_df.itertuples()
        ]
        axes[1, 0].barh(labels, top_df[metric], color="#84a98c", edgecolor="#2f3e46")
        axes[1, 0].set_title(f"Top {len(top_df)} worst case-observable rows by {metric}")
        axes[1, 0].set_xlabel(metric)
        axes[1, 0].grid(axis="x", alpha=0.25)
    else:
        axes[1, 0].text(0.5, 0.5, "No successful cases.", ha="center", va="center")
        axes[1, 0].set_axis_off()

    axes[1, 1].set_title("Run summary")
    axes[1, 1].axis("off")
    summary_lines = [
        f"run_id: {aggregate.get('config_id', '')}",
        f"n_total_cases: {aggregate.get('n_total_cases', 0)}",
        f"n_success_cases: {aggregate.get('n_success_cases', 0)}",
        f"n_failed_cases: {aggregate.get('n_failed_cases', 0)}",
    ]
    if len(failed_df) > 0 and "group_id" in failed_df:
        group_counts = failed_df.groupby("group_id")["case_id"].nunique().to_dict()
        summary_lines.append("")
        summary_lines.append("failed cases by group:")
        summary_lines.extend(f"  {group_id}: {count}" for group_id, count in group_counts.items())
    axes[1, 1].text(0.02, 0.98, "\n".join(summary_lines), ha="left", va="top", family="monospace")

    fig.tight_layout()
    return fig, axes


def plot_case_overlay(
    case_result: Mapping[str, Any],
    *,
    include_gates: bool = True,
):
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(case_result["time_ms"], dtype=float)
    gate_alignments = list(case_result["alignment"]["gates"])
    n_gate_axes = len(gate_alignments) if include_gates else 0
    n_axes = 2 + n_gate_axes
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 3.0 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]

    axes[0].plot(time_ms, case_result["neuron"]["voltage_mV"], linewidth=1.5, label="NEURON")
    axes[0].plot(time_ms, case_result["braincell"]["voltage_mV"], linewidth=1.3, linestyle="--", label="braincell")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].set_title("Voltage overlay")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    current_time_ms, braincell_current_ix, neuron_current_ix = _aligned_current_view(case_result)
    axes[1].plot(current_time_ms, neuron_current_ix, linewidth=1.5, label="NEURON")
    axes[1].plot(current_time_ms, braincell_current_ix, linewidth=1.3, linestyle="--", label="braincell")
    axes[1].set_ylabel("Current ix")
    axes[1].set_title("Current overlay")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for offset, gate_alignment in enumerate(gate_alignments):
        if not include_gates:
            break
        axis = axes[2 + offset]
        canonical_name = gate_alignment["canonical_name"]
        neuron_gate = gate_alignment["neuron_gate"]
        braincell_gate = gate_alignment["braincell_gate"]
        axis.plot(time_ms, case_result["neuron"]["gates"][neuron_gate], linewidth=1.5, label=f"NEURON {neuron_gate}")
        axis.plot(
            time_ms,
            case_result["braincell"]["gates"][braincell_gate],
            linewidth=1.3,
            linestyle="--",
            label=f"braincell {braincell_gate}",
        )
        axis.set_ylabel(canonical_name)
        axis.set_title(f"Gate overlay: {canonical_name}")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig, axes


def plot_case_error_summary(case_result: Mapping[str, Any]):
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(case_result["time_ms"], dtype=float)
    metric_rows = [("voltage", case_result["metrics"]["voltage"])]
    metric_rows.extend((f"current.{name}", metrics) for name, metrics in case_result["metrics"]["current"].items())
    metric_rows.extend((f"gates.{name}", metrics) for name, metrics in case_result["metrics"]["gates"].items())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [label for label, _ in metric_rows]
    mae_values = [row["mae"] for _, row in metric_rows]
    max_abs_values = [row["max_abs"] for _, row in metric_rows]
    y = range(len(labels))
    axes[0].barh([idx - 0.2 for idx in y], mae_values, height=0.35, label="MAE", color="#bfd7ea", edgecolor="#2f3e46")
    axes[0].barh([idx + 0.2 for idx in y], max_abs_values, height=0.35, label="Max abs", color="#84a98c", edgecolor="#2f3e46")
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Error")
    axes[0].set_title("Observable error summary")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend()

    voltage_abs = np.abs(
        np.asarray(case_result["braincell"]["voltage_mV"], dtype=float)
        - np.asarray(case_result["neuron"]["voltage_mV"], dtype=float)
    )
    current_time_ms, braincell_current_ix, neuron_current_ix = _aligned_current_view(case_result)
    current_abs = np.abs(
        np.asarray(braincell_current_ix, dtype=float)
        - np.asarray(neuron_current_ix, dtype=float)
    )
    axes[1].plot(time_ms, voltage_abs, linewidth=1.5, label="voltage")
    axes[1].plot(current_time_ms, current_abs, linewidth=1.5, label="current.ix")
    for gate_alignment in case_result["alignment"]["gates"]:
        canonical_name = gate_alignment["canonical_name"]
        braincell_gate = gate_alignment["braincell_gate"]
        neuron_gate = gate_alignment["neuron_gate"]
        gate_abs = np.abs(
            np.asarray(case_result["braincell"]["gates"][braincell_gate], dtype=float)
            - np.asarray(case_result["neuron"]["gates"][neuron_gate], dtype=float)
        )
        axes[1].plot(time_ms, gate_abs, linewidth=1.2, label=f"gates.{canonical_name}")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("|delta|")
    axes[1].set_title("Absolute difference traces")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


def plot_observable_metric_boxplots(
    summary_tables: Mapping[str, Any],
    *,
    metrics: tuple[str, ...] = ("mae", "rmse", "max_abs", "rel_mae_pct"),
):
    import matplotlib.pyplot as plt
    import numpy as np

    ok_df = summary_tables["ok_df"]
    if "observable" in ok_df:
        ok_df = ok_df.copy()
        ok_df["observable"] = ok_df["observable"].astype(str)
    families = (
        ("voltage", "Voltage"),
        ("current.", "Current"),
        ("gates.", "Gates"),
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.0))
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        plot_data = []
        labels = []
        for prefix, label in families:
            if prefix == "voltage":
                values = ok_df.loc[ok_df["observable"] == "voltage", metric].dropna().to_numpy()
            else:
                values = ok_df.loc[ok_df["observable"].str.startswith(prefix), metric].dropna().to_numpy()
            if values.size == 0:
                continue
            plot_data.append(values)
            labels.append(label)

        if len(plot_data) == 0:
            axis.text(0.5, 0.5, "No successful cases.", ha="center", va="center")
            axis.set_axis_off()
            continue

        axis.boxplot(plot_data, tick_labels=labels, showfliers=True)
        axis.set_title(f"{metric} by observable family")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)

        positive_values = np.concatenate([values[values > 0] for values in plot_data if np.any(values > 0)])
        if positive_values.size > 0:
            dynamic_range = positive_values.max() / positive_values.min()
            if dynamic_range >= 100.0:
                axis.set_yscale("log")

    fig.tight_layout()
    return fig, axes


def _aligned_current_view(case_result: Mapping[str, Any]):
    import numpy as np

    aligned = case_result.get("aligned", {}).get("current", {})
    if isinstance(aligned, Mapping) and {"time_ms", "braincell_ix", "neuron_ix"} <= set(aligned):
        return (
            np.asarray(aligned["time_ms"], dtype=float),
            np.asarray(aligned["braincell_ix"], dtype=float),
            np.asarray(aligned["neuron_ix"], dtype=float),
        )
    return (
        np.asarray(case_result["time_ms"], dtype=float),
        np.asarray(case_result["braincell"]["current"]["ix"], dtype=float),
        np.asarray(case_result["neuron"]["current"]["ix"], dtype=float),
    )


def _resolve_template_path(*, config_path: Path, template_path: str | Path) -> Path:
    raw_path = Path(template_path).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (config_path.parent / raw_path).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
