"""Notebook-facing workflow helpers for cable comparisons."""

import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_ENGINE_ROOT = _ROOT / "engine"
for candidate in (_HERE, _ROOT, _ENGINE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import experiment_schema
    import run as cable_run
    from outputs import default_config_run_output_dir, default_sweep_output_dir
except ImportError as exc:  # pragma: no cover
    raise ImportError("Could not import cable engine modules for notebook workflow helpers.") from exc


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


def resolve_selected_files(
    morphology_root: str | Path | None,
    selected_files: Sequence[str | Path],
    *,
    morphology_kind: str = "swc",
) -> list[Path]:
    if len(selected_files) == 0:
        raise ValueError("selected_files must be a non-empty sequence.")

    root = None if morphology_root is None else Path(morphology_root).expanduser().resolve()
    expected_suffixes = _expected_suffixes_for_kind(morphology_kind)
    resolved: list[Path] = []
    for item in selected_files:
        path = Path(item).expanduser()
        if not path.is_absolute():
            if root is None:
                raise ValueError("morphology_root is required when selected_files contains relative paths.")
            path = root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Selected morphology file does not exist: {path!s}.")
        if path.is_dir():
            raise IsADirectoryError(f"Selected morphology path must be a file, got directory: {path!s}.")
        if expected_suffixes and path.suffix.lower() not in expected_suffixes:
            raise ValueError(
                f"Selected file {path!s} does not match morphology_kind={morphology_kind!r}; "
                f"expected suffix in {sorted(expected_suffixes)!r}."
            )
        resolved.append(path)
    return resolved


def load_workflow_inputs(config_path: str | Path, template_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    resolved_template_path = _resolve_template_path(config_path=resolved_config_path, template_path=template_path)
    config = experiment_schema.load_sweep_config(resolved_config_path, resolved_template_path)
    normalized_config = experiment_schema.config_to_payload(config)
    expanded_cases = experiment_schema.expand_cases(config)

    default_out_dir = default_sweep_output_dir(
        cable_root=_ROOT,
        config_id=config.config_id,
    ).resolve()

    return {
        "config_path": resolved_config_path,
        "template_path": resolved_template_path,
        "config_name": config.config_name,
        "template_name": config.template_name,
        "run_id": config.config_id,
        "group_id": normalized_config["template"]["group"]["group_id"],
        "normalized_config": normalized_config,
        "n_expanded_cases": len(expanded_cases),
        "default_out_dir": default_out_dir,
    }


def load_config_workflow_inputs(config_path: str | Path) -> dict[str, Any]:
    resolved_config_path = Path(config_path).expanduser().resolve()
    model_config = experiment_schema.load_model_config(resolved_config_path)
    template_paths = tuple(Path(path).expanduser().resolve() for path in model_config.template_paths)
    default_out_dir = default_config_run_output_dir(
        cable_root=_ROOT,
        config_name=model_config.config_name,
    ).resolve()
    return {
        "config_path": resolved_config_path,
        "config_name": model_config.config_name,
        "template_paths": template_paths,
        "template_names": tuple(path.stem for path in template_paths),
        "n_templates": len(template_paths),
        "default_out_dir": default_out_dir,
    }


def discover_batch_configs(config_dir: str | Path) -> list[dict[str, Any]]:
    resolved_config_dir = Path(config_dir).expanduser().resolve()
    if not resolved_config_dir.exists():
        raise FileNotFoundError(f"cable batch config directory does not exist: {resolved_config_dir!s}.")
    if not resolved_config_dir.is_dir():
        raise NotADirectoryError(f"cable batch config path must be a directory: {resolved_config_dir!s}.")

    config_paths = sorted(
        path
        for path in resolved_config_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    )
    if len(config_paths) == 0:
        raise ValueError(f"No cable config JSON files were found in {resolved_config_dir!s}.")

    records: list[dict[str, Any]] = []
    for config_path in config_paths:
        try:
            info = load_config_workflow_inputs(config_path)
        except Exception as exc:
            raise RuntimeError(f"cable batch discovery failed for {config_path!s}: {exc}") from exc
        records.append(
            {
                "config_path": info["config_path"],
                "config_name": info["config_name"],
                "run_id": info["config_name"],
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

    status = cable_run.run_sweep_config(
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
            "cable workflow completed without successful cases: "
            f"{aggregate.get('n_failed_cases', 'unknown')} failed."
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
    result = cable_run.run_config_file(
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
            "cable config workflow completed with failed templates: "
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


def run_notebook_batch(
    config_records: Sequence[Mapping[str, Any]],
    *,
    plot: bool | None = None,
    expand_only: bool = False,
    summary_dir: str | Path | None = None,
    batch_run_id: str | None = None,
) -> dict[str, Any]:
    if len(config_records) == 0:
        raise ValueError("run_notebook_batch requires at least one config record.")

    config_dirs = {
        Path(str(record["config_path"])).expanduser().resolve().parent
        for record in config_records
    }
    if len(config_dirs) != 1:
        raise ValueError("run_notebook_batch requires all config records to come from the same directory.")
    config_dir = next(iter(config_dirs))

    resolved_batch_run_id = batch_run_id or make_batch_run_id()
    resolved_summary_dir = default_batch_run_output_dir(
        batch_run_id=resolved_batch_run_id,
        summary_dir=summary_dir,
    )
    config_outputs_dir = resolved_summary_dir / "configs"
    config_outputs_dir.mkdir(parents=True, exist_ok=True)

    config_run_infos: list[dict[str, Any]] = []
    for record in config_records:
        config_name = str(record["config_name"])
        config_out_dir = config_outputs_dir / config_name
        try:
            config_run_infos.append(
                run_notebook_config_workflow(
                    config_path=record["config_path"],
                    out_dir=config_out_dir,
                    plot=plot,
                    expand_only=bool(expand_only),
                    raise_on_failure=False,
                )
            )
        except Exception as exc:
            config_run_infos.append(
                _build_batch_exception_run_info(
                    record=record,
                    out_dir=config_out_dir,
                    error_message=str(exc),
                )
            )

    artifacts = write_batch_summary_artifacts(
        config_dir=config_dir,
        summary_dir=resolved_summary_dir,
        batch_run_id=resolved_batch_run_id,
        config_records=config_records,
        config_run_infos=config_run_infos,
        plot_cases=bool(plot),
    )
    return {
        "batch_run_id": resolved_batch_run_id,
        "summary_dir": resolved_summary_dir,
        "manifest_path": artifacts["manifest_path"],
        "config_runs_path": artifacts["config_runs_path"],
        "batch_observable_summary_path": artifacts["batch_observable_summary_path"],
        "batch_observable_summary_json_path": artifacts["batch_observable_summary_json_path"],
        "batch_failures_path": artifacts["batch_failures_path"],
        "config_rows": artifacts["config_rows"],
        "observable_rows": artifacts["observable_rows"],
        "failure_rows": artifacts["failure_rows"],
        "config_records": list(config_records),
        "config_run_infos": config_run_infos,
        "expand_only": bool(expand_only),
        "plot": plot,
    }


def build_batch_summary_tables(batch_result: Mapping[str, Any]) -> dict[str, Any]:
    import pandas as pd

    config_rows_df = pd.DataFrame(batch_result.get("config_rows", []), columns=_BATCH_CONFIG_COLUMNS)
    observables_df = pd.DataFrame(batch_result.get("observable_rows", []), columns=_BATCH_OBSERVABLE_COLUMNS)
    failures_df = pd.DataFrame(batch_result.get("failure_rows", []), columns=_BATCH_FAILURE_COLUMNS)

    for frame, columns in (
        (config_rows_df, ("n_templates", "n_total_cases", "n_success_cases", "n_failed_cases")),
        (observables_df, ("n_cases", "mae_mean", "rmse_mean", "max_abs_max", "rel_mae_pct_mean")),
        (failures_df, ("n_failed_cases",)),
    ):
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return {
        "config_rows_df": config_rows_df,
        "observables_df": observables_df,
        "failures_df": failures_df,
    }


def load_run_artifacts(out_dir: str | Path) -> dict[str, Any]:
    import pandas as pd

    resolved_out_dir = Path(out_dir).expanduser().resolve()
    metrics_df = pd.read_csv(resolved_out_dir / "case_metrics.csv")
    for column in ("dt_ms", "cv_per_branch", "n_samples", "mae", "rmse", "max_abs", "rel_mae_pct"):
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
    if "morphology.path" in merged_df.columns:
        merged_df["morphology_name"] = merged_df["morphology.path"].map(
            lambda value: Path(str(value)).name if value == value else ""
        )
    else:
        merged_df["morphology_name"] = ""

    ok_df = merged_df[merged_df["status"] == "ok"].copy()
    failed_df = merged_df[merged_df["status"] != "ok"].copy()
    worst_cases_df = ok_df.sort_values(["max_abs", "rmse", "mae"], ascending=False).reset_index(drop=True)

    if len(ok_df) == 0:
        by_observable_df = metrics_df.iloc[0:0].copy()
        by_group_df = metrics_df.iloc[0:0].copy()
        by_morphology_df = metrics_df.iloc[0:0].copy()
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
        by_morphology_df = (
            ok_df.groupby("morphology_name", dropna=False)
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
        "by_morphology_df": by_morphology_df,
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

    ok_df = summary_tables["ok_df"]
    failed_df = summary_tables["failed_df"]
    by_morphology_df = summary_tables["by_morphology_df"]
    worst_cases_df = summary_tables["worst_cases_df"]
    aggregate = summary_tables["aggregate"]
    grouped_metric = {
        "mae": "mae_mean",
        "rmse": "rmse_mean",
        "max_abs": "max_abs_max",
        "rel_mae_pct": "rel_mae_pct_mean",
    }.get(metric, "max_abs_max")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    observable_names = list(aggregate.get("observables", {}).keys())
    observable_values = [
        float(aggregate["observables"][name].get(grouped_metric, 0.0))
        for name in observable_names
    ]
    axes[0, 0].bar(observable_names, observable_values, color="#bfd7ea", edgecolor="#2f3e46")
    axes[0, 0].set_title(f"Aggregate observable summary ({grouped_metric})")
    axes[0, 0].set_ylabel(grouped_metric)
    axes[0, 0].grid(axis="y", alpha=0.25)

    if len(by_morphology_df) > 0:
        top_morphology_df = by_morphology_df.head(min(10, len(by_morphology_df))).iloc[::-1]
        axes[0, 1].barh(
            top_morphology_df["morphology_name"],
            top_morphology_df[grouped_metric],
            color="#c7e9c0",
            edgecolor="#2f3e46",
        )
        axes[0, 1].set_title(f"Top morphologies by {metric}")
        axes[0, 1].set_xlabel(metric)
        axes[0, 1].grid(axis="x", alpha=0.25)
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
    elif len(ok_df) > 0:
        summary_lines.append("")
        summary_lines.append("successful cases by group:")
        group_counts = ok_df.groupby("group_id")["case_id"].nunique().to_dict()
        summary_lines.extend(f"  {group_id}: {count}" for group_id, count in group_counts.items())
    axes[1, 1].text(0.02, 0.98, "\n".join(summary_lines), ha="left", va="top", family="monospace")

    fig.tight_layout()
    return fig, axes


def build_case_metric_table(case_result: Mapping[str, Any]):
    import pandas as pd

    per_cv = pd.DataFrame(case_result["metrics"]["per_cv"])
    braincell_labels = case_result["alignment"]["braincell_labels"]
    neuron_labels = case_result["alignment"]["neuron_labels"]
    if len(per_cv) != len(braincell_labels) or len(per_cv) != len(neuron_labels):
        raise ValueError("Per-CV metrics and alignment labels must have the same length.")
    per_cv["braincell_label"] = [
        label.get("canonical_label", f"bc:{label.get('compartment_index', index)}")
        for index, label in enumerate(braincell_labels)
    ]
    per_cv["neuron_label"] = [
        label.get("canonical_label", f"nrn:{label.get('compartment_index', index)}")
        for index, label in enumerate(neuron_labels)
    ]
    per_cv["braincell_canonical_name"] = [
        label.get("canonical_name", label.get("branch_name", ""))
        for label in braincell_labels
    ]
    per_cv["neuron_canonical_name"] = [
        label.get("canonical_name", label.get("section_name", ""))
        for label in neuron_labels
    ]
    return per_cv.sort_values(["mae", "max_abs"], ascending=False).reset_index(drop=True)


def plot_case_overlay(
    case_result: Mapping[str, Any],
    *,
    compartment_indices: Sequence[int] | None = None,
    max_compartments: int = 3,
):
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(case_result["time_ms"], dtype=float)
    braincell_voltage = np.asarray(case_result["braincell"]["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(case_result["neuron"]["voltage_mV"], dtype=float)
    abs_error = np.abs(braincell_voltage - neuron_voltage)
    metric_table = build_case_metric_table(case_result)
    if compartment_indices is None:
        selected = metric_table["compartment_index"].head(max_compartments).astype(int).tolist()
    else:
        selected = [int(index) for index in compartment_indices]
    braincell_labels = case_result["alignment"]["braincell_labels"]
    neuron_labels = case_result["alignment"]["neuron_labels"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for index in selected:
        bc_label = braincell_labels[index].get("canonical_label", f"bc:{index}")
        nrn_label = neuron_labels[index].get("canonical_label", f"nrn:{index}")
        label = f"{bc_label} / {nrn_label}"
        axes[0].plot(time_ms, neuron_voltage[:, index], linewidth=1.5, alpha=0.85, label=f"NEURON {label}")
        axes[0].plot(
            time_ms,
            braincell_voltage[:, index],
            linewidth=1.5,
            linestyle="--",
            alpha=0.85,
            label=f"braincell {label}",
        )
        axes[1].plot(time_ms, abs_error[:, index], linewidth=1.5, alpha=0.85, label=label)
    axes[0].set_title("Voltage overlay for selected CV midpoints")
    axes[0].set_ylabel("Voltage (mV)")
    axes[1].set_title("Absolute error for selected CV midpoints")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("|delta V| (mV)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_case_error_summary(
    case_result: Mapping[str, Any],
    *,
    top_k: int = 10,
):
    import matplotlib.pyplot as plt
    import numpy as np

    time_ms = np.asarray(case_result["time_ms"], dtype=float)
    braincell_voltage = np.asarray(case_result["braincell"]["voltage_mV"], dtype=float)
    neuron_voltage = np.asarray(case_result["neuron"]["voltage_mV"], dtype=float)
    abs_error = np.abs(braincell_voltage - neuron_voltage)
    metric_table = build_case_metric_table(case_result).head(top_k).copy()
    metric_table = metric_table.sort_values(["mae", "max_abs"], ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(time_ms, abs_error.mean(axis=1), linewidth=1.6, label="mean |delta V|")
    axes[0].plot(time_ms, abs_error.max(axis=1), linewidth=1.6, label="max |delta V|")
    axes[0].set_title("Aggregate midpoint voltage error over time")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("|delta V| (mV)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].barh(metric_table["braincell_label"], metric_table["mae"], color="#bfd7ea", edgecolor="#2f3e46")
    axes[1].set_title(f"Top {len(metric_table)} CVs by MAE")
    axes[1].set_xlabel("MAE (mV)")
    axes[1].set_ylabel("CV midpoint")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    return fig, axes


def plot_observable_metric_boxplots(
    summary_tables: Mapping[str, Any],
    *,
    metrics: tuple[str, ...] = ("mae", "rmse", "max_abs", "rel_mae_pct"),
):
    import matplotlib.pyplot as plt

    ok_df = summary_tables["ok_df"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.0))
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        grouped = {
            observable: ok_df.loc[ok_df["observable"] == observable, metric].dropna().to_numpy()
            for observable in dict.fromkeys(ok_df["observable"].tolist())
        }
        grouped = {label: values for label, values in grouped.items() if values.size > 0}
        if len(grouped) == 0:
            axis.text(0.5, 0.5, "No successful cases.", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = list(grouped.keys())
        data = [grouped[label] for label in labels]
        axis.boxplot(data, tick_labels=labels, showfliers=True)
        axis.set_title(f"{metric} by observable")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig, axes


def _expected_suffixes_for_kind(morphology_kind: str) -> set[str]:
    if morphology_kind == "swc":
        return {".swc"}
    if morphology_kind == "asc":
        return {".asc"}
    raise ValueError(f"Unsupported morphology_kind {morphology_kind!r}.")


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


def _build_batch_exception_run_info(
    *,
    record: Mapping[str, Any],
    out_dir: Path,
    error_message: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    observable_summary_json_path = out_dir / "observable_summary.json"
    _write_json(
        observable_summary_json_path,
        {
            "all_templates": {"observables": {}},
        },
    )
    return {
        "status": 1,
        "config_path": str(Path(str(record["config_path"])).expanduser().resolve()),
        "config_name": str(record["config_name"]),
        "out_dir": out_dir,
        "observable_summary_json_path": observable_summary_json_path,
        "failures": [
            {
                "template_name": "",
                "template_path": "",
                "out_dir": str(out_dir),
                "n_failed_cases": "",
                "error_message": error_message,
            }
        ],
        "status_counts": {"ok": 0, "partial": 0, "failed": 1},
        "n_templates": int(record.get("n_templates", 0)),
        "n_total_cases": 0,
        "n_success_cases": 0,
        "n_failed_cases": 0,
    }


__all__ = [
    "build_batch_config_rows",
    "build_batch_failure_rows",
    "build_batch_observable_rows",
    "build_batch_summary_tables",
    "build_case_metric_table",
    "build_summary_tables",
    "default_batch_run_output_dir",
    "discover_batch_configs",
    "find_repo_root",
    "load_case_result",
    "load_config_workflow_inputs",
    "load_run_artifacts",
    "load_workflow_inputs",
    "make_batch_run_id",
    "plot_case_error_summary",
    "plot_case_overlay",
    "plot_observable_metric_boxplots",
    "plot_sweep_summary",
    "resolve_selected_files",
    "run_notebook_batch",
    "run_notebook_config_workflow",
    "run_notebook_workflow",
    "write_batch_summary_artifacts",
]
