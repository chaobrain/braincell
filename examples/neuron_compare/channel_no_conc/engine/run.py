#!/usr/bin/env python3
"""Dispatcher for channel_no_conc compare inputs."""



import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

try:
    from .compare import compare_case
    from .experiment_schema import ChannelNoConcCase, config_to_payload, expand_cases, load_model_config, load_sweep_config
    from .outputs import (
        aggregate_case_metrics,
        build_config_observable_summary_payload,
        default_config_run_output_dir,
        default_sweep_output_dir,
        default_template_run_output_dir,
        iter_metric_rows,
        save_all_templates_observable_summary_plot,
        save_boxplot_by_observable_family,
        save_boxplot_by_template,
        save_observable_metric_boxplots,
        save_case_plot,
        save_sweep_summary_plot,
        write_case_metrics_csv,
        write_json,
    )
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from compare import compare_case  # type: ignore
    from experiment_schema import ChannelNoConcCase, config_to_payload, expand_cases, load_model_config, load_sweep_config  # type: ignore
    from outputs import (  # type: ignore
        aggregate_case_metrics,
        build_config_observable_summary_payload,
        default_config_run_output_dir,
        default_sweep_output_dir,
        default_template_run_output_dir,
        iter_metric_rows,
        save_all_templates_observable_summary_plot,
        save_boxplot_by_observable_family,
        save_boxplot_by_template,
        save_observable_metric_boxplots,
        save_case_plot,
        save_sweep_summary_plot,
        write_case_metrics_csv,
        write_json,
    )


CONFIG_TEMPLATE_RUN_COLUMNS = (
    "run_id",
    "config_name",
    "template_name",
    "config_path",
    "template_path",
    "mod_dir",
    "out_dir",
    "batch_status",
    "n_total_cases",
    "n_success_cases",
    "n_failed_cases",
    "worst_observable",
    "worst_max_abs_max",
    "error_message",
)
CONFIG_OBSERVABLE_COLUMNS = (
    "run_id",
    "config_name",
    "template_name",
    "config_path",
    "template_path",
    "mod_dir",
    "batch_status",
    "observable",
    "n_cases",
    "mae_mean",
    "rmse_mean",
    "max_abs_max",
    "rel_mae_pct_mean",
)
CONFIG_FAILURE_COLUMNS = (
    "run_id",
    "config_name",
    "template_name",
    "config_path",
    "template_path",
    "out_dir",
    "batch_status",
    "n_failed_cases",
    "error_message",
)
SUMMARY_METRICS = ("mae", "rmse", "max_abs", "rel_mae_pct")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch channel_no_conc compare inputs.")
    parser.add_argument("config_path", help="Path to a channel_no_conc model config JSON.")
    parser.add_argument("template_path", nargs="?", help="Optional path to one channel_no_conc scan template JSON.")
    parser.add_argument("--out-dir", help="Optional output directory for sweep runs.")
    parser.add_argument("--expand-only", action="store_true", help="Only expand a sweep config without running compare.")
    parser.set_defaults(plot=None)
    parser.add_argument("--plot", dest="plot", action="store_true", help="Generate per-case comparison plots for sweeps.")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation for sweeps.")
    return parser


def run_sweep_config(
    config,
    *,
    out_dir: str | Path | None = None,
    expand_only: bool = False,
    plot: bool | None = None,
) -> int:
    expanded_cases = expand_cases(config)
    root = Path(__file__).resolve().parents[1]
    resolved_out_dir = (
        Path(out_dir)
        if out_dir is not None
        else default_sweep_output_dir(
            channel_no_conc_root=root,
            config_id=config.config_id,
        )
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    legacy_notebook_plots_dir = resolved_out_dir / "notebook_plots"
    if legacy_notebook_plots_dir.exists():
        shutil.rmtree(legacy_notebook_plots_dir)

    write_json(resolved_out_dir / "normalized_config.json", config_to_payload(config))
    write_json(resolved_out_dir / "expanded_cases.json", expanded_cases)
    if expand_only:
        return 0

    do_plot = config.outputs.plot if plot is None else bool(plot)
    case_results_dir = resolved_out_dir / "case_results"
    if case_results_dir.exists():
        shutil.rmtree(case_results_dir)
    case_results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = resolved_out_dir / "plots"
    if do_plot:
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
    elif plots_dir.exists():
        shutil.rmtree(plots_dir)

    csv_rows: list[dict[str, object]] = []
    success_results: list[dict[str, object]] = []
    failed_cases: list[dict[str, object]] = []

    for case_payload in expanded_cases:
        group_id = str(case_payload.get("group_id", ""))
        case_id = str(case_payload["case_id"])
        try:
            case = ChannelNoConcCase.from_dict(case_payload)
            result = compare_case(case)
            success_results.append(result)
            write_json(case_results_dir / f"{case_id}.json", result)
            for observable, metric in iter_metric_rows(result["metrics"]):
                csv_rows.append(
                    {
                        "case_id": case_id,
                        "group_id": group_id,
                        "status": "ok",
                        "stimulus_kind": str(case.stimulus.kind),
                        "temperature_celsius": case.simulation.temperature_celsius,
                        "v_init_mV": case.simulation.v_init_mV,
                        "n_samples": len(result["time_ms"]),
                        "observable": observable,
                        "mae": metric["mae"],
                        "rmse": metric["rmse"],
                        "max_abs": metric["max_abs"],
                        "rel_mae_pct": metric["rel_mae_pct"],
                        "error_message": "",
                    }
                )
            if do_plot:
                save_case_plot(plots_dir / f"{case_id}.png", result)
        except Exception as exc:
            failed_cases.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "error_message": str(exc),
                    "case": case_payload,
                }
            )
            csv_rows.append(
                {
                    "case_id": case_id,
                    "group_id": group_id,
                    "status": "failed",
                    "stimulus_kind": case_payload.get("stimulus", {}).get("kind", ""),
                    "temperature_celsius": case_payload.get("simulation", {}).get("temperature_celsius", ""),
                    "v_init_mV": case_payload.get("simulation", {}).get("v_init_mV", ""),
                    "n_samples": "",
                    "observable": "",
                    "mae": "",
                    "rmse": "",
                    "max_abs": "",
                    "rel_mae_pct": "",
                    "error_message": str(exc),
                }
            )

    aggregate = aggregate_case_metrics(
        config,
        success_results=success_results,
        failed_cases=failed_cases,
        total_cases=len(expanded_cases),
    )
    write_case_metrics_csv(csv_rows, resolved_out_dir / "case_metrics.csv")
    write_json(
        resolved_out_dir / "aggregate.json",
        aggregate,
    )
    if do_plot and len(success_results) > 0:
        for metric in SUMMARY_METRICS:
            save_sweep_summary_plot(
                plots_dir / f"summary_{metric}.png",
                aggregate=aggregate,
                metric_rows=csv_rows,
                metric=metric,
            )
        save_observable_metric_boxplots(
            plots_dir / "observable_metric_boxplots.png",
            metric_rows=csv_rows,
            metrics=SUMMARY_METRICS,
        )
    return 0 if len(success_results) > 0 else 1


def run_config_file(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    expand_only: bool = False,
    plot: bool | None = None,
) -> dict[str, Any]:
    model_config = load_model_config(config_path)
    root = Path(__file__).resolve().parents[1]
    resolved_out_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else default_config_run_output_dir(
            channel_no_conc_root=root,
            config_name=model_config.config_name,
        ).resolve()
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    template_runs: list[dict[str, Any]] = []
    observables: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []

    for template_path in model_config.template_paths:
        resolved_template_path = Path(template_path).expanduser().resolve()
        template_out_dir = default_template_run_output_dir(
            config_out_dir=resolved_out_dir,
            template_name=resolved_template_path.stem,
        ).resolve()
        legacy_notebook_plots_dir = template_out_dir / "notebook_plots"
        if legacy_notebook_plots_dir.exists():
            shutil.rmtree(legacy_notebook_plots_dir)
        try:
            config = load_sweep_config(model_config.config_path, resolved_template_path)
            status = run_sweep_config(
                config,
                out_dir=template_out_dir,
                expand_only=bool(expand_only),
                plot=plot,
            )
            run_row, observable_rows, failure_row = _build_config_template_success_rows(
                model_config=model_config,
                config=config,
                template_path=resolved_template_path,
                out_dir=template_out_dir,
                status=int(status),
                expand_only=bool(expand_only),
            )
        except Exception as exc:
            run_row = _build_config_template_exception_row(
                model_config=model_config,
                template_path=resolved_template_path,
                out_dir=template_out_dir,
                error_message=str(exc),
            )
            observable_rows = []
            failure_row = _build_config_failure_row(run_row)

        template_runs.append(run_row)
        observables.extend(observable_rows)
        if failure_row is not None:
            failures.append(failure_row)
        if not expand_only:
            metric_rows_path = template_out_dir / "case_metrics.csv"
            if metric_rows_path.exists():
                all_metric_rows.extend(
                    _load_metric_rows(
                        metric_rows_path,
                        template_name=resolved_template_path.stem,
                    )
                )

    manifest = _build_config_manifest(
        model_config=model_config,
        out_dir=resolved_out_dir,
        template_runs=template_runs,
        expand_only=bool(expand_only),
        plot=plot,
    )
    manifest_path = resolved_out_dir / "config_manifest.json"
    config_runs_path = resolved_out_dir / "config_runs.csv"
    observable_summary_path = resolved_out_dir / "observable_summary.csv"
    observable_summary_json_path = resolved_out_dir / "observable_summary.json"
    failures_path = resolved_out_dir / "failures.csv"
    config_summary_plot_paths = [
        resolved_out_dir / "all_templates_observable_summary.png",
        resolved_out_dir / "boxplot_by_template.png",
        resolved_out_dir / "boxplot_by_observable_family.png",
    ]
    for plot_path in config_summary_plot_paths:
        if plot_path.exists():
            plot_path.unlink()

    write_json(manifest_path, manifest)
    _write_csv_rows(config_runs_path, CONFIG_TEMPLATE_RUN_COLUMNS, template_runs)
    _write_csv_rows(observable_summary_path, CONFIG_OBSERVABLE_COLUMNS, observables)
    _write_csv_rows(failures_path, CONFIG_FAILURE_COLUMNS, failures)
    observable_summary_payload = build_config_observable_summary_payload(
        config_name=model_config.config_name,
        config_path=model_config.config_path,
        template_runs=template_runs,
        observable_rows=observables,
        status_counts=manifest["status_counts"],
    )
    write_json(observable_summary_json_path, observable_summary_payload)

    if bool(plot) and len(all_metric_rows) > 0:
        save_all_templates_observable_summary_plot(
            resolved_out_dir / "all_templates_observable_summary.png",
            summary_payload=observable_summary_payload,
            metrics=SUMMARY_METRICS,
        )
        save_boxplot_by_template(
            resolved_out_dir / "boxplot_by_template.png",
            metric_rows=all_metric_rows,
            metrics=SUMMARY_METRICS,
        )
        save_boxplot_by_observable_family(
            resolved_out_dir / "boxplot_by_observable_family.png",
            metric_rows=all_metric_rows,
            metrics=SUMMARY_METRICS,
        )

    status = 0 if any(row["batch_status"] in {"ok", "partial"} for row in template_runs) else 1
    return {
        "status": status,
        "config_path": str(model_config.config_path),
        "config_name": model_config.config_name,
        "mod_dir": model_config.mod_dir,
        "out_dir": resolved_out_dir,
        "manifest_path": manifest_path,
        "config_runs_path": config_runs_path,
        "observable_summary_path": observable_summary_path,
        "observable_summary_json_path": observable_summary_json_path,
        "failures_path": failures_path,
        "template_runs": template_runs,
        "observables": observables,
        "failures": failures,
        "status_counts": manifest["status_counts"],
        "n_templates": manifest["n_templates"],
    }


def _build_config_template_success_rows(
    *,
    model_config,
    config,
    template_path: Path,
    out_dir: Path,
    status: int,
    expand_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    base_row = {
        "run_id": config.config_id,
        "config_name": config.config_name,
        "template_name": config.template_name,
        "config_path": str(model_config.config_path),
        "template_path": str(template_path),
        "mod_dir": str(Path(model_config.mod_dir).expanduser().resolve()),
        "out_dir": str(out_dir),
        "error_message": "",
    }

    expanded_cases = json.loads((out_dir / "expanded_cases.json").read_text())
    if expand_only:
        run_row = {
            **base_row,
            "batch_status": "ok",
            "n_total_cases": len(expanded_cases),
            "n_success_cases": "",
            "n_failed_cases": "",
            "worst_observable": "",
            "worst_max_abs_max": "",
        }
        return run_row, [], None

    aggregate = json.loads((out_dir / "aggregate.json").read_text())
    worst_observable, worst_max_abs_max = _find_worst_observable(aggregate)
    n_failed_cases = int(aggregate.get("n_failed_cases", 0))
    if status != 0:
        batch_status = "failed"
        error_message = f"Sweep exited with status {status}."
    elif n_failed_cases > 0:
        batch_status = "partial"
        error_message = f"{n_failed_cases} case(s) failed inside the sweep."
    else:
        batch_status = "ok"
        error_message = ""

    run_row = {
        **base_row,
        "batch_status": batch_status,
        "n_total_cases": int(aggregate.get("n_total_cases", 0)),
        "n_success_cases": int(aggregate.get("n_success_cases", 0)),
        "n_failed_cases": n_failed_cases,
        "worst_observable": worst_observable,
        "worst_max_abs_max": worst_max_abs_max,
        "error_message": error_message,
    }
    observable_rows = [
        {
            "run_id": run_row["run_id"],
            "config_name": run_row["config_name"],
            "template_name": run_row["template_name"],
            "config_path": run_row["config_path"],
            "template_path": run_row["template_path"],
            "mod_dir": run_row["mod_dir"],
            "batch_status": batch_status,
            "observable": observable,
            "n_cases": int(metrics.get("n_cases", 0)),
            "mae_mean": float(metrics.get("mae_mean", 0.0)),
            "rmse_mean": float(metrics.get("rmse_mean", 0.0)),
            "max_abs_max": float(metrics.get("max_abs_max", 0.0)),
            "rel_mae_pct_mean": float(metrics.get("rel_mae_pct_mean", 0.0)),
        }
        for observable, metrics in aggregate.get("observables", {}).items()
    ]
    failure_row = _build_config_failure_row(run_row) if batch_status != "ok" else None
    return run_row, observable_rows, failure_row


def _build_config_template_exception_row(
    *,
    model_config,
    template_path: Path,
    out_dir: Path,
    error_message: str,
) -> dict[str, Any]:
    template_name = template_path.stem
    return {
        "run_id": f"{model_config.config_name}__{template_name}",
        "config_name": model_config.config_name,
        "template_name": template_name,
        "config_path": str(model_config.config_path),
        "template_path": str(template_path),
        "mod_dir": str(Path(model_config.mod_dir).expanduser().resolve()),
        "out_dir": str(out_dir),
        "batch_status": "failed",
        "n_total_cases": "",
        "n_success_cases": "",
        "n_failed_cases": "",
        "worst_observable": "",
        "worst_max_abs_max": "",
        "error_message": error_message,
    }


def _build_config_failure_row(run_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_row["run_id"],
        "config_name": run_row["config_name"],
        "template_name": run_row["template_name"],
        "config_path": run_row["config_path"],
        "template_path": run_row["template_path"],
        "out_dir": run_row["out_dir"],
        "batch_status": run_row["batch_status"],
        "n_failed_cases": run_row["n_failed_cases"],
        "error_message": run_row["error_message"],
    }


def _find_worst_observable(aggregate: Mapping[str, Any]) -> tuple[str, float | str]:
    observables = aggregate.get("observables", {})
    if not isinstance(observables, Mapping) or len(observables) == 0:
        return "", ""
    worst_name, worst_metrics = max(
        observables.items(),
        key=lambda item: float(item[1].get("max_abs_max", float("-inf"))),
    )
    return str(worst_name), float(worst_metrics.get("max_abs_max", 0.0))


def _build_config_manifest(
    *,
    model_config,
    out_dir: Path,
    template_runs: Sequence[Mapping[str, Any]],
    expand_only: bool,
    plot: bool | None,
) -> dict[str, Any]:
    status_counts = {
        status: sum(1 for row in template_runs if row["batch_status"] == status)
        for status in ("ok", "partial", "failed")
    }
    return {
        "config_name": model_config.config_name,
        "config_path": str(model_config.config_path),
        "mod_dir": str(Path(model_config.mod_dir).expanduser().resolve()),
        "out_dir": str(out_dir),
        "n_templates": len(template_runs),
        "expand_only": bool(expand_only),
        "plot": plot,
        "status_counts": status_counts,
        "template_runs": [dict(row) for row in template_runs],
        "summary_files": {
            "config_runs_csv": str(out_dir / "config_runs.csv"),
            "observable_summary_csv": str(out_dir / "observable_summary.csv"),
            "observable_summary_json": str(out_dir / "observable_summary.json"),
            "failures_csv": str(out_dir / "failures.csv"),
        },
    }


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_metric_rows(path: Path, *, template_name: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["template_name"] = template_name
    return rows


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.template_path is None:
        result = run_config_file(
            args.config_path,
            out_dir=args.out_dir,
            expand_only=bool(args.expand_only),
            plot=args.plot,
        )
        return int(result["status"])

    config = load_sweep_config(args.config_path, args.template_path)
    return run_sweep_config(
        config,
        out_dir=args.out_dir,
        expand_only=bool(args.expand_only),
        plot=args.plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
