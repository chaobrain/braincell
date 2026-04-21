#!/usr/bin/env python3
"""Dispatcher for cable sweep inputs."""



import argparse
from pathlib import Path

try:
    from .compare import compare_case
    from .case_schema import MultiCompartmentCableCase
    from .experiment_schema import config_to_payload, expand_cases, load_sweep_config
    from .outputs import (
        aggregate_case_metrics,
        default_sweep_output_dir,
        iter_metric_rows,
        save_case_plot,
        write_case_metrics_csv,
        write_json,
    )
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from compare import compare_case  # type: ignore
    from case_schema import MultiCompartmentCableCase  # type: ignore
    from experiment_schema import config_to_payload, expand_cases, load_sweep_config  # type: ignore
    from outputs import (  # type: ignore
        aggregate_case_metrics,
        default_sweep_output_dir,
        iter_metric_rows,
        save_case_plot,
        write_case_metrics_csv,
        write_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch cable compare inputs.")
    parser.add_argument("config_path", help="Path to a cable model config JSON.")
    parser.add_argument("template_path", help="Path to a cable scan template JSON.")
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
            cable_root=root,
            config_id=config.config_id,
        )
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    write_json(resolved_out_dir / "normalized_config.json", config_to_payload(config))
    write_json(resolved_out_dir / "expanded_cases.json", expanded_cases)
    if expand_only:
        return 0

    do_plot = config.outputs.plot if plot is None else bool(plot)
    case_results_dir = resolved_out_dir / "case_results"
    case_results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = resolved_out_dir / "plots"
    if do_plot:
        plots_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, object]] = []
    success_results: list[dict[str, object]] = []
    failed_cases: list[dict[str, object]] = []

    for case_payload in expanded_cases:
        group_id = str(case_payload.get("group_id", ""))
        case_id = str(case_payload["case_id"])
        try:
            case = MultiCompartmentCableCase.from_dict(case_payload)
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
                        "morphology_kind": case.morphology.kind,
                        "morphology_path": case.morphology.path,
                        "dt_ms": case.simulation.dt_ms,
                        "cv_per_branch": case.cv_policy.cv_per_branch,
                        "observable": observable,
                        "n_samples": len(result["time_ms"]),
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
                    "morphology_kind": case_payload.get("morphology", {}).get("kind", ""),
                    "morphology_path": case_payload.get("morphology", {}).get("path", ""),
                    "dt_ms": case_payload.get("simulation", {}).get("dt_ms", ""),
                    "cv_per_branch": case_payload.get("cv_policy", {}).get("cv_per_branch", ""),
                    "observable": "",
                    "n_samples": "",
                    "mae": "",
                    "rmse": "",
                    "max_abs": "",
                    "rel_mae_pct": "",
                    "error_message": str(exc),
                }
            )

    write_case_metrics_csv(csv_rows, resolved_out_dir / "case_metrics.csv")
    write_json(
        resolved_out_dir / "aggregate.json",
        aggregate_case_metrics(
            config,
            success_results=success_results,
            failed_cases=failed_cases,
            total_cases=len(expanded_cases),
        ),
    )
    return 0 if len(success_results) > 0 else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_sweep_config(args.config_path, args.template_path)
    return run_sweep_config(
        config,
        out_dir=args.out_dir,
        expand_only=bool(args.expand_only),
        plot=args.plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
