"""Output helpers shared by channel_no_conc sweep dispatch."""



import csv
import json
from pathlib import Path
from typing import Any

METRIC_AGGREGATE_FIELDS = {
    "mae": "mae_mean",
    "rmse": "rmse_mean",
    "max_abs": "max_abs_max",
    "rel_mae_pct": "rel_mae_pct_mean",
}
SUMMARY_METRICS = ("mae", "rmse", "max_abs", "rel_mae_pct")


def write_json(out_path: str | Path, payload: Any) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def default_sweep_output_dir(
    *,
    channel_no_conc_root: str | Path,
    config_id: str,
) -> Path:
    return Path(channel_no_conc_root) / "results" / "sweeps" / config_id


def default_config_run_output_dir(
    *,
    channel_no_conc_root: str | Path,
    config_name: str,
) -> Path:
    return Path(channel_no_conc_root) / "results" / "config_runs" / config_name


def default_template_run_output_dir(
    *,
    config_out_dir: str | Path,
    template_name: str,
) -> Path:
    return Path(config_out_dir) / "templates" / template_name


def iter_metric_rows(metrics: dict[str, Any]) -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = [("voltage", metrics["voltage"])]
    rows.extend((f"current.{name}", record) for name, record in metrics.get("current", {}).items())
    rows.extend((f"gates.{name}", record) for name, record in metrics.get("gates", {}).items())
    return rows


def write_case_metrics_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    fieldnames = [
        "case_id",
        "group_id",
        "status",
        "stimulus_kind",
        "temperature_celsius",
        "v_init_mV",
        "observable",
        "n_samples",
        "mae",
        "rmse",
        "max_abs",
        "rel_mae_pct",
        "error_message",
    ]
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_case_metrics(
    config,
    *,
    success_results: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    total_cases: int,
) -> dict[str, Any]:
    observable_rows: dict[str, list[dict[str, float]]] = {}
    for result in success_results:
        for observable, metric in iter_metric_rows(result["metrics"]):
            observable_rows.setdefault(observable, []).append(metric)

    observables = {
        observable: {
            "n_cases": len(rows),
            "mae_mean": float(sum(row["mae"] for row in rows) / len(rows)),
            "rmse_mean": float(sum(row["rmse"] for row in rows) / len(rows)),
            "max_abs_max": float(max(row["max_abs"] for row in rows)),
            "rel_mae_pct_mean": float(sum(row["rel_mae_pct"] for row in rows) / len(rows)),
        }
        for observable, rows in observable_rows.items()
    }

    return {
        "config_id": config.config_id,
        "n_total_cases": total_cases,
        "n_success_cases": len(success_results),
        "n_failed_cases": len(failed_cases),
        "observables": observables,
        "failed_cases": failed_cases,
    }


def save_case_plot(out_path: str | Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    gate_alignments = list(result.get("alignment", {}).get("gates", []))
    if len(gate_alignments) == 0:
        gate_alignments = [
            {
                "canonical_name": gate_name,
                "braincell_gate": gate_name,
                "neuron_gate": gate_name,
            }
            for gate_name in sorted(result["braincell"]["gates"])
        ]

    observables = [
        ("voltage_mV", "Voltage (mV)"),
        ("current.ix", "Current ix"),
        *[
            (
                f"gates.{gate_alignment['canonical_name']}",
                f"Gate {gate_alignment['canonical_name']}",
            )
            for gate_alignment in gate_alignments
        ],
    ]
    fig, axes = plt.subplots(len(observables), 1, figsize=(10, 3.0 * len(observables)), sharex=True)
    if len(observables) == 1:
        axes = [axes]
    for axis, (observable, label) in zip(axes, observables):
        if observable == "voltage_mV":
            plot_time_ms = result["time_ms"]
            braincell_trace = result["braincell"]["voltage_mV"]
            neuron_trace = result["neuron"]["voltage_mV"]
        elif observable == "current.ix":
            plot_time_ms, braincell_trace, neuron_trace = _aligned_current_view(result)
        else:
            plot_time_ms = result["time_ms"]
            canonical_name = observable.split(".", 1)[1]
            gate_alignment = next(
                item for item in gate_alignments
                if str(item["canonical_name"]) == canonical_name
            )
            braincell_trace = result["braincell"]["gates"][str(gate_alignment["braincell_gate"])]
            neuron_trace = result["neuron"]["gates"][str(gate_alignment["neuron_gate"])]
        axis.plot(plot_time_ms, neuron_trace, label="NEURON", linewidth=1.5)
        axis.plot(plot_time_ms, braincell_trace, label="braincell", linewidth=1.2, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (ms)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_sweep_summary_plot(
    out_path: str | Path,
    *,
    aggregate: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    metric: str = "max_abs",
    top_k: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    ok_rows = [
        row for row in metric_rows
        if str(row.get("status", "")) == "ok" and str(row.get("observable", "")) != ""
    ]
    failed_rows = [
        row for row in metric_rows
        if str(row.get("status", "")) != "ok"
    ]
    aggregate_field = METRIC_AGGREGATE_FIELDS.get(metric, "max_abs_max")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    observables = aggregate.get("observables", {})
    observable_names = list(observables.keys())
    observable_values = [
        float(observables[name].get(aggregate_field, 0.0))
        for name in observable_names
    ]
    axes[0, 0].bar(observable_names, observable_values, color="#bfd7ea", edgecolor="#2f3e46")
    axes[0, 0].set_title(f"Aggregate observable summary ({aggregate_field})")
    axes[0, 0].set_ylabel(aggregate_field)
    axes[0, 0].grid(axis="y", alpha=0.25)

    by_observable: dict[str, list[float]] = {}
    for row in ok_rows:
        observable = str(row["observable"])
        value_raw = row.get(metric)
        if value_raw in ("", None):
            continue
        by_observable.setdefault(observable, []).append(float(value_raw))
    if len(by_observable) > 0:
        labels = list(by_observable.keys())
        data = [by_observable[label] for label in labels]
        axes[0, 1].boxplot(data, tick_labels=labels, orientation="vertical")
        axes[0, 1].set_title(f"Per-case distribution of {metric}")
        axes[0, 1].set_ylabel(metric)
        axes[0, 1].grid(axis="y", alpha=0.25)
    else:
        axes[0, 1].text(0.5, 0.5, "No successful cases.", ha="center", va="center")
        axes[0, 1].set_axis_off()

    ranked_rows = sorted(
        ok_rows,
        key=lambda row: (
            float(row.get(metric, 0.0) or 0.0),
            float(row.get("rmse", 0.0) or 0.0),
            float(row.get("mae", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if len(ranked_rows) > 0:
        top_rows = ranked_rows[:top_k][::-1]
        labels = [f"{row['case_id']} / {row['observable']}" for row in top_rows]
        values = [float(row.get(metric, 0.0) or 0.0) for row in top_rows]
        axes[1, 0].barh(labels, values, color="#84a98c", edgecolor="#2f3e46")
        axes[1, 0].set_title(f"Top {len(top_rows)} worst case-observable rows by {metric}")
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
    if len(failed_rows) > 0:
        summary_lines.append("")
        summary_lines.append("failed rows present in metrics csv")
    axes[1, 1].text(0.02, 0.98, "\n".join(summary_lines), ha="left", va="top", family="monospace")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_observable_metric_boxplots(
    out_path: str | Path,
    *,
    metric_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> None:
    import matplotlib.pyplot as plt

    ok_rows = [
        row for row in metric_rows
        if str(row.get("status", "")) == "ok" and str(row.get("observable", "")) != ""
    ]
    families = (
        ("voltage", "Voltage"),
        ("current.", "Current"),
        ("gates.", "Gates"),
    )

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.0))
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        plot_data: list[list[float]] = []
        labels: list[str] = []
        for prefix, label in families:
            values = [
                float(row[metric])
                for row in ok_rows
                if row.get(metric) not in ("", None)
                and (
                    (prefix == "voltage" and str(row["observable"]) == "voltage")
                    or (prefix != "voltage" and str(row["observable"]).startswith(prefix))
                )
            ]
            if len(values) == 0:
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

        positive_values = [value for values in plot_data for value in values if value > 0]
        if positive_values:
            dynamic_range = max(positive_values) / min(positive_values)
            if dynamic_range >= 100.0:
                axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_config_observable_summary_payload(
    *,
    config_name: str,
    config_path: str | Path,
    template_runs: list[dict[str, Any]],
    observable_rows: list[dict[str, Any]],
    status_counts: dict[str, int],
) -> dict[str, Any]:
    per_template: dict[str, dict[str, Any]] = {}
    for template_run in template_runs:
        template_name = str(template_run["template_name"])
        per_template[template_name] = {
            "batch_status": str(template_run["batch_status"]),
            "n_total_cases": _as_optional_int(template_run.get("n_total_cases")),
            "n_success_cases": _as_optional_int(template_run.get("n_success_cases")),
            "n_failed_cases": _as_optional_int(template_run.get("n_failed_cases")),
            "observables": {},
        }

    grouped_all: dict[str, list[dict[str, Any]]] = {}
    for row in observable_rows:
        observable = str(row["observable"])
        grouped_all.setdefault(observable, []).append(row)
        template_name = str(row["template_name"])
        per_template.setdefault(
            template_name,
            {
                "batch_status": "",
                "n_total_cases": None,
                "n_success_cases": None,
                "n_failed_cases": None,
                "observables": {},
            },
        )["observables"][observable] = _observable_record_from_row(row)

    return {
        "config_name": config_name,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "n_templates": len(template_runs),
        "n_total_cases": sum(
            _as_optional_int(row.get("n_total_cases")) or 0
            for row in template_runs
        ),
        "n_success_cases": sum(
            _as_optional_int(row.get("n_success_cases")) or 0
            for row in template_runs
        ),
        "n_failed_cases": sum(
            _as_optional_int(row.get("n_failed_cases")) or 0
            for row in template_runs
        ),
        "status_counts": dict(status_counts),
        "all_templates": {
            "observables": {
                observable: _aggregate_observable_group(rows)
                for observable, rows in sorted(grouped_all.items())
            }
        },
        "templates": {
            template_name: payload
            for template_name, payload in sorted(per_template.items())
        },
    }


def save_all_templates_observable_summary_plot(
    out_path: str | Path,
    *,
    summary_payload: dict[str, Any],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> None:
    import matplotlib.pyplot as plt

    observables = dict(summary_payload.get("all_templates", {}).get("observables", {}))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    for axis, metric in zip(axes.flat, metrics):
        aggregate_field = METRIC_AGGREGATE_FIELDS[metric]
        observable_names = list(observables.keys())
        observable_values = [
            float(observables[name].get(aggregate_field, 0.0))
            for name in observable_names
        ]
        if len(observable_names) == 0:
            axis.text(0.5, 0.5, "No successful observables.", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.bar(observable_names, observable_values, color="#bfd7ea", edgecolor="#2f3e46")
        axis.set_title(f"All templates observable summary ({aggregate_field})")
        axis.set_ylabel(aggregate_field)
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_boxplot_by_template(
    out_path: str | Path,
    *,
    metric_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> None:
    import matplotlib.pyplot as plt

    successful_rows = [
        row for row in metric_rows
        if str(row.get("status", "")) == "ok" and str(row.get("observable", "")) != ""
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    for axis, metric in zip(axes.flat, metrics):
        grouped: dict[str, list[float]] = {}
        for row in successful_rows:
            value = row.get(metric)
            if value in ("", None):
                continue
            grouped.setdefault(str(row["template_name"]), []).append(float(value))
        if len(grouped) == 0:
            axis.text(0.5, 0.5, "No successful cases.", ha="center", va="center")
            axis.set_axis_off()
            continue
        labels = list(grouped.keys())
        data = [grouped[label] for label in labels]
        axis.boxplot(data, tick_labels=labels, showfliers=True)
        axis.set_title(f"{metric} by template")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
        positive_values = [value for values in data for value in values if value > 0]
        if positive_values:
            dynamic_range = max(positive_values) / min(positive_values)
            if dynamic_range >= 100.0:
                axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_boxplot_by_observable_family(
    out_path: str | Path,
    *,
    metric_rows: list[dict[str, Any]],
    metrics: tuple[str, ...] = SUMMARY_METRICS,
) -> None:
    import matplotlib.pyplot as plt

    successful_rows = [
        row for row in metric_rows
        if str(row.get("status", "")) == "ok" and str(row.get("observable", "")) != ""
    ]
    families = (
        ("voltage", "Voltage"),
        ("current.", "Current"),
        ("gates.", "Gates"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.reshape(2, 2)

    for axis, metric in zip(axes.flat, metrics):
        plot_data: list[list[float]] = []
        labels: list[str] = []
        for prefix, label in families:
            values = [
                float(row[metric])
                for row in successful_rows
                if row.get(metric) not in ("", None)
                and (
                    (prefix == "voltage" and str(row["observable"]) == "voltage")
                    or (prefix != "voltage" and str(row["observable"]).startswith(prefix))
                )
            ]
            if len(values) == 0:
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
        positive_values = [value for values in plot_data for value in values if value > 0]
        if positive_values:
            dynamic_range = max(positive_values) / min(positive_values)
            if dynamic_range >= 100.0:
                axis.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _observable_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_cases": int(row.get("n_cases", 0)),
        "mae_mean": float(row.get("mae_mean", 0.0)),
        "rmse_mean": float(row.get("rmse_mean", 0.0)),
        "max_abs_max": float(row.get("max_abs_max", 0.0)),
        "rel_mae_pct_mean": float(row.get("rel_mae_pct_mean", 0.0)),
    }


def _aggregate_observable_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_cases = sum(int(row.get("n_cases", 0)) for row in rows)
    if total_cases <= 0:
        return {
            "n_cases": 0,
            "mae_mean": 0.0,
            "rmse_mean": 0.0,
            "max_abs_max": 0.0,
            "rel_mae_pct_mean": 0.0,
        }
    return {
        "n_cases": total_cases,
        "mae_mean": sum(float(row.get("mae_mean", 0.0)) * int(row.get("n_cases", 0)) for row in rows) / total_cases,
        "rmse_mean": sum(float(row.get("rmse_mean", 0.0)) * int(row.get("n_cases", 0)) for row in rows) / total_cases,
        "max_abs_max": max(float(row.get("max_abs_max", 0.0)) for row in rows),
        "rel_mae_pct_mean": sum(float(row.get("rel_mae_pct_mean", 0.0)) * int(row.get("n_cases", 0)) for row in rows) / total_cases,
    }


def _as_optional_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    return int(value)


def _aligned_current_view(result: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    aligned = result.get("aligned", {}).get("current", {})
    if isinstance(aligned, dict) and {"time_ms", "braincell_ix", "neuron_ix"} <= set(aligned):
        return aligned["time_ms"], aligned["braincell_ix"], aligned["neuron_ix"]
    return (
        result["time_ms"],
        result["braincell"]["current"]["ix"],
        result["neuron"]["current"]["ix"],
    )
