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

"""Report complete-model controls without presenting differences as kernel costs."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from benchmarks.performance.synapse_events.benchmark import HERE
from benchmarks.performance.synapse_events.report import _document_link
from benchmarks.performance.synapse_events.delivery_benchmark import CONTROL_JOBS
from benchmarks.performance.synapse_events.delivery_report import aggregate, collect

REPORT_FILENAME = "scheduled-event-controls-tables.md"
CONTROLS = ("cell_only", "unconnected", "silent", "active")
LABELS = dict(cell_only="A 无突触", unconnected="B 突触无连接", silent="C 连接静默", active="D 正常输入")


def collect_controls(root):
    """Validate model identities and enrich the shared per-worker timing rows.

    Parameters
    ----------
    root : pathlib.Path
        Root containing controls manifests and isolated worker results.

    Returns
    -------
    tuple
        Valid timing rows, failures, exclusions, configurations and environments.
    """
    root = Path(root)
    flat, excluded, failures, profiles = collect(root)
    metadata, signatures, configs, environments = {}, {}, {}, {}
    snapshot = None
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        if protocol["phase"] != "controls" or protocol.get("trace_full"):
            failures.append(f"{path}: requires non-profiling controls only")
            continue
        if snapshot is None:
            snapshot = protocol["sources"]
        elif snapshot != protocol["sources"]:
            failures.append(f"{path}: different source snapshots")
        for record in manifest["rows"]:
            worker = path.parent / record["file"]
            if not worker.exists():
                continue
            data = json.loads(worker.read_text())
            if data["status"] != "ok":
                continue
            job = record["case"]
            if job not in CONTROL_JOBS:
                failures.append(f"{worker}: unknown controls job")
                continue
            control, methods = CONTROL_JOBS[job]
            if record.get("methods") != ["production", *methods] or data.get("control") != control:
                failures.append(f"{worker}: wrong control/method assignment")
                continue
            cfg = data["case"]
            if cfg != protocol.get("case_configs", {}).get(job) or (configs and cfg != next(iter(configs.values()))):
                failures.append(f"{worker}: incompatible model configuration")
                continue
            configs[job] = cfg
            env = data["environment"]
            if environments and env["versions"] != next(iter(environments.values()))["versions"]:
                failures.append(f"{worker}: incompatible software environment")
                continue
            environments[record["device"]] = env
            for row in data["rows"]:
                if row["status"] != "ok":
                    continue
                v = row.get("validation", {})
                try:
                    assert v["synapse_count"] == (0 if control == "cell_only" else cfg["n"] * cfg["m"])
                    assert v["connection_rows"] == (cfg["n"] * cfg["m"] if methods else 0)
                    assert v["finite"] and v["reset_reproducible"]
                    assert v["final_step"] == round(cfg["duration_ms"] / cfg["dt_ms"])
                    assert abs(v["final_time_ms"] - cfg["duration_ms"]) < cfg["dt_ms"] / 2
                    assert v["arrivals"] > 0 if control == "active" else v["arrivals"] == 0
                    assert v["source_shape"] == ([cfg["n"] * cfg["m"], cfg["k"]] if methods else None)
                    for field in ("voltage_max_error_mV", "g_max_error", "drive_max_error"):
                        assert math.isfinite(row[field]) and row[field] >= 0
                    for field in ("reset_voltage_max_error_mV", "reset_g_max_error"):
                        assert math.isfinite(v[field]) and v[field] >= 0
                    assert math.isfinite(row["reset_timing"]["median_s"]) and row["reset_timing"]["median_s"] >= 0
                except (AssertionError, KeyError, TypeError, ValueError):
                    failures.append(f"{worker}/{row['method']}: invalid numerical/structure checks")
                    continue
                signature = (v["schedule_sha256"], v["drive_sha256"], v["arrivals"])
                if control in signatures and signatures[control] != signature:
                    failures.append(f"{worker}/{row['method']}: source/input mismatch across methods/devices")
                    continue
                signatures[control] = signature
                key = (str(path.parent.relative_to(root)), record["round"], record["device"], job, row["method"])
                metadata[key] = dict(
                    control=control,
                    n_steps=round(cfg["duration_ms"] / cfg["dt_ms"]),
                    synapse_count=v["synapse_count"],
                    connection_rows=v["connection_rows"],
                    arrivals=v["arrivals"],
                    reset_ms=row["reset_timing"]["median_s"] * 1000,
                    build_ms=row["build_s"] * 1000,
                    reset_cold_ms=row["reset_cold_s"] * 1000,
                    schedule_sha256=v["schedule_sha256"],
                    drive_sha256=v["drive_sha256"],
                    reset_voltage_max_error_mV=v["reset_voltage_max_error_mV"],
                    reset_g_max_error=v["reset_g_max_error"],
                )
    rows = []
    for row in flat:
        key = tuple(row[k] for k in ("run", "round", "device", "case", "method"))
        if key not in metadata:
            continue
        item = row | metadata[key]
        item["us_per_step"] = item["median_ms"] * 1000 / item["n_steps"]
        rows.append(item)
    if profiles:
        failures.append("profiling results cannot enter controls comparisons")
    return rows, failures, excluded, configs, environments


def summarize_controls(rows):
    """Aggregate matched candidates and same-round control differences.

    Parameters
    ----------
    rows : list of dict
        Validated controls timing records.

    Returns
    -------
    tuple
        Aggregate rows and differences between complete-model controls.
    """
    # Keep each worker identity when combining two production cohorts.
    paired = [r | dict(case=r["control"], run=r["run"] + "/" + r["case"]) for r in rows]
    summary = aggregate(paired)
    grouped = defaultdict(list)
    rounds = defaultdict(list)
    for row in rows:
        grouped[(row["device"], row["control"], row["method"])].append(row)
        rounds[(row["run"], row["round"], row["device"], row["control"], row["method"])].append(row["median_ms"])
    for item in summary:
        values = grouped[(item["device"], item["case"], item["method"])]
        for field in ("us_per_step", "reset_ms", "build_ms", "reset_cold_ms"):
            item[field] = statistics.median(r[field] for r in values)
    delta = defaultdict(list)
    for run, round_id, device in sorted({k[:3] for k in rounds}):

        def value(control, method):
            samples = rounds.get((run, round_id, device, control, method))
            return statistics.median(samples) if samples else None

        edges = [("B-A", "unconnected", "production", "cell_only", "production")]
        for method in ("production", "direct", "padded"):
            edges += [
                ("C-B", "silent", method, "unconnected", "production"),
                ("D-C", "active", method, "silent", method),
            ]
        for label, right, method, left, left_method in edges:
            a, b = value(left, left_method), value(right, method)
            if a is not None and b is not None:
                delta[(device, label, method)].append(b - a)
    differences = [
        dict(
            device=device,
            comparison=label,
            method=method,
            rounds=len(values),
            median_ms=statistics.median(values),
            min_ms=min(values),
            max_ms=max(values),
            samples_ms=values,
        )
        for (device, label, method), values in sorted(delta.items())
    ]
    return summary, differences


def export_report(root, report=None):
    """Write controls CSV, validation, plots and reproducible data tables.

    Parameters
    ----------
    root : pathlib.Path
        Artifact root containing controls manifests.
    report : pathlib.Path or None, optional
        Generated Markdown; defaults to scheduled-event-controls-tables.md in root.

    Returns
    -------
    dict
        Validation outcomes and measurement totals.
    """
    root = Path(root)
    report = root / REPORT_FILENAME if report is None else Path(report)
    rows, failures, excluded, configs, environments = collect_controls(root)
    summary, differences = summarize_controls(rows)
    for name, data in (
        ("controls_raw.csv", rows),
        ("controls_summary.csv", summary),
        ("controls_differences.csv", differences),
    ):
        with (root / name).open("w", newline="") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    validation = dict(
        rows=len(rows),
        aggregates=len(summary),
        failures=failures,
        excluded=excluded,
        configurations=configs,
        environments=environments,
    )
    (root / "controls_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# 完整模型四层对照数据表",
        "",
        "所有时间均为完整模型推进的墙钟时间；每步时间为整段运行的摊销值。构建、准备、编译和 reset 不计入稳态，完整轨迹校验另行运行。",
        "",
    ]
    if configs:
        cfg = next(iter(configs.values()))
        lines += [
            f"配置：N={cfg['n']}，M={cfg['m']}，每源 K={cfg['k']}；{cfg['duration_ms']} ms 生物时间，dt={cfg['dt_ms']} ms。A 无突触；B 挂突触无连接；C 与 D 时间表形状相同，但发放整体后移 {cfg['duration_ms'] + 10:g} ms。",
            "",
        ]
    lines += [
        "每个 worker 预热两次、同步计时五次，表中为进程中位数的中位数；实际轮次和全部样本以 manifest/worker JSON 为准。production 在 C/D 各有两个候选配对批次，汇总进程数因此是候选的两倍。",
        "",
        "| 设备 | 对照 | 方法 | 进程数 | 稳态 ms | 每步 µs | 进程中位数范围 ms | 配对加速 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary:
        ratio = "—" if row['speedup'] is None else f"{row['speedup']:.3f}×"
        lines.append(
            f"| {row['device']} | {LABELS[row['case']]} | {row['method']} | {row['independent_runs']} | {row['median_ms']:.3f} | {row['us_per_step']:.3f} | {row['process_min_ms']:.3f}–{row['process_max_ms']:.3f} | {ratio} |"
        )
    lines += [
        "",
        "## 首次使用与存储",
        "",
        "首次使用包含模型构建、准备、初始 reset 与首次编译执行，不含进程启动、模块导入和另行轨迹校验。稳态 reset 独立记录。显式计划数组不含源表、准备临时数据或 executable 常量。",
        "",
        "| 设备 | 对照 | 方法 | 构建 ms | 准备 ms | 首次使用 ms | 稳态 reset ms | 计划 KiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['device']} | {LABELS[row['case']]} | {row['method']} | {row['build_ms']:.1f} | {row['prepare_ms']:.1f} | {row['one_use_ms']:.1f} | {row['reset_ms']:.3f} | {row['array_bytes'] / 1024:.2f} |"
        )
    lines += [
        "",
        "## 对照变化的整体增量",
        "",
        "每个轮次先聚合同一对照的 production 配对批次，再作差；三个轮次差值取中位数。B-A 是挂载突触后的整体变化，C-B 是增加静默连接后的整体变化，D-C 是激活输入后的整体变化。差值可以为负；静默常量优化、状态和计算图变化使其不能解释为内核独占时间。",
        "",
        "| 设备 | 对照差 | 方法 | 轮次 | 差值 ms | 各轮范围 ms |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in differences:
        lines.append(
            f"| {row['device']} | {row['comparison']} | {row['method']} | {row['rounds']} | {row['median_ms']:.3f} | {row['min_ms']:.3f}–{row['max_ms']:.3f} |"
        )
    lines += ["", "## 校验与复现", "", f"有效计时 {len(rows)} 条；失败 {len(failures)} 项；排除 {len(excluded)} 项。"]
    lines += [f"- {item}" for item in failures]
    lines += [f"- {item['device']}/{item['case']}: {item['status']} — {item['reason']}" for item in excluded]
    lines += [
        "",
        _document_link("已维护结果解释", HERE / "results/scheduled-event-controls.md", report)
        + " · "
        + _document_link("复现命令", HERE / "README.md", report)
        + " · "
        + _document_link("第二轮投递结果", HERE / "results/scheduled-event-delivery.md", report),
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    if summary:
        _plot(root, summary)
    return validation


def _plot(root, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {(r['device'], r['case'], r['method']): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, device in zip(axes, ("cpu", "gpu")):
        x = np.arange(4)
        for i, (method, color) in enumerate((("production", "#526477"), ("direct", "#168a76"), ("padded", "#9752a3"))):
            values = [lookup.get((device, c, method)) for c in CONTROLS]
            heights = [r['median_ms'] if r else np.nan for r in values]
            bars = ax.bar(x + (i - 1) * 0.23, heights, 0.22, label=method, color=color)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        ax.set_xticks(x, ["A: cells", "B: +synapses", "C: +silent routes", "D: active"])
        ax.set_title(device.upper())
        ax.set_ylabel("4000-step full rollout / wall ms")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, frameon=False)
    fig.savefig(root / "controls_comparison.png", dpi=170)
    fig.savefig(root / "controls_comparison.svg")
    plt.close(fig)


def main(argv=None):
    """Export controls; return nonzero for failed or incompatible evidence.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments.

    Returns
    -------
    int
        One when validation failed, otherwise zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, help="Output Markdown; defaults to the input artifact directory")
    args = parser.parse_args(argv)
    result = export_report(args.directory, args.report)
    print(json.dumps({k: result[k] for k in ('rows', 'aggregates', 'failures')}, indent=2))
    return int(bool(result['failures']))


if __name__ == "__main__":
    raise SystemExit(main())
