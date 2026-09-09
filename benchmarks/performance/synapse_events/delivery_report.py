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

"""Summarize bounded delivery trials without hiding missing or failed work."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from benchmarks.performance.synapse_events.benchmark import ROOT, HERE
from benchmarks.performance.synapse_events.report import _document_link

REPORT_FILENAME = "scheduled-event-delivery-tables.md"


def collect(root):
    """Validate manifests and flatten measurements, preserving exclusions.

    Parameters
    ----------
    root : pathlib.Path
        Root containing phase/run subdirectories with manifests.

    Returns
    -------
    tuple
        Flat timing records, worker exclusions, validation failures and profiles.
    """
    flat, excluded, failures, profiles = [], [], [], []
    pairs = defaultdict(dict)
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        failures.append("no manifests found")
    for path in manifests:
        manifest = json.loads(path.read_text())
        protocol = manifest["protocol"]
        phase = protocol["phase"]
        expected = len(protocol["cases"]) * len(protocol["devices"]) * protocol["rounds"]
        if len(manifest["rows"]) != expected:
            failures.append(f"{path.parent.name}: incomplete workers {len(manifest['rows'])}/{expected}")
        identities = set()
        for record in manifest["rows"]:
            identity = (record["device"], record["case"], record["round"])
            if identity in identities:
                failures.append(f"{path.parent.name}: duplicate worker {identity}")
            identities.add(identity)
            worker = path.parent / record["file"]
            if not worker.exists():
                failures.append(f"missing {worker.relative_to(root)}")
                continue
            data = json.loads(worker.read_text())
            prefix = dict(
                run=str(path.parent.relative_to(root)),
                round=record["round"],
                phase=phase,
                precision=protocol["precision"],
                device=record["device"],
                case=record["case"],
            )
            if data["status"] != "ok":
                excluded.append(prefix | dict(status=data["status"], reason=data.get("reason", data.get("error", ""))))
                failures.append(f"{worker.relative_to(root)}: {data['status']}")
                continue
            expected_methods = set(record.get("methods", protocol["methods"])) | (
                {"production"} if phase in ("full", "controls") else set()
            )
            if {r["method"] for r in data["rows"]} != expected_methods or len(data["rows"]) != len(expected_methods):
                failures.append(f"{worker.relative_to(root)}: incomplete methods")
            if phase in ("micro", "profile"):
                key = (record["case"], protocol["precision"])
                signature = (data["schedule_sha256"], data["count_sha256"], data["arrivals"])
                if pairs[key] and any(other != signature for other in pairs[key].values()):
                    failures.append(f"{worker.relative_to(root)}: source/count mismatch across runs/devices")
                pairs[key][str(worker)] = signature
            for row in data["rows"]:
                if row["status"] != "ok":
                    excluded.append(
                        prefix | dict(method=row["method"], status=row["status"], reason=row.get("reason", ""))
                    )
                    continue
                if phase == "profile" or protocol.get("trace_full", False):
                    profiles.append(
                        prefix
                        | dict(
                            method=row["method"],
                            status=row.get("profile_status", "missing"),
                            path=str(worker.with_suffix("").relative_to(root)),
                        )
                    )
                    continue
                for mode, metric in row["timing"].items():
                    cold = (
                        data.get("build_s", row.get("build_s", 0))
                        + row["prepare_cold_s"]
                        + row.get("reset_cold_s", 0)
                        + metric["first_call_s"]
                    )
                    value = prefix | dict(
                        control=data.get("control", "active"),
                        method=row["method"],
                        mode=mode,
                        median_ms=metric["median_s"] * 1000,
                        min_ms=metric["min_s"] * 1000,
                        max_ms=metric["max_s"] * 1000,
                        prepare_ms=row["prepare_cold_s"] * 1000,
                        first_call_ms=metric["first_call_s"] * 1000,
                        one_use_ms=cold * 1000,
                        reuse_10_ms=(cold + 9 * metric["median_s"]) * 1000,
                        reuse_100_ms=(cold + 99 * metric["median_s"]) * 1000,
                        array_bytes=row["array_bytes"],
                        arrivals=data.get("arrivals"),
                        voltage_max_error_mV=row.get("voltage_max_error_mV"),
                        g_max_error=row.get("g_max_error"),
                        count_max_error=row.get("count_max_error"),
                        drive_max_error=row.get("drive_max_error"),
                        source_snapshot=str(path.parent.relative_to(root) / "source_snapshot"),
                    )
                    flat.append(value)
    return flat, excluded, failures, profiles


def aggregate(flat):
    """Compare medians across independent workers and paired baselines.

    Parameters
    ----------
    flat : list of dict
        Per-worker timing rows.

    Returns
    -------
    list of dict
        Aggregate timings and evidence-qualified speedup classifications.
    """
    groups = defaultdict(list)
    baselines = {}
    for row in flat:
        key = tuple(row[k] for k in ("phase", "precision", "device", "case", "mode"))
        groups[key + (row["method"],)].append(row)
        if row["method"] == ("production" if row["phase"] in ("full", "controls") else "current"):
            baselines[key + (row["run"], row["round"])] = row
    result = []
    for key, rows in groups.items():
        medians = [r["median_ms"] for r in rows]
        pairs = [(r, baselines.get(key[:-1] + (r["run"], r["round"]))) for r in rows]
        pairs = [(r, b) for r, b in pairs if b is not None]
        ratios = [b["median_ms"] / r["median_ms"] for r, b in pairs]
        ratio = statistics.median(ratios) if ratios else None
        stable = len(pairs) >= 3 and all(r["median_ms"] < b["median_ms"] for r, b in pairs) and ratio >= 1.1
        item = dict(zip(("phase", "precision", "device", "case", "mode", "method"), key))
        item.update(
            independent_runs=len(rows),
            median_ms=statistics.median(medians),
            process_min_ms=min(medians),
            process_max_ms=max(medians),
            speedup=ratio,
            stable_gain=stable,
        )
        for field in ("prepare_ms", "one_use_ms", "reuse_10_ms", "reuse_100_ms", "array_bytes"):
            item[field] = statistics.median(r[field] for r in rows)
        result.append(item)
    return sorted(
        result, key=lambda r: tuple(str(r[k]) for k in ("phase", "precision", "device", "case", "mode", "method"))
    )


def export_report(root, report=None):
    """Write CSV, charts and an evidence-scoped Chinese results page.

    Parameters
    ----------
    root : pathlib.Path
        Artifact root containing run manifests.
    report : pathlib.Path or None, optional
        Output Markdown; defaults to scheduled-event-delivery-tables.md in root.

    Returns
    -------
    dict
        Measurement count and all validation failures/exclusions.
    """
    root = Path(root)
    report = root / REPORT_FILENAME if report is None else Path(report)
    flat, excluded, failures, profiles = collect(root)
    summary = aggregate(flat)
    for name, rows in (("delivery_raw.csv", flat), ("delivery_summary.csv", summary)):
        with (root / name).open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    validation = dict(rows=len(flat), aggregates=len(summary), failures=failures, excluded=excluded, profiles=profiles)
    (root / "delivery_validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False))
    lines = [
        "# 预定事件直接投递完整数据表",
        "",
        "第二轮比较 current、预计算 scan、原 bucket、紧凑 direct 和固定形状 padded，分别评估 CPU/GPU。完整模型由实验 Cell 子类替换预定事件输入计算，复用真实 Network/Cell solver；production 为未安装适配器的对照。",
        "",
        "默认 100 ms、dt=.025 ms、4000 步、float32、种子 7；每 worker 预热 2 次、重复 5 次、同步等待。表中为独立进程中位数的中位数。profile 运行不进入计时汇总；本轮 float64 仅做正确性测试。",
        "",
        "## 稳态运行时间",
        "",
        "单位 ms，越小越好；— 表示本组合没有成功计时，原因见排除记录。synapse 包含查询、权重归约和 ExpSyn 更新；full 包含真实膜电位积分。",
        "",
    ]
    methods = ("production", "current", "scan", "bucket", "direct", "padded")
    lines += [
        "| 模式 | 设备 | 场景 | " + " | ".join(methods) + " |",
        "| --- | --- | --- | " + " | ".join(["---:"] * len(methods)) + " |",
    ]
    table = defaultdict(dict)
    for row in summary:
        if row["precision"] == 32 and row["mode"] in ("synapse", "full"):
            table[(row["mode"], row["device"], row["case"])][row["method"]] = row
    for key, values in sorted(table.items()):
        lines.append(
            "| "
            + " | ".join(key)
            + " | "
            + " | ".join(f"{values[m]['median_ms']:.3f}" if m in values else "—" for m in methods)
            + " |"
        )
    lines += [
        "",
        "## 稳定收益与复用成本",
        "",
        "稳定收益要求至少三个独立配对进程均更快，配对加速比中位数 ≥1.10。此规则只支持所测场景；不等于统计置信区间或全局自动选择阈值。",
        "",
        "| 模式 | 设备 | 场景 | 方法 | 独立进程 | 稳态加速 | 稳定收益 | 首次使用 ms | 100 次估算 ms | 数组 KiB |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        if row["precision"] != 32 or row["mode"] not in ("synapse", "full"):
            continue
        ratio = "—" if row["speedup"] is None else f"{row['speedup']:.2f}×"
        lines.append(
            f"| {row['mode']} | {row['device']} | {row['case']} | {row['method']} | {row['independent_runs']} | {ratio} | {'是' if row['stable_gain'] else '否'} | {row['one_use_ms']:.1f} | {row['reuse_100_ms']:.1f} | {row['array_bytes'] / 1024:.2f} |"
        )
    lines += [
        "",
        "首次使用包含源/模型构建、计划准备、首次执行及 full 的初始 reset；首次执行含编译，不能称为纯执行。full 稳态和 100 次重用估算不含后续运行间 reset，不能视为独立重复实验的完整总时间。micro 和 full 的构建范围不同，不跨模式比较冷启动。数组字节只统计显式计划存储，current/production 为零不代表实际内存为零；准备临时完整事件表和 executable 常量没有计入，allocator 统计保留在 worker JSON，不能当作内核峰值。",
        "",
        "## 校验与排除",
        "",
        f"导出 {len(flat)} 条计时记录；校验失败 {len(failures)} 项，排除记录 {len(excluded)} 项。",
    ]
    lines += [f"- {item}" for item in failures]
    unique_excluded = sorted(
        {
            f"{r['phase']}/{r['device']}/{r['case']}/{r.get('method', 'worker')}: {r['status']} — {r['reason']}"
            for r in excluded
        }
    )
    lines += [f"- {item}" for item in unique_excluded]
    lines += [
        "",
        "## 复现与证据",
        "",
        "原始记录位于本次输入目录；delivery_validation.json 保留所有未通过、超时和不适用项。",
        _document_link("输入目录", root, report)
        + " · "
        + _document_link("benchmark README", HERE / "README.md", report),
        "",
        _document_link("已维护结果解释与选型", HERE / "results/scheduled-event-delivery.md", report)
        + " · "
        + _document_link("方案", ROOT / "docs/design/synapse/proposals/event-delivery-optimization.md", report)
        + " · "
        + _document_link("已维护查询结果", HERE / "results/scheduled-event-queries.md", report),
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    if table:
        _plot(root, table)
    return validation


def _plot(root, table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    methods = ("production", "current", "scan", "bucket", "direct", "padded")
    colors = ("#526477", "#8799aa", "#3577b5", "#e6a14a", "#168a76", "#9752a3")
    for i, mode in enumerate(("synapse", "full")):
        for j, device in enumerate(("cpu", "gpu")):
            ax = axes[i, j]
            cases = [
                c
                for c in ("small", "large", "long", "sparse", "burst", "dense", "shared")
                if (mode, device, c) in table
            ]
            x = np.arange(len(cases))
            for k, (method, color) in enumerate(zip(methods, colors)):
                y = [table[(mode, device, c)].get(method, {}).get("median_ms", np.nan) for c in cases]
                ax.bar(x + (k - 2.5) * 0.13, y, 0.12, label=method, color=color)
            ax.set_xticks(x, cases)
            ax.set_yscale("log")
            ax.set_ylabel("100 ms rollout / wall ms (log)")
            ax.set_title(f"{device.upper()} · {mode}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=6, frameon=False)
    fig.savefig(root / "delivery_comparison.png", dpi=170)
    fig.savefig(root / "delivery_comparison.svg")
    plt.close(fig)


def main(argv=None):
    """Export an artifact root; return nonzero on incomplete validation.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments.

    Returns
    -------
    int
        One when any worker or cross-device validation fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, help="Output Markdown; defaults to the input artifact directory")
    args = parser.parse_args(argv)
    result = export_report(args.directory, args.report)
    print(json.dumps({k: v for k, v in result.items() if k not in ("excluded", "profiles")}, indent=2))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
