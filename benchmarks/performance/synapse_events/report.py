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

"""Generate inspectable tables, plots and numerical checks from event benchmarks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.performance.synapse_events.benchmark import Case, HERE


REPORT_FILENAME = "netstim-baseline.md"


def _document_link(label, target, document):
    """Link a local file relative to the actual report, including paths with spaces."""
    relative = Path(os.path.relpath(Path(target).resolve(), Path(document).resolve().parent)).as_posix()
    return f"[{label}](<{relative}>)"


def load_results(directory):
    """Load only successful rows referenced by a run manifest.

    Parameters
    ----------
    directory : pathlib.Path or str
        Run directory containing manifest.json and its measurement files.

    Returns
    -------
    tuple
        Manifest, successful measurements keyed by (case key, device), and failures.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows, failures = {}, []
    for entry in manifest["rows"]:
        path = directory / entry["file"]
        if entry["status"] != "ok" or not path.exists():
            failures.append(entry)
            continue
        row = json.loads(path.read_text())
        if row.get("status") != "ok":
            failures.append(entry)
            continue
        rows[entry["key"], entry["device"]] = dict(row, groups=entry["groups"], device=entry["device"])
    return manifest, rows, failures


def validate_results(rows):
    """Compare devices, layouts and schedule lengths independently of timings.

    Parameters
    ----------
    rows : dict
        Successful measurements indexed by case key and device.

    Returns
    -------
    dict
        Comparison counts and explicit mismatch descriptions.
    """
    checks = dict(
        device_pairs=0,
        layout_pairs=0,
        schedule_pairs=0,
        declaration_pairs=0,
        max_voltage_error_mv=0.0,
        max_conductance_error_us=0.0,
        failures=[],
    )

    def compare(a, b, label):
        if a["arrivals_per_source"] != b["arrivals_per_source"]:
            checks["failures"].append(label + ": event counts differ")
        for field in ("final_voltage_mv", "final_conductance_us"):
            error = float(np.max(np.abs(np.asarray(a[field]) - np.asarray(b[field]))))
            error_key = "max_voltage_error_mv" if field.endswith("mv") else "max_conductance_error_us"
            checks[error_key] = max(checks[error_key], error)
            if not np.allclose(a[field], b[field], rtol=3e-4, atol=2e-5 if field.endswith("mv") else 2e-7):
                checks["failures"].append(label + f": {field} differs (max absolute error {error:.9g})")

    for (key, device), row in rows.items():
        case = Case(**row["case"])
        if device == "cpu" and (key, "gpu") in rows:
            checks["device_pairs"] += 1
            compare(row, rows[key, "gpu"], key + " CPU/GPU")
        if case.layout == "independent" and (replace(case, layout="shared").key, device) in rows:
            checks["layout_pairs"] += 1
            compare(row, rows[replace(case, layout="shared").key, device], key + " independent/shared")
        if "schedule" in row["groups"] and case.number != 10:
            counterpart = rows.get((replace(case, number=10).key, device))
            if counterpart:
                checks["schedule_pairs"] += 1
                compare(row, counterpart, key + " schedule length")
        if case.declaration == "per_cell":
            counterpart = rows.get((replace(case, declaration="batched").key, device))
            if counterpart:
                checks["declaration_pairs"] += 1
                compare(row, counterpart, key + " declaration")
    return checks


def export_csv(rows, destination):
    """Export one row per case, device and measurement mode.

    Parameters
    ----------
    rows : dict
        Successful measurements indexed by case key and device.
    destination : pathlib.Path or str
        CSV output path, with parent directories created as needed.
    """
    flat = []
    for row in rows.values():
        for mode, timing in row["timing"].items():
            flat.append(
                dict(
                    key=row["key"],
                    device=row["device"],
                    mode=mode,
                    **row["case"],
                    groups=";".join(row["groups"]),
                    arrival_count=row["arrival_count"],
                    synapse_count=row["synapse_count"],
                    schedule_events_per_source=(row["schedule_shape"] or [0, 0])[1],
                    build_s=row["build_s"],
                    source_build_s=row["source_build_s"],
                    prepare_s=row["prepare_s"],
                    first_call_s=timing["first_call_s"],
                    median_s=timing["median_s"],
                    stdev_s=timing["stdev_s"],
                    us_per_step=timing["us_per_step"],
                    events_per_second=timing["events_per_second"],
                    samples_s=json.dumps(timing["samples_s"]),
                )
            )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else ["key", "device", "mode"])
        writer.writeheader()
        writer.writerows(flat)


def plot_results(rows, directory):
    """Write standalone scaling and schedule-length figures with error bars.

    Parameters
    ----------
    rows : dict
        Successful measurements indexed by case key and device.
    directory : pathlib.Path or str
        Destination for PNG and SVG figures.

    Returns
    -------
    list of pathlib.Path
        Generated PNG paths; matching SVGs are also written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for group, xfield, title in (("scaling", "n", "Cell scaling"), ("schedule", "number", "Future schedule length")):
        selected = [r for r in rows.values() if group in r["groups"]]
        if not selected:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, layout in zip(axes, ("independent", "shared")):
            subsets = {}
            for row in selected:
                if row["case"]["layout"] == layout:
                    subsets.setdefault((row["device"], row["case"]["m"]), []).append(row)
            for (device, m), members in sorted(subsets.items()):
                members.sort(key=lambda r: r["case"][xfield])
                x = [r["case"][xfield] for r in members]
                y = [r["timing"]["full"]["median_s"] * 1000 for r in members]
                err = [r["timing"]["full"]["stdev_s"] * 1000 for r in members]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{device.upper()}, M={m}")
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="Cells N" if xfield == "n" else "Events stored per source K",
                ylabel="Full rollout median (ms)",
                title=layout,
            )
            ax.grid(True, which="both", alpha=0.2)
            if subsets:
                ax.legend(fontsize=8)
        protocol = selected[0]["case"]
        fig.suptitle(title + f" ({protocol['duration_ms']:g} ms, dt={protocol['dt_ms']:g} ms; bars: SD)")
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"{group}.{suffix}", dpi=160)
        plt.close(fig)
        outputs.append(directory / f"{group}.png")
    return outputs


def _ms(row, mode="full"):
    return row["timing"][mode]["median_s"] * 1000


def findings(rows):
    """Derive bounded performance observations from matched configurations.

    Parameters
    ----------
    rows : dict
        Successful measurements indexed by case key and device.

    Returns
    -------
    list of str
        Observations limited to the measured configuration pairs.
    """
    text = []
    pairs = [
        (rows[key, "cpu"], row)
        for (key, device), row in rows.items()
        if device == "gpu" and (key, "cpu") in rows and "scaling" in row["groups"]
    ]
    if pairs:
        gains = [_ms(cpu) / _ms(gpu) for cpu, gpu in pairs]
        text.append(
            f"规模 sweep 中 GPU 快于 CPU 的配置为 {sum(g > 1 for g in gains)}/{len(gains)}；"
            f"CPU/GPU 完整运行时间比范围 {min(gains):.2f}–{max(gains):.2f}。小于 1 表示 CPU 更快。"
        )
    for device in ("cpu", "gpu"):
        for layout in ("independent", "shared"):
            selected = sorted(
                [
                    r
                    for r in rows.values()
                    if r["device"] == device and r["case"]["layout"] == layout and "schedule" in r["groups"]
                ],
                key=lambda r: r["case"]["number"],
            )
            if len(selected) >= 2:
                first, last = selected[0], selected[-1]
                text.append(
                    f"{device.upper()} / {layout}：时间表从 {first['case']['number']} 增至 "
                    f"{last['case']['number']} 条/源，完整运行 {_ms(first):.3f} → {_ms(last):.3f} ms "
                    f"（{_ms(last) / _ms(first):.2f}×），查询微基准 {_ms(first, 'query'):.3f} → "
                    f"{_ms(last, 'query'):.3f} ms。窗口内到达计数及最终状态的匹配结果见数值检查。"
                )
        large = [
            (r, rows.get((replace(Case(**r["case"]), layout="shared").key, device)))
            for r in rows.values()
            if r["device"] == device and "scaling" in r["groups"] and r["case"]["layout"] == "independent"
        ]
        large = [(a, b) for a, b in large if b is not None]
        if large:
            independent, shared = max(large, key=lambda pair: pair[0]["case"]["n"] * pair[0]["case"]["m"])
            c = independent["case"]
            text.append(
                f"{device.upper()}，N={c['n']}、M={c['m']}：独立突触 {_ms(independent):.3f} ms，"
                f"共享突触 {_ms(shared):.3f} ms；独立/共享时间比 {_ms(independent) / _ms(shared):.2f}。"
            )
        declarations = [
            row
            for row in rows.values()
            if row["device"] == device
            and row["case"]["declaration"] == "per_cell"
            and row["case"]["layout"] == "independent"
        ]
        if declarations:
            split = max(declarations, key=lambda row: row["case"]["n"])
            batch = rows.get((replace(Case(**split["case"]), declaration="batched").key, device))
            if batch:
                text.append(
                    f"{device.upper()}，N={split['case']['n']}、M={split['case']['m']}："
                    f"批量连接 {_ms(batch):.3f} ms，每细胞一次连接 {_ms(split):.3f} ms；"
                    f"完整模拟首次调用分别 {batch['timing']['full']['first_call_s']:.3f} / "
                    f"{split['timing']['full']['first_call_s']:.3f} s（含编译和执行）。"
                )
    return text


def write_report(directory, markdown=None):
    """Produce the complete reproducible report and numerical validation record.

    Parameters
    ----------
    directory : pathlib.Path or str
        Input run directory; generated tables and figures are also stored here.
    markdown : pathlib.Path or str or None, optional
        Destination for the generated report; defaults to netstim-baseline.md
        in the input directory. Maintained summaries are not overwritten by default.

    Returns
    -------
    dict
        Numerical comparison counts and any discrepancies.
    """
    directory = Path(directory)
    markdown = directory / REPORT_FILENAME if markdown is None else Path(markdown)
    manifest, rows, failures = load_results(directory)
    checks = validate_results(rows)
    export_csv(rows, directory / "summary.csv")
    plots = plot_results(rows, directory)
    lines = [
        "# NetStim 突触事件性能实测",
        "",
        f"运行记录：{manifest['created_utc']}。成功 {len(rows)} 个设备/配置，失败或超时 {len(failures)} 个。",
        f"基于提交 `{manifest['git_head']}` 的工作区；完整未提交文件清单保存在 manifest。",
        "",
        "## 测量口径",
        "",
        "每个配置使用独立进程。CPU/GPU 使用同一 Python 环境及 CPU 生成的固定种子事件表。"
        "每条轨迹采用 jit + for_loop；reset 在计时区间外执行并同步，计时包含设备完成等待。",
        f"首次调用单列（含编译及执行），另预热 {manifest['warmup']} 次、测量 {manifest['repeat']} 次。"
        "查询与聚合为独立编译微基准，不能相加或除以完整运行时间解释成阶段占比。",
        "",
        "NetStim 构造生成日程，Cell 每步查询日程并加权 scatter-add；本实验不测 live Cell spike 延迟队列，"
        "也不比较 scatter/brainevent 后端。完整运行使用 prepare_run + update，不包括 Network.run 的结果表物化。",
        "",
        "## 主要观察",
        "",
    ]
    lines += ["- " + observation for observation in findings(rows)] or ["暂无足够的匹配配置。"]
    lines += [
        "",
        "## 数值检查",
        "",
        f"CPU/GPU 配对 {checks['device_pairs']}；独立/共享突触配对 {checks['layout_pairs']}；"
        f"时间表长度配对 {checks['schedule_pairs']}；连接声明方式配对 {checks['declaration_pairs']}。",
        "每个 worker 还检查查询计数、聚合权重和最终状态有限性。",
        f"配对最大电压差 {checks['max_voltage_error_mv']:.9g} mV；最大电导差 "
        f"{checks['max_conductance_error_us']:.9g} µS。事件计数要求精确一致；状态 rtol=3e-4，"
        "电压 atol=2e-5 mV，电导 atol=2e-7 µS。未因观测结果放宽阈值。",
        "结果："
        + (
            "全部配对通过。"
            if not checks["failures"]
            else "存在不匹配，不能声称所有配置在 CPU/GPU 上严格数值等价；下列条目保留具体误差。"
        ),
        "",
    ]
    lines += ["- " + failure for failure in checks["failures"]]
    for group, title in (
        ("smoke", "最小案例"),
        ("scaling", "规模 sweep"),
        ("activity", "事件活动"),
        ("schedule", "时间表长度"),
        ("delay", "延迟"),
        ("declarations", "连接声明"),
        ("controls", "对照组"),
    ):
        keys = list(dict.fromkeys(key for (key, _), row in rows.items() if group in row["groups"]))
        if not keys:
            continue
        lines += [
            f"## {title}",
            "",
            "完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。",
            "",
            "| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for key in keys:
            cpu, gpu = rows.get((key, "cpu")), rows.get((key, "gpu"))
            row = cpu or gpu
            c = row["case"]
            detail = (
                f"{c['rate_hz']:g} Hz, {c['pattern']}"
                if group == "activity"
                else f"K={c['number']}"
                if group == "schedule"
                else c["delay"]
                if group == "delay"
                else c["declaration"]
                if group == "declarations"
                else c["control"]
                if group == "controls"
                else "默认"
            )

            def total(r):
                return f"{_ms(r):.3f}" if r else "—"

            def micro(r):
                return f"{_ms(r, 'query'):.3f}/{_ms(r, 'aggregate'):.3f}" if r and "query" in r["timing"] else "—"

            lines.append(
                f"| {c['n']} | {c['m']} | {c['layout']} | {detail} | {row['arrival_count']} | "
                f"{total(cpu)} | {total(gpu)} | {micro(cpu)} | {micro(gpu)} |"
            )
    lines += [
        "",
        "## 构造、准备与首次调用",
        "",
        "选取各布局最大的规模配置；全部配置及每次重复值见 summary.csv / 原始 JSON。单位 s。",
        "",
        "| 设备 | N×M | 布局 | 构造（含源） | 其中源构造 | 初始化准备 | 完整模拟首次调用 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for device in ("cpu", "gpu"):
        for layout in ("independent", "shared"):
            candidates = [
                r
                for r in rows.values()
                if r["device"] == device and r["case"]["layout"] == layout and "scaling" in r["groups"]
            ]
            if candidates:
                row = max(candidates, key=lambda r: r["case"]["n"] * r["case"]["m"])
                c = row["case"]
                lines.append(
                    f"| {device} | {c['n']}×{c['m']} | {layout} | {row['build_s']:.3f} | "
                    f"{row['source_build_s']:.3f} | {row['prepare_s']:.3f} | {row['timing']['full']['first_call_s']:.3f} |"
                )
    precision_rows = []
    for precision in (32, 64):
        diagnostic_dir = directory / "diagnostics" / f"precision{precision}"
        if (diagnostic_dir / "manifest.json").exists():
            _, diagnostics, diagnostic_failures = load_results(diagnostic_dir)
            for (key, device), cpu in diagnostics.items():
                gpu = diagnostics.get((key, "gpu"))
                if device == "cpu" and gpu:
                    error = float(
                        np.max(np.abs(np.asarray(cpu["final_voltage_mv"]) - np.asarray(gpu["final_voltage_mv"])))
                    )
                    same_counts = cpu["arrivals_per_source"] == gpu["arrivals_per_source"]
                    precision_rows.append((precision, cpu["case"], error, same_counts))
            if diagnostic_failures:
                lines += ["", f"精度 {precision} 诊断有 {len(diagnostic_failures)} 个未完成配置。"]
    if precision_rows:
        lines += [
            "",
            "## 精度补充诊断",
            "",
            "CPU/GPU 分进程运行，所有精度使用同样的 CPU float32 随机源日程；该对照用于调查单精度电压差，"
            "不与主性能基线混合，也不改变主检查的阈值。",
            "",
            "| 精度 | N×M | 布局 | 刺激 | CPU/GPU 最大电压差 (mV) | 事件计数一致 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {precision} | {case['n']}×{case['m']} | {case['layout']} | "
            f"{case['rate_hz']:g} Hz {case['pattern']} | {error:.9g} | {same} |"
            for precision, case, error, same in precision_rows
        ]
        lines += [
            "",
            "原始配置和结果保存在同一运行目录的 `diagnostics/precision32` 与 `diagnostics/precision64`。",
            "主表的电压不匹配仍保留；精度诊断仅覆盖上表配置，不能外推为所有形态和求解器均已验证。",
        ]
        single_errors = [error for precision, _, error, _ in precision_rows if precision == 32]
        double_errors = [error for precision, _, error, _ in precision_rows if precision == 64]
        if single_errors and double_errors and max(double_errors) < max(single_errors) * 1e-3:
            lines += [
                "",
                "代表配置的跨设备电压差在双精度下显著减小，支持误差来自单精度细胞数值计算的判断。"
                "事件计数检查独立通过；尚未把电压差定位到某个求解器运算，不能将其归因于 NetStim 路由错误。",
            ]
    trace_summaries = []
    for trace_name in ("trace", "trace_scopes"):
        trace_root = directory / "diagnostics" / trace_name
        for path in sorted(trace_root.glob("*_summary.json")):
            trace_summaries.append((path, json.loads(path.read_text())))
    if trace_summaries:
        lines += [
            "",
            "## GPU trace：执行粒度与归因边界",
            "",
            "trace 在预热和正式计时后单独采集。下面是 GPU 事件时长之和，不是完整运行墙钟时间；"
            "关闭 command buffer 的运行仅供归因诊断，不纳入基准速度比较。",
            "",
            "| command buffer | 布局 | GPU 事件数 | 每步约事件数 | GPU 事件时长之和 (ms) | 匹配 scope 事件数 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for path, summary in trace_summaries:
            case = Case(**summary["case"])
            lines.append(
                f"| {summary['command_buffers']} | {case.layout} | {summary['total_gpu_events']} | "
                f"{summary['total_gpu_events'] / case.steps:.2f} | {summary['total_gpu_time_ms']:.3f} | "
                f"{summary['matched_gpu_events']} |"
            )
        for path, summary in trace_summaries:
            if summary["command_buffers"] == "default":
                lines += [
                    "",
                    f"{summary['case']['layout']} 的前五个 GPU kernel 分组（按设备事件时长之和）：",
                    "",
                    "| kernel 名称 | 次数 | 累计 ms | 平均 µs |",
                    "| --- | --- | --- | --- |",
                ]
                lines += [
                    f"| `{kernel['name']}` | {kernel['calls']} | {kernel['total_ms']:.3f} | {kernel['mean_us']:.3f} |"
                    for kernel in summary["kernels"][:5]
                ]
                lines += ["", _document_link("原始 trace 解析结果", path, markdown)]
        if not any(summary["matched_gpu_events"] for _, summary in trace_summaries):
            lines += [
                "",
                "本环境的 XPlane 解析未匹配到 BrainCell named scope，因此不提供 event/积分的精确占比。"
                "融合 kernel 名称本身不足以证明它只执行某个阶段；可确认的是大量短 GPU 事件反复执行。",
            ]
    lines += ["", "## 图与原始记录", ""]
    for path in plots:
        lines += ["!" + _document_link(path.stem, path, markdown), ""]
    lines += [
        _document_link("原始 manifest", directory / "manifest.json", markdown)
        + " · "
        + _document_link("完整 CSV", directory / "summary.csv", markdown),
        _document_link("公共实验入口", HERE / "README.md", markdown)
        + " · "
        + _document_link("已维护结果索引", HERE / "RESULTS.md", markdown),
        "",
        "## 环境与限制",
        "",
    ]
    environments = {}
    for row in rows.values():
        environments.setdefault(row["device"], row["environment"])
    for device, env in environments.items():
        lines += [
            f"- {device.upper()}：{env['device_kind']}，x64={env['x64']}，"
            f"Python {env['python'].split()[0]}，版本 `{json.dumps(env['versions'])}`。"
        ]
    cpu_info = manifest.get("cpu_info", "")
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in cpu_info.splitlines() if line.startswith("model name")), "未记录"
    )
    lines += [f"- 宿主 CPU：{cpu_model}；进程可用逻辑 CPU 数：{len(manifest.get('cpu_affinity') or [])}。"]
    lines += [
        "",
        "GPU 初始占用快照：",
        "",
        "```text",
        manifest.get("gpu_status", "未记录").strip(),
        "```",
        "",
        "构造阶段包含首次随机生成及初始化操作的编译，不能等同于纯 Python 拓扑构造。"
        "进程内分配器的内存统计是设备观测值，不代表单个 event kernel 的独占峰值。",
        "静默对照保留时间表形状，但编译器仍可能优化常量；差值只表示该对照实验的整体变化。"
        "其他 GPU/CPU 上同时运行的工作会影响宿主机负载；小差异需结合重复值判断。",
        "",
    ]
    if failures:
        lines += ["## 未完成配置", ""] + [f"- {r['device']} {r['key']}：{r['status']}。" for r in failures]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n")
    (directory / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def main(argv=None):
    """Generate a report from command-line paths and fail on numerical mismatches.

    Parameters
    ----------
    argv : list of str or None
        Arguments to parse, defaulting to process arguments.

    Returns
    -------
    int
        One if paired numerical checks failed, otherwise zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "artifacts" / "baseline")
    parser.add_argument("--markdown", type=Path, help="Output Markdown; defaults to the input run directory")
    args = parser.parse_args(argv)
    checks = write_report(args.input, args.markdown)
    print(json.dumps(checks, indent=2))
    return int(bool(checks["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
