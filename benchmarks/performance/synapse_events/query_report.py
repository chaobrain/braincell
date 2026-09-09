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

"""Export scheduled query measurements and device comparisons."""

import argparse
import csv
import json
import math
from pathlib import Path

from benchmarks.performance.synapse_events.benchmark import ROOT, HERE
from benchmarks.performance.synapse_events.report import _document_link
from benchmarks.performance.synapse_events.query_benchmark import cases_for

METHODS = ("current", "scan", "cursor", "bucket")
REPORT_FILENAME = "scheduled-event-queries.md"


def export_report(directory, report=None):
    """Write CSV, figures and a Chinese results page from a manifest.

    Parameters
    ----------
    directory : pathlib.Path or str
        Directory containing the measurement manifest and worker JSON files.
    report : pathlib.Path or str or None, optional
        Destination Markdown; defaults to scheduled-event-queries.md in directory.

    Returns
    -------
    dict
        Case count, flattened rows and validation failures.
    """
    directory = Path(directory)
    report = directory / REPORT_FILENAME if report is None else Path(report)
    manifest = json.loads((directory / "manifest.json").read_text())
    cases, failures, flat, pairs = [], [], [], {}
    protocol = manifest["protocol"]
    if "suite" in protocol:
        expected = len(cases_for(protocol["suite"])) * (2 if protocol["devices"] == "both" else 1)
        if len(manifest["rows"]) < expected:
            failures.append(f"incomplete run: {len(manifest['rows'])}/{expected} worker records")
    for item in manifest["rows"]:
        path = directory / item["file"]
        if not path.exists():
            failures.append(f"missing {item['file']}")
            continue
        data = json.loads(path.read_text())
        if data["status"] != "ok":
            failures.append(f"failed {item['file']}: {data.get('error', '')}")
            continue
        if {r["method"] for r in data["rows"]} != set(METHODS):
            failures.append(f"incomplete methods: {item['file']}")
            continue
        cases.append((item, data))
        pairs.setdefault(item["key"], {})[item["device"]] = data
        base = next(r for r in data["rows"] if r["method"] == "current")
        for row in data["rows"]:
            for mode, metric in row["timing"].items():
                ref = base["timing"][mode]
                overhead = row["prepare_cold_s"] + row["reset_cold_s"] + metric["first_call_s"]
                base_overhead = base["prepare_cold_s"] + base["reset_cold_s"] + ref["first_call_s"]
                saving = ref["median_s"] - metric["median_s"]
                break_even = max(1, 1 + math.ceil((overhead - base_overhead) / saving)) if saving > 0 else None
                flat.append(
                    dict(
                        device=item["device"],
                        key=item["key"],
                        groups=",".join(item["groups"]),
                        **data["case"],
                        method=row["method"],
                        mode=mode,
                        arrivals=data["arrivals"],
                        median_ms=metric["median_s"] * 1000,
                        speedup=ref["median_s"] / metric["median_s"],
                        first_call_ms=metric["first_call_s"] * 1000,
                        prepare_cold_ms=row["prepare_cold_s"] * 1000,
                        reset_cold_ms=row["reset_cold_s"] * 1000,
                        one_use_ms=(data["build_s"] + overhead) * 1000,
                        reuse_100_ms=(data["build_s"] + overhead + 99 * metric["median_s"]) * 1000,
                        warm_saving_break_even_runs=break_even,
                        query_array_bytes=row["query_array_bytes"],
                        cursor_bytes=row["cursor_bytes"],
                        source_host_bytes=data["source_host_bytes"],
                        **row["preparation"],
                    )
                )
    device_pairs = 0
    for key, devices in pairs.items():
        if "cpu" in devices and "gpu" in devices:
            device_pairs += 1
            for field in ("schedule_sha256", "count_sha256", "arrivals"):
                if devices["cpu"][field] != devices["gpu"][field]:
                    failures.append(f"CPU/GPU {field} mismatch: {key}")
    with (directory / "query_summary.csv").open("w", newline="") as handle:
        if flat:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    if flat:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for i, device in enumerate(("cpu", "gpu")):
            for j, mode in enumerate(("query", "synapse")):
                ax = axes[i, j]
                for method in METHODS:
                    rows = sorted(
                        (
                            r
                            for r in flat
                            if r["device"] == device
                            and r["mode"] == mode
                            and r["method"] == method
                            and "schedule" in r["groups"].split(",")
                        ),
                        key=lambda r: r["k"],
                    )
                    if rows:
                        ax.plot([r["k"] for r in rows], [r["median_ms"] for r in rows], "o-", label=method)
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlabel="Events stored per source (K)",
                    ylabel="Rollout (ms)",
                    title=f"{device.upper()} / {mode}",
                )
                ax.grid(alpha=0.25)
                if ax.lines:
                    ax.legend()
        for ext in ("png", "svg"):
            fig.savefig(directory / f"query_schedule.{ext}", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for i, device in enumerate(("cpu", "gpu")):
            for j, layout in enumerate(("independent", "shared")):
                ax = axes[i, j]
                for method in METHODS:
                    candidates = [
                        r
                        for r in flat
                        if r["device"] == device
                        and r["layout"] == layout
                        and r["mode"] == "synapse"
                        and r["method"] == method
                        and "scaling" in r["groups"].split(",")
                    ]
                    by_size = {}
                    for row in sorted(candidates, key=lambda r: r["n"]):
                        by_size.setdefault(row["n"] * row["m"], row)
                    if by_size:
                        sizes = sorted(by_size)
                        ax.plot(sizes, [by_size[size]["median_ms"] for size in sizes], "o-", label=method)
                ax.set(
                    xscale="log",
                    yscale="log",
                    xlabel="Connections (N*M)",
                    ylabel="Query + ExpSyn (ms)",
                    title=f"{device.upper()} / {layout}",
                )
                ax.grid(alpha=0.25)
                if ax.lines:
                    ax.legend()
        for ext in ("png", "svg"):
            fig.savefig(directory / f"query_scaling.{ext}", dpi=150)
        plt.close(fig)
    lines = [
        "# 预定事件查询第一版实测",
        "",
        "四种查询方式使用相同 spike 和连接，对照现有 NetStim.event_count。实验只包含事件查询、乘权归约与 ExpSyn 电导推进，未包含完整 Cell 电压求解或 Network 接入。",
        "",
        f"记录创建于 {manifest['created_utc']}。成功 {len(cases)} 个设备/场景，CPU/GPU 输入与计数配对 {device_pairs} 组；记录或配对失败 {len(failures)} 项。",
        "",
        f"窗口 {protocol['duration_ms']} ms，dt={protocol['dt_ms']} ms，float{protocol['precision']}，预热 {protocol['warmup']} 次，测量 {protocol['repeat']} 次。每方法清理编译缓存，GPU 同步等待；稳态表不包含准备和首次编译执行。",
        "",
    ]
    lines += ["## 主要比较", ""]
    for device in ("cpu", "gpu"):
        selected = [x for x in flat if x["device"] == device and x["mode"] == "synapse"]
        if selected:
            keys = sorted({x["key"] for x in selected})
            wins = {method: 0 for method in METHODS}
            for key in keys:
                winner = min((x for x in selected if x["key"] == key), key=lambda x: x["median_ms"])
                wins[winner["method"]] += 1
            lines.append(
                f"- {device}：{len(keys)} 个场景中，synapse 最小中位数的计数为 "
                + "、".join(f"{m} {wins[m]}" for m in METHODS)
                + "。这是本矩阵的描述，未作噪声显著性检验。"
            )
            schedule = [x for x in selected if "schedule" in x["groups"].split(",")]
            if schedule:
                longest = max(x["k"] for x in schedule)
                rows = [x for x in schedule if x["k"] == longest]
                lines.append(
                    f"- {device}，K={longest}、窗口内 {rows[0]['arrivals']} 个到达事件："
                    + "、".join(f"{x['method']} {x['median_ms']:.3f} ms" for x in rows)
                    + "。"
                )
    lines += ["", "## 环境与验证", ""]
    environments = {}
    cpu_info = manifest.get("cpu_info", "")
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in cpu_info.splitlines() if line.startswith("model name")), "未记录"
    )
    lines.append(f"CPU 型号：{cpu_model}。GPU 为 worker 实际选择设备，物理设备编号见 manifest 协议。")
    lines.append("")
    for item, data in cases:
        environments.setdefault(item["device"], data["environment"])
    for device, env in environments.items():
        lines.append(
            f"- {device}: {env['device_kind']}；JAX {env['versions']['jax']}，BrainState {env['versions']['brainstate']}，BrainUnit {env['versions']['brainunit']}。"
        )
    lines += [
        "",
        "每个成功 worker 已逐步逐连接比较全部计数，逐步比较全部目标的 ExpSyn 电导；各候选在同一设备上对照 current，计数必须完全相同，电导 rtol=3e-5、atol=1e-7 uS。跨设备检查相同 source 和计数的 SHA256，不把这项检查宣称为跨设备电压一致性。",
        "",
        "## 稳态运行时间",
        "",
        "单位 ms，均为完整窗口的中位数。query 仅查询；synapse 为查询、乘权归约和 ExpSyn 的统一事件后精确衰减。",
        "",
        "| 设备 | N/M/K | 布局/活动/扇出/delay/block | 模式 | current | scan | cursor | bucket |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item, data in cases:
        c = data["case"]
        methods = {r["method"]: r for r in data["rows"]}
        for mode in ("query", "synapse"):
            nums = " | ".join(f"{methods[m]['timing'][mode]['median_s'] * 1000:.3f}" for m in METHODS)
            lines.append(
                f"| {item['device']} | {c['n']}/{c['m']}/{c['k']} | {c['layout']}/{c['pattern']}/{c['fanout']}/{c['delay']}/{c['block']} | {mode} | {nums} |"
            )
    lines += [
        "",
        "## 准备、内存与重用",
        "",
        "CSV 的 one_use_ms 包含源构建、冷准备、初始游标和首次执行；reuse_100_ms 再加 99 次稳态运行。break-even 在候选稳态更快时估算需要多少次重用才能抵消冷启动差额，未计入运行间修改参数导致的重建。",
        "",
        "prepare_cold_ms 拆为主机校验/收集、设备量化冷执行、设备到主机、桶排序、上传与打包。量化冷执行包含算子编译和输入传输，不是纯算术时间。current 的常量时间表转入编译程序的成本留在首次执行中。",
        "",
        "query_array_bytes 与 cursor_bytes 是持久设备调度数组的精确字节数；source_host_bytes 是源时间表和 mask。arrival_table_bytes 和 gathered_source_bytes 记录准备中间数组的逻辑大小，不能相加称为峰值。bucket 持久存储为 O(E+T+block)，但准备仍有 O(CK) 临时工作；allocator 统计保留在 JSON，不能解释为每内核峰值。",
        "current 的 query_array_bytes=0 表示没有新增显式调度数组，不代表没有设备内存：实际生产路径的时间表常量包含在编译程序中，此列不统计 executable 常量。",
        "",
        "代表场景的 synapse 成本（ms）；设备调度列只列显式数组和游标，单位 KiB：",
        "",
        "| 设备 | N/M/K/活动 | 方法 | 冷准备 | 首次执行 | 一次使用 | 100次重用 | 设备调度 KiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in flat:
        representative = (row["n"], row["m"], row["k"]) in ((1, 10, 10), (100, 100, 10), (10, 10, 10000)) or row[
            "pattern"
        ] == "burst"
        if (
            representative
            and row["mode"] == "synapse"
            and row["block"] == 128
            and row["layout"] == "independent"
            and row["fanout"] == 1
        ):
            memory_kib = (row["query_array_bytes"] + row["cursor_bytes"]) / 1024
            lines.append(
                f"| {row['device']} | {row['n']}/{row['m']}/{row['k']}/{row['pattern']} | {row['method']} | {row['prepare_cold_ms']:.2f} | {row['first_call_ms']:.2f} | {row['one_use_ms']:.2f} | {row['reuse_100_ms']:.2f} | {memory_kib:.2f} |"
            )
    lines += [
        "",
        "## 解释边界",
        "",
        "- scan 对照用于分离预计算收益。cursor 与 bucket 的数组为动态 JIT 参数，current 保留实际生产源码的常量处理；这是接口候选实验，不是完全相同 HLO 的算法复杂度证明。",
        "- cursor 按连接推进，仍每步检查连接；同一步多重事件会增加批量循环。bucket 按块读紧凑区间，再形成每连接计数；后续稀疏直接投递可能有不同结果。",
        "- burst 是显式受控时间表（每源每 10 ms 同时到达 32 条），由原有 NetStim.event_count 消费；dense 在同一默认 100 ms 窗口具有相同事件数但均匀分布。它们不是 NetStim 标准参数能直接表达的同一种随机过程。",
        "- query 与 synapse 独立编译，不能相加为完整模型的时间占比。这里的 ExpSyn 精确衰减消费者不复刻 Cell solver，生产加速需要下一阶段验证。",
        "",
        "## 复现",
        "",
        "复现实验时读取本批次 manifest 的协议并重新确认执行清单；公共入口提供参数化命令。",
        "",
        _document_link("方案", ROOT / "docs/design/synapse/proposals/event-delivery-optimization.md", report)
        + " · "
        + _document_link("固定版本参考", ROOT / "docs/design/synapse/references/scheduled-event-delivery.md", report)
        + " · "
        + _document_link("实验接口", ROOT / "docs/design/synapse/current/experimental-scheduled-events.md", report)
        + " · "
        + _document_link("驱动与命令", HERE / "README.md", report)
        + " · "
        + _document_link("已维护结果索引", HERE / "RESULTS.md", report),
        "",
    ]
    if failures:
        lines += ["## 未通过记录", ""] + [f"- {x}" for x in failures] + [""]
    if manifest.get("revisions"):
        lines += [
            "## 测量版本与边界修正",
            "",
            "原始 72 个 worker 完成后，dense 的间隔由 100/K 改为 100/(K+1)，避免末尾事件取整到窗口外；仅重测受影响的 CPU/GPU 两个 worker，原记录保存在 superseded/。修正后 dense 与 burst 的窗口到达数均为 320000。",
            "",
            "最终实验接口追加了时钟数值范围检查（包括 float32 大绝对步号），原矩阵全部满足该条件。query/reset 的 AST 与原测量版本完全相同；其他场景的准备时间未为这项额外主机校验重测。manifest 的每条记录标明 source_snapshot，baseline 与 corrected 两份完整测量源码及哈希随 artifacts 保留，protocol.sources 对应原始扫描版本。",
            "",
        ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))
    summary = dict(cases=len(cases), device_pairs=device_pairs, failures=failures, rows=len(flat))
    (directory / "query_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    """Export a completed or partial benchmark directory.

    Parameters
    ----------
    argv : list of str or None
        CLI arguments.

    Returns
    -------
    int
        One when records or device comparisons failed, otherwise zero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).parent / "artifacts" / "queries")
    parser.add_argument("--report", type=Path, help="Output Markdown; defaults to the input artifact directory")
    args = parser.parse_args(argv)
    result = export_report(args.directory, args.report)
    print(json.dumps(result, ensure_ascii=False))
    return int(bool(result["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
