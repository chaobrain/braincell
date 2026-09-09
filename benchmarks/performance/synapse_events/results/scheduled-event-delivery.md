# 预定事件直接投递实测

原始运行材料仅保存在运行者本地的 artifacts，未随 Git 提供；本页的正文与表格随仓库维护。原始目录、日志或图像链接仅为可选核查入口，缺失时不影响读取已保存的数据。

固定窗口的直接投递在 CPU 长表和共享布局中有稳定收益；GPU 的固定宽度方案也在共享布局完整模型中达到 **1.23×** 加速。设备、事件分布和目标状态布局共同决定选择，当前证据不支持一个全局最优方法。[完整数据表](scheduled-event-delivery-tables.md) 保存所有方法、稳态时间、首次成本和排除项；[方案](../../../../docs/design/synapse/proposals/event-delivery-optimization.md) 维护生产接入方向。

## 配置与测量口径

测量日期为 2026-09-08。CPU 为 Intel Xeon Platinum 8358P，进程可用 128 个逻辑 CPU；GPU 为 NVIDIA A100 SXM4 80GB，物理设备 7，开始时空闲。Python 3.11.15、JAX/JAXlib 0.10.1、brainstate 0.5.4、brainunit 0.3.0。CPU/GPU 使用独立进程和同种子输入；未做 CPU 核数或线程数搜索。

本轮实现、验证、测量、诊断和报告的执行记录约 113 分钟，低于约定的 180 分钟上限；时间戳与预算保存在 artifacts/delivery/execution.json。单次模拟的毫秒级耗时与整轮实验墙钟时间是不同指标。

每次模拟 100 ms，dt=0.025 ms，共 4000 步，float32、种子 7。每个方法先清理编译缓存，准备计划、首次执行，再预热 2 次和同步计时 5 次。胜出候选增加两个独立进程复测。稳态统计取各进程中位数的中位数，加速比取**同一进程内候选与基线之比**的中位数；至少三个配对进程均更快且中位加速 ≥1.10 才标为稳定收益。

| 场景 | N 个 cell / 每 cell M 个 NetStim / 每源 K 条记录 | 目标突触状态数 | 窗口内到达数（micro） |
| --- | --- | ---: | ---: |
| small | 1 / 10 / 10 | 10 | 100 |
| large | 100 / 100 / 10 | 10000 | 99990 |
| long | 10 / 10 / 10000 | 100 | 1000 |
| sparse | 10 / 100 / 10，间隔 100 ms | 1000 | 1000 |
| burst | 10 / 100 / 320，同源每次 32 条同刻事件 | 1000 | 320000 |
| dense | 10 / 100 / 320，分散在窗口内 | 1000 | 320000 |
| shared | 100 / 100 / 10，共享目标 | 100 | 99990（布局诊断） |

small/large/long 使用随机相位、10 ms 间隔；窗口右端按生产量化规则排除。N 在完整模型中由一个 Cell 的 population 轴表达。每个 cell 为一个 CV 的被动膜，加 ExpSyn；权重 0.001 uS、tau=2 ms。正式完整模型覆盖 small、large、long、shared；sparse/burst/dense 是投递与突触动力学的微基准。

independent 每连接有独立突触状态；shared 每个 cell 的 M 个源投递到同一个突触状态。这是两种明确构造的模型布局，不能据此自动合并一般模型的独立状态。微基准与完整模型的总工作不同，耗时不能相减为事件占比，也不能把微基准加速直接套到完整模型。无突触、无连接和静默输入的同批次完整模型对照见 [四层对照](scheduled-event-controls.md)。

## 完整模型的选型证据

下表每个候选均有三个独立配对进程。时间单位 ms，不含构建、准备、编译及运行间 reset。

| 设备与布局 | 候选 | 配对 production | 候选稳态 | 配对加速 | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| CPU large independent | direct | 476.516 | 332.706 | 1.43× | 稳定收益 |
| CPU large independent | padded | 451.424 | 319.981 | 1.41× | 稳定收益 |
| CPU long independent | direct | 1208.438 | 11.247 | 107.44× | 稳定收益 |
| CPU shared | direct | 224.799 | 18.452 | 12.22× | 稳定收益 |
| CPU shared | padded | 225.058 | 16.981 | 13.30× | 稳定收益 |
| GPU large independent | padded | 143.543 | 138.376 | 1.04× | 小幅收益，未达推荐门槛 |
| GPU shared | padded | 129.210 | 104.663 | 1.23× | 稳定收益，耗时约减少 19% |

CPU large 的 direct/padded 测量批次中，基线相差约 5%；两者配对收益都约 1.4×，不能据候选绝对中位数宣称 padded 稳定胜过 direct。GPU shared 的 padded 三个进程中位数范围为 104.487–104.782 ms。

小模型仍适合生产扫描路径：CPU small production 为 3.803 ms，direct 为 35.317 ms；GPU small 分别为 91.899 和 188.154 ms。GPU long 完整模型 production 115.306 ms、scan 117.697 ms、direct 211.104 ms；微基准中 scan 的优势没有在这个完整模型中重现。这些劣势候选只做初筛，不标记为三轮稳定结论。

CPU long 的 107× 来自固定窗口仅消费 1000 次到达，而原源表存储 100 万条事件；这是长表、低窗口活动的特定条件。CPU large 微基准 direct 约 6.31×，完整模型约 1.43×，说明投递之外的独立突触状态更新与细胞计算限制了整体收益。

## 准备成本与计划重用

| 配置与候选 | production 首次使用 | 候选首次使用 | 候选计划数组 |
| --- | ---: | ---: | ---: |
| CPU large direct | 14.132 s | 15.358 s | 836.9 KiB |
| CPU long direct | 4.656 s | 4.800 s | 24.8 KiB |
| CPU shared padded | 12.712 s | 13.756 s | 1351.6 KiB |
| GPU shared padded | 13.186 s | 13.892 s | 1351.6 KiB |

首次使用包括模型/源构建、计划准备、初始 reset 和第一次编译执行。长表的百倍稳态加速并不是第一次使用快百倍。按这批配对中位数粗估，GPU shared 的额外冷成本约需 30 次后续运行才能抵消；估算不含运行间 reset，也不是对任意模型的盈亏阈值。

数组字节只计最终显式计划，不含源表、准备时完整连接事件表、排序/压缩临时数据和 executable 常量。direct 消除逐连接计数向量，但仍分配目标输入并更新突触状态。padded 以较高存储换固定形状：large/shared 宽度 42、填充比例约 1.68；dense 宽度 103、比例约 1.29。small/long/sparse/burst 的比例分别为 40/12/16/400，超过约定的 8 倍上限，明确标为不适用，没有截断事件或静默回退。

## GPU trace 能解释什么

另开进程采集 sparse/burst 的 current、bucket、direct，shared 微基准的 current/padded，以及 shared 完整模型 production/padded，共 10 个 trace。诊断运行全部排除出正式计时。原始 XPlane、Trace Viewer JSON、lowered IR 和解析 JSON 均保存在 artifacts/delivery/profile*。

| 诊断场景 | 方法 | GPU 活动事件数（含复制） | GPU 活动时长之和 |
| --- | --- | ---: | ---: |
| sparse micro | current / bucket / direct | 20005 / 39569 / 31569 | 32.47 / 75.10 / 62.63 ms |
| burst micro | current / bucket / direct | 24005 / 46005 / 28325 | 39.52 / 87.65 / 54.81 ms |
| shared micro | current / padded | 20005 / 16005 | 55.19 / 33.99 ms |
| shared full | production / padded | 68013 / 64013 | 137.35 / 109.35 ms |

sparse 的 bucket/direct 各有 4891 次与 while 相关的 MemcpyD2H；burst 分别为 6500/4080 次。trace 同时显示大量约 2 us 的比较和动态切片内核。这支持“动态循环的条件检查、小任务和同步开销抵消稀疏收益”的解释，而不能把所有差额归于扫描算术。

shared full 的 `input_scatter_fusion_2` 累积时间从 29.748 ms 降为 8.235 ms，每步一个调用；其 launch grid 从 79 个 block 减为 1 个。总活动数少了 4000，约少每步一个活动。结合 direct/padded 的执行结构，证据支持固定形状投递减少全连接归约工作。command buffer 融合使部分 trace 丢失细分的 BrainCell scope，因而这里只报告实际内核及活动，不能给出精确的“event 占总时间百分比”。解析器在这些 trace 中使用 Trace Viewer JSON，`device_planes=[]` 不表示没有 GPU 活动。

活动时长之和包含 profiler 影响，可能包含重叠，**不是完整运行的关键路径墙钟时间**；优化是否有效以三轮非 profiling 的配对计时为准。专用 GPU 内核的后续问题因此应聚焦消除桶内主机条件检查及小任务，而不只是继续缩短被扫描的表。

## 正确性、覆盖与证据边界

本轮导出 336 条正式计时记录。微基准逐步计数与加权输入一致，CPU/GPU 源表和计数哈希配对通过。完整模型逐步驱动输入最大误差为 0，电导最大绝对误差 1.863e-9 uS，膜电位最大绝对误差 4.006e-5 mV；允许同一步归约顺序引起的浮点差异。

新增四个测试文件共 19 个测试、39 个子场景通过，另有既有 benchmark 的 12 个测试通过；GPU 10 个核心测试、JAX 0.8.0 CPU 上全部 19 个新增测试通过。新四个源模块语句与分支综合覆盖率为 98.66%。覆盖空桶、重复事件、冲突目标、尾块、异质 delay、半步边界、分段/seek/reset、运行时权重变化、float32/64、资源拒绝和一阶权重/tau 梯度。实际 TrainableManager/Network 路径检查权重、tau 梯度及未连接布局；[接口页](../../../../docs/design/synapse/current/experimental-scheduled-delivery.md) 的编译示例也已执行。

初始 full 大批次在 CPU/GPU 的 large/shared 共 4 个 worker 超过 120 秒，记录保留。随后拆为“production + 一个候选”成功完成 scan/direct/padded，并对关键候选复测；large/shared 的旧 current 适配器和 bucket 没有补测，表中留空。报告程序因保留这四个历史超时而返回非零；除此之外无缺失、输入不一致或数值校验失败。当前 full 默认已收窄为 direct + production，回归测试验证这一配置。

尚未验收多 CV 主动模型、一般 live 与 scheduled 混合网络、自动续建窗口、跨窗口 BPTT、前向/高阶微分，以及生产自动选择器。未运行专用 GPU 内核、源端共享查询或全参数/硬件搜索。当前实验适配器要求固定窗口和一次安装；这些限制见接口页。

## 复现与源码

执行入口为 [delivery_benchmark.py](../delivery_benchmark.py)，命令、依赖、输出和进程限时见 [README](../README.md)。可调用实现为 [直接投递](../../../../braincell/experimental/scheduled_delivery.py) 与 [真实 Cell 适配器](../../../../braincell/experimental/scheduled_cell.py)。所有阶段保存 manifest、worker JSON/log 和源码快照；CSV/PNG/SVG、验证和 profiling 汇总位于被忽略的 `benchmarks/performance/synapse_events/artifacts/delivery/`。

初始 micro 计时后，完整模型增加了不计时的目标输入轨迹核对；后续增加保守 full 默认与 profiling 开关。投递算法与 micro 计时函数保持一致，快照与 AST 核对记录保存在 verification.json。报告生成器默认在输入 artifacts 中重建数据表；本页及[已审阅数据表](scheduled-event-delivery-tables.md) 随 Git 维护，更新需审阅，不由默认报告命令覆盖。

## 原工作流命令记录

以下保留原 README 的运行命令与示例，包含运行者的解释器路径和设备选择。它们不代表各命令均已执行，实际执行范围以本页实测记录为准，也不构成重新运行授权。其他用户使用公共 README 的参数化入口。

```bash
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase micro --out benchmarks/performance/synapse_events/artifacts/delivery/micro_initial
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase micro --methods current,scan,direct,padded --rounds 2 --order 1 --out benchmarks/performance/synapse_events/artifacts/delivery/micro_confirm
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase full --cases large,shared --methods padded --out benchmarks/performance/synapse_events/artifacts/delivery/full_padded
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase full --cases small,long --methods direct --out benchmarks/performance/synapse_events/artifacts/delivery/full_direct
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase profile --cases sparse,burst --devices gpu --methods current,bucket,direct --out benchmarks/performance/synapse_events/artifacts/delivery/profile
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase profile --cases shared --devices gpu --methods current,padded --out benchmarks/performance/synapse_events/artifacts/delivery/profile_shared
python -m benchmarks.performance.synapse_events.delivery_benchmark --phase full --cases shared --devices gpu --methods padded --trace-full --out benchmarks/performance/synapse_events/artifacts/delivery/profile_full_shared
python -m benchmarks.performance.synapse_events.delivery_report benchmarks/performance/synapse_events/artifacts/delivery
```


## 测量源码标识

批次创建时间：`2026-09-08T10:20:47.908975+00:00`。以下为 `delivery/full_initial` manifest 中保存的测量源码 SHA-256；代表该批次，后续修正/补测的适用范围以正文说明为准。当前仓库中的报告代码和文档可能已更新，不能将当前工作区视为原始测量快照。

| 测量源码 | SHA-256 |
| --- | --- |
| `benchmarks/performance/synapse_events/delivery_benchmark.py` | `325c4c42f471d794fce50f4e2eaa55fae4457f21c06da19dcf064086c91e8d3e` |
| `braincell/experimental/scheduled_events.py` | `7dba07e0e441ae37b03cfcd6d4001d0ad3d6fb9bac83f869814f0bc1c17a9279` |
| `braincell/experimental/scheduled_delivery.py` | `e903b9873f8e583452f53cdc8e1a2fbb557b03780939fae7d7d72464bab24456` |
| `braincell/experimental/scheduled_cell.py` | `212c7c6d3ca66fd32836f06255317b28fa583d201fd98f5efa939ea9af66e341` |
| `benchmarks/performance/synapse_events/query_benchmark.py` | `4173681e3e295ea058e990bbeb2b442623c5e7bb031230f7fb713c1645c257ed` |
| `benchmarks/performance/synapse_events/benchmark.py` | `add520a3d688a29514bc0a5d1ce5e79f7055a6596afd5cbbf15823e3959c3972` |
| `braincell/_multi_compartment/cell.py` | `82887dd7dadbad6d19cb06ad8a25fb718ebdef3e5afec5e454413f4620a02e8c` |
| `braincell/network/event.py` | `40c1288eebdb05f27f533bd455f4c5902353b0649fb5d1b5ec8ce97de8bfbccb` |
| `braincell/network/engine.py` | `81455f117238259e1d59d5345ae976df424fc365bd4149b8814ff3330416aafd` |
