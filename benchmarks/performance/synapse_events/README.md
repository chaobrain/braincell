# NetStim 突触事件性能测试

本目录比较 `NetStim → Connection → ExpSyn → Cell` 的构造、准备、事件查询、直接投递和完整模型运行成本，帮助判断事件表长度、输入密度、声明方式和突触布局怎样影响性能。公共入口不依赖已有原始结果；已完成实验的环境与数据见 [结果索引](RESULTS.md)。

## 实验与脚本

| 测量脚本 | 实验目标与支持范围 | 报告脚本 |
| --- | --- | --- |
| [benchmark.py](benchmark.py) | 现有生产路径：smoke、scaling、activity、schedule、delay、declarations、controls；比较独立与共享 ExpSyn 布局 | [report.py](report.py) |
| [query_benchmark.py](query_benchmark.py) | 比较 current、预计算 scan、cursor、bucket；覆盖表长度、密度、扇出、延迟和块大小 | [query_report.py](query_report.py) |
| [delivery_benchmark.py](delivery_benchmark.py) | micro 比较查询与直接投递；full 比较真实 Cell/Network；controls 做四层对照；profile 采集诊断 | [delivery_report.py](delivery_report.py)、[controls_report.py](controls_report.py) |

`*_test.py` 为各脚本的同目录测试。可复用算法来自 `braincell.experimental`，本目录提供实验驱动；[查询接口](../../../docs/design/synapse/current/experimental-scheduled-events.md)、[直接投递接口](../../../docs/design/synapse/current/experimental-scheduled-delivery.md) 和 [设计方案](../../../docs/design/synapse/proposals/event-delivery-optimization.md) 说明算法合同与采用方向。

生产基准中的 N 是逻辑 cell 数，M 是每 cell 的独立 NetStim 源数，每个目标恰好接受 M 个源。独立布局为每源一个 ExpSyn；共享布局为每 cell 一个 ExpSyn。查询实验中的 fanout 另外控制每源连接数。共享与独立是显式构造的两种模型，不能据此自动合并一般模型的独立状态。

生产基准使用单 CV 被动膜与 ExpSyn，物理参数、随机种子和配置矩阵由脚本定义；配置帮助可以在不启动测量的情况下查看。schedule 实验保持窗口内到达不变并增加未来记录；activity 会同时改变事件密度及默认日程长度。查询实验中的 burst 是受控重复时间表，不能视作普通 Cell 阈值发放。

## 依赖与运行前确认

从仓库根目录运行，使用已安装本项目及开发依赖的 Python 环境。测量需要 JAX、brainstate、brainunit，报告使用 NumPy 和 Matplotlib；CPU 实验无需 GPU。GPU 实验需要与本机设备及驱动相匹配的 JAX GPU 安装。设备选择必须在 worker 导入 JAX 前完成，`--gpu` 接收用户选择的设备编号，`--python` 可指定 worker 解释器，默认使用当前 Python；GPU 或双设备协调器必须显式给出 `--gpu`。

```bash
python -m benchmarks.performance.synapse_events.benchmark --help
python -m benchmarks.performance.synapse_events.query_benchmark --help
python -m benchmarks.performance.synapse_events.delivery_benchmark --help
```

启动测量前按 [Benchmark 执行规范](../../AGENTS.md) 列出并确认设备、配置、独立轮数、进程数、首次执行、额外预热、计时重复、校验执行和时间预算。以下是命令模板，尖括号内容须替换为已确认的值；模板和 CLI 默认值均不是运行授权。单独调用一次命令不代表只执行一次模型。

```text
python -m benchmarks.performance.synapse_events.benchmark \
  --suite <suite> --devices <cpu|gpu|both> --gpu <device-index> \
  --warmup <warmup-count> --repeat <timed-count> --timeout <seconds> \
  --out <new-run-directory>

python -m benchmarks.performance.synapse_events.query_benchmark \
  --suite <suite> --devices <cpu|gpu|both> --gpu <device-index> \
  --duration-ms <duration> --dt-ms <dt> \
  --warmup <warmup-count> --repeat <timed-count> --timeout <seconds> \
  --out <new-run-directory>

python -m benchmarks.performance.synapse_events.delivery_benchmark \
  --phase <micro|full|controls|profile> --cases <case-list> \
  --devices <cpu|gpu|both> --gpu <device-index> --rounds <round-count> \
  --warmup <warmup-count> --repeat <timed-count> \
  --timeout <worker-seconds> --budget-seconds <total-seconds> \
  --out <new-run-directory>
```

CPU-only 时可以省略 `--gpu`。前两个驱动一次调用执行一套选定配置；delivery 驱动另有 `--rounds`。`--precision 32|64` 显式选择数值精度，不同精度用不同目录。随机源通过 `brainstate.random` 在 CPU 生成，确保设备间输入可配对。

生产基准通过 `--cells`、`--inputs`、`--rate-hz`、`--number` 等参数调整规模和日程；查询实验支持 smoke/scaling/schedule/activity/fanout/blocks/all。delivery 的 `--methods` 可选择候选；full 自动配对 production，建议一次只选择一个候选以限制重复构建和编译。具体可选配置以对应 `--help` 和脚本矩阵为准。

使用新的输出目录，改变代码、环境或测量协议后不得混用旧结果。前两个驱动的 `--resume` 只用于满足代码、环境、协议一致条件的原实验；delivery 要求新目录。失败、超时和 GPU 不可用会记录状态，不静默回退成 CPU 结果。

## 计时与正确性边界

- 完整模拟通过编译后的 `brainstate.transform.for_loop` 调用真实 `Network.update`；计时保留最终状态，不包括 `Network.run` 结果表物化。完整轨迹校验另外执行，不计入稳态时间。
- 每次正式运行前 reset 并同步，计时结束等待设备计算完成。首次调用包含编译与执行，额外预热和正式计时次数分别报告；controls 另列 reset 成本。
- 查询、查询加归约、投递加 ExpSyn 消费者各自独立编译。消费者使用 `ExpSyn.apply_events` 和精确指数衰减；它们没有完整膜电位积分，不能相加或从完整模型耗时中相减得到阶段占比。
- current/scan/cursor/bucket 生成逐连接计数；direct/padded 直接归约目标输入。固定宽度 padded 有填充与存储限制，超限明确记录不适用。
- NetStim 为预定日程，live Cell spike 走另一条路由与延迟队列。本目录的结果不能代表 live 事件性能，也不能用生产 event backend 切换替代预定查询比较。
- 校验覆盖事件计数、目标输入、状态轨迹、数值有限性和 reset；具体容差与误差必须随结果记录。设备 allocator 统计不等于单个内核独占内存。

## Complete-model controls (A/B/C/D)

`delivery_benchmark --phase controls` 当前提供固定的单 CV 被动模型：100 个逻辑 cell，每个 100 个独立 ExpSyn，每源 10 条记录，模拟 100 ms，dt=0.025 ms。

| 控制组 | 结构与输入 | `--cases` 名称 |
| --- | --- | --- |
| A | 无突触、无连接 | `cell_only` |
| B | 挂载突触，无连接 | `unconnected` |
| C | 保留连接与源表形状，发放整体后移 110 ms | `silent_direct`、`silent_padded` |
| D | 正常输入 | `active_direct`、`active_padded` |

A/B 仅 production；C/D 的每个 worker 配对 production 与所选桶，候选由 case 固定，不额外传 `--methods`。选择全部六个 case 时，每设备每轮有六个进程、十条方法结果；这些数量不包含方法内部的首次执行、预热、计时与额外校验，运行前还须逐项确认。

四组差值表示不同完整模型之间的整体变化。B−A 不是单独突触衰减耗时，D−C 也不是全部 event 时间。时钟检查采用最终量化步与半 dt 偏差界限；重复结果使用数值容差，不能将 float32 的逐位相等作为普遍保证。

## 报告、总结与原始文件

```text
python -m benchmarks.performance.synapse_events.report --input <baseline-run-directory>
python -m benchmarks.performance.synapse_events.query_report <query-run-directory>
python -m benchmarks.performance.synapse_events.delivery_report <delivery-run-root>
python -m benchmarks.performance.synapse_events.controls_report <controls-run-directory>
```

报告只读取已有结果，不运行模型。四个生成器默认分别在输入目录写入 `netstim-baseline.md`、`scheduled-event-queries.md`、`scheduled-event-delivery-tables.md`、`scheduled-event-controls-tables.md`，并保留各自的 CSV、校验 JSON 和图表输出。`report.py --markdown` 或其他报告的 `--report` 可以显式指定输出文件，目标父目录自动创建。

| 位置 | 用途 |
| --- | --- |
| [RESULTS.md](RESULTS.md) | 已保存结果的主题索引 |
| `results/` | Git 维护的完整实测总结；记录实际环境、协议、数据和局限 |
| `artifacts/<run>/` | 本地 manifest、原始 JSON/log、源码快照、生成报告及图表 |

自动生成报告默认不覆盖 Git 总结或 Design。将新结果纳入 `results/` 时，需要审阅证据与适用条件。原始 artifacts 不随 Git 提供，也不要求新用户具备它们；已保存总结中的原始材料链接仅用于可选核查。

报告按 manifest 核查完整性，并保留失败、缺失和不适用记录。非零退出码可能表示数值不匹配、未完成 worker 或源版本混用，应读取报告原因。controls 报告输入必须限定为配置与源码一致的批次；profile 运行不进入稳态汇总。

## Profiling 与测试

production 驱动支持 `--trace`；delivery 支持 `--phase profile` 和 `--phase full --trace-full`。这些都会额外执行模型，需单独确认次数与预算；已有 trace 可通过 `python -m benchmarks.profiling.parse_xplane_trace --help` 查看解析方式。内核时间之和与完整墙钟不同，诊断运行也不作为正常性能样本。

仅报告逻辑的测试使用合成记录，不运行模型：

```bash
python -m pytest \
  benchmarks/performance/synapse_events/report_test.py \
  benchmarks/performance/synapse_events/query_report_test.py \
  benchmarks/performance/synapse_events/delivery_report_test.py \
  benchmarks/performance/synapse_events/controls_report_test.py -q
```

其他 `*_test.py` 包含模型执行及测量驱动检查，不应把全目录 pytest 当成无成本的报告检查。真实测量与额外执行均须遵循运行前确认规范。
