# 完整模型四层对照

原始运行材料仅保存在运行者本地的 artifacts，未随 Git 提供；本页的正文与表格随仓库维护。原始目录、日志或图像链接仅为可选核查入口，缺失时不影响读取已保存的数据。

本实验用同一批次的完整 Cell/Network 运行，比较无突触、挂载突触、连接静默和正常输入四种配置。它回答加入这些结构后整段运行时间怎样变化；独立投递微基准不作为完整运行的阶段占比。全部样本统计见 [完整数据表](scheduled-event-controls-tables.md)，前两轮算法比较见 [直接投递结果](scheduled-event-delivery.md)。

## 四组模型与计时

| 对照 | 逻辑 cell 数 | 独立 ExpSyn 状态数 | 连接行数 | 输入 |
| --- | ---: | ---: | ---: | --- |
| A 无突触 | 100 | 0 | 0 | 无 |
| B 突触无连接 | 100 | 10000 | 0 | 无 |
| C 连接静默 | 100 | 10000 | 10000 | 保留每源 10 条记录，发放时间整体后移 110 ms |
| D 正常输入 | 100 | 10000 | 10000 | 每源 10 条记录、10 ms 间隔、随机相位 |

N=100 由一个 Cell 的 population 轴表达。每个逻辑 cell 为一个 CV 的被动膜，初始电压及漏电反转电位均为 -65 mV；形态长 20 um、半径 10 um，漏电导密度 0.1 mS/cm²。ExpSyn 的 tau=2 ms、反转电位 0 mV，每连接权重 0.001 uS，零延迟，一次批量连接声明。C 与 D 具有相同数量和形状的源表及相同路由；B/C 的电导始终为零。D 的 100000 条源记录中，99990 条量化到本窗口的 step 0–3999；另 10 条量化到 step 4000，属于下一个窗口，不能按 100000 次本窗口到达计算吞吐。A 是静息被动细胞的基础成本，不能作为复杂主动细胞的通用基线。

模拟生物时间 100 ms，dt=0.025 ms，共 4000 步，float32，seed=7。四组使用同一个 JIT runner，通过 brainstate.transform.for_loop 调用真实 Network.update。计时返回最终电压和电导；A 的电导输出为空数组。完整轨迹另行采集做校验，正式计时不包含完整记录输出，也不计 Network.run 结果表物化。

每个配置分别清理编译缓存、构建并准备、记录首次执行，然后预热两次，同步计时五次。每次运行前 reset 并等待完成，reset 耗时另列。CPU/GPU 各做三个独立进程轮次；每个 C/D worker 最多 production 加一个候选，因此 production 在 C/D 各有六个进程，direct/padded 各三个。A/B 各三个进程。候选加速使用同一 worker 中的 production，跨控制组差值则每轮先聚合对应中位数再相减。

CPU 为 Intel Xeon Platinum 8358P，worker 可用 128 个逻辑 CPU；GPU 为 A100 SXM4 80GB、物理索引 7，开始时空闲。Python 3.11.15、JAX/JAXlib 0.10.1、brainstate 0.5.4、brainunit 0.3.0。没有搜索 CPU 线程数、其他 GPU 或更大规模。

## 实测结果与选择

36 个 worker 全部通过，得到 60 条计时记录、16 组方法汇总。下表为三轮进程中位数的中位数，单位是完整 4000 步的墙钟 ms。

| 对照 | 方法 | CPU ms | GPU ms |
| --- | --- | ---: | ---: |
| A 无突触 | production | 8.024 | 58.375 |
| B 突触无连接 | production | 322.622 | 131.898 |
| C 连接静默 | production | 475.508 | 143.238 |
| C 连接静默 | direct | 331.929 | 226.356 |
| C 连接静默 | padded | 319.573 | 135.426 |
| D 正常输入 | production | 474.901 | 143.089 |
| D 正常输入 | direct | 333.335 | 301.909 |
| D 正常输入 | padded | 320.748 | 138.341 |

CPU 正常输入下，direct 配对加速 1.428×，padded 为 1.480×；padded 的三个进程中位数范围 320.082–323.218 ms，均低于 direct 的 332.196–334.027 ms。本配置下 padded 最快，但不能推为所有表长度、布局或存储预算下的最优。优化后的完整模型已接近 B 的运行成本；B−A 的同轮差值中位数为 314.598 ms，说明剩余成本首先需要在挂载独立突触后的机制与求解结构中定位。该差值不能区分电导更新、point 电流归约、投影或内存访问各自占多少。

CPU 静默扫描已需约 475.5 ms。增加静默连接的 C−B 中位数为 152.464 ms，而激活输入的 D−C 扫描差值只有 0.543 ms，且各轮为 -1.431–11.709 ms。有一个静默 production 进程中位数为 450.311 ms，低于其余约 475 ms，全部样本保留；不根据这一小差值声称测得精确的实际事件处理成本。padded 的 C−B 为 -2.950 ms，同样只是不同完整程序的比较，不存在“负的投递耗时”。

GPU 正常输入下，padded 配对加速仅 1.034×（约少用 3.3% 墙钟），direct 为 0.474×（耗时约为扫描的 2.11 倍）。direct 在静默时已比 B 增加约 94.493 ms，实际激活输入再增加 75.516 ms；其不适合作为本配置的 GPU 默认。保留生产扫描，固定窗口反复复用且存储允许时可手动尝试 padded。CPU 从改算法获得的收益更大，但本规模完整运行仍是 GPU 更快：padded 为 138.341 ms，对比 CPU 的 320.748 ms。

正常输入的首次使用成本约 14.1–15.6 秒，远大于一次稳态运行；CPU/GPU padded 的显式计划均约 1351.56 KiB，direct 为 836.86 KiB。这些数组不含源表和编译器内部存储。稳态 reset 约 0.4–1.3 ms/次，独立列出；完整冷成本、每步摊销和各轮范围见 [数据表](scheduled-event-controls-tables.md)。

## 如何解释差值

- B−A 表示挂载 10000 个突触后整个模型增加的运行成本，包含机制状态、求解和归约结构变化，不能单独称为电导衰减耗时。
- C−B 表示添加静默连接和预定调度后的整体变化；对 direct/padded 来说还包括实验适配器的执行结构。
- D−C 表示从没有窗口内到达变成正常输入后的整体变化。它不包含 C 中已经存在的查询、路由和空桶开销，不能称为全部 event 时间。

三个差值都是不同完整程序之间的对照，编译器融合、静默常量优化、状态轨迹及系统负载可能变化。差值可以为负；小于进程间波动的变化不支持更细的归因。每步 µs 仅为整段墙钟除以 4000，不是单独调用一步的延迟。

这些数据不能与第二轮的“单独投递 4000 步”或“投递＋ExpSyn 消费者”相减。后两者输出校验值、没有膜电位积分，编译图和消费者与完整模型不同。精确的内核独占时间仍需在完整模型中结合 profiler，不能用控制组差值替代。

## 正确性与失败记录

结构校验检查实际突触状态数及连接行数；source table 形状、源时间哈希、逐步目标输入哈希和到达数在方法/设备/轮次间核对。正常输入以生产量化表达式独立构造目标驱动参考；静默输入要求全零，B/C 电导要求全零，所有轨迹要求有限值。每个候选的逐步输入、电导及膜电位与 production 对照，并验证 reset 后重复运行的最终结果在 float32 容差内一致：rtol=3e-5，电压 atol=1e-4 mV，电导/输入 atol=1e-7 uS；实际误差写入原始记录。

生产 float32 时钟逐次累加 dt，在本配置 4000 步后为约 100.0038 ms。结束时间量化后必须对应第 4000 步，偏差小于半个 dt；这与要求时钟精确等于 host 的 100 ms 不同。逐步驱动校验独立验证事件没有错步，不能仅凭结束时钟通过判断事件正确。

首次尝试使用了过严的绝对时间断言，7 个 worker 被拒绝；停止后另有 29 个未执行，记录保存在 artifacts/controls/initial_attempt。增加长 float32 时钟回归测试并修正 benchmark 校验后，重跑后，GPU 活跃输入的逐位 reset 结果检查又拒绝了约 1.43e-6 mV 的重复运行电压差；该批次 15 个通过、2 个失败、19 个未执行，保存在 reset_bitwise_attempt。重复运行改用与算法等价性相同的 float32 容差，并增加接受舍入差、拒绝明显残留状态的测试。matched 从修正后的同一源码快照完整重跑；生产时钟及投递算法没有改动，前两次尝试均不进入匹配统计。

最终 matched 的输入与电导最大误差均为 0，候选相对 production 的逐步电压最大误差为 4.0054e-5 mV；reset 后重复运行的电压最大差为 2.8133e-5 mV，电导差为 0。36 个 worker 最长约 65.4 秒，均低于 120 秒上限。实施至报告约 62 分钟，低于两小时总预算；执行与验收明细保存在 artifacts/controls/execution.json 和 verification.json。

代码回归通过 33 个测试、50 个子场景；测量驱动及两个报告模块的语句与分支综合覆盖率为 98.78%。JAX 0.8.0 上另有 8 个相关测试通过。新增边界覆盖无突触空输出、静默零输入、长 float32 时钟、reset 舍入与残留状态拒绝，以及错配配置/缺失 worker/失败样本/跨设备输入哈希不一致。后续建议的多 CV 主动模型与 live/scheduled 混合模型测试由 proposal 的有限模型回归阶段维护，本轮不提供这些场景的性能结论。

## 复现与范围

命令见 [benchmark README](../README.md#complete-model-controls-abcd)。[测量驱动](../delivery_benchmark.py) 的 controls 阶段复用现有模型构造和实验 Cell；[报告生成器](../controls_report.py) 在输入 artifacts 中保存原始/汇总 CSV、差值 CSV、验证 JSON、PNG/SVG 和生成数据表。本页及已审阅数据表随 Git 维护，默认重跑报告不会覆盖它们。

原始有效批次位于被忽略的 benchmarks/performance/synapse_events/artifacts/controls/matched，每个 worker 保留日志、源码快照、全部五次计时、构建/准备/编译首次执行/reset 成本和显式计划存储。执行时间与初次停止原因保存在上一层 execution.json。报告输入指定 matched；将失败的旧源码批次混入时，报告会返回失败而非静默拼接。

本轮只评价单 CV 被动模型、10000 个独立突触及固定 NetStim 调度。共享布局、多 CV 主动膜、live 阈值事件、训练循环和生产窗口续建不在该四层对照中。后续执行方向由 [proposal](../../../../docs/design/synapse/proposals/event-delivery-optimization.md#完整模型四层对照) 维护。

## 原工作流命令记录

以下保留原 README 的运行命令与示例，包含运行者的解释器路径和设备选择。它们不代表各命令均已执行，实际执行范围以本页实测记录为准，也不构成重新运行授权。其他用户使用公共 README 的参数化入口。

```bash
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.delivery_benchmark \
  --phase controls --devices both --gpu 7 --rounds 3 \
  --warmup 2 --repeat 5 --timeout 120 --budget-seconds 3600 \
  --out benchmarks/performance/synapse_events/artifacts/controls/matched

/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.controls_report \
  benchmarks/performance/synapse_events/artifacts/controls/matched
```


## 测量源码标识

批次创建时间：`2026-09-08T15:06:52.091055+00:00`。以下为 `controls/matched` manifest 中保存的测量源码 SHA-256；代表该批次，后续修正/补测的适用范围以正文说明为准。当前仓库中的报告代码和文档可能已更新，不能将当前工作区视为原始测量快照。

| 测量源码 | SHA-256 |
| --- | --- |
| `benchmarks/performance/synapse_events/delivery_benchmark.py` | `e5832b23c1bd0c3ac884296aece2601fe25dea29e8a2f4dddb79ad316e6b060b` |
| `braincell/experimental/scheduled_events.py` | `7dba07e0e441ae37b03cfcd6d4001d0ad3d6fb9bac83f869814f0bc1c17a9279` |
| `braincell/experimental/scheduled_delivery.py` | `e903b9873f8e583452f53cdc8e1a2fbb557b03780939fae7d7d72464bab24456` |
| `braincell/experimental/scheduled_cell.py` | `212c7c6d3ca66fd32836f06255317b28fa583d201fd98f5efa939ea9af66e341` |
| `benchmarks/performance/synapse_events/query_benchmark.py` | `4173681e3e295ea058e990bbeb2b442623c5e7bb031230f7fb713c1645c257ed` |
| `benchmarks/performance/synapse_events/benchmark.py` | `add520a3d688a29514bc0a5d1ce5e79f7055a6596afd5cbbf15823e3959c3972` |
| `braincell/_multi_compartment/cell.py` | `82887dd7dadbad6d19cb06ad8a25fb718ebdef3e5afec5e454413f4620a02e8c` |
| `braincell/network/event.py` | `40c1288eebdb05f27f533bd455f4c5902353b0649fb5d1b5ec8ce97de8bfbccb` |
| `braincell/network/engine.py` | `81455f117238259e1d59d5345ae976df424fc365bd4149b8814ff3330416aafd` |
