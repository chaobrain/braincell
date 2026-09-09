# Synapse and Network Learning Results

## 来源与配置边界

本页保存 2026-09-07 的实验与会话测量，以及 2026-09-08 的提交验收；两组记录分别注明配置。
公共支持见 [参数支持度](../parameter-support.md)，事件导数合同见
[架构](../architecture.md#event-derivatives)，完整 RTRL 仍是实验代码。
历史来源：[Synapse/Network](../../../../specs/2026-09-07-synapse-network-learning.md)、
[双向 population](../../../../specs/2026-09-07-bidirectional-population-learning.md)。

本页保留单/双 Cell 与自连接、A(2)/B(3) 双向 population 的正确性及训练验证。
独立 CPU 计时见 [benchmark 实测](../../../../../benchmarks/performance/optim_gradient_scaling/results/synapse-network-cpu.md)。
不能把它们与 [历史多 CV/A100 scaling](../../../../../benchmarks/performance/optim_gradient_scaling/results/bptt-rtrl-scaling.md) 合成同一配置。
各组都在 rollout 内固定参数；不验证每个 timestep 更新优化器的语义。

## 事件与自连接

既有 autapse 实验在 JAX 0.8.0 和 0.10.1 CPU 上验证 voltage/spike loss、零及 0.1 ms
固定 delay、two-Cell sensitivity 和 carry shape。Notebook 的 float64 比较中，
最大绝对梯度差不超过 5.24e-10；spike loss 的差不超过 3.33e-15。
这是同一 surrogate 图上的 forward/reverse 一致性，不是硬事件时间的有限差分导数。
其完整模型、步长和长度见 [autapse.py](../../../../../validation/optim/gradient_correctness/autapse.py)；
单参数 tau/weight/threshold 教学结果单列于 [参数学习](parameter-learning.md#synapse-与-connection)。

## 双向 Population

A 含 2 个成员、B 含 3 个成员，每成员 1 CV HH；A 接收 ExpSyn，B 接收 Exp2Syn。
每方向 6 个 contact，含汇聚与发散。dt=0.025 ms，共 800 步/20 ms，错开的两次 clamp
使每个成员都发放两次 spike，并在接收非零突触电导后继续放电。

梯度验证共 15 个具名向量根、45 个独立标量坐标：两边 Channel 的 g_max/V_sh、
SodiumFixed.E、突触时间常数/e、检测器 threshold，以及每个 contact 的 weight。
拟合时改成各 population/方向内共享，共 15 个标量根，A/B 不共享；A.weight 指 B 到 A，
B.weight 指 A 到 B。scale 根是无量纲因子，shift/reversal/threshold 根是以 mV 计的偏移。

| Loss | Fixed delays | Delivery |
| --- | --- | --- |
| Joint voltage | Heterogeneous, including zero | scatter |
| A-only voltage | Homogeneous positive | scatter |
| B-only voltage | Heterogeneous, including zero | brainevent |
| Joint spikes | Zero | brainevent |

全部坐标逐项比较，float64 接受界为 atol=1e-8、rtol=1e-7。已执行 Notebook 的 joint voltage
最大绝对梯度差约 3.35e-10。还验证单侧 loss 到另一侧 gmax/threshold 的非零梯度；
仅截断 event 反向时，前向 voltage/spike/conductance 完全相同，但这些跨 population 梯度为零。
共享 gmax 根得到独立 A/B 梯度之和；前缀、双向电压/队列敏感度、编译后 reset/root 更新和
恢复也通过检查。四组组合不是 loss/delay/backend 的全部笛卡尔积。

| 训练方法 | Adam updates | Initial voltage MSE | Final voltage MSE | Final/initial |
| --- | ---: | ---: | ---: | ---: |
| BPTT | 100 | 25.51068369 | 0.10422649 | 0.0040856 |
| Full RTRL | 100 | 25.51068369 | 0.10422649 | 0.0040856 |

使用同一个 synthetic spiking target、相同扰动初值、Adam lr=0.01，MSE 单位 mV squared。
全部 15 根变化；这不证明唯一恢复生成参数。自动化测试允许 200 次更新，要求至少十倍下降，
两个 JAX 环境均通过。实现与测试见
[bidirectional.py](../../../../../validation/optim/gradient_correctness/bidirectional.py)、
[bidirectional_test.py](../../../../../validation/optim/gradient_correctness/bidirectional_test.py)。

## 提交验收

功能提交 `5f90f69`，验收日期 2026-09-08。以下检查在导出的暂存快照上使用 CPU 执行；
验收后仅补充 Design 文字，提交中的代码和示例与受测版本一致。

| 环境 | 检查范围 | 结果 |
| --- | --- | --- |
| Python 3.11.4 / JAX 0.8.0 | Channel、Ion、compute、trainable、Cell、Network、Synapse、三项基类测试及新增训练示例测试 | 1523 passed, 2 skipped |
| Python 3.11.15 / JAX 0.10.1 | Network、Synapse、Network roots、点参数目标及新增训练示例测试 | 175 passed, 2 skipped；另 25 subtests passed |
| Python 3.11.4 / JAX 0.8.0 | synapse_learning.ipynb 全部 6 个代码单元 | 全部通过 |

新增示例测试为 autapse_test.py、bidirectional_test.py 和 synapse_learning_test.py，
分别覆盖反馈梯度、双向 population 和单参数拟合，位置见本页各实验入口。
JAX 0.10.1 本次执行的是所列相关套件；组合 Ion 回归的已有精度问题仍见下方记录。

## 验证与限制

| 历史检查 | 结果 |
| --- | --- |
| JAX 0.8.0 相关 Channel/Ion/runtime/trainable/Cell/Network/Synapse | 1509 passed, 2 skipped |
| Synapse 扩展的覆盖率运行 | 381 passed, 2 skipped；改动行 364/378 = 96.3% |
| JAX 0.10.1 focused Synapse/Network/target/base | 173 passed, 2 skipped；另 8 subtests |
| JAX 0.8.0 双向网络与 delivery | 11 passed |
| JAX 0.10.1 同组测试 | 11 passed |
| 网络、参数聚合、autapse 回归 | 135 passed, 1 skipped |
| Python tracer 的选定覆盖率运行 | 3 passed |
| 新 bidirectional 实验代码覆盖率 | 139/146 = 95.2%；缺七行命令行报告入口 |
| 新稀疏导数实现 | 无未覆盖可执行行 |
| 最新 synapse_learning Notebook | 六个 code cells 执行成功 |

这些覆盖率都不是整仓库覆盖率或正确概率。测试组存在重叠，不相加为独立测试总数。
本次只整理历史结果，没有重新跑测试、Notebook 或 benchmark。

brainevent 的 weight JVP 批处理曾在两个环境报 weight_info 参数错误，已有先失败后修复的
delivery 回归；前向仍用 coomv，精确双线性 JVP 使用 scatter，没有修改依赖文件。

JAX 0.10.1 的组合 Ion float32/float64 测试存在执行顺序/上下文敏感失败：旧 HEAD 特定顺序
也复现五个失败，17 个 manager Ion 测试在新进程通过；旧 HEAD 完整相关运行另有
1488 passed、2 skipped。不能把 focused 通过写成当前完整 JAX 0.10.1 套件无条件全绿。
覆盖率插桩是否为原因未确定，该问题未在本轮修复。

JAX 0.8.0 的一次 C tracer 覆盖率运行在 AD tracing 原生崩溃，不计成功；随后普通测试及
JAX 0.10.1 Python tracer 通过。原因未确定，不以文档重组宣称解决。
GPU 上的新事件网络、多 CV × 多 population、自定义机制以及 rollout 内更新参数尚未验证。

## 复查入口

[已执行 Notebook](../../../../../examples/optim/parameter_learning/synapse_learning.ipynb) 保存表格和图。
在选定依赖环境后，功能检查的已有命令为：

```bash
python -m pytest -q braincell/network/delivery_test.py validation/optim/gradient_correctness/bidirectional_test.py
python -m pytest -q validation/optim/gradient_correctness/autapse_test.py examples/optim/parameter_learning/synapse_learning_test.py
```

这些命令验证功能而不是生成上述独立计时表；构造耗时、整轮测试耗时不能替代梯度内核时间。
