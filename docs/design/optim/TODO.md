# Optimization TODO

## 当前需要推进的事项

参数学习的协作入口。[全局 TODO](../TODO.md) 管理宏观依赖，[Design 规范](../AGENTS.md) 定义文档分工和状态。

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| 可塑性参数训练与 weight_initial | 讨论中 | 在 Network 运行时契约基础上确定参数绑定、状态初值与梯度验收 | [训练提案](proposals/connection-plasticity.md)、[Network 可塑性](../network/proposals/connection-plasticity.md) |
| plateau、SGDR、perturb 自动恢复 | 讨论中 | 确定 effective-LR、恢复状态和比较协议 | [训练恢复](proposals/training-recovery.md) |
| Cell 初值与 cable 参数 owner | 待讨论 | 明确构造转换、缓存及初始化依赖 | [后续方向](proposals/roadmap.md#cell-初值与-cable-参数) |
| 公共训练协议与稳定 grouping | 待讨论 | 确定复用边界、持久化和 ownership | [后续方向](proposals/roadmap.md#公共训练协议与分组) |
| 多 CV、多 population、GPU 与 checkpoint 验证 | 待讨论 | 确定组合矩阵和公平测量协议；已有单项结果不代替组合验证 | [验证缺口](proposals/roadmap.md#验证缺口) |
| rollout 内更新参数 | 待讨论 | 定义梯度对应的参数历史 | [研究方向](proposals/roadmap.md#rollout-内更新参数) |

## 阅读入口

optim 是参数学习的问题域；公共参数映射位于 `braincell.trainable`，优化算法由 BrainTools 等提供。

| 想知道什么 | 阅读位置 |
| --- | --- |
| Channel、Ion、Synapse 等到底能学什么 | [参数支持度](current/parameter-support.md) |
| 如何注册、分组、共享、读写参数 | [公共 API](current/api.md) |
| 参数依赖、reset、materialization、事件和队列如何连接 | [架构](current/architecture.md) |
| 已实现的 RTRL、训练、诊断和 OED 实验如何调用 | [实验工作流](current/experimental-workflows.md) |
| 还准备实现什么、缺什么验证 | [路线图](proposals/roadmap.md) |
| 可塑性与恢复策略的候选方案 | [Network 可塑性](../network/proposals/connection-plasticity.md)、[可塑性训练](proposals/connection-plasticity.md)、[训练恢复](proposals/training-recovery.md) |
| 精度、训练收敛、耗时、内存的已有证据 | [结果导航](#实验结果) |
| 原理、方法选择和外部研究 | [参考导航](#理论与方法) |

## 状态口径

Current 描述工作区实现：实验核心在 `braincell.experimental`，使用流程在 `examples/optim/`，精度验证和性能测试分别在 `validation/optim/` 与 `benchmarks/`。
提交进度沿用 [项目进度](../TODO.md#进度口径)；Synapse/Network 扩展已随 `5f90f69` 提交，
CPU 回归及示例验收见 [验证记录](current/results/synapse-network-learning.md#提交验收)。

| 标记 | 含义 |
| --- | --- |
| 公共接口已实现 | 当前 braincell 代码存在接口；支持边界和测试证据另列 |
| 实验代码已实现 | 实验命名空间或工作流中的可调用实现；以对应 Current 描述的入口为准 |
| 待设计 | 尚需决定的候选方向 |
| 暂不支持 | 当前接口拒绝或尚未接入 |

## 能力地图

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| Channel、Ion、Synapse 参数 source 与映射 | 公共接口已实现 | 签名发现；不是科学参数白名单 |
| Connection weight、检测器 threshold | 公共接口已实现 | threshold 的事件路径使用代理梯度 |
| Cell／Network roots 管理与聚合 | 公共接口已实现 | Network 按对象身份去重，不复制根 |
| direct／scale／parameterized、分组、单位和 reset | 公共接口已实现 | 统一 binding 主链 |
| 完整 RTRL、BPTT、敏感度诊断 | 实验代码已实现 | 固定参数 rollout；不能切断跨 Cell 敏感度 |
| 数据、loss、分阶段拟合、搜索、archive、OED | 实验代码已实现 | 不发布 Dataset／Trainer／checkpoint 等公共训练类型 |
| 可塑性规则及动态 weight 初始化 | 待设计 | 与现有静态 weight 参数区分 |
| plateau／restart／perturb 自动恢复 | 待设计 | 已实现观测与历史 archive 不等于已实现 controller |
| Connection delay 可训、Cell.V_init／cable trainable owner | 暂不支持 | 静态配置能力不等于参数训练能力 |

## 实验结果

性能实验及解释它所需的正确性证据在对应 benchmark 的 results 维护；独立数值与训练验证保留验证职责。
实验条件不同的数字不能直接比较；每页记录来源、口径、复查入口和未验证范围。

- [参数学习示例](current/results/parameter-learning.md)：Channel、Ion、单突触单参数教学拟合。
- [突触与网络](current/results/synapse-network-learning.md)：事件、自连接、双向 population 的正确性及训练验证。
- [独立网络 CPU 计时](../../../benchmarks/performance/optim_gradient_scaling/results/synapse-network-cpu.md)：不同参数数与轨迹长度的梯度耗时和内存。
- [BPTT/RTRL scaling](../../../benchmarks/performance/optim_gradient_scaling/results/bptt-rtrl-scaling.md)：历史多 CV、CPU、A100 和 Adam 一致性。
- [Batch 与吞吐](../../../benchmarks/performance/parameter_fitting/results/batch-size-and-throughput.md)：batch、candidate lanes、GPU 容量和训练质量。
- [拟合与可辨识性](current/results/fitting-and-identifiability.md)：multi-start、优化器对照、诊断、FIM 和 ensemble。

“接口可调用”“梯度存在”“拟合成功”“参数唯一可辨识”“性能占优”是不同结论。
多 CV 与多 population 分别已有证据，不合并为其任意组合已经实测。

## 理论与方法

- [BPTT 到 RTRL](references/bptt-to-rtrl-neuron-derivation.md)：链式法则、初始化、online 与 e-prop。
- [Solver 梯度](references/staggered-solver-gradient-analysis.md)：DHS、机制更新和离散程序导数。
- [电压与 spike 拟合](references/voltage-and-spike-parameter-fitting.md)：loss、评价与 curriculum 方法。
- [训练诊断](references/modular-training-diagnostics.md)：诊断解释、archive 和 region 方法。
- [刺激与可辨识性](references/stimulus-design-and-identifiability.md)：方法原理和数据隔离。
- [Jaxley 参数模型](references/jaxley-parameter-model.md)：外部实现取舍。

## 文档对齐

相关教程：
[Channel](../../../examples/optim/parameter_learning/channel_learning.ipynb)、
[Ion](../../../examples/optim/parameter_learning/ion_learning.ipynb)、
[Synapse／Network](../../../examples/optim/parameter_learning/synapse_learning.ipynb)。
提交前检查遵循 [仓库约定](../../../AGENTS.md#design-code-and-examples)。

已知 JAX 0.10.1 Ion 精度切换/执行上下文问题，以及覆盖率 C tracer 的原生崩溃记录，
见 [验证限制](current/results/synapse-network-learning.md#验证与限制)。

旧 [P0 计划](../../specs/2026-08-29-trainable-parameter-implementation.md) 仅供历史追溯。
