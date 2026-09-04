# Model Optimization Design Overview

## 文档状态

本目录讨论 BrainCell 模型参数优化所需的接口、架构和实验依据。目录名 `optim` 表示
问题域，不对应 `braincell.optim` Python 模块，也不表示 BrainCell 自己实现优化算法。

当前已经实现的首个公共 API 是：在三个 Channel 上选择可训参数，并把低维参数或 latent
函数映射到 runtime 物理字段。数据、loss、搜索、诊断和结果协议仍处于需求或实验阶段，
在形成稳定合同前不预先占用公共类型名。

## 文档导航

### 规范文档

- [API](api.md)：当前公共接口、调用方式、参数和错误合同。
- [Architecture](architecture.md)：参数所有权、binding、materialization 和单位边界。
- [Implementation plan](implementation-plan.md)：实现阶段、文件边界和验收场景。
- [Optimization Experiments](../../../examples/experimental/README.md)：experimental gradient core、正确性、scaling 和训练实验导航。
- [Parameter Learning Experiments](../../../examples/experimental/optim_parameter_fitting/README.md)：Python组合式模型、数据、loss、gradient/non-gradient stage和结果合同。

### References

以下文档提供实验、方法调研和技术分析，不定义公共 API。与规范文档冲突时，以 API、
Architecture 和 Implementation plan 为准。推荐按问题选择阅读，不需要顺序通读。

#### 训练目标与工作流

- [电压轨迹与 Spike-Aware 参数拟合](references/voltage-and-spike-parameter-fitting.md)：设计
  subthreshold/spike loss、mask、curriculum 和成功标准时阅读。
- [模块化训练诊断与优化恢复](references/modular-training-diagnostics.md)：实现 observer、
  archive、plateau、spike-region 或非局部恢复时阅读。
- [优化恢复消融协议](references/optimization-ablation-protocol.md)：比较 scheduler、controller
  和 perturb 策略时使用的固定实验合同。

#### 梯度理论与 Solver

- [BPTT/RTRL 通用理论](references/bptt-to-rtrl-neuron-derivation.md)：通用链式法则、`v/w`
  梯度路径、online/e-prop 边界和 1-CV HH 示例。
- [Staggered Solver 梯度分析](references/staggered-solver-gradient-analysis.md)：当前 DHS 与
  post-voltage 离散程序的一步梯度，以及替换 solver 时的验证边界。

#### 性能

- [Batch Size 与 GPU 吞吐](references/batch-size-and-gpu-throughput.md)：protocol batch、
  candidate lanes、显存、吞吐和统计效率。

#### 外部方法

- [Jaxley 参数模型](references/jaxley-parameter-model.md)：参数选择、sharing 和 simulation
  replacement 的实现取舍。

### 主题所有权

同一概念只在主文档中完整定义，其他文档只提供链接和本地上下文：

| 主题 | 唯一主文档 |
| --- | --- |
| 公共 trainable API 与错误合同 | [API](api.md) |
| ownership、binding、materialization、单位 | [Architecture](architecture.md) |
| voltage/spike loss 与 curriculum | [电压轨迹与 Spike-Aware 参数拟合](references/voltage-and-spike-parameter-fitting.md) |
| diagnostics、archive、region、recovery | [模块化训练诊断与优化恢复](references/modular-training-diagnostics.md) |
| BPTT/RTRL 通用公式 | [BPTT/RTRL 通用理论](references/bptt-to-rtrl-neuron-derivation.md) |
| 当前 solver 的程序导数 | [Staggered Solver 梯度分析](references/staggered-solver-gradient-analysis.md) |
| 测量数据 | 对应实验结果或性能文档 |

## 功能地图

参数拟合工作流可以拆成七个能力域：

| 能力域 | BrainCell 应负责 | 复用或用户负责 | 当前状态 |
| --- | --- | --- | --- |
| 数据集 | 单位、shape、PyTree 兼容性要求 | 文件、切分、增强、DataLoader | 尚未形成 API |
| 可训参数 | 参数选择、共享自由度、稳定状态树 | BrainState `nn.Param` / `ParamState` | P0 已设计 |
| 参数映射 | direct、scale、latent function 到 runtime field | 用户空间函数 | P0 已设计 |
| 损失 | 仿真输出与单位边界的互操作原则 | BrainTools metric 或用户 callable | 实验阶段 |
| 优化器 | 提供原始 ParamState tree | `braintools.optim` 或用户 optimizer | 直接复用 |
| 预采点 | 参数 tree 的批量赋值和候选形状 | Sobol/LHS/Nevergrad/SciPy | 需求阶段 |
| 诊断与评价 | 模型特有 metadata 和结果语义 | 成功规则、profiler、科学结论 | 实验模块已实现 |

BrainCell 不应因为涵盖这个问题域就再实现一套 Trainer。可训参数系统只负责把模型暴露
为一个结构清楚、单位正确、可由 BrainState 求导或被黑盒搜索写入的参数化函数。

## 公共模型

```text
View selection
  -> View.trainable(field=source)
  -> target Cell TrainableManager
      roots: nn.Param / ParamState
      bindings: source -> runtime field
      ParameterSet: optimizer-facing tree
  -> init/reset/run materialization
  -> existing BrainCell simulation

Future: Network.trainables
  -> aggregate target Cell managers
  -> one deduplicated ParameterSet
```

公共 helper 位于 `braincell.trainable`，不是 `braincell.optim`：

```python
view.trainable(
    g_max=braincell.trainable.scale(group_by="all")
)

parameters = cell.trainables.parameters()
states = parameters.states()

optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(states)
```

`braincell.trainable` 负责模型参数化；`braintools.optim` 负责优化算法。两个 namespace 的
职责不能合并。

## 当前范围

P0 只覆盖 multi-compartment `ChannelView` 上 `IL`、`Na_HH1952`、`K_HH1952` 的连续
物理参数，并要求在
`init_state()` 前声明。以下内容不属于首批实现：

- Ion、Synapse 和 Connection weight；
- Network 参数聚合与自动物化；
- cable、Cell initial value 和 topology 参数；
- 初始化后改变 trainable ownership；
- Dataset、loss composition、history、checkpoint 和 convergence 公共类型；
- optimizer、scheduler 或通用搜索算法。

Synapse 与 Connection 不需要另一套参数系统。它们后续实现相同的 `View.trainable()`；
因为 SynapseStore 和 ConnectionStore 都由目标 Cell 持有，binding 仍进入该 Cell 的
`trainables` manager。未来 `Network.trainables` 负责跨 Cell 聚合。

## 设计原则

- 只把真实训练自由度包装为 `nn.Param`，不批量改写所有 runtime 字段类型。
- root 与 materialized runtime value 是两个 view；graph state 只暴露 root。
- direct、shared scale 和任意 latent function 使用同一 binding 主链。
- 物理量保留 `brainunit` 单位，loss 在明确尺度上无量纲化。
- 参数选择、group 和 output shape 在 JIT 内固定。
- BrainCell 只提供模型特有互操作，不隐藏 BrainState 的 grad/state 接口。
- 尚未通过真实实验证明通用的 helper 不进入公共 API。

## 演进方式

新增接口时按以下顺序更新文档：

1. 在本页确定能力边界和与现有系统的关系；
2. 在 `architecture.md` 锁定 ownership、数据流和生命周期；
3. 在 `api.md` 增加完整调用合同；
4. 在 `implementation-plan.md` 增加落地阶段和验收场景；
5. References 继续保留原始方法、数据和结论，不承担公共 API 规范职责。
