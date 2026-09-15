# Experimental Optimization Workflows

## 边界与入口

实验梯度核心位于 `braincell.experimental.optim`；拟合、初始化和刺激设计工作流位于 `examples/optim/`。
公共参数声明见 [API](api.md)。本页定义实验工具的能力与调用入口；数值结果见
[结果导航](../TODO.md#实验结果)，未实现的恢复动作见 [proposal](../proposals/training-recovery.md)。

| 工具 | 当前实现入口 | 已有能力 |
| --- | --- | --- |
| 梯度引擎 | [optim/gradients.py](../../../../braincell/experimental/optim/gradients.py) | additive rollout、trajectory objective、BPTT/full RTRL、诊断 |
| 参数拟合 | [training.py](../../../../examples/optim/parameter_fitting/training.py) | 模型/数据/loss 组合、gradient stage、候选交接 |
| 梯度外搜索 | [search.py](../../../../examples/optim/parameter_fitting/search.py) | 有界候选搜索，与 physical CandidateSet 交接 |
| 优化器适配 | [optimizers.py](../../../../examples/optim/parameter_fitting/optimizers.py) | Adam、Rprop、Optax Rprop、SGD/momentum/Nesterov stage |
| 诊断与归档 | [diagnostics.py](../../../../examples/optim/parameter_fitting/diagnostics.py) | 状态/更新观测、历史总结、训练后 best archive |
| OED | [robust_oed.py](../../../../examples/optim/stimulus_design/robust_oed.py) | observation sensitivity、sampled-prior FIM、候选刺激排序 |
| 全局可辨识性探查 | [global_ensemble.py](../../../../examples/optim/stimulus_design/global_ensemble.py) | forward-only 候选池，不等于完整 posterior |

## 梯度与 Network

```python
from braincell.experimental.optim import build_rollout_value_and_grad
```

build_rollout_value_and_grad(target, step=..., method="bptt" 或 "rtrl") 返回实验引擎。
step 接收一个时间片、推进一次模型并返回 scalar additive loss；prepare 追踪参数相关
初始化和完整一步。调用返回逐步 losses、总 loss 和按稳定 root 名组织的 gradients。
非逐步相加目标使用同模块的 trajectory engine；不要把两者的 loss 合同混用。

准备阶段自动选择参数物化时机。`engine.materialization_mode` 为 `"rollout"` 时，
通用映射留在入口，满足布局条件的直接参数在单步直接读取 root；为 `"step"` 时保留
逐步物化。准备前该属性为 `None`。判定条件及 BPTT/RTRL 梯度连接见
[自动物化调度](architecture.md#实验梯度引擎的自动物化调度)。两种方法的公开梯度都保留
optimizer root 的 PyTree，包含直接物理参数的 Quantity metadata。

Network 在求导外 prepare_run 固定路由和队列，step 内 update。引擎在每轮开始物化当前
root 并 reset，参数在该 rollout 内固定。完整 RTRL 的 carry 包括全网已捕获状态及其
parameter-major sensitivities；跨 CV/Cell 或 queue 的依赖不能省略。
常规路径不输出敏感度历史，diagnose(at=...) 是单独编译的诊断路径。
初值、loss 直接依赖参数以及总/前缀梯度的区别见 [理论](../references/bptt-to-rtrl-neuron-derivation.md)。

`inspect_state_sensitivity(functional_step, ...)` 可在 rollout 外列出 traced
state 的 shape、dtype、参数根标记和逻辑 tangent bytes，用于决定 compact
carry 的候选集合。该接口只提供证据，不自动删除状态。DHS 的
`comp_triang_jvp` 与 `comp_backsub_jvp` 是可选的显式 JVP 诊断核，分别覆盖
前向消元和 recursive backsub；`comp_triang_raw` 与 `comp_backsub_raw` 可通过
`BRAINCELL_DHS_CUSTOM_JVP=1` 接入它们。该变量默认关闭，generic AD 是当前
默认路径。primal solver 的数值路径和状态更新顺序保持不变，完整 HH 短
rollout 已完成 loss/gradient 对照；固定性能比较显示 generic AD 更快，因此
显式 JVP 不作为默认速度优化。

正常 RTRL 仍返回逐步 loss，输入和输出可随 T 增长；不随 T 增长的是递归敏感度 carry，
不是整个 Python 进程或所有输出。详细测量口径见 [网络结果](results/synapse-network-learning.md)。

RTRL 默认使用 `jax.linearize` 后批量应用线性化函数。实验开关
`BRAINCELL_RTRL_JVP_MODE=direct` 可切换为逐方向 `vmap(jax.jvp)`，仅用于同一
workload 的工程对照；当前结果未显示 direct 路径更快，因此默认路径不变。

## 分阶段训练

实验配置把模型、数据、loss、梯度引擎、优化器和评价分开。run_pipeline 在 stage 之间
用 physical CandidateSet 交接；梯度优化在 root 的 optimizer 坐标工作，非梯度搜索可用
有界归一化坐标。非梯度阶段改变候选后重新建立后续优化器状态，不复用过时 moments。

训练协议、validation 和 final-only test 分开。固定目标预处理、mask、normalizer 和 split，
不要让评价修改训练目标或把观察过的 holdout 继续称为未使用的 test。
这些实验类型不是公共 Dataset、Trainer、Search 或 Checkpoint API。
具体配置入口见 [参数拟合目录](../../../../examples/optim/parameter_fitting/README.md)。

## 诊断与历史

当前 diagnostics 实现 capture_state、capture_update、finalize_history、
extract_best_archives、summarize_history、save_artifacts 和 plot_diagnostics。
输入 prediction 为 [time,start,probe]，target 为 [time,probe]；协议与单位通过 metadata 表达。
状态历史有 N+1 个位置（含初值和最终 endpoint），更新历史有 N 个位置。
update 前 loss 必须与 update 前参数配对，不能和更新后参数错位。

continuous-best 与 spike-feasible-best 分开提取；无 finite 或无可行项时使用 invalid 标记，
不能把连续 loss 最低当成 spike 成功。当前历史提取不回写训练参数；在线 archive/controller、
restart/perturb 以及完整 resume 不是因此自动具备的能力。
诊断分类是启发式，不证明局部最优或参数可辨识；详见 [诊断方法](../references/modular-training-diagnostics.md)。

## 刺激设计

robust OED 对 train candidates 的 observation sensitivity 累计 per-protocol FIM，
不必保存完整时间敏感度；global ensemble 用纯 forward 比较有限候选池。
rank、condition 或候选排序改善不保证实际训练成功率提高，更不证明全局唯一性。
方法、数据隔离与固定协议见 [刺激设计参考](../references/stimulus-design-and-identifiability.md)，
历史结果见 [拟合与可辨识性](results/fitting-and-identifiability.md)。

## 验证入口

梯度 core 的共置测试在 [optim](../../../../braincell/experimental/optim/README.md)，
跨 Cell 事件验证在 [gradient correctness](../../../../validation/optim/gradient_correctness/README.md)。
拟合、诊断和 OED 模块均有对应的 `*_test.py`。
上述实际相关示例仅在 commit 前纳入本次提交的一致性检查，日常开发不要求同步维护
docs/examples 或增加教程副本；检查范围见 [文档对齐](../TODO.md#文档对齐)。
