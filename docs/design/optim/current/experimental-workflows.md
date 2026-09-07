# Experimental Optimization Workflows

## 边界与入口

以下代码已存在于 root examples/experimental，但不是 braincell 公共训练 API。
公共参数声明见 [API](api.md)。本页定义实验工具的能力与调用入口；数值结果见
[结果导航](../TODO.md#实验结果)，未实现的恢复动作见 [proposal](../proposals/training-recovery.md)。

| 工具 | 当前实现入口 | 已有能力 |
| --- | --- | --- |
| 梯度引擎 | [optim/gradients.py](../../../../examples/experimental/optim/gradients.py) | additive rollout、trajectory objective、BPTT/full RTRL、诊断 |
| 参数拟合 | [training.py](../../../../examples/experimental/optim_parameter_fitting/training.py) | 模型/数据/loss 组合、gradient stage、候选交接 |
| 梯度外搜索 | [search.py](../../../../examples/experimental/optim_parameter_fitting/search.py) | 有界候选搜索，与 physical CandidateSet 交接 |
| 优化器适配 | [optimizers.py](../../../../examples/experimental/optim_parameter_fitting/optimizers.py) | Adam、Rprop、Optax Rprop、SGD/momentum/Nesterov stage |
| 诊断与归档 | [diagnostics.py](../../../../examples/experimental/optim_parameter_fitting/diagnostics.py) | 状态/更新观测、历史总结、训练后 best archive |
| OED | [robust_oed.py](../../../../examples/experimental/optim_stimulus_design/robust_oed.py) | observation sensitivity、sampled-prior FIM、候选刺激排序 |
| 全局可辨识性探查 | [global_ensemble.py](../../../../examples/experimental/optim_stimulus_design/global_ensemble.py) | forward-only 候选池，不等于完整 posterior |

## 梯度与 Network

build_rollout_value_and_grad(target, step=..., method="bptt" 或 "rtrl") 返回实验引擎。
step 接收一个时间片、推进一次模型并返回 scalar additive loss；prepare 追踪参数相关
初始化和完整一步。调用返回逐步 losses、总 loss 和按稳定 root 名组织的 gradients。
非逐步相加目标使用同模块的 trajectory engine；不要把两者的 loss 合同混用。

Network 在求导外 prepare_run 固定路由和队列，step 内 update。引擎在每轮开始物化当前
root 并 reset，参数在该 rollout 内固定。完整 RTRL 的 carry 包括全网已捕获状态及其
parameter-major sensitivities；跨 CV/Cell 或 queue 的依赖不能省略。
常规路径不输出敏感度历史，diagnose(at=...) 是单独编译的诊断路径。
初值、loss 直接依赖参数以及总/前缀梯度的区别见 [理论](../references/bptt-to-rtrl-neuron-derivation.md)。

正常 RTRL 仍返回逐步 loss，输入和输出可随 T 增长；不随 T 增长的是递归敏感度 carry，
不是整个 Python 进程或所有输出。详细测量口径见 [网络结果](results/synapse-network-learning.md)。

## 分阶段训练

实验配置把模型、数据、loss、梯度引擎、优化器和评价分开。run_pipeline 在 stage 之间
用 physical CandidateSet 交接；梯度优化在 root 的 optimizer 坐标工作，非梯度搜索可用
有界归一化坐标。非梯度阶段改变候选后重新建立后续优化器状态，不复用过时 moments。

训练协议、validation 和 final-only test 分开。固定目标预处理、mask、normalizer 和 split，
不要让评价修改训练目标或把观察过的 holdout 继续称为未使用的 test。
这些实验类型不是公共 Dataset、Trainer、Search 或 Checkpoint API。
具体配置入口见 [参数拟合目录](../../../../examples/experimental/optim_parameter_fitting/README.md)。

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

梯度 core 的共置测试在 [optim](../../../../examples/experimental/optim/README.md)，
跨 Cell 事件验证在 [gradient correctness](../../../../examples/experimental/optim_gradient_correctness/README.md)。
拟合、诊断和 OED 模块均有对应的 *_test.py；本次文档整理不重新执行这些实验。
上述实际相关示例仅在 commit 前纳入本次提交的一致性检查，日常开发不要求同步维护
docs/examples 或增加教程副本；检查范围见 [文档对齐](../TODO.md#文档对齐)。
