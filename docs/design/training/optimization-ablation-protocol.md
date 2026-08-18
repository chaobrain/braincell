# 非凸优化恢复的消融与测试协议

## 文档状态

本文是四协议三电导实验的下一轮测试合同。它定义怎样判断 SGDR、plateau recovery、
局部撒点和 spike-region archive 是否真正改善 basin robustness。

机制定义见
[Nonconvex Search and Restarts](nonconvex-search-and-restarts.md) 与
[Spike-Count Region Control](spike-count-region-control.md)。

本轮只记录协议；在实现获得单独批准前，不修改训练 API，也不运行正式消融。

## 固定实验条件

为隔离优化策略，所有方法固定：

- 三 compartment morphology：soma、`dend_a`、`dend_b`；
- 三个全 compartment 共享参数：leak、HH sodium、HH potassium `g_max`；
- physical target：`(0.6, 120, 36) mS/cm^2`；
- 同一组四协议与三个 voltage probes；
- 每次 rollout 为 `100 ms`，`dt = 0.025 ms`；
- 同一 bounded sigmoid transform、Composite component 和 normalizer；
- 同一八个 `2 x 2 x 2` physical initial points；
- CPU backend、batch=8；
- Adam baseline 的 `betas=(0.9, 0.999)`、global clip norm `1.0`；
- 每次 rollout 前 reset dynamic state，不增加 warm-up。

正式比较必须重新运行 CPU baseline，不能把旧 GPU 结果直接当作逐值对照，因为 spike
边界附近的 backend 浮点差异可能改变轨迹。

## 当前 Baseline

现有四协议、固定 Adam `lr=0.02`、180-update 结果为：

| 指标 | 当前值 |
| --- | ---: |
| trace success | `3 / 8` |
| parameter success | `4 / 8` |
| median common loss | `0.235153` |
| median aggregate RMSE | `7.8201 mV` |
| median mean parameter error | `0.1562` |
| best common loss | `0.0153285` |

这些数字是 promotion reference，不是单元测试常量。重新运行 baseline 后还要报告 CPU
结果和旧结果的差异。

## 方法矩阵

第一阶段每种方法固定 180 optimizer updates。

| ID | 方法 | LR / recovery | 目的 |
| --- | --- | --- | --- |
| A | Adam baseline | fixed `0.02` | 重建公平 CPU baseline |
| B | cosine decay | `0.02 -> 0.001`, no restart | 验证后期稳定性 |
| C | periodic SGDR | `0.02`, `eta_min=0.001`, `T_0=30`, `T_mult=2` | 验证周期 restart |
| D | plateau LR recovery | flat kick / oscillatory cooldown，无 perturb | 隔离自适应 LR 控制 |
| E | perturb-and-select | fixed Adam + plateau perturb，无 SGDR | 隔离非局部跳跃 |
| F | combined | cosine/SGDR、adaptive recovery、双 archive | 验证完整策略 |

第一轮不同时加入 global 1024-point screening 或 loss curriculum；它们作为后续独立
实验轴。否则即使 F 胜出，也无法判断收益来自初始化、目标变形还是 restart。

方法 E/F 的随机扰动使用 `brainstate.random`，第一阶段固定 seed `0`。所有方法都
保存 continuous-best；D/E/F 额外保存 recovery event；F 保存 spike-feasible-best。

## 两阶段预算

### Stage 1：公平筛选

- A-F 全部运行 180 updates；
- 相同八个 starts，一次 batch=8；
- 每个方法只运行一个 seed；
- optimizer update 数相同；
- perturb forward evaluations 作为额外预算单列，不伪装成免费计算；
- 按 promotion score 选择两个非 baseline 方法。

### Stage 2：长期与随机性

- 两个入选方法运行 360 updates；
- seeds 为 `0, 1, 2`；
- 保留 180-update 截面，既能与 Stage 1 比较，也能分析额外预算；
- 记录每个 seed 的 region transitions，不只报告平均值；
- 若方法包含周期 restart，必须覆盖至少两个完整周期和 restart 后的收敛窗口。

## Promotion 规则

方法进入正式候选流程需要同时满足：

```text
trace_success >= 5 / 8
parameter_success >= 4 / 8
median_common_loss <= 0.8 * CPU_baseline_median
best_common_loss <= 1.1 * CPU_baseline_best
all endpoints finite
```

用现有 baseline 估算，20% median improvement 对应约 `0.1881`，但正式阈值必须使用
本轮 CPU baseline 计算。

选择 Stage 2 两个方法时依次比较：

1. trace-success 数；
2. spike-feasible start 数；
3. median common loss；
4. median aggregate RMSE；
5. parameter-success 数；
6. 总 forward rollout 和 wall time。

不能只按 best start 选择，因为本轮目标是提高 basin robustness。

## Scheduler 回归测试

任何正式运行前，先用单参数、常梯度系统验证有效 LR。

### Eager 常梯度测试

配置：

```text
parameter = 0
gradient = 1
optimizer = SGD
base_lr = 0.1
T_0 = 2
T_mult = 1
eta_min = 0.01
```

期望：actual parameter delta 与用于该 update 的 schedule LR 数值一致，不能出现
reported LR 为 `0.055` 而 delta 仍为 `0.1`。

### 编译控制流测试

同一序列分别通过：

- eager update；
- `brainstate.transform.jit` 单步；
- `brainstate.transform.for_loop` 多步；
- state-aware `vmap` 两条独立 schedule。

四种路径必须产生相同 LR 和参数序列。`vmap` 测试给两个 start 不同 plateau history，
确认 scheduler/restart state 没有跨 start 共享。

### Restart 与 Resume

- 检查 restart updates 精确为 30 和 90；
- checkpoint 后恢复 scheduler，下一 update 的 LR 与不中断运行一致；
- `lr_restart_only` 保留 moments；
- `lr_and_moment_restart` 将 moments 清零；
- 非局部 jump 后 parameters 保持在 accepted candidate，而不是 optimizer reset 前值。

上述测试失败时，SGDR 方法 C/F 必须标为 blocked，不能回退成“reported LR 曲线正确”
就继续实验。

## Plateau Controller 单元测试

使用纯数组 loss/gradient fixture 覆盖：

1. 单调显著下降永不触发；
2. warmup 内平坦不触发；
3. 25-update 无 0.5% 改善在预期 update 触发；
4. 一次显著改善重置 patience；
5. flat plateau 进入 LR kick；
6. high-gradient sign-flip plateau 进入 cooldown；
7. slow-progress 参数位移阻止误触发；
8. cooldown 内不重复恢复；
9. 最多执行三次恢复；
10. NaN/Inf 不覆盖 finite best，并触发 failure record。

测试状态必须能进入 JAX pytree，并在 `for_loop` 中保持固定 shape/dtype。

## Checkpoint 与 Region 单元测试

- loss 在 update 前计算时，对齐 `trajectory[t]`；
- final、continuous-best 和 feasible-best 可以指向三个不同 epoch；
- count 错误但 loss 更低的状态不能覆盖 feasible-best；
- exact target signature 为 `(1, 2, 3, 4)`；
- 保存完整 signed error，而不只保存 distance；
- feasible checkpoint 的 timing tie tolerance 为 `0.025 ms`；
- non-finite candidate 排在所有 finite candidate 后；
- resume 后两个 archive 与未中断运行一致。

## Perturb-And-Select 单元测试

- 三个 radii、每个八个候选，加 incumbent 后 shape 为 `(25, 3)`；
- directions 使用 `brainstate.random`，相同 seed 完全复现，不同 seed 至少一个候选不同；
- optimizer-space candidate finite，physical transform 后严格位于 bounds 内；
- 每个 start 和 recovery event 使用独立 random key；
- count distance 下降优先于同 signature 内的小幅 loss 改善；
- signature 相同但相对改善低于 0.5% 时拒绝；
- 接受候选后 Adam moments 清零、LR phase 重启、recovery count 增加；
- 无候选改善时 incumbent 和 optimizer state 不被污染；
- forward candidate evaluation 不构建反向图。

## 短真实集成测试

使用真实三-compartment、四协议模型，但限制训练和搜索规模：

- batch 取两个 starts；
- optimizer updates 取 4-8；
- perturb radii 保持不变，每个 radius 只取两个候选；
- 验证 target spike counts、finite loss、state sharing 和输出 shape；
- 人工构造 plateau controller state，使测试无需等待 40 updates 即可触发 recovery；
- 检查 accepted physical parameters 确实参与下一次仿真。

测试仍使用 `brainstate.transform.for_loop`/`vmap`，不能为了测试方便引入重复模型的裸
Python loop。控制器纯逻辑测试与真实仿真测试分开，避免每个边界测试都重新编译
4000-step rollout。

## 正式实验记录

每个 method/start/seed 保存：

```text
loss and component histories
optimizer gradients and norms
effective learning-rate history
physical and optimizer parameter trajectories
signature and signed-count-error histories
continuous-best and feasible-best checkpoints
restart, plateau, cooldown and perturb events
perturb candidate summaries
initial/final/best voltage traces
common four-protocol metrics
forward rollout count
compile, training, recovery-evaluation and total wall time
backend and precision
```

SGDR 图必须画 effective LR，而不是只画 scheduler 报告值。

## 必须输出的图表

1. 8-panel loss、LR 和 restart marker；
2. 8-panel normalized physical parameter path；
3. 每条 start 的 spike signature timeline；
4. region transition matrix；
5. continuous-best 与 feasible-best 指标对比；
6. method x start 的 common loss、RMSE、count 和参数误差表；
7. accepted perturb 的 incumbent/candidate 局部 landscape；
8. success-rate 与额外 forward/wall-time 的 Pareto 图；
9. 180/360 update 截面的稳定性比较；
10. CPU baseline 与旧 GPU baseline 的数值差异摘要。

## 失败与停止规则

- scheduler 实际 LR 测试失败：阻止 SGDR 方法；
- 任一 start non-finite 且无法恢复最后 finite checkpoint：该 start 失败但其余继续；
- 三次 recovery 后仍无改善：停止该 start 的恢复，保留 archives；
- 所有 starts 在同一 signature、相似 loss 但参数分散：转入 identifiability 分析；
- 方法只提高 best start、不提高成功率：不晋升为默认流程；
- 方法收益来自超过两倍 forward 预算：报告但不与等更新 baseline 宣称同成本优势；
- 360 updates 结果显著低于其 180-update checkpoint：检查 restart、checkpoint 和
  scheduler resume 语义后再解释为优化现象。

## 后续独立实验轴

本矩阵完成后再分别测试：

- 1024-point global screening + 16 diverse starts；
- spike-aware curriculum/continuation；
- Adam `beta2=0.99`、RAdam 和局部 L-BFGS 精修；
- held-out current amplitude 与 injection location；
- adaptive landscape refinement；
- 更多 parameter 或 density-function coefficient。

这些方向不能在第一轮 F 中同时开启，否则无法归因。
