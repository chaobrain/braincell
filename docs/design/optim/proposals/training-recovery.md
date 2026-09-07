# Training Recovery Proposal

## 状态与目的

待设计，未实现自动恢复 controller，未运行正式 A-F 消融。观测、历史总结与训练后
best archives 已有实验代码，不等于具有 online archive、完整 resume、SGDR 或 perturb。
当前能力见 [实验工作流](../current/experimental-workflows.md)，方法解释见
[诊断参考](../references/modular-training-diagnostics.md)。

本页合并原恢复参考中的候选动作和消融协议。所有阈值、预算、接受规则均属于候选实验
设计，不是 BrainCell 公共默认值。实施或正式运行仍需单独确认；本次整理不授权它们。

## 状态分类与动作

plateau 以 best loss 的相对改善为主，并使用每条轨迹自身的梯度和位移尺度：

```text
warmup_updates = 40
patience = 25
relative_improvement = 0.005
cooldown = 20
max_recoveries = 3
epsilon = 1e-8

relative_gain = (old_best - new_best) / max(abs(old_best), epsilon)
```

只有 warmup 后连续 25 updates 未达到 `0.5%` 改善、不在 cooldown 且 recovery 未达三次
时才判定 plateau。单次 raw-loss 上升或一次 spike-boundary 抖动不触发恢复。

| 状态 | 主要证据 | 下一动作 |
| --- | --- | --- |
| 正常下降 | best loss 持续改善 | 保持优化器 |
| flat plateau | 近期梯度中位数低于早期参考的 `10%`，物理位移小 | LR kick；失败后 perturb |
| oscillatory plateau | 梯度未衰减、频繁符号翻转或 loss 往复 | LR 降至 `0.001`，冷却至少 20 updates |
| slow progress | loss 趋势和参数位移仍一致 | 延长预算或正常退火 |
| spike feasible | count 正确，timing/trace 未收敛 | 小 LR、小扰动、完整 loss |
| 等价低损失解 | held-out 也成功但参数分散 | 报告不可辨识集合 |

flat plateau 的第一层恢复为 `restart_lr=0.02`、`kick_updates=10`。每条 start 独立维护
plateau、cooldown 和 recovery 状态，不能由 batch mean loss 统一触发。


## 非局部恢复

### Cosine 与 SGDR

普通 cosine decay 适合正确 basin 内收敛；SGDR 可跨浅 barrier，但不会在严格零梯度区
创造方向。固定周期消融使用：

```text
base_lr = 0.02
eta_min = 0.001
T_0 = 30 updates
T_mult = 2
total = 180 updates  # restart at update 30 and 90
```

`lr_restart_only` 保留 Adam moments；`lr_and_moment_restart` 清空 moments，二者必须分开
消融。所有 restart 都依赖 best archive。

历史 effective-LR 缺陷观察已迁至 [结果记录](../current/results/fitting-and-identifiability.md#scheduler-的历史缺陷观察)；正式使用前仍须回归验证。

### Perturb-and-select

扰动在 bounded sigmoid 前的无约束 `z` 空间执行：

```text
radii = (0.1, 0.25, 0.5)
candidates_per_radius = 8
incumbent = 1
total_forward_candidates = 25
z_candidate = z_checkpoint + radius * normalized_direction
```

direction 由 `brainstate.random` 生成并归一化；random key 按 start 和 recovery event 独立。
候选经过 transform 后仅做批量 forward，不保留反向图。接受顺序为：

1. finite 优先；
2. `count_distance` 更小优先；
3. distance 相同而 signature 不同时，优先减少缺失 spike 的 protocol 数，并记录此选择；
4. signature 相同时，Composite loss 至少相对改善 `0.5%`；
5. loss 并列时选离 incumbent 更近的候选；
6. 完全并列时选固定 candidate index。

feasible incumbent 默认不能被 infeasible candidate 替换；`allow_feasible_escape` 只能作为
显式消融，且不得清除 feasible archive。接受 jump 后写入参数、reset dynamic state、清空
Adam moments、重启 LR phase并记录 region transition；无改善则保留 incumbent 并 cooldown。

### 更高成本入口

全局筛选以 1024 个 optimizer-space 候选运行 forward，再选择 16 个多样化 starts。候选
必须包含原八个角点、`z=0`、确定性 low-discrepancy 点和用户先验；选择同时考虑 signature、
loss、距离、bound proximity 与 finite，不能只取同一 compensation valley 中 loss 最低的
16 点。

curriculum 则依次引入 subthreshold/multiscale/smooth peak、threshold margin/event、count/
latency/alignment、AP shape/AHP/full trace，最后降低 surrogate temperature 并低 LR 精修。
它与全局筛选、SGDR 和 perturb 必须分别消融。

常见替代方案的边界：

| 方法 | 不能替代恢复策略的原因 |
| --- | --- |
| 只提高固定 LR | 对严格零梯度无效，在 spike boundary 上更不稳定 |
| AdamW | 对无约束 `z` 的 decay 会拉向 physical bounds 中点，不等于生理先验 |
| L-BFGS | 适合正确 basin 内精修，不提供全局逃逸方向 |
| parameter averaging | 两个可行参数的均值可能位于错误 spike region |
| 只增加 epochs | 只帮助仍在移动的轨迹，不能保证离开错误 basin |
| 只保留 batch best | 隐藏其他 starts 的失败和 basin robustness |


## 渐进加入顺序

| Stage | Module | 是否改变训练 | 目的 / 状态 |
| ---: | --- | --- | --- |
| 0 | manifest | 否 | 固定环境与配置；metadata 已支持 |
| 1 | observer + evaluator | 否 | 梯度、位移、region、finite；实验版已实现 |
| 2 | dual archives | 仅模型选择 | continuous/feasible best；历史提取已实现 |
| 3 | protocol suite + held-out | 数据/评价 | 泛化与可辨识性 |
| 4 | loss components | 是 | 逐项消融 voltage/count/timing/shape |
| 5 | initializer | 是 | LHS/Sobol/先验与 basin diversity |
| 6 | optimizer policy | 是 | LR、schedule、optimizer space |
| 7 | plateau controller | 是 | 区分 flat/oscillatory/slow |
| 8 | perturb-and-select | 是 | 显式跨 basin |
| 9 | identifiability | 否 | profile、Hessian/Fisher、compensation valley |
| 10 | performance | 否 | compile、step time、memory、throughput |

每次消融只改变一个行为模块，固定 starts、updates、protocol 和评价规则；额外 forward 单独
计数。报告所有 starts 的 success rate，而非 batch best。正式比较预算见
[优化消融协议](#固定条件与-baseline)。

## 必须验证的边界

- target 必须经同一 hard evaluator 得到 `(1, 2, 3, 4)`；
- non-finite trace 使用 invalid region，不能以极大整数参与普通距离；
- continuous-best loss 更低但 count 错误时不能覆盖 feasible-best；
- Adam 离开可行区或 resume 后，feasible archive 仍保持一致；
- count 相同但 spike 配对错误时 timing metric 必须失败；
- window 边缘 crossing 只计一次，必要时显式定义 refractory；
- CPU/GPU 边界差异必须随 backend、precision 一起报告；
- scheduler、plateau、random key、optimizer moments 和 archives 都属于完整 resume 状态。


## 消融协议

以下保留原固定协议，不因文档迁移而变成已经执行的结果。

## 固定条件与 Baseline

| 类别 | 固定值 |
| --- | --- |
| morphology | soma、`dend_a`、`dend_b` 三 compartment |
| 参数 | 全 compartment 共享 leak、HH sodium、HH potassium `g_max` |
| target | `(0.6, 120, 36) mS/cm^2` |
| 数据 | 相同四协议、三个 voltage probes；每次 `100 ms`，`dt=0.025 ms` |
| 参数化/loss | 同一 bounded sigmoid、Composite components 和 normalizers |
| starts | 同一八个 `2 x 2 x 2` physical initial points |
| 执行 | CPU、batch=8；每次 rollout 前 reset dynamic state，无 warm-up |
| optimizer | Adam，`betas=(0.9, 0.999)`，global clip norm `1.0` |

正式比较必须重跑 CPU baseline，不能用旧 GPU 结果逐值替代。spike boundary 附近的浮点
差异可能改变轨迹。

历史 Adam promotion 数值见 [结果页](../current/results/fitting-and-identifiability.md#消融的历史参考点)；不把它当作本轮已执行的 CPU baseline。

## 方法与预算

Stage 1 对所有方法使用相同八个 starts、一个 seed、batch=8 和 180 optimizer updates：

| ID | 方法 | LR / Recovery | 隔离变量 |
| --- | --- | --- | --- |
| A | Adam baseline | fixed `0.02` | 公平 CPU baseline |
| B | cosine decay | `0.02 -> 0.001`，无 restart | 后期稳定性 |
| C | periodic SGDR | `eta_min=0.001, T_0=30, T_mult=2` | 周期 restart |
| D | plateau LR | flat kick / oscillatory cooldown，无 perturb | 自适应 LR |
| E | perturb-and-select | fixed Adam + plateau perturb，无 SGDR | 非局部跳跃 |
| F | combined | cosine/SGDR、adaptive recovery、双 archive | 完整策略 |

第一轮不加入 1024-point screening 或 loss curriculum。E/F 的扰动使用
`brainstate.random(seed=0)`；所有方法保存 continuous-best，D/E/F 保存 recovery events，
F 还保存 spike-feasible-best。perturb forward 数单列，不能视作免费预算。

Stage 2 选择两个非 baseline 方法，运行 360 updates、seeds `0, 1, 2`，并保留 180-update
截面。包含周期 restart 的方法必须覆盖至少两个完整周期和 restart 后收敛窗口；每个 seed
单独报告 region transitions。

### Promotion

候选必须同时满足：

```text
trace_success >= 5 / 8
parameter_success >= 4 / 8
median_common_loss <= 0.8 * CPU_baseline_median
best_common_loss <= 1.1 * CPU_baseline_best
all endpoints finite
```

基于旧 baseline，median 阈值约为 `0.1881`；正式值必须由本轮 CPU baseline 计算。Stage 2
选择顺序为 trace success、spike-feasible starts、median common loss、median aggregate RMSE、
parameter success、总 forward 和 wall time，不能按单个 best start 晋级。

## 实现前测试矩阵

| 子系统 | 必须覆盖的场景 | 阻断条件 |
| --- | --- | --- |
| effective LR | 单参数、常梯度 SGD；eager、JIT、`for_loop`、state-aware `vmap` 序列一致 | reported LR 与实际 delta 不一致时阻断 C/F |
| restart/resume | restart 精确在 30、90；resume 后 LR 连续；两种 moment policy 分离 | 中断与连续运行不同 |
| plateau | 单调下降、warmup、25-update patience、0.5% gain、flat/oscillatory/slow、cooldown、最多三次 recovery、NaN/Inf | 状态不能作为 fixed-shape JAX pytree |
| archive/region | update 前 loss 对齐 `trajectory[t]`；final/continuous/feasible 可在不同 epoch；target `(1,2,3,4)`；timing tie `0.025 ms` | infeasible 或 non-finite 覆盖 feasible-best |
| perturb | 三个 radii x 八候选 + incumbent 为 `(25,3)`；seed、bounds、独立 key、0.5% 接受阈值、moment reset | 无改善时污染 incumbent/optimizer state |
| integration | 真实三-compartment、四协议，2 starts、4--8 updates、每 radius 2 candidates | accepted 参数未进入下一 rollout |

常梯度 scheduler fixture 为 `parameter=0, gradient=1, base_lr=0.1, T_0=2,
T_mult=1, eta_min=0.01`；每次 actual parameter delta 必须等于该 update 的 schedule LR。
真实集成测试仍使用 `brainstate.transform.for_loop`/`vmap`，controller 数组测试与昂贵 rollout
分离。

## 记录与输出

每个 method/start/seed 保存：

| 类别 | 内容 |
| --- | --- |
| histories | total/component loss、optimizer gradients、effective LR、physical/optimizer 参数 |
| region | signature、signed error、region transitions、plateau/cooldown/restart events |
| archives | continuous-best、spike-feasible-best、initial/final/best traces |
| perturb | candidate summary、accepted jump、额外 forward 数 |
| performance | compile、training、recovery evaluation、total wall time、backend、precision |

SGDR 图必须画 effective LR。最小输出集合为：

| 图/表 | 回答的问题 |
| --- | --- |
| per-start loss/LR/restart 与参数轨迹 | 是否稳定、何时恢复 |
| signature timeline 与 transition matrix | 是否进入并保持正确 region |
| continuous vs feasible archive | 连续目标是否偏离 hard 成功条件 |
| method x start 指标表 | 收益是否覆盖多数 basin |
| perturb 局部 landscape | jump 为什么被接受 |
| success-cost Pareto | 额外 forward 是否值得 |
| 180/360 截面 | 延长预算还是策略带来收益 |
| CPU/旧 GPU 摘要 | backend 差异有多大 |

## 停止与后续

| 条件 | 处理 |
| --- | --- |
| scheduler 实际 LR 测试失败 | 阻断 SGDR 方法 |
| start non-finite 且无法恢复 finite checkpoint | 标记该 start 失败，其余继续 |
| 三次 recovery 无改善 | 停止该 start 的恢复，保留 archives |
| 相同 signature/loss 但参数分散 | 转入 identifiability 分析 |
| 只改善 best start、不提高成功率 | 不晋升默认流程 |
| 收益使用超过两倍 forward | 报告结果，但不宣称同成本优势 |
| 360-update 终点差于 180 checkpoint | 先审计 restart、archive 和 resume 语义 |

矩阵完成后再独立测试：1024-point screening + 16 starts、spike-aware curriculum、Adam
`beta2=0.99`/RAdam/L-BFGS、held-out amplitude/location、adaptive landscape refinement，以及
更多参数或 density coefficients。不得同时塞入方法 F，否则无法归因。

## 方法来源

原恢复参考的 SGDR、basin-hopping、CMA-ES 与 curriculum 引文见
[文献列表](../references/modular-training-diagnostics.md#references)。
