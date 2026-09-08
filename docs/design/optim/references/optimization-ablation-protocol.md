# 非凸优化恢复的消融协议

## 文档定位

本文记录四协议、三电导实验的固定比较合同，不定义 BrainCell 公共 API，也不重复解释
controller 机制。状态分类、双 archive、SGDR、perturb 和 spike-region 规则见
[模块化训练诊断与优化恢复](modular-training-diagnostics.md)。本轮只定义协议；未经单独批准
不修改训练 API，也不运行正式消融。

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

现有 fixed Adam `lr=0.02`、180-update 结果只作为 promotion reference，不是测试常量：

| 指标 | 当前值 |
| --- | ---: |
| trace success | `3/8` |
| parameter success | `4/8` |
| median common loss | `0.235153` |
| median aggregate RMSE | `7.8201 mV` |
| median mean parameter error | `0.1562` |
| best common loss | `0.0153285` |

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
