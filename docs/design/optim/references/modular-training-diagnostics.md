# 模块化训练诊断与优化恢复

## 文档定位

本文是参数学习实验的非规范性 reference，统一定义观测、归档、spike-region 和非凸恢复
合同；不定义 BrainCell 公共 API，也不引入 Trainer。当前实验实现位于
[`training_diagnostics.py`](../../../../examples/experimental/parameter_learning/training_diagnostics.py)。
参数选择与 runtime 映射仍由 `braincell.trainable` 负责，优化器由 BrainTools 或用户代码
负责。

## 当前证据

默认 1-CV HH 基线以 `RandomState(seed=123)` 产生 32 个 initialization starts，并在固定
完整数据上运行 Adam；这些 lane 不是独立随机实验。一次 100-update 运行得到：

| 观测 | 结果 |
| --- | ---: |
| 终点 loss 低于初值 | `32/32` |
| 三个电导 scale 平均相对误差 `<10%` | `10/32` |
| best loss 出现在最终 update 以前 | `19/32` |
| final loss 比 best 高 `>10%` | `5/32` |
| 接近 `[0.1, 2.0]` transform bounds | `0/32` |
| final MSE | 中位数 `4.7627 mV^2`，范围 `0.0923--44.4395 mV^2` |

因此不能用 final loss 或 bound saturation 单独解释失败。至少要区分 spike basin、低梯度
平原、高梯度震荡、慢速移动、spike phase 误差和 `gNa/gK` 补偿。

## 可组合观测合同

训练循环保持四个独立角色：

```text
predictions = rollout(protocols)
loss, components = objective(predictions, targets)
metrics = evaluator(predictions, targets)
state = observer.capture(parameters, loss, components, metrics)
update = observer.capture_update(gradients, learning_rate)
```

角色使用普通函数和 immutable history，不要求继承框架基类。替换 objective 不应迫使用户
改动 rollout、evaluator 或 observer。

| 合同 | Shape / 内容 | 约束 |
| --- | --- | --- |
| prediction | `[time, start, probe]` | protocol 可有不同 time/probe 数 |
| target | `[time, probe]` | 与同名 prediction 对齐 |
| evaluator | spike count、signed count error、RMSE、finite | 不参与梯度 |
| state | optimizer roots、physical values、loss、metrics | optimizer update 前捕获 |
| update | optimizer-space gradients、effective LR | 与一次参数位移对应 |

协议名、probe 的物理含义、单位和采样规则进入 metadata，不编码为隐式数组位置。当前
`voltage_mse_objective()` 对协议分别产生 `[start]` MSE 后等权平均，只是最小基线。

硬 spike count 使用统一的上穿规则 `v[t] < threshold` 且 `v[t+1] >= threshold`。每个
protocol 可指定一个 spike probe；voltage RMSE 仍保留全部 probe。

### 时间对齐

训练结束后再评价 endpoint，并由 `finalize_history()` 形成：

```text
state axis:  N + 1  # initial，N 次 update 对应的状态，endpoint
update axis: N      # gradient、LR、state[t] -> state[t + 1] 位移
```

update 前计算的 loss 对应 `parameter_trajectory[t]`，绝不能与更新后的
`parameter_trajectory[t + 1]` 配对。archive、resume 和 plateau controller 都依赖这一
不变量。

### 派生诊断

`TrainingHistory` 应派生 optimizer-space gradient L2 norm、相邻 gradient cosine、step
norm、bound-normalized physical step、bound position、spike region、finite、initial /
best / final loss、best epoch 和可选 parameter error。分类阈值集中保存于
`DiagnosticConfig`；标签只是观测启发式，不证明局部最优。

## Spike Region

固定 protocol 下，spike count 在参数区域内是整数常量，跨兴奋性边界时跳变。当前四协议
目标及已观察失败示例为：

| 类型 | Signature |
| --- | --- |
| target | `(1, 2, 3, 4)` |
| low excitability | `(1, 1, 1, 2)` |
| mixed mismatch | `(1, 1, 3, 4)` |
| high excitability | `(2, 3, 4, 5)` |

每次 forward 至少记录：

```text
signature[p]
signed_count_error[p] = signature[p] - target_signature[p]
count_distance = sum(abs(signed_count_error))
count_feasible = all(signed_count_error == 0)
spike_times[p]
finite
composite_loss
component_losses
```

完整 signature 是 region ID；相同 distance 的不同 signature 不能合并。`dt`、threshold
和 refractory 规则共同定义 region，必须写入实验配置。

| Hard 指标用于 | Hard 指标不用于 |
| --- | --- |
| region、archive、候选接受、landscape、成功判定、held-out | 参数梯度、动态裁剪 trace、改变 JAX 输出结构、替代连续/surrogate loss |

梯度始终来自连续或 surrogate component。region 只允许控制下一阶段的固定形状 loss 配置。

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

## Archive 与模型选择

每条 start 维护两个固定 shape archive：

| Archive | 更新条件 | 用途 |
| --- | --- | --- |
| `continuous_best` | finite 且 Composite loss 更低 | 保存连续目标最优状态 |
| `spike_feasible_best` | 所有 protocol signed count error 为零 | 保存满足离散成功条件的最优状态 |

每项保存 valid、epoch、optimizer roots、physical values、loss components 和 metrics。无
finite 状态或从未进入可行区时使用 `valid=False, epoch=-1`，浮点 payload 为 NaN；调用方
必须先检查 valid。continuous-best 不能冒充 spike-feasible 成功。

多个 feasible 候选依次比较 maximum matched spike-time error、Composite loss、aggregate
voltage RMSE、到 bounds 的参数距离。timing 差异小于 `0.025 ms` 视为并列。第一版由
`extract_best_archives(history)` 在训练后逐 start 提取，不回写当前参数，因此不改变训练
轨迹；长训练才考虑在 JAX loop 内在线维护。

held-out 模型选择依次比较：train/held-out finite、held-out signature、held-out timing 与
voltage、train Composite loss、生理先验或不确定性。synthetic 数据同时报告 trace 和
parameter success；真实数据不能把不可见的真参数作为唯一标准。

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

`braintools 0.1.9` 的 `CosineAnnealingWarmRestarts` 曾出现 reported LR 更新但实际参数增量
仍固定的问题（`base_lr=0.1, T_0=2, eta_min=0.01` 时报告 `0.1, 0.055, ...`，实际 delta
始终 `-0.1`）。正式使用前必须以常梯度回归测试验证 effective LR；临时 controller 只放
在 example 内。

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

## Region-aware loss 与可视化

| Region | 连续目标调整 | 搜索约束 |
| --- | --- | --- |
| 缺失 spike | threshold margin、smooth peak/event、target-fixed window | 不硬编码 `gNa`/`gK` 方向 |
| 额外 spike | unmatched-event、late no-event、AHP/steady/late voltage | 检查 rebound，不永久裁掉 peak |
| count 正确 | 增加 timing、ISI、AP shape、AHP、full trace | LR 约 `0.001`，radius 至多 `0.1` |

landscape 图必须显示真实 grid 或 grid size，强制加入 target、initial、checkpoint 和 candidate
坐标，并把 contour 标为 sampled boundary estimate。边界附近的自适应细化要报告额外
forward 数；endpoint-anchored 二维切片中的真值只能标为 `target projection`。

## Artifact 与事件合同

```text
history.npz    optimizer/physical/loss/metric histories and archives
metadata.json  seed, backend, precision, dt, duration, solver, protocols, probes, units
summary.json   per-start classification and aggregate counts
```

每条轨迹还应记录 signature/count-distance history、首次和最后 feasible epoch、entry/exit
次数、region dwell、restart 与 perturb events、effective LR 和额外 forward 数。高分辨率 trace
不按每个 epoch 保存，避免 `time x start x epoch` 膨胀；只保留 initial/final/best 或另设采样
模块。

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
[优化消融协议](optimization-ablation-protocol.md)。

## 必须验证的边界

- target 必须经同一 hard evaluator 得到 `(1, 2, 3, 4)`；
- non-finite trace 使用 invalid region，不能以极大整数参与普通距离；
- continuous-best loss 更低但 count 错误时不能覆盖 feasible-best；
- Adam 离开可行区或 resume 后，feasible archive 仍保持一致；
- count 相同但 spike 配对错误时 timing metric 必须失败；
- window 边缘 crossing 只计一次，必要时显式定义 refractory；
- CPU/GPU 边界差异必须随 backend、precision 一起报告；
- scheduler、plateau、random key、optimizer moments 和 archives 都属于完整 resume 状态。

## References

1. Loshchilov, I. & Hutter, F. *SGDR: Stochastic Gradient Descent with Warm
   Restarts*. ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983).
2. Wales, D. J. & Doye, J. P. K. *Global Optimization by Basin-Hopping and the
   Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110
   Atoms*. J. Phys. Chem. A 101, 5111-5116 (1997).
   [arXiv:cond-mat/9803344](https://arxiv.org/abs/cond-mat/9803344).
3. Hansen, N. *The CMA Evolution Strategy: A Tutorial*.
   [arXiv:1604.00772](https://arxiv.org/abs/1604.00772).
4. Bengio, Y., Louradour, J., Collobert, R. & Weston, J. *Curriculum Learning*.
   ICML 2009. [PDF](https://icml.cc/2009/papers/119.pdf).
