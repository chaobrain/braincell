# Spike-Count 区域控制与可行 Checkpoint

## 文档状态

本文是非规范性方法 reference，讨论含 spike 的参数拟合如何使用 hard spike count
划分参数空间、控制非局部搜索并选择 checkpoint。它不把 hard count 变成可微机制，
也不修改 BrainCell 的 spike API。规范性范围见
[Design Overview](../design-overview.md)。

上层训练原则见
[Voltage-Trace and Spike-Aware Parameter Fitting](voltage-and-spike-parameter-fitting.md)，
局部盆地恢复见
[Nonconvex Search and Restarts](nonconvex-search-and-restarts.md)，验证矩阵见
[Optimization Ablation Protocol](optimization-ablation-protocol.md)。

## 为什么 Spike Count 是区域变量

对固定 protocol，spike count 在一片参数区域内保持整数不变，跨过兴奋性边界时
突然增加或减少。四协议实验因此不是一个单纯的平滑 loss surface，而是连续 loss
覆盖在多个离散区域之上。

当前目标 signature 为：

```text
target_signature = (1, 2, 3, 4)
```

当前失败终点已经出现：

```text
low-excitability:  (1, 1, 1, 2)
mixed mismatch:    (1, 1, 3, 4)
high-excitability: (2, 3, 4, 5)
```

这些 signature 的总 mismatch 可能相同，但恢复方向和 loss component 不同。因此
必须保存完整 vector，不能只保存一个 mismatch 标量。

## Region 数据模型

每次 forward evaluation 至少生成：

```text
signature[p]          hard soma spike count for protocol p
signed_count_error[p] signature[p] - target_signature[p]
count_distance        sum(abs(signed_count_error))
count_feasible        all(signed_count_error == 0)
spike_times[p]
finite
composite_loss
component_losses
```

`signed_count_error < 0` 表示缺失 spike，`> 0` 表示额外 spike。region ID 使用完整
signature tuple。不同 signature 即使 count distance 相同，也不能合并日志或 archive。

hard count 必须由统一的 upward crossing 规则和物理时间轴计算。改变 `dt`、阈值或
refractory 规则会改变 region 定义，必须作为实验配置保存。

## Hard 指标与可微目标的边界

hard spike count 用于：

- region 分类；
- checkpoint 归档；
- perturb candidate 接受；
- landscape 边界；
- 成功判定和 held-out 评价。

hard count 不直接用于：

- 对机制参数求梯度；
- 根据当前 prediction 动态裁剪 voltage trace；
- 在一个 JAX trace 内改变 loss 输出结构；
- 替代连续 voltage、event、margin 和 timing loss。

梯度仍来自连续或 surrogate component。控制器可以根据 region 调整下一阶段权重，
但每个阶段的 loss 定义必须显式记录，并保持固定形状。

## 双 Checkpoint Archive

每条 start 始终维护两个独立 archive。

### Continuous-Best

保存 finite Composite loss 最低的状态，不要求 count 正确。它用于分析连续目标是否
与 hard 成功标准冲突，也为完全没有进入可行区的轨迹提供最优结果。

更新规则：

```text
finite and loss < continuous_best.loss
```

### Spike-Feasible-Best

只保存 signature 完全等于目标的状态。候选都 count-feasible 时按以下顺序比较：

1. maximum matched spike-time error；
2. Composite loss；
3. aggregate voltage RMSE；
4. parameter distance to bounds。

第一项使用 tolerance-aware 比较：时间误差差异小于 `0.025 ms` 时视为并列，再比较
Composite loss。这样可避免浮点噪声使 checkpoint 在几乎相同 timing 间反复切换。

若训练从未进入可行区，则 `spike_feasible_best = None`，不能用 continuous-best
冒充成功结果。

## Perturb Candidate 接受规则

非局部撒点时，将 incumbent 与所有候选一起 forward evaluation。采用单调、
lexicographic 接受规则：

1. finite 优先于 non-finite；
2. 更小的 `count_distance` 优先；
3. distance 相同但 signed error pattern 不同时，优先减少缺失 spike 的 protocol 数，
   同时记录该规则为实验选择；
4. signature 相同时，Composite loss 至少相对改善 `0.5%` 才接受；
5. loss 在 tolerance 内并列时，选择 optimizer-space 距离 incumbent 更小的候选；
6. 完全并列时选择固定 candidate index，保证可复现。

一旦 incumbent 已经 count-feasible，不接受 count-infeasible 候选作为当前优化状态，
除非实验显式运行 `allow_feasible_escape` 消融。即使允许离开，可行 archive 也必须
保留。

这套规则只控制非局部 jump。正常 Adam 更新仍由 differentiable loss 决定，否则
hard region 会使每个 epoch 的更新不连续。

## Region-Aware Curriculum

控制器根据 signed count error 选择下一训练阶段，但不硬编码某个参数应增大或减小。

### 缺失 Spike

若一个或多个 protocol 的 count 过少：

- 保留 subthreshold voltage 和 multiscale loss；
- 增加 stimulus window 内 smooth threshold margin；
- 增加 smooth peak 与 filtered event 权重；
- 对目标 spike 附近使用 target-fixed window；
- 不直接命令 `gNa` 上升或 `gK` 下降，让动力学梯度决定方向。

margin 需要在模型远离 threshold、surrogate event 梯度接近零时仍提供连续方向。

### 额外 Spike

若一个或多个 protocol 的 count 过多：

- 对 unmatched event 增加 penalty；
- 对目标最后一个 spike 之后的窗口加入 no-extra-event margin；
- 保留 AHP、steady-state 和 late-voltage 约束；
- 检查刺激结束后的 rebound spike，不把它错误归入 stimulus spike；
- 不通过永久裁掉 spike peak 来隐藏额外放电。

### Count 正确

首次进入目标 signature 后：

- 立即更新 feasible archive；
- LR 进入 `0.001` 量级的精修阶段；
- perturb radius 缩小到最多 `0.1`；
- 提高 latency、ISI、AP shape、AHP 和完整 trace 权重；
- 继续记录 signature，允许识别训练是否反复离开可行区。

count 正确不是最终成功。还必须满足 spike timing、三 probe voltage RMSE、finite 和
参数恢复等当前实验标准。

## Region 转移日志

每条轨迹保存按 epoch 对齐的：

```text
signature_history
count_distance_history
first_feasible_epoch
last_feasible_epoch
feasible_entry_count
feasible_exit_count
region_dwell_lengths
restart_events
perturb_accept_events
```

建议输出 region transition table：

| start | from | to | trigger | epoch | accepted loss change |
| --- | --- | --- | --- | ---: | ---: |

这可以区分“从未靠近正确区域”“进入后被大 LR 推走”“在边界反复震荡”和“可行区内
精修失败”。

## Landscape 与边界采样

spike-count map 是离散采样，不应使用平滑 contour 的视觉分辨率冒充仿真分辨率。
所有图必须：

- 标出真实 evaluated grid points 或明确写出 grid size；
- 强制插入需要评价的 target、initial、checkpoint 和 perturb candidate 坐标；
- 将 white contour 描述为 sampled boundary estimate；
- 对孤立的 count-correct cell 说明它可能只是粗网格下的菱形区域；
- 在 boundary 附近自适应细化，并报告新增 forward 数；
- 分开显示完整 signature 或 signed error，不能只显示“是否正确”。

二维切片中的 target star 只有在固定的第三个参数也等于 target 时，才代表完整真值。
在 endpoint-anchored 切片中应标注为 `target projection`，避免把投影误认为切片上的
真实 target。

## Held-Out Protocol 与不可辨识性

多个参数点可能都位于训练 signature 的正确区域，并具有相似 trace loss。此时需要
held-out current amplitude、不同注入位置或额外 probe 区分，而不是继续提高 spike
count 权重。

模型选择顺序为：

1. train 和 held-out 均 finite；
2. held-out signature 正确；
3. held-out timing 和 voltage 指标；
4. train Composite loss；
5. 生理先验或参数不确定性。

真实实验数据不存在可见的“真参数”，因此不能把 parameter recovery 作为唯一目标。
synthetic 实验则同时报告 trace success 与 parameter success。

## 边界情况

- target spike 恰好落在窗口边缘时，crossing 只能计入一个窗口；
- 两个相邻 sample 都接近阈值时，避免把一次 crossing 重复计数；
- prediction 与 target count 相同但 spike 配对顺序错误时，timing metric 必须失败；
- burst 内的小幅跨阈可能需要 refractory 语义，否则 signature 不稳定；
- non-finite trace 的 signature 为 invalid，不设为极大整数参与普通距离计算；
- 不同 start 共享参数 batch 时，每条 start 独立维护 signature 和 archive；
- CPU/GPU 浮点差异可能改变临界边界，正式报告需保存 backend；
- target 自身必须通过相同 hard-count evaluator，不能手写期望值后不验证。

## 最小验证场景

1. 精确 target 得到 `(1, 2, 3, 4)` 并创建 feasible checkpoint；
2. silent candidate、extra-spike candidate 和 non-finite candidate 的排序正确；
3. continuous-best loss 更低但 count 错误时，不能覆盖 feasible-best；
4. 一次 Adam 更新离开可行区后，feasible archive 保留；
5. perturb 候选从 `(1, 1, 1, 2)` 进入 `(1, 2, 3, 4)` 时优先接受；
6. signature 相同且 loss 改善不足 `0.5%` 时拒绝无意义 jump；
7. target-anchored landscape 强制包含精确 target 网格节点；
8. endpoint-anchored 图把 target 标成 projection。
