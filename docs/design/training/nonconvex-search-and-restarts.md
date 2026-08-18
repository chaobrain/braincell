# 非凸搜索、学习率重启与局部盆地恢复

## 文档状态

本文定义 BrainCell 参数学习实验中的非凸优化恢复策略，是研究合同，不是稳定公共
API。

spike-count 区域的语义见
[Spike-Count Region Control](spike-count-region-control.md)，具体消融和测试预算见
[Optimization Ablation Protocol](optimization-ablation-protocol.md)。

## 问题定义

当前四协议实验是确定性的 full-batch 优化。每个 epoch 使用同样四条协议，参数只有
三个，但 loss 同时包含 HH 动力学、spike 出现或消失、时间偏移和 `gNa-gK`
补偿。因此低维不等于凸，也不保证局部梯度能找到目标盆地。

需要区分以下状态：

| 状态 | 观测 | 正确动作 |
| --- | --- | --- |
| 正常下降 | best loss 持续改善 | 保持当前优化器 |
| 低梯度 plateau | 长期无改善，近期梯度显著低于自身历史 | LR kick，随后非局部扰动 |
| 高梯度震荡 | 长期无改善，但梯度和符号翻转较大 | 降 LR、冷却，不直接加速 |
| 仍在慢速移动 | loss 和参数仍有系统趋势 | 延长预算或正常退火 |
| 正确 spike 区域内精修 | count 正确但 timing/trace 未收敛 | 小 LR、小扰动、完整 loss |
| 等价低损失解 | 多组参数都拟合良好 | 归类为不可辨识，不强行跳出 |

任何确定性的局部一阶方法在严格局部极小点都没有离开方向。Adam、RAdam 或
L-BFGS 可以改善局部条件数和步长，但不能代替全局入口、目标函数 continuation
或显式非局部搜索。

## 恢复层级

恢复按成本和侵入性从低到高排列：

1. 保留 best checkpoint，防止后续更新破坏已经找到的解；
2. 正常 cosine decay，使正确盆地内的轨迹稳定下来；
3. 周期 SGDR 或 plateau-triggered LR kick，跨过浅的 loss barrier；
4. 在 optimizer space 批量扰动并选择候选，跨过零梯度盆地；
5. 重新进行全局候选筛选，避免持续围绕错误区域搜索；
6. 改变 curriculum 或增加辨识性更强的 protocol。

不能把“最后一个 epoch”视为模型选择规则。当前结果已经出现训练中间 loss 低于
最终 loss 的情况，warm restart 会进一步增加这种情况。

## Checkpoint 合同

每条独立轨迹至少维护两个 archive：

- `continuous_best`：四协议 differentiable Composite loss 最低的 finite 状态；
- `spike_feasible_best`：四协议 hard spike signature 正确时，Composite loss 最低的
  finite 状态。

每个 checkpoint 保存：

```text
epoch
optimizer_parameter_z
physical_parameter_theta
composite_loss
component_losses
gradient_norm
learning_rate
spike_signature
protocol/probe metrics
recovery_count
recovery_reason
```

loss 与参数必须语义对齐。若 epoch `t` 的 loss 在 optimizer update 前计算，则该
loss 对应 `parameter_trajectory[t]`，而不是更新后的 `parameter_trajectory[t + 1]`。
测试必须专门防止这个 off-by-one 错误。

非有限 loss、trace 或参数不得覆盖最后一个 finite checkpoint。保存 optimizer
state 用于恢复同一局部轨迹；保存纯参数 checkpoint 用于重新开始新的局部搜索。

## Cosine Annealing 与 SGDR

普通 cosine annealing 从较大学习率平滑降到 `eta_min`，适合正确盆地内收敛。SGDR
在每个 cosine 周期结束后把学习率恢复到 `base_lr`，用于重新获得探索能力 [1]。

第一轮固定周期消融采用：

```text
base_lr = 0.02
eta_min = 0.001
T_0 = 30 updates
T_mult = 2
total = 180 updates
```

周期长度为 `30, 60, 120, ...`，180-update 实验在 update 30 和 90 发生 restart。
SGDR 只改变步长，不创造新的梯度方向。因此：

- 浅盆地或狭窄 valley 中可能受益；
- 严格零梯度区域不会因为提高 LR 自动离开；
- 高梯度 spike boundary 上可能因 restart 变得更不稳定；
- 必须依赖 best checkpoint 防止 restart 破坏已找到解。

标准 SGDR 只重启学习率，不重置 Adam moments。实验必须把以下两个变体分开：

1. `lr_restart_only`：保留一阶、二阶 moment，符合常规 warm restart；
2. `lr_and_moment_restart`：清空 moment，用于判断历史尖峰梯度是否抑制后续更新。

### 当前 `braintools` 阻塞项

安装版本 `braintools 0.1.9` 提供
`CosineAnnealingWarmRestarts`，但不能直接假定其有效。已观察到：

```text
SGD, constant gradient = 1
base_lr = 0.1, T_0 = 2, eta_min = 0.01

reported LR: 0.1, 0.055, 0.1, 0.055, ...
actual delta: -0.1, -0.1, -0.1, -0.1, ...
```

`scheduler.step()` 更新了周期状态和报告值，但 Optax 主 transformation 读取的
callable LR state 没有同步。后续实现必须先用常梯度回归测试证明实际参数增量正确，
再用于四协议实验。临时方案应位于 example-local controller 中；本文不要求修改
BrainCell API 或外部安装包。

## Plateau 检测

plateau 以 best loss 的相对改善为主，不以单个梯度阈值为主。默认状态为：

```text
warmup_updates = 40
patience = 25
relative_improvement = 0.005
cooldown = 20
max_recoveries = 3
epsilon = 1e-8
```

一次显著改善定义为：

```text
(old_best - new_best) / max(abs(old_best), epsilon) >= 0.005
```

只有在 warmup 后、距离最近显著改善至少 25 updates、当前不在 cooldown 且没有
达到最大恢复次数时，才触发 plateau。raw loss 的短期上升或一次 spike-boundary
抖动不能单独触发。

plateau 触发后再分类：

- **flat plateau**：近期梯度中位数低于该轨迹早期参考值的 10%，且物理参数窗口
  位移小；
- **oscillatory plateau**：近期梯度没有衰减、signed gradient 频繁翻转，或 loss
  在窄区间反复上升下降；
- **slow progress**：未满足显著改善阈值，但 loss 趋势和参数位移仍一致，此时延迟
  recovery，避免误判慢收敛。

使用轨迹自身的历史尺度而不是全局绝对梯度阈值，因为不同 loss normalizer、参数
transform 和 spike 区域会改变梯度量级。

## Adaptive LR Recovery

flat plateau 首先进行一次受限 LR kick：

```text
restart_lr = 0.02
kick_updates = 10
```

kick 期间保留 best checkpoint。如果 10 updates 后没有显著改善，则进入
perturb-and-select。高梯度震荡不执行 kick，而是将 LR 退火到 `0.001`，完成至少
20 updates 的冷却后重新判断。

每条 start 独立维护 plateau、cooldown 和 restart 状态。把所有 starts 放入同一个
`vmap` 不得意味着它们共享“是否停滞”的布尔值或只由 batch 平均 loss 触发恢复。

## Perturb-And-Select

参数扰动在 sigmoid 前的无约束 `z` 空间进行，而不是直接对带单位物理电导相加。
这样可复用现有 bounded transform，并使不同量纲的参数具有可比较的扰动尺度。

默认每次恢复生成：

```text
radii = (0.1, 0.25, 0.5)
candidates_per_radius = 8
incumbent = 1
total_forward_candidates = 25
```

每个候选为：

```text
z_candidate = z_checkpoint + radius * normalized_direction
```

`normalized_direction` 由 `brainstate.random` 生成并规范化。候选先经过 bounded
transform，再批量执行四协议前向仿真。选择规则由 spike-region 文档定义，不能只
比较总 loss。

若没有候选优于 incumbent，则保持 checkpoint、增加 recovery count 并进入
cooldown。若接受候选：

1. 写入新的 optimizer parameter；
2. reset cell dynamic state；
3. 清空 Adam first/second moments；
4. 重启 LR phase；
5. 记录从旧 signature 到新 signature 的迁移；
6. 保留跳跃前的两个 best archives。

旧 optimizer moments 不可跨非局部跳跃复用，否则 moment 可能立即把参数拉回旧
basin。候选评价只做 forward，不需要为 25 个候选同时保留反向图。

## 全局候选筛选

八个 `2 x 2 x 2` 角点不能代表三维 basin coverage。第一版全局入口采用 1024 个
optimizer-space 候选，批量运行四协议 forward。候选集合必须包含：

- 当前八个角点；
- physical bounds 中点对应的 `z=0`；
- deterministic low-discrepancy coverage；
- 用户指定的先验或已知 baseline 点。

从中选择 16 个 local-training starts。选择时分层考虑：

1. spike signature 和 count mismatch；
2. differentiable Composite loss；
3. optimizer-space 距离；
4. 是否贴近 transform bound；
5. simulation 是否 finite。

不能简单取得 loss 最低的 16 点，因为它们可能全部来自同一条补偿 valley。全局
筛选和局部 plateau recovery 是两个独立方向，消融时必须分开。

## Curriculum 作为 Landscape Continuation

curriculum 不是数据展示顺序，而是将易优化目标逐渐变形成最终目标 [4]：

1. subthreshold voltage、multiscale 和 smooth peak；
2. threshold margin 与平滑 event；
3. spike count、latency 和 alignment；
4. AP shape、AHP 和完整 trace；
5. 降低 surrogate temperature 并执行低 LR 精修。

这可以扩大早期有效梯度区域，减少从静默状态直接面对尖锐 spike boundary 的概率。
curriculum 与 SGDR、perturbation 必须单独消融，否则无法判断收益来自 loss 变形还是
搜索策略。

## 不适用或需要谨慎的方法

- **只提高固定 LR**：可能越过浅 barrier，但对零梯度无效，对高梯度边界更危险；
- **AdamW**：weight decay 作用于 `z`，会拉向 physical bounds 中点，不是通用先验；
- **L-BFGS**：适合正确 basin 内精修，不是全局逃逸方法；
- **parameter averaging**：非线性模型中两个可行参数的平均值可能落入错误 spike
  区域；
- **只增加 epoch**：对仍在移动的轨迹有效，对已停在错误 basin 的轨迹无效；
- **只保留 batch 最优**：会隐藏不同 starts 的失败模式和 basin robustness。

## 边界情况

- transform 在上下界附近饱和时，`z` 扰动可能产生很小 physical 位移；记录两种
  空间中的距离；
- candidate 产生 non-finite trace 时标记失败，不覆盖 incumbent；
- plateau 期间进入正确 spike region 时立即更新 feasible archive，并取消待执行的
  非局部 jump；
- 多个候选完全相同 loss 时优先选择离当前 checkpoint 更近、离 bounds 更远者；
- resume 必须恢复 scheduler、plateau、random key、optimizer moments 和 archives；
- 多 start vmap 下随机 key 必须按 start 和 recovery event 独立 fold-in；
- 若多组参数均在 held-out protocol 上成功，应报告等价解集合，而不是继续强制跳跃。

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
