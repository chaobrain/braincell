# Modular Training Diagnostics and Ablation Workflow

## 文档状态

本文定义参数学习实验的第一层可组合诊断合同，以及后续逐模块消融顺序。它是实验性
reference，不定义 BrainCell 公共 API。当前实现位于
[`examples/experimental/parameter_learning/training_diagnostics.py`](../../../../examples/experimental/parameter_learning/training_diagnostics.py)，
首先服务于 multi-compartment 实验；接口稳定并经过多类任务验证后，再讨论是否提升到
公共模块。

参数选择和 runtime 映射仍由 `braincell.trainable` 负责，优化器仍由 BrainTools 或用户
代码负责。本文不引入全功能 Trainer。

## 当前问题与已知证据

默认 1-CV HH 实验使用一个 `RandomState(seed=123)` 产生 32 个初始点，在固定完整数据上
执行 Adam。因此这里的 32 条 lane 是 32 个 initialization starts，不是 32 次独立随机
实验；初始化以后训练是确定性的。

一次 32-start、100-update 基线观察到：

- 32 个终点 loss 均低于各自初值；
- 只有 10/32 的三个电导 scale 平均相对误差低于 10%；
- 19/32 的最低 loss 出现在最终 update 以前；
- 5/32 的最终 loss 比各自 best loss 高 10% 以上；
- 没有 start 接近 `[0.1, 2.0]` 的 transform bounds；
- final MSE 中位数约为 `4.7627 mV^2`，范围约为 `0.0923--44.4395 mV^2`。

这些数据排除了“全部失败都由 bound 饱和造成”，但不能只靠 total loss 区分以下原因：

- 不同 spike-count region 对应的非凸 basin；
- 低梯度平原、高梯度震荡或只是仍在缓慢前进；
- raw voltage MSE 对 spike phase 的敏感性；
- `gNa/gK` 补偿导致的弱可辨识性；
- 一个共同 learning rate 对不同 optimizer-space 尺度不合适。

因此第一步不是立即叠加 scheduler、扰动和新 loss，而是让每条 lane 的状态可以被比较和
复现。

## 最小组合合同

实验循环拆成四个独立角色：

```text
predictions = rollout(protocols)
loss, components = objective(predictions, targets)
metrics = evaluator(predictions, targets)
state = observer.capture(parameters, loss, components, metrics)
update = observer.capture_update(gradients, learning_rate)
```

当前代码使用普通函数和 immutable history 类型，不要求继承框架基类。用户可以只替换
其中一个角色：例如保留 rollout 和 observer，只将 MSE objective 换成 spike-aware loss。

### Prediction contract

```text
predictions[protocol]: [time, start, probe]
targets[protocol]:     [time, probe]
```

协议可以具有不同的 time 长度和 probe 数；同一协议的 prediction 与 target 必须对齐。
`start` 同时适用于并行初值、population lane 或显式候选 batch。协议名和 probe 的物理
含义进入 metadata，不编码进数组位置的隐式约定。

当前 `voltage_mse_objective()` 对每个协议分别计算 `[start]` MSE，再对协议等权平均。
它只是最小基线；协议权重、时间 mask 和复合 loss 应以新的 objective 模块加入。

### Evaluator contract

`evaluate_voltage_protocols()` 只做不参与梯度的评价，当前产生：

```text
spike_count/{protocol}         [start]
signed_count_error/{protocol}  [start]
voltage_rmse/{protocol}        [start, probe]
finite/{protocol}              [start]
```

每个协议可指定一个用于硬 spike count 的 probe。RMSE 保留全部 probe，不因 spike probe
选择丢失 dendrite 信息。硬上穿使用 `v[t] < threshold` 且 `v[t+1] >= threshold`。

### Observer contract

每次 optimizer update 前记录一个 `StateSignals`：

- optimizer-space root values；
- materialized physical parameter values；
- total 和 component losses；
- evaluator metrics。

同一次 update 另记录一个 `UpdateSignals`：

- optimizer-space gradients；
- 本次实际使用的 learning rate。

训练结束后再评价终点，并由 `finalize_history()` 形成严格对齐的历史：

```text
state axis:  N + 1  # 初始状态、每次更新前状态、最终状态
update axis: N      # 梯度、LR、从 state[t] 到 state[t+1] 的位移
```

不得把 update 前计算的 loss 与 update 后的参数放在同一 epoch。这个约束也是后续 best
archive、resume 和 plateau controller 的基础。

### Derived diagnostics

`TrainingHistory` 当前派生：

- 每条 start 的 optimizer-space gradient L2 norm；
- 相邻 gradient 的 cosine；
- optimizer-space step norm；
- 有 bounds 时，bound-normalized physical step norm 和最终 bound position；
- spike region 与有限性；
- initial、best、final loss、best epoch 和终点是否显著退化；
- 可选的最终 parameter relative error。

`summarize_history()` 的 `flat`、`oscillatory`、`slow-progress` 是窗口启发式标签，不是
“已证明局部最优”的结论。分类阈值集中在 `DiagnosticConfig`，正式对比必须保存该配置。

## 当前调用方式

```python
def loss():
    predictions = rollout()
    per_start, components = voltage_mse_objective(predictions, targets)
    metrics = evaluate_voltage_protocols(predictions, targets)
    return per_start.sum(), (per_start, components, metrics)

def train_step(epoch):
    gradients, _, (per_start, components, metrics) = gradient()
    state = capture_state(
        parameters,
        total_loss=per_start,
        components=components,
        metrics=metrics,
    )
    update = capture_update(gradients, learning_rate=learning_rate)
    optimizer.update(gradients)
    return state, update

states, updates = brainstate.transform.for_loop(train_step, epochs)
endpoint = capture_state(parameters, total_loss=final_loss, ...)
history = finalize_history(states, endpoint, updates, bounds=bounds)
summary = summarize_history(history, target_parameters=target_parameters)
```

完整接入示例见
[`trainable_hh_multistart.py`](../../../../examples/multi_compartment/trainable_hh_multistart.py)。
模块测试还使用真实 soma+dendrite 形态、亚阈值/放电两个协议和两个电压 probe，验证同一
prediction contract 不依赖 1-CV 假设。

## Artifact contract

`save_artifacts()` 将结果拆成：

```text
history.npz   # 数组历史；key 保留 optimizer/physical/loss/metric 等 namespace
metadata.json # seed、dt、duration、solver、protocol、probe、单位和实验配置
summary.json  # 每条 start 的归类和聚合计数
```

`history.npz` 还包含 `archive/continuous/*` 和 `archive/spike_feasible/*`：每条 start 的
valid、epoch、optimizer/physical 参数、loss components 和 metrics。当前 artifact 不保存
best checkpoint 的高分辨率 voltage trace 或 Adam moments；HH 示例只在内存结果中保留
target 和最终 best-fit trace。trace 采样策略后续作为独立模块加入，避免历史尺寸随
`time x start x epoch` 膨胀。

## 渐进实施顺序

每一步只增加一个可归因能力，并与前一步做相同预算对照：

| Stage | Module | Changes behavior | Purpose | Status |
| --- | --- | --- | --- | --- |
| 0 | Run manifest | No | 固定 seed、backend、precision、模型、协议和配置 | metadata 已支持 |
| 1 | Observer + evaluator | No | 分离 region、梯度、位移、bound 和有限性 | 已实现实验版 |
| 2 | Best archives | Model selection only | 保存 continuous-best 和 spike-feasible-best | 已实现历史提取版 |
| 3 | Protocol suite + held-out | Evaluation/data | 区分 trace 拟合与可辨识性/泛化 | 待实现 |
| 4 | Loss components | Yes | 逐个加入 subthreshold、count、timing、shape | 待消融 |
| 5 | Initializer | Yes | LHS/Sobol/先验及 basin diversity | 待消融 |
| 6 | Optimizer policy | Yes | 独立比较 LR、schedule、optimizer space | 待消融 |
| 7 | Plateau controller | Yes | 只识别并处理 flat/oscillatory/slow lane | 待消融 |
| 8 | Perturb-and-select | Yes | 为错误 basin 提供显式非局部跳跃 | 待消融 |
| 9 | Landscape/identifiability | No | profile、Hessian/Fisher、补偿 valley | 待实现 |
| 10 | Performance | No | compile、step time、显存、吞吐 | 待实现 |

下一步进入 Stage 3 前，应先用完整 32-start 运行检查 archive 的实际分布，并确定 held-out
协议。当前结果中 19/32 的 best 不在终点，因此模型选择应优先使用 archive，而不是 final。

## Best archive 当前合同

每条 start 维护两个固定 shape archive：

- `continuous_best`：finite total objective 最低；
- `spike_feasible_best`：所有协议 signed count error 为零时 objective 最低。

每个 archive 保存 valid、epoch、optimizer roots、physical values、loss components 和
metrics。第一版由 `extract_best_archives(history)` 在训练结束后从完整、对齐的历史逐
start 提取，不能由 batch mean 或 batch best 代替。它只用于报告和导出，不回写当前参数，
因此与未启用 archive 的训练轨迹严格一致。

若某条 start 没有 finite 状态，continuous archive 为 `valid=False, epoch=-1`；若没有任何
同时满足全部协议 signed spike-count error 为零的状态，spike-feasible archive 使用相同的
无效表示。无效 archive 的浮点 payload 使用 NaN，调用方必须先检查 valid。

这版仍保留全部历史，不以降低内存为目标。长训练需要省内存时，再将相同选择规则移入
JAX loop，在线维护固定 shape archive；跨进程恢复和 Adam moments 不属于当前实现。

## 模块加入规则

- 每个行为改变模块都有独立 enable flag、配置和事件日志；
- 一次消融只改变一个模块，保持 starts、update 数、协议和评价规则相同；
- 额外 forward rollout 单独计数，不伪装成相同计算预算；
- 报告每条 start 和 success rate，不只报告 batch best；
- 同时报告 trace success、spike-region success 和 parameter recovery；
- 多组参数在 held-out protocol 上同样成功时，归类为不可辨识，不强迫参数回到唯一真值；
- 诊断模块在证明可跨 1-CV、树突、多协议和不同 objective 复用前不进入公共 namespace。

更激进的恢复策略和正式比较预算分别见
[Nonconvex Search and Restarts](nonconvex-search-and-restarts.md) 与
[Optimization Ablation Protocol](optimization-ablation-protocol.md)。
