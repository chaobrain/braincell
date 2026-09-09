# Network Random Context Proposal

## 状态与目标

状态：讨论中。已确认方向是使用 BrainState 随机上下文管理一段代码内的随机过程，
移除对 `Network(seed=...)` 及逐层回退到 Network seed 的依赖。
局部随机流、生命周期和兼容迁移尚未形成完整实施契约，当前代码仍使用原有 seed 机制。
进度见 [Network TODO](../TODO.md)，当前行为见 [Network API](../current/api.md)。

问题不只是 Network 自己怎样抽样。用户可能在构造模型、初始化参数或自定义 callback 中
调用随机函数；如果每条路径都必须显式实现“seed 为 None 时取 Network seed”，漏掉任何一处
都会使外部设置无法完整控制实验。默认随机调用应自然使用所在区域的 BrainState 随机流，
用户不需要认识 Network 的 seed owner 或手动转发它。

## 当前差距

| 路径 | 当前处理 | 提案需要解决的问题 |
| --- | --- | --- |
| Network | 保存自己的 seed | 取消 Network 作为另一套默认 seed 管理入口 |
| NetStim | 显式局部 seed，或由 Network seed 和 population 名称派生；独立使用时隐式 root 为 0 | 明确事件计划实际生成时从哪里取得随机流 |
| Endpoint pairing | Network seed 与规范路径派生，显式 rule seed 覆盖；直接 connect 的隐式 root 为 0 | 统一默认来源，并明确局部子流与顺序稳定性的取舍 |
| 空间采样 | 现有 sample 要求显式 seed，内部构造局部 RNG | 讨论如何参与上下文及与现有采样复现约定兼容 |
| 用户 callback 与随机初值 | 用户自己的调用不自动读取 Network seed | 使用默认 BrainState 随机调用时无需额外转发 seed |

实现定位：[Network](../../../../braincell/network/engine.py)、
[NetStim](../../../../braincell/network/event.py)、[pairing](../../../../braincell/network/pairing.py)、
[空间采样](../../../../braincell/filter/_sampling.py)。本表说明改造边界，不表示已经完成统一。

## 上下文方向

实际接口名称是 `brainstate.random.seed_context(seed_or_key)`。以下只是上下文写法示意，
不表示当前 Network 内部已经消费该上下文：

```python
import braincell
import brainstate

with brainstate.random.seed_context(42):
    user_values = brainstate.random.normal(size=(4,))
    net = braincell.Network("demo")
```

目标是库内默认随机操作与用户默认随机调用都受这个区域控制。
BrainState 的 `default_rng()` 无参数时返回默认随机状态；显式给定 seed 时则产生独立状态。
是否保留现有局部 seed 参数、用嵌套上下文表达局部复现，或从默认流派生持久子流，仍待讨论。
默认路径不应继续隐式回退到 Network seed 或固定 root 0。

上下文管理的是实际执行的随机调用，不是对象所有权。已经生成的数组和事件计划不会因
后来进入另一个上下文而重新生成；只在构造 Network 时包一次，也不能自动控制之后的初始化和运行。

## 控制边界与待决定问题

| 问题 | 需要确定的契约 |
| --- | --- |
| 随机源覆盖 | 默认 BrainState 调用纳入统一控制；显式 key、独立 RandomState、NumPy Generator 和其他随机库不自动被接管 |
| 调用顺序 | 相同 seed、相同调用顺序的复现不等于注册顺序无关；插入一次默认流抽样可能改变后续结果 |
| 局部子流 | 是否保留按名称或阶段派生的隔离；若保留，确定从上下文取 key 的时机及用户 callback 的随机语义 |
| 延迟执行 | 分别规定构造、事件计划生成、init_state、reset 和运行时的抽样时机，不在退出上下文后悄悄依赖旧 seed |
| 持久随机状态 | 明确模型保存的 RNG 是否参与 reset、连续 run 和恢复；外部上下文不自动重置既有独立 RNG |
| JIT 与循环变换 | 区分追踪时和执行时随机操作，验证缓存命中、重新追踪及 BrainState 循环中的状态推进 |
| 兼容迁移 | 确定 Network.seed、构造参数及局部 seed 的移除或弃用步骤；已有固定数值和顺序无关测试需要逐项评估 |

不宣称这个上下文能管理任意用户随机代码，也不承诺不同线程或并行任务天然隔离。
若后续需要这些能力，应单独确定状态隔离方式。

## 候选验收

- 在同一个上下文内混合用户默认随机调用、callback、NetStim、pairing 和采样；相同 seed 与相同执行顺序可复现。
- 验证不同 seed 可改变随机结果；无随机因素的路径不因 seed 改变而改变。
- 验证嵌套上下文以及正常、异常退出后外层默认随机状态的恢复。
- 加入额外抽样或改变注册顺序，验证最终选定的顺序依赖或子流隔离契约，不继续无条件沿用旧保证。
- 对独立 RNG、显式 key 和其他随机库验证控制边界，避免测试误称它们也被外层上下文重置。
- 覆盖上下文外创建、上下文内初始化及相反情形，以及延迟抽样、JIT 重用、reset 和分段运行。

以上是未来实现的验收方向，本次文档工作不新增运行时接口或仿真测试。

## 依据与关联

- [BrainState seed_context](https://brainstate.readthedocs.io/apis/generated/brainstate.random.seed_context.html)：接口名称和默认随机状态的临时切换与恢复。
- [BrainState default_rng](https://brainstate.readthedocs.io/apis/generated/brainstate.random.default_rng.html)：默认状态与显式独立状态的区别。
- [当前 Network 架构](../current/architecture.md)：现有按路径派生 seed 和顺序无关的约定。

本地已检查的 BrainState 实现使用 try/finally 保存和恢复默认 RNG key；本文仅据此描述
BrainState 默认流，不将其扩展为对 NumPy 或任意 RNG 的统一保存恢复保证。
