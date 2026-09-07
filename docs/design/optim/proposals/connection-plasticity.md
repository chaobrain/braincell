# Connection Plasticity Training Proposal

## 状态

讨论中，尚未实现可塑性接口、动态权重规则或 weight_initial 训练入口。
本文管理参数训练与可微性；两类可塑性模型的划分、规则在 Connection 上的挂载、pre/post
信号选择和运行时更新时序由 [Network 可塑性提案](../../network/proposals/connection-plasticity.md) 管理。
本文只记录已经讨论的方向，不定义可调用的类、方法、签名或默认窗口。
当前可用的是静态 Connection weight 参数和事件 threshold，见 [支持度](../current/parameter-support.md)。
局部训练事项由 [Optimization TODO](../TODO.md) 跟踪。

## 参数与状态分开

沿用 Ion 初始浓度的分离思路：初始权重参数决定 reset 时的动态 weight；仿真期间规则
更新的是动态状态，跨 epoch 优化器更新的是 root。概念关系为：

```text
trainable initial-weight root -> reset -> dynamic weight(t)
rule parameter roots + selected pre/post events or voltage -> plasticity state transition
dynamic weight(t) -> synaptic delivery -> loss
```

这只是候选数据流，不是目前已经存在的调用方式。固定 weight 的现有行为必须保留，
接收 Cell 持有 Connection 的所有权原则也应复用，不另造一套参数 manager。

## 可微性边界

以“pre 后 5 ms 内出现 post，则 weight 增加 alpha”为讨论例子，5 ms 不是已确定默认值。
当窗口命中分支执行且之后的 loss 对 weight 敏感时，连续 alpha 可能具有梯度；未命中、
未影响后续输出或被截断时也可能是零。窗口长度、事件配对、硬比较和离散索引不保证可微。

不要求所有规则参数都可训，也不另加科学白名单；是否能经统一 source 注册、如何处理
静态配置及报错边界仍需在接口设计时确认。现有 spike surrogate 不能自动使时间窗口、
取整或事件配对可微。更不能把 surrogate 梯度解释为硬事件时间的普通有限差分导数。

## 实现前必须决定

- 先由 [Network 提案](../../network/proposals/connection-plasticity.md#实现前必须决定) 确定规则挂载、
  信号、同一步更新与投递顺序、delay、单位和 reset 契约，再按该契约建立可微的数据流。
- 明确初始权重和规则参数的 source 注册、共享分组、持久化，以及状态初值对参数的依赖。
- 区分初始权重、动态 weight、可塑性 trace、事件历史和优化器 root，避免 setter 改断绑定。
- 固定 shape 后把全部影响未来输出的可塑性状态纳入 BPTT/full RTRL；不能只跟踪 weight。

## 候选验收

先用固定外部 pre/post 事件验证 alpha 路径、无命中零梯度和重复 reset；再加入 cell-generated
事件验证代理梯度和完整 queue/trace 敏感度。对离散窗口分别检查前向规则与求导边界，
不承诺可训。电压依赖规则还需覆盖选定电压及滤波状态到后续权重和 loss 的梯度路径。
最后对同一固定参数 rollout 比较 BPTT/RTRL，再讨论其他 online 语义。
这些是后续实施条件，本次只整理文档，没有创建上述接口或测试。
