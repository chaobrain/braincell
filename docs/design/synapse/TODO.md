# Synapse TODO

突触内部动力学的协作入口。Connection 路由归 [Network](../network/TODO.md)，文档规则见 [Design 规范](../AGENTS.md)。

| 事项 | 状态 | 已有证据与下一步 | 详情 |
| --- | --- | --- | --- |
| 预定事件查询第一版 | 已完成 | 四方法、72 个设备场景与边界/梯度检查已通过；后续作为窗口推进和源端共享查询的对照 | [完整实测](../../../benchmarks/performance/synapse_events/results/scheduled-event-queries.md)、[参考](references/scheduled-event-delivery.md) |
| 完整模型四层开销对照 | 已完成 | 36 个 worker、60 条计时及四层结构/输入/轨迹校验已完成；后续扩展多 CV 主动模型和 live/scheduled 混合负载 | [完整实测](../../../benchmarks/performance/synapse_events/results/scheduled-event-controls.md)、[方案](proposals/event-delivery-optimization.md#完整模型四层对照) |
| CPU/GPU 事件计划生产接入 | 讨论中 | 与 Network 明确窗口、手动方法入口、缓存失效、整数时钟及跨窗口验收 | [方案](proposals/event-delivery-optimization.md#生产接入前的待决定事项) |
| CPU/GPU 直接事件投递第二轮 | 已完成 | 两种投递、代表性完整模型、梯度/兼容性及 GPU trace 已验证；后续结合窗口合同验证自动重建和跨窗口梯度 | [完整实测](../../../benchmarks/performance/synapse_events/results/scheduled-event-delivery.md)、[最终顺序](proposals/event-delivery-optimization.md#第二轮决定与最终执行顺序) |
| GPU 专用内核与源端查询共享 | 待讨论 | 留到后续阶段，依据第二轮瓶颈再决定内核与共享查询实验 | [方案](proposals/event-delivery-optimization.md) |
| 突触内部动力学可塑性 | 讨论中 | 用具体模型区分内部状态变化与 Connection weight 规则 | [可塑性方案](../network/proposals/connection-plasticity.md) |
| 模型验证覆盖 | 待讨论 | 扩展事件序列、时间常数与电压驱动力的参考对照 | [API](current/api.md) |

当前实现：[ExpSyn/Exp2Syn API](current/api.md)、[状态与事件架构](current/architecture.md)。
