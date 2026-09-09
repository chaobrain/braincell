# Synapse TODO

突触内部动力学的协作入口。Connection 路由归 [Network](../network/TODO.md)，文档规则见 [Design 规范](../AGENTS.md)。

| 事项 | 状态 | 下一步 | 详情 |
| --- | --- | --- | --- |
| 突触内部动力学可塑性 | 讨论中 | 用具体模型区分内部状态变化与 Connection weight 规则 | [可塑性方案](../network/proposals/connection-plasticity.md) |
| 模型验证覆盖 | 待讨论 | 扩展事件序列、时间常数与电压驱动力的参考对照 | [API](current/api.md) |

当前实现：[ExpSyn/Exp2Syn API](current/api.md)、[状态与事件架构](current/architecture.md)。
