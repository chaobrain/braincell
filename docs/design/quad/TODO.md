# Quad TODO

数值积分和电压求解的协作入口。宏观依赖见 [全局 TODO](../TODO.md)，规范见 [Design 规范](../AGENTS.md)。

| 事项 | 状态 | 下一步 | 详情 |
| --- | --- | --- | --- |
| 显式积分消费边界 point 输入 | 讨论中 | 对照单 branch、3 CV 的五行方程，补端点电流与 synapse 的反馈项 | [Cell 边界输入提案](../cell/proposals/explicit-solver-boundary-inputs.md) |
| Single ODE 统一后的积分路径 | 讨论中 | 先确定单 branch single policy，再比较复用装配与独立 ODE 路径 | [Cell 统一提案](../cell/proposals/single-multi-compartment-unification.md) |
| 自适应步长 | 待讨论 | 明确 embedded RK 误差估计、事件时间与 recording 对齐 | [积分 API](current/api.md) |
| 标准模型性能对照 | 待讨论 | 统一 Mainen/Hay/L5PC 模型、精度、编译与计时口径 | [积分架构](current/architecture.md) |

当前实现：[API](current/api.md)、[架构](current/architecture.md)。
