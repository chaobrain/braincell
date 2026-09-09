# Network TODO

Network 的协作入口。[全局 TODO](../TODO.md) 管理宏观目标与跨模块阻塞，
[Design 规范](../AGENTS.md) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| 预定事件查询生产接入 | 讨论中 | Synapse 两轮及代表性真实 Cell/Network 已验证、待提交验收；明确生产窗口、时钟、缓存失效、续跑与跨窗口训练合同 | [方案](../synapse/proposals/event-delivery-optimization.md) |
| 随机上下文替代 Network seed | 讨论中 | 确定默认流、局部子流、生命周期与兼容迁移边界 | [随机上下文](proposals/random-context.md) |
| Synapse 动力学与 Connection 权重可塑性 | 讨论中 | 细化共用的挂载、信号绑定、事件输入与生命周期合同，核对状态共享及调度语义 | [可塑性](proposals/connection-plasticity.md) |
| I-09 稀疏 delay slots | 待讨论 | 比较表示、选择规则、静态 shape 与性能基准 | [运行时扩展](proposals/runtime-extensions.md#i-09-sparse-delay-slots) |
| I-10 可学习 topology | 待讨论 | 明确结构 mutation、state 与重编译协议 | [运行时扩展](proposals/runtime-extensions.md#i-10-trainable-topology) |
| I-11 大规模 endpoint generators | 待讨论 | 确定 chunking、专用生成器与内存验收 | [运行时扩展](proposals/runtime-extensions.md#i-11-scalable-endpoint-generators) |
| Network batch runtime | 待讨论 | 明确网络 batch 轴、拓扑和事件约定 | [运行时扩展](proposals/runtime-extensions.md#network-batch-runtime) |

## 已实现内容索引

- [API](current/api.md)：公开入口、参数合同与用法。
- [事件与连接](current/connections.md)、[端点配对](current/pairing.md)、[记录与结果](current/recording.md)：按任务查阅接口。
- [架构](current/architecture.md)：公开模型、Cell-owned storage、事件调度、生命周期和 v1 边界。
- [模块分层](current/module-layout.md)：内部职责、导入边界和命名。

历史编号和原测试记录见 [决定快照](../../specs/2026-09-07-network-decisions-snapshot.md)、
[验证快照](../../specs/2026-09-07-network-verification-snapshot.md)。现行决定由 API 和架构维护。

## 参考入口

- [平台调研](references/platform-survey-2026-06.md)：架构取舍及扩展背景。
- [Connection 与 Synapse 语义](references/bmtk-netpyne-synapse-sharing.md)：既有 owner 边界与候选扩展的依据。
- [可塑性模型](references/plasticity-models.md)：BrainPy 规则与释放模型、来源、横向延伸及数值例子。

具体采用状态见各参考页。
