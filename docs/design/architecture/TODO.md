# Architecture TODO

Architecture 维护系统级模块关系、数据流、状态归属和跨模块接口约定。
宏观目标见 [全局 TODO](../TODO.md)，文档分工与状态定义见 [Design 规范](../AGENTS.md)。

## 当前需要推进的事项

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| 声明类型与运行时基类同名 | 待讨论 | 比较命名空间限定与显式声明名，核对示例和类型标注的迁移成本 | [命名方案](proposals/interface-consistency.md#声明与运行时类型的名称) |
| 公共导出与内部实现路径 | 待讨论 | 确定导出、支持的成员和兼容策略如何共同表达公共契约 | [导出方案](proposals/interface-consistency.md#公共导出与实现路径) |
| Python 版本覆盖 | 待讨论 | 对齐 classifiers 的 3.11 至 3.14 声明和主要测试 3.13 的 CI：扩展矩阵或调整支持范围 | [CI](../../../.github/workflows/CI.yml)、[Daily CI](../../../.github/workflows/CI-daily.yml) |

## 当前架构

- [系统总览](current/system-overview.md)：模块职责、依赖与数据流、Cell/Network/Trainable/Vis 示例、状态归属和执行约定。
- [Cell 架构](../cell/current/architecture.md)、[Network 架构](../network/current/architecture.md)、[Trainable 架构](../optim/current/architecture.md)：模块内部构建与运行细节。
- [Morph 分层约束](../morph/current/layering-invariants.md)、[Vis 架构](../vis/current/visualization.md)：几何层依赖及绘图数据组织。

## 模块议题入口

Cell 的 reset 和查询风格见 [Cell TODO](../cell/TODO.md)，Morph 子包导出见
[Morph TODO](../morph/TODO.md)，通道命名现状见 [Channel TODO](../channel/TODO.md)。
Vis 向 BrainTools 迁移涉及的接口与模块归属由 [Vis TODO](../vis/TODO.md) 推进。
