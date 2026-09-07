# Reduction TODO

约化模型的协作入口。[全局 TODO](../TODO.md) 管理跨模块目标与阻塞，
[维护规范](../../../AGENTS.md#module-documents-and-project-progress) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| DBNN 数据生成、训练和部署闭环 | 讨论中 | 确定输入布局、规模、资产格式及分阶段验收；沿用已有 Cell 挂载契约 | [DBNN 设计](proposals/DBNN-plan.md) |

公共 ReductionModel 接入已存在，不代表 DBNN 模型或训练流程已经实现。
DBNN 的数学与流程设计仍是提案，范围决定不等于完整实施契约。

## 已实现内容索引

- [约化模型接入指南](current/model-integration-guide.md)：Cell 挂载、生命周期、输入输出与最低测试要求。
- [Cell API](../cell/current/api.md)：详细模型与约化模型的宿主接口。

历史决策见 [Cell reduction runtime](../../specs/2026-09-04-cell-reduction-runtime.md)，
当前接入合同以 current 中的指南和实现为准。
