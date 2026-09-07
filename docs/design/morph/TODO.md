# Morph TODO

形态模块的协作入口。[全局 TODO](../TODO.md) 管理宏观进度，
[Design 规范](../AGENTS.md) 定义文档分工和事项状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步 | 文档 |
| --- | --- | --- | --- |
| Morphology / Branch 的子包导出 | 待讨论 | 当前使用顶层 `braincell.Morphology` / `braincell.Branch`；比较同时从 `braincell.morph` 导出的可发现性收益与维护成本，检查导入依赖 | [子包导出](../../../braincell/morph/__init__.py)、[分层约束](current/layering-invariants.md) |
| 子树编辑 | 待讨论 | 定义删除、splice、两树连接和分支替换的身份及方向保持规则 | [当前 attach API](current/api.md#morphology-与连接) |
| 几何变换 | 待讨论 | 确定平移、旋转、缩放与主轴对齐后的 revision、指标和 Cell 缓存失效 | [当前查询与缓存](current/api.md#查询视图和指标) |

## 已实现内容索引

- [API](current/api.md)：Branch 构造、树连接、views、指标与独立复制。
- [Morph 分层约束](current/layering-invariants.md)：依赖方向、延迟导入及测试守护。
- [SWC Reader 约束](../io/current/swc-reader-invariants.md)：由 IO 维护的形态导入语义。
