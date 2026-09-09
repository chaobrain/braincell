# Vis TODO

`braincell.vis` 已提供脚本和 Notebook 可视化。下一步讨论将可视化迁入 braintools，
由同一模块提供简单绘图和 GUI 两类入口。

| 事项 | 状态 | 下一步 | 详情 |
| --- | --- | --- | --- |
| 可视化迁入 braintools | 讨论中 | 比较直接传 Cell 与共享数据入口，确定模块归属及旧入口兼容方案 | [迁移提案](proposals/braintools-migration.md) |
| 渲染回归基线 | 待讨论 | 选取代表性图像，准备基线和实际执行比较的 CI | [验证现状](current/visualization.md#验证现状) |

## 已有能力

[Visualization](current/visualization.md) 汇总形态、数据、拓扑、交互和导出能力，
并列出后端差异、代码入口和教程。
[Vis API](current/api.md) 提供 Cell 调用示例、全部公共接口规格、数据映射和结果对象说明。

项目进度见[全局 TODO](../TODO.md)，文档组织与事项状态遵循[设计文档规范](../AGENTS.md)。
