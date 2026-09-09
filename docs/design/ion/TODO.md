# Ion TODO

离子模板与离子实现的协作入口。[全局 TODO](../TODO.md) 管理宏观进度，
[Design 规范](../AGENTS.md) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| 文献来源与归因缺口 | 待讨论 | 按文献表中的未核实项补充来源证据，不推断已经确认 | [共享文献表](references/ion-channel-bibliography.md) |
| 钠、钾动态浓度 | 待讨论 | 定义 Detailed/FirstOrder 模型的内外浓度、泵和 current 输入，复用共享生命周期 | [Ion API](current/api.md) |
| Chloride 家族 | 待讨论 | 确定固定/动态反转电位与 GABA 模型所需浓度方程 | [生命周期模板](current/kinetic-ion-api.md) |
| 外部电流一致性 | 待讨论 | 审计各动态模型是否纳入 include_external 及缓存电流 | [电流契约](current/api.md#生命周期和电流) |
| CalciumFirstOrder 单位错误 | 待讨论 | 确定 alpha/beta 的物理单位，修复电流密度到浓度导数的转换并补对照 | [具体失败条件](current/api.md#动态钙浓度) |

小脑机制导入、PC 数值比较及剩余 MOD 覆盖由 [示例进度](../../../validation/neuron/cerebellum-import-progress.md) 管理，
不在这里复制逐模型任务表。Single 兼容范围由 [Cell 统一提案](../cell/proposals/single-multi-compartment-unification.md) 讨论。

## 已实现内容索引

- [API](current/api.md)：Fixed/Nernst 家族、动态钙浓度和生命周期。
- [KineticIon 模板 API](current/kinetic-ion-api.md)：反应、source、factor、守恒及 Cell 示例。
- [KineticIon 契约](current/kinetic-ion.md)：声明、物种状态和电流输入边界。
- [Cell 电流快照与调度](../cell/current/architecture.md#离子电流快照与调度)：由 Cell 管理的运行时语义。

## 参考入口

- [Ion/Channel 文献表](references/ion-channel-bibliography.md)：Ion 与 Channel 共用的来源记录。
