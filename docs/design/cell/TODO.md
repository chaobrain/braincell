# Cell TODO

Cell 的具体事项与下一步集中在这里；跨模块目标与依赖见 [全局 TODO](../TODO.md)，
文档分工、状态定义和维护规则见 [Design 规范](../AGENTS.md)。

## 当前需要推进的事项

| 事项 | 状态 | 下一步 | 文档 |
| --- | --- | --- | --- |
| 显式 solver 的边界输入完整性 | 讨论中 | 根据五行方程及消元缺项，比较恢复边界约束与共享 point 装配的方案 | [边界输入提案](proposals/explicit-solver-boundary-inputs.md) |
| SingleCompartment 与 MultiCompartment 统一 | 讨论中 | 从单 branch、单 CV 开始，衔接现有集中参数模型与 Cell 的用法 | [统一提案](proposals/single-multi-compartment-unification.md) |
| 特殊 policy 与 single 位点表达 | 讨论中 | 确定中点校验的归属及省略 locset 的快捷方式 | [Policy 与位点](proposals/single-multi-compartment-unification.md#特殊-policy-与位点表达) |
| single 与 Cell 的状态轴差异 | 待讨论 | 对应状态读写、population、batch 及参数单位 | [状态与参数](proposals/single-multi-compartment-unification.md#状态形状与参数) |
| 分叉 morphology 的 single 表示 | 讨论中 | 比较等效圆柱体与跨 branch 单 CV 的几何、机制和位置映射 | [形态兼容](proposals/single-multi-compartment-unification.md#形态与旧接口的兼容) |
| single 专用积分路径 | 讨论中 | 先验证现有路径的等价性，再评估省去边界装配的收益和 solver 一致性 | [积分路径](proposals/single-multi-compartment-unification.md#积分路径是否需要单独实现) |
| 生命周期命名 | 待讨论 | 比较保留 reset/reset_state 与增加明确别名，保留重建结构和重置状态的语义区别 | [生命周期](current/api.md#生命周期) |
| 查询接口风格 | 待讨论 | 按读取缓存、触发构建及参数需求检查 node_tree/runtime/layouts，判断是否需要统一 property/method | [静态查询](current/api.md#静态查询)、[运行时查询](current/api.md#运行时查询) |

## 已实现内容

| 内容 | 入口 |
| --- | --- |
| 构造与分段、paint/place、生命周期、运行结果、状态查询及积分协议 | [Cell API](current/api.md) |
| 模块与数据归属、离散构建、状态布局、时间推进及求解路径 | [Cell 架构](current/architecture.md) |
| 离子电流快照与更新排序 | [调度契约](current/architecture.md#离子电流快照与调度) |

## 参考

- [Arbor CV Discretization](references/arbor-cv-discretization.md)：跨 branch CV 表示、属性汇总与近似局限。
- [系统总览](../architecture/current/system-overview.md)：跨模块关系、数据归属与典型流程。
