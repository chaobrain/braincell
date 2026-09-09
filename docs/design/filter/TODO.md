# Filter TODO

区域、位点与空间参数选择的协作入口。[全局 TODO](../TODO.md) 管理宏观进度，
[Design 规范](../AGENTS.md) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步 | 文档 |
| --- | --- | --- | --- |
| 半径与距离区域 | 待讨论 | 复用 Morph 空间指标，定义跨阈值区间切分及单位规则 | [预留类型](current/api.md#解析和缓存) |
| 子树区域 | 待讨论 | 确定根选择、连接方向及树编辑后的缓存失效 | [预留类型](current/api.md#解析和缓存)、[Morph TODO](../morph/TODO.md) |
| RegionAnchors 与 StepSamples | 待讨论 | 定义区域相对坐标与固定物理步长的端点和重复值规则 | [预留类型](current/api.md#解析和缓存) |
| 旧 RandomSamples 随机流 | 待讨论 | 将 NumPy 局部流与项目 BrainState 随机上下文方案对齐 | [当前采样](current/api.md#位点和批次)、[随机上下文](../network/proposals/random-context.md) |

Cell single 模式的位点表达见 [Cell 统一提案](../cell/proposals/single-multi-compartment-unification.md#特殊-policy-与位点表达)。

## 已实现内容索引

- [空间 callable 参数](current/spatial-callable-parameters.md)：形态上下文、metric 和 paint/sampling 参数。
- [API](current/api.md)：区域、位点、集合运算、批次与解析缓存。
- [连续采样](current/sampling.md)：几何测度、density 与 SamplingContext。
- [Network Pairing](../network/current/pairing.md)：已有端点的连接配对。
