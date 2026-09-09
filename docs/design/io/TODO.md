# IO TODO

形态读写的协作入口。[全局 TODO](../TODO.md) 管理宏观进度，
[Design 规范](../AGENTS.md) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步 | 文档 |
| --- | --- | --- | --- |
| ASC 几何与标记覆盖 | 实施中 | 核对 spine、轮廓 soma、多树案例及 reader 测试的具体缺口 | [API](current/api.md#asc-与-neuroml)、[测试](../../../braincell/io/asc/reader_test.py) |
| NeuroML2 导入 | 待讨论 | 从 cell/segment-group 到 Morphology 的最小映射与 fixture 开始；read 当前为 stub | [API](current/api.md#asc-与-neuroml) |
| NeuroMorpho 指标自动对照 | 待讨论 | 从 notebook 提取固定数据集、单位转换及容差，避免在线数据变化影响回归 | [NeuroMorpho API](current/neuromorpho.md)、[示例](../../../examples/io/neuromorpho.ipynb) |

## 已实现内容索引

- [API](current/api.md)：SWC、ASC、预留 NeuroML 入口、写出与 checkpoint。
- [NeuroMorpho](current/neuromorpho.md)：检索、下载、缓存与结果对象。
- [SWC Reader 约束](current/swc-reader-invariants.md)：读取流程、soma 处理、分支连接及相关测试。
- [Morph 分层约束](../morph/current/layering-invariants.md)：IO 与形态便利入口之间的依赖边界。

当前 NEURON 对照的具体数值缺口及复现入口见 [验证记录](../../../validation/neuron/known-differences.md)。
