# Channel TODO

通道模板与通道实现的协作入口。[全局 TODO](../TODO.md) 管理宏观进度，
[Design 规范](../AGENTS.md) 定义分类和状态。

## 当前需要推进的事项

| 事项 | 状态 | 下一步或待决定问题 | 文档 |
| --- | --- | --- | --- |
| 参数单位元数据 | 待讨论 | 对齐构造签名和 Mech 错误定位，明确元数据 owner | [API](current/api.md)、[Mech TODO](../mech/TODO.md) |
| GHK 与 Q10 审计 | 实施中 | 逐家族核对原模型驱动力及参考温度，补剩余参数来源 | [辅助函数](current/api.md#电流与辅助函数)、[文献表](../ion/references/ion-channel-bibliography.md) |
| 逐模型 MOD 与刚性检查 | 待讨论 | 电压钳和电流钳对照，并按 dt/solver 检查收敛 | [Mech 验证框架](../mech/proposals/runtime-extensions.md#机制生成与验证) |
| 氯通道 | 待讨论 | 在 Chloride 家族确定后补对应通道 | [Ion TODO](../ion/TODO.md) |
| 门变量命名 | 待讨论 | 核对 p/q 与模型自定义名称的下游使用，设计兼容路径 | [模板 API](current/api.md#hh-与-gate) |

## 已实现内容索引

- [API](current/api.md)：HH/Markov 声明、生命周期、单位及可运行示例。
- [通道模板约束](current/template-invariants.md)：HH、OhmicHH、Markov 的声明校验、单位和裁剪策略。
- [空间 callable 参数](../filter/current/spatial-callable-parameters.md)：跨模块能力，由 Filter 维护。

通道已使用 `Na_HH1952`、`K_HH1952` 等现行名称，旧 `INa_HH1952` 等别名已移除；
`IL` 是当前漏通道名称。可用名称以 [Channel 导出](../../../braincell/channel/__init__.py) 为准，
迁移记录见 [通道简化](../../specs/2026-09-02-channel-simplification.md)。

## 参考与示例

- [Ion/Channel 文献表](../ion/references/ion-channel-bibliography.md)：共享来源记录，仅保留这一份；缺失归因不视为已核实。
- [小脑导入与比较进度](../../../examples/neuron_compare/cerebellum-import-progress.md)：具体模型的验证工作由示例维护。
