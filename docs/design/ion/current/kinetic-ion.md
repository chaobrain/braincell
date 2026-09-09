# KineticIon 契约

构造与反应声明见 [模板 API](kinetic-ion-api.md)，公共模型用法见 [Ion API](api.md)。

状态：已有实现说明。本页从小脑导入记录中提取可复用的模板契约，
不把模型比较进度当作模板或整细胞数值等价性的验收结果。入口见 [Ion TODO](../TODO.md)。

## 声明与状态

`braincell.ion` 导出 `Factor`、`Species`、`Reaction`、`Source`、`Conserve` 和 `KineticIon`。
模板支持微分物种、代数物种、反应与源项，以及用 `Conserve` 解析的守恒关系。
物种表必须包含且只包含一个名为 `Ci` 的物种，它继续通过标准 `Ion.pack_info()` 提供胞内浓度。
`Co`、温度和价态属于离子字段。

物种以可见单位存储和积分；factor 只在守恒求解与导数映射时转换可见量和缩放量，
不会把持久状态改成缩放后的数值。解析后的完整物种视图包含代数物种。

模板初始化要求显式温度，子步数至少为 1；默认 solver 为 `backward_euler`，默认子步数为 1。
具体离子负责给出初值、反应和来源模型，不由模板统一指定生理参数。

## 电流输入与运行时边界

需要总离子电流的模型通过模板的电流输入路径求导。调用方提供快照时，模板可以复用该快照。
快照何时产生、离子与通道如何排序属于 [Cell 调度契约](../../cell/current/architecture.md#离子电流快照与调度)，
不由反应网络声明决定。

本页记录现有模板及详细 Cell 接入，不宣称新的 single 模式已经实现或验证。
合并范围见 [两个 Compartment 类的统一提案](../../cell/proposals/single-multi-compartment-unification.md)。

## 实现与验证入口

- [模板实现](../../../../braincell/ion/_base.py)、[模板测试](../../../../braincell/ion/_base_test.py)。
- [具体钙池](../../../../braincell/ion/calcium.py)、[离子运行时测试](../../../../braincell/_compute/ions_test.py)。
- [共享文献表](../references/ion-channel-bibliography.md)：来源与模型归因，不代表全部导入模型已完成比较。
- [小脑导入与比较进度](../../../../validation/neuron/cerebellum-import-progress.md)：具体模型、复现入口和验证限制。
