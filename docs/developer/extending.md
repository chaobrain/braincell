# 扩展模型与积分器

添加机制或积分器时，先选择已有模板，再实现模型特有的方程，并验证它能通过 Cell 的实际路径运行。
下表给出阅读顺序；完整签名、公式与可运行模型例子由对应 Design Current 维护。

| 扩展对象 | 先读模板与契约 | 再读注册与集成 |
| --- | --- | --- |
| Channel | [HH/Markov 模板](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/current/api.md)、[模板校验](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/current/template-invariants.md) | [Mech 注册](https://github.com/chaobrain/braincell/blob/main/docs/design/mech/current/api.md#注册与事件契约)、[Cell.paint](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/api.md#paint-与-place) |
| Ion | [公共离子模型](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/current/api.md)、[生命周期与 KineticIon 扩展模板](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/current/kinetic-ion-api.md) | [电流快照与调度](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/architecture.md#离子电流快照与调度) |
| Synapse | [动力学与事件契约](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/current/api.md)、[状态归属](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/current/architecture.md) | [事件源与连接](https://github.com/chaobrain/braincell/blob/main/docs/design/network/current/connections.md) |
| Integrator | [积分注册与调用](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/current/api.md)、[目标协议](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/current/api.md#积分协议) | [求解路径](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/current/architecture.md) |

## 添加 Channel 或 Ion

1. 找到所属家族及相近模型，在 `braincell/channel/` 或 `braincell/ion/` 的合适模块中实现。
2. 按 Current 选模板，声明所需参数、状态和离子依赖，注明方程来源及本地调整。
3. 用 `register_channel` 或 `register_ion` 注册具体模型；进入公共模型目录时更新包导出及相邻测试。
4. 先验证独立模型的初始化、reset、导数、单位和电流符号，再通过 paint 放入 Cell 验证绑定、状态形状及运行结果。

HH/Markov 还需验证门稳态或概率守恒；动态 Ion 需检查浓度、守恒约束及电流驱动。
可复用的模型出处记录在
[Ion/Channel 文献表](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/references/ion-channel-bibliography.md)，
具体模型的对照配置和结果随对应验证工作流维护。

## 添加 Synapse

以 `braincell/synapse/exponential.py` 及相邻测试为起点，确定内部状态和事件输入契约，
实现动力学后用 `register_synapse` 注册。通过 Cell.place 和 Network.connect 完成一次真实事件投递。

验证事件前后的状态变化、多事件聚合、电流单位与符号、衰减轨迹及 reset。
如果改变的是 Connection weight 更新规则，先阅读
[可塑性讨论](https://github.com/chaobrain/braincell/blob/main/docs/design/network/proposals/connection-plasticity.md)，
确定状态应该属于突触模型还是连接。

## 添加 Integrator

在 `braincell/quad/` 选择相邻算法作为实现和测试参照，按目标协议实现步函数，
使用 `register_integrator` 声明名称、别名和阶数。
明确该算法适用于通用 ODE，还是需要 Cell 的专用电压接口。

用解析解检查单步误差和多步收敛，验证阶段钩子、单位、状态形状及编译循环。
面向 Cell 的算法还需覆盖多 CV、分叉与边界输入，现有问题见
[边界输入提案](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/proposals/explicit-solver-boundary-inputs.md)。

## 完成贡献

更新对应 Current 和实际相关示例，在模块 TODO 中关联完成内容或剩余问题。
测试命令与 fixture 用法见 [测试指南](testing.md)，提交要求见 [贡献流程](contributing.md#提交-pr)。
