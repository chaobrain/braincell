# Synapse Architecture

目标 Cell 持有逻辑 Synapse rows 和运行时动力学；Network 只组织到这些 rows 的事件路由。

```text
place declaration -> stable Synapse ID -> type-grouped runtime rows
source event -> Connection weight/delay -> event buffer -> apply_events
synapse state -> current(V_post) -> point current -> Cell voltage solve
```

同类型模型合并进一个 runtime 节点以向量化计算，逻辑 ID、name、位置和参数行保持独立。
一个 Synapse 可以被多个 Connection 指向，所有输入作用于同一套内部状态。
Connection 保存 weight/delay/source routing，不拥有另一份突触时间常数或电导状态。

运行时基类从构造签名发现参数，通过 `_init_parameters` 物化；states 类属性声明 StateSpec，
event_input 声明输入单位及聚合规则。ExpSyn/Exp2Syn 在事件边界更新电导状态，之后由积分器推进衰减，
并在 point 电压上求点电流。点与 CV 电压装配见 [Cell 架构](../../cell/current/architecture.md)。

可训练的构造参数与 Connection weight 由各自 owner 的 TrainableManager 管理；
改变突触内部动力学的可塑性需要新的模型类，纯 weight 规则的挂载方式还在
[Network proposal](../../network/proposals/connection-plasticity.md) 中讨论。

实现入口：[运行时基类](../../../../braincell/_base_channel.py)、
[逻辑 storage](../../../../braincell/_multi_compartment/synapses.py)、
[Exp 模型](../../../../braincell/synapse/exponential.py)。公开调用见 [API](api.md)。
