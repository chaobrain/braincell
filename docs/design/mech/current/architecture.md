# Mech Architecture

Mech 将机制类型、参数和输入契约表示为声明。离散层决定覆盖哪些 CV/point，计算层解析 registry 并分配运行时对象。

```text
mech.Channel / Ion -> paint rules -> CV coverage -> runtime Channel / Ion
mech.Synapse       -> place rules -> logical rows -> runtime Synapse
event_input       -> weight validation + event buffer allocation
```

| 对象 | 数据与职责 |
| --- | --- |
| Density | category、class_name、name、覆盖比例、params、积分配置 |
| Point | 几何位置之外的机制声明；位置由 place rule 保存 |
| Params | 不可变参数映射，等价参数按值分组，关键字顺序不改变身份 |
| MechanismRegistry | 分类名称/别名到运行时类的单一解析入口 |
| EventInput / StateSpec | 运行时可以在构造之前读取的静态契约 |

通道到离子家族的依赖来自 runtime class.root_type，逻辑 owner 选择由声明名称决定。
同名密度 owner 的 CV 覆盖冲突由 Cell 检查；改变参数不能规避覆盖冲突。
覆盖面积比例保留为几何元数据，不混进目标模型的普通参数。

Mech 不导入其他 braincell 包或数值运行时。具体 Channel/Ion/Synapse 模块导入时主动注册自己，
因此 registry 不需要反向导入模型来找类。事件输入和字段 schema 留在这里，使运行时基类和 Network
共享契约而不形成导入环，约束见 [Network 模块布局](../../network/current/module-layout.md)。

接口见 [API](api.md)，完整转换阶段见 [Cell 架构](../../cell/current/architecture.md)。
