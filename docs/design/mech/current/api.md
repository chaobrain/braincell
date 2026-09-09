# Mech Declaration API

`braincell.mech` 定义 Cell.paint/place 消费的不可变声明。它不持有积分状态；
`braincell.Channel/Ion/Synapse` 是运行时基类，名称相同但导入位置不同。

## 声明与查询

```python
import braincell as bc
import brainunit as u

leak = bc.mech.Channel("IL", name="leak", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV)
partial = leak.with_coverage(0.5)
assert leak.instance_name == "leak"
assert partial.coverage_area_fraction == 0.5
assert bc.mech.get_registry().get("channel", "IL") is bc.channel.IL
```

## 密度与电缆属性

```text
Channel(class_name, /, *, name=None, coverage_area_fraction=1.0,
        ion_name=None, ion_names=None, solver=None, substeps=None, **params)
Ion(class_name, /, *, name=None, coverage_area_fraction=1.0,
    solver=None, substeps=None, **params)
CableProperty(resting_potential, membrane_capacitance, axial_resistivity,
              temperature=309.15*u.kelvin)
```

class_name 接受 registry 名称或已注册类，params 是目标运行时构造参数；
name 决定逻辑 owner 名称，None 回退为类名。coverage_area_fraction 为 `[0,1]` 的无量纲覆盖面积比例，独立于 params。
Channel 的 ion_name 选择单离子 owner，ion_names 为多离子依赖提供名称映射；不能混淆类型与逻辑名称。
solver/substeps 配置机制独立积分，None 使用目标模型默认值。
CableProperty 四个字段依次为电压、单位面积电容、电阻率和绝对温度。
安装方法、重叠检查和空间广播见 [Cell.paint](../../cell/current/api.md#paint-与-place)。

`with_coverage(fraction)` 返回覆盖比例不同的新声明；名称和参数不同的声明通过构造器创建。
CableProperty.with_updates(**kwargs) 同样返回副本。原声明不变。
`Params(data=None, /, **kwargs)` 是不可变 Mapping，参数字典顺序不影响哈希相等；
`with_updates(**kwargs)` 构造新映射，不能用它直接写运行时状态。

## 点机制

```text
Synapse(synapse_type, /, *, name=None, **params)
CurrentClamp(delay=0*u.ms, durations=1*u.ms, amplitudes=0*u.nA)
FunctionClamp(fn)
SineClamp(amplitude, frequency, phase=0.0, offset=0*u.nA,
          delay=0*u.ms, duration=1*u.ms)
Junction(params=Params())
```

Synapse 的 synapse_type 为 registry 中的突触模型名；参数和动态方程见 [Synapse API](../../synapse/current/api.md)。
CurrentClamp 的 durations/amplitudes 描述连续分段刺激，形状、区间及 population 广播见
[Cell 刺激声明](../../cell/current/api.md#paint-与-place)。FunctionClamp 的 fn 接收带单位时间，返回电流；
SineClamp 的 frequency 是频率量，phase 为弧度裸数，offset/amplitude 为电流，delay/duration 为时间。
Clamp 在主步中点求值并缓存，观测与 solver 消费同一值，见 [ClampView](../../cell/current/views.md#clampview)。
Junction 当前只有声明，还没有 partner 接线和电压求解贡献。

旧 ProbeMechanism、StateProbe、CurrentProbe、MechanismProbe 仍在导出列表，新的观测使用
[Cell.record 与 observe](../../network/current/recording.md)，不占用点机制位置。

## 注册与事件契约

```text
get_registry() -> MechanismRegistry
register_channel(name, *, aliases=()) -> class decorator
register_ion(name, *, aliases=()) -> class decorator
register_synapse(name, *, aliases=()) -> class decorator
NoEventInput()
TriggerEventInput(*, aggregation="count")
ScalarEventInput(unit, *, aggregation="sum")
ParameterSpec(default, validator=None)
StateSpec(initial=<required>)
positive(value, name) -> None
```

装饰器注册类并返回原类。registry 按 channel/ion/synapse 分类，未知名称查询抛出 KeyError 并给近似建议；
重复冲突名称拒绝注册。具体查询与变更签名：

```text
MechanismEntry(category, name, cls, aliases=())
MechanismRegistry()
registry.register(entry) -> None
registry.unregister(category, name) -> None
registry.clear() -> None
registry.contains(category, name) -> bool
registry.get(category, name) -> type
registry.entry(category, name) -> MechanismEntry
registry.names(category=None, *, include_aliases=False) -> tuple[str, ...]
registry.items(category=None) -> tuple[tuple[str, type], ...]
```

category=None 查询所有分类；names 默认只列 canonical 名称，include_aliases=True 追加别名。
entry 返回冻结的注册元数据；register/unregister/clear 就地修改所操作的 registry，
全局 registry 的变更影响后续声明解析，独立 MechanismRegistry 实例有自己的内容。
注册冲突和别名处理的依据见 [_registry.py](../../../../braincell/mech/_registry.py)。
事件契约决定目标是否接收事件、计数或带单位的标量累加；ScalarEventInput 的 unit 决定 Connection.weight 单位。
StateSpec 是运行时 Synapse 状态初值声明；ParameterSpec 是旧显式 schema 辅助类型，
现行参数发现从构造签名读取，见 [Synapse 架构](../../synapse/current/architecture.md)。

源码入口：[density](../../../../braincell/mech/_density.py)、[point](../../../../braincell/mech/_point.py)、
[事件契约](../../../../braincell/mech/_event_contract.py)。
