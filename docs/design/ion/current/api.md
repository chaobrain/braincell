# Ion API

`braincell.ion` 提供离子容器，持有内外浓度、反转电位和对应通道。Cell 通过
`braincell.mech.Ion` 安装离子；自定义反应系统见 [KineticIon 模板](kinetic-ion-api.md)。

## 最小用法

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1))
cell.paint(AllRegion(), bc.mech.Ion("CalciumDetailed", name="ca"))
cell.record("ci", bc.observe.ion(name="ca").state("Ci"))
result = cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
ci = result.samples["ci"].values
assert ci.shape == (4, 1)
assert u.math.all(ci > 0.0 * u.mM)
```

## 固定值与 Nernst 模型

下列签名中的 `**channels` 是命名的运行时 Channel 子对象；使用 Cell 时由 paint 声明通道。

```text
SodiumFixed(size, E=50*u.mV, Ci=None, Co=None, valence=None, name=None, **channels)
PotassiumFixed(size, E=-95*u.mV, Ci=None, Co=None, valence=None, name=None, **channels)
CalciumFixed(size, E=120*u.mV, Ci=None, Co=None, valence=None, name=None, **channels)
SodiumInitNernst(size, temp=309.15*u.kelvin, Ci=None, Co=None, valence=None, name=None, **channels)
PotassiumInitNernst(size, temp=309.15*u.kelvin, Ci=None, Co=None, valence=None, name=None, **channels)
CalciumInitNernst(size, temp=309.15*u.kelvin, Ci=None, Co=None, valence=None, name=None, **channels)
```

| 参数 | 类型与含义 |
| --- | --- |
| `size` | 整数或形状序列，决定运行时离子状态形状 |
| `E` | 电压值或 shape initializer；Fixed 模型保持给定值 |
| `Ci`、`Co` | 摩尔浓度或 initializer；None 使用具体离子类的 default_Ci/default_Co |
| `valence` | 无量纲价态或 initializer；None 使用具体类 default_valence |
| `temp` | 绝对温度或 initializer，默认 36 摄氏度对应的 kelvin |
| `name` | 运行时节点名称，None 由模块命名机制处理 |

构造参数广播到 `size`，普通 initializer 接受 shape；Cell 声明的空间 callable 先按
[CVContext](../../filter/current/spatial-callable-parameters.md) 求值。模型间的 None 默认值见各家族源码，
不能用零替代。浓度必须为正才能使 Nernst 对数有定义，电价不能为零。

$$
E=\frac{RT}{zF}\log\frac{C_o}{C_i}.
$$

Fixed 模型直接持有 E；InitNernst 在初始化和 reset 时计算 E，此后改浓度不自动刷新缓存。
动态 Nernst 模型则在读取 E 时根据当前 Ci 求值。

## 动态钙浓度

完整签名中 `Constant` 来自 `braintools.init`：

```text
CalciumDetailed(size, temp=309.15*u.kelvin, d=1*u.um, tau=5*u.ms,
                C_rest=0.00024*u.mM, Co=None,
                Ci_initializer=Constant(0.00024*u.mM), name=None, **channels)
CalciumFirstOrder(size, temp=309.15*u.kelvin, alpha=0.13, beta=0.075,
                  Co=None, Ci_initializer=Constant(0.00024*u.mM), name=None, **channels)
derivative(Ci, V, total_current=None) -> concentration / time
```

Detailed 使用厚度 d 的膜下薄壳，将向内钙电流密度换算为浓度流入，再以 tau 回到 C_rest：

$$
\dot C_i=\max\left(\frac{I_{Ca}}{2Fd},0\right)+\frac{C_{rest}-C_i}{\tau}.
$$

CalciumFirstOrder 当前存在单位错误：alpha、beta 是裸数，derivative 却把 `alpha * total_current`
与 `0*u.mM` 比较，缺少电流密度到浓度导数的转换。传入带单位的总电流会抛出 UnitMismatchError；
无通道时 current 返回 None，也无法计算导数。它目前可构造，但不能用默认参数完成这条动力学路径。
源码依据见 [CalciumFirstOrder.derivative](../../../../braincell/ion/calcium.py)，修复由 [Ion TODO](../TODO.md) 跟踪。
`Ci_initializer` 接受浓度或 shape initializer；Ci 存储为 DiffEqState，`Co` 为外部浓度参数。
`total_current=None` 使用离子容器汇总电流，传值时使用提供的快照；该输入是电流密度。
Cell 的快照与更新顺序见 [离子调度](../../cell/current/architecture.md#离子电流快照与调度)。

## 生命周期和电流

```text
init_state(V, batch_size=None) -> None
reset_state(V, batch_size=None) -> None
compute_derivative(V) -> None
current(V, include_external=False) -> current density
pack_info() -> IonInfo
```

`init_state` 分配离子及子通道状态，reset 重新应用浓度初值并重置子通道；
compute_derivative 计算离子和子通道导数，不推进状态。运行时 E 是电压，Ci/Co 是浓度，
`pack_info()` 将这些当前值组成 IonInfo 供通道读取。current 汇总子通道的向内正电流，
`include_external=True` 同时纳入已注册的外部贡献。

独立构造的 Ion 要在给定 V 下初始化后再使用动态 Ci；Cell 自动管理这一过程。
IonInfo 的字段是本次读取的值，不是用于替换 owner 状态的视图。
所有状态的形状应与所属 Cell population/CV 布局一致；状态修改与参数训练分别通过
[Cell Views](../../cell/current/views.md) 和 [Trainable](../../optim/current/api.md) 完成。

模板生命周期的内部扩展钩子集中在 [KineticIon 与生命周期模板](kinetic-ion-api.md)。
现成模型的全部导出见 [ion/__init__.py](../../../../braincell/ion/__init__.py)，
模型方程出处见 [共享文献表](../references/ion-channel-bibliography.md)。
