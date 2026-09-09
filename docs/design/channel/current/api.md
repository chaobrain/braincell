# Channel API

`braincell.channel` 提供现成通道和 HH、Markov 模板。通道计算膜电流密度；
`braincell.mech.Channel` 则是安装到 Cell 的声明，二者的关系见 [Mech](../../mech/current/api.md)。

| 任务 | 入口 |
| --- | --- |
| 使用现成模型 | [Cell 中的通道](#cell-中的通道)、[公开模型目录](../../../../braincell/channel/__init__.py) |
| 定义门控动力学 | [HH 与 Gate](#hh-与-gate) |
| 定义状态转移 | [Markov 与 Transition](#markov-与-transition) |
| 选择驱动力与温度处理 | [电流与辅助函数](#电流与辅助函数) |
| 校验和裁剪的实现依据 | [模板约束](template-invariants.md) |

## Cell 中的通道

下面安装一个钠通道，显式指定离子反转电位，并读取门状态。运行时通道的最后一轴是该 owner 覆盖的 CV；
前面的轴属于 Cell population。`g_max` 是单位面积电导，电压和浓度均需物理单位。

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1))
cell.paint(AllRegion(), bc.mech.Ion("SodiumFixed", E=50.0 * u.mV))
cell.paint(AllRegion(), bc.mech.Channel("Na_HH1952", name="na", g_max=1.0 * u.mS / u.cm**2))
cell.record("p", bc.observe.channel(name="na").state("p"))
result = cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
assert result.samples["p"].values.shape == (4, 1)
```

现成通道通过具体构造器声明参数，默认值随模型而异，完整模型签名和出处分别见
[源码目录](../../../../braincell/channel) 和 [共享文献表](../../ion/references/ion-channel-bibliography.md)。
参数的空间 callable 和 CV 上的求值规则见 [Filter](../../filter/current/spatial-callable-parameters.md)。

## HH 与 Gate

模板构造及共同方法的完整签名：

```text
HH(size, name=None)
OhmicHH(size, name=None)
GhkHH(size, name=None)
Gate(name, power=1, phi=None, q10=None, temp_ref=None, time_unit=u.ms, clip=False)
init_state(V, *ions, batch_size=None) -> None
reset_state(V, *ions, batch_size=None) -> None
compute_derivative(V, *ions) -> None
conductance_factor(V, *ions) -> dimensionless array
gate_phi(gate) -> dimensionless value
current(V, *ions) -> current density
```

`size` 是整数或形状序列。`V` 为电压数组，`*ions` 是与 `root_type` 对应的 IonInfo，提供 `E`、`Ci`、`Co`。
模板本身不提供具体速率，子类用类属性 `gates` 声明 Gate，并为每个门选一种速率形式：

| 回调方法，均为 `(self, V, *ions)` | 返回值 | 方程 |
| --- | --- | --- |
| `f_m_inf`、`f_m_tau` | 无量纲稳态值、时间常数 | `dm/dt = phi * (m_inf - m) / tau` |
| `f_m_alpha`、`f_m_beta` | 两个逆时间速率 | `dm/dt = phi * (alpha * (1-m) - beta * m)` |

回调可省略不使用的 IonInfo 位置参数；返回值应广播到门状态形状。
带单位的 tau/rate 按实际单位使用，裸数分别解释为 `time_unit`、`1/time_unit`。
`power` 决定电导因子 `product(gate ** power)`，不改变门方程。

`phi` 直接给温度倍率；或成对给 `q10`、`temp_ref`，由实例的绝对温度 `temp` 求倍率。
这三个字段支持值、实例属性名字符串和 `(owner) -> value` 回调，动力学求值时解析。
`phi` 与 q10 组合互斥，非法组合抛出 `ValueError`。`clip=True` 只裁剪电导求值使用的门值。

`init_state` 分配零值 DiffEqState；`reset_state` 写入当前电压下的稳态门值；
`compute_derivative` 写 `.derivative`，不推进 `.value`。独立使用时先 init 再 reset，
随后由积分器推进。Cell 自动管理这些调用。门名冲突、重复门、速率形式缺失或同时存在两套，
在类定义或状态绑定时报告 `ValueError`，规则见 [模板校验](template-invariants.md)。

这个独立示例定义一个门和固定反转电位，直接检查导数及电流：

```python
import braincell as bc
import brainunit as u

class DemoHH(bc.channel.OhmicHH):
    root_type = bc.HHTypedNeuron
    gates = (bc.channel.Gate("m", power=2),)

    def __init__(self, size=1):
        super().__init__(size=size)
        self.g_max = 0.1 * u.mS / u.cm**2
        self.E = 0.0 * u.mV

    def f_m_inf(self, V):
        return 1.0 / (1.0 + u.math.exp(-(V + 40.0 * u.mV) / (10.0 * u.mV)))

    def f_m_tau(self, V):
        return u.math.ones_like(V / u.mV) * 2.0 * u.ms

    def reversal_potential(self, V, *ions):
        return self.E

channel = DemoHH()
V = u.math.asarray([-65.0]) * u.mV
channel.init_state(V)
channel.reset_state(V)
channel.compute_derivative(V)
assert channel.m.value.shape == (1,)
assert u.math.allclose(channel.m.derivative, 0.0 / u.ms)
assert u.math.all(channel.current(V) > 0.0 * u.uA / u.cm**2)
```

## Markov 与 Transition

```text
Markov(size, name=None, solver=None, substeps=None)
OhmicMarkov(size, name=None, solver=None, substeps=None)
Transition(src, dst, forward, backward=None)
init_state(V, *ions, batch_size=None) -> None
reset_state(V, *ions, batch_size=None) -> None
reset_steady_state(V, *ions, batch_size=None) -> None
state_values() -> dict[str, array]
compute_derivative(V, *ions) -> None
make_integration(*args, **kwargs) -> None
```

`pairs` 是 Transition 元组。`src`、`dst` 是状态名，`forward`、`backward` 是实例速率方法名；
省略 backward 表示单向反应。速率回调 `(self, V, *ions)` 返回逆时间，裸数解释为 `/ms`。
`dependent_state` 必须明确指定一个已声明状态，否则类定义抛出 `ValueError`。

对 `C <-> O`，设速率为 alpha、beta，总概率 `conserve=1`：

$$
\dot O=\alpha C-\beta O,\qquad C=1-O.
$$

只积分独立状态；`state_values()` 返回包含重建依赖状态的字典。默认 reset 将独立状态置零，
`reset_to_steady_state=True` 改为求定常分布。`clip_states=True` 默认裁剪动力学求值所用的独立状态，
不覆盖储存值。`conserve` 可为值、属性名或 owner 回调。
`solver=None`、`substeps=None` 分别采用类默认 `backward_euler`、1；substeps 必须至少为 1。
独立积分通过环境的 `t`、`dt` 获取时间，协议见 [Quad](../../quad/current/api.md)。

```python
import braincell as bc
import brainunit as u

class DemoMarkov(bc.channel.OhmicMarkov):
    root_type = bc.ion.Potassium
    pairs = (bc.channel.Transition("C", "O", "alpha", "beta"),)
    dependent_state = "C"
    open_states = ("O",)

    def __init__(self, size=1):
        super().__init__(size=size)
        self.g_max = 0.1 * u.mS / u.cm**2

    def alpha(self, V, K):
        return u.math.ones_like(V / u.mV) * 0.2 / u.ms

    def beta(self, V, K):
        return u.math.ones_like(V / u.mV) * 0.1 / u.ms

channel = DemoMarkov()
V = u.math.asarray([-65.0]) * u.mV
K = bc.ion.PotassiumFixed(size=1).pack_info()
channel.init_state(V, K)
channel.reset_state(V, K)
channel.compute_derivative(V, K)
states = channel.state_values()
assert u.math.allclose(states["C"] + states["O"], 1.0)
assert u.math.allclose(channel.O.derivative, 0.2 / u.ms)
```

## 电流与辅助函数

HH、Markov 只定义状态动力学，具体子类需实现 `current(V, *ions)`。
OhmicHH 和 OhmicMarkov 共享 `g_max * conductance_factor * (E - V)`，向内为正。
默认从第一个 IonInfo 取 E；直接附着于神经元的固定 E 通道可覆盖 `reversal_potential`。
OhmicMarkov 的 `open_states=("O",)` 决定哪些概率相加形成电导因子。

| 完整签名或属性 | 行为 |
| --- | --- |
| `q10_factor(q10, temp, temp_ref)` | 返回 `q10 ** ((temp-temp_ref)/(10*u.kelvin))`；温度为绝对温度，倍率无量纲 |
| `freeze_gradient(value)` | 返回数值、形状、单位不变的值，停止经过它的梯度 |
| `ghk_flux(V, ci, co, z, temp)` | 电压、内外摩尔浓度、无量纲价态、绝对温度；返回尚未乘 permeability 的恒场通量 |
| `GhkHH.permeability()` | 默认返回 `self.g_max`，其单位需与通量相乘得到电流密度；不是 ohmic 电导单位 |
| `GhkHH.current(V, *ions)` | 使用首个 IonInfo 的 Ci/Co，以及 self.z、self.temp，计算 `-permeability * gate_factor * ghk_flux` |
| `GhkHH.freeze_drive_gradient=False` | 为 True 时仅冻结驱动力中的 V，门控的电压依赖保留 |

GHK 在零电压附近使用解析极限分支。具体实现及测试见
[_base.py](../../../../braincell/channel/_base.py)、[_base_test.py](../../../../braincell/channel/_base_test.py)。
