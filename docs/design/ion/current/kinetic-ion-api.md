# KineticIon 与生命周期模板

自定义 Ion 可复用 `braincell.ion._base` 中的生命周期 mixin。该路径是内部扩展入口，
这些类型没有从 `braincell.ion` 重导出。现成模型的公共用法见 [Ion API](api.md)，
状态、守恒和电流归属见 [KineticIon 架构契约](kinetic-ion.md)。

## 生命周期钩子

具体类同时继承一个离子家族和 mixin，在自己的构造器中调用以下初始化钩子：

```text
FixedIon._init_fixed_ion(*, Ci=None, Co=None, E=None, valence=None)
InitNernstIon._init_nernst_ion(*, Ci=None, Co=None, temp=None, valence=None)
DynamicNernstIon._init_dynamic_nernst_ion(*, Co=None, temp=None, valence=None, Ci_initializer=None)
KineticIon._init_kinetic_ion(*, Co=None, temp=None, valence=None,
                            species_initializers=None, solver=None, substeps=None)
```

FixedIon 要求 E；其余三种要求绝对温度 temp。缺少时抛出 ValueError。
None 的 Co、Ci、valence 使用具体家族的 class defaults。DynamicNernstIon 子类实现
`derivative(self, Ci, V, total_current=None)` 返回浓度导数。
这些钩子物化参数并配置初始化规则，运行时分配由 Ion.init_state 执行。

## 反应声明

```text
Factor(name, value)
Species(name, init, factor=None)
Reaction(lhs, rhs, forward, backward=None)
Source(target, flux)
Conserve(species, algebraic, total)
species_values() -> dict[str, quantity]
make_integration(V, recursive_child=True) -> None
```

| 声明 | 参数与回调 |
| --- | --- |
| Factor | name 是唯一名称；value 是 `(owner) -> factor`，如胞质体积 |
| Species | name 是状态名；init 为可见单位的值或 `(shape) -> initial` initializer；factor 可引用 Factor 名称 |
| Reaction | lhs/rhs 为 `species_name -> 正整数化学计量`；forward/backward 是 `(owner, V, values) -> rate coefficient` |
| Source | target 是物种名；flux 为 `(owner, V, values, total_current=None) -> scaled derivative` |
| Conserve | species 为物种名元组；algebraic 为被消去的物种；total 为 `(owner, V, values) -> scaled total` |

`values` 是可见物种量的字典。Source 回调会收到 `total_current` 关键字，即使规则不使用也应接受它。
各声明分别放进类属性 `factors/species/reactions/sources/conserves` 元组。
必须声明 `Ci`，它供 Nernst 电位和 IonInfo 使用；未知物种、重复定义和非法守恒引用在模板配置时拒绝。

若 `y_s = f_s * x_s`，其中 x 是可见浓度，f 是 Factor，则积分和守恒在 y 空间计算：

$$
\dot y_s=\sum_r \nu_{sr}J_r+S_s,\qquad
J_r=k_r^+\prod_s x_s^{\nu^-_{sr}}-k_r^-\prod_s x_s^{\nu^+_{sr}}.
$$

反应系数回调只返回 k，模板乘反应物浓度。其单位应让 J 与 scaled derivative 一致；
例如体积缩放后的二阶结合，需要 `volume / (concentration * time)`。
Source 直接给 scaled derivative。Conserve.total 与参与物种的 scaled value 同单位。
Factor 在一个积分步内作为既定转换量；不要在回调里推进另一套隐含状态。

`species_values()` 返回重建守恒物种后的可见单位字典。独立物种是 DiffEqState，
代数物种通过守恒写回；具体写回时机见 [架构契约](kinetic-ion.md)。
默认独立积分器是 backward_euler、substeps=1，外层 dt 从 brainstate 环境获取。
`uses_total_current=True` 请求总离子电流，来源与缓存规则同 [Ion API](api.md#动态钙浓度)。

## 最小结合模型与 Cell

这个例子把 `Ci + B <-> BC` 放到一个 Cell 中，体积设为 1，守恒量为 `B + BC`。
注册名称用于 mech.Ion 查找类；同一进程重复定义时需先注销旧的示例名称。

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion
from braincell.ion._base import KineticIon, Species, Reaction, Conserve

@bc.mech.register_ion("DocBindingCalcium")
class DocBindingCalcium(bc.ion.Calcium, KineticIon):
    species = (
        Species("Ci", 0.1 * u.mM),
        Species("B", 1.0 * u.mM),
        Species("BC", 0.0 * u.mM),
    )
    reactions = (
        Reaction({"Ci": 1, "B": 1}, {"BC": 1},
                 lambda self, V, x: 0.2 / (u.mM * u.ms),
                 lambda self, V, x: 0.1 / u.ms),
    )
    conserves = (Conserve(("B", "BC"), "B", lambda self, V, x: 1.0 * u.mM),)

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self._init_kinetic_ion(temp=309.15 * u.kelvin)

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1))
cell.paint(AllRegion(), bc.mech.Ion("DocBindingCalcium", name="binding"))
cell.record("bound", bc.observe.ion(name="binding").state("BC"))
cell.record("free", bc.observe.ion(name="binding").state("B"))
result = cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms)
assert result.samples["bound"].values.shape == (4, 1)
assert u.math.allclose(result.samples["bound"].values + result.samples["free"].values, 1.0 * u.mM)
```

模板实现和边界用例见 [_base.py](../../../../braincell/ion/_base.py)、
[_base_test.py](../../../../braincell/ion/_base_test.py)。真实的多壳层模型与 MOD 对照由
[小脑示例进度](../../../../validation/neuron/cerebellum-import-progress.md) 管理。
