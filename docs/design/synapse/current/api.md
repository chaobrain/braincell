# Synapse API

`braincell.synapse.ExpSyn` 和 Exp2Syn 计算突触内部状态及点电流；Cell 用户通过
`braincell.mech.Synapse` 声明位置，通过 [Connection](../../network/current/connections.md) 连接事件源。

## 独立状态示例

```python
import braincell as bc
import brainunit as u

syn = bc.synapse.ExpSyn(size=1, tau=2.0 * u.ms, e=0.0 * u.mV)
syn.init_state()
syn.apply_events(u.math.asarray([0.001]) * u.uS)
syn.compute_derivative()
assert u.math.allclose(syn.g.value, 0.001 * u.uS)
assert u.math.allclose(syn.current(-65.0 * u.mV), 0.065 * u.nA)
assert u.math.allclose(syn.g.derivative, -0.0005 * u.uS / u.ms)
```

完整 Cell 连接与运行例子见 [Network 最小用法](../../network/current/api.md#最小用法)。

## 构造和方法

```text
ExpSyn(size, name=None, tau=0.1*u.ms, e=0*u.mV)
Exp2Syn(size, name=None, tau1=0.1*u.ms, tau2=10*u.ms, e=0*u.mV)
init_state(V_post=None, batch_size=None) -> None
reset_state(V_post=None, batch_size=None) -> None
apply_events(payload, V_post=None) -> None
compute_derivative(V_post=None) -> None
current(V_post) -> point current
```

size 是运行时 packed rows 形状，Cell 自动按逻辑突触布局分配；name 为运行时节点名。
时间常数和 e 接受带单位值或 shape initializer，广播到 size。tau/tau1/tau2 必须正，
Exp2Syn 还要求 tau1 < tau2，否则构造参数校验抛出 ValueError。
init 分配状态，reset 将状态恢复到零，apply_events 就地增加状态，compute_derivative 只写导数。
V_post 是突触处电压；current 返回电流而不是电流密度。

## 方程与事件

ExpSyn 的 g 单位为 uS，事件 payload 也为 uS：

$$
g\leftarrow g+w n,\qquad \dot g=-g/\tau,\qquad I=g(e-V).
$$

n 是事件计数，w 是 Connection.weight；多个事件按 sum 累加，向内电流为正。
默认权重由 ScalarEventInput(u.uS) 决定，传错单位在连接/事件校验时失败。

Exp2Syn 保存 A、B 两个 uS 状态，g 是只读计算值 `B-A`：

$$
\dot A=-A/\tau_1,\quad \dot B=-B/\tau_2,\quad I=(B-A)(e-V),
$$
$$
t_p=\frac{\tau_1\tau_2}{\tau_2-\tau_1}\log(\tau_2/\tau_1),\qquad
f=\left(e^{-t_p/\tau_2}-e^{-t_p/\tau_1}\right)^{-1}.
$$

事件使 A、B 同时增加 `f*w*n`，使单个事件的峰值电导为 w。
记录 g 时注意 ExpSyn.g 是 State，Exp2Syn.g 是派生属性；可观测字段选择以
[Recording](../../network/current/recording.md) 的 runtime field 解析为准。
参数训练见 [Optim 支持度](../../optim/current/parameter-support.md)。
来源与数值用例见 [exponential.py](../../../../braincell/synapse/exponential.py)、
[exponential_test.py](../../../../braincell/synapse/exponential_test.py)。
