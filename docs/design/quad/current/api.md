# Quad API

`braincell.quad` 把注册名或 callable 解析为状态积分步骤。Cell 选择 solver 后自动调用；
独立 DiffEqModule 使用 brainstate 时间环境及编译后的步函数。

## 最小用法

```python
import braincell as bc
import brainstate
import brainunit as u

class Decay(brainstate.nn.Module, bc.DiffEqModule):
    def __init__(self):
        super().__init__()
        self.x = bc.DiffEqSingleState(1.0 * u.mV)

    def compute_derivative(self):
        self.x.derivative = -self.x.value / (2.0 * u.ms)

model = Decay()
step = bc.quad.get_integrator("rk4")

@brainstate.transform.jit
def advance():
    with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
        step(model)
    return model.x.value

value = advance()
assert u.math.abs(value - u.math.exp(-0.05) * u.mV) < 1e-5 * u.mV
```

## 解析、注册与调用

```text
get_integrator(method) -> callable
register_integrator(name, *, aliases=(), category="general", order=None,
                    description="", deprecated=False, override=False) -> decorator
step(target, *args) -> None
```

method 为注册名或 callable；callable 原样返回，未知名称抛出 KeyError 并给近似名称建议。
注册装饰器记录名称、别名、类别、阶数和描述，返回原函数；override 控制重复注册是否覆盖。
registry 对象和只读 all_integrators 查询表见 [_registry.py](../../../../braincell/quad/_registry.py)。

target 遵循 DiffEqModule 协议，并是可遍历状态的 brainstate Module/Node。
DiffEqState 是状态协议，实际使用 DiffEqSingleState 或 DiffEqGroupState；args 传给目标导数/阶段钩子。
step 读取环境 t、dt，就地写状态，不为外层推进时钟。多步驱动使用
brainstate.transform.for_loop/scan，Cell/Network.run 已封装此循环。
完整目标协议见 [Cell 积分协议](../../cell/current/api.md#积分协议)。

| 家族 | 步函数 |
| --- | --- |
| 显式 | euler_step、midpoint_step、rk2/3/4_step、heun2/3_step、ralston2/3/4_step、ssprk3_step |
| 隐式 | backward_euler_step、implicit_euler_step |
| 指数 | exp_euler_step、ind_exp_euler_step |
| Cell staggered | staggered_step |

函数名去掉 `_step` 为常用注册名，实际别名以 registry 为准。不同积分器对目标的要求不同，
staggered 需要电压与机制分步接口；将它用于任意只有 compute_derivative 的模型会失败。
`dhs_voltage_step`、dense_voltage_step、sparse_voltage_step 是电压系统后端，
调用与数组契约见 [staggered 实现](../../../../braincell/quad/_staggered.py)，不与普通 target-step 签名混用。

Cell 显式路径当前遗漏边界输入反馈，例子和缺项见
[边界输入提案](../../cell/proposals/explicit-solver-boundary-inputs.md)。
