# 实验性预定事件直接投递

`braincell.experimental.scheduled_delivery` 把固定窗口的预定到达直接归约到目标输入，保留运行时权重与一阶反向梯度。它与第一轮 [逐连接查询](experimental-scheduled-events.md) 并列，生产默认路径不变。[方案](../proposals/event-delivery-optimization.md#第二轮三小时内的直接投递实验) 维护选型边界，[第二轮实测](../../../../benchmarks/performance/synapse_events/results/scheduled-event-delivery.md) 记录完整实验条件与性能数据。

```python
import brainstate
import brainunit as u
import jax.numpy as jnp
from braincell.network.event import NetStim
from braincell.experimental.scheduled_delivery import prepare_delivery

source = NetStim(size=1, start=1*u.ms, interval=2*u.ms, number=3, seed=7)
plan = prepare_delivery(source, [0, 0], [0, 1], n_targets=2,
                        delay=0*u.ms, dt=1*u.ms, n_steps=8)

@brainstate.transform.jit
def run(arrays, weights):
    return brainstate.transform.for_loop(
        lambda step: plan.deliver(arrays, step, weights), jnp.arange(8))

inputs = run(plan.arrays, jnp.array([0.1, 0.2])*u.uS)
# inputs[1] == [0.1, 0.2] uS; steps 3 and 5 have the same input.
```

## 准备与执行

`prepare_delivery(source, source_index, target_index, *, n_targets, delay, dt, n_steps, start_step=0, method="direct", block_size=128, max_bytes=256*1024**2, max_padding_ratio=8.0)` 返回 DeliveryPlan。

source、source_index、delay、dt 和窗口沿用第一轮准备合同，包括单位、有序时间、尾部 mask、半步 snapping 和 float32 窗口右端不超过 2**20 的限制。target_index 是与连接数同长的一维整数数组，值在 `[0, n_targets)`；多个连接可以指向同一目标。n_targets 是正整数。

| method | 设备数组 | 运行方式 |
| --- | --- | --- |
| direct | 压缩连接编号、整数重数、逐步 offsets、连接到目标映射 | 按 block_size 个记录循环，直接乘权并 scatter 到目标；尾部留一个块避免 dynamic_slice 起点钳制 |
| padded | `(n_steps, width)` 连接编号/重数矩阵、连接到目标映射 | 每步固定宽度并行投递，padding 重数为零 |

同一步同一连接的多条记录压缩为一个整数重数；不同连接保持独立权重，目标布局中的独立突触状态不会合并。padded 的 width 是窗口内单步最大压缩记录数，至少为 1；总数组超过 max_bytes，或填充槽位数/有效压缩记录数超过 max_padding_ratio 时抛出 `PaddingLimitError`，不会截断或自动切换。全空窗口豁免填充比例限制，但保留字节限制。

`plan.deliver(arrays, step, weights)` 返回形状 `(n_targets,)` 的 Quantity，单位与 weights 相同。weights 必须是有限物理量、形状为 `(连接数,)`，运行时读取；梯度参数使用浮点值。权重变化不要求重新准备。arrays 应由调用者作为动态 JIT 参数传入。step 是准备窗口内的绝对整数步号，调用者保证范围；没有游标状态，所以重跑、seek 或连续分段无需 reset。

source、source_index、目标、delay、dt 或窗口变化要求重新准备。调用者不得原地修改这些静态输入。一般输入错误为 TypeError/ValueError；不兼容量纲由 brainunit 报错。

## 梯度与成本

direct 的显式反向沿同一桶把目标输入的 cotangent 按整数重数累加到连接权重，避免让 JAX 对动态 while 自动反向。支持固定调度下权重的一阶反向，tau 的梯度由外部突触动力学传播；时间、delay 的梯度及 direct 的前向模式/高阶导数不在合同内。padded 使用常规 JAX 自动微分，统一验收范围仍是一阶反向。

preparation 字典记录第一轮量化/分桶、压缩/打包、上传耗时，以及到达数量、压缩记录数、固定宽度需求和最终数组字节。准备阶段仍使用完整连接事件表和主机压缩；这些临时数据与旧计划会增加峰值，array_bytes 不是峰值内存。direct 没有全连接计数中间向量，但仍产生完整目标输入，并不省去每步突触状态衰减或细胞积分。

## 真实模型实验适配器

`braincell.experimental.scheduled_cell.ScheduledDeliveryCell` 是 Cell 子类。构建、连接和 `Network.prepare_run` 与原 Cell 一致；在首次编译 rollout 前调用：

```text
cell.prepare_scheduled_delivery(*, method, dt, n_steps, start_step=0, block_size=128)
```

方法接受 current/scan/bucket/direct/padded，返回每条连接声明对应的计划 tuple。它只替换 Cell 的预定事件输入查询和目标归约；Network 的事件时序、积分器、记录和时间推进仍执行原代码。计划数组保存在只读使用的 brainstate State，进入编译 runner 的动态输入；连接权重每步通过原连接接口读取。

仅适用于物理量、加法聚合的事件缓冲；本轮以 ExpSyn 验证。一个实例只允许安装一次计划；改变网格、拓扑或调度后构造新实验实例。网络 reset 会重置模型状态，投递计划仍可复用。调用者须保证所有步号在准备窗口内，Network 不会自动续建本实验计划。生产 `event_backend` 选项主要管理 live spike 路由，不能代替本实验的预定事件方法选择。
