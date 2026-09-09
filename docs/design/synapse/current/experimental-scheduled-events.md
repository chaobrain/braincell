# 实验性固定窗口事件查询

`braincell.experimental.scheduled_events` 提供四种 NetStim 查询方式，输出逐连接事件计数。它是实验接口，尚未接入 Network；[方案与限制](../proposals/event-delivery-optimization.md) 与 [完整实测](../../../../benchmarks/performance/synapse_events/results/scheduled-event-queries.md) 分别维护选型依据和性能结果。

```python
import brainstate
import brainunit as u
import jax.numpy as jnp
from braincell.network.event import NetStim
from braincell.experimental.scheduled_events import prepare_events

source = NetStim(size=1, start=1*u.ms, interval=2*u.ms, number=3, seed=7)
plan = prepare_events(source, [0], delay=1*u.ms, dt=1*u.ms,
                      n_steps=8, method="cursor")

@brainstate.transform.jit
def run(arrays, cursor, steps):
    return brainstate.transform.scan(
        lambda p, k: plan.query(arrays, p, k), cursor, steps)

cursor, counts = run(plan.arrays, plan.reset(), jnp.arange(8))
# counts[:, 0] == [0, 0, 1, 0, 1, 0, 1, 0]
```

`prepare_events(source, source_index, *, delay, dt, n_steps, start_step=0, method="cursor", block_size=128)` 返回 EventPlan。source 为 NetStim；source_index 是非空一维有效整数索引（每连接一项，可重复）；delay 为非负标量或同长向量时间量；dt 为正标量时间量；窗口为 `[start_step,start_step+n_steps)`，长度正，步索引受 int32 限制。时钟为 float32 时窗口右端还必须不超过 2**20：超过该范围，生产四 ULP 半步 snapping 可把整数比值映射到下一步；实验计划直接拒绝，避免按原始整数索引时错位。float64 时仍受 int32 索引上限约束。时钟乘法溢出、dt 在时钟精度中下溢为零等不可表示网格也被拒绝。method 为 current/scan/cursor/bucket，block_size 为正整数。

`plan.reset(start_step=None)` 返回全新整数状态，允许准备窗口内任意起点。`plan.query(arrays, state, step)` 返回 `(new_state, counts)`，counts 为 `(连接数,)` int32。连续推进时携带 new_state；回退或 seek 时 reset；同一计划可以重复运行。query 是编译内部操作，调用者必须保证 step 在窗口内且连续，避免每步主机检查。weights、目标映射和动力学由消费者处理。EventPlan.preparation 提供准备阶段各段耗时与临时数组逻辑字节数，不代表峰值内存。

source、delay、dt、拓扑和窗口在计划生命周期内固定；改变后重建，不能修改 source 的私有时间表或传入数组。prepare 验证有序有效时间、尾部 padding、单位、形状、有限值和范围。类型/单位错误为 TypeError（不兼容量纲由 brainunit 报错），值/范围错误为 ValueError。

scan 保存 `(C,K)` 到达步；cursor 保存带终止哨兵的 `(C,K+1)` 表和长度，运行状态为 C 个位置；bucket 保存 E 个窗口事件的连接索引、末尾一个 block padding 和 T+1 个 offsets。cursor 不合并源端查询。bucket 的持久查询数组为 O(E+T+block)，但准备暂存表及原始 source 仍可能为 O(CK)/O(SK)。

量化沿用生产 event_count 的设备精度与半步 snapping。重复事件保留，不提前乘权。动态循环只处理整数调度状态，实验训练消费者覆盖 weight/tau；没有定义 source time、delay 的可微接口。tau 为实验消费者传入的参数，梯度检查不等同于 Network TrainableManager 接入验收。

实现与测试：[scheduled_events.py](../../../../braincell/experimental/scheduled_events.py)、[测试](../../../../braincell/experimental/scheduled_events_test.py)。
