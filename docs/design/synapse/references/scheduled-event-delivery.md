# 预定事件推进参考实现

NEURON、CoreNEURON 和 Arbor 将事件源推进、延迟调度和突触投递分层组织，避免每步匹配所有历史与未来事件。本文作为 [事件优化方案](../proposals/event-delivery-optimization.md) 的依据；采用的是推进与紧凑区间的思路，不移植各平台的事件边界语义。

核查日期：2026-09-08。源码快照：NEURON/CoreNEURON `bd4f3287089a680582d0350a6832fd7d996cd39b`；Arbor `c5206636b56d8e06ef6fe371c276ea1e3602aa7f`。论文描述对应发表版本，当前实现结论以固定提交源码为准。

## NEURON：逐次生成与延迟队列

NetStim 初始化安排首次自事件，每次触发后调用 `net_event(t)` 发出 spike，再计算间隔并安排下一次 `net_send`。官方 VecStim 示例则保存时间数组及 `index`，每次读取下一条时间。两者都不需要每步扫描完整 spike 时间表。

- [NetStim：INITIAL、NET_RECEIVE](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/src/nrnoc/netstim.mod)
- [VecStim：index、element、net_send](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/share/examples/nrniv/netcon/vecevent.mod)

源端产生 spike 后，连接按 delay 安排目标事件。NEURON 队列包含 splay tree 与固定步长 bin queue 支持；不能据此说所有配置默认使用分桶。动态队列中的桶保存已经安排的未来事件，与预编译完整模拟窗口的静态桶不同。

依据：[NEURON TQueue](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/src/nrncvode/tqueue.hpp)。

## CoreNEURON：CPU 调度与 GPU 批量投递

[Kumbhar et al., 2019, CoreNEURON](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2019.00063/pdf)，DOI `10.3389/fninf.2019.00063`，描述 CPU 负责事件队列与 MPI，GPU 执行批量突触事件处理。

当前源码可以核对完整边界：`PreSyn::send` 对连接安排 `time + delay`；`bin_event` 按配置进入 bin queue 或一般队列；`deliver_net_events` 消费到期事件，然后更新接收缓冲区并调用机制的批量处理入口。不能把 GPU 仿真概括为完全在 GPU 上管理动态优先队列。

- [调度、扇出与到期处理](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/src/coreneuron/network/netcvode.cpp)
- [splay tree、priority_queue 与 BinQ](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/src/coreneuron/network/tqueue.hpp)
- [接收缓冲区排序、CPU→GPU、发送缓冲区 GPU→CPU](https://github.com/neuronsimulator/nrn/blob/bd4f3287089a680582d0350a6832fd7d996cd39b/src/coreneuron/gpu/nrn_acc_manager.cpp)

接收缓冲区按目标实例分组，并维持同一实例的接收顺序以避免竞争。该分组面向有状态的机制事件处理，不能等同于任意模型都允许无序求和或共享内部状态。

## Arbor：区间推进与紧凑分桶

[Abi Akar et al., 2019, Arbor](https://arxiv.org/abs/1901.07454)，DOI `10.1109/EMPDP.2019.8671560`，提供 CPU/GPU 后端架构背景。下面的数据结构按当前固定提交核查。

显式 schedule 保存 `start_index_`，通过有序查找返回当前时间区间的事件，并推进进度；Poisson schedule 保存 `next_` 并增量生成。积分窗口开始时，来自各 cell 的事件 lane 被整理为按机制分组、按时间步划分的事件数组。

- [schedule 的区间访问](https://github.com/arbor-sim/arbor/blob/c5206636b56d8e06ef6fe371c276ea1e3602aa7f/arbor/schedule.cpp)
- [窗口初始化](https://github.com/arbor-sim/arbor/blob/c5206636b56d8e06ef6fe371c276ea1e3602aa7f/arbor/fvm_lowered_cell_impl.hpp)
- [event_stream_base 的 ev_data、ev_spans、mark](https://github.com/arbor-sim/arbor/blob/c5206636b56d8e06ef6fe371c276ea1e3602aa7f/arbor/backends/event_stream_base.hpp)

CPU 和 GPU 共用区间组织。CPU 引用主机事件数组，GPU 在初始化时异步复制事件数据到设备。每步通过区间起止位置取当步事件；事件数据不按最大桶大小填充。

- [CPU event_stream](https://github.com/arbor-sim/arbor/blob/c5206636b56d8e06ef6fe371c276ea1e3602aa7f/arbor/backends/multicore/event_stream.hpp)
- [GPU event_stream](https://github.com/arbor-sim/arbor/blob/c5206636b56d8e06ef6fe371c276ea1e3602aa7f/arbor/backends/gpu/event_stream.hpp)

例如 A 的到达步为 `[2, 2, 6]`，B 为 `[2, 7]`，可组织为：

```text
事件数组    [A, A, B, A, B]
第 2 步     [0:3]
第 3—5 步  [3:3]  空区间
第 6 步     [3:4]
第 7 步     [4:5]
```

有 T 个时间步、E 个事件时，主体存储为 O(T+E)。源端游标与投递端分桶可以同时存在，二者不是互斥的全局架构。

## 对 BrainCell 的采用边界

| 层次 | 参考方法 | 第一版实验 |
| --- | --- | --- |
| spike 生成 | NEURON 逐次生成、Arbor 增量 schedule | 保留同一份预生成 spike，隔离查询算法 |
| 到达步查询 | VecStim 进度、Arbor 区间 | 比较按连接游标与紧凑桶 |
| 突触投递 | CoreNEURON 批量机制处理 | 共用计数→权重→目标归约→ExpSyn |
| CPU/GPU | 共享组织与后端差异 | 两个设备实测，不提前指定赢家 |

BrainCell 固定步长的 `round_half_up((spike_time+delay)/dt)`、精度及半步 snapping 规则必须保持；Arbor 的区间约定不能直接作为 BrainCell 的量化规则。第一版使用 JAX 固定大小块读取紧凑桶，动态循环仅推进整数状态；训练检查覆盖权重与 ExpSyn 时间常数。源时间与 delay 的可微语义、非线性事件顺序、动态插入与 MPI 不属于本实验。

旧 [平台调研](../../network/references/platform-survey-2026-06.md) 保留为历史背景。本页是预定事件推进专题的固定版本依据，实测性能只从 [第一版结果](../../../../benchmarks/performance/synapse_events/results/scheduled-event-queries.md) 获取，不从外部平台的速度推断。
