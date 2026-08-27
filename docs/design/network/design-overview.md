# Network Design Overview

BrainCell Network 只负责四件事：注册模型 owner、建立直接事件路由、统一初始化与推进、聚合结果。
Cell 持有 morphology、机制声明、Synapse/Connection SoA storage 和 runtime；Network 不复制这些数据。
Recording 同样由 Cell 声明，Network 只按 population name 收集规则 samples 与稀疏 events。

## 文档导航

- [API](./api.md)：面向用户的完整入口和示例。
- [Architecture](./architecture.md)：Cell-owned storage、事件调度和生命周期。
- [Issues](./issues.md)：已经锁定和仍开放的设计问题。
- [Implementation plan](./implementation-plan.md)：实现状态与验收项。
- [References](./references/platform-survey-2026-06.md)：其他平台的行为参考。

## 公开模型

```text
Network
  populations[name] -> Population(Cell | NetStim | EventSequence)
  connections[target_population] -> Cell-owned ConnectionView
  run result -> samples[population][recording] + events[population][port]

EventSource -- Connection(weight, delay) --> Synapse --> target Cell runtime
CellView -- RecordingSpec(observable, schedule) --> SampleBlock(schema, values)
```

`Synapse` 拥有 postsynaptic parameters、state 和 dynamics。`Connection` 只拥有 source routing、
weight 和 delay。一次命名 `connect` 调用可以批量产生多行 routing；connection 数量指命名调用数，
row 数量指实际稀疏路由数。

Cell/CellView 先选择 population 与空间，Channel/Ion/Synapse View 再选择机制 identity。Channel 使用
type/name，Ion 使用 species/type/name，Synapse 使用 type/name/stable IDs。Recording selector 复用同一
identity 模型，不创建另一套机制对象。

## v1 边界

- source 和 target 必须属于同一 Network，初始化后拓扑冻结。
- 连接已有 Synapse，或通过 `Network.connect` 快捷完成 place + connect。
- source/target 等长、`1 -> N`、`N -> 1` 自动对齐；任意显式 pairs 使用重复索引后的 views。
- `pairing=` 支持固定行数 marginal/conditional sampling、单侧 degree 和双侧 stub matching；它只生成
  临时端点索引，不进入 Network storage，也不重新引入第二套连接对象。
- v1 pairing 只消费已有 EventSourceView 与 SynapseView；不会从 Region 同时创建 Synapse。
- recording 只支持静态 schema；初始化前声明，运行中不能增加或改变记录行。
- 规则 state/current samples 与稀疏 source events 分开保存；legacy Probe 不是新接口的一部分。
- 不支持初始化后新增/删除机制、异质 morphology 或 Network batch runtime。
