# Network 决定历史快照

归档日期：2026-09-07。来源：`docs/design/network/current/decisions.md` 的整理前工作区版本。
以下保留原记录；其中的实现状态和测试数量没有在归档时重新验收。现行入口为 [Design TODO](../design/TODO.md)。

---

# Network 已实现设计决定

本文保留 I-01 至 I-08 的既有决定及编号。具体契约见 [API](../design/network/current/api.md) 和 [架构](../design/network/current/architecture.md)，
验证记录见 [实现与验证记录](../design/network/current/implementation-status.md)。
开放问题 I-09 至 I-11 已移至 [运行时扩展提案](../design/network/proposals/runtime-extensions.md)，由 [Network TODO](../design/network/TODO.md) 跟踪。

| ID | Topic | Status |
| --- | --- | --- |
| I-01 | Cell-owned Synapse/Connection SoA 与 Network 聚合 | `LOCKED` |
| I-02 | connection call、row 与名称作用域 | `LOCKED` |
| I-03 | weight/event-input 单位与符号 | `LOCKED` |
| I-04 | 初始化生命周期与拓扑冻结 | `LOCKED` |
| I-05 | delay 量化与 continued run | `LOCKED` |
| I-06 | density paint CV overlap | `LOCKED` |
| I-07 | recording selector 与 current reduction | `LOCKED` |
| I-08 | endpoint pairing 语义与 RNG | `LOCKED` |

## Locked decisions

- Synapse 拥有 postsynaptic dynamics；Connection 拥有 source routing、weight 和 delay。
- 每次命名 connect call 可以生成多行；名称在目标 Cell 内唯一，row ID 稳定且删除后不复用。
- scalar event target 的 weight 必须与 model `event_input` 单位兼容，默认值为 `1 * unit`；允许负值。
- Network source/target 必须先注册。初始化后结构冻结，reset 不返回编辑态。
- connection 数量是 active named calls；实际稀疏规模单独报告 rows。
- density owner 的 CV overlap 直接报错；不比较参数，不执行后写覆盖。
- observable selector 明确区分 `type`、`name`、ion `species` 和 synapse stable `ids`；一次调用最多选择
  一个 identity 维度，空间范围由调用 `record()` 的 CellView 独立决定。
- state 保留 logical mechanism rows；current 默认按 `(population, CV)` 求和，`reduce="none"` 保留
  contributor rows，归约后的 schema 记录 contributor positions。
- RecordingSpec 在初始化前声明、首次 run 时按 dt 编译；规则 samples 使用带静态 RecordingSchema 的
  SampleBlock，EventSource 输出单独使用稀疏 EventSeries。
- pairing 只物化临时 endpoint positions，最终写入普通 Connection rows；不建立 topology owner。
- fixed-count、one-sided degree 与 dual-stub matching 是三种独立行数语义。
- target-cell grouping 只分割 Synapse pool；候选 views 必须 unique，输出 rows 可以重复。
- Network 隐式 seed 由 Network seed 和 canonical connection path 派生；显式 rule seed 完全覆盖。
