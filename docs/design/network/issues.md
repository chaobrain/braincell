# Network Design Issues

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
| I-09 | 稀疏 delay slot representation | `OPEN` |
| I-10 | 可学习 topology 与结构 mutation | `OPEN` |
| I-11 | 大规模 endpoint generator 优化 | `OPEN` |

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

## Open work

### I-09 Sparse delay slots

当前每个 target layout 使用 dense time ring，成本与最大 delay 和 layout width 相关。后续评估只保存
实际 event rows 的 sparse slots，并比较 JIT 静态 shape、scatter 成本和事件密度阈值。

### I-10 Trainable topology

当前只允许初始化前结构编辑和初始化后 shape-preserving 参数更新。可学习连接存在性、位置或新增/
删除 rows 会改变 JAX shapes，需要独立的 masked/padded 或重编译协议，不能复用普通参数训练接口。

### I-11 Scalable endpoint generators

当前通用 pairing 会按实际候选矩阵计算 conditional score；语义已经锁定，但大 N 下仍需增加不改变
结果的 score chunking、Bernoulli/all-to-all specialized generator，并记录 host peak-memory contract。
