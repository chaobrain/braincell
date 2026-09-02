# Network Architecture

## Ownership

Cell 是静态声明与 runtime 的 owner。一个 Cell population 内：

```text
paint declarations
  -> logical density owners (category, type, name, CV coverage)
  -> ChannelView / IonView rows (population, CV, owner)
  -> runtime SoA layouts

Synapse declarations
  -> _SynapseStore (logical IDs, locations, parameters)
  -> SynapseView
  -> runtime nodes grouped by synapse type

connect calls
  -> _ConnectionStore (SoA routing rows)
  -> ConnectionView

record calls
  -> RecordingSpec (spatial scope + observable + schedule)
  -> compiled RecordingSchema + layout-free gather
  -> immutable SampleBlock
```

同 type 的 Synapse 合并到一个 runtime SoA node，但 logical ID、name、location 和参数行保持独立。
Connection store 保存稳定 row ID、connect ID、source index、synapse ID、weight、delay 和 active mask。
Synapse 参数/state 不进入 Connection；weight/delay 不进入 Synapse。

Channel/Ion 不建立另一份显式 instance store。它们的 logical rows 由 density declarations、离散 CV coverage
和 population scope 派生，身份为 `(category, type, name, population, CV)`。View 只保存这些选择；初始化后
通过 layout metadata 映射到 runtime SoA buffers。

`NetworkConnections` 只保存 Network 引用并动态遍历 Cell populations。选择 target 后直接返回原始
Cell `ConnectionView`，因此没有跨 Cell row ID，也没有第二份 columns。

## Connection creation

`braincell.connect` 规范化 source 与 SynapseView，验证 target event-input contract 和单位，并向目标
Cell store 追加 rows。`Network.connect` 在其外增加：

1. source/target owner 必须已注册；
2. Network 尚未初始化；
3. 可选的 Synapse placement transaction；
4. topology cache invalidation。

快捷 placement 通过调用前后的 stable logical IDs 找到本次新建 Synapse。异常时恢复 place rules 和
origin metadata，并重新失效声明缓存。

可选 pairing spec 位于规范化和 store append 之间：

```text
unique source/synapse candidate views
  -> endpoint contexts
  -> temporary (source_position, synapse_position) columns
  -> weight/delay row broadcasting
  -> existing _ConnectionStore
```

固定行数策略包括两端独立 marginal sampling，以及先采一端、再按固定端分批计算 `(B, K)` score 的
conditional sampling。单侧 degree 策略先展开该侧 stub，再条件采 partner；双侧 degree 策略检查
每个分组的 stub 总和后随机匹配。`target_cell` 分组仅切分 Synapse 候选池，不改变 storage owner。

随机数由 `brainstate.random.RandomState` 提供。规则先获得一个 effective base seed，再按 stage、
target-cell group 和固定 endpoint ID 派生子流，因此 Network population 添加顺序和条件 score 的内部
batch 划分不会改变结果。显式规则 seed 不读取 Network seed。

## Runtime delivery

scheduled source 由目标 Cell 按绝对时间直接求 event count。live Cell source 在 Network setup 时
lower 为 `ConnectionBlock`，然后生成共享 target-layout delay queues：

```text
source crossing
  -> source population event vector
  -> sparse routing operator
  -> immediate target input or future ring slot
  -> scatter-add into Synapse runtime input
  -> vectorized Synapse dynamics
```

delay 保存为物理时间，在 run setup 按 dt 量化；支持 `nearest`、`ceil`、`floor`、`strict`。
scatter 与 brainevent operator 接收相同的完整 presynaptic population vector，delay grouping 只改变
operator lowering，不改变事件语义。

## Lifecycle

Network 只有 editable 和 initialized 两个外部状态。`init_state` 验证 source ownership，然后统一
初始化 Cell runtime。成功后不提供 build/deinit 或返回声明态的操作。`reset_state` 只重置动态状态、
时间、detector 和 queues。重复运行复用 setup 和 compiled loop caches。

## Mechanism views

空间选择顺序为 population -> region/locset/branch -> CV -> mechanism。Channel/Ion views 使用
`(type, name, population, CV)` logical rows；Ion 同时公开由 runtime class 决定的 species；SynapseView
使用 stable logical IDs。type 决定动力学实现，name 决定用户逻辑 owner/group，两者不能混为一个层级。

同 category、type、name 的 density paint 在离散后 CV 有交集即报错；无交集时属于同一 logical
owner。参数是否相同不参与冲突判断，修改通过 view `set()` 完成。Density views 不允许数字 row indexing；
Synapse/Connection views 的数字索引基于 stable IDs。

## Recording lowering

RecordingSpec 在 editable 阶段只保存 Cell-local name、静态空间 scope、observable selector 和 schedule。
它不参与 point placement，也不创建 mechanism。首次 run 已知 dt 并完成 runtime materialization 后，
每个 spec 编译为：

```text
CellView scope + observe selector
  -> Channel/Ion/Synapse logical rows or Cell/CV rows
  -> RecordingRow metadata + runtime gather function
  -> regular snapshots in the JIT run loop
  -> SampleBlock(values, time, RecordingSchema)
```

Channel/Ion state gathers 按 `(type, name)` owner 分组后写回原 logical order；Synapse state gathers 按 type
分组并通过 logical ID -> runtime row mapping 读取。`current(reduce="none")` 保留相同 contributor rows；
`reduce="sum"` 在 sampler 内按 `(population, CV)` 聚合，并把归约前 positions 写入 schema 的
`contributor_ids`。`membrane_current` 直接读取每个所选 CV 的总膜电流密度。

规则样本使用不可变 SampleBlock；EventSource 输出使用稀疏 EventSeries，不经过 RecordingSpec。NetworkResult
按 population/name 或 population/port 聚合二者。continued run 沿用全局 schedule；concat 要求相邻时间、
dt 和静态 schema 一致。旧 Probe 仅保留 deprecated 兼容路径。
