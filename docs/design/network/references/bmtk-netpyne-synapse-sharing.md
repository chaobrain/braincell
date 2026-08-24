# BMTK / NetPyNE 的 Connection 与 Synapse 语义

> **Historical, non-normative reference (2026-08).** 本文记录 BMTK 和 NetPyNE
> 如何从 cell-pair connectivity 创建 Connection 与 Synapse，并提炼 BrainCell 需要覆盖的
> 更一般语义。正式接口和内部数据结构仍以 [Network Builder API](../api.md) 与
> [内部架构规范](../architecture.md) 为准。

## 1. 要区分的三个问题

讨论 `connection_rule=5`、`synsPerConn=5` 或 `all_to_one` 时，容易把三个不同层级混在
一起：

1. 哪些 source cell 与哪些 target cell 相连；
2. 一个 cell pair 生成多少条 event routes；
3. 这些 routes 指向多少个独立的 postsynaptic Synapse instances。

前两项不能唯一决定第三项。例如两个 source cell 都连接同一个 target cell 时，下面两种
网络具有相同的 cell-pair 统计，但 Synapse state ownership 不同：

```text
independent                         shared

source A -> Connection A -> Syn A  source A -> Connection A -+
                                                             +-> Syn X
source B -> Connection B -> Syn B  source B -> Connection B -+
```

因此，`(source_cell, target_cell, count)` 只能描述连接数量，不能完整表达 Synapse identity。

## 2. NetPyNE

### 2.1 Cell-pair rule

NetPyNE 的 `connParams` 先用 `preConds`、`postConds` 和
`probability`/`convergence`/`divergence`/`connList` 等规则选择 cell pairs，再用
`synsPerConn` 指定每个 cell-to-cell connection 创建多少个 individual synaptic contacts。
`synsPerConn=5` 默认表示同一 cell pair 创建五个 contacts；weight、delay、section 和
location 可以广播，也可以逐 contact 给值。

### 2.2 Synapse 是否共享

NetPyNE 默认设置 `simConfig.oneSynPerNetcon=True`。因此每个 NetCon 创建并绑定一个独立
Synapse object：

```text
cell pair, synsPerConn=2

source -> NetCon 0 -> Synapse 0
       -> NetCon 1 -> Synapse 1
```

设置 `oneSynPerNetcon=False` 后，NetPyNE 可以复用已经存在且具有相同 synapse mechanism
label、section 和 location 的 Synapse。于是多个 NetCon 可能指向同一 point process。

这个能力证明底层 NEURON 支持 Synapse sharing，但 NetPyNE 的表达仍有两个限制：

- sharing 由 simulation-wide boolean 控制，而不是逐 Connection group 声明；
- target identity 通过 mechanism/section/location 匹配推断，而不是由用户显式提供稳定的
  `synapse_id`。

因此用户很难在同一网络中清楚表达“这一组 routes 共享，但另一组同位置 routes 保持独立”。
参数或位置碰巧相同也不应天然等价于 state identity 相同。

## 3. BMTK

### 3.1 `connection_rule` 与 `iterator`

BMTK Builder 的 `connection_rule` 返回每个 source-target cell pair 的连接数量。
`connection_rule=5` 表示每个候选 cell pair 的 `nsyns=5`。

`iterator` 只改变 rule 如何看到和遍历候选 cells：

```text
one_to_one: rule(source_i, target_j) -> nsyns_ij

all_to_one: rule([source_0, source_1, ...], target_j)
            -> [nsyns_0j, nsyns_1j, ...]

one_to_all: rule(source_i, [target_0, target_1, ...])
            -> [nsyns_i0, nsyns_i1, ...]
```

`all_to_one` 因而适合控制一个 postsynaptic cell 的总入度或在全部候选 source 中联合采样，
但它不是“所有 source 共享一个 Synapse”的开关。iterator 最终仍生成逐 cell-pair 的
`(source_id, target_id, nsyns)`。

### 3.2 Runtime 实例化

BioNet 收到 `nsyns=N` 后会选择 N 个 target locations，调用 synapse factory N 次，并为每个
返回的 Synapse 创建一个 NetCon。默认 `Exp2Syn` factory 每次调用都新建
`h.Exp2Syn(...)`。因此：

```text
cell pair, connection_rule=2

source -> NetCon 0 -> Exp2Syn instance 0
       -> NetCon 1 -> Exp2Syn instance 1
```

即使两个实例随机选择到相同 section 和 location，它们仍是两个拥有独立动态状态的 point
processes。

BMTK Builder 可以在没有逐 Synapse properties 时把一个 cell pair 压缩保存为一条带
`nsyns=N` 的 edge row；存在逐 Synapse weight/location 等属性时则会展开 rows。这只是文件
表示优化，不会让运行时的 N 个 Synapse instances 共享状态。

标准 Builder/BioNet/SONATA 路径没有公开的 shared `synapse_id`，也没有让多条 edge rows
显式引用同一 postsynaptic Synapse 的接口。自定义 synapse factory 理论上可以自行缓存并返回
同一个 NEURON object，但这不是标准数据模型保证的语义，保存、检查和重建也无法可靠保留该
关系。

## 4. 两个平台的共同边界

| 能力 | NetPyNE | BMTK |
| --- | --- | --- |
| 先选择 source/target cell pairs | 支持 | 支持 |
| 指定每个 pair 的 contact 数量 | `synsPerConn` | `connection_rule -> nsyns` |
| 每个 route 默认创建独立 Synapse | 是 | 是 |
| 多个 routes 共享一个 Synapse | 全局开关下隐式匹配 | 标准 Builder API 不表达 |
| 用稳定 identity 显式选择已有 Synapse | 不提供 | 不提供 |
| 同时混用 shared 与 independent groups | 难以清楚表达 | 标准路径不能表达 |

二者的高层接口都很适合下面这条生成路径：

```text
cell-pair rule -> contact count -> create the same number of Synapses and routes
```

但这只是常用 recipe，不是 Connection/Synapse 关系的完整底层模型。cell-pair count 无法回答
多个 routes 是否共享 state，也无法从统计表无损重建共享关系。

## 5. BrainCell 要扩展的语义

BrainCell 应覆盖两个平台的便捷行为，同时把它们没有显式表达的 Synapse identity 提升为一等
概念。目标不是自动合并更多实例，而是让用户准确选择 identity 和 ownership。

### 5.1 独立保存两个事实表

概念上的规范关系是：

```text
Synapse instances
    synapse_id -> target cell + continuous location + model + parameters + state

Connections
    connection_id -> source endpoint + synapse_id + weight + delay
```

由 Connection 的 `synapse_id` 查到 target cell 后，可以聚合出 cell-pair connectivity；反过来
只有 `(source_cell, target_cell, count)` 时，不能恢复 Synapse sharing。

### 5.2 同时提供两类高层 recipe

第一类保留 BMTK/NetPyNE 熟悉的默认行为：

```text
generate pairs -> choose N locations -> create N new Synapses
               -> bind one Connection to each new Synapse
```

第二类允许用户先放置或选取已有 Synapses，再建立 routes：

```text
place/select Synapse X
bind source A -> Synapse X
bind source B -> Synapse X
```

最终 API 名称可以继续讨论，但必须能够分别表达 `new synapse per connection` 和
`connect to existing synapse`，而不是试图让一个 `nsyn` 数字同时承担两种含义。

### 5.3 显式共享，不按相等性自动合并

多个 declarations 即使具有相同 location、model、tau 和 reversal potential，也默认是独立
Synapse instances。只有 Connection 明确引用同一个 `synapse_id` 或 Synapse view 时才共享
动态状态。

这个规则避免以下问题：

- 用户以后把两个 tau 修改为不同值时，自动合并造成 identity 无法拆分；
- nonlinear、plastic 或 saturating mechanisms 被错误地当作可线性合并；
- 同位置的实验性独立实例因为参数碰巧相同而丢失独立 state。

### 5.4 State identity 与 SoA grouping 分离

独立 Synapses 不意味着逐对象执行。具有同一 mechanism layout 的 Synapse instances 仍可把
parameters 和 states 打包为 SoA arrays，由同一个 JAX kernel 批量更新：

```text
tau[N_syn], e[N_syn], g[N_syn], ...
```

因此需要分别处理：

- **semantic sharing**：多条 Connection 是否引用同一个 `synapse_id`；
- **runtime grouping**：多少个独立 Synapse instances 共用一个 vectorized kernel；
- **storage compression**：声明或文件是否压缩重复的 cell-pair 信息。

后两项是性能实现，不能改变第一项。

## 6. 对 BrainCell direct API 的含义

BrainCell 不保存第二套 cell-pair topology object。用户或独立 pairing helper 先产生重复保持的
source/synapse selections，再由 `connect` 生成 canonical routing rows。常用快捷入口可以同时创建
新 Synapse，但最终仍落到同一个 Cell-owned store。

这一层次允许：

- 为每条 Connection 分配新 Synapse；
- 从预先放置的 SynapseView 中选择 target；
- 让多条 Connections 引用同一个显式 Synapse；
- 从 Connection rows 派生 cell-pair 统计，而不把统计表变成 runtime 的第二事实来源。

因此 BrainCell 可以复现 BMTK/NetPyNE 常见的 cell-pair-first 结果，同时保留 NEURON 底层允许的
Synapse sharing，并让 JAX runtime 始终消费统一的 SoA rows。

## 7. 参考资料

### NetPyNE

- [User Documentation: Connectivity rules and `synsPerConn`](https://doc.netpyne.org/user_documentation.html)
- [`SimConfig.oneSynPerNetcon`](https://doc.netpyne.org/_modules/netpyne/specs/simConfig.html)
- [`CompartCell.addConn` implementation](https://github.com/Neurosim-lab/netpyne/blob/development/netpyne/cell/compartCell.py)

### BMTK

- [`BioNetBuilder.add_edges`](https://alleninstitute.github.io/bmtk/bmtk/bmtk.builder.html#bmtk.builder.network_builder.BioNetBuilder.add_edges)
- [Builder iterator implementation](https://github.com/AllenInstitute/bmtk/blob/develop/bmtk/builder/iterator.py)
- [Builder connection map and per-pair `nsyns`](https://github.com/AllenInstitute/bmtk/blob/develop/bmtk/builder/connection_map.py)
- [BioNet Synapse/NetCon instantiation](https://github.com/AllenInstitute/bmtk/blob/develop/bmtk/simulator/bionet/biocell.py)
- [Default NEURON synapse factories](https://github.com/AllenInstitute/bmtk/blob/develop/bmtk/simulator/bionet/default_setters/synapse_models.py)
