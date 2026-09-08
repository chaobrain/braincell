# Trainable Parameter Architecture

## 目标

参数系统需要让 BrainCell 模型暴露可训练自由度，同时不改变现有 mechanism 的动力学
实现和 runtime owner。它必须同时支持：

- 直接训练一个物理字段；
- 多个 runtime row 共享一个自由度；
- 保持当前空间分布、只训练比例；
- runtime field 由一个或多个 latent 参数和 `CVContext` 生成；
- BrainState graph 自动发现 roots；
- gradient-based 和 gradient-free 方法写入同一个 ParameterSet。

公共调用合同见 [API](api.md)，落地顺序见 [Implementation plan](implementation-plan.md)。

## 三层参数模型

```text
optimizer raw variable z
        | brainstate.nn.Param transform
        v
physical root q / latent theta
        | ParameterBinding
        v
runtime physical field, for example g_max
```

BrainState 管理第一层 transform、ParamState 和 autodiff。BrainCell 只管理 root 的空间
含义以及 root 到 runtime field 的映射。

root 不一定是原模型参数：

- direct：`g_max[row] = q[group(row)]`；
- scale：`g_max[row] = baseline[row] * theta[group(row)]`；
- function：`g_max[row] = f(ctx[row], a[group(row)], b[group(row)], fixed)`。

只有 root 是 ParamState。runtime field 是 materialized physical buffer，不进入 optimizer
tree。`brainstate.graph.states(cell, ParamState)` 因而看到 `q/theta/a/b`，不会看到每个 CV
已经计算出的 `g_max`。

## 所有权与隔离

实现集中在独立的 `braincell.trainable` 模块。P0 由 Cell 持有 `trainables` 门面：

```text
Cell.trainables -> TrainableManager
  root registry
  ParameterBinding collection
  ParameterSet construction
  materialization

Future: Network.trainables -> aggregate TrainableManager
```

Cell manager 是实际 owner，并作为 Cell graph 的子 module 保存 `nn.Param`。未来 Network
manager 只形成跨 Cell 聚合 view，不复制 roots。

这使核心 Cell 的侵入保持在两个边界：

1. 构造并暴露一个 `trainables` manager；
2. 在 init/reset/run 的既定位置调用 manager materialization。

View 只把 target selection 和 source 交给目标 Cell manager，不保存 optimizer state。

## 不同 View 的归属

当前仓库的物理 ownership 已经适合统一参数系统：

| View | 实际 storage owner | Binding owner |
| --- | --- | --- |
| ChannelView | target Cell density/runtime layout | target Cell |
| IonView | target Cell density/runtime layout | target Cell |
| SynapseView | target Cell SynapseStore/runtime node | target Cell |
| ConnectionView | target Cell ConnectionStore/runtime delivery | target Cell |

`NetworkConnections` 只聚合查询 Cell-owned ConnectionView，没有第二份 connection columns。
因此未来训练 connection weight 时，`ConnectionView.trainable(weight=...)` 仍注册到目标
Cell manager。未来全网优化仍应聚合 Cell managers，不在 Connection 或 Population 上增加
独立 manager。

delay 影响离散调度和 queue schema，不作为首批连续 trainable field。weight 只有在
runtime delivery buffer 能保持 JAX trace 和固定 shape 后才接入。

## Parameter Schema

Channel、Ion 和后续 Synapse 统一使用显式字段角色：

| Role | 含义 | 例子 | 可训练 |
| --- | --- | --- | --- |
| `parameter` | 连续、shape-preserving、进入 forward | `g_max`, `E`, `tau`, `V_sh` | 候选 |
| `state` | 随时间积分或由 reset 初始化 | gates, dynamic concentration | 否 |
| `derived` | 根据 parameter/state 计算 | Nernst-derived `E`, factor | 否 |
| `static` | 决定结构、shape 或调度 | solver, substeps, valence, topology | 否 |

P0 采用显式 `ParameterSpec` 白名单，不根据构造函数签名或数值 dtype 自动开放。spec
只保存 unit/default prototype 和 validator；是否开放训练属于 owner binding 策略，不污染
物理 schema。Synapse 现有 spec 迁移为公共 schema，但首批 trainable owner 只有三个
Channel 类。

## Runtime 参数列

schema Channel 的物理参数由非可训 `RuntimeParameterState(LongTermState)` 保存。状态内部按
`uniform`、`population`、`cv` 或 `row` 保存最小值，Channel 读取时才广播为执行矩形；旧的
`get_state()` 和 buffer inspection 仍获得矩形兼容视图。point mask 在读取 conductance 时
应用，因此 scalar `g_max` 不必为未 paint 点分配完整数组。

首次 materialization 冻结参数列的轴语义。optimizer 之后只改变同 shape 的 state value，
不会因数值从相同变为不同而触发第二步 JIT 重编译。

## TrainableManager

Cell manager 内部维护：

```text
TrainableManager
  roots             stable name -> nn.Param
  bindings          ordered ParameterBinding collection
  target ownership  logical row/field -> binding
  parameters()      ParameterSet facade
  materialize()     evaluate and scatter all bindings
```

注册是事务性的：source 解析、schema、单位、group、root name 和 target overlap 全部验证
成功后才提交。一个逻辑 row/field 只能由一个 binding 消费；同一个 `nn.Param` 可以被多个
binding 引用，并按对象身份去重。

自动生成的 root name 来自稳定 owner、field、group 或 function argument。显式名称优先，
重复名称指向不同对象时失败。共享对象的多个显式名称不一致时同样失败。

## ParameterBinding

binding 是 source 与 runtime target 之间的持续关系，至少保存：

```text
target category / owner / field
stable selected row keys
root references
group and gather indices
expected output unit and shape
source evaluator
optional frozen baseline
```

binding 只需要 forward materialization，不要求 inverse/reduce。direct 的物理 setter 通过
root transform inverse 完成；任意 latent function 不存在通用的 target-to-root 逆映射。

普通 `View.set()` 与 binding 的区别是：

- `set()` 立即修改声明 override 或 runtime buffer；
- `trainable()` 注册持续关系；
- optimizer 只更新 root；
- `materialize()` 根据当前 root 刷新 target。

已由 binding 占有的 target 不允许普通 set 静默替换关系。最终实现应明确拒绝，或要求
用户先删除 binding；P0 不提供初始化后改变 ownership。

## Grouping 与广播

View selection 展开为稳定 logical rows。P0 的 group key 为：

| Group | Root identity |
| --- | --- |
| `row` | `(population, cv, owner-row)` |
| `population` | population member |
| `cv` | CV identity |
| `all` | one constant key |

group engine 根据 row metadata 创建 `row -> root index` gather，不依赖矩形 reshape，因而
可以扩展到 ragged selection。

direct source 从当前 target 初始化 grouped root 时，同组物理值必须相等；否则不平均。
scale source 将当前不同值保存为 row-aligned baseline，只共享 factor。

对 `P=4, C=10` 的 selection：

```text
row         40 DOF
population   4 DOF
cv          10 DOF
all          1 DOF
runtime     40 rows in every case
```

branch、region、组合 key 和用户 grouper 延后，直到稳定 row metadata 可以为 checkpoint
提供可靠 fingerprint。

## Parameterized Function

custom source 是一次 deferred normal function call。binding 按函数签名绑定 trainable
和 fixed arguments，并为每个 logical row 提供 `CVContext`。

- `nn.Param(fit=True)` 是 root，调用时传 `.value()`；
- `fit=False` 或普通 Quantity/array 是 fixed argument；
- grouped nested source 在调用前按当前 row gather；
- raw vector Param 作为完整函数参数，不靠 shape 猜测 population/CV 轴；
- callable output 必须与 target unit 兼容且 shape 固定；
- 热路径不得转换为 NumPy 或 Python scalar。

scalar `a/b` 表示全 selection 共 2 DOF。若 `a/b` 按 population 分组，4 个 population
member 共 8 DOF；CV 变化来自 `ctx`，不是自动增加的 latent 轴。

任意函数没有通用 inverse。用户负责 root 初值；系统不会根据旧 `g_max` 反求 `a/b`，
也不会把最终 target bounds 反推到 latent bounds。

## Transform、Bounds 与单位

root 的约束完全由 `brainstate.nn.Param.transform` 管理。BrainCell 不把 bounds 固定为
sigmoid，也不再维护第二份 target physical bounds。

lower/upper 按 root shape 广播，而不是 runtime row shape。例如 population group 的 root
shape 是 `(P,)`，可以使用 scalar 或 `(P,)` bounds。

direct root 保留 target 的物理单位；scale factor 通常无量纲；custom root 由函数合同决定。
进入训练不需要整体去单位。只有 loss 在选择 canonical unit 或观测尺度后形成无量纲
scalar。

多个 runtime values 依赖同一 latent 时，逐 target bounds 会形成耦合约束。P0 不自动求
约束交集；用户直接约束 latent，或构造始终满足条件的 parameterized function。

## Materialization 生命周期

现有 density callable 在 lowering 时求值一次，不能承载 optimizer 持续更新的 latent。
TrainableManager 必须提供新的 JAX-traceable materialization 路径：

1. `init_state()`：runtime buffer 建立后、mechanism state 初始化前；
2. `reset_state()`：先物化，再重置 gates 和 ion dynamic state；
3. `Cell.run()`：rollout 入口保证当前 roots 已同步；
4. 直接连续调用 `update()` 时不在每一步求值，用户在 root 更新后显式 materialize 一次。

materialization 必须位于 differentiated trace 内。若参数影响 reset 初值，仅在 run 开始后
刷新已经太晚，因此 reset 入口也必须接入。

`reset_state()` 不改变 roots 或 frozen baseline。完整 `Cell.reset()` 清除 runtime；重新
初始化必须从仍有效的声明和 manager metadata 重建 binding target，旧 runtime 引用不可复用。

## 与 Jaxley 的对比

完整实现调研见 [Jaxley Parameter Model](references/jaxley-parameter-model.md)。本架构只冻结
直接影响 BrainCell 的结论：

- 使用显式 parameter/state schema，而不是从构造函数或 dtype 猜测；
- View selection 和稳定 row metadata 决定 sharing；
- 低维 root 通过 gather/scatter 映射到 dense runtime values；
- BrainCell root 由 `nn.Param` graph state 持有，不要求显式 `simulate(params)`；
- 保留物理单位、支持 arbitrary latent function，且不平均不一致的 grouped direct 初值。

## 架构不变量

- root 才是 ParamState，runtime target 始终是物理 materialized value。
- manager 是参数系统唯一 owner，View 和 ParameterSet 不复制 state。
- direct、scale 和 function source 共享一条 binding 主链。
- 参数 ownership、group、row fingerprint 和 output shape 在 JIT 内固定。
- reset dynamic state 不得回滚 roots。
- 任意 callable 不作为 checkpoint 数据序列化。
- 不在 BrainCell 内实现 optimizer 或通用局部最优证明。
