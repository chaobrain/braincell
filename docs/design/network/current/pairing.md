# Network Endpoint Pairing

配对规则将已有端点候选集变成 Connection rows。先按所需行数选择策略，再定义 score 和分组；调用入口见 [Connections](connections.md)。

## 最小用法

两个源和两个已有突触，从候选池独立抽取三行连接。规则只生成路由索引，不增加 Synapse。

```python
import braincell as bc
import brainunit as u
from braincell.filter import RootLocation

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), pop_size=2, cv_policy=bc.CVPerBranch(1))
cell.place(RootLocation(0.5), bc.mech.Synapse("ExpSyn", name="ampa"))
net = bc.Network(seed=7)
pre = net.add_population("pre", bc.NetStim(size=2))
post = net.add_population("post", cell)
rows = net.connect("sampled", source=pre, synapse=post.synapses["ampa"],
                   pairing=bc.network.connection.independent(3, seed=8), weight=0.001*u.uS)
assert len(rows) == 3
assert len(post.synapses["ampa"]) == 2
```

## Endpoint Pairing

`pairing=` 从已有 source 和 Synapse candidate views 中生成临时局部索引，最终仍写入普通 Connection
rows，不建立第二套 topology 或 storage。它当前只接受已存在的 `SynapseView`。

### Strategy comparison

| Helper | Row count | Sampling order | Typical use |
| --- | --- | --- | --- |
| `independent(number, ...)` | 固定为 `number` | source 与 Synapse 独立采样 | 已知总 Connection 数。 |
| `source_first(number, ...)` | 固定为 `number` | 先 source，后条件采样 Synapse | Synapse 偏好依赖已选 source。 |
| `synapse_first(number, ...)` | 固定为 `number` | 先 Synapse，后条件采样 source | source 偏好依赖已选 Synapse。 |
| `by_source(degree, ...)` | source degrees 之和 | 每个 source 采样其 Synapse partners | 指定出度。 |
| `by_synapse(degree, ...)` | Synapse degrees 之和 | 每个 Synapse 采样其 source partners | 指定入度。 |
| `match_degrees(source_degree, synapse_degree, ...)` | 两侧 degree 和 | 展开两侧 stubs 后随机匹配 | 同时固定两侧 degree sequence。 |

### `braincell.network.connection.independent`

```text
braincell.network.connection.independent(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace=True,
    synapse_replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

固定总 row 数，分别从 source 和 Synapse candidate pools 独立采样。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 总 rows，或每个 `target_cell` group 的 rows。 |
| `source_score` | callable or `None` | `None` | source 边际非负权重。 |
| `synapse_score` | callable or `None` | `None` | Synapse 边际非负权重。 |
| `source_replace` | `bool` | `True` | source pool 是否放回采样。 |
| `synapse_replace` | `bool` | `True` | Synapse pool 是否放回采样。 |
| `group_by` | `None or "target_cell"` | `None` | 是否按 target cell 独立运行规则。 |
| `seed` | `int or None` | `None` | 显式局部 seed；给定后覆盖 Network seed 派生。 |

### `braincell.network.connection.source_first`

```text
braincell.network.connection.source_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace=True,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

先采样 source，再让 `synapse_score(ctx)` 在已选 source 条件下为 Synapse candidates 赋权。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 生成 rows 数。 |
| `source_score` | callable or `None` | `None` | 第一阶段 source 边际权重。 |
| `synapse_score` | callable or `None` | `None` | 第二阶段条件 Synapse 权重。 |
| `source_replace` | `bool` | `True` | 第一阶段 source 是否放回。 |
| `replace` | `bool` | `True` | 同一固定 source 的 Synapse partners 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.synapse_first`

```text
braincell.network.connection.synapse_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    synapse_replace=True,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

先采样 Synapse，再让 `source_score(ctx)` 在已选 Synapse 条件下为 source candidates 赋权。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 生成 rows 数。 |
| `synapse_score` | callable or `None` | `None` | 第一阶段 Synapse 边际权重。 |
| `source_score` | callable or `None` | `None` | 第二阶段条件 source 权重。 |
| `synapse_replace` | `bool` | `True` | 第一阶段 Synapse 是否放回。 |
| `replace` | `bool` | `True` | 同一固定 Synapse 的 source partners 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.by_source` and `by_synapse`

```text
braincell.network.connection.by_source(
    degree,
    *,
    synapse_score=None,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec

braincell.network.connection.by_synapse(
    degree,
    *,
    source_score=None,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

分别为每个 source 指定下游 Synapse 数，或为每个 Synapse 指定上游 source 数。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `degree` | non-negative integer scalar, array, or callable | required | 每个固定 endpoint 的 partner 数；callable 签名为 `(ctx, rng) -> counts`。 |
| `synapse_score` / `source_score` | callable or `None` | `None` | partner candidate 权重。 |
| `replace` | `bool` | `True` | 同一个固定 endpoint 内 partner 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.match_degrees`

```text
braincell.network.connection.match_degrees(
    source_degree,
    synapse_degree,
    *,
    group_by=None,
    seed=None,
) -> PairingSpec
```

展开 source 与 Synapse stubs，然后随机一一配对。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source_degree` | scalar, array, or degree callable | required | 每个 source 的 stub 数。 |
| `synapse_degree` | scalar, array, or degree callable | required | 每个 Synapse 的 stub 数。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

两侧 degree 总和必须严格相等。该策略 v1 不接受 score 或额外约束。

### Degree helpers

| Signature | Distribution / meaning |
| --- | --- |
| `braincell.network.connection.degree.poisson(lam)` | Poisson degree callable。 |
| `braincell.network.connection.degree.binomial(n, p)` | Binomial degree callable。 |
| `braincell.network.connection.degree.negative_binomial(n, p)` | Negative-binomial degree callable，要求 \(p\in(0,1]\)。 |
| `braincell.network.connection.degree.empirical(values, probabilities)` | 从显式离散 PMF 采样 degree。 |

这些 callable 使用 `brainstate.random.RandomState`，返回非负整数 counts。

### Score and grouping contracts

score callable 接收 `ctx`，必须返回有限、非负、无量纲权重。权重 \(w_i\) 的归一化概率为：

$$
p_i=\frac{w_i}{\sum_j w_j}.
$$

条件采样时固定端形状为 `(B, 1)`，候选端为 `(1, K)`，score 应可广播到 `(B, K)`；边际 score
使用 `B=1`。每个被归一化的候选行至少需要一个正值。

| Context | Available information |
| --- | --- |
| Synapse | logical/location/CV/branch IDs、population index、radius、树路径距离、3-D position、`get(parameter)`。 |
| Source | source ID、type、name、owner、可用时的 population index、`get(field)`。 |

默认候选 endpoints 属于一个全局池。`group_by="target_cell"` 按 Synapse `population_index` 升序分组，
每组独立执行规则后拼接。固定行数规则的 `number` 可以是 scalar，或长度等于实际分组数的一维整数数组。

候选 source/Synapse views 不能含重复 ID，但生成结果允许重复。`replace=False` 在边际采样中分别作用于
对应池；条件采样只保证同一个固定 endpoint 的 partners 不重复，不保证全局 pair 唯一。生成零行会报错，
且不会修改 Connection store。

```text
net.connect(
    "distance_conditioned",
    source=pre.event_outputs["spike"],
    synapse=post.synapses["ampa"],
    pairing=braincell.network.connection.source_first(
        500,
        synapse_score=lambda ctx: distance_kernel(
            ctx.source.get("position"),
            ctx.synapse.position,
        ),
        seed=8,
    ),
)

pairing = braincell.network.connection.by_synapse(
    braincell.network.connection.degree.poisson(5.0),
    source_score=lambda ctx: source_preference(ctx.source),
    replace=False,
    seed=9,
)
```

直接 `braincell.connect` 的隐式 pairing seed root 为 0。`Network.connect` 从 Network seed 与 source
Population、target Population 和 connection name 派生，与 Population 添加顺序无关。显式 pairing
seed 完全覆盖 Network seed。
