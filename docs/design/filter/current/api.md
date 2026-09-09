# Filter API

`braincell.filter` 的 RegionExpr 选择形态区间，LocsetExpr 选择离散位置。表达式在获得 Morphology 后解析；
Cell.on/paint 使用区域，Cell.loc/place 使用位置。连续概率采样见 [Sampling](sampling.md)，
空间参数回调见 [Callable 参数](spatial-callable-parameters.md)。

## 最小用法

```python
import braincell as bc
import brainunit as u
from braincell import filter as f

branch = bc.Branch.from_lengths(lengths=[60.0] * u.um, radii=[2.0, 2.0] * u.um, type="dendrite")
morpho = bc.Morphology.from_root(branch, name="dend")
region = f.branch_in("type", ["dendrite"]) & f.BranchSlice(0, 0.25, 0.75)
cache = f.SelectionCache()
mask = region.evaluate(morpho, cache)
assert mask.intervals == ((0, 0.25, 0.75),)
points = (f.at("dend", 0.25) + f.at("dend", 0.75)).evaluate(morpho, cache)
assert points.branch_id.shape == (2,)
assert len(points) == 2
sampled = f.sample(region, number=4, seed=7).evaluate(morpho)
assert sampled.branch_x.shape == (4,)
assert ((sampled.branch_x >= 0.25) & (sampled.branch_x <= 0.75)).all()
```

## 区域

```text
AllRegion()
EmptyRegion()
BranchSlice(branch_index, prox, dist)
branch_in(property, values) -> BranchInFilter
branch_range(property, bounds, *, closed="neither") -> BranchRangeFilter
RegionExpr.evaluate(morpho, cache=None) -> RegionMask
RegionMask(intervals)
```

BranchSlice 的 branch_index/prox/dist 支持广播，坐标是分支弧长归一化的 `[0,1]`；
RegionMask.intervals 是 `(branch_index, prox, dist)` 元组集合。
branch_in 选择离散属性如 type/name/branch_id/parent_id/branch_order/n_children/n_tapers。
values 是一个或多个允许值。branch_range 用 `(lower, upper)` 筛选分支标量属性，
支持 length、mean_radius、area、volume 以及数值拓扑属性；物理量 bounds 必须使用匹配单位。
closed 为 neither/left/right/both，分别决定开闭边界；任一 bound 为 None 表示该侧不设限。

区域代数 `a | b`、`a & b`、`a - b`、`a.complement()` 返回新表达式，不修改原对象；
解析时分别取区间并、交、差以及相对全部形态的补集。操作类型为
`RegionSetOp(op, operands)`，通常用运算符构造。

## 位点和批次

```text
at(branch, x) -> AtLocation
AtLocation(branch, x)
RootLocation(x)
ForkPoints()
Terminals()
UniformSamples(region, count)
RandomSamples(region, count, seed)
LocsetExpr.evaluate(morpho, cache=None) -> LocsetMask
LocsetMask(points=(), display_names=None)
LocsetMask.from_columns(branch_id, branch_x, *, display_names=None)
LocsetBatch.from_columns(branch_id, branch_x, *, display_names=None)
```

branch 为分支整数索引或名称，x 为 `[0,1]` 弧长坐标。RootLocation 使用根分支；
ForkPoints 返回分叉位置，BranchPoints 是它的别名；Terminals 返回终端位置。
UniformSamples 将所选区间按物理长度连接，在总长的 count 个等分中点取样；
RandomSamples 先按区间长度选择区间，再在区间内均匀抽样。count 为正整数，空或零长度区域抛出 ValueError。
旧 RandomSamples 使用独立 NumPy RNG；需要显式测度、density 和保留抽样顺序时使用 [sample](sampling.md)。

LocsetMask.points 是 `(branch_id, branch_x)` 元组，列表示为只读 `(L,)` 数组；
LocsetBatch 列为 `(P,L)`，P 是 population 行，L 是每行位置数。两列形状必须匹配。
普通索引返回位置子集；batch 的一行索引返回 LocsetMask。display_names 为可选对齐标签。

| 运算 | 顺序与重复值 |
| --- | --- |
| `a + b` | 连接两个位置序列，保留顺序和重复位置 |
| `a | b`、`a & b`、`a - b` | 集合并、交、差 |
| `a.unique()` | 按首次出现顺序去重 |

对应表达式类型为 `LocsetConcatOp(operands)`、`LocsetSetOp(op, operands)`、
`LocsetUniqueOp(operand)`。Cell 将几何位置映射到 CV/边界 point 的规则见
[Cell 架构](../../cell/current/architecture.md)，Locset 本身不分配电气节点。

## 解析和缓存

evaluate 需要 Morphology，错误对象抛出 TypeError；无效分支索引/名称、坐标和广播形状在解析时报错。
`morpho.select(expr, cache=cache)` 是同一解析能力的便利入口。
同一个 SelectionCache 可复用子表达式结果；换 morphology 或 attach 导致 revision 改变时清空缓存。
不可哈希的表达式仍可解析，只跳过缓存。实现见 [cache.py](../../../../braincell/filter/cache.py)。

以下已导出类型目前只能构造，evaluate 会抛出 NotImplementedError：

| 签名 | 尚缺能力 |
| --- | --- |
| `RadiusRangeRegion(minimum, maximum)` | 按局部半径截取区域 |
| `TreeDistanceRegion(minimum, maximum)` | 按树路径距离截取 |
| `EuclideanDistanceRegion(minimum, maximum)` | 按三维距离截取 |
| `SubtreeRegion(root_branch_index)` | 按子树选择 |
| `RegionAnchors(region, x)` | 区域相对锚点 |
| `StepSamples(region, step)` | 固定物理步长采样 |

这些类型的实现工作见 [Filter TODO](../TODO.md)。region、locset 的行为用例分别在
[region_test.py](../../../../braincell/filter/region_test.py)、[locset_test.py](../../../../braincell/filter/locset_test.py)。
