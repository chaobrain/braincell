# Morph API

`braincell.Branch` 保存不可变分支几何，`braincell.Morphology` 管理可追加的树。
两者从 braincell 顶层导入；`braincell.morph` 导出 MorphoBranch、MorphoEdge、MorphoMetric 等辅助类型。
文件读写见 [IO](../../io/current/api.md)，依赖规则见 [分层约束](layering-invariants.md)。

## 构造一棵树

```python
import copy
import braincell as bc
import brainunit as u

soma = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
dend = bc.Branch.from_lengths(lengths=[40.0, 60.0] * u.um, radii=[2.0, 1.5, 1.0] * u.um, type="dendrite")
morpho = bc.Morphology.from_root(soma, name="soma")
child = morpho.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=0.5)
assert child.parent is morpho.root
assert morpho.n_branches == 2
assert u.math.allclose(morpho.total_length, 120.0 * u.um)
assert len(morpho.edges) == 1
copied = copy.deepcopy(morpho)
assert copied is not morpho
print(morpho.topo())
```

## Branch 几何

完整签名；`<class default>` 表示省略 type 时采用调用类的默认分支类型，Branch 为 custom：

```text
Branch(lengths, radii_proximal, radii_distal,
       points_proximal=None, points_distal=None, type="custom")
Branch.from_lengths(*, lengths, radii=None, radii_proximal=None,
                    radii_distal=None, type=<class default>) -> Branch
Branch.from_points(*, points, radii=None, radii_proximal=None,
                   radii_distal=None, type=<class default>) -> Branch
```

| 参数 | 单位和形状 |
| --- | --- |
| lengths | 长度量 `(n,)`，n 为 frustum 数 |
| points | 长度量 `(n+1, 3)`，每行 xyz，段长由相邻点计算 |
| radii | 长度量 `(n+1,)`，相邻值给出每段两端半径 |
| radii_proximal、radii_distal | 长度量 `(n,)`；与 radii 二选一，允许段间半径跳变 |
| points_proximal、points_distal | 直接构造时的逐段端点 `(n,3)`，需同时提供 |
| type | 形态语义字符串，如 soma、axon、dendrite、basal_dendrite、apical_dendrite、custom |

缺少半径、混用两种半径形式、非有限几何、非法形状或单位会在构造时失败，
具体检查见 [branch.py](../../../../braincell/morph/branch.py)。
`from_lengths` 提供电缆几何，但没有三维坐标；路径长度和面积仍可用，xyz 范围、欧氏距离及三维绘图需要 points。
段长允许零以保留半径跳变的面积语义；这类几何如何导入见 [SWC 约束](../../io/current/swc-reader-invariants.md)。

## Morphology 与连接

```text
Morphology(*, root_name, root_branch)
Morphology.from_root(branch, *, name="soma") -> Morphology
Morphology.attach(*, parent, child_branch, child_name=None,
                  parent_x=1.0, child_x=0.0) -> MorphoBranch
MorphoBranch.attach(branch, name=None, *, parent_x=1.0, child_x=0.0) -> MorphoBranch
```

root_branch/child_branch 是 Branch；parent 是本树的分支名或 MorphoBranch。
`parent_x` 为父分支归一化弧长坐标 `[0,1]`，`child_x` 为子分支连接端点 0 或 1。
child_name=None 自动按类型分配名称，显式名称需唯一且不能占用保留属性名。
attach 就地增加节点和边、递增 revision、失效派生缓存，返回新节点的 view。
错误父节点、重复名称和非法连接坐标会在写入前拒绝。

属性式 `morpho.soma.dend = branch` 等价于在 soma 末端追加命名分支。
已有名称不能用该语法替换几何；删除、splice、整体变换的设计由 [Morph TODO](../TODO.md) 跟踪。
要编辑独立副本，使用 `copy.deepcopy(morpho)`；当前没有公共 `clone()` 方法。

## 查询、视图和指标

```text
Morphology.branch(*, name=None, index=None, order=None) -> MorphoBranch
Morphology.branch_by_order(*, order="default") -> tuple[MorphoBranch, ...]
Morphology.path_to_root(branch_index) -> tuple[int, ...]
Morphology.topo() -> str
Morphology.select(expr, *, cache=None) -> RegionMask | LocsetMask
MorphoBranch.index_by(*, order="default") -> int
MorphoMetric.as_dict() -> dict
```

branch 按 name 或 index 选一个分支，两者互斥。order 支持 default/type/depth：
分别按节点创建顺序、分支类型与名称、根路径深度排序；index 属于所选顺序。
持久定位宜使用名称，按 type/depth 排列时追加结构可能改变索引。依据见
[树查询实现](../../../../braincell/morph/morphology.py)。
select 消费 RegionExpr/LocsetExpr，参数和缓存行为见 [Filter](../../filter/current/api.md)。

| 属性 | 返回值和归属 |
| --- | --- |
| root、branches、edges | 根 view、分支 view 元组、只读 MorphoEdge 元组，均由当前树拥有 |
| MorphoBranch.parent、children、n_children | 父 view 或 None、子 view 元组、子节点数 |
| MorphoBranch.index / branch_id、branch_order、n_tapers | 空间选择所用整数索引、拓扑层次和几何段数 |
| revision | 当前结构修订号，用于缓存失效 |
| metric | 当前统计的冻结 MorphoMetric 快照，修改树后应重新读取 |
| total_length、mean_radius | 长度量 |
| total_area、total_volume | 面积和体积量 |
| n_branches、n_stems、n_bifurcations、max_branch_order | 整数拓扑指标 |
| max_path_distance、max_path_distance_excluding_soma | 树路径距离 |
| max_euclidean_distance 及 excluding_soma 变体、x/y/z_range | 依赖完整三维坐标的距离量；缺失坐标时抛出 ValueError |

几何存储、连接方向与导入分层见 [架构约束](layering-invariants.md)。
读写和 checkpoint 方法统一由 [IO API](../../io/current/api.md) 维护，
vis2d/vis3d 完整调用见 [Vis API](../../vis/current/api.md)。
