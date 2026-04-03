# 开发进度 TODO

以下状态按当前代码与测试实况整理。

## Morpho

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| Branch | 基础属性、`from_lengths` / `from_points` 两类构建、面积/体积计算 | 已完成 |
| Morpho | 根分支构建、`attach`、属性语法糖挂接、拓扑查询、`topo` 树输出 | 已完成 |
| Morpho | `from_swc` / `from_asc` 入口 | 已完成 |
| Morpho | 高级树编辑（删除 subtree、拼接/合并 tree） | 未完成 |
| Metric | 总长度、总面积、总体积、branch order、path distance、euclidean distance、height/width/depth 等度量 | 已完成 |

## IO

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| 文件格式 | SWC 导入、rulebook 检查/修正、report | 已完成 |
| 文件格式 | ASC 导入：简单 Neurolucida 树、metadata、`Morpho.from_asc(..., return_report=True)` | 大部分完成 |
| 文件格式 | NeuroML2 导入 | 未完成 |
| 对比检测 | 通过 NEURON 导入 SWC，进行 metric 对比 | 已完成 |
| 对比检测 | 通过 Neuromorpho API 下载并进行对比 | 未完成 |

## Vis

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| 3D | 具备 points 的 Branch / Morpho 3D 渲染，scene 构建与 PyVista backend | 已完成 |
| 2D | projected 模式：基于真实 points 的 2D 投影 | 已完成 |
| 2D | tree 自动布局模式 | 已完成 |
| 2D | frustum 自动布局模式 | 已完成 |
| 2D | stem / balloon / radial360 布局族与 matplotlib 对比图 | 已完成 |
| Overlay | `region` / `locset` / `values` 参数已接入统一绘图入口，但真实高亮/着色语义仍较弱 | 部分完成 |

## Filter

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| Region | `BranchSlice` 区间选择、广播输入、集合运算（并/交/差/补） | 已完成 |
| Region | 离散变量筛选：按 `type` / `name` / `branch_order` / `parent_id` / `n_children` | 已完成 |
| Region | 连续变量筛选：`branch_range(...)`，支持数值和 quantity bounds | 已完成 |
| Region | 按半径范围筛选 | 未完成 |
| Region | 按树上距离筛选 | 未完成 |
| Region | 按欧氏距离筛选 | 未完成 |
| Region | Subtree region | 未完成 |
| Locset | 根位置、branch points、terminals | 已完成 |
| Locset | 由 Region 生成均匀采样 / 随机采样 | 已完成 |
| Locset | 由 Region 生成 anchors / 固定步长采样 | 未完成 |

## mech

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| Cable | `CableProperties` 数据容器 | 已完成 |
| Density | `DensityMechanism` 数据容器 | 已完成 |
| Point | `CurrentClamp` / `ProbeMechanism` / `SynapseMechanism` / `GapJunctionMechanism` | 已完成 |
| 导出 | `ion` / `channel` / `synapse` 顶层导出 | 已完成 |
| 运行时集成 | 与真实仿真执行链路的完整编译/运行集成 | 部分完成 |

## cell

| 模块子项 | 内容 | 完成状态 |
| --- | --- | --- |
| Cell | `Cell(morpho, cv_policy)` 前端入口、形态快照、`paint` / `place`、懒重建 | 已完成 |
| CV 离散 | `CVPolicy(cv_per_branch / max_cv_len)`、CV 几何、轴向电阻拆分 | 已完成 |
| 机制映射 | cable paint、density paint、point place 映射到 CV | 已完成 |
| PointTree | 计算点、边、attachment 处理 | 已完成 |
| Scheduling | `PointScheduling` / DHS 分组 | 已完成 |
| 执行层 | `run()`、`HHTypedNeuron` 编译、quad/JAX 真实仿真执行 | 未完成 |
