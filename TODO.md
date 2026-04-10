# 开发进度 TODO

以下状态按当前代码与测试实况整理。

## `braincell.morph` module

- [x] Branch：基础属性、`from_lengths` / `from_points` 两类构建、面积/体积计算
- [x] Morpho：根分支构建、`attach`、属性语法糖挂接、拓扑查询、`topo` 树输出
- [x] Morpho：`from_swc` / `from_asc` 入口
- [x] Morpho：`save_checkpoint` / `load_checkpoint`：`.bcm` 自包含格式 + pickle / `copy.deepcopy` 支持
- [ ] Morpho：高级树编辑（删除 subtree、拼接/合并 tree）
- [x] Metric：总长度、总面积、总体积、branch order、path distance、euclidean distance 等度量


## `braincell.io` module

- [x] 文件格式：SWC 导入、rulebook 检查/修正、report
- [~] 文件格式：ASC 导入：简单 Neurolucida 树、metadata、`Morpho.from_asc(..., return_report=True)`（大部分完成)
- [ ] 文件格式：NeuroML2 导入
- [x] 对比检测：通过 `develop_doc/neuron_diff.py` 走 NEURON 导入 SWC，进行开发期 metric 对比
- [ ] 对比检测：通过 Neuromorpho API 下载并进行对比
- [x] checkpoints（`braincell/io/checkpoint.py`：`save_branch` / `load_branch` / `save_morpho` / `load_morpho`，`.bcm` 单文件格式 + 教程 `develop_doc/morpho-checkpoint.ipynb`）
- [ ] NMODL parsing compiler
- [ ] xxx
- [ ] xx


## `braincell.vis` module

- [x] 3D：具备 points 的 Branch / Morpho 3D 渲染，scene 构建与 PyVista backend
- [x] 2D：projected 模式：基于真实 points 的 2D 投影
- [x] 2D：tree 自动布局模式
- [x] 2D：frustum 自动布局模式
- [x] 2D：stem / balloon / radial360 布局族与 matplotlib 对比图
- [~] Overlay：`region` / `locset` / `values` 参数已接入统一绘图入口，但真实高亮/着色语义仍较弱（部分完成）


## `braincell.filter` module

- [x] Region：`BranchSlice` 区间选择、广播输入、集合运算（并/交/差/补）
- [x] Region：离散变量筛选：按 `type` / `name` / `branch_order` / `parent_id` / `n_children`
- [x] Region：连续变量筛选：`branch_range(...)`，支持数值和 quantity bounds
- [ ] Region：按半径范围筛选
- [ ] Region：按树上距离筛选
- [ ] Region：按欧氏距离筛选
- [ ] Region：Subtree region
- [x] Locset：根位置、branch points、terminals
- [x] Locset：由 Region 生成均匀采样 / 随机采样
- [ ] Locset：由 Region 生成 anchors / 固定步长采样


## `braincell.mech` module

- [x] Cable：`CableProperties` 数据容器
- [x] Density：`DensityMechanism` 数据容器
- [x] Point：`CurrentClamp` / `ProbeMechanism` / `SynapseMechanism` / `GapJunctionMechanism`
- [x] 导出：`ion` / `channel` / `synapse` 顶层导出
- [~] 运行时集成：与真实仿真执行链路的完整编译/运行集成（部分完成）


## `braincell.cell` module

- [x] Cell：`Cell(morpho, cv_policy)` 前端入口、形态快照、`paint` / `place`、懒重建
- [x] CV 离散：`CVPolicy` 基类 + `CVPerBranch / MaxCVLen / DLambda`、CV 几何、轴向电阻拆分
- [x] 机制映射：cable paint、density paint、point place 映射到 CV
- [x] PointTree：计算点、边、attachment 处理
- [x] Scheduling：`PointScheduling` / DHS 分组
- [ ] 执行层：`run()`、`HHTypedNeuron` 编译、quad/JAX 真实仿真执行


## `braincell.quad` module




