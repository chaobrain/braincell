# 当前进展

这份文档记录当前仓库已经落地到什么程度，方便下次继续开发时快速对齐代码现实。

## 已完成

### 包结构

- 包名稳定为 `braincell`
- 当前包目录为 `braincell/`
- 公开子模块已经拆分为 `morpho / io / filter / vis / mech / cell / quad`
- `develop_doc/` 主要承载设计文档、notebook、fixture 和开发验证工具

### `morpho`

- `Branch` 已支持当前两类主构造入口：
  - `Branch.from_lengths(...)`
  - `Branch.from_points(...)`
- `Branch` 已支持几何量计算：
  - `length`
  - `area`
  - `volume`
  - 分段半径、分段点等查询
- `Morpho` 是当前唯一公开树对象，支持原位编辑
- `MorphoBranch` 是树内 branch 视图，支持：
  - `tree.soma.dend = ...`
  - `tree.soma.attach(...)`
  - `tree.attach(...)`
  - `parent / children / parent_x / child_x / n_children`
- `Morpho` 已支持：
  - `from_root(...)`
  - `from_swc(...)`
  - `from_asc(...)`
  - `branch(...)`
  - `branch_by_order(...)`
  - `path_to_root(...)`
  - `summary()`
  - 直接暴露的 whole-morphology metrics，如 `total_length / total_area / total_volume / n_branches / max_path_distance / max_euclidean_distance`
- `Morpho.metric` 已简化为字符串属性，用于格式化打印 summary；不再保留独立 `metrics.py` 包装层
- `Morpho.vis2d(...)` 和 `Morpho.vis3d(...)` 都已可调用

### `io`

- `braincell/io/` 现在承载公开 IO 适配层：
  - `swc/` 已落地
  - `asc/` 已有 reader 与测试
  - `neuroml2/` 已有 reader 骨架
  - `neuromorpho.py` 已提供下载/缓存客户端
- `Morpho.from_swc(...)` 与 `Morpho.from_asc(...)` 是当前推荐入口
- `SwcReader.read(..., return_report=True)` 与 `Morpho.from_swc(..., return_report=True)` 可返回 `(Morpho, SwcReport)`
- SWC 规则已集中到单一 rulebook，支持 `check + correct`
- 真实文件 smoke fixture 当前放在 `develop_doc/morpho_files/`
- 通过 NEURON 对比 SWC metric 的逻辑已迁到：
  - `develop_doc/neuron_diff.py`
  - `develop_doc/neuron_diff_test.py`
  - 它们是开发验证资产，不再是 `braincell.io` 的公开接口

### `cell`

- 当前 `cell/` 已经不是早期骨架，而是现有前端层：
  - `cell.py`：`Cell`
  - `cv.py`：`CV`
  - `cv_policy.py`：`CVPolicy / CVPerBranch / MaxCVLen / DLambda`
  - `cv_geo.py`：CV 几何离散
  - `cv_mech.py`：`PaintRule / PlaceRule`
  - `point_tree.py`：`PointTree`
  - `point_scheduling.py`：`PointScheduling`
- `Cell(morpho, cv_policy=...)`、`paint(...)`、`place(...)`、懒重建已经落地
- `PointTree` 和 scheduling 相关基础结构已存在并有测试覆盖

### `vis`

- 2D 与 3D 渲染入口都已落地
- `braincell/vis/` 当前包含：
  - `plot2d.py` / `plot3d.py`
  - `scene2d.py` / `scene3d.py`
  - `layout2d.py`
  - `backend_matplotlib.py`
  - `backend_pyvista.py`
- 2D 布局族、比较工具、真实文件 smoke 测试已经存在
- 3D PyVista backend 已可用于真实 morphology 文件验证

### `filter`

- Region / Locset 的基础表达式与部分筛选能力已落地
- 现有测试包括：
  - `filter_region_test.py`
  - `filter_locset_test.py`
  - `filter_branch_filters_test.py`
  - `filter_vis_test.py`

### `mech`

- `mech` 下已有 `ion / channel / synapse / cable / density / point` 基础容器与测试
- NMODL 相关开发资产和验证脚本已经进入仓库，但仍属于持续建设中

## 当前验证

当前仓库至少已有以下自动化测试入口：

- `braincell/morpho/branch_test.py`
- `braincell/morpho/morpho_test.py`
- `braincell/io/io_swc_test.py`
- `braincell/io/io_asc_test.py`
- `braincell/io/io_real_files_test.py`
- `braincell/io/neuromorpho_test.py`
- `braincell/cell/cell_test.py`
- `braincell/cell/cv_policy_test.py`
- `braincell/vis/vis_plot_test.py`
- `braincell/vis/vis_real_files_test.py`
- `braincell/vis/backend_pyvista_test.py`

开发验证测试还包括：

- `develop_doc/neuron_diff_test.py`

## 仍未完成

### `io`

- `NeuroMlReader` 仍未形成完整导入能力
- ASC / NeuroML2 的设计与异常输入处理仍可继续补强
- `develop_doc/neuromorpho_diff.ipynb` 相关批量对比工作流仍在演进

### `filter`

- 按半径范围筛选
- 按树上距离筛选
- 按欧氏距离筛选
- Subtree region
- 更完整的 anchors / 固定步长 locset 语义

### `cell`

- `run()` 与完整执行层尚未打通
- `HHTypedNeuron` 编译链路仍未形成对外稳定入口
- 与 `quad` / `_base.py` / 真实 runtime 的完整仿真集成仍未完成

### `vis`

- overlay 的真实高亮/着色语义仍有继续增强空间
- 除 PyVista / matplotlib 之外的 backend 仍可继续扩展

## 重要设计决策

- `Branch` 保持纯几何，不存拓扑
- `Morpho` 是唯一公开树对象，允许原位编辑
- whole-morphology metrics 直接挂在 `Morpho` 上
- `Morpho.metric` 仅保留为格式化 summary 字符串
- `braincell.io` 只保留正式 IO/下载接口；NEURON 对比逻辑放在 `develop_doc`
- `Cell + CV + PointTree + PointScheduling` 是当前 cell 前端层主干

## 下次继续时最值得做的事

优先级建议：

1. 打通 `cell` 的执行层与真实仿真链路
2. 补强 `filter` 的距离类 region / locset 表达式
3. 继续完善 ASC / NeuroML2 导入
4. 继续收紧 `develop_doc/` 文档与 notebook 的事实同步
5. 扩展 vis overlay 与额外 backend

## 快速恢复上下文

建议下次先读：

1. `README.md`
2. `develop_doc/io.md`
3. `develop_doc/cell.md`
4. `develop_doc/todo.md`
5. `braincell/morpho/branch.py`
6. `braincell/morpho/morpho.py`
7. `braincell/io/swc/reader.py`
8. `braincell/cell/cell.py`
9. `braincell/cell/point_tree.py`
10. `braincell/vis/plot2d.py`
