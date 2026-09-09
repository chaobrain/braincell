# Vis API

`braincell.vis` 提供形态、Cell 离散拓扑、数值与时间序列的可视化。函数从
`from braincell import vis` 导入；`Morphology` 和 `Branch` 另有 `vis2d()`、`vis3d()`
快捷方法。Cell 拓扑直接传 `cell`，形态绘图传 `cell.morpho`。

| 使用场景 | 接口说明 |
| --- | --- |
| 查看 Cell 的分支、CV、求解节点及状态 | [Cell 拓扑](#cell-拓扑) |
| 画形态、着色与选择位点 | [形态绘图](#形态绘图)、[对象快捷方法](#对象快捷方法)、[数值与选择](#数值与选择) |
| 比较模型和结果、画轨迹或动画 | [对比](#对比)、[时间轨迹](#时间轨迹)、[动画](#动画) |
| 分析树结构与几何 | [结构分析](#结构分析) |
| 设置布局和样式、响应点击、保存结果 | [布局与缓存](#布局与缓存)、[样式配置](#样式配置)、[交互回调](#交互回调)、[后端与导出](#后端与导出) |

以下签名保留参数顺序、关键字限定及默认值，类型与组合规则在对应表格中说明。
覆盖范围以[公共导出](../../../../braincell/vis/__init__.py)为准；返回对象的字段也在正文列出。

## 从 Cell 开始

下面构造一个带 3D 坐标的 soma 和两段 dendrite，共两个 branch、三个几何 segment。
`CVPerBranch(2)` 为每个 branch 生成两个 CV。后续示例复用这里的 `cell`、`morpho` 和导入。

```python
import numpy as np
import brainunit as u
import braincell as bc
from braincell import vis

soma = bc.Branch.from_points(
    points=[[0., 0., 0.], [10., 0., 0.]] * u.um,
    radii=[5., 5.] * u.um, type="soma",
)
dend = bc.Branch.from_points(
    points=[[10., 0., 0.], [30., 10., 0.], [50., 20., 5.]] * u.um,
    radii=[2., 1.5, 1.] * u.um, type="basal_dendrite",
)
morpho = bc.Morphology.from_root(soma, name="soma")
morpho.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=1.)
cell = bc.Cell(morpho, cv_policy=bc.CVPerBranch(2))

shape_ax = vis.plot2d(cell.morpho)
branch_ax = vis.plot_cell_topology(cell, level="branch", layout="kamada_kawai")
cv_ax = vis.plot_cell_topology(cell, level="cv", layout="kamada_kawai")

cell.init_state()
voltage_ax = vis.plot_cell_topology(
    cell, level="node", value="V", layout="kamada_kawai",
)
vis.save_figure(voltage_ax, "voltage.png")
assert voltage_ax.figure is not None
```

绘图读取调用时的数据，不推进 Cell。`init_state()` 用于初始化模型，不是刷新图像的步骤。
再次绘图会创建新图；传入已有 `ax` 时在该轴上继续添加图元。修改 Cell 后，旧图不会自动刷新。

## Cell 拓扑

源码：[cell_topology.py](../../../../braincell/vis/cell_topology.py)；数值与选择映射由
[field_resolution.py](../../../../braincell/_multi_compartment/field_resolution.py)处理。

```text
plot_cell_topology(cell, *, level="node", preset="dendrotweaks",
    layout=None, layout_scale=1.0, region=None, locset=None,
    coverage_mode="fraction", highlight_color="#ef4444", value=None,
    cmap=None, vmin=None, vmax=None, norm=None, value_label=None,
    show_colorbar=True, node_color=None, edge_color=None, root_color=None,
    ax=None) -> matplotlib.axes.Axes
```

| `level` | 一个图节点表示什么 | 调用条件与可用数据 |
| --- | --- | --- |
| `"branch"` | 一个 morphology branch | 构造 Cell 后可调用；支持 `region` 覆盖，拒绝 `locset`、`value` 及显式色条参数 |
| `"cv"` | 一个 CV | 结构与选择可在初始化前查看；提供 value 时先初始化，CV 树必须恰有一个根 |
| `"node"`，默认 | 一个 NodeTree 空间位点，包括 CV 中点和边界点 | 要求 `cell.init_state()`；支持选择与数值映射 |

### 选择与覆盖

| 参数 | 类型与语义 |
| --- | --- |
| `cell` | `bc.Cell`，必填 |
| `region=None` | `RegionExpr` 或 `RegionMask`；表达式由 Cell 解析。按膜面积计算与 CV 或 branch 的重叠比例 |
| `locset=None` | `LocsetExpr` 或 `LocsetMask`；位置归属到 CV，在 node 图中高亮该 CV 中点 |
| `coverage_mode="fraction"` | `fraction` 按比例混色；`any` 有重叠即全亮；`all` 完全覆盖才全亮。locset 命中强度为 1 |
| `highlight_color="#ef4444"` | Matplotlib 颜色，高亮目标色 |
| `value=None` | 数值或字段选择器，见下表；与任意非 `None` 的 `region`、`locset` 互斥 |

region 和 locset 可以组合，重叠位置取较强高亮。node 图中的选择只标记 CV 中点，
包括端点在内的 locset 也按所属 CV 映射，图中被高亮的点不一定是原始几何位置。

```python
region = bc.filter.BranchSlice(branch_index=1, prox=0.0, dist=0.5)
locset = bc.filter.RootLocation(0.5)
vis.plot_cell_topology(cell, level="cv", region=region, layout="kamada_kawai")
vis.plot_cell_topology(cell, locset=locset, layout="kamada_kawai")
```

### 数值来源与空间

| `value` 形式 | 解释与限制 |
| --- | --- |
| 标量 | 在目标空间广播，可带 `brainunit` 单位 |
| `(n_cv,)` 数组 | CV 图直接使用；node 图只写入各 CV 中点，其他点为 `NaN` |
| `(n_point,)` 数组 | node 图保留每点值；CV 图读取各 CV 中点值 |
| `"V"`、`"voltage"` | 读取 `cell.V.value` 的 CV 电压；node 图中的边界点为 `NaN` |
| `("ion", ion_name, field)` | 读取指定离子字段，按命名膜字段规则映射，node 图仅显示中点 |
| `("channel", class_name, field)` | 按通道类名查找唯一 runtime layout；同类有多个 layout 时用 ID 选择 |
| `("layout_id", layout_id, field)` | 读取具体机制 layout 的状态或参数，按其实际挂载位置映射；未覆盖处为 `NaN` |

运行时数组的空间轴在最后。前置 population 轴全为 1 时自动去掉；含多个成员时抛出
`ValueError`，调用者应先选一个成员，再通过 `value=` 传入一维数组。空间数组长度不能
按颜色自动推断成形态 segment；这里的 `n_point` 指求解节点，区别于形态中心线点数。

```python
cv_voltage = cell.V.value[0]
vis.plot_cell_topology(cell, level="cv", value=cv_voltage, layout="kamada_kawai")
```

### 拓扑图的公共样式与错误

以下参数也用于 `plot_point_topology`。

| 参数 | 类型、默认值与效果 |
| --- | --- |
| `preset="dendrotweaks"` | 样式预设；可选名称见下方离散点拓扑说明 |
| `layout=None` | 默认取预设布局；可指定 `twopi`、`dot`、`neato`、`kamada_kawai` |
| `layout_scale=1.0` | 正的有限数，控制图布局间距，不代表真实几何长度 |
| `node_color/edge_color/root_color=None` | Matplotlib 颜色；`None` 沿用预设 |
| `cmap=None` | 数值色图；省略时使用预设或 `viridis` |
| `vmin/vmax=None` | 数值范围，省略时根据有效数值计算；使用数据当前单位对应的裸数值 |
| `norm=None` | Matplotlib `Normalize` 对象，优先于范围参数 |
| `value_label=None` | 色条标题；Cell 命名字段自动生成名称，带单位数组自动补单位 |
| `show_colorbar=True` | 是否创建数值色条 |
| `ax=None` | 目标 Matplotlib `Axes`；`None` 新建，返回实际绘制的同一个轴 |

错误类型：输入不是 Cell 为 `TypeError`；node 层未初始化为 `RuntimeError`；非法 level、
互斥参数、不支持的数组形状、多成员 population 或不唯一的通道选择为 `ValueError`。
缺少机制字段可能产生 `AttributeError`，不存在的 layout ID 为 `KeyError`。
branch 层还拒绝非默认的 `show_colorbar`，因为这一层只展示拓扑和覆盖。
Graphviz 不可用时发出警告并退回 `kamada_kawai`。

## 形态绘图

源码：[plot2d.py](../../../../braincell/vis/plot2d.py)、[plot3d.py](../../../../braincell/vis/plot3d.py)。

```text
plot2d(morpho, *, region=None, locset=None, values=None, cmap=None,
    vmin=None, vmax=None, norm=None, value_label=None, show_colorbar=True,
    layout=None, shape=None, branch_type_colors=None,
    branch_type_edge_colors_2d=None, frustum_edge_linewidth_2d=None,
    backend=None, chooser=None, ax=None, notebook=None, jupyter_backend=None,
    return_plotter=False, projection_plane="xy", min_branch_angle_deg=25.0,
    root_layout="type_split", layout_config=None, hooks=None) -> object

plot3d(morpho, *, region=None, locset=None, values=None, cmap=None,
    vmin=None, vmax=None, norm=None, value_label=None, show_colorbar=True,
    mode=None, backend=None, chooser=None, notebook=None, jupyter_backend=None,
    return_plotter=False, hooks=None) -> object
```

`morpho` 必须是 `bc.Morphology`；传 Cell 或 Branch 会抛出 `TypeError`。
2D 返回 `Axes`，3D 返回对象由后端与 Notebook 设置决定，见[后端与导出](#后端与导出)。

| 参数 | 类型与语义 |
| --- | --- |
| `region/locset=None` | 已求值的 `RegionMask`、`LocsetMask`，由 `expr.evaluate(morpho)` 获得；与 Cell 接口接收表达式的方式不同 |
| `values=None` | 一维形态标量数组或 `ValueSpec`；形状与优先级见[数值与选择](#数值与选择) |
| `cmap/vmin/vmax/norm/value_label` | 着色参数；`None` 保留传入 ValueSpec 的对应设置。裸数组默认色图为 `viridis`；norm 用于 Matplotlib，3D 后端按 vmin/vmax 线性着色 |
| `show_colorbar=True` | 控制色条；始终覆盖传入 ValueSpec 的同名字段 |
| `layout=None` | 使用全局默认，初始为 `fan`；另有 `stem`（别名 `trunk_first`）、`balloon`、`radial_360`、`projected` |
| `shape=None` | 使用全局默认，初始为 `frustum`；`line` 画中心线，`frustum` 画截锥投影 |
| `projection_plane="xy"` | `projected` 使用的投影平面，可选 `xy`、`xz`、`yz` |
| `min_branch_angle_deg=25.0` | 自动布局的最小分叉角提示，单位度；可传 `None`，由布局算法处理 |
| `root_layout="type_split"` | 根部按分支类型分组；另接受已弃用的 `legacy`，使用时发出 DeprecationWarning |
| `layout_config=None` | `LayoutConfig`；`None` 使用内置布局参数 |
| `mode=None` | 3D 使用全局默认，初始为 `geometry`；PyVista 的 `skeleton` 画中心线，`geometry` 画管状几何；Plotly 均画分支线 |
| `branch_type_colors=None` | 类型名到颜色的映射，临时覆盖当前绘图的默认填色 |
| `branch_type_edge_colors_2d=None` | 类型名到截锥边框色的映射 |
| `frustum_edge_linewidth_2d=None` | 非负边框线宽；`None` 沿用全局配置 |
| `backend=None` | 默认 2D 选 Matplotlib，3D 优先 PyVista，其次 Plotly |
| `chooser=None` | 高级扩展点，传 `braincell.vis.backend.BackendChooser`；`None` 使用默认注册表 |
| `ax=None` | 2D 的目标 Axes；3D 无此参数 |
| `notebook=None` | PyVista 自动检测 Notebook；`True/False` 显式控制 |
| `jupyter_backend=None` | PyVista 的 Notebook 展示方式，如 `client`、`html`、`trame` |
| `return_plotter=False` | PyVista Notebook 返回展示对象；`True` 请求原始 Plotter，详见后端说明 |
| `hooks=None` | `VisHooks`，在支持回调的后端接入拾取等事件 |

真实投影和 3D 绘图需要分支坐标；`layout="projected"` 要配合 `shape="line"`，否则为
`ValueError`。只有长度和半径的 morphology 可使用自动 2D 布局。
非法布局、模式、形状、后端名称或后端维度组合为 `ValueError`；显式指定但未安装的后端为
`RuntimeError`。没有 `values` 却设置 `cmap/vmin/vmax/norm/value_label` 也会抛出 `ValueError`。

```python
vis.plot2d(morpho, layout="projected", shape="line", projection_plane="xz")
figure_3d = vis.plot3d(morpho, backend="plotly", mode="skeleton")
vis.save_figure(figure_3d, "morphology.html")
```

## 对象快捷方法

源码：[Morphology](../../../../braincell/morph/morphology.py)、[Branch](../../../../braincell/morph/branch.py)。
这里的分支对象是 `Branch`；`MorphoBranch` 是形态内分支视图，可通过其 `.branch` 取得 Branch。

```text
Morphology.vis2d(self, *, layout=None, shape=None, branch_type_colors=None,
    branch_type_edge_colors_2d=None, frustum_edge_linewidth_2d=None,
    backend=None, region=None, locset=None, values=None, chooser=None, ax=None,
    notebook=None, jupyter_backend=None, return_plotter=False, show=True,
    projection_plane="xy", min_branch_angle_deg=25.0, root_layout="type_split")

Morphology.vis3d(self, *, mode=None, backend=None, region=None, locset=None,
    values=None, chooser=None, notebook=None, jupyter_backend=None,
    return_plotter=False, show=True)

Branch.vis2d(self, *, layout=None, shape=None, branch_type_colors=None,
    branch_type_edge_colors_2d=None, frustum_edge_linewidth_2d=None,
    backend=None, chooser=None, projection_plane="xy", return_plotter=False,
    show=True)

Branch.vis3d(self, *, mode=None, backend=None, chooser=None, notebook=None,
    jupyter_backend=None, return_plotter=False, show=True)
```

同名参数沿用绘图函数的含义。Branch 方法先创建临时 morphology，再调用相应方法。
这些方法均返回底层绘图结果，即使 `return_plotter=False` 也不统一返回 `None`。
`show=True` 会在绘图后调用 `matplotlib.pyplot.show()`；它不是统一的 3D 显示开关，
PyVista 的 Notebook 渲染仍由 `notebook` 等参数控制。

快捷方法接收的参数是固定子集：例如 `hooks`、`layout_config` 和独立色图参数应使用
`vis.plot2d/plot3d`，或者通过 `values=ValueSpec(...)` 传递着色设置。

```python
shortcut_ax = cell.morpho.vis2d(show=False, return_plotter=True)
single_branch_ax = morpho.root.branch.vis2d(show=False)
```

## 数值与选择

源码：[scene.py](../../../../braincell/vis/scene.py)、[_values.py](../../../../braincell/vis/_values.py)。

```text
ValueSpec(values, cmap="viridis", vmin=None, vmax=None, norm=None,
    label=None, unit_label=None, show_colorbar=True)
OverlaySpec(region=None, locset=None, values=None)
OverlaySpec.values_spec(self) -> ValueSpec | None
```

两个类型均为冻结 dataclass，但数组和映射内容仍是引用。`ValueSpec` 的 `values` 必填，
其余参数设置色图、范围、Normalize、标题、单位后缀和色条显示；色图范围使用所传数据单位的裸数值。
自动范围忽略非有限值。构造 spec 本身不完成形状验证，绘图时才解析。

形态值只接受一维数组，按下表顺序匹配长度。数组可带 `brainunit` 单位，单位用于色条标签；
这里的裸数组同样合法。多个长度相等时优先采用表中靠前的解释。

| 长度 | 排列与显示 |
| --- | --- |
| `n_branches` | 按 `morpho.branches` 顺序，每分支一个值 |
| `sum(n_segments)` | 按分支顺序拼接，每个分支由近端向远端排列；内部中心线点取相邻 segment 的均值 |
| `sum(n_segments + 1)` | 按分支拼接中心线点值；分支连接处的点仍分别计数 |

后端消费中心线值；2D segment 色值由其两个端点均值形成，因此逐 segment 输入在转换后
可能被平滑。例如一条双 segment 分支输入 `[0, 2]`，中心线值成为 `[0, 1, 2]`，
2D 两段着色对应 `[0.5, 1.5]`。CV 数组应通过 Cell 接口解释，不能靠长度相同假定映射正确。

`OverlaySpec.region/locset` 接收已求值的 mask；`values` 可为数组、ValueSpec 或 `None`。
`values_spec()` 对数组创建默认 ValueSpec，对已有 spec 返回原对象，对空值返回 `None`。
高层绘图函数分别接收 `region=`、`locset=`、`values=`，没有 `overlay=` 参数。
形态图可以同时叠加高亮、位置标记和数值着色。

```python
branch_values = np.array([-65., -60.]) * u.mV
region_mask = region.evaluate(morpho)
location_mask = locset.evaluate(morpho)
spec = vis.ValueSpec(branch_values, label="Voltage", vmin=-70., vmax=-50.)
overlay = vis.OverlaySpec(region=region_mask, locset=location_mask, values=spec)
vis.plot2d(morpho, region=overlay.region, locset=overlay.locset, values=overlay.values)
```

## 对比

源码：[compare.py](../../../../braincell/vis/compare.py)。

```text
compare_morphologies(morphologies, *, titles=None, layout=None, shape=None,
    align="soma", figsize=None, min_branch_angle_deg=25.0,
    root_layout="type_split", layout_config=None) -> (Figure, tuple[Axes, ...])

compare_values(morpho, value_arrays, *, titles=None, cmap=None, vmin=None,
    vmax=None, value_label=None, layout=None, shape=None, figsize=None,
    min_branch_angle_deg=25.0, root_layout="type_split", layout_config=None)
    -> (Figure, tuple[Axes, ...])
```

`morphologies` 是非空 Morphology 序列；`value_arrays` 是非空形态值数组序列，每项一幅图。
`titles=None` 使用形态名或 `panel i`，显式标题序列长度必须等于面板数。`figsize=None`
生成 `(4.5 * 面板数, 4.5)` 英寸的图，返回新 Figure 与按面板顺序排列的 Axes 元组。
布局和着色参数沿用 `plot2d`。各面板默认独立色标，传相同 `vmin/vmax` 才固定共同范围。
`align="soma"` 目前只影响标题后缀，`None` 去掉后缀，不执行额外几何对齐；各面板不共享轴限。
空序列、标题数量不符为 `ValueError`。

```python
comparison, panels = vis.compare_values(
    morpho, [branch_values, branch_values + 5. * u.mV],
    titles=["Before", "After"], vmin=-70., vmax=-50.,
)
assert len(panels) == 2
```

## 时间轨迹

源码：[traces.py](../../../../braincell/vis/traces.py)。

```text
plot_traces(morpho, time, values_over_time, *, locset=None, labels=None,
    colors=None, cmap="tab10", layout=None, shape=None, time_unit_label=None,
    value_unit_label=None, figsize=None, sharex=True, show_morphology=True)
    -> TracesResult
```

| 参数 | 类型与语义 |
| --- | --- |
| `morpho` | Morphology |
| `time` | `(T,)` 时间数组，可带时间单位 |
| `values_over_time` | `(T, n_locations)` 数组，可带物理单位；列顺序对应 locset 点顺序 |
| `locset=None` | 已求值 LocsetMask；有值时其点数必须等于列数，用于形态上标记位置；省略时按列绘制轨迹 |
| `labels=None` | 每列一个标题，默认 `Loc i` |
| `colors=None`、`cmap="tab10"` | 每列一个 Matplotlib 颜色；省略 colors 时从 cmap 取色 |
| `layout/shape=None` | 形态面板的布局与形状；显式传值可固定轨迹标记与底图使用的布局 |
| `time_unit_label/value_unit_label=None` | 单位标签覆盖；省略时从 Quantity 提取 |
| `figsize=None` | 英寸，默认 `(10, max(2*n_locations, 3))` |
| `sharex=True` | 轨迹面板共享时间轴 |
| `show_morphology=True` | 同时绘制形态；设 False 只画轨迹 |

返回 `braincell.vis.traces.TracesResult`，冻结 dataclass，字段为 `figure: Figure`、
`morpho_axes: Axes | None`、`trace_axes: tuple[Axes, ...]`。这些字段引用新建图对象，
可继续自定义。时间维度、列数、标签或颜色数量不匹配会抛出 `ValueError`；构图需要至少一列。

当前省略 layout/shape 时，底图使用全局默认，标记场景却回退到 stem/frustum，可能导致
标记与形态错位。绘制带 locset 的轨迹时同时显式传入这两个参数，下面的示例使用 fan/line。

下面用两帧样例数据说明输入形状，实际使用时传入对应位点的记录结果：

```python
time = np.array([0., 0.1]) * u.ms
trace_values = np.array([[-65.], [-64.]]) * u.mV
traces = vis.plot_traces(
    morpho, time, trace_values, locset=location_mask, layout="fan", shape="line",
)
assert len(traces.trace_axes) == 1
vis.save_figure(traces.figure, "traces.png")
```

## 动画

源码：[movie.py](../../../../braincell/vis/movie.py)。

```text
plot_movie(morpho, values_over_time, *, dt=None, dimensionality="2d",
    out=None, fps=30, cmap="viridis", vmin=None, vmax=None, value_label=None,
    layout=None, shape=None, layout_config=None, mode=None, ax=None,
    figsize=None) -> MovieResult
```

`values_over_time` 是 `(T, N)`，`T > 0`，每帧 `N` 遵循形态值的三种长度；时间或空间形状
错误为 `ValueError`。`dt=None` 用帧号标记，时间量用于标题时间；`fps=30` 决定播放/导出帧率，
两者职责不同。`vmin/vmax=None` 从所有帧统一计算范围，动画中保持色标固定。

| 参数 | 行为 |
| --- | --- |
| `dimensionality="2d"` | Matplotlib FuncAnimation；`"3d"` 使用 PyVista，其余值为 ValueError |
| `out=None` | 2D 返回内存动画；提供路径时导出，父目录需已存在。2D 支持 GIF/MP4，3D 用 movie writer |
| `fps=30` | 输出帧率，使用正整数 |
| `cmap/value_label` | 色图及显式色条标签；当前动画入口先提取裸数值，需用 value_label 显式写明物理单位 |
| `layout/shape/layout_config=None` | 2D 布局，含义同 plot2d |
| `mode=None` | 3D 场景默认 skeleton；动画渲染使用粗中心线，不生成逐帧管网 |
| `ax=None`、`figsize=None` | 2D Axes 与图尺寸；有 ax 时 figsize 不生效，3D 忽略这两个参数 |

返回 `braincell.vis.movie.MovieResult`：`animation` 为 FuncAnimation 或 Plotter，
`frames` 为输入帧数，`output_path` 为导出 Path 或 `None`。保留结果引用可避免动画提前回收。
3D 未指定 out 时返回初始场景，逐帧写入发生在输出文件路径；有输出时最终关闭 Plotter。
GIF 使用 Pillow writer，MP4 通常需要 FFmpeg；缺失环境由对应 writer 报错。

```python
movie_values = np.array([[-65., -60.], [-64., -58.]]) * u.mV
movie = vis.plot_movie(
    morpho, movie_values, dt=0.1 * u.ms, out="voltage.gif",
    layout="fan", value_label="Voltage [mV]",
)
assert movie.frames == 2
```

## 结构分析

### 形态树与 Sholl

源码：[morphometry.py](../../../../braincell/vis/morphometry.py)。

```text
plot_dendrogram(morpho, *, ax=None, color_by_type=True, linewidth=1.5) -> Axes
plot_topology(morpho, *, ax=None, color_by_type=True) -> Axes
plot_sholl(morpho, *, ax=None, step_um=10.0, max_radius_um=None,
    color="tab:blue") -> Axes
plot_branch_order_histogram(morpho, *, ax=None, color="tab:gray") -> Axes
```

四个函数接收 Morphology，`ax=None` 新建 Axes；每个都返回实际绘图轴。
`color_by_type=True` 按 branch 类型着色，False 使用统一颜色；`linewidth=1.5` 控制树状图线宽。
`color` 接收 Matplotlib 颜色。

树状图展示累计路径长度，拓扑图展示分支连接关系；分支阶数直方图以根为 0 阶，按树深度计数。
Sholl 在有坐标时使用根近端为中心的径向距离，缺少完整坐标时使用路径距离。
`step_um=10.0`、`max_radius_um=None` 接收以微米计的裸数值；后者省略时从几何推导最大半径。
步长非正为 `ValueError`。Sholl 当前只统计近端在阈值内、远端在阈值外的 segment，
径向向内穿越和同一线段两次穿过球面的情况不做完整几何求交。

```python
vis.plot_dendrogram(morpho)
vis.plot_topology(morpho)
vis.plot_sholl(morpho, step_um=5.)
vis.plot_branch_order_histogram(morpho)
```

### 离散点拓扑

源码：[point_topology.py](../../../../braincell/vis/point_topology.py)。

```text
plot_point_topology(node_tree, *, preset="dendrotweaks", layout=None,
    layout_scale=1.0, highlight_point_ids=None, highlight_fractions=None,
    coverage_mode="fraction", highlight_color="#ef4444", color_mode=None,
    values=None, cmap=None, vmin=None, vmax=None, norm=None, value_label=None,
    value_unit_label=None, show_colorbar=True, node_color=None,
    edge_color=None, root_color=None, ax=None) -> Axes
```

`node_tree` 为非空 `bc.NodeTree`，从已初始化 Cell 可取 `cell.node_tree`。
它接收已经解析的点数据，区别于 Cell 接口负责解析表达式与字段。
`preset` 可选 `dendrotweaks`（默认配色）、`mono`（单色）、`depth`（按树深度着色），
三者默认布局均为 `twopi`，数值模式默认色图为 `viridis`；`dendrotweaks` 使用统一节点色与根色。

| 特有参数 | 类型与行为 |
| --- | --- |
| `highlight_point_ids=None` | 整数 ID 可迭代对象，对指定点全强度高亮 |
| `highlight_fractions=None` | `dict[int, float]`，点 ID 到 `[0,1]` 覆盖比例；优先于 ID 高亮 |
| `color_mode=None` | 根据 values 或预设推断，可选 `solid`、`depth`、`values` |
| `values=None` | `(n_point,)` 数组或 Quantity；与高亮参数互斥，显式 values 模式要求提供数组 |
| `value_unit_label=None` | 色条单位后缀；省略时从 Quantity 提取 |

其他参数沿用[拓扑图公共样式](#拓扑图的公共样式与错误)。输入类型错误为 `TypeError`；
空树、错误形状、未知预设/布局/颜色模式、无效间距或参数互斥为 `ValueError`。

```python
vis.plot_point_topology(cell.node_tree, layout="kamada_kawai", highlight_point_ids=[0])
```

## 布局与缓存

### LayoutConfig

`LayoutConfig` 是冻结 dataclass；构造参数顺序与下表一致，均有默认值，可按关键字覆盖。
`collision_retry_limit`、`stem_collision_window` 为整数，其余字段为 float。
`layout_config=None` 使用同样的内置默认配置。几何参数使用名称中注明的微米、弧度或比例裸数值。
完整实现见 [layout/_config.py](../../../../braincell/vis/layout/_config.py)。

| 构造参数 | 默认值 | 含义 |
| --- | --- | --- |
| `collision_margin_um` | `2.0` | 碰撞软惩罚距离 |
| `collision_retry_limit` | `8` | 候选放置重试次数 |
| `stem_collision_window` | `24` | stem 检查的最近分支数 |
| `collision_cell_size_um` | `20.0` | 碰撞空间哈希网格尺寸 |
| `default_bend_fraction` | `0.4` | 默认弯曲段占分支长度的比例 |
| `balloon_bend_fraction` | `0.22` | balloon 弯曲比例 |
| `fan_bend_fraction` | `0.24` | fan 弯曲比例 |
| `radial_bend_fraction` | `0.25` | radial 弯曲比例 |
| `stem_root_full_span_rad` | `radians(150)` | stem 根统一扇区角宽 |
| `stem_root_group_span_rad` | `radians(120)` | stem 类型分组扇区角宽 |
| `balloon_root_span_rad` | `radians(180)` | balloon 根扇区角宽 |
| `balloon_child_span_rad` | `radians(120)` | balloon 子分叉角宽 |
| `balloon_type_split_span_rad` | `radians(110)` | balloon 类型分组角宽 |
| `fan_root_left_span_rad` | `radians(95)` | fan 左侧根扇区角宽 |
| `fan_root_middle_upper_span_rad` | `radians(70)` | fan 中部上扇区角宽 |
| `fan_root_middle_lower_span_rad` | `radians(70)` | fan 中部下扇区角宽 |
| `fan_root_right_span_rad` | `radians(95)` | fan 右侧根扇区角宽 |
| `radial_root_span_rad` | `2*pi` | radial 根扇区角宽 |
| `radial_child_span_rad` | `radians(150)` | radial 子分叉角宽 |
| `legacy_root_child_span_rad` | `radians(120)` | 历史布局参数，当前高层布局枚举不提供 legacy |
| `fan_root_left_max_parent_x` | `0.02` | fan 左扇区最大父位置 |
| `fan_root_middle_min_parent_x` | `0.35` | fan 中部父位置下界 |
| `fan_root_middle_max_parent_x` | `0.65` | fan 中部父位置上界 |
| `stem_collision_weight` | `100.0` | 碰撞代价权重 |
| `stem_tail_delta_weight` | `3.0` | 尾部角偏移权重 |
| `stem_settle_delta_weight` | `0.8` | 稳定角偏移权重 |
| `stem_overturn_weight` | `6.0` | 反向弯曲代价权重 |
| `stem_trunk_tail_delta_weight` | `0.75` | 主干尾部角偏移权重 |
| `stem_side_opening_weight` | `2.0` | 侧枝展开权重 |

构造器本身不统一校验参数范围，算法在使用时消费字段。修改配置通过构造新对象完成。

```python
layout_config = vis.LayoutConfig(collision_margin_um=3., collision_retry_limit=12)
vis.plot2d(morpho, layout="stem", layout_config=layout_config)
```

### LayoutCache

```text
LayoutCache(maxsize=64)
LayoutCache.get_or_build(self, morpho, *, mode, layout_family, root_layout,
    min_branch_angle_deg, layout_config, build) -> tuple[LayoutBranch2D, ...]
LayoutCache.clear(self) -> None
len(cache) -> int
```

源码：[layout/_cache.py](../../../../braincell/vis/layout/_cache.py)。`maxsize` 为正整数，非正值
抛出 `ValueError`。缓存键包含形态几何、连接与布局参数；`mode/layout_family/root_layout`
是布局构建器对应的字符串，角度可为 float 或 None，`layout_config` 可为 None。
`build()` 是无参数回调，仅 miss 时调用，返回布局分支序列。
命中时返回同一缓存 tuple，`hits/misses` 是可读取的计数；超容量淘汰最久未访问项。
`clear()` 清空内容并归零计数。缓存几何键有微米数值的小数舍入，极小变化可能复用同一项。

高层 `plot2d` 自动使用布局分发器的缓存，公开签名没有 `cache=` 参数。
这个类型供直接使用布局构建器的高级调用方管理缓存。

## 样式配置

源码：[config.py](../../../../braincell/vis/config.py)。

```text
configure_defaults(*, layout_2d_default=None, shape_2d_default=None,
    mode_3d_default=None, branch_type_colors=None, branch_type_edge_colors_2d=None,
    replace_branch_type_colors=False, replace_branch_type_edge_colors_2d=False,
    alpha_2d=None, alpha_2d_poly=None, alpha_2d_line=None,
    frustum_edge_linewidth_2d=None, alpha_3d_tube=None, highlight_color=None,
    highlight_alpha=None, marker_color=None, marker_size_2d=None,
    marker_radius_3d_um=None) -> VisDefaults
set_defaults(...)  # configure_defaults 的同一函数别名，完整参数相同
get_defaults() -> VisDefaults
reset_defaults() -> VisDefaults
theme(**overrides) -> context manager yielding VisDefaults
publication_theme(preset=None, *, rc_overrides=None)
    -> context manager yielding VisDefaults
```

`configure_defaults` 就地替换全局配置，`None` 表示保持原值。颜色映射默认合并，
`replace_branch_type_colors`、`replace_branch_type_edge_colors_2d` 为 True 时替换对应映射。
修改只影响后续绘图。`get_defaults()` 返回独立配置副本，包含颜色字典的复制；
修改副本不会更新全局。`reset_defaults()` 恢复初始配置并返回副本。

`VisDefaults` 是可变 dataclass，构造字段顺序与下表一致，每项可用同名关键字传入。

| 字段 | 初始默认值 | 类型与含义 |
| --- | --- | --- |
| `layout_2d_default` | `"fan"` | 2D 布局名 |
| `shape_2d_default` | `"frustum"` | 2D 图形模式 |
| `mode_3d_default` | `"geometry"` | 3D 模式 |
| `branch_type_colors` | 内置类型配色字典的副本 | 类型到 RGB，统一 2D/3D |
| `branch_type_edge_colors_2d` | `None` | 边框色映射；省略时从填色派生 |
| `alpha_2d` | `0.8` | 2D 透明度 |
| `alpha_2d_poly` | `None` | 截锥透明度，默认继承 alpha_2d |
| `alpha_2d_line` | `None` | 中心线透明度，默认继承 alpha_2d |
| `frustum_edge_linewidth_2d` | `0.9` | 截锥边框宽度 |
| `alpha_3d_tube` | `1.0` | 3D 管透明度 |
| `highlight_color` | `(255, 215, 0)` | region 高亮 RGB |
| `highlight_alpha` | `0.9` | 高亮透明度 |
| `marker_color` | `(30, 144, 255)` | locset 标记 RGB |
| `marker_size_2d` | `36.0` | 2D scatter 标记面积 |
| `marker_radius_3d_um` | `1.5` | 3D 标记半径，微米 |

颜色配置接受颜色名称、十六进制色和 RGB 序列。通过 configure 设置时，透明度必须在
`[0,1]`，边框宽度非负，标记尺寸为正；非法值为 `ValueError`。直接构造 dataclass 不执行
同样的校验。`theme(**overrides)` 使用 configure 的完整参数，退出时恢复配置，包括异常退出。

```text
PublicationTheme(branch_type_colors=<PUBLICATION_BRANCH_TYPE_COLORS 的副本>,
    branch_type_edge_colors_2d=None, rc_params=<PUBLICATION_RC_PARAMS 的副本>,
    alpha_2d=0.7, frustum_edge_linewidth_2d=0.9, alpha_3d_tube=1.0)
```

`PublicationTheme` 为冻结 dataclass，参数分别是类型配色、边框配色、Matplotlib rcParams
映射、2D 透明度、边框宽度及 3D 透明度。`publication_theme(preset=None)` 使用默认实例，
`rc_overrides=None` 可改为字典覆盖其 rc 设置；块退出时恢复两类全局配置。
未知 rc 键被忽略；未安装 Matplotlib 时仅应用 vis 配置。

`PUBLICATION_BRANCH_TYPE_COLORS` 是类型名到整数 RGB 的字典，包含 soma、axon、
basal_dendrite、apical_dendrite、dendrite 和 custom；`PUBLICATION_RC_PARAMS` 是
出版样式字典，包含字体、线宽、轴样式和默认 300 dpi 保存设置。通过副本定制可避免改动共享常量。

```python
with vis.theme(alpha_2d=0.6):
    vis.plot2d(morpho)
with vis.publication_theme(rc_overrides={"font.size": 10}):
    paper_ax = vis.plot2d(morpho)
    vis.save_figure(paper_ax, "morphology.pdf")
```

## 交互回调

源码：[hooks.py](../../../../braincell/vis/hooks.py)。

```text
VisHooks(on_pick=None, on_hover=None, on_leave=None)
VisHooks.is_active(self) -> bool
PickInfo(branch_index, branch_name, branch_type, segment_index=None,
    x=None, value=None, position_um=None, artist=None)
```

两者都是冻结 dataclass。回调签名为 `on_pick(info: PickInfo) -> None`、
`on_hover(info: PickInfo) -> None`、`on_leave() -> None`，由图形事件循环同步调用，
返回值被忽略。任意回调非 None 时 `is_active()` 为 True。回调中的耗时操作会阻塞交互。

| PickInfo 字段 | 类型与含义 |
| --- | --- |
| `branch_index` | int，morpho.branches 中的索引 |
| `branch_name/branch_type` | str，分支名与类型 |
| `segment_index` | int 或 None，分支内从近端开始的 segment 索引 |
| `x` | float 或 None，分支归一化弧长 `[0,1]` |
| `value` | float 或 None，所拾取图元的标量；单位需从原始数据获知 |
| `position_um` | ndarray 或 None，2D/3D 场景坐标，单位微米；自动布局坐标区别于原始 3D 坐标 |
| `artist` | 底层图元引用或 None，便于后端特定操作 |

Matplotlib 支持三个回调，但 Agg 等静态后端不会自动产生鼠标事件。
PyVista 仅支持 on_pick，部分位置与数值字段可能为空；Plotly 有内置悬停提示但没有接入 VisHooks。
拾取信息目前描述形态分支与 segment，没有统一的 CV/node ID 字段。

```python
picked = []
hooks = vis.VisHooks(on_pick=lambda info: picked.append((info.branch_index, info.x)))
interactive_ax = vis.plot2d(morpho, hooks=hooks)
assert hooks.is_active()
```

## 后端与导出

### 返回与显示

| 调用 | 返回结果 | 显示方式 |
| --- | --- | --- |
| Matplotlib 绘图 | `Axes`，或对应的复合结果 | `plt.show()`；Notebook 可展示图对象 |
| Plotly plot3d | `plotly.graph_objects.Figure` | `figure.show()` 或 Notebook 展示 |
| PyVista，`notebook=False` | `pyvista.Plotter` | `plotter.show()` |
| PyVista，Notebook，`return_plotter=False` | viewer，或支持 `_repr_html_()` 的 HTML 展示对象 | Notebook 渲染；HTML 路径用于嵌入展示 |
| PyVista，Notebook，`return_plotter=True` | `pyvista.Plotter` | 某些 Notebook 路径仍会先建立 viewer；需要原始对象且不触发 Notebook 展示时用 `notebook=False` |

每次绘图生成独立的图形对象或向指定 ax 添加内容；调用者负责保留动画引用和关闭不用的图。
PyVista Notebook 的环境与后端尝试失败时会抛出带诊断信息的 `RuntimeError`。

### save_figure

```text
save_figure(figure, path, *, dpi=None, transparent=False, format=None) -> pathlib.Path
```

源码：[export.py](../../../../braincell/vis/export.py)。
`figure` 接受 Matplotlib Axes/Figure、PyVista Plotter 或 Plotly Figure，复合结果先取其
`.figure`；PyVista HTML 展示对象不是此函数接收的图句柄。
`path` 为 str 或 PathLike，写入指定文件，父目录需存在，已有目标可能被覆盖。
`dpi=None` 使用后端默认，`transparent=False` 控制透明背景；`format=None` 从路径后缀推断。

| 后端 | 支持与差异 |
| --- | --- |
| Matplotlib | 调用 Figure.savefig，支持 PNG/PDF/SVG 等格式；dpi、transparent、format 直接传递 |
| PyVista | 位图截图或 HTML 导出；矢量路径调用可用的 `save_graphic`，缺少该能力时拒绝；dpi 用于截图缩放 |
| Plotly | HTML 使用 write_html；其他格式使用 write_image，通常需要 Kaleido；dpi 和 transparent 不传递 |

返回写入路径，不改变输入句柄的类型。未知图对象为 `TypeError`，PyVista 缺少矢量导出方法时为
`ValueError`，缺少目录或导出依赖时传播文件系统或后端异常。HTML/矢量分派依据路径后缀，
因此 `format` 应与后缀一致。动画文件通过 `plot_movie(out=...)` 输出。

## 验证入口与迁移讨论

接口边界和空间映射的测试位于 [cell_topology_test.py](../../../../braincell/vis/cell_topology_test.py)，
各绘图模块有同目录测试；完整使用流程见[可视化教程](../../../../examples/vis/vis.ipynb)。
[Visualization](visualization.md) 汇总后端能力和视觉回归缺口。
下一步接口调整和两个入口的候选设计见[迁移提案](../proposals/braintools-migration.md)。
