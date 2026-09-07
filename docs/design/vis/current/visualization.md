# Visualization

`braincell.vis` 面向脚本和 Notebook，提供形态展示、数据着色、拓扑分析和结果导出。
公共入口见[包导出](../../../../braincell/vis/__init__.py)，完整用法见
[可视化教程](../../../../examples/multi_compartment/vis.ipynb)和 [API 文档](../../../apis/vis.rst)。
完整签名、参数、返回值和调用条件见 [Vis API](api.md)。

## 当前支持什么

| 能力 | 主要内容 | 代表接口 |
| --- | --- | --- |
| 形态展示 | 2D 真实坐标投影、树形与截锥布局，stem、balloon、radial 等布局；3D 骨架与几何展示 | `plot2d`、`plot3d`、`LayoutConfig` |
| 数据展示 | 按 branch、segment 或中心线采样点着色，带单位色条，region 高亮与 locset 标记 | `ValueSpec`、`OverlaySpec` |
| 对比与动态结果 | 多形态和多组数值对比，位点时间轨迹与形态联动，随时间着色的动画 | `compare_morphologies`、`compare_values`、`plot_traces`、`plot_movie` |
| 结构分析 | 树状图、拓扑图、Sholl 分析、分支阶数统计，以及 Cell 的 branch、CV、node 三级拓扑 | `plot_dendrogram`、`plot_topology`、`plot_sholl`、`plot_branch_order_histogram`、`plot_cell_topology`、`plot_point_topology` |
| 交互与输出 | 拾取回调、主题与出版样式、图片和动画导出 | `VisHooks`、`PickInfo`、`theme`、`publication_theme`、`save_figure` |

已有 Cell 时，形态图传 `cell.morpho`，拓扑与运行时数据图直接传 `cell`：

```python
from braincell import vis

ax = vis.plot2d(cell.morpho)
vis.save_figure(ax, "morphology.png")
vis.plot_cell_topology(cell, level="cv")
# cell 已初始化时，可读取当前膜电压。
vis.plot_cell_topology(cell, level="node", value="V")
```

带对象构造和初始化的完整示例见 [从 Cell 开始](api.md#从-cell-开始)。

## 后端与数据

| 后端 | 展示与输出 | 交互 |
| --- | --- | --- |
| Matplotlib | 2D 图、拓扑、轨迹，位图与矢量图导出；`plot_movie` 使用 `FuncAnimation` | `VisHooks` 支持拾取、悬停和离开，事件触发需要交互式后端 |
| PyVista | 3D 中心线与管状几何、截图和 HTML 导出；`plot_movie` 写出 3D 动画 | `VisHooks` 支持拾取 |
| Plotly | 3D 分支线、数值着色、HTML 导出；静态图片导出需要额外图像引擎 | 支持视角操作和悬停提示，尚未接入 `VisHooks` |

绘图依赖按需加载，`braincell[vis]` 包含 Matplotlib、NetworkX、PyVista 和 Plotly。
动画编码、Notebook 展示及部分导出格式还需要对应后端的运行环境。

现有数据流是“形态与位点数据 → 布局和场景 → 后端渲染”。场景构建负责将几何转换为
微米数值，保留着色数据的单位标签；布局配置和缓存用于重复绘图。
形态图读取 `Morphology`，Cell 拓扑图还读取离散结果及 region/locset 的解析结果。
`plot_cell_topology(level="node")` 需要先调用 `cell.init_state()`。

## 验证现状

[vis 源码目录](../../../../braincell/vis/)已有与模块相邻的布局、场景、后端、着色、
交互、导出和动画测试；布局、场景及 2D 渲染另有可选的 `pytest-benchmark` 用例。
当前缺少实际运行的像素回归基线。已有 Matplotlib artist 断言可检查图元和属性，
完整视觉回归还需要提交代表性基线图，并配置执行图像比较的 CI。

迁移方向见 [Braintools Migration](../proposals/braintools-migration.md)，事项进度见 [Vis TODO](../TODO.md)。
