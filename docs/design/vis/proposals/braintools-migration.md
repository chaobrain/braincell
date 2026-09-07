# Braintools Migration

状态：讨论中。准备把可视化集中到 braintools 的一个模块，提供简单绘图和 GUI 两类入口。
现有 `braincell.vis` 承接简单绘图，GUI 后续接入 BrainVis。两个入口需要共用形态、
数值与位点映射，当前接口契约见 [Vis API](../current/api.md)。

## 当前调用与问题

已有且已初始化的 Cell，当前这样查看形态和膜电压：

```python
from braincell import vis

shape_ax = vis.plot2d(cell.morpho)
voltage_ax = vis.plot_cell_topology(cell, level="cv", value="V")
```

这两个调用的输入语义不同：形态图接收 Morphology，`values` 按 branch、segment 或
中心线点解释；Cell 图接收 Cell，`value` 按 CV/node 或字段选择器解释。
例如 CV 电压数组即使恰好与 segment 数量相同，直接交给形态图也可能画到错误位置。
GUI 选择一个位点后查看曲线，同样需要明确它对应哪个 CV 或 node。

vis 的场景构建依赖 Morphology，Cell 拓扑依赖 NodeTree、region/locset 和内部字段解析。
BrainCell 已经依赖 braintools，搬动代码时需要处理反向导入。迁移的核心是让两个入口
使用一致的数据解释，并把 BrainCell 对象适配与绘图、窗口显示分开。

## 模块位置与入口

下面两处位置可选；本提案的候选代码统一用 `braintools.visualize.cell` 展示：

| 位置 | 好处 | 成立条件与代价 |
| --- | --- | --- |
| `braintools.visualize.cell` | 沿用已有 visualize 命名空间，将细胞形态绘图与 GUI 放在一起 | 检查 visualize 的导入路径和命名，确保普通绘图不会加载 GUI |
| `braintools.vis` 等同级模块 | 可以独立组织这套接口与依赖 | 需要向用户说明它与已有 visualize 中绘图功能的分工 |

建议先采用已有命名空间下的专用子模块；最终名称需要结合 braintools 源码中的公开导出、
重名函数和可选依赖配置确认。“两个入口”指脚本绘图函数与 GUI 启动函数，两者可以复用数据层。

## Cell 如何传入

### 候选 A：直接传 Cell

以下为候选接口：

```python
from braintools.visualize import cell as cellvis

ax = cellvis.plot2d(cell, value="V")
window = cellvis.gui(cell)
```

调用者直接使用现有 Cell。适配器负责取形态、解析字段、确定空间位置，再交给绘图或 GUI。
现有 Morphology 调用也可继续接收 `cell.morpho`；增加 Cell 输入后，需要明确哪些参数
随输入对象改变含义。

这条路径改写调用较少，但必须安排适配器的归属与延迟导入。若适配器在 braintools，
它应是按需加载的 BrainCell 集成层；基础绘图模块不能在导入时依赖 BrainCell。
还需要记录 BrainCell 内部接口变化的适配责任。

### 候选 B：显式转换共享数据

以下为候选接口：

```python
from braintools.visualize import cell as cellvis

data = cellvis.from_cell(cell, value="V")
ax = cellvis.plot2d(data)
window = cellvis.gui(data)
```

`from_cell` 是候选便捷转换器，内部可调用 BrainCell 提供的数据导出能力。
`data` 表示某一时刻的形态、拓扑、字段值、单位和位置映射。绘图与 GUI 消费同一份数据，
可直接比较显示结果，也便于离线数据接入。

代价是需要定义转换结果和更新时间。数据快照不是活模型，Cell 推进后应重新提取；
位置回传使用稳定的 branch/CV/node 标识，不能只依赖渲染网格的临时下标。

| 比较项 | 直接传 Cell | 显式数据 |
| --- | --- | --- |
| 使用步骤 | 一次调用 | 转换后再绘图，可供多个视图复用 |
| BrainCell 依赖 | 调用时通过适配器读取 | 集中在转换器，渲染只消费数据 |
| 数据时刻 | 绘图时读取；GUI 持续读取需要另定规则 | 明确对应一次快照 |
| 迁移工作 | 保留现有对象调用习惯，持续维护适配 | 先定义共享数据和映射，便于独立验证 |

建议以共享数据为基础，支持直接传 Cell 的便捷调用，并让显式转换保持可用。
决定前先用一个带 region、多个 CV 和两帧电压结果的 Cell 验证两种调用能显示一致位置与数值。

## 还需要商量的接口选择

### 数值是否显式标明空间

可以保留当前形态 `values=` 与 Cell `value=` 两套语义，也可以采用统一的数值描述。
后一种方式需要显式标明 `branch/segment/centerline/cv/node` 空间，避免数组长度相等时误判。
候选调用示意：

```python
ax = cellvis.plot2d(cell, values=cv_voltage, space="cv")
```

建议保留命名字段的便捷写法 `value="V"`，显式数组增加空间信息。
准备时需明确：CV 值怎样投到中心线、边界点怎样着色、未覆盖机制怎样显示，以及如何指定
population 成员。现有行为及 NaN 映射见 [Cell 数值来源](../current/api.md#数值来源与空间)。

### 返回结果与显示方式

| 方案 | 使用体验 | 代价 |
| --- | --- | --- |
| 保留 Axes、Plotly Figure、PyVista Plotter | 可以继续使用原生后端的自定义方法 | 保存、显示和关闭方式随后端变化，Notebook 返回值还需明确 |
| 返回统一绘图结果，内部保存后端对象 | 可以提供一致的显示、导出和关闭方法 | 增加包装层，需要覆盖现有后端特性及旧返回值兼容 |

建议迁移简单绘图时保留原生结果，GUI 返回独立窗口句柄。简单绘图的创建与显示行为应明确，
尤其要处理现有 `show`、`notebook` 与 `return_plotter` 的差异。
若常见调用仍需大量后端分支，再根据示例评估统一结果对象。

### GUI 刷新与选择回传

共享数据至少要表达坐标和半径、拓扑连接、空间 ID、数值单位、时间轴及已解析选择。
GUI 的单次查看可先读取快照；需要查看新状态时显式刷新。持续实时跟随 Cell 需要确定
采样时机和数据提供方，已有图形回调也需要从 branch/segment 扩展到可追溯的空间标识。

下一步用“点选一个 CV，展示该 CV 的轨迹”检验数据是否足够，并明确选中结果返回给调用者
的内容。窗口刷新与选择接口据此讨论。

### 旧入口如何过渡

可保留 `braincell.vis` 和 Morphology/Branch 快捷方法，内部转发到 braintools；
也可以要求调用者迁移到新导入路径。建议先转发，以便现有脚本逐步迁移。

转发成立的条件是对应参数、返回对象、显示行为和异常仍可兼容。新增空间描述等改变需要
显式转换，不能只替换 import。保留期限和弃用提示需结合版本发布安排决定；具体旧调用
清单以 [Vis API](../current/api.md) 为依据。

## 准备工作与验证

1. 对照公共接口规格清点源码、测试、样例和依赖，定位 Morphology、Cell、NodeTree、
   region/locset、单位和内部字段解析的适配位置。
2. 检查 braintools 模块布局与可选依赖，确定共享数据和适配器归属；分别验证基础导入、
   简单绘图及 GUI 入口加载了哪些依赖。
3. 准备小型分叉形态、真实形态、CV/node 拓扑和两帧电压数据，验证几何、数值、单位、
   选择与轨迹在两个入口中对应同一位置。
4. 迁移现有测试与必要夹具，核对旧入口转发、返回值、初始化要求、population 选择、
   动画及文件导出；补齐代表性图像基线并比较大形态布局和渲染耗时。
5. 用实际可运行的接口更新教程和 Sphinx API，列出旧调用到新调用的对应关系。

已确认的方向是迁入 braintools 并提供简单绘图与 GUI；模块位置、数据入口和兼容方式仍在讨论。
事项进度见 [Vis TODO](../TODO.md)。
