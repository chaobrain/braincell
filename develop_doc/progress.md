# 当前进展

这份文档记录当前仓库已经落地到什么程度，方便下次直接接上。

## 已完成

### 包结构

- 包名已经稳定为 `braincell`
- 当前包目录为 `braincell/`
- `tests` 和 `examples` 都已经适配当前目录结构
- `pyproject.toml` 已能匹配当前非 `src` 包布局

### `morpho`

- `Branch` 已完成当前主数据模型：
  - 字段：`lengths / radii_prox / radii_dist / proximal_points / distal_points / name / type`
  - 单位统一基于 `brainunit`
  - 主构造器：
    - `Branch.lengths_shared(...)`
    - `Branch.lengths_paired(...)`
    - `Branch.xyz_shared(...)`
    - `Branch.xyz_paired(...)`
- `Branch` 已支持单段标量糖衣：
  - `Branch(lengths=10 * u.um, radii_prox=1 * u.um, radii_dist=0.5 * u.um, ...)`
  - `Branch.lengths_shared(lengths=10 * u.um, radii=[1, 0.5] * u.um, ...)`
- `braincell/morpho/morpho.py` 现在承载：
  - `BranchConnection`
  - `Morpho`
  - `MorphoBranch`
- `Morpho` 是唯一公开树对象，当前是可变的
- `Morpho` 现在还提供 `Morpho.from_swc(...)` 作为 SWC 导入主入口
- `MorphoBranch` 是树内分支视图，支持：
  - `tree.soma.dend = ...`
  - `tree.soma.attach(...)`
  - `tree.attach(...)`
  - 拓扑查询：`parent / parent_x / child_x / children`
  - 几何查询：`total_length / radius_proximal / radius_distal`
- `Morpho` 已支持按 `type` 自动命名：
  - 若 `Branch.name is None`，最终名字为 `type_N`
  - `N` 从 `0` 开始
  - 会跳过已占用名字，例如已有 `dend_20` 时，其他自动名仍可先得到 `dend_0`, `dend_1`, ...
  - `tree.soma.dend` 中的 `dend` 只是父节点下的访问槽位，不一定等于最终 branch 名
- `Morpho.topo()` 已可返回只读文本树，例如：
  - `soma`
  - `├── apical_dendrite_0`
  - `└── axon_0`
- `Morpho.vis3d(...)` 已可作为 3D 可视化便捷入口
- `Morpho.vis2d(...)` 已预留，但当前仍是 `NotImplementedError`
- `vis / filter / cell` 的公开入口现在只接受整棵 `Morpho`
- `Morphology` 和公开 `snapshot()` 已移除
- `MultiCompartment` 运行时已归档到 `legacy/morph`（仅阅读，不再保证可导入）
- `builder` 已移除
- 连接参数统一为 `parent_x / child_x`
- `Branch.type` 目前按有限集合校验：
  - `soma`
  - `axon`
  - `dend`
  - `basal_dend`
  - `basal_dendrite`
  - `apical_dend`
  - `apical_dendrite`
  - `custom`

### 单位处理

- `_units.py` 已收缩为参数规范化层
- `normalize_param(...)` 负责：
  - 补 base unit
  - 量纲兼容检查
  - shape 校验
  - bounds 校验
- 普通 quantity 运算已尽量还给 `brainunit`
- `segment_lengths_from_points(...)` 已使用 `u.math.linalg.norm(...)`

### 其他层

- `io/` 现在是与 `morpho` 同级的独立输入适配层
  - `braincell/io/swc/` 负责 SWC 导入
  - `braincell/io/asc/` 与 `braincell/io/neuroml2/` 当前仍是占位 reader
  - `SWC` 规则已改成单一 rulebook；每条规则同时负责 `check + correct`
  - `Morpho.from_swc(...)` 是主推入口
  - `SwcReader.read(..., return_report=True)` 与 `Morpho.from_swc(..., return_report=True)` 都可返回 `(Morpho, SwcReport)`
  - `SwcReader.check(...)` 会返回结构化 `SwcReport`
  - 支持标准 7 列 SWC
  - 特殊三点 soma 当前按“第一个点是中心点，后两个点挂在中心两侧”识别
  - 连续 degree-2 链会压缩成多段 `Branch`
  - SWC type 会映射到当前 `Branch.type`
  - 未知 SWC type 默认降级为 `custom`，同时记 warning
  - 当前已内置 rulebook 和 options 入口，便于后续继续补规则
  - `tests/morpho_files/*.swc` 已纳入 smoke + 基本不变量测试
- `mech` 已有基本数据容器
  - `ion / channel / synapse` 实现已落到 `braincell/mech/` 下
  - 当前推荐通过 `import braincell` 后访问 `braincell.ion / braincell.channel / braincell.synapse`
- `cell` 已有 `CellSpec`、`Discretizer`、`CellCompiler` 等骨架类型
- `vis` 现在已有：
  - `Morpho -> RenderGeometry3D` 的统一数据提取
  - 按 branch type 聚合的 3D batch 数据
  - `PyVistaBackend` 可选 backend
  - 当前仍只实现 3D，不含 2D
- `filter` 已有表达式类型和部分基础实现

### 当前验证

已确认以下命令可运行：

```bash
conda run -n brainunit python -m unittest discover -s tests -p 'test_*.py'
pytest --collect-only -q
```

当前测试数量：64 个（`pytest --collect-only -q`）。

## 仍未完成

### `morpho`

- `Branch.lateral_areas()`
- `Branch.volumes()`
- `morpho/metrics.py` 里的度量实现
- `io/asc/` 与 `io/neuroml2/` 中的 `AscReader` / `NeuroMlReader`

### `filter`

以下表达式仍大量是占位实现：

- 距离相关 region
- 欧氏距离相关 region
- 多种 locset 采样表达式
- 更完整的区间集合代数

### `cell`

- `cell/discretize.py` 仍未真正生成可用 CV 网格
- `cell/compile.py` 仍未把声明 lower 到 JAX-ready runtime
- 还未接入 `_base.py / channel / ion / quad`
- 还没有真实仿真执行路径



### `vis`

- `vis2d` 尚未实现
- `PyVistaBackend` 是可选依赖，当前 `brainunit` 环境已可运行真实文件 smoke 测试
- 还没有 Plotly 等其他 backend
- 还未实现 overlay 的真实渲染语义



## 重要设计决策

- `Branch` 保持纯几何，不存拓扑
- `Morpho` 是公开树对象，允许原位编辑
- `Morpho.topo()` 属于核心拓扑调试接口，不放在 `vis`
- 自动命名使用 `type_N`，而父节点属性槽位和最终 branch 名可以不同
- 不再公开冻结态 `Morphology`
- 不再保留 `builder`
- 单位全部基于 `brainunit`
- 包名固定为 `braincell`

## 下次继续时最值得做的事

优先级建议：

1. 补齐 `Branch` 的几何量实现：面积、体积
2. 在当前 SWC rulebook 基础上补 ASC 或 NeuroML 中的一种
3. 让 `filter` 的 region / locset 真正可用
4. 明确 `cell` 的运行时数据结构，再对接 `_base.py / channel / ion / quad`
5. 扩展 `vis`：2D 入口、更多 backend、overlay

## 快速恢复上下文

建议下次先读：

1. `README.md`
2. `AGENTS.md`
3. `braincell/develop_doc/io.md`
4. `braincell/develop_doc/架构与迁移.md`
5. `braincell/develop_doc/morpho_结构图.md`
6. `braincell/morpho/branch.py`
7. `braincell/morpho/morpho.py`
8. `tests/test_morpho.py`
