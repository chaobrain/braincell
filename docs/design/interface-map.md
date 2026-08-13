# BrainCell 模块依赖与公共接口速览

本文基于当前仓库源码扫描整理，用于和合作者讨论接口命名与子模块分工。它不是完整 API 文档，也不替代 `README.md`、`TODO.md` 或自动生成的 `docs/apis/*` 文档。

## 1. 总体结构

当前主要代码在 `braincell/` 下，按职责大致分为以下层次：

```text
io -> morph -> filter
           -> mech
           -> _cv -> _compute -> _multi_compartment.Cell -> quad / brainstate / JAX

ion / channel / synapse -> concrete runtime mechanisms
vis -> consumes morph / filter / Cell data for plotting and export
_single_compartment -> classic single-compartment HH-style neuron frontend
_base / _base_ion / _base_channel -> shared runtime base classes
```

### 依赖强弱

- 较独立的声明层：`braincell.mech`。它主要定义机制声明对象、参数容器和注册表，不直接承担 JAX runtime 状态。
- 几何基础层：`braincell.morph`。多数上层模块依赖它；但当前实现里 `Morphology.from_*`、`vis2d`、`vis3d` 也会反向引用 `io` 和 `vis`，属于用户便利入口。
- 选择层：`braincell.filter`。依赖 `morph`，被 CV 划分、Cell paint/place、可视化 overlay 使用。
- CV 与 compute 层：`braincell._discretization` 和 `braincell._compute`。负责把 morphology、filter、mech 声明 lowering 到 control volumes、node tree 和 runtime layout。
- 最强编排层：`braincell._multi_compartment.Cell`。集中依赖 `morph`、`filter`、`mech`、`_discretization`、`_compute`、`quad`、`vis`。
- 数值积分层：`braincell.quad`。提供 integrator registry 和 solver step；会依赖 shared protocol，并有部分 solver 适配 multi/single-compartment 对象。
- 可视化层：`braincell.vis`。主要消费 `morph`、`filter`、Cell/point topology，不应作为核心计算依赖。

## 2. 模块职责与主要依赖

| 模块 | 当前职责 | 主要内部依赖 | 被谁依赖 |
| --- | --- | --- | --- |
| `braincell.morph` | Branch 几何、Morphology 树、metric、加载/绘图便利入口 | `_misc`、`io`、`filter`、`vis` | `filter`、`_discretization`、`_compute`、`Cell`、`io`、`vis` |
| `braincell.io` | SWC/ASC/NeuroML2/NeuroMorpho/checkpoint 输入输出 | `morph`、`io.swc`、`io.asc` | `Morphology.from_*`、用户 |
| `braincell.filter` | Region/locset 表达式与选择缓存 | `morph` | `_discretization`、`Cell`、`vis` |
| `braincell.mech` | 机制声明、参数、registry | 基本只依赖自身和 `brainunit` | `_discretization`、`_compute`、`Cell`、`channel`、`ion`、`synapse` |
| `braincell._discretization` | CV 对象、CV policy、paint/place lowering | `morph`、`filter`、`mech` | `_compute`、`Cell` |
| `braincell._compute` | point topology、runtime state、layout/table | `_discretization`、`morph`、`mech`、`ion` | `Cell`、`vis.point_topology` |
| `braincell._multi_compartment` | 多隔室 Cell 前端和运行时 facade | 几乎所有核心层 | 顶层 `braincell.Cell` |
| `braincell._single_compartment` | 单隔室 HH-style neuron | `_base`、`quad` | 顶层 `braincell.SingleCompartment` |
| `braincell.ion` | Na/K/Ca ion containers | `_base`、`mech`、`quad.protocol` | `channel`、runtime |
| `braincell.channel` | 具体 ion channel 实现 | `_base`、`ion`、`mech` | 用户、runtime registry |
| `braincell.synapse` | 具体 synapse 实现 | `mech` | 用户、runtime registry |
| `braincell.quad` | integrator registry 和 step functions | `_misc`、`quad.protocol` | single/multi-compartment runtime |
| `braincell.vis` | 2D/3D scene、backend、morphometry、traces | `morph`、`filter`、layout/backend | 用户、`Morphology.vis*`、`Cell.vis*` |

## 3. 顶层导出面

`braincell/__init__.py` 当前主要导出：

- 基础协议/基类：`DiffEqState`、`DiffEqModule`、`IndependentIntegration`、`HHTypedNeuron`、`IonChannel`、`Ion`、`MixIons`、`Channel`、`IonInfo`、`mix_ions`
- morphology：`Branch`、`Soma`、`Dendrite`、`Axon`、`BasalDendrite`、`ApicalDendrite`、`CustomBranch`、`Morphology`
- multi-compartment：`Cell`、`RunResult`
- single-compartment：`SingleCompartment`
- CV policy：`CV`、`CVPolicy`、`CVPerBranch`、`MaxCVLen`、`DLambda`、`CVPolicyByTypeRule`、`CompositeByTypePolicy`
- declaration helpers：`CableProperty`、`CurrentClamp`、`FunctionClamp`、`SineClamp`
- 子包：`channel`、`ion`、`mech`、`quad`、`synapse`、`vis`

命名注意点：

- `braincell.Channel` 是 runtime base class；`braincell.mech.Channel` 是 Cell paint 使用的声明层对象。后续讨论接口名时要明确这两个概念是否继续同名。
- `braincell.Ion` 同样是 runtime ion base class；`braincell.mech.Ion` 是声明层 density mechanism。
- `braincell.morph.__all__` 当前没有导出 `Branch` 和 `Morphology`，但顶层 `braincell` 有导出；如果希望 `braincell.morph.Branch` 成为稳定公共路径，需要单独确认。

## 4. Morphology / Branch 接口

### `Branch`

文件：`braincell/morph/branch.py`

主要构造与属性：

- `Branch.from_lengths(...)`
- `Branch.from_points(...)`
- `radii`
- `points`
- `n_segments`
- `length`
- `mean_radius`
- `areas`
- `area`
- `volumes`
- `volume`
- `vis2d(...)`
- `vis3d(...)`
- `save_checkpoint(path)`
- `Branch.load_checkpoint(path)`

类型化子类：

- `Soma`
- `Dendrite`
- `Axon`
- `BasalDendrite`
- `ApicalDendrite`
- `CustomBranch`
- `branch_class_for_type(branch_type)`

### `Morphology`

文件：`braincell/morph/morphology.py`

主要构造/加载：

- `Morphology.from_root(branch, name="soma")`
- `Morphology.from_swc(path, options=None, mode=None, return_report=False)`
- `Morphology.from_asc(path, return_report=False)`
- `Morphology.from_neuromorpho(...)`
- `Morphology.save_checkpoint(path)`
- `Morphology.load_checkpoint(path)`

树结构与查询：

- `root`
- `branches`
- `edges`
- `branch_by_order(order="default")`
- `branch(name=None, index=None, order=None)`
- `path_to_root(branch_index)`
- `path_length_to_root(branch_index)`
- `shortest_path_length(from_site, to_site)`
- `topo()`
- `attach(parent=..., child_branch=..., child_name=None, parent_x=1.0, child_x=0.0)`
- `select(expr, cache=None)`

metric 属性：

- `metric`
- `has_full_point_geometry`
- `total_length`
- `mean_radius`
- `total_area`
- `total_volume`
- `n_branches`
- `n_stems`
- `n_bifurcations`
- `x_range`
- `y_range`
- `z_range`
- `max_branch_order`
- `max_euclidean_distance`
- `max_euclidean_distance_excluding_soma`
- `max_path_distance`
- `max_path_distance_excluding_soma`

可视化便利入口：

- `vis2d(...)`
- `vis3d(...)`

辅助类型：

- `MorphoBranch`：`index`、`index_by(...)`、`parent`、`children`、`n_children`、`attach(...)`，并支持 attribute-style child assignment。
- `MorphoEdge`：parent-child attachment edge。
- `MorphoMetric`：`from_morpho(...)`、`as_dict()`。

## 5. Cell / Multi-Compartment 接口

文件：`braincell/_multi_compartment/cell.py`

### 构造与声明期接口

- `Cell(morpho, cv_policy=None, V_th=..., V_init=None, spk_fun=..., solver="staggered", name=None)`
- `morpho`
- `cv_policy`
- `paint_rules`
- `place_rules`
- `V_th`
- `V_init`
- `solver`
- `solver_name`
- `spk_fun`
- `name`
- `paint(region, *mechanisms)`
- `place(locset, *mechanisms)`

### CV preview 与生命周期

- `n_cv`
- `cvs`
- `init_state(batch_size=None)`
- `reset()`
- `reset_state(batch_size=None)`
- `runtime`

### Runtime / simulation 接口

- `n_point`
- `pop_size`
- `varshape`
- `n_compartment`
- `node_tree`
- `node_scheduling(max_group_size=32, algorithm="dhs")`
- `current_time`
- `pre_integral(I_ext=0.0)`
- `compute_derivative(I_ext=0.0)`
- `compute_membrane_derivative(V, I_ext=0.0)`
- `compute_axial_derivative(V)`
- `compute_voltage_derivative(V, I_ext=0.0)`
- `post_integral(I_ext=0.0)`
- `update(I_ext=None)`
- `run(dt=..., duration=...)`

### 状态与机制查询

- `layouts`
- `voltage_shape`
- `get_point_layouts(point_id)`
- `get_cv_layouts(cv_id)`
- `expected_state_shape(layout_id, var_name)`
- `get_state(layout_id, var_name)`
- `set_state(layout_id, var_name, value)`
- `get_point_state(point_id)`
- `get_cv_state(cv_id)`
- `get_runtime_node(layout_id)`
- `get_ion(name)`
- `sample_probe(name)`
- `sample_probes()`
- `mech_table()`

### Cell 可视化入口

- `vis_topology(...)`
- `vis_node(...)`
- `vis_cv(...)`
- `vis_branch(...)`

命名建议关注点：

- `reset()` 与 `reset_state()` 语义不同：前者回到声明期，后者重置 runtime state。建议后续文档中明确命名或别名策略。
- `node_tree` 和 `runtime` 现在都是属性风格查询接口。

## 6. CV / Discretization 接口

主要文件：`braincell/_discretization/base.py`、`braincell/_discretization/policy.py`

公开类型：

- `CV`
- `CVPolicy`
- `CVPerBranch`
- `MaxCVLen`
- `DLambda`
- `CVPolicyByTypeRule`
- `CompositeByTypePolicy`

主要接口：

- `CV.region`
- `CV.diam_mid`
- `CVPolicy.resolve_cv_bounds(morpho, paint_rules=None)`
- `build_discretization(morpho, policy=..., paint_rules=..., place_rules=...)`

当前 CV 层更像内部 lowering 层，但 `CV` 和 policy 类已经从顶层 `braincell` 导出，属于需要稳定命名的接口。

## 7. Mechanism Declaration 接口

文件：`braincell/mech/*`

### 基础与参数

- `Mechanism`
- `Params`
- `Params.keys()`
- `Params.values()`
- `Params.items()`
- `Params.get(...)`
- `Params.with_updates(...)`
- `Params.without(...)`
- `Params.coerce(...)`

### Cable / density mechanisms

- `CableProperty`
- `CableProperty.with_updates(...)`
- `Density`
- `Density.instance_name`
- `Density.identity`
- `Density.with_params(...)`
- `Density.with_coverage(...)`
- `Density.with_name(...)`
- `mech.Channel(class_name, ..., ion_name=None, ion_names=None, **params)`
- `mech.Ion(class_name, ..., **params)`

### Point mechanisms

- `Point`
- `CurrentClamp`
- `CurrentClamp(delay=..., durations=duration, amplitudes=amplitude)`
- `SineClamp`
- `FunctionClamp`
- `StateProbe`
- `MechanismProbe`
- `CurrentProbe`
- `ProbeMechanism`
- `Synapse`
- `Synapse.instance_name`
- `Synapse.identity`
- `Junction`

### Registry

- `MechanismEntry`
- `MechanismRegistry`
- `MechanismRegistry.register(...)`
- `MechanismRegistry.unregister(...)`
- `MechanismRegistry.add_alias(...)`
- `MechanismRegistry.clear()`
- `MechanismRegistry.contains(...)`
- `MechanismRegistry.get(...)`
- `MechanismRegistry.entry(...)`
- `MechanismRegistry.names(...)`
- `MechanismRegistry.items(...)`
- `get_registry()`
- `register_channel(...)`
- `register_ion(...)`
- `register_synapse(...)`

## 8. Filter / Selection 接口

文件：`braincell/filter/region.py`、`braincell/filter/locset.py`

### Region

- `RegionMask`
- `RegionExpr`
- `RegionExpr.complement()`
- `RegionExpr.evaluate(morpho, cache=None)`
- `AllRegion`
- `EmptyRegion`
- `BranchSlice`
- `BranchInFilter`
- `BranchRangeFilter`
- `RadiusRangeRegion`
- `TreeDistanceRegion`
- `EuclideanDistanceRegion`
- `SubtreeRegion`
- `RegionSetOp`
- `branch_in(property, values)`
- `branch_range(property, bounds, closed="neither")`

### Locset

- `LocsetMask`
- `LocsetExpr`
- `LocsetExpr.evaluate(morpho, cache=None)`
- `AtLocation`
- `at(branch, x)`
- `RootLocation`
- `BranchPoints`
- `Terminals`
- `RegionAnchors`
- `UniformSamples`
- `RandomSamples`
- `StepSamples`
- `LocsetSetOp`

### Cache

- `SelectionCache`

## 9. Runtime Base / Single-Compartment 接口

### Runtime base classes

文件：`braincell/_base.py`、`braincell/_base_ion.py`、`braincell/_base_channel.py`

- `HHTypedNeuron`：`pop_size`、`n_compartment`、`current(...)`、`pre_integral(...)`、`compute_derivative(...)`、`post_integral(...)`、`init_state(...)`、`reset_state(...)`、`add(...)`、`get_spike(...)`
- `Ion`：`external_currents`、`pre_integral(V)`、`compute_derivative(V)`、`post_integral(V)`、`current(V, include_external=False)`、`init_state(V, batch_size=None)`、`reset_state(V, batch_size=None)`、`update(V, ...)`、`register_external_current(...)`、`pack_info()`、`add(...)`
- `MixIons`：`ion_types`、`pre_integral(V)`、`compute_derivative(V)`、`post_integral(V)`、`current(V)`、`init_state(...)`、`reset_state(...)`、`update(...)`、`add(...)`
- `mix_ions(...)`
- `IonChannel`：`varshape`、`current(...)`、`pre_integral(...)`、`compute_derivative(...)`、`post_integral(...)`、`reset_state(...)`、`init_state(...)`、`update(...)`
- `IonInfo`
- `Channel`
- `Synapse`

### `SingleCompartment`

文件：`braincell/_single_compartment/base.py`

- `SingleCompartment(...)`
- `pop_size`
- `n_compartment`
- `area`
- `init_state(batch_size=None)`
- `reset_state(batch_size=None)`
- `pre_integral(I_ext=...)`
- `compute_derivative(I_ext=...)`
- `post_integral(I_ext=...)`
- `update(I_ext=...)`
- `soma_spike()`

## 10. Ion / Channel / Synapse 具体实现

### `braincell.ion`

主要导出：

- `FixedIon`
- `InitNernstIon`
- `DynamicNernstIon`
- `Calcium`、`CalciumFixed`、`CalciumInitNernst`、`CalciumDetailed`、`CalciumFirstOrder`
- `Potassium`、`PotassiumFixed`、`PotassiumInitNernst`
- `Sodium`、`SodiumFixed`、`SodiumInitNernst`
- `build_placeholder_ions(size=(1,))`

### `braincell.channel`

按文件分组：

- `leaky`：`LeakageChannel`、`IL`
- `sodium`：`Na_Ba2002`、`Na_TM1991`、`Na_HH1952`、`NaF_SU2015_DCN`、`NaP_SU2015_DCN`、`Na_ZH2019_IO`、`Nav1p6_MA2020_GoC`、`Nav1p6_MA2024_PC`、`Nav1p6_MA2025_BC`、`Nav1p6_RI2021_SC`、`Nav1p1_MA2025_BC`、`Nav1p1_RI2021_SC`、`Nav_MA2020_GrC`、`NaFHF_MA2020_GrC`
- `potassium`：`KDR_Ba2002`、`K_TM1991`、`K_HH1952`、`KA1_HM1992`、`KA2_HM1992`、`KK2A_HM1992`、`KK2B_HM1992`、`KNI_Ya1989`、`K_Leak`、`KM_*`、`Kv*`、`Kir*`、`Kdr_ZH2019_IO` 等
- `calcium`：`CaN_IS2008`、`CaT_HM1992`、`CaT_HP1992`、`CaHT_HM1992`、`CaHT_Re1993`、`CaL_IS2008`、`Cav*`、`CaHVA_*`、`Ca_ZH2019_IO`
- `potassium_calcium`：`AHP_De1994`、`Kca3p1_MA2020_GoC`、`Kca2p2_MA2020_GoC`、`Kca1p1_MA2020_GoC`
- `hyperpolarization_activated`：`HCN_HM1992`、`HCN1_*`、`HCN2_MA2020_GoC`、`HCN_SU2015_DCN`、`HCN_ZH2019_IO`
- `channel._base`：`Gate`、`Transition`、`HH`、`Markov`、`ghk_flux`

命名注意点：

- 当前实际类名多为 `Na_HH1952`、`KDR_Ba2002`、`CaL_IS2008`，但 README/docs 的部分示例仍使用 `INa_HH1952`、`IKDR_Ba2002`、`ICaL_IS2008` 这类旧命名。统一接口名时需要决定是否保留 alias、更新文档，或迁移到无 `I` 前缀命名。

### `braincell.synapse`

- `AMPA`
- `GABAa`
- `NMDA`

## 11. IO 接口

主要导出：

- `SwcReader`
- `SwcReadOptions`
- `SwcReport`
- `SwcIssue`
- `AscReader`
- `AscReport`
- `AscIssue`
- `AscMetadata`
- `AscSpineRecord`
- `NeuroMlReader`
- `NeuroMorphoClient`
- `NeuroMorphoCache`
- `NeuroMorphoQuery`
- `NeuroMorphoNeuron`
- `NeuroMorphoMeasurement`
- `NeuroMorphoSearchPage`
- `NeuroMorphoDetail`
- `NeuroMorphoUrls`
- `NeuroMorphoFilePlan`
- `NeuroMorphoDownloadItem`
- `NeuroMorphoDownloadRecord`
- `NeuroMorphoCacheStatus`
- `fetch_neuromorpho(...)`
- `load_neuromorpho(...)`
- `save_branch(...)`
- `load_branch(...)`
- `save_morpho(...)`
- `load_morpho(...)`

## 12. Quad / Integration 接口

主要导出：

- `get_integrator(method)`
- `register_integrator(...)`
- `get_registry()`
- `IntegratorEntry`
- `IntegratorRegistry`
- `all_integrators`
- explicit / RK：`euler_step`、`midpoint_step`、`rk2_step`、`heun2_step`、`ralston2_step`、`rk3_step`、`heun3_step`、`ssprk3_step`、`ralston3_step`、`rk4_step`、`ralston4_step`
- exponential：`exp_euler_step`、`ind_exp_euler_step`
- implicit：`backward_euler_step`、`implicit_euler_step`、`splitting_step`、`implicit_rk4_step`、`implicit_exp_euler_step`、`cn_rk4_step`、`cn_exp_euler_step`、`exp_exp_euler_step`
- cable-specific：`staggered_step`
- protocol：`DiffEqState`、`DiffEqModule`、`IndependentIntegration`

## 13. Visualization 接口

主要导出：

- `plot2d(...)`
- `plot3d(...)`
- `plot_traces(...)`
- `plot_movie(...)`
- `plot_topology(...)`
- `plot_point_topology(...)`
- `plot_dendrogram(...)`
- `plot_sholl(...)`
- `plot_branch_order_histogram(...)`
- `compare_morphologies(...)`
- `compare_values(...)`
- `save_figure(...)`
- `configure_defaults(...)`
- `get_defaults()`
- `set_defaults(...)`
- `reset_defaults()`
- `publication_theme(...)`
- `theme(...)`
- `LayoutCache`
- `LayoutConfig`
- `ValueSpec`
- `OverlaySpec`
- `VisDefaults`
- `PublicationTheme`
- `PickInfo`
- `VisHooks`

## 14. 后续统一接口名时建议优先讨论的问题

1. `braincell.Channel` / `braincell.Ion` 与 `braincell.mech.Channel` / `braincell.mech.Ion` 是否继续共用名称。
2. channel 类名是否统一使用无 `I` 前缀，例如 `Na_HH1952`，并为旧 `INa_HH1952` 提供 alias 或弃用提示。
3. `Morphology` 是否应该在 `braincell.morph` 子包直接导出，和顶层 `braincell.Morphology` 保持一致。
4. Cell 生命周期接口是否保留 `reset()` 与 `reset_state()` 两套名称，或增加更明确的别名。
5. `Cell` 的查询接口是否统一 property/method 风格，例如 `node_tree`、`runtime`、`layouts`。
6. `_discretization`、`_compute` 当前以下划线标识内部模块，但 `CV` 和 policy 已从顶层导出；需要确定哪些类型是稳定公共 API。
7. docs 中旧结构文件如 `docs/apis/morphology.rst` 仍包含 `Section`、`Segment` 等旧名字，建议后续单独清理。
