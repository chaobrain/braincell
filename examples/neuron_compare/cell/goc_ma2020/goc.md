# GoC formal template

这个目录的正式 GoC 模板由四个文件组成：

- `Optimization_result.txt`: 本 cell 模板自带的优化参数，避免正式模板依赖 debug 目录。
- `parameters.py`: 只放参数、默认路径、region 规则和 nseg/CV 规则。
- `goc_braincell.py`: BrainCell 构建版本。
- `goc_neuron.py`: NEURON 构建版本。

`debug/` 目录保留调试、对照、probe、table、toggle 等内容；正式模板不 import `debug/`，也不改动 `debug/`。

正式最小验证入口是 `run.ipynb`，使用正式 BrainCell/NEURON builders，温度与当前 debug notebook 对齐为 `30.0` Celsius。

## Parameter layout

参数按使用位置分层：

- `GoCChannelParameters`: channel conductance/permeability，继续按 `soma`、`dend_apical`、`dend_basal`、`axon_ais`、`axon_regular` 分区。
- `GoCCableParameters`: `Ra`、leak reversal/conductance、`cm`、`cv_max_len_um`。
- `GoCIonParameters`: Na/K/Ca reversal 和 `CdpStC` pump。
- `GoCParameters`: 聚合 `channel`、`cable`、`ion`。

正式代码里使用 `params.channel`、`params.cable`、`params.ion`。不要再生成自动加单位的中间 `SimpleNamespace`；BrainCell 侧在 `paint()` 调用处手动写单位，例如 `g_max=ch.soma.nav * (u.siemens / u.cm**2)`。

`load_goc20_params()` 只负责读取 `Optimization_result.txt` 并生成 `GoCParameters`。如果换一组优化结果，只替换路径或文件内容，不改构建类。

## Region rules

GoC morphology 期望导入后有：

- soma: 1
- dendrite: 151
- axon: 75

dendrite 分区沿用源实现：

- `dend_basal`: `0-3`、`16-17`、`33-41`、`84`、`105-150`
- `dend_apical`: `4-15`、`18-32`、`42-83`、`85-104`

axon 分区：

- `axon_ais`: axon index `0`
- `axon_regular`: 其他 axon

这些规则集中在 `parameters.py` 的 `dend_region_name()` 和 `axon_region_name()`，BrainCell 和 NEURON 都复用同一套逻辑。

## CV and nseg

GoC 使用 `goc20_nseg_rule()`：

```python
1 + 2 * int(length_um / max_len_um)
```

默认 `max_len_um` 是 `GoCCableParameters.cv_max_len_um`，当前为 `40.0` um。

BrainCell 侧通过 `CVPerBranchList` 对每条 branch 指定 CV 数；NEURON 侧对每个 section 指定 `nseg`。两边都使用同一个规则，避免只用 `MaxCVLen` 时出现“刚好对齐但数量关系不一致”的情况。

## Class construction

正式类保持最小构建接口：

```python
params = load_goc20_params()
cell = GoC(params=params).build()
```

BrainCell `GoC`:

- `self.morph_path`
- `self.params`
- `self.temperature_celsius`
- `self.v_init_mV`
- `self.morph`
- `self.cell`
- `self.regions`

NEURON `GoC`:

- `self.morph_path`
- `self.params`
- `self.nrnmech_path`
- `self.sections`
- `self.soma_sections`
- `self.dend_sections`
- `self.axon_sections`
- `self.root_soma`

正式模板不带 channel toggle。默认按下表完整插入源 GoC 机制；需要逐通道 debug 时放在 `debug/`。

## Build sequence

BrainCell:

1. `Morphology.from_asc()`
2. 按 branch length 和 `goc20_nseg_rule()` 构建 `CVPerBranchList`
3. 创建 `Cell`
4. 定义 region
5. paint cable
6. paint ions
7. paint channels

NEURON:

1. `h.load_file()` 加载 support hoc
2. `h.nrn_load_dll()` 加载 compiled mod library
3. `Import3d_Neurolucida3` 导入 ASC
4. 按 section prefix 拆分 soma/dend/axon
5. 配置 cable
6. 插入 leak、ions、channels

方法名使用 `_define_regions()`、`_paint_cable()`、`_paint_ions()`、`_paint_channels()` 这种内部步骤即可。Python 的单下划线只是约定“类内部流程使用”，不影响调用能力；正式外部接口保持 `build()` 和必要的 `cleanup()` 更清楚。

## Mechanism table

| GoC | Soma | Dend_apical | Dend_basal | Axon_AIS | Axon_regular |
| --- | --- | --- | --- | --- | --- |
| Leak | yes | yes | yes | yes | yes |
| HCN1 |  |  |  | yes |  |
| HCN2 |  |  |  | yes |  |
| Nav1.6 | yes | yes | yes | yes | yes |
| Kv1.1 | yes |  |  |  |  |
| Kv3.4 | yes |  |  |  | yes |
| Kv4.3 | yes |  |  |  |  |
| KM |  |  |  | yes |  |
| Kca1.1 | yes | yes | yes | yes |  |
| Kca2.2 |  | yes | yes |  |  |
| Kca3.1 | yes |  |  |  |  |
| CaHVA | yes |  | yes | yes |  |
| Cav2.3 |  | yes |  |  |  |
| Cav3.1 | yes | yes |  |  |  |
| CdpStC | yes | yes | yes | yes | yes |
| Na_ion | yes | yes | yes | yes | yes |
| K_ion | yes | yes | yes | yes | yes |
| Ca_ion | yes | yes | yes | yes | yes |
