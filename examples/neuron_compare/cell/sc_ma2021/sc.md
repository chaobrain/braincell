# SC formal template

这个目录保存 Stellate Cell 2021 的正式 `NEURON` / `braincell` 单细胞对照模板。

正式模板文件：

- `parameters.py`: 默认路径、共享常量、区域参数、region 规则和 nseg/CV 规则。
- `sc_neuron.py`: `NEURON` 侧全机制开启的最小 `SC` 组装器。
- `sc_braincell.py`: `braincell` 侧全机制开启的最小 `SC` 组装器。
- `run.ipynb`: 正式最小运行入口。
- `debug/`: 调试入口，保留 toggles、summary、table、probe helper；正式模板不依赖 debug 文件。

## Morphology and regions

形态默认使用 repo 内：

```text
examples/neuron_compare/Cerebellum_mod/SC/morphology/SC.asc
```

机制库默认使用：

```text
examples/neuron_compare/Cerebellum_mod/SC/x86_64/.libs/libnrnmech.so
```

正式模板期望 ASC 导入后有：

- `soma`: 1
- `dendrite`: 104
- `axon`: 15

区域划分集中在 `parameters.py`：

- `soma`: 唯一 soma branch。
- `dendprox`: dendrite index in `(2, 3, 15, 16, 20, 31, 34, 35, 36, 50, 66, 67, 81, 103)`。
- `denddist`: 其他 dendrite。
- `axon_ais`: axon index `0`。
- `axon_regular`: 其他 axon。

BrainCell 和 NEURON 都复用同一套 `dend_region_name()` / `axon_region_name()` 规则，避免两边区域漂移。

## Parameter and build rules

`SCParameters` 按正式区域拆分：

- `soma`
- `dendprox`
- `denddist`
- `axon_ais`
- `axon_regular`

正式 BrainCell 侧在 `paint()` 调用处显式写单位；参数对象只保存源数值。`Cav2p1` 默认使用 frozen BrainCell 类 `Cav2p1_RI2021_SC_Frozen`，对应当前正式模板的稳定对照路径。需要切换 frozen/unfrozen 时使用 `debug/` 版本。

离子语义：

- Na 使用固定 reversal `NA_E_MV`。
- K 在 soma/dend 使用 `K_E_MV`，axon 使用 `K_E_AXON_MV`。
- Ca 使用 `CdpStC_RI2021_SC`，每个正式区域有独立 pump 参数。

## Mechanism table

| SC | Soma | Dendprox | Denddist | Axon_AIS | Axon_regular |
| --- | --- | --- | --- | --- | --- |
| Leak | yes | yes | yes | yes | yes |
| Nav1.1 | yes |  |  |  |  |
| Nav1.6 |  |  |  | yes | yes |
| Cav2.1 frozen | yes | yes | yes |  |  |
| Cav3.2 | yes | yes |  |  |  |
| Cav3.3 | yes | yes |  |  |  |
| Kir2.3 | yes |  |  |  |  |
| Kv1.1 | yes | yes | yes | yes | yes |
| Kv3.4 | yes |  |  | yes | yes |
| Kv4.3 | yes | yes |  |  |  |
| KM |  |  |  | yes |  |
| Kca1.1 | yes | yes | yes |  |  |
| Kca2.2 | yes | yes | yes |  |  |
| HCN1 | yes |  |  | yes | yes |
| CdpStC | yes | yes | yes | yes | yes |
| Na_ion active use | yes |  |  | yes | yes |
| K_ion active use | yes | yes | yes | yes | yes |
| Ca_ion active use | yes | yes | yes | yes | yes |

## Notebook entry

`run.ipynb` 使用正式 `SC` builders，同时设置 soma current clamp 和 soma voltage probe。时间轴对齐沿用 cell 总 README 中的约定：NEURON 记录包含 `t=0`，BrainCell `StateProbe` 第一个样本对齐到 `t=dt`。
