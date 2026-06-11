# GrC formal templates

这个目录保存 Granule Cell 2020 的正式 `NEURON` / `braincell` 单细胞对照模板。

GrC 当前有两套正式入口：

- ASC-only `GrC`: 只导入 ASC 中天然存在的 `soma + dend`。
- Full manual morphology `GrCFull`: 在 ASC `soma + dend` 基础上手工补 `hilock`、`ais`、`aa_*` 和两条 PF 链。

`debug/` 目录保留 toggles、summary、table、probe helper 和逐通道排查入口；正式模板不依赖 debug 文件。

## File layout

ASC-only 正式文件：

- `parameters.py`: ASC-only 默认路径、共享常量、区域参数和 nseg/CV 规则。
- `grc_neuron.py`: `NEURON` 侧 ASC-only `GrC` 组装器。
- `grc_braincell.py`: `braincell` 侧 ASC-only `GrC` 组装器。
- `run.ipynb`: ASC-only 最小运行入口。

Full morphology 正式文件：

- `grc_full_parameters.py`: full 形态区域参数和手工 branch 数量常量。
- `grc_full_neuron.py`: `NEURON` 侧 `GrCFull` 组装器。
- `grc_full_braincell.py`: `braincell` 侧 `GrCFull` 组装器。
- `run_full.ipynb`: full morphology 最小运行入口。

## Morphology and regions

形态默认使用 repo 内：

```text
examples/neuron_compare/Cerebellum_mod/GrC/morphology/GrC.asc
```

机制库默认使用：

```text
examples/neuron_compare/Cerebellum_mod/GrC/x86_64/.libs/libnrnmech.so
```

ASC-only 期望导入后有：

- `soma`: 1
- `dendrite`: 4
- `axon`: 0

Full morphology 期望：

- `soma`: 1
- `dend`: 4
- `hilock`: 1
- `ais`: 1
- `aa`: 4
- `pf1`: 142
- `pf2`: 142

Full morphology 仍不包含源脚本中注释掉的 synapse / Syntype 部分。

## Parameter and build rules

ASC-only `GrCParameters` 只有两个区域：

- `soma`
- `dend`

Full `GrCFullParameters` 有六个正式区域：

- `soma`
- `dend`
- `hilock`
- `ais`
- `aa`
- `pf`

两套模板都使用同一个 nseg/CV 规则：

```python
1 + 2 * int(length_um / CV_MAX_LEN_UM)
```

BrainCell 侧通过 `CVPerBranchList` 指定每条 branch 的 CV 数；NEURON 侧设置 section `nseg`。

## ASC-only mechanism table

| GrC ASC-only | Soma | Dend |
| --- | --- | --- |
| Leak | yes | yes |
| Kv3.4 | yes |  |
| Kv4.3 | yes |  |
| Kir2.3 | yes |  |
| CaHVA | yes | yes |
| Kv1.1 | yes | yes |
| Kv1.5 | yes |  |
| Kv2.2 | yes |  |
| Kca1.1 |  | yes |
| CdpCR | yes | yes |
| Na_ion active use |  |  |
| K_ion active use | yes | yes |
| Ca_ion active use | yes | yes |

## Full morphology mechanism table

| GrC full | Soma | Dend | Hilock | AIS | AA | PF |
| --- | --- | --- | --- | --- | --- | --- |
| Leak | yes | yes | yes | yes | yes | yes |
| Nav |  |  |  |  | yes | yes |
| NaFHF |  |  | yes | yes |  |  |
| Kv3.4 | yes |  | yes | yes | yes | yes |
| Kv4.3 | yes |  |  |  |  |  |
| Kir2.3 | yes |  |  |  |  |  |
| CaHVA | yes | yes | yes | yes | yes | yes |
| Kv1.1 | yes | yes |  |  |  |  |
| Kv1.5 | yes |  |  |  |  |  |
| Kv2.2 | yes |  |  |  |  |  |
| Kca1.1 |  | yes |  |  |  |  |
| KM |  |  |  | yes |  |  |
| CdpCR | yes | yes | yes | yes | yes | yes |
| Na_ion active use |  |  | yes | yes | yes | yes |
| K_ion active use | yes | yes | yes | yes | yes | yes |
| Ca_ion active use | yes | yes | yes | yes | yes | yes |

## Notebook entries

- `run.ipynb`: ASC-only 对照入口。
- `run_full.ipynb`: full morphology 对照入口。

两个 notebook 都只做正式最小对照：构建两边 cell、挂 soma current clamp、记录 soma voltage 并画图。逐通道开关和全 compartment probe 继续放在 `debug/`。
