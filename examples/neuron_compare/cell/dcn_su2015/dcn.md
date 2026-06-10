# DCN formal template

这个目录保存 Deep Cerebellar Nucleus 2015 的正式 `NEURON` / `braincell` 单细胞对照模板。

DCN 和其他 ASC 模板不同：正式形态来源是源 HOC 文件，而不是 repo 内 ASC/SWC。

## File layout

- `parameters.py`: 默认路径、区域计数、温度缩放参数和 `DcnTemplateParameters`。
- `dcn_neuron.py`: `NEURON` 侧最小 `DCN` 组装器。
- `dcn_braincell.py`: `braincell` 侧最小 `DCN` 组装器。
- `run.ipynb`: 正式最小运行入口。
- `debug/`: 调试入口，保留 toggles、summary、table、probe helper 和逐通道排查。

## Morphology source

NEURON 侧加载源形态：

```text
/home/swl/Cerebellum_circuit/DCN/DCN/DCN_mor.hoc
```

BrainCell 侧复用 repo 内 native parser：

```text
examples/neuron_compare/Cerebellum_mod/DCN/morphology/dcn_native.py
```

`dcn_native.py` 解析同一份 HOC 的 `create`、`connect`、`pt3dadd` 和 `SectionList`，生成带 region 前缀的 BrainCell branch 名。这样两边都以源 HOC region 为准。

机制库默认使用：

```text
examples/neuron_compare/Cerebellum_mod/DCN/x86_64/.libs/libnrnmech.so
```

## Regions

正式模板期望源 HOC 区域计数：

| Region | Count | Branch type |
| --- | ---: | --- |
| soma | 1 | soma |
| axHillock | 1 | axon |
| axIniSeg | 10 | axon |
| axNode | 20 | axon |
| proxDend | 83 | dendrite |
| distDend | 402 | dendrite |

总 section / branch 数为 `517`。

## Parameter and ion rules

`DcnTemplateParameters` 保存源模板常量和温度缩放：

- channel gating 使用 `q10_channel_gating`。
- conductance/permeability 使用 `q10_conductances`。
- calcium concentration tau 使用 `q10_ca_conc`。

正式模板默认全机制开启，不暴露 toggle。需要 leak-only 或逐通道排查时使用 `debug/`。

`TNC` 是 ohmic current：

- BrainCell 侧用 `IL` class、`E=-35 mV` 和 `TNC_*` 实例名复现。
- NEURON 侧 repo 内没有独立 `TNC` suffix，因此把 leak 与 TNC 合并为等效 `pas`。

HVA 和 LVA calcium concentration pools 分别用：

- `CdpHVA_SU2015_DCN` / `ca_hva`
- `CdpLVA_SU2015_DCN` / `ca_lva`

BrainCell 侧按源 section 单独 paint calcium pools，因为 shell depth 是 section-specific。

## Mechanism table

| DCN | Soma | axHillock | axIniSeg | axNode | proxDend | distDend |
| --- | --- | --- | --- | --- | --- | --- |
| Leak | yes | yes | yes | yes | yes | yes |
| NaF | yes | yes | yes |  | yes |  |
| NaP | yes |  |  |  |  |  |
| fKdr | yes | yes | yes |  | yes |  |
| sKdr | yes | yes | yes |  | yes |  |
| SK | yes |  |  |  | yes | yes |
| HCN | yes |  |  |  | yes | yes |
| TNC | yes | yes | yes |  | yes |  |
| CaLVA | yes |  |  |  | yes | yes |
| CaHVA | yes |  |  |  | yes | yes |
| CdpHVA | yes |  |  |  | yes | yes |
| CdpLVA | yes |  |  |  | yes | yes |
| Na_ion active use | yes | yes | yes |  | yes |  |
| K_ion active use | yes | yes | yes |  | yes | yes |
| Ca_HVA ion active use | yes |  |  |  | yes | yes |
| Ca_LVA ion active use | yes |  |  |  | yes | yes |

## Notebook entry

`run.ipynb` 使用正式 `DCN` builders，设置 soma current clamp 和 soma voltage probe。时间轴对齐沿用 cell 总 README 中的约定：NEURON 记录包含 `t=0`，BrainCell `StateProbe` 第一个样本对齐到 `t=dt`。
