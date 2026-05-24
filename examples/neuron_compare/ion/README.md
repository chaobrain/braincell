# ion

最小化的 ion-vs-NEURON 对照目录。

当前目录现在包含这些最小 notebook：

- `cdpcam_ma2024_pc_zero_ica.ipynb`
- `cdphva_su2015_dcn_zero_ica.ipynb`
- `cdpstc_ma2020_goc_zero_ica.ipynb`
- `cdpstc_ma2020_goc_with_cahva.ipynb`
- `cdpstc_ma2020_goc_with_cav2p1_ri21_sc.ipynb`
- `cdpstc_ma2020_goc_with_cav3p1.ipynb`
- `toy_ca_binding_kinetic_su2015_dcn.ipynb`
- `toy_ca_binding_source_kinetic_su2015_dcn.ipynb`
- `toy_ca_pump_factor_with_cahva_su2015_dcn.ipynb`
- `toy_diam_factor_kinetic_su2015_dcn.ipynb`

它们分别覆盖：

- `CdpCAM_MA24_PC` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 PC calcium-pool / CB / CAM 稳态回归
- `CdpHVA_SU15_DCN` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 `cai/Ci` 回归
- `CdpStC_MA20_GoC` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 GoC calcium-pool 回归
- `CdpStC_MA20_GoC + CaHVA_MA20_GoC` 的 NEURON-vs-BrainCell 耦合轨迹
- `CdpStC_MA20_GoC + Cav2p1_RI21_SC` 的 NEURON-vs-BrainCell 跨来源耦合轨迹
- `CdpStC_MA20_GoC + Cav3p1_MA20_GoC` 的 NEURON-vs-BrainCell 耦合轨迹
- 若干 DCN toy kinetic ion，用于隔离验证 binding/source/pump/factor/geometry 行为

当前刻意不做：

- 通用 JSON config / batch workflow
- 多 mechanism 批量对照
- 更大范围的参数扫描或自动拟合
- 与 `channel_no_conc` 同级别的大型 compare harness

这一目录的目标只是给 imported ion template 建一个最小、可解释、可继续扩展的起点。

补充说明：

- 当前本地 NEURON 环境里，`CaLVA/CdpLVA` 的 DCN `x86_64` 机制如果需要重编，先按仓库根目录 `agent.md` 的说明，把 `CPP/CC/CXX` 切到 `/usr/bin` 下的系统编译器再编。
- GoC 的耦合 notebook 默认都从 `examples/neuron_compare/Cerebellum_mod/GoC/x86_64` 加载机制。
- 如果当前 `GoC/x86_64` 里没有你要的 channel 组合，先在 `examples/neuron_compare/Cerebellum_mod/GoC` 下重编；如果当前 Python / Jupyter 进程已经加载过旧的 `libnrnmech.so`，需要先重启 kernel 再运行 notebook。
- `cdpstc_ma2020_goc_with_cahva.ipynb` 的最小编译命令：

```bash
CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ \
nrnivmodl ion/CdpStC_MA20_GoC.mod channel/CaHVA_MA20_GoC.mod
```

- `cdpstc_ma2020_goc_with_cav2p1_ri21_sc.ipynb` 混合了 GoC 的 ion 和 SC 的 channel，建议在临时目录下编译：

```bash
tmpdir=$(mktemp -d)
cd "$tmpdir"
CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ \
nrnivmodl \
  /home/swl/braincell-ion_dyn/examples/neuron_compare/Cerebellum_mod/GoC/ion/CdpStC_MA20_GoC.mod \
  /home/swl/braincell-ion_dyn/examples/neuron_compare/Cerebellum_mod/SC/channel/Cav2p1_RI21_SC.mod
```

- `cdpstc_ma2020_goc_with_cav3p1.ipynb` 的最小编译命令：

```bash
CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ \
nrnivmodl ion/CdpStC_MA20_GoC.mod channel/Cav3p1_MA20_GoC.mod
```
