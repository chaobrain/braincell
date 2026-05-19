# ion

最小化的 ion-vs-NEURON 对照目录。

当前目录现在包含这些最小 notebook：

- `cdphva_su2015_dcn_zero_ica.ipynb`
- `cdpstc_ma2020_goc_zero_ica.ipynb`
- `cdpstc_ma2020_goc_with_cahva.ipynb`
- `toy_ca_binding_kinetic_su2015_dcn.ipynb`
- `toy_ca_binding_source_kinetic_su2015_dcn.ipynb`
- `toy_ca_pump_factor_with_cahva_su2015_dcn.ipynb`
- `toy_diam_factor_kinetic_su2015_dcn.ipynb`

它们分别覆盖：

- `CdpHVA_SU15_DCN` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 `cai/Ci` 回归
- `CdpStC_MA20_GoC` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 GoC calcium-pool 回归
- `CdpStC_MA20_GoC + CaHVA_MA20_GoC` 的 NEURON-vs-BrainCell 耦合轨迹
- 若干 DCN toy kinetic ion，用于隔离验证 binding/source/pump/factor/geometry 行为

当前刻意不做：

- 通用 JSON config / batch workflow
- 多 mechanism 批量对照
- 更大范围的参数扫描或自动拟合
- 与 `channel_no_conc` 同级别的大型 compare harness

这一目录的目标只是给 imported ion template 建一个最小、可解释、可继续扩展的起点。

补充说明：

- 当前本地 NEURON 环境里，`CaLVA/CdpLVA` 的 DCN `x86_64` 机制如果需要重编，先按仓库根目录 `agent.md` 的说明，把 `CPP/CC/CXX` 切到 `/usr/bin` 下的系统编译器再编。
- GoC 的 `cdpstc_ma2020_goc_with_cahva.ipynb` 需要同一个 `x86_64` 里同时包含 ion 和 channel，可在 `examples/neuron_compare/Cerebellum_mod/GoC` 下运行：

```bash
CPP=/usr/bin/cpp CC=/usr/bin/cc CXX=/usr/bin/c++ \
nrnivmodl ion/CdpStC_MA20_GoC.mod channel/CaHVA_MA20_GoC.mod
```
