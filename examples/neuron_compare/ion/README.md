# ion

最小化的 ion-vs-NEURON 对照目录。

当前目录现在包含 4 个最小 notebook：

- `cdphva_su2015_dcn_zero_ica.ipynb`
- `cdphva_su2015_dcn_with_cahva.ipynb`
- `cdplva_su2015_dcn_zero_ical.ipynb`
- `cdplva_su2015_dcn_with_calva.ipynb`

它们分别覆盖：

- `CdpHVA_SU15_DCN` 在 **不插入 Ca channel**、因此 `ica = 0` 时的 `cai/Ci` 回归
- `CdpHVA_SU15_DCN + CaHVA_SU15_DCN` 的 NEURON-vs-BrainCell 耦合轨迹
- `CdpLVA_SU15_DCN` 在 **不插入 CaLVA channel**、因此 `ical = 0` 时的 `cali/Ci` 回归
- `CdpLVA_SU15_DCN + CaLVA_SU15_DCN` 的 NEURON-vs-BrainCell 耦合轨迹

当前刻意不做：

- 通用 JSON config / batch workflow
- 多 mechanism 批量对照
- 更大范围的参数扫描或自动拟合
- 与 `channel_no_conc` 同级别的大型 compare harness

这一目录的目标只是给 imported ion template 建一个最小、可解释、可继续扩展的起点。

补充说明：

- 当前本地 NEURON 环境里，`CaLVA/CdpLVA` 的 DCN `x86_64` 机制如果需要重编，先按仓库根目录 `agent.md` 的说明，把 `CPP/CC/CXX` 切到 `/usr/bin` 下的系统编译器再编。
