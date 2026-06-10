# Cell comparison examples

这个目录保存单细胞 `NEURON` / `braincell` 对照入口。

相关资源位置：

- `.mod` 机制文件和形态学文件统一放在 `../Cerebellum_mod` 下。
- PC 的机制文件在 `../Cerebellum_mod/PC`。
- PC 的形态学文件在 `../Cerebellum_mod/PC/morphology/PC.asc`。
- GoC 的机制文件在 `../Cerebellum_mod/GoC`，顶层 `x86_64/.libs/libnrnmech.so` 是 cell 对照默认库。
- GoC 的形态学文件在 `../Cerebellum_mod/GoC/morphology/GoC.asc`。
- BC 的机制文件在 `../Cerebellum_mod/BC`，顶层 `x86_64/.libs/libnrnmech.so` 是 cell 对照默认库。
- BC 的形态学文件在 `../Cerebellum_mod/BC/morphology/BC.asc`。
- BC 旧的 `../Cerebellum_mod/BC/channel/x86_64` 只包含 channel 编译产物，不包含 `CdpStC_MA25_BC`，不要作为 cell 对照默认库。
- SC 的机制文件在 `../Cerebellum_mod/SC`，顶层 `x86_64/.libs/libnrnmech.so` 是 cell 对照默认库。
- SC 的形态学文件在 `../Cerebellum_mod/SC/morphology/SC.asc`。
- GrC 的机制文件在 `../Cerebellum_mod/GrC`，顶层 `x86_64/.libs/libnrnmech.so` 是 cell 对照默认库。
- GrC 的形态学文件在 `../Cerebellum_mod/GrC/morphology/GrC.asc`。
- DCN 的机制文件在 `../Cerebellum_mod/DCN`，顶层 `x86_64/.libs/libnrnmech.so` 是 cell 对照默认库。
- DCN 的形态当前使用源 HOC `/home/swl/Cerebellum_circuit/DCN/DCN/DCN_mor.hoc` native 解析路径，不依赖 SWC/CSV。

`pc_ma2024` 是当前正式 PC 对照文件夹：

- `pc_neuron.py`
  - 保留一份 `NEURON` 组装版本。
  - 这份版本去掉了原始脚本里手工搭建的 axon、myelin、node、spine 等额外结构。
  - 当前只直接导入 ASC/SWC 形态文件里天然存在的形态，也就是这里的 `soma + dendrite`。
- `pc_braincell.py`
  - 按 `pc_neuron.py` 的同语义重新写成 `braincell` 组装版本。
  - 目标是让两边的区域、参数、机制插入和离子语义尽量一一对应，方便对照转换。
- `run.ipynb`
  - 最小运行入口。
  - 同时导入 `pc_neuron.py` 和 `pc_braincell.py`，构建两边 cell，设置同一类刺激与观测，然后一起仿真比较。
- `debug/`
  - 调试版本。
  - 文件里保留逐步检查用的开关、summary、table、probe helper 等。
  - 用于定位某个 channel、ion、region 或调度差异；正式版本尽量保持简洁。

当前文件：

- `pc_ma2024/parameters.py`
  - 正式版共享常量、`PCParameters` 和 `R_01_final_pop.txt` 参数读取
- `pc_ma2024/pc_neuron.py`
  - `NEURON` 侧最小 `PC` 组装器
- `pc_ma2024/pc_braincell.py`
  - `braincell` 侧最小 `PC` 组装器
- `pc_ma2024/debug/pc_parameters.py`
  - 调试版共享开关、默认常量、`R_01_final_pop.txt` 参数读取
- `pc_ma2024/debug/pc_neuron_debug.py`
  - `NEURON` 侧带 summary/table/probe helper 的调试组装器
- `pc_ma2024/debug/pc_braincell_debug.py`
  - `braincell` 侧带 summary/table/probe helper 的调试组装器
- `pc_ma2024/run.ipynb`
  - 最小运行 notebook
- `pc_ma2024/debug/run_debug.ipynb`
  - 薄 notebook：导入两个 cell、设置开关、添加刺激与 probe、运行和分析
- `goc_ma2020/debug/`
  - GoC 调试版对照入口。
  - 参数来自 `/home/swl/Cerebellum_circuit/GoC/GoC/Optimization_result.txt`，形态默认使用 repo 内 `Cerebellum_mod/GoC/morphology/GoC.asc`。
- `bc_ma2025/debug/`
  - BC 调试版对照入口。
  - 组装语义来自 `/home/swl/Cerebellum_circuit/BC2025/basket.py`，形态默认使用 repo 内 `Cerebellum_mod/BC/morphology/BC.asc`。
  - 参数是 `basket.py` 中的硬编码区域参数，集中放在 `bc_parameters.py`。
- `bc_ma2025/parameters.py`
  - 正式版共享常量和 `BCParameters`
- `bc_ma2025/bc.md`
  - BC 正式模板说明和机制表
- `bc_ma2025/bc_neuron.py`
  - `NEURON` 侧全机制开启的最小 `BC` 组装器
- `bc_ma2025/bc_braincell.py`
  - `braincell` 侧全机制开启的最小 `BC` 组装器
- `bc_ma2025/run.ipynb`
  - 最小运行 notebook
- `sc_ma2021/debug/`
  - SC 调试版对照入口。
  - 组装语义来自 `/home/swl/Cerebellum_circuit/SC2021/stellate.py`，参数来自 `/home/swl/Cerebellum_circuit/SC2021/SC_param.py`。
  - 形态默认使用 repo 内 `Cerebellum_mod/SC/morphology/SC.asc`。
  - 区域划分为 `soma`、`dendprox`、`denddist`、`axon_ais`、`axon_regular`；`dendprox` 使用源 NEURON 导入后的 canonical dend index。
- `sc_ma2021/parameters.py`
  - 正式版共享常量和 `SCParameters`
- `sc_ma2021/sc.md`
  - SC 正式模板说明、区域规则和机制表
- `sc_ma2021/sc_neuron.py`
  - `NEURON` 侧全机制开启的最小 `SC` 组装器
- `sc_ma2021/sc_braincell.py`
  - `braincell` 侧全机制开启的最小 `SC` 组装器
- `sc_ma2021/run.ipynb`
  - 最小运行 notebook
- `grc_ma2020/debug/`
  - GrC ASC-only 调试版对照入口。
  - 组装参数来自 `/home/swl/Cerebellum_circuit/GrC/GrC/Cereb_GrC_regular.py` 中当前未注释的 soma/dend 配置。
  - 形态默认使用 repo 内 `Cerebellum_mod/GrC/morphology/GrC.asc`。
  - 当前只导入 ASC 内的 `soma + dend`，不包含源脚本手动创建的 `hilock`、`ais`、`aa_*`，也不包含已注释的 `pf_*` 和 synapse。
  - 同目录的 `run_full_debug.ipynb`、`grc_full_neuron_debug.py` 和 `grc_full_braincell_debug.py` 是完整手工形态对照入口：ASC `soma + dend` 加上 `hilock`、`ais`、`aa_*` 和两条 PF 链；仍不包含注释掉的 synapse/Syntype 部分。
- `grc_ma2020/parameters.py`
  - GrC ASC-only 正式版共享常量和 `GrCParameters`
- `grc_ma2020/grc.md`
  - GrC ASC-only 与 full morphology 正式模板说明和机制表
- `grc_ma2020/grc_neuron.py`
  - `NEURON` 侧 ASC-only 最小 `GrC` 组装器
- `grc_ma2020/grc_braincell.py`
  - `braincell` 侧 ASC-only 最小 `GrC` 组装器
- `grc_ma2020/run.ipynb`
  - GrC ASC-only 最小运行 notebook
- `grc_ma2020/grc_full_parameters.py`
  - GrC 完整手工形态正式版共享常量和 `GrCFullParameters`
- `grc_ma2020/grc_full_neuron.py`
  - `NEURON` 侧完整手工形态 `GrCFull` 组装器，包含 `hilock`、`ais`、`aa_*` 和两条 PF 链
- `grc_ma2020/grc_full_braincell.py`
  - `braincell` 侧完整手工形态 `GrCFull` 组装器
- `grc_ma2020/run_full.ipynb`
  - GrC 完整手工形态最小运行 notebook
- `dcn_su2015/debug/`
  - DCN 调试版对照入口。
  - NEURON 侧加载 `/home/swl/Cerebellum_circuit/DCN/DCN/DCN_mor.hoc` 得到源形态和 `SectionList`，但通道插入使用 repo 内 `../Cerebellum_mod/DCN/x86_64/.libs/libnrnmech.so` 的重命名机制。
  - BrainCell 侧复用 `Cerebellum_mod/DCN/morphology/dcn_native.py` 直接解析源 HOC 的 `create/connect/pt3dadd/SectionList`，分支名携带区域前缀。
  - `TNC` 是 ohmic current。BrainCell debug 中用 `IL` 机制、`E=-35 mV` 和 `TNC_*` 实例名复现；NEURON debug 中由于 repo 内没有独立 `TNC` suffix，将 leak 与 TNC 合并为等效 `pas`。
- `dcn_su2015/parameters.py`
  - DCN 正式版共享常量、区域计数和 `DcnTemplateParameters`
- `dcn_su2015/dcn.md`
  - DCN 正式模板说明、HOC native morphology 路径和机制表
- `dcn_su2015/dcn_neuron.py`
  - `NEURON` 侧最小 `DCN` 组装器，加载源 `DCN_mor.hoc` 并使用其 `SectionList` 区域
- `dcn_su2015/dcn_braincell.py`
  - `braincell` 侧最小 `DCN` 组装器，复用 `dcn_native.py` 解析同一份源 HOC 形态
- `dcn_su2015/run.ipynb`
  - DCN 最小运行 notebook

debug建议流程：

1. 在 `pc_ma2024/debug/run_debug.ipynb` 里设置 `PCToggles` 和仿真参数
2. `build()` 两边 cell
3. 查看 `summary()` 和 `branch_table()`
4. 用标准电压 helper 挂 `soma` 和全部 compartment midpoint 电压
5. 在 notebook 里手写额外的 `soma` 观测，例如 `cai`、`ica`、gate
6. 运行并比较

注意：

- 电压采样 helper 只覆盖：
  - `soma(0.5)` voltage
  - 全 compartment midpoint voltage
- 其他变量默认不封装 helper，保持 notebook 手写
- 时间轴对齐继续建议使用统一参考网格：
  - NEURON天然会有初始时间点，Braincell没有，例如duration = 10ms，dt =1ms，NEURON的总时间长度是11(多一个t=0)，Braincell是10(t=dt开始)
