# PC Cell General

这个目录现在只保留一个通用版 Purkinje cell 对照入口。

相关资源位置：

- `.mod` 机制文件和形态学文件统一放在 `../Cerebellum_mod` 下。
- PC 的机制文件在 `../Cerebellum_mod/PC`。
- PC 的形态学文件在 `../Cerebellum_mod/PC/morphology/PC.asc`。

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
