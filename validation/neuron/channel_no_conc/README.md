# channel_no_conc

单 `soma`、单 channel、无浓度语义的专用对比 harness。

主说明入口：

- 配置编写总说明： [configs/README.md](configs/README.md)
- workflow / notebook / 输出说明： [workflows/README.md](workflows/README.md)

当前正式输入与入口：

- `configs/**/*.json`
- `templates/*.json`
- `engine/run.py`
- `workflows/workflow_api.py`
- `workflows/workflow.ipynb`
- `workflows/batch_workflow.ipynb`

这个目录只负责：

- 根据一个 `config` 运行其声明的全部 `template`
- 也支持显式指定单个 `config + template` 作为低层调试入口
- 输出 `v / ix / gates` 结果和汇总产物
- 提供 notebook 和 batch summary 的分析入口

当前内置的通用模板为：

- 电流钳：`templates/vinit_celsius.json`、`templates/dc.json`、`templates/ac.json`
- 电压钳：`templates/vc_steps.json`、`templates/vc_ramp.json`

电压钳模板由 `engine/voltage_clamp.py` 运行：NEURON 用 `SEClamp`（`rs = 1e-9 MΩ`）
钳住 soma，braincell 用同一个 Cell、以回放 NEURON 电压的 solver 只推进通道状态
（共享模块 `validation/neuron/_clamp.py`）。观测量为电流 `ix`（按协议内最大
NEURON 电流归一，`e_I`）与 `gates.*`（`e_s`）。这两个模板暂不写进各配置的
`templates` 列表，只由 `tests/test_voltage_clamp.py` 对代表配置显式运行：

```python
compare_protocol("configs/ma20_goc/kv3p4_ma20_goc.json", "templates/vc_steps.json")
```

全部限值来自容差表 [`tolerances.py`](tolerances.py)：表中没有的配置与协议为
T0（`e_V ≤ 1e-6 mV`；`e_I`、`e_s ≤ 1e-10`）；每个 T1 项写明偏差类别与
[已知差异](../known-differences.md)中的小节。`tests/test_neuron_agreement.py`
（电流钳）与 `tests/test_voltage_clamp.py`（电压钳）都从这张表读限值。

注册表中的每个通道要么在 `configs/` 下有配置，要么列在 `tolerances.py` 的
`UNCOVERED` 中并写明原因；`tests/test_coverage.py` 检查这一点，新增通道两者都没有
时失败。`configs/testing/kv_testing.json` 对照 `data/mechanisms/testing/kv.mod`
（先运行 `python -m validation.neuron._mechanisms testing --kind channel`），也是
转换器示例 `examples/convert_mod/mod_validate/channel_validate.ipynb` 运行的配置。

当前推荐的 `MA24_PC` batch 配置目录为：

- `configs/ma24_pc/`
