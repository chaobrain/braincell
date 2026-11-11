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
NEURON 电流归一，`e_I`）与 `gates.*`（`e_s`）。这两个模板不写进配置的
`templates` 列表（列入时 `load_model_config` 抛 `ValueError`，否则 sweep 工作流会把它
当作电流钳运行），由测试显式运行：`tests/test_voltage_clamp.py` 跑每种模板的代表配置，
`tests/test_catalogue_*.py`（`slow`，共用 `tests/_catalogue.py`）按家族跑全部配置与
`tolerances.VARIANTS`（默认关闭的可选路径，两侧同改 `channel_params`）的五个协议：

```python
compare_protocol("configs/ma20_goc/kv3p4_ma20_goc.json", "templates/vc_steps.json")
```

全部限值来自容差表 [`tolerances.py`](tolerances.py)：表中没有的配置与协议为
T0（`e_V ≤ 1e-6 mV`；`e_I`、`e_s ≤ 1e-10`）；每个 T1 项写明偏差类别与
[已知差异](../known-differences.md)中的小节。`tests/test_neuron_agreement.py`
（电流钳）与 `tests/test_voltage_clamp.py`（电压钳）都从这张表读限值。

注册表中的每个通道要么在 `configs/` 下有配置，要么列在 `tolerances.py` 的
`UNCOVERED` 中并写明原因；`tests/test_coverage.py` 检查这一点，新增通道两者都没有
时失败；它同时检查每个用例的电导（`g_max` 或 `perm`）大于 0。`configs/testing/kv_testing.json` 对照 `data/mechanisms/testing/kv.mod`
（先运行 `python -m validation.neuron._mechanisms testing --kind channel`），也是
转换器示例 `examples/convert_mod/mod_validate/channel_validate.ipynb` 运行的配置。
`configs/testing/cl_leak_testing.json` 对照同一库中的 `clleak.mod`：没有门变量
（`gate_names` 为空列表，电压钳只比较 `e_I`）；`clleak` 只读 `ecl`，NEURON 侧由
runner 对 `cl_ion` 设 `ion_style`，使 `ecl` 按 `ion_state` 的浓度取 Nernst 值。
`configs/builtin/` 以 NEURON 自带机制为参照（任一已编译的库都含有）：`hh` 只留一种
电导（其余设为 0，runner 对带 `TABLE` 的机制设 `usetable_<mech> = 0`），对照
`K_HH1952` 与 `Na_HH1952`（braincell 侧 `temp_ref = 6.3 °C`）；`pas` 对照 `IL`。
`.mod` 中的 `GLOBAL` 参数（如 Nav1.1 的 `gateCurrent`）由 runner 写到
`h.<参数>_<机制>`，在同一进程内对之后的模型保持有效，所以改变它的配置在 `defaults`
中显式给出默认值。

当前推荐的 `MA24_PC` batch 配置目录为：

- `configs/ma24_pc/`

## 收敛与刚性

`validation/neuron/_convergence.py` 不用 NEURON，测 Cell 的积分路径在不同 `dt`
下的误差，结果与限值在 [`convergence.py`](convergence.py)（与 `tolerances.py`
互不读取）：

- 协议：`vc_steps`、`vc_ramp` 的 8 条（或 1 条）命令作为 Cell 的 8 个
  population 一次回放；第 n 步写入 V_{n+1}，阶跃取该步区间上的命令值，斜坡取
  V(t_{n+1})。松弛协议把通道的独立状态全部置 0、保持 −80 mV 35 ms、只比较
  `e_s`，用于电压钳下状态不变的通道（只受钙门控）。
- 参照：阶跃与松弛取 `dt = 0.025 ms` 的 `exp_euler`（电压在步内为常数时精确；
  HH 门在回放中换用 `exp_euler`，Markov 用 Cell 的 `subsolver`）；`dt` 取
  0.1、0.05、0.025、0.0125、0.00625 ms，每 0.1 ms 比较一次。斜坡不设参照，
  用 0.05、0.025、0.0125 ms 三个解的相邻差与 Richardson 阶数；在这三个 `dt`
  下仍未进入渐近区的配置在表项中写更细的三个 `dt`（`ramp_dts_ms`，现在只有
  `builtin/na_hh1952_hh`）。无状态通道（`IL`、`Cl_Leak`）不换门积分器，
  只比较 `e_I`。
- 分类：所有 `dt` 的 `e_I`、`e_s` ≤ 1e-10 为"精确"；否则最小三个 `dt`
  的拟合阶数须在 [0.8, 1.2]，0.025 ms 下 ≤ 1e-2 为"收敛"，超过为"收敛但误差
  超过指引线"；出现非有限值或阶数在窗口外为"不收敛"（需修正，不写限值）。
  斜坡阶数窗口：`e_s` [0.8, 1.2]，`e_I` [0.7, 1.3]。
- 刚性：`rho_max` 为 −100…60 mV（0.25 mV 步长）、36 °C 下 Markov 转移矩阵
  的谱半径或 HH 门的 α + β（含温度倍率），`rho_max · 0.025 > 2` 为刚性（显式
  solver 在该 `dt` 下不可用）。奇点扫描在 0.001 mV 网格上检查门值、门导数与
  电流都有限，已知的可去奇点列在 `KNOWN_SINGULARITIES`（现在为空）。

```python
from validation.neuron import _convergence as cv

report = cv.check_config("ma20_goc/nav1p6_ma20_goc")  # 或 subsolver="exp_euler", substeps=1
report.classification, report.errors[0.025], report.orders
cv.stiffness(cv.channel_class(cv.load_case("ma20_goc/nav1p6_ma20_goc", "vc_steps")[0])).rho_max
```

限值取实测的 1.5 倍，上取整到 1、2 或 5 乘 10 的幂（`cv.rounded_limit`）。
表的覆盖检查（`COVERAGE_SCOPE` 为 `None`）要求 registry 的每个通道类与
`configs/` 的每个配置都有表项；三个家族的类与配置另写成 `*_FAMILY_*` 元组，由各
家族的 slow 模块对照通道模块核对。`validation/neuron/_convergence_test.py` 在每个
PR 运行，含覆盖检查、全部类的刚性分类、对 registry 全部通道类的奇点扫描，以及
`testing/cl_leak_testing` 的收敛。按家族的收敛测试标 `slow`：
`tests/test_convergence_calcium.py`（本机约 1.7 分钟）、
`tests/test_convergence_potassium.py`（本机约 40 秒）、
`tests/test_convergence_sodium_hcn_leak.py`（本机约 1.5 分钟）。标 `slow` 的模块（按家族的
收敛测试与 `tests/test_catalogue_*.py`）只在每日 CI 的 `slow_tests_neuron` job 中
运行（`pytest tests -m slow`），PR 的 channel job 用 `-m "not slow"` 跳过它们。
