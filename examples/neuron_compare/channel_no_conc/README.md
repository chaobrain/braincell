# channel_no_conc

单 `soma`、单 channel、无浓度语义的专用对比 harness。

主说明入口：

- 配置编写总说明： [configs/README.md](/home/swl/braincell/examples/neuron_compare/channel_no_conc/configs/README.md)
- workflow / notebook / 输出说明： [workflows/README.md](/home/swl/braincell/examples/neuron_compare/channel_no_conc/workflows/README.md)

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

- `templates/vinit_celsius.json`
- `templates/dc.json`
- `templates/ac.json`

当前推荐的 `MA24_PC` batch 配置目录为：

- `configs/ma24_pc/`
