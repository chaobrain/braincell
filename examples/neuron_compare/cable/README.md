# cable

`cable` 用来比较同一份 morphology 在 `NEURON` 和 `braincell` 两侧的纯 cable 电压响应。

- 官方输入面是 `configs/*.json + templates/*.json`。
- 一个 config 表示一个 cell，并通过 `defaults.morphology.path` 固定该 cell 的 morphology。
- template 表示不同刺激方案，同时携带该测试的 `simulation / cable / cv_policy / stimulus` 设置，以及刺激相关 sweep。
- `engine/` 是实现代码，`templates/` 是扫描模板 JSON，`workflows/` 是 notebook 与 `workflow_api.py`。
- 默认结果目录是 `results/sweeps/`、`results/config_runs/`、`results/batch_runs/`。
- 推荐 notebook 入口：
  - `workflows/workflow.ipynb`
  - `workflows/batch_workflow.ipynb`
- `cases/` 只保留单 case 样例和 ASC fixture。
