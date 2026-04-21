# channel_no_conc Doc

## Summary

`channel_no_conc` 用来比较单 `soma` 上一个具体 channel 在 `NEURON` 和 `braincell` 两侧的数值一致性。

这里不区分 HH / Markov。只要是“单通道、无浓度依赖、单 `soma`、只看 `v / ix / gates`”的比较，都走这一套 harness。

## Current Layout

- `configs/**/*.json`: 主配置。声明 `mod_dir`、channel 映射和模板列表。
- `configs/ma24_pc/*.json`: 当前推荐的 `MA24_PC` batch 配置集合。
- `templates/*.json`: 扫描模板。每个模板只描述一个 `base + group`。
- `engine/`: 扁平实现目录，含 `mapping_schema.py`、`experiment_schema.py`、`neuron_runner.py`、`braincell_runner.py`、`compare.py`、`run.py`、`metrics.py`、`outputs.py`、`stimulus.py`。
- `workflows/workflow_api.py`: notebook-facing 调用层。
- `tests/`: 当前实现专属测试。
- `workflows/workflow.ipynb`: 单个 `config x template` 工作流。
- `workflows/batch_workflow.ipynb`: 批量 many-config 工作流。
- `results/`: 运行结果输出目录；默认不再入库。

## Authoring

`config` / `template` 的完整写法、`owner` 的用法、`defaults` 与 `base` 的合并顺序，都统一写在：

- [configs/README.md](/home/swl/braincell/examples/neuron_compare/channel_no_conc/configs/README.md)

这里不再重复展开 schema 细节，避免出现两份不一致的说明。

## Workflow

典型运行流：

1. `workflows/workflow_api.py` 或 `engine/run.py` 读取一个主配置。
2. config 主入口会顺序运行该 config 声明的全部模板。
3. 每个模板内部仍会先合成为一个 sweep config。
4. runner 分别运行 `NEURON` 和 `braincell`。
5. compare 层做时间轴修正、gate 对齐和指标计算。
6. config 级目录下输出模板级子目录和 config 级 summary。

默认 config 级输出目录为 `results/config_runs/<config_stem>/`。
每个模板的 sweep 结果位于 `results/config_runs/<config_stem>/templates/<template_stem>/`。

低层调试入口仍支持显式传入一个 `config + template`，这时行为与原来的单 sweep 运行保持一致。

Notebook 入口：

- 单 config run： [workflow.ipynb](/home/swl/braincell/examples/neuron_compare/channel_no_conc/workflows/workflow.ipynb)
- 批量 many-config run： [batch_workflow.ipynb](/home/swl/braincell/examples/neuron_compare/channel_no_conc/workflows/batch_workflow.ipynb)
- 推荐批量目录：`CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc"`

说明：

- notebook 参数区承担“配置注释”的角色；`configs/**/*.json` 与 `templates/*.json` 保持纯数据，不在 JSON 内写注释。
- 首次使用的 notebook 参数区只保留最核心的 `config_path`、`out_dir`、`plot_cases`；更细的模板选择与 case 选择在看到 `run_info` 和模板子目录后再做。
- 模板子目录里的 raw 数据是主结果面；辅助性的 `worst` 排行和临时 notebook 图都建立在这些 raw 数据之上，而不是反过来。

模板子目录原始产物约定：

- `normalized_config.json`
- `expanded_cases.json`
- `case_results/<case_id>.json`
- `case_metrics.csv`
- `aggregate.json`
- `plots/`
  - `<case_id>.png`: 单个 case 的所有观测变量对比图
  - `summary_mae.png`
  - `summary_rmse.png`
  - `summary_max_abs.png`
  - `summary_rel_mae_pct.png`
  - `observable_metric_boxplots.png`: 模板级四误差箱图

config 根目录原始产物约定：

- `config_manifest.json`
- `config_runs.csv`
- `observable_summary.csv`
- `observable_summary.json`
- `failures.csv`
- `all_templates_observable_summary.png`
- `boxplot_by_template.png`
- `boxplot_by_observable_family.png`

batch run 目录约定：

- `results/batch_runs/<YYMMDD_HHMMSS>/`
- `configs/<config_name>/...` 下放每个 config 的完整输出副本
- batch 根目录只放跨全部 config 的 summary 文件：
  - `batch_manifest.json`
  - `config_runs.csv`
  - `batch_observable_summary.csv`
  - `batch_observable_summary.json`
  - `batch_failures.csv`

## Notes

- 这个 harness 只接受单 channel、单 `soma`、无浓度语义的比较输入。
- 结果面固定只看 `voltage`、`current.ix`、`gates.*`。
