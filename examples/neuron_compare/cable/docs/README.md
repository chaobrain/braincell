# cable Doc

## Summary

`cable` 用来比较多房室 morphology 在 `NEURON` 和 `braincell` 两侧的 cable-only 数值一致性。

这里固定不插入 `ion`、`channel`、`leak`，只比较：

- 同一份 morphology 导入结果
- 同一组 `Ra / cm / dt / duration / v_init`
- 同一个 root `soma(0.5)` 刺激
- 所有离散 compartment 的 `voltage`

## Current Layout

- `configs/*.json`: 主配置。声明模板列表。
- `scan_templates/*.json`: 扫描模板。每个模板只描述一个 `base + group`。
- `cases/*.json`: 单 case 样例。
- `templates/`: 扁平实现目录，含 `case_schema.py`、`experiment_schema.py`、`braincell_runner.py`、`neuron_runner.py`、`compare.py`、`run.py`、`mapping.py`、`morphology_io.py`、`outputs.py`、`stimulus.py`。
- `workflow_api.py`: notebook-facing 调用层。
- `tests/`: 当前实现专属测试。
- `docs/workflow.ipynb`: 单个 `config x template` 工作流。
- `docs/batch_workflow.ipynb`: 批量 `config x template` 工作流。
- `artifacts/`: 运行结果输出目录；默认写到 `artifacts/sweeps/` 和 `artifacts/batch_runs/`。

## Config Contract

官方 sweep 输入面只有 “main config + scan template”。

主配置示例：

```json
{
  "meta": {
    "label": "Cable example sweeps"
  },
  "templates": [
    "../scan_templates/io_dc_smoke_v1.json",
    "../scan_templates/dc_morphology_cv_scan_v1.json",
    "../scan_templates/sine_frequency_scan_v1.json"
  ]
}
```

扫描模板示例：

```json
{
  "meta": {
    "label": "DC morphology x CV scan"
  },
  "base": {
    "morphology": {
      "kind": "swc",
      "path": "/home/swl/braincell/examples/multi_compartment/morpho_files/unbranched_soma.swc"
    },
    "simulation": {
      "dt_ms": 0.1,
      "duration_ms": 2.0,
      "v_init_mV": -65.0
    },
    "cable": {
      "ra_ohm_cm": 100.0,
      "cm_uF_cm2": 1.0
    },
    "cv_policy": {
      "kind": "CVPerBranch",
      "cv_per_branch": 3
    },
    "stimulus": {
      "kind": "dc_step",
      "target": "root_soma_midpoint",
      "delay_ms": 0.5,
      "dur_ms": 1.0,
      "amp_nA": 0.05
    }
  },
  "group": {
    "group_id": "dc_morphology_cv",
    "sweep_axes": {
      "morphology.path": [
        "/home/swl/braincell/examples/multi_compartment/morpho_files/unbranched_soma.swc",
        "/home/swl/braincell/examples/multi_compartment/morpho_files/branched_dend.swc"
      ],
      "cv_policy.cv_per_branch": [1, 3]
    }
  },
  "outputs": {
    "plot": false
  }
}
```

约束：

- `templates` 必须是相对主配置目录的 JSON 路径列表。
- 一个模板只表达一个 `base + group`。
- `sweep_axes` 使用 dotted path。
- `cv_policy.cv_per_branch` 仍然要求正奇数。
- `morphology.kind` 当前正式支持 `swc` 和 `asc`；`neuroml2` 保留但运行时未实现。
- 旧的单文件 `config_id + case_groups[]` schema 不再作为官方输入。

## Workflow

典型运行流：

1. `workflow_api.py` 读取一个主配置和一个模板。
2. 两者合成为内部 sweep config。
3. runner 分别运行 `NEURON` 和 `braincell`。
4. compare 层做 tree mapping、compartment 对齐和指标计算。
5. 输出 `normalized_config.json`、`expanded_cases.json`、`case_results/*.json`、`case_metrics.csv`、`aggregate.json`。

默认输出目录名为 `<config_stem>__<template_stem>`，位于 `artifacts/sweeps/` 下。

当前单跑 notebook 默认参数直接指向 `configs/cable_demo.json + scan_templates/io_dc_smoke_v1.json`，方便先对 `io.swc` 做一个稳定的 DC smoke 检查，再切到其他模板。

Notebook 入口：

- 单 run： [workflow.ipynb](/home/swl/braincell/examples/neuron_compare/cable/docs/workflow.ipynb)
- 批量 run： [batch_workflow.ipynb](/home/swl/braincell/examples/neuron_compare/cable/docs/batch_workflow.ipynb)

## Notes

- 这里不做 morphology metric gatekeeping。
- 正式观测量固定是 all-compartment `voltage`。
- `mapping` 仍然使用 branch/section 结构对齐，不靠字符串名字直接配对。
