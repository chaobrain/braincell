# cable Workflow Doc

## Summary

`cable` 用来比较多房室 morphology 在 `NEURON` 和 `braincell` 两侧的 cable-only 数值一致性。

这里固定不插入 `ion`、`channel`、`leak`，只比较：

- 同一份 morphology 导入结果
- 同一组 `Ra / cm / dt / duration / v_init`
- 同一个 root `soma(0.5)` 刺激
- 多 CV 中点电压

## Current Layout

- `configs/*.json`: 主配置。一个 config 表示一个 cell，在 `defaults.morphology` 里固定 morphology，并声明模板列表。
- `templates/*.json`: 扫描模板。每个模板只描述一个刺激方案的 `base + group`。
- `engine/`: 实现目录，含 `case_schema.py`、`experiment_schema.py`、`braincell_runner.py`、`neuron_runner.py`、`compare.py`、`metrics.py`、`run.py`、`mapping.py`、`morphology_io.py`、`outputs.py`、`stimulus.py`。
- `workflows/workflow_api.py`: notebook-facing 调用层。
- `tests/`: 当前实现专属测试。
- `workflows/workflow.ipynb`: 单个 config 工作流。
- `workflows/batch_workflow.ipynb`: 批量 many-config 工作流。
- `results/`: 运行结果输出目录；当前 notebook 默认写到 `results/config_runs/` 和 `results/batch_runs/`。

## Config Contract

官方输入面是 “main config + templates”。

主配置示例：

```json
{
  "meta": {
    "label": "Inferior olive cable sweeps"
  },
  "defaults": {
    "morphology": {
      "path": "../../../multi_compartment/morpho_files/Cerebellum_morph/IO.swc"
    }
  },
  "templates": [
    "../templates/ac.json",
    "../templates/dc.json",
    "../templates/vinit.json",
    "../templates/cv.json"
  ]
}
```

当前仓库内置的 config 对应 `Cerebellum_morph/` 下这 6 个 morphology：

- `BC.json`
- `GoC.json`
- `GrC.json`
- `IO.json`
- `PC.json`
- `SC.json`

模板示例：

```json
{
  "meta": {
    "label": "AC current stimulus"
  },
  "base": {
    "simulation": {
      "dt_ms": 0.025,
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
      "kind": "sine",
      "target": "root_soma_midpoint",
      "start_ms": 0.0,
      "duration_ms": 2.0,
      "amplitude_nA": 0.05,
      "frequency_hz": 100.0,
      "phase_rad": 0.0,
      "offset_nA": 0.0
    }
  },
  "group": {
    "group_id": "ac",
    "sweep_axes": {
      "stimulus.amplitude_nA": [0.02, 0.05, 0.1],
      "stimulus.frequency_hz": [25.0, 100.0, 250.0]
    }
  },
  "outputs": {
    "plot": false
  }
}
```

## Workflow

典型运行流：

1. `workflows/workflow_api.py` 读取一个主配置。
2. `config.defaults.morphology` 固定 cell morphology。
3. config 主入口顺序运行该 config 声明的全部模板。
4. 每个 template 提供该测试的 `simulation / cable / cv_policy / stimulus`。
5. `template.group.sweep_axes` 只负责测试参数扫描。
6. config 与 template 合成为内部 sweep config。
7. runner 分别运行 `NEURON` 和 `braincell`。
8. compare 层做 tree mapping、CV 对齐和指标计算。
9. config 根目录输出模板级子目录和 config 级 summary。

默认 config 级输出目录为 `results/config_runs/<config_stem>/`。
每个模板的 sweep 结果位于 `results/config_runs/<config_stem>/templates/<template_stem>/`。

Notebook 入口：

- 单 config run： [workflow.ipynb](/home/swl/braincell/examples/neuron_compare/cable/workflows/workflow.ipynb)
- 批量 many-config run： [batch_workflow.ipynb](/home/swl/braincell/examples/neuron_compare/cable/workflows/batch_workflow.ipynb)
