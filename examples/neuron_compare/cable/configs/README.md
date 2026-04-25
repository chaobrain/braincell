# cable Config Guide

这份文档是 `cable` 的唯一配置编写说明。

这里的层级分工是：

- `config`
  - 表示一个 cell
  - 只负责固定 morphology
  - 声明要跑哪些 `templates/*.json`
- `template`
  - 表示一个刺激方案
  - 在 `base` 里给出该测试的 `simulation / cable / cv_policy / stimulus`
  - 在 `group.sweep_axes` 里定义会扫的测试参数

当前 `cable` 与 `channel_no_conc` 的主要区别只在：

- cell 构建来自 morphology 导入，而不是单 `soma`
- 观测量是多 CV 电压，结果导出为 `voltage_midpoint_mean` 和 `voltage_sum`

一个最小 config：

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

说明：

- `morphology.kind` 可以省略
- 目前只支持 `.swc` 和 `.asc`
- kind 会按 `morphology.path` 的后缀自动推断
- 相对路径按当前 config 文件所在目录解析

一个最小 template：

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
  }
}
```

合成 case 时的顺序是：

1. `config.defaults`
2. `template.base`
3. `template.group.sweep_axes`

也就是说：

- cell 身份只写在 `config.defaults.morphology`
- 测试设置写在 `template.base`
- 真正要扫的参数写在 `template.group.sweep_axes`
