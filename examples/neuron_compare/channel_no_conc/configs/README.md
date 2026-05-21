# channel_no_conc Config Guide

这份文档是 `channel_no_conc` 的唯一配置编写说明。

它完整覆盖：

- `config` 怎么写
- `template` 怎么写
- `defaults` / `base` / `sweep_axes` 怎么配合
- `mapping.current` 和 `mapping.channel_params` 怎么写

这里的原则是：

- 这是单点单 channel compare
- 作者不需要关心 `mechanism / section`
- 作者也不需要关心 `channel / ion` 分支
- 只需要写“这个 current 叫什么”“这个参数叫什么”

运行时会自动处理：

- `ik/ina/ica` 对应的 ion reversal
- NEURON 参数到底挂在 mechanism 还是 section 上

## 1. 适用范围

`channel_no_conc` 只适合下面这类场景：

- 单 `soma`
- 单 channel
- 不涉及浓度动力学
- 结果只比较 `voltage`、`current.ix`、`gates.*`

这里的 `gates.*` 是统一结果名：

- 对 HH channel，通常就是 activation / inactivation gate
- 对 Markov no-conc channel，也可以是被显式比较的独立状态轨迹

不适合：

- 依赖 `cai` / `nai` / `ki` 等浓度状态，或需要显式比较这些浓度轨迹
- 需要显式控制多个 ion 状态的 mixed-ion 机制
- manual-structure 机制
- 多 compartment cable

例外说明：

- 少数 mixed-ion channel 如果只需要默认固定 ion 占位、并且比较目标仍然只有 `v / ix / gates`，可以接进来
- 例如 `Kca3p1_MA20_GoC`

## 2. 文件分工

这里只用两类 JSON：

- `config`
  - 定义这个 channel 是谁
  - 定义 NEURON 和 braincell 两侧怎么对应
  - 定义这个 channel 自己的默认参数
  - 声明要跑哪些模板
- `template`
  - 定义实验的公共 `base`
  - 定义 sweep 轴

一句话记忆：

- `config` 负责 channel 身份和映射
- `template` 负责实验形状和扫轴

## 3. 合并顺序

真正跑 case 之前，配置会按这个顺序合并：

1. `config.defaults`
2. `template.base`
3. `group.sweep_axes`

后者覆盖前者。

所以：

- channel 专属默认值写在 `config.defaults`
- 通用实验条件写在 `template.base`
- 真正要扫的东西写在 `group.sweep_axes`

## 4. 一个完整 config 长什么样

```json
{
  "meta": {
    "label": "PC HCN1 MA24"
  },
  "identity": {
    "mod_dir": "/home/swl/braincell/examples/neuron_compare/Cerebellum_mod/PC/channel"
  },
  "defaults": {
    "channel_params": {
      "g_max_S_cm2": 0.0001,
      "E_mV": -34.4
    }
  },
  "mapping": {
    "current": "ih",
    "impl_name": {
      "common": "HCN1_MA24_PC"
    },
    "gate_names": {
      "common": ["h"]
    },
    "channel_params": {
      "g_max_S_cm2": {
        "neuron": "gbar",
        "braincell": "g_max"
      },
      "E_mV": {
        "neuron": "eh",
        "braincell": "E"
      }
    }
  },
  "templates": [
    "../../templates/vinit_celsius.json",
    "../../templates/dc.json",
    "../../templates/ac.json"
  ]
}
```

## 5. config 顶层字段

### `meta`

只放人类可读说明。

当前最常见的是：

```json
{
  "label": "PC HCN1 MA24"
}
```

建议至少写 `label`。

### `identity`

当前只要求：

```json
{
  "mod_dir": "/abs/path/to/mods"
}
```

用途：

- 指向 NEURON 已编译机制目录

要求：

- 必填
- 目录必须存在
- 推荐写绝对路径

### `defaults`

`defaults` 可选。

用途：

- 给该 config 的全部模板提供一层 channel 专属默认值

形状：

- 和 template 的 `base` 同形状

也就是说它可以包含：

- `morphology`
- `simulation`
- `stimulus`
- `ion_state`
- `channel_params`
- `leak`

最常见用途：

- 给 channel 写固定默认 `channel_params`
- 给某个 config 写固定的 `simulation` 或 `stimulus`

HCN 的 `E_mV` 就应该写在这里：

```json
{
  "defaults": {
    "channel_params": {
      "E_mV": -34.4
    }
  }
}
```

### `mapping`

`mapping` 只负责回答 3 个问题：

1. 这个 channel 在 NEURON 里 current 变量叫什么
2. 两侧实现名字是什么
3. 公共 IR 参数在两侧分别叫什么

它不负责填参数数值。

### `templates`

这是模板 JSON 路径列表。

要求：

- 必填
- 必须是相对当前 config 目录的路径
- 路径都要存在

例如：

```json
{
  "templates": [
    "../../templates/vinit_celsius.json",
    "../../templates/dc.json",
    "../../templates/ac.json"
  ]
}
```

## 6. `mapping.current` 怎么写

现在只写一个字符串：

```json
{
  "mapping": {
    "current": "ik"
  }
}
```

这表示：

- NEURON 侧比较 `_ref_ik`
- braincell 侧统一比较这个 mechanism 的 current probe

常见值：

- `ik`
- `ina`
- `ica`
- `ih`
- 其他机制自己暴露的 current 名

自动规则：

- `ik` 自动识别成 K channel
- `ina` 自动识别成 Na channel
- `ica` 自动识别成 Ca channel
- 其他值例如 `ih`，视为 pure-channel current

这意味着作者不用再写：

- `owner = "channel"`
- `owner = "ion"`
- `ion_name`

统一只写 current 名。

## 7. `ion_state` 什么时候写

### 自动识别到 `ik/ina/ica`

如果 `mapping.current` 是：

- `ik`
- `ina`
- `ica`

那么 schema 会自动识别这是离子通道。

这时允许写：

```json
{
  "ion_state": {
    "E_mV": -80.0
  }
}
```

或者：

```json
{
  "ion_state": {
    "Ci_mM": 0.00024,
    "Co_mM": 2.0
  }
}
```

如果不写，会自动补默认值：

- `na = 50 mV`
- `k = -80 mV`
- `ca = 120 mV`

### 其他 current，例如 `ih`

如果 `mapping.current` 不是 `ik/ina/ica`，那么它被视为 pure-channel current。

这时：

- 不允许写 `ion_state`
- reversal potential 如果需要，应该写在 `channel_params.E_mV`

HCN 就是这样。

## 8. `mapping.impl_name` 怎么写

两侧名字相同就写：

```json
{
  "impl_name": {
    "common": "Kv1p1_MA24_PC"
  }
}
```

两侧名字不同就写：

```json
{
  "impl_name": {
    "neuron": "Kv",
    "braincell": "IK_Kv_test"
  }
}
```

不要混写 `common` 和 `neuron/braincell`。

## 9. `mapping.gate_names` 怎么写

两侧 gate 名相同：

```json
{
  "gate_names": {
    "common": ["n"]
  }
}
```

两侧 gate 名不同：

```json
{
  "gate_names": {
    "canonical": ["m", "h"],
    "neuron": ["m", "h"],
    "braincell": ["m_gate", "h_gate"]
  }
}
```

规则：

- 三个列表长度必须一致

补充约定：

- `gate_names` 只声明“要比较的可观测状态名”，不限制它必须是 HH gate
- 对 Markov channel，推荐只列两侧都能直接采样的独立状态
- 如果某个状态是通过概率守恒重建出来的冗余状态，就不要放进这里

例如 `Nav1p6_MA20_GoC` 在这里比较的是：

- `C1` 到 `C5`
- `I1` 到 `I5`
- `O`
- `B`

不包含冗余状态 `I6`。

## 10. `mapping.channel_params` 怎么写

这里写的是“公共 IR 参数名到两侧真实名字的映射”，不是数值。

例如：

```json
{
  "channel_params": {
    "g_max_S_cm2": {
      "neuron": "gbar",
      "braincell": "g_max"
    }
  }
}
```

或者：

```json
{
  "channel_params": {
    "E_mV": {
      "neuron": "eh",
      "braincell": "E"
    }
  }
}
```

作者现在只需要写参数名本身。

不需要再写：

- `mechanism`
- `section`

运行时会自动处理：

- 如果这个 NEURON 参数在 mechanism 上，就写 mechanism
- 如果不在 mechanism、但在 section 上，就写 section
- 如果两边都不存在，会报错
- 如果 mechanism 和 section 同时都有同名字段，会报歧义错误

### `channel_params` 可以写 `common` 吗

可以。

如果两侧参数名相同：

```json
{
  "channel_params": {
    "alpha_shift_mV": {
      "common": "alpha_shift"
    }
  }
}
```

### 数值写在哪里

不要把数值写进 `mapping.channel_params`。

真正的数值写在：

- `config.defaults.channel_params`
- `template.base.channel_params`
- `group.sweep_axes` 里的 `channel_params.<name>`

## 11. 一个完整 template 长什么样

```json
{
  "meta": {
    "label": "Quiet v_init and temperature scan"
  },
  "base": {
    "morphology": {
      "length_um": 10.0,
      "diam_um": 31.830988618379067,
      "cm_uF_cm2": 1.0
    },
    "simulation": {
      "dt_ms": 0.025,
      "duration_ms": 10.0,
      "v_init_mV": -65.0,
      "temperature_celsius": 25.0
    },
    "stimulus": {
      "kind": "dc",
      "delay_ms": 0.0,
      "dur_ms": 10.0,
      "amp_nA": 0.0
    },
    "leak": {
      "enabled": false,
      "g_S_cm2": 0.0,
      "e_mV": -65.0
    }
  },
  "group": {
    "group_id": "vinit_celsius",
    "description": "Quiet DC sweep over initial voltage and temperature.",
    "sweep_axes": {
      "simulation.v_init_mV": [-80.0, -65.0, -50.0],
      "simulation.temperature_celsius": [22.0, 25.0, 37.0]
    }
  },
  "outputs": {
    "plot": false
  }
}
```

## 12. template 顶层字段

### `meta`

放人类可读说明。

### `base`

`base` 是实验公共底板，允许包含以下块。

#### `morphology`

支持：

- `length_um`
- `radius_um` 或 `diam_um`
- `cm_uF_cm2`

#### `simulation`

支持：

- `dt_ms`
- `duration_ms`
- `v_init_mV`
- `temperature_celsius`

#### `stimulus`

支持两种形状。

`dc`：

```json
{
  "kind": "dc",
  "delay_ms": 0.0,
  "dur_ms": 10.0,
  "amp_nA": 0.0
}
```

`sine`：

```json
{
  "kind": "sine",
  "start_ms": 0.0,
  "duration_ms": 20.0,
  "amplitude_nA": 0.02,
  "frequency_hz": 100.0,
  "phase_rad": 0.0,
  "offset_nA": 0.0
}
```

#### `ion_state`

只有在 `mapping.current` 自动识别到 `ik/ina/ica` 时才有意义。

支持两种互斥模式：

- 固定 reversal：`E_mV`
- 固定浓度输入：`Ci_mM` + `Co_mM`

#### `channel_params`

只允许出现已经在 `mapping.channel_params` 里声明过的公共参数名。

#### `leak`

支持：

```json
{
  "leak": {
    "enabled": false,
    "g_S_cm2": 0.0,
    "e_mV": -65.0
  }
}
```

### `group`

#### `group_id`

模板自己的 sweep 组名。

#### `description`

可选说明。

#### `sweep_axes`

使用 dotted path。

例如：

```json
{
  "sweep_axes": {
    "simulation.v_init_mV": [-80.0, -65.0, -50.0],
    "simulation.temperature_celsius": [22.0, 25.0, 37.0]
  }
}
```

### `outputs`

当前最常见的是：

```json
{
  "outputs": {
    "plot": false
  }
}
```

## 13. `sweep_axes` 允许扫什么

所有模板都允许：

- `simulation.dt_ms`
- `simulation.v_init_mV`
- `simulation.temperature_celsius`
- `leak.enabled`
- `leak.g_S_cm2`
- `leak.e_mV`
- `channel_params.<任何已声明公共参数>`

如果 `mapping.current` 自动识别到 `ik/ina/ica`，额外允许：

- `ion_state.E_mV`
- `ion_state.Ci_mM`
- `ion_state.Co_mM`

当 `stimulus.kind == "dc"`，额外允许：

- `stimulus.amp_nA`
- `stimulus.delay_ms`
- `stimulus.dur_ms`

当 `stimulus.kind == "sine"`，额外允许：

- `stimulus.start_ms`
- `stimulus.duration_ms`
- `stimulus.amplitude_nA`
- `stimulus.frequency_hz`
- `stimulus.phase_rad`
- `stimulus.offset_nA`

## 14. 通用模板和 channel 专属默认值

当前内置 3 个通用模板：

- [vinit_celsius.json](/home/swl/braincell/examples/neuron_compare/channel_no_conc/templates/vinit_celsius.json)
- [dc.json](/home/swl/braincell/examples/neuron_compare/channel_no_conc/templates/dc.json)
- [ac.json](/home/swl/braincell/examples/neuron_compare/channel_no_conc/templates/ac.json)

这些模板只负责公共实验条件。

如果某个 channel 有自己的固定默认值，例如：

- HCN 的 `E_mV = -34.4`
- 某个 channel 的默认 `g_max_S_cm2`

优先写到 `config.defaults`，不要写死进通用模板。

## 15. 两个可直接仿写的例子

### K channel

参考：

- [kv_test.json](/home/swl/braincell/examples/neuron_compare/channel_no_conc/configs/kv_test.json)

它展示了：

- `mapping.current = "ik"`
- `g_max_S_cm2` 映射到 `gbar / g_max`
- `ion_state` 可以用默认 `k = -80 mV`

### HCN pure-channel

参考：

- [hcn1_ma24_pc.json](/home/swl/braincell/examples/neuron_compare/channel_no_conc/configs/ma24_pc/hcn1_ma24_pc.json)

它展示了：

- `mapping.current = "ih"`
- `E_mV` 写在 `channel_params`
- `defaults.channel_params.E_mV = -34.4`
- 不需要显式声明 `section`

## 16. BC / PC / GoC 现成批次

当前已经整理好的 compare config 目录有：

- `configs/ma20_grc/`
- `configs/ma24_pc/`
- `configs/ma25_bc/`
- `configs/ma20_goc/`
- `configs/su15_dcn/`
- `configs/ri21_sc/`
- `configs/su15_dcn/`

`ma24_pc/` 目前放了 PC 里已经可直接从 `braincell.channel` 导入、且适合走 `channel_no_conc` compare 的 7 个 channel：

- `HCN1_MA24_PC`
- `Kir2p3_MA24_PC`
- `Kv1p1_MA24_PC`
- `Kv1p5_MA24_PC`
- `Kv3p3_MA24_PC`
- `Kv3p4_MA24_PC`
- `Kv4p3_MA24_PC`

其中 `Kv3p3_MA24_PC` 曾被误分到 `Markov_no_conc`，但 `.mod` 注释和方程都是 HH `n^4`，这里只按 `hh_special_current` 接入。

PC 其余机制不在这个目录的原因：

- `Nav1p6_MA24_PC` 属于 `Markov_no_conc`
- `CdpCAM_MA24_PC` 属于 `Ion_dyn`
- `Cav2p1_MA24_PC`、`Cav3p1_MA24_PC`、`Cav3p2_MA24_PC`、`Cav3p3_MA24_PC`、`Kca3p1_MA24_PC` 属于 `HH_conc`
- `Kca1p1_MA24_PC`、`Kca2p2_MA24_PC` 属于 `Markov_conc`

`ma25_bc/` 目前放了 BC 里已经可直接从 `braincell.channel` 导入、且适合走 `channel_no_conc` compare 的 6 个 channel：

- `HCN1_MA25_BC`
- `Nav1p6_MA25_BC`
- `Kir2p3_MA25_BC`
- `Kv1p1_MA25_BC`
- `Kv3p4_MA25_BC`
- `Kv4p3_MA25_BC`

为什么不是 BC 全部 15 个机制都在这里：

- `Nav1p1_MA25_BC` 属于 `Markov_no_conc`
- `CdpStC_MA25_BC` 属于 `Ion_dyn`
- `Cav1p2_MA25_BC`、`Cav1p3_MA25_BC`、`Cav2p1_MA25_BC`、`Cav3p2_MA25_BC`、`Kca3p1_MA25_BC` 属于 `HH_conc`
- `Kca1p1_MA25_BC`、`Kca2p2_MA25_BC` 属于 `Markov_conc`

其中 `Nav1p6_MA25_BC` 虽然是 `Markov_no_conc`，但已经按 12 个独立状态接入。

`ma20_grc/` 当前覆盖的是 GrC 里已经可直接从 `braincell.channel` 导入、并且适合走当前 compare 流程的 1 个 channel：

- `Kv1p5_MA20_GrC`

其中 `Kv1p5_MA20_GrC` 是一个默认参数特例：

- NMODL 同时声明 `ik` 和 `ino`
- `ino` 由 `gnonspec` 控制，但 `gnonspec` 默认是 `0`
- 当前 BrainCell 实现只转换默认 `ik` path，不支持非零 `gnonspec / ino`

GrC 其余机制不在这个目录的原因：

- `Nav_MA20_GrC`、`NaFHF_MA20_GrC` 属于 `Markov_no_conc`
- `CdpCR_MA20_GrC` 属于 `Ion_dyn`
- `Kca1p1_MA20_GrC`、`Kca2p2_MA20_GrC` 属于 `Markov_conc`
- 其他 GrC HH_no_conc / HH_conc channel 已在对应批次或通道测试中覆盖，后续可按需补 config

`ma20_goc/` 当前覆盖的是 GoC 里已经可直接从 `braincell.channel` 导入、并且适合走 `channel_no_conc` compare 的 12 个 channel：

- `HCN1_MA20_GoC`
- `HCN2_MA20_GoC`
- `KM_MA20_GoC`
- `Kca1p1_MA20_GoC`
- `Kca2p2_MA20_GoC`
- `Kca3p1_MA20_GoC`
- `Kv1p1_MA20_GoC`
- `Kv3p4_MA20_GoC`
- `Kv4p3_MA20_GoC`
- `CaHVA_MA20_GoC`
- `Cav2p3_MA20_GoC`
- `Nav1p6_MA20_GoC`

这 12 个已经能直接写成 compare config。

其中 `Nav1p6_MA20_GoC` 虽然是 `Markov_no_conc`，但当前已经按 12 个独立状态接入：

- `C1` 到 `C5`
- `I1` 到 `I5`
- `O`
- `B`

不比较冗余状态 `I6`。

`Kca3p1_MA20_GoC` 虽然是 mixed-ion KCa，但当前可以在默认固定 `ca` 占位下进入 compare：

- `mapping.current` 仍然写 `ik`
- `ion_state` 只显式控制 `k` reversal
- `ca` 侧不暴露浓度配置，也不比较浓度轨迹

`Kca2p2_MA20_GoC` 和 `Kca1p1_MA20_GoC` 也是同类特例，但由于它们属于守恒约束的 Markov 模型，compare config 只比较独立状态：

- `Kca2p2_MA20_GoC` 不比较冗余态 `c1 / C1`
- `Kca1p1_MA20_GoC` 不比较冗余态 `C0`

注意：

- `GoC/channel` 这批 mod 在 conda toolchain 下容易编译失败
- 当前环境需要用系统编译器重新编译 GoC mod，推荐命令是：

```bash
env PATH=/usr/bin:/bin:$PATH \
  CC=/usr/bin/cc \
  CXX=/usr/bin/c++ \
  GCC=/usr/bin/gcc \
  CPP=/usr/bin/cpp \
  CFLAGS= \
  CXXFLAGS= \
  LDFLAGS= \
  nrnivmodl
```

- 目录位置：
  - `examples/neuron_compare/Cerebellum_mod/GoC/channel`
- 编译成功后会生成：
  - `x86_64/libnrnmech.so`
  - `x86_64/special`
- `ma20_goc/` 这批 config 现在按上面的编译方式已经可以进入真实 compare 流程

GoC 其余机制不在这个目录的原因：

- `CdpStC_MA20_GoC` 属于 `Ion_dyn`
- `Cav1p2_MA20_GoC`、`Cav1p3_MA20_GoC`、`Cav3p1_MA20_GoC` 属于 `HH_conc`

`su15_dcn/` 当前覆盖的是 DCN 里已经可直接从 `braincell.channel` 导入、并且适合走当前 compare 流程的 7 个 channel：

- `HCN_SU15_DCN`
- `NaF_SU15_DCN`
- `NaP_SU15_DCN`
- `fKdr_SU15_DCN`
- `sKdr_SU15_DCN`
- `SK_SU15_DCN`
- `CaL_SU15_DCN`

其中 `SK_SU15_DCN` 是 KCa / `HH_conc` 特例：

- `mapping.current` 仍然写 `ik`
- BrainCell 实现绑定 `Potassium + Calcium`，并以 `Potassium` 作为 current owner
- 当前 config 只固定 `k` reversal；`ca` 侧沿用两边默认 `cai`，不比较浓度轨迹

`CaL_SU15_DCN` 也是 `HH_conc` 来源的特例，但 NMODL 使用自定义 `call` ion 并在 current 中固定 `carev = 139 mV`：

- `mapping.current` 写 `icall`
- BrainCell 实现用 `HHTypedNeuron` + 显式 `E = 139 mV`
- compare 走 mechanism current probe，不绑定标准 `ca` ion

DCN 其余机制不在这个目录的原因：

- `CaHVA_SU15_DCN`、`CaLVA_SU15_DCN` 属于 `HH_conc`
- `CdpHVA_SU15_DCN`、`CdpLVA_SU15_DCN` 属于 `Ion_dyn`

`ri21_sc/` 当前覆盖的是 SC 里已经可直接从 `braincell.channel` 导入、并且适合走 `channel_no_conc` compare 的 9 个 channel：

- `Cav2p1_RI21_SC`
- `Cav3p2_RI21_SC`
- `Cav3p3_RI21_SC`
- `HCN1_RI21_SC`
- `KM_RI21_SC`
- `Kir2p3_RI21_SC`
- `Kv1p1_RI21_SC`
- `Kv3p4_RI21_SC`
- `Kv4p3_RI21_SC`

SC 其余机制不在这个目录的原因：

- `Nav1p1_RI21_SC`、`Nav1p6_RI21_SC` 属于 `Markov_no_conc`
- `CdpStC_RI21_SC` 属于 `Ion_dyn`
- `Kca1p1_RI21_SC`、`Kca2p2_RI21_SC` 属于 `Markov_conc`

`su15_dcn/` 当前覆盖的是 DCN 里已经可直接从 `braincell.channel` 导入、并且适合走当前 compare 流程的 7 个 channel：

- `HCN_SU15_DCN`
- `NaF_SU15_DCN`
- `NaP_SU15_DCN`
- `fKdr_SU15_DCN`
- `sKdr_SU15_DCN`
- `SK_SU15_DCN`
- `CaL_SU15_DCN`

其中 `SK_SU15_DCN` 是 KCa / `HH_conc` 特例：

- `mapping.current` 仍然写 `ik`
- BrainCell 实现绑定 `Potassium + Calcium`，并以 `Potassium` 作为 current owner
- 当前 config 只固定 `k` reversal；`ca` 侧沿用两边默认 `cai`，不比较浓度轨迹

`CaL_SU15_DCN` 也是 `HH_conc` 来源的特例，但 NMODL 使用自定义 `call` ion 并在 current 中固定 `carev = 139 mV`：

- `mapping.current` 写 `icall`
- BrainCell 实现用 `HHTypedNeuron` + 显式 `E = 139 mV`
- compare 走 mechanism current probe，不绑定标准 `ca` ion

DCN 其余机制不在这个目录的原因：

- `CaHVA_SU15_DCN`、`CaLVA_SU15_DCN` 属于 `HH_conc`
- `CdpHVA_SU15_DCN`、`CdpLVA_SU15_DCN` 属于 `Ion_dyn`

## 17. 常见错误

### 错误 1：把运行值写进 `mapping.channel_params`

错误：

```json
{
  "mapping": {
    "channel_params": {
      "g_max_S_cm2": 0.001
    }
  }
}
```

正确：

- `mapping.channel_params` 只声明映射
- 真正数值写进 `defaults.channel_params`、`base.channel_params` 或 `sweep_axes`

### 错误 2：把 HCN 的 `E_mV` 写成 `ion_state`

HCN 这种 pure-channel 应该写：

- `channel_params.E_mV`

不要写：

- `ion_state.E_mV`

### 错误 3：把 channel 专属默认值写进通用模板

例如把 HCN 的 `E_mV=-34.4` 写进 `vinit_celsius.json`。

正确做法：

- 模板保持通用
- channel 自己的默认值写到 `config.defaults`

### 错误 4：写了一个不存在的 NEURON 参数名

现在作者只写参数名字符串。

如果这个名字在 mechanism 和 section 上都找不到，运行时会报错。

所以参数名必须真的对应 NEURON 侧字段。

## 18. 最后检查清单

写完一个新 `config + template` 后，至少自查这些：

- `mod_dir` 存在
- `templates` 路径相对当前 config 目录正确
- `mapping.current` 写的是 NEURON current 名
- `channel_params` 里出现的公共参数都已在 `mapping.channel_params` 里声明
- pure-channel 的 `E_mV` 写在 `channel_params`，不是 `ion_state`
- channel 专属默认值写在 `defaults`，不是写死进通用模板
- `sweep_axes` 里的 dotted path 在白名单内
