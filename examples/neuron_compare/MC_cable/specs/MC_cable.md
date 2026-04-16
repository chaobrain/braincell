# Multi Compartment Cable

## Goal

这份文档定义 `NEURON vs braincell` 的多房室 `cable` 对比测试模板。

这个模板专门验证：

- 同一份 `SWC`
- 不挂任何 `ion`
- 不挂任何 `channel`
- 不挂任何 `leak`
- 只保留被动电缆参数与轴向耦合
- 只在 root `soma` 中点打一段电流刺激

然后比较 `NEURON` 与 `braincell` 的所有房室电压轨迹是否一致。

这里的核心目标不是测膜机制，而是单独验证：

- 在相同离散条件下，电压传播是否一致
- 不同刺激下，电压响应是否一致

## Position In This Directory

这个模板放在 `examples/neuron_compare/MC_cable` 里，作为新的对比方向：

- `channel`
- `ion`
- `cable`

它不挂到 `multi_compartment_ion` 或 `multi_compartment_channel` 下面。

## Upstream Boundary

这份模板不负责 morphology metric gatekeeping。

即：

- 这里不复用 `neuron_diff` 的 metric 输出
- 这里不负责先判断 morphology metric 是否通过
- 这里只假设输入的 `SWC` 已经由上游流程判定可进入 voltage compare

## V1 Boundary

`v1` 只做下面这一个最小闭环：

- 同一份 `SWC`
- 同一套 `Ra / cm / dt / duration / v_init`
- 同一个 root `soma(0.5)` 刺激定位
- 比较所有 CV 的 `voltage`

`v1` 明确不做：

- `ion`
- `channel`
- `leak`
- `E / xi / xo`
- `channel states`
- 把 `axial current` 单独作为正式比较输出项

`axial current / axial coupling` 是这个模板的设计动机，但 `v1` 的正式判据只看所有房室的电压。

## Shared Build Rule

### Morphology Input

- `NEURON` 与 `braincell` 必须导入同一份 `SWC`
- `braincell` 侧固定使用：
  - `Morphology.from_swc(..., mode="neuron")`
- `NEURON` 侧固定使用：
  - `import3d` 导入同一份 `SWC`

### Cable-Only Rule

两边都只允许设置电缆参数：

- `Ra`
- `cm`
- `v_init`

不插入任何膜机制：

- 不插 `ion`
- 不插 `channel`
- 不插 `pas`

### Time Integration Rule

- 固定步长
- 两边使用相同 `dt`
- 两边使用相同 `duration`
- 不使用变步长

## Discretization Rule

### V1 Supported `cv_policy`

`v1` 只正式支持：

- `CVPerBranch`

即 case/spec 中固定写：

- `cv_policy.kind = "CVPerBranch"`
- `cv_policy.cv_per_branch = <odd positive int>`

### Why `cv_per_branch` Must Be Odd

`v1` 要求：

- `cv_per_branch` 必须是正奇数

原因是这份模板把刺激位置固定在 root `soma(0.5)`。

只有当刺激所在 branch 使用奇数个离散段时：

- `0.5` 才是一个真实的 compartment midpoint
- `braincell` 与 `NEURON` 才能把刺激放在同一个离散中心

如果使用偶数分段：

- 在 `braincell` 中，`0.5` 会落在 CV 边界
- 在 `NEURON` 中，`soma[0](0.5)` 仍是 section 内部位置

这样两边刺激语义不再天然一致，因此 `v1` 直接禁止偶数分段。

### NEURON Side Discretization

对于每个由 `SWC` 导入得到的 `NEURON` section：

- `nseg = cv_per_branch`

### braincell Side Discretization

`braincell` 固定使用：

- `Cell(..., cv_policy=CVPerBranch(cv_per_branch=<same odd int>))`

## Stimulus Rule

### Stimulus Location

只允许刺激 root `soma` 的中点。

规范写法：

- `braincell`: `soma(0.5)`
- `NEURON`: `soma[0](0.5)`

### Multiple Soma Rule

如果同一个 `SWC` 导入后存在多个 `soma` branch / section：

- `braincell` 侧后续名字可能是：
  - `soma_0`
  - `soma_1`
  - ...
- `NEURON` 侧后续名字可能是：
  - `soma[1]`
  - `soma[2]`
  - ...

但 `v1` 仍然只刺激 root `soma`：

- `braincell`: `soma(0.5)`
- `NEURON`: `soma[0](0.5)`

不切换到 `soma_0(0.5)` 或 `soma[1](0.5)`。

### V1 Stimulus Suite

`v1` 正式支持三类刺激：

1. `dc_step`
2. `piecewise_step`
3. `sine`

### `dc_step`

字段：

- `delay_ms`
- `dur_ms`
- `amp_nA`

### `piecewise_step`

字段：

- `start_ms`
- `durations_ms[]`
- `amplitudes_nA[]`

要求：

- `durations_ms` 与 `amplitudes_nA` 长度一致
- 两者都非空

### `sine`

字段：

- `start_ms`
- `duration_ms`
- `amplitude_nA`
- `frequency_hz`
- `phase_rad`
- `offset_nA`

## Required Observable

`v1` 只要求一个正式观测量：

- all-compartment `voltage`

具体来说：

- 比较 `braincell` 所有 `CV` 的电压轨迹
- 比较 `NEURON` 所有 segment midpoint 的电压轨迹
- 两边按统一离散顺序配对后逐一比较

## Mapping Rule

### Mapping Principle

比较顺序固定使用两级排序：

1. branch / section 顺序
2. branch / section 内从 proximal 到 distal 的 CV / segment 顺序

这里不使用字符串名字作为主映射键。

原因是：

- `braincell` 名字可能是 `soma / basal_dendrite_0 / soma_0`
- `NEURON` 名字可能是 `soma[0] / dend[0] / soma[1]`

名字只用于诊断输出，不用于主映射。

### braincell Ordering

`braincell` 侧按下面顺序展开：

1. `Morphology.branches` 的顺序
2. 每个 branch 内按 proximal -> distal 的 `CV` 顺序

### NEURON Ordering

`NEURON` 侧按下面顺序展开：

1. `import3d` 实例化后的 section 顺序
2. 每个 section 内按 `for seg in sec` 的顺序

### Failure On Mismatch

如果出现下面任一情况，直接报错，不进入数值比较：

- 两边离散后的 compartment 数不一致
- 无法稳定定位 root `soma`
- `cv_per_branch` 不是正奇数

## Suggested Case Fields

`v1` 建议 case 至少包含：

- `template_family = "multi_compartment_cable"`
- `case_id`
- `swc.path`
- `simulation.dt_ms`
- `simulation.duration_ms`
- `simulation.v_init_mV`
- `cable.ra_ohm_cm`
- `cable.cm_uF_cm2`
- `cv_policy.kind = "CVPerBranch"`
- `cv_policy.cv_per_branch`
- `stimulus.target = "root_soma_midpoint"`
- `stimulus.kind in {"dc_step", "piecewise_step", "sine"}`

不同 `kind` 的具体字段见上面的 `Stimulus Suite`。

## Suggested Output Fields

每个 case 输出建议至少包含：

- `time_ms`
- `braincell.voltage_mV`
- `neuron.voltage_mV`
- `alignment.branch_order`
- `alignment.compartment_labels`
- `metrics.overall`
- `metrics.per_compartment`
- `plots.voltage_overlay`

## Metrics

对每个 case，至少输出：

- `mae`
- `rmse`
- `max_abs`
- `rel_mae_pct`

建议同时给两层汇总：

- overall summary
- per-compartment summary

## V1 Test Matrix

`v1` 至少覆盖下面三类 case：

1. 单 `soma` + 单 `dend`，`cv_per_branch = 1`
2. 同一份 `SWC`，`cv_per_branch = 3`
3. 含多个 `soma` section 的 `SWC`，用于验证仍只刺激 root `soma`

在刺激维度上，`v1` 至少各覆盖一次：

1. `dc_step`
2. `piecewise_step`
3. `sine`

## Acceptance Rule

一个 case 通过的含义是：

- 所有房室都参与比较
- 每个房室都有有效的电压配对
- 整体指标与逐房室指标都在阈值内

阈值本身不在这份文档中展开，后续由具体 runner/case 文档补充。
