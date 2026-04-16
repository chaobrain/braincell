# Single Compartment Channel

## Goal

这份文档定义 `NEURON vs braincell` 的单房室 `channel` 对比测试模板。

当前分类按 **对比检测需求** 来划分，而不是按 channel 内部的具体实现细节来划分。

当前只使用两个维度：

1. `channel kinetics`
   - `HH`
   - `Markov`
2. `ion side`
   - `fixed ion`
   - `dynamic ion`

因此当前共有四类模板：

1. `HH + fixed ion`
2. `Markov + fixed ion`
3. `HH + dynamic ion`
4. `Markov + dynamic ion`

## Classification Rule

### Why not split `ohmic` and `GHK`

在这份 **对比检测模板** 中，不需要单独区分 `ohmic` 和 `GHK`。

原因是：

- 它们在测试流程上没有区别
- 差异只体现在 channel 的构建方式
- 这里不关心 channel 如何构建，只关心如何进行对比检测

因此：

- 对于 `fixed ion` 模板，不管内部 drive model 是 `ohmic` 还是 `GHK`
  - 都归到同一个模板里
  - 统一只观测 `v / ix / channel states`
  - 不观测 `E / xi / xo`
- 对于 `dynamic ion` 模板，不管内部 drive model 是 `ohmic` 还是 `GHK`
  - 都归到同一个模板里
  - 统一观测 `v / ix / channel states / E / xi / xo`

## Template Matrix

| Template | Description | Dependency | Priority |
|---|---|---|---|
| `HH + fixed ion` | HH gate channel with fixed ion side | none | P0 |
| `Markov + fixed ion` | Markov-state channel with fixed ion side | none | P1 |
| `HH + dynamic ion` | HH gate channel with dynamic ion side | requires `single_compartment_ion` | P2 |
| `Markov + dynamic ion` | Markov-state channel with dynamic ion side | requires `single_compartment_ion` | P3 |

当前优先顺序：

1. `HH + fixed ion`
2. `Markov + fixed ion`
3. 两个 `dynamic ion` 模板

## Shared Template Layer

下面这些流程对四类模板都通用。

### Morphology Template

- 单房室
- 一个 `soma`
- `NEURON` 与 `braincell` 使用相同几何参数
- 建议固定参数：
  - `L`
  - `diam` 或 `radius`
  - `cm`

### Cell Lifecycle Template

- 每个 sweep case 独立初始化
- 不做连续运行比较
- 每个 case 都从同一初始状态开始
- 默认流程：
  - build model
  - set temperature / ion / mechanism / stimulus
  - initialize
  - run once

### Stimulus Template

默认刺激模板包含三类：

1. `DC clamp`
   - 覆盖正负 `amp`
   - 固定 `delay`
   - 固定 `dur`
2. `AC clamp`
   - `frequency`
   - `amplitude`
   - `offset`
   - `start`
   - `duration`
3. `temperature sweep`
   - 固定几个代表性温度点

### Shared Observables Categories

公共观测类别如下：

- `v`
- `ix`
- `channel states`
- `E`
- `xi`
- `xo`

不同模板会从中选择子集。

### Preprocessing Template

统一预处理步骤：

- `NEURON` 结果裁掉初始采样点
- shape squeeze
- pairwise align
- 对 unsupported observable 明确记录
- 每个 case 的同类观测量按统一 key 聚合

### Metrics Template

统一统计：

- `mae`
- `rmse`
- `max_abs`
- `rel_mae_pct`

### Output Template

每个 case 输出至少包含：

- 原始时间轴
- 对齐后的对比 traces
- 指标统计
- 对比图

## Template-Specific Layer

### 1. HH + fixed ion

#### Goal

验证单房室下，`fixed ion + HH channel` 组合的数值行为在 `NEURON` 与 `braincell` 之间是否一致。

#### braincell setup

- 单房室 cell
- `fixed ion`
- 一个 HH 型 channel
- 可选固定 leak
- 不使用动态浓度更新
- 不要求使用 Nernst

#### NEURON setup

- 单个 `soma` section
- 插入一个被测 channel
- 可选固定 leak
- 不比较浓度动态
- 不要求观测 `E / xi / xo`

#### Required Observables

- `v`
- `ix`
- HH gate vars 全集

这里的 gate vars 指：

- `m / h / n / p / q ...`
- 以被测机制真实存在的 gate/state 变量为准

#### Unsupported Observables

这一类模板中，不观测：

- `E`
- `xi`
- `xo`

即使该机制内部是 `GHK` drive，也不单独增加这些观测。

#### Notes

- 不区分 `ohmic` / `GHK`
- 两者的检测模板完全一致
- 差异只体现在 channel 构建和当前值计算，不影响对比流程

#### First Implementation Target

第一批实现建议固定为：

- 一个最简单的 HH 单 channel
- 一个固定 ion side
- 一个固定观测位置
- 一个最小刺激 sweep

### 2. Markov + fixed ion

#### Goal

验证单房室下，`fixed ion + Markov channel` 组合的数值行为。

#### Required Observables

- `v`
- `ix`
- Markov state vars 全集

#### Unsupported Observables

- `E`
- `xi`
- `xo`

#### Notes

- 与 `HH + fixed ion` 相同，不区分 `ohmic / GHK`
- 仅将 `channel states` 从 HH gates 换成 Markov states

### 3. HH + dynamic ion

#### Goal

验证单房室下，`dynamic ion + HH channel` 组合的数值行为。

#### Dependency

- 需要 `single_compartment_ion` 先通过

#### Required Observables

- `v`
- `ix`
- HH gate vars 全集
- `E`
- `xi`
- `xo`

#### Notes

- 与 `single_compartment_ion` 有部分重叠
- 当前先接受这种重叠，不做去重设计

### 4. Markov + dynamic ion

#### Goal

验证单房室下，`dynamic ion + Markov channel` 组合的数值行为。

#### Dependency

- 需要 `single_compartment_ion` 先通过

#### Required Observables

- `v`
- `ix`
- Markov state vars 全集
- `E`
- `xi`
- `xo`

#### Notes

- 同样允许与 `single_compartment_ion` 模板有重叠

## Current V1 Boundary

当前文档只先细化：

- `HH + fixed ion`

其余三类目前只保留骨架，不展开具体 sweep、参数表与通过阈值。

## Not In This Document Yet

当前还不在这份文档里展开：

- 每个模板的具体参数 sweep 表
- 每个模板的具体误差阈值
- 具体机制名映射表
- 具体 case 文件格式
- 具体 runner 输出字段

这些后续在：

- 对应 task 的实现脚本
- 或更细的 task/spec 文档

中继续补。
