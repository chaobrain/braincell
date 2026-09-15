# Nonlinear Pattern Separation 实验报告

## 1. 实验概览（Overview）

本实验验证 BrainCell 是否能够复现 Jaxley reference implementation 的 **Nonlinear Pattern Separation** 任务。实验比较两种实现的训练结果、参数变化、分类行为和 nonlinear response surface。

本实验属于 **Validation / Reproduction Study**，不是 **Performance Benchmark**。运行时间、吞吐量、显存和扩展性应在仓库的 `benchmarks/` 中单独测量。

## 2. 数据来源（Data Provenance）

BrainCell 的任务定义位于 `runners/braincell/task.py`，训练入口位于 `runners/braincell/`。Jaxley 结果来自配套仓库 **jaxley_nonlinear_reproduction**，本仓库只保存本次 comparison 使用的 raw export。

Jaxley raw data 的来源、source commit、软件版本和 SHA256 见：

- [`raw/jaxley/SOURCE.md`](../artifacts/nonlinear_pattern_separation_2026-09-15/raw/jaxley/SOURCE.md)

分析和绘图不依赖外部 Jaxley checkout。

## 3. 实验数据（Dataset）

| 项目 | 设置 |
|---|---|
| 输入维度 | 2 |
| 输入分布 | 三个 Gaussian clusters |
| Cluster centers | `(1.5, 3.5)`、`(2.5, 2.5)`、`(3.5, 1.5)` |
| Standard deviation | `0.1` |
| Train samples | 32 |
| Test samples | 16 |
| Class 1 | 两个 outer clusters，目标为 spike |
| Class 0 | 中间 cluster，目标为 non-spike |
| Voltage target | spike `35 mV`，non-spike `-70 mV` |
| Readout time | `3.0 ms` |
| Time step | `0.025 ms` |
| Observation window | `5.95 ms` |

Train/test 数据均由 seed 独立生成。BrainCell 使用与 reference task 相同的 cluster 定义和标签规则。

## 4. 模型构建（Model Construction）

模型是一个双树突 HH cell：一个 soma/root branch 连接两个 dendritic branches（`dend_a` 和 `dend_b`）。每个 branch 被离散为 4 个 CV，因此总数为 12 CV。

| 模型组件 | 设置 |
|---|---|
| CV 数量 | 12（每个 branch 4 CV） |
| Solver | `staggered` |
| Initial membrane voltage | `-70 mV` |
| Membrane capacitance | `1 uF/cm²`，固定 |
| Axial resistivity 初始值 | `3000 ohm cm` |
| Sodium channel | `Na_HH1952` |
| Potassium channel | `K_HH1952` |
| Leak channel | `IL`，`E = -54.387 mV` |
| Sodium reversal potential | `50 mV` |
| Potassium reversal potential | `-77 mV` |

## 5. 可训练参数和边界（Trainable Parameters and Bounds）

每个参数在 12 个 CV 上独立训练，因此共有 `6 × 12 = 72` 个 trainable parameters。

| 参数 | BrainCell 字段 | 物理含义 | Lower bound | Upper bound | Unit |
|---|---|---|---:|---:|---|
| Sodium conductance | `na.g_max` → `gNa` | Sodium conductance density | 50 | 1100 | `mS/cm²` |
| Potassium conductance | `k.g_max` → `gK` | Potassium conductance density | 10 | 300 | `mS/cm²` |
| Leak conductance | `leak.g_max` → `gLeak` | Leak conductance density | 0.1 | 1.0 | `mS/cm²` |
| Length | `length` | CV length | 1 | 20 | `um` |
| Radius | `radius_scale` → `radius_mid` | Physical CV radius | 0.1 | 5.0 | `um` |
| Axial resistivity | `Ra` | Axial resistivity | 500 | 5500 | `ohm cm` |

BrainCell 不能直接把 radius 当作简单圆柱半径进行训练，因此优化的是 dimensionless `radius_scale`，最终通过 reference radius `2.55 um` 映射为 physical `radius_mid`。所有结果图和汇总均使用 materialized physical values。

所有边界通过 bounded sigmoid transform 实现。优化器内部的 parameter roots 可以是 unconstrained representation，但导出的物理参数始终受表中 bounds 约束。

## 6. Loss Function

本次正式对照使用：

```text
loss_kind = voltage_at_3ms
```

其定义为训练样本在 3 ms readout 时刻的平均绝对电压误差：

```text
L = mean(abs(V_pred(3 ms) - V_target))
```

代码仍支持 `peak_margin` 和 `soft_spike` 两种 loss，但它们不用于本次正式 BrainCell/Jaxley comparison。

## 7. 优化器和梯度（Optimizer and Gradients）

| 项目 | 设置 |
|---|---|
| Optimizer | `braintools.optim.Adam` |
| Learning rate | `0.01` |
| Gradient method | exact full RTRL |
| Batch size | 1 |
| Epochs | 200 |
| Updates per seed | `200 × 32 = 6400` |
| Precision | float64 |
| Backend | CPU |

BPTT 只用于 geometry gradient correctness validation；正式 nonlinear pattern training 使用 exact full RTRL。

## 8. 训练结果（Results）

BrainCell 结果位于：

- [`raw/braincell/summary.json`](../artifacts/nonlinear_pattern_separation_2026-09-15/raw/braincell/summary.json)

跨框架汇总位于：

- [`analysis/summaries/framework_performance_summary.json`](../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/summaries/framework_performance_summary.json)

主要 figures 位于 [`analysis/figures/`](../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/figures/)：

| Figure | 内容 |
|---|---|
| `surface_comparison_jaxley_braincell.png` | Jaxley/BrainCell nonlinear voltage surface 对照 |
| `performance_before_after_by_seed.png` | 各 seed/restart 的表现 |
| `parameter_before_after_seed_*.png` | BrainCell 每个 seed 的参数 before/after |
| `jaxley_parameter_before_after_seed_*.png` | Jaxley 每个 restart 的参数 before/after |
| `braincell_all_seed_voltage_surfaces.png` | BrainCell 多 seed voltage surfaces |
| `braincell_all_seed_spike_surfaces.png` | BrainCell 多 seed spike surfaces |
| `braincell_test_voltage_traces_6ms.png` | 10 个 seed 的最终 test voltage traces，标记 3 ms 和两个 voltage labels |
| `jaxley_test_voltage_traces_6ms.png` | 10 个 restart 的最终 test voltage traces，使用相同标记 |

结果汇总时应同时报告 mean、median、best 和成功率，避免只展示单个最佳 seed。seed/restart 编号用于并排展示，不表示两种实现使用了完全相同的随机数流。

## 9. 几何梯度验证（Geometry Gradient Validation）

几何 validation 覆盖：

- cable JAX gradient 与 central finite difference 对照；
- runtime geometry update 后 area 改变且 CV topology 不变；
- `row`、`cv`、`population` 和 `all` parameter grouping；
- BPTT/RTRL rollout 和 finite-difference gradient 一致性；
- synthetic cable parameter recovery；
- 多步 synthetic voltage loss descent。

测试文件为：

```text
runners/braincell/geometry_checks_test.py
```

这些检查验证 gradient correctness 和数值 descent，不代表 nonlinear task 一定能够唯一恢复真实参数。

## 10. 数值稳定性（Numerical Stability）

历史无界 geometry 运行中，Ra 可能被更新到负值，随后 forward sensitivity 爆炸并产生 NaN。当前实现对 geometry 和 channel parameters 使用 bounded physical transforms。最新 10 个 BrainCell seeds 的参数均为 finite，且位于定义的物理 bounds 内。

## 11. 限制（Limitations）

- BrainCell 与 Jaxley 的内部 parameterization 不同；
- BrainCell radius 优化的是 `radius_scale`，Jaxley 导出的是 physical radius；
- 分类成功不代表参数被唯一恢复；
- seed/restart 的编号只用于结果对齐，不代表随机过程完全等价；
- 本报告不包含性能 benchmark 结论。
