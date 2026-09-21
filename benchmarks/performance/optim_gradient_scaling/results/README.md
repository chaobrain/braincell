# Gradient Performance Results

本目录按实验层级组织结果，而不是按脚本文件类型组织。正式主实验是 HH crossover；complexity 用于拆解复杂度；diagnostics 用于 profiling 和候选优化；historical 保存独立 workload。

## 主实验：HH crossover

- [H200 优化前](hh_crossover/before/hh-crossover-h200-20260910.md)
- [H200 优化后](hh_crossover/after/optimized.md)
- [H200 优化前后对比](hh_crossover/comparison.md)
- [A100 scaling 历史结果](hh_crossover/a100/scaling.md)
图表保存在对应 artifact 的 `analysis/figures/`，由统一 plot 函数离线生成。

## Supporting experiment：complexity decomposition

`controlled_complexity` 通过 synthetic state/parameter、HH state 和 HH parameter grouping workload 拆解 BPTT/RTRL 的复杂度。当前没有独立的 Git 结果页；新测量应写入 `complexity/`，并保持与 HH crossover 分开。

## Diagnostics

- [RTRL profiling](diagnostics/rtrl-profile-20260912.md)
- [RTRL speed baseline](diagnostics/rtrl-speed-baseline-20260912.md)
- [JVP engineering comparison](diagnostics/rtrl-jvp-engineering-20260912.md)
- [Candidate screening](diagnostics/rtrl-candidate-screening-20260912.md)
- [Speed exploration review](diagnostics/rtrl-speed-exploration-review-20260914.md)
- [Materialization compile investigation](diagnostics/compile-materialization-investigation-20260910.md)

## Historical independent workloads

- [Synapse network CPU](historical/synapse-network-cpu.md)
