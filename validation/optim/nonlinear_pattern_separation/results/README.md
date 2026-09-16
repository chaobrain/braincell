# Nonlinear Pattern Separation Results

本目录保存 **Nonlinear Pattern Separation Validation / Reproduction Study** 的结果说明。该实验验证 BrainCell 是否能够复现 Jaxley reference implementation 的任务行为，不是运行性能 benchmark；性能测试位于仓库的 `benchmarks/` 目录。

## 从哪里开始

- 完整实验报告：[nonlinear-pattern-separation-summary.md](nonlinear-pattern-separation-summary.md)
- 原始数据说明：[`../artifacts/nonlinear_pattern_separation_2026-09-15/raw/README.md`](../artifacts/nonlinear_pattern_separation_2026-09-15/raw/README.md)
- 图表：[`../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/figures/`](../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/figures/)
- 结构化汇总：[`../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/summaries/`](../artifacts/nonlinear_pattern_separation_2026-09-15/analysis/summaries/)

## 复现入口

BrainCell runner 位于 `../runners/braincell/`。测试命令：

```bash
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q validation/optim/nonlinear_pattern_separation
```
