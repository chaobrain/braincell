# Nonlinear pattern separation validation

本实验比较 BrainCell 与 Jaxley 在 12 CV、72 个可训练物理参数上的 nonlinear pattern 学习结果。

## 目录

- `runners/braincell/`：BrainCell 实验和训练脚本；测试文件与脚本放在一起。
- `raw/jaxley/`：Jaxley reference raw data；本仓库不运行 Jaxley runner。
- `plotting/`：跨框架结果绘图。
- `artifacts/nonlinear_pattern_separation_2026-09-15/raw/`：本次实验的 BrainCell 与 Jaxley 原始数据。
- `artifacts/nonlinear_pattern_separation_2026-09-15/analysis/`：图表、JSON 汇总和分析说明。

## 最新结果

综合对比图位于 `artifacts/nonlinear_pattern_separation_2026-09-15/analysis/comparison/`。
原始数据结构和来源见 `artifacts/nonlinear_pattern_separation_2026-09-15/raw/README.md`。

## 验证

```bash
JAX_PLATFORM_NAME=cpu JAX_ENABLE_X64=1 python -m pytest -q validation/optim/nonlinear_pattern_separation
```
