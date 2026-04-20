# NEURON vs braincell Compare

这个目录用于组织 `NEURON vs braincell` 的真实仿真对比测试。其核心目标是检验两类输入在真实仿真中的精度一致性：

- 从 `mod` 文件转入后的机制模型
- 导入形态学后的多房室 / cable 模型

同时，这里也用于在不同刺激与环境条件下做综合验证，例如：

- `DC`:直流电
- `AC`:交流电
- `temperature`:温度

因此这里按不同模板家族拆分为多个子文件夹，分别处理受限的单通道对比与多房室 cable 对比。

## Typical Flow

通常按下面的流程组织每个家族：

1. 准备输入配置。
2. 构建 `NEURON` 与 `braincell` 的仿真入口并执行 compare。
3. 在 `artifacts/` 与 notebook/workflow 中查看结果、汇总误差并做调试分析。

当前目录已按模板家族重组，每个家族单独维护自己的说明、样例、模板、结果与 notebook。

## Families

- `cable/`
- `channel_no_conc/`

## Common Layout

当前两个家族都保留自己的 `README.md`、实现模板、输入配置、测试与工作流入口。

目前活跃维护的家族：

- `cable`
- `channel_no_conc`
