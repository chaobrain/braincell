# NEURON vs braincell Compare

这个目录用于组织 `NEURON vs braincell` 的真实仿真对比测试。其核心目标是检验两类输入在真实仿真中的精度一致性：

- 从 `mod` 文件转入后的机制模型
- 导入形态学后的多房室 / cable 模型

同时，这里也用于在不同刺激与环境条件下做综合验证，例如：

- `DC`:直流电
- `AC`:交流电
- `temperature`:温度

因此这里按不同模板家族拆分为多个子文件夹，分别处理 `single-compartment` / `multi-compartment`、`channel` / `ion` / `cable` 等不同对比场景。

## Typical Flow

通常按下面的流程组织每个家族：

1. 在 `specs/` 中定义该家族要验证的对象、边界和比较目标。
2. 在 `cases/` 中准备单 case 或 sweep 输入，配置 morphology、mechanism、stimulus、temperature 等条件。
3. 在 `templates/` 中分别构建 `NEURON` 与 `braincell` 的仿真入口并执行 compare。
4. 在 `artifacts/` 与 `notebooks/` 中查看结果、汇总误差并做调试分析。

当前目录已按模板家族重组，每个家族单独维护自己的说明、样例、模板、结果与 notebook。

## Families

- `MC_cable/`
- `MC_ion/`
- `MC_channel/`
- `SC_ion/`
- `SC_channel/`

## Common Layout

每个家族目录统一包含：

- `README.md`
- `specs/`
- `cases/`
- `templates/`
- `artifacts/`
- `notebooks/`

目前已具备较完整模板与产物的家族：

- `MC_cable`
- `SC_channel`

其余家族先保留骨架、spec、样例和入口模板，后续逐步补齐实现。
