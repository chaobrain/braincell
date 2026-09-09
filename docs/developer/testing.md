---
myst:
  heading_anchors: 2
---

# 测试指南

先运行修改模块的测试，再按影响范围检查集成行为。BrainCell 使用 pytest，
测试文件与所测源码相邻；测试收集配置在
[pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml)。
以下命令在完成 [开发安装](contributing.md#配置开发环境) 后，从仓库根目录运行。

## 运行相关测试

以 SWC reader 为例，分别运行一个模块、查看其收集结果，或运行整个 IO 包：

```bash
python -m pytest braincell/io/swc/reader_test.py -q
python -m pytest braincell/io/swc/reader_test.py --collect-only -q
python -m pytest braincell/io/ -q
```

涉及多个模块的共享行为时，扩大到相应包；完整核心测试命令为：

```bash
python -m pytest braincell/ -q
```

测试会使用根 conftest 配置的 CPU JAX 和无窗口 Matplotlib 环境。
GPU 性能或大型数值对照需要按对应工作流另外运行，不能由这些单元测试推断结果。

## 编写复现与回归测试

错误修复先增加一个会失败的最小复现，修复后保留它作为回归测试。
新功能选择有可观察结果的场景：返回形状、单位、状态变化、事件时序或数值误差。
共享行为变更还需验证使用它的 Cell 或 Network 路径。

新测试使用与源码对应的 `*_test.py` 文件。拆分规则、包级导出检查、共享辅助代码和依赖跳过方式见
[仓库测试规则](https://github.com/chaobrain/braincell/blob/main/AGENTS.md#testing)。

## 复用形态 fixture

仓库形态数据位于 `data/morphology/`，通过共享常量获取路径：

```python
import braincell as bc
from braincell.io._testing import FIXTURE_DIR

morpho, report = bc.io.SwcReader().read(
    FIXTURE_DIR / "three_points_soma.swc", return_report=True,
)
assert morpho.n_branches > 0
assert not report.has_errors
```

同一辅助模块还提供 `VALID_SWC_FIXTURES`、`ALLOWED_TYPES`。HTTP 等外部服务使用已有测试替身，
例如 `braincell/io/neuromorpho/_testing.py`；这样普通回归测试可以离线运行。

## 数值结果如何验收

先声明参照和容差，再比较结果。固定模型参数、单位、温度、初态、solver 与 dt；
用解析解或可信参照检查动力学，改变 dt 检查收敛，区分实现错误与离散误差。
需要重复推进模型时使用仓库约定的 brainstate 编译循环。
模型扩展的具体检查见 [扩展指南](extending.md)。

PR 中列出实际执行的命令、通过或跳过情况，以及结果依据。
文档中的可运行示例单独执行；Sphinx 网站构建当前关闭 notebook 执行。
