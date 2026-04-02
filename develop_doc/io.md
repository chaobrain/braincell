# IO 设计

当前 `braincell` 将 IO 明确放在顶层 `braincell/io/`，与 `morpho` 同级。

这样分层的原因是：

- `morpho` 负责核心形态模型：`Branch`、`Morpho`、`MorphoBranch`
- `io` 负责外部格式适配：SWC、ASC、NeuroML2
- 输入格式不应反向决定形态核心模块的目录结构

## 当前目录

- `braincell/io/__init__.py`
  - 顶层导出 `SwcReader`、`AscReader`、`NeuroMlReader` 及 SWC report 类型
- `braincell/io/swc/`
  - `reader.py`：`SwcReader`
  - `rules.py`：单一 `check + correct` rulebook
  - `soma.py`：soma 分类与 contour/cylinder 判定
  - `types.py`：`SwcReadOptions`、`SwcIssue`、`SwcReport` 与内部 SWC 数据结构
- `braincell/io/asc/`
  - `reader.py`：`AscReader` 占位
- `braincell/io/neuroml2/`
  - `reader.py`：`NeuroMlReader` 占位

## 入口关系

推荐入口：

```python
from braincell import Morpho

tree = Morpho.from_swc("file.swc")
tree, report = Morpho.from_swc("file.swc", return_report=True)
```

或者直接用 reader：

```python
from braincell.io.swc import SwcReader

reader = SwcReader()
tree = reader.read("file.swc")
report = reader.check("file.swc")
```

边界约定：

- `Morpho.from_swc(...)` 是便捷入口
- 真正的格式适配实现放在 `braincell/io/...`
- `check(...)` 只返回 `SwcReport`，不构建 `Morpho`
- `read(..., return_report=True)` 返回 `(Morpho, SwcReport)`

## SWC 当前状态

`SWC` 是目前唯一已落地的格式。

当前已实现：

- 标准 7 列 SWC 解析
- 单一 rulebook 的 `check + correct`
- `SwcReport` 结构化报告
- 单点 soma / 特殊三点 soma / 普通多点 soma / contour soma 处理
  - 特殊三点 soma 当前按“第一个点是中心点，后两个点挂在中心两侧”识别
- 连续 degree-2 链压缩为 `Branch`
- `tests/morpho_files/*.swc` 的真实 fixture smoke 测试

当前主 pipeline：

1. parse raw rows
2. rulebook `check + correct`
3. build graph index
4. soma classification
5. 连续 degree-2 链压缩为 `Branch`
6. build `Morpho`

当前需要注意的一点：

- `type=1` 的 branched soma graph 语义仍未最终定稿
- 目前实现倾向于先把连通的 soma 组件视为一个 soma 组件处理
- 这部分后续仍可能继续调整

当前未完成：

- `AscReader`
- `NeuroMlReader`
- 更完整的 SWC abnormal compartments 规则
- 更系统的 ASC / NeuroML2 格式设计文档

## 测试与 fixture

- `tests/test_io_swc.py`
  - synthetic SWC case
  - rulebook、report、soma 逻辑
- `tests/test_io_real_files.py`
  - `tests/morpho_files/*.swc` / `*.asc`
  - 当前主要做 smoke + 基本不变量

真实 morphology fixture 目前放在：

- `tests/morpho_files/grc.swc`
- `tests/morpho_files/io.swc`
- `tests/morpho_files/bc.swc`
- `tests/morpho_files/goc.asc`
- `tests/morpho_files/pc.asc`

## 下一步建议

如果继续做 IO，优先级建议是：

1. 在 `braincell/io/swc/rules.py` 里继续补 SWC 规则
2. 为 `AscReader` 设计与 `SwcReader` 对齐的分层结构
3. 明确 `NeuroMlReader` 的最小导入范围
