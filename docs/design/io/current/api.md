# IO API

`braincell.io` 读取 SWC、ASC，保存 Morphology/Branch checkpoint；SWC 写出入口位于
`braincell.io.swc.write_swc`。在线形态检索和缓存单列在 [NeuroMorpho](neuromorpho.md)。
读取结果是 `braincell.Morphology`，几何数据模型见 [Morph API](../../morph/current/api.md)。

## 本地读写示例

以下使用仓库自带 SWC fixture，在临时目录验证写出和 checkpoint；不会访问网络。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import braincell as bc
from braincell.io._testing import FIXTURE_DIR

morpho, report = bc.io.SwcReader().read(FIXTURE_DIR / "three_points_soma.swc", return_report=True)
assert morpho.n_branches > 0
assert not report.has_errors
with TemporaryDirectory() as directory:
    directory = Path(directory)
    swc_path = morpho.to_swc(directory / "copy.swc")
    reread = bc.Morphology.from_swc(swc_path)
    checkpoint = bc.io.save_morpho(morpho, directory / "shape.bcm")
    restored = bc.io.load_morpho(checkpoint)
    assert reread.n_branches == morpho.n_branches
    assert restored == morpho
```

## SWC 读取与检查

```text
SwcReadOptions(standardize_safe_fixes=True, unknown_type_as_custom=True,
               require_root_type_soma=False, mode="neuron")
SwcReader(options=SwcReadOptions())
SwcReader.read(path, *, return_report=False) -> Morphology | (Morphology, SwcReport)
SwcReader.check(path) -> SwcReport
Morphology.from_swc(path, *, options=None, mode=None, return_report=False)
```

path 为 str/PathLike，SWC 数值长度解释为微米。options 默认是每个 reader 独立构造的对象。
mode 为 neuron 或 neuromorpho，决定 soma 重建和连接语义；选择方法及完整几何例子见
[SWC Reader 约束](swc-reader-invariants.md)。便利入口同时传 options 和 mode 时两者必须一致，否则抛出 ValueError；
仅传 mode 时据此创建默认 options。

standardize_safe_fixes 决定是否应用规则允许的修复；unknown_type_as_custom 把未知类型保留为 custom；
require_root_type_soma=True 将非 soma 根作为错误。read 构造新树，return_report=True 同时返回诊断；
check 只返回检查结果。不可读取的文件和失败的格式验证分别通过文件异常或 ValueError 报告。

SwcReport 的 `issues` 保存 level/code/message 及行号，`has_errors` 表示存在错误；
SwcIssue 另带 node_id、fix_message、fix_applied，便于区分发现的问题和实际应用的修复。
重复读取构造独立树，不共享后续 attach 修改。

## ASC 与 NeuroML

```text
AscReader()
AscReader.read(path, return_report=False) -> Morphology | (Morphology, AscReport)
Morphology.from_asc(path, *, return_report=False)
NeuroMlReader()
NeuroMlReader.read(path) -> Morphology
```

ASC 读取 Neurolucida 树，并在 AscReport 中保留诊断、metadata 和 spine 记录。
几何导入与非几何标记的区别见 [ASC reader](../../../../braincell/io/asc/reader.py) 和
[ASC report 类型](../../../../braincell/io/asc/types.py)。轮廓 soma、多树及 spine 细节的剩余工作在
[IO TODO](../TODO.md) 中跟踪。

NeuroMlReader 当前是预留入口，read 直接抛出 NotImplementedError，尚不能导入 NeuroML 文件。

## SWC 写出

```text
braincell.io.swc.write_swc(morpho, path) -> Path
Morphology.to_swc(path) -> Path
```

写出需要完整三维点几何；只有 lengths 的 Branch 无法直接转成 SWC 坐标。
writer 将树展开成 SWC rows，保留分支连接、方向和 soma 附着，文件长度为微米；返回实际输出路径。
已有路径会被写入覆盖。非法几何或不能表示的连接在验证阶段报错；
结构回读用例见 [writer_test.py](../../../../braincell/io/swc/writer_test.py)。

## Checkpoint

```text
save_branch(branch, path) -> Path
load_branch(path) -> Branch
save_morpho(morpho, path) -> Path
load_morpho(path) -> Morphology
Branch.save_checkpoint(path) -> Path
Branch.load_checkpoint(path) -> Branch
Morphology.save_checkpoint(path) -> Path
Morphology.load_checkpoint(path) -> Morphology
```

函数从 braincell.io 导入。`.bcm` 是自包含的形态格式，保存几何、类型、连接和命名信息，
加载返回独立对象。它保存形态，不保存 Cell 电压或 Network 队列。
保存会写入指定文件，父目录和错误文件格式处理见
[checkpoint.py](../../../../braincell/io/checkpoint.py)。格式错误抛出 CheckpointError，
不支持的版本抛出其子类 CheckpointVersionError；文件系统异常保留原异常类型。

完整示例见 [morphology-checkpoint.ipynb](../../../../examples/io/morphology-checkpoint.ipynb)。
