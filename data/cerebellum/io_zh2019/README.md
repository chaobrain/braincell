# IO mod sources

## Source

- ModelDB: https://modeldb.science/257028?tab=1
- GitHub: https://github.com/ModelDBRepository/257028

## Reference

- Zhang X, Santaniello S. (2019).
- Role of cerebellar GABAergic dysfunctions in the origins of essential tremor.
- Proceedings of the National Academy of Sciences.

## Local note

该目录中的 `.mod` 文件基于上述公开模型中的 IO2019 子模块整理而来。
在本地整理过程中，这些文件可能包含轻微修改、提取或重分类，因此这里将其视为“基于/提取自”该模型的 IO 机制集合，而不强行声明为与原始发布版本完全一致。

## Folder content

机制源文件在 `mechanisms/` 下按 channel、ion、synapse、other 分类；形态在 `morphology/`。

## Use and variants

[Whole-cell comparison](../../../validation/neuron/cell/io_zh2019/) uses the morphology and mechanisms in this bundle.
Parameter tables are under `parameters/` when supplied. Original reference files and their provenance hashes are under `reference/` when supplied. Source notices are retained; local variants keep their distinct filenames.

`morphology/io_fixture/` preserves the different morphology previously used by IO tests. It is not interchangeable with the whole-cell comparison morphology.

Compile from the repository root with `python -m validation.neuron._mechanisms io_zh2019`. Outputs go to validation artifacts.
