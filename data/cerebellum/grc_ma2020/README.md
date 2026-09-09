# GrC mod sources

## Source

- ModelDB: https://modeldb.science/265584
- GitHub: https://github.com/ModelDBRepository/265584

## Reference

- Masoli S, Tognolina M, Laforenza U, Moccia F, D'Angelo E. (2020).
- Parameter tuning differentiates granule cell subtypes enriching transmission properties at the cerebellum input stage.
- Communications Biology, 3.

## Local note

该目录中的 `.mod` 文件基于上述公开模型整理而来。
在本地整理过程中，这些文件可能包含轻微修改、提取或重分类，但整体来源与上述模型一致。

## Folder content

机制源文件在 `mechanisms/` 下按 channel、ion、synapse、other 分类；形态在 `morphology/`。

## Use and variants

[Whole-cell comparison](../../../validation/neuron/cell/grc_ma2020/) uses the morphology and mechanisms in this bundle.
Parameter tables are under `parameters/` when supplied. Original reference files and their provenance hashes are under `reference/` when supplied. Source notices are retained; local variants keep their distinct filenames.

`morphology/io_fixture/` preserves the different morphology previously used by IO tests. It is not interchangeable with the whole-cell comparison morphology.

Compile from the repository root with `python -m validation.neuron._mechanisms grc_ma2020`. Outputs go to validation artifacts.
