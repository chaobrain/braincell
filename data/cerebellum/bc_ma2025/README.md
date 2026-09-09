# BC mod sources

## Source

- ModelDB: https://modeldb.science/2018018
- GitHub: https://github.com/ModelDBRepository/2018018

## Reference

- Masoli S, Rizza MF, Soda T, Sanchez-Ponce D, Munoz A, Prestori F, D'Angelo E. (2025).
- Cerebellar basket cell filtering of Purkinje cell responses elicited by low frequency parallel fibre transmission.
- Scientific Reports, 15.

## Local note

该目录中的 `.mod` 文件基于上述公开模型整理而来。
在本地整理过程中，这些文件可能包含轻微修改、提取或重分类，但整体来源与上述模型一致。

## Folder content

机制源文件在 `mechanisms/` 下按 channel、ion、synapse、other 分类；形态在 `morphology/`。

## Use and variants

[Whole-cell comparison](../../../validation/neuron/cell/bc_ma2025/) uses the morphology and mechanisms in this bundle.
Parameter tables are under `parameters/` when supplied. Original reference files and their provenance hashes are under `reference/` when supplied. Source notices are retained; local variants keep their distinct filenames.

The IO fixture and whole-cell morphology were byte-identical; both now reference the single file in `morphology/`.

Compile from the repository root with `python -m validation.neuron._mechanisms bc_ma2025`. Outputs go to validation artifacts.
