# DCN mod sources

## Source

- ModelDB: https://modeldb.science/185513
- GitHub: https://github.com/ModelDBRepository/185513

## Reference

- Sudhakar SK, Torben-Nielsen B, De Schutter E. (2015).
- Cerebellar Nuclear Neurons Use Time and Rate Coding to Transmit Purkinje Neuron Pauses.
- PLoS Computational Biology, 11.

## Local note

该目录中的 `.mod` 文件基于上述公开模型整理而来。
在本地整理过程中，这些文件可能包含轻微修改、提取或重分类，但整体来源与上述模型一致。

## Folder content

本目录保存 DCN `.mod` 源文件，按 `channel/ion/synapse/other` 子目录分类组织。

`morphology/` 保存从源 `DCN_mor.hoc` 抽取出的形态。需要重新解析源 HOC
时，通过 `DCN_SOURCE_HOC` 环境变量或 `make_dcn_swc.py --source-hoc` 显式传入路径：

- `morphology/dcn_native.py`
  - BrainCell DCN 复现的主路径。
  - 直接解析源 HOC 的 `create/connect/pt3dadd/SectionList`，构建 BrainCell `Morphology`。
  - 分支名直接带生理区域前缀，例如 `proxDend__p1b2__1`、`distDend__p1b2b2__0`、`axIniSeg__axIS__0`。
  - 后续通道插入应按这些区域前缀选择 branch，不需要 `DCN_section_map.csv`。
- `morphology/dcn_cell.py`
  - 基于 native morphology 构建 BrainCell `Cell`，按 `DCN_template_1.hoc` 的区域逻辑 paint cable、ion、channel。
  - `TNC` 是 `gbar * (v - eTNC)` 形式的 ohmic current，当前直接用 `IL` 机制、`E=-35 mV` 和 `TNC_*` 实例名复现。
- `morphology/DCN.swc`
  - 面向 NEURON round-trip / 外部可视化的 legacy SWC。
  - 使用交替 SWC type code 阻止 `Import3d_SWC_read` 把同类型连续链合并成更少 section。
  - 不要只用 SWC type 推断原始生理区域。
- `morphology/DCN_section_map.csv`
  - SWC 方案的 legacy sidecar，保存 SWC node、导入后 section index、原 HOC section 名、父 section、`parent_x`、`proxDend/distDend/axHillock/axIniSeg/axNode` 区域和原始 pt3d 端点。
  - BrainCell 复现路径不再依赖这个表。
- `morphology/make_dcn_swc.py`
  - 可重复生成上述文件，并运行 NEURON round-trip 几何校验。
- `morphology/DCN_morphology_summary.json`
  - 最近一次生成和校验摘要。

当前校验结果：原 HOC 与 `DCN.swc` 经 `Import3d_SWC_read + Import3d_GUI.instantiate`
导入后均为 517 个 section、1034 个 total pt3d、每个 section 2 个 pt3d，
NEURON 计算的总长度一致。`Import3d_SWC_read.err == 1` 是预期现象，因为
SWC 中有意使用 type 变化制造 section 边界；这是为了保留原 HOC 的 section
数量，而不是表示真实解剖 type 差异。
