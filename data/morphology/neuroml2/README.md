# NeuroML2 morphology fixtures

Cells read by `braincell/io/neuroml2/reader_test.py` and by
`docs/file_formats/neuroml2.ipynb`. Every file holds one `<cell>` except
`two_cells.nml`.

| File | Source | What it exercises |
|---|---|---|
| `minimal.cell.nml` | written for BrainCell | a cable group cut by a child at `fractionAlong="0.25"`, a diameter jump at a merged joint, children at `fractionAlong="0"` and at the soma midpoint, an omitted `<proximal>`, a `subTree` group |
| `branched.cell.nml` | written for BrainCell | two cuts in one cable group, a child at the start of a segment that is not the first of its branch, two children at one distal end, a type group that includes a cable group, `path` and `subTree` groups |
| `sphere_soma.cell.nml` | written for BrainCell | a spherical soma (proximal and distal at the same point) with two children |
| `two_cells.nml` | written for BrainCell | two cells: `a` refers to a top-level `<morphology>`, `b` has no type groups |
| `pyr_4_sym.cell.nml` | libNeuroML, see below | a published 9-segment pyramidal cell with 9 cable groups, channel `<include>`s and `<biophysicalProperties>` |

## pyr_4_sym.cell.nml

Copied unchanged from libNeuroML (https://github.com/NeuralEnsemble/libNeuroML)
at commit `29e74c377243482622dfdd79257b58f2bbeee799`, path
`neuroml/examples/test_files/pyr_4_sym.cell.nml`, retrieved on 2026-11-11.

| File | SHA-256 |
|---|---|
| `pyr_4_sym.cell.nml` | `bec845120435ec350df09402cd0e86636738d0086f805209a8c7fcea3a3bd35c` |
| `LICENSE-libNeuroML` | `cd1a1dc13535eb9eb92d4f825d1a502ee35887d5ceb4a7168e764ae2130d4809` |

libNeuroML is distributed under the BSD 3-Clause License. `LICENSE-libNeuroML`
is libNeuroML's `LICENSE` file at the same commit, copied unchanged; its
copyright notice and conditions apply to `pyr_4_sym.cell.nml`.

The `<include>` elements name channel files that are not copied here. The
reader does not follow includes, so they do not affect the morphology.
