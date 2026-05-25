# Documentation & Examples Reorganization

**Date:** 2026-05-25
**Status:** Approved for direct execution (user opted to skip implementation plan)
**Branch base:** `origin/main` at 784c8d3 (merged into `worktree-zesty-yawning-taco`)

## Goal

Reorganize `braincell` documentation and example assets along three axes:

1. Group documentation by cell-model class — `docs/single_compartment/` and `docs/multi_compartment/`.
2. Promote the runnable multi-compartment tutorials from `examples/multi_compartment/` into `docs/multi_compartment/` so Sphinx builds them.
3. Consolidate data fixtures under a single top-level `data/` directory, subdivided by data kind.
4. Drop Chinese (`*-zh.ipynb`) translations for single-compartment material; rename the surviving English files to drop the redundant `-en` suffix.

## Non-goals

- No restructuring of `examples/neuron_compare/` or `examples/convert_mod/` (only minimal import-line fixes inside `neuron_compare/cable/` to follow the data move).
- No new public helper API for data paths — notebooks/scripts/tests resolve `data/` via their own relative paths.
- No content rewrites to notebooks beyond the path references they contain.

## Target layout

```
braincell/                          # source (unchanged except 4 hardcoded test paths)
data/                               # NEW — repo-root data directory
├── morphology/
│   ├── generic/                    # ← examples/multi_compartment/morpho_files/*
│   └── cerebellum/                 # ← examples/multi_compartment/morpho_files/Cerebellum_morph/
├── neuron_traces/                  # ← examples/single_compartment/neuron_data/
├── neuromorpho/                    # cache target referenced by neuromorpho.ipynb (empty initially; gitignored)
└── vis_outputs/                    # ← examples/multi_compartment/data/plot/*.png
                                    #   (publication.pdf was deleted upstream in #94)

docs/
├── single_compartment/             # NEW
│   ├── index.rst                   # NEW landing page
│   ├── quickstart/                 # ← docs/quickstart/  (English only)
│   ├── tutorial/                   # ← docs/tutorial/    (English only)
│   └── advanced_tutorial/          # ← docs/advanced_tutorial/  (English only)
│       └── examples/               # sc02..sc05.ipynb (English content, suffix dropped)
├── multi_compartment/              # NEW
│   ├── index.rst                   # NEW landing page; toctree controls ordering
│   ├── morphology.ipynb            # ← examples/multi_compartment/1.morphology.ipynb
│   ├── filter.ipynb                # ← 2.filter.ipynb
│   ├── mech.ipynb                  # ← 3.mech.ipynb
│   ├── cell.ipynb                  # ← 4.cell.ipynb
│   ├── vis.ipynb                   # ← 5.vis.ipynb
│   ├── channel.ipynb               # ← 6.channel.ipynb
│   └── ion.ipynb                   # ← 7.ion.ipynb
├── apis/                           # unchanged
├── _static/, _templates/           # unchanged
├── conf.py                         # unchanged
├── index.rst                       # toctree updated
├── cell.md, interface-map.md,
│   module-dependency-map.md,
│   auto_generater.py, Makefile,
│   make.bat                        # unchanged

examples/
├── single_compartment/             # only SC0*.py scripts; neuron_data/ removed
│   └── SC01..SC07.py               # paths updated to `data/neuron_traces/...`
├── multi_compartment/              # scratch / validation area
│   ├── neuron_diff.{ipynb,py}, neuron_diff_test.py
│   ├── neuromorpho.ipynb, neuron_test.ipynb
│   ├── morphology-checkpoint.ipynb, quad.ipynb
│   ├── vis2d.ipynb, vis_old.ipynb
│   └── (morpho_files/, data/ removed — content in /data/)
├── convert_mod/                    # untouched
└── neuron_compare/                 # untouched except 2 import-path fixes in cable/
```

## File-level changes

### Deletes (11 files)

```
docs/quickstart/concepts-zh.ipynb
docs/tutorial/cell-zh.ipynb
docs/tutorial/channel-zh.ipynb
docs/tutorial/ion-zh.ipynb
docs/advanced_tutorial/rationale-zh.ipynb
docs/advanced_tutorial/differential_equation-zh.ipynb
docs/advanced_tutorial/more-zh.ipynb
docs/advanced_tutorial/examples/sc02.ipynb     ← unsuffixed = zh (sc02-en.ipynb is the en sibling)
docs/advanced_tutorial/examples/sc03.ipynb     ← same
docs/advanced_tutorial/examples/sc04.ipynb     ← same
docs/advanced_tutorial/examples/sc05.ipynb     ← same
```

### Directory moves (`git mv`)

```
docs/quickstart/                                    → docs/single_compartment/quickstart/
docs/tutorial/                                      → docs/single_compartment/tutorial/
docs/advanced_tutorial/                             → docs/single_compartment/advanced_tutorial/
examples/multi_compartment/morpho_files/Cerebellum_morph/ → data/morphology/cerebellum/
examples/multi_compartment/morpho_files/            → data/morphology/generic/   (after the Cerebellum subdir is hoisted out)
examples/multi_compartment/data/plot/               → data/vis_outputs/
examples/single_compartment/neuron_data/            → data/neuron_traces/
```

### File renames (drop `-en` suffix after deletes)

| Old | New |
|---|---|
| `docs/single_compartment/quickstart/concepts-en.ipynb` | `docs/single_compartment/quickstart/concepts.ipynb` |
| `docs/single_compartment/tutorial/cell-en.ipynb` | `docs/single_compartment/tutorial/cell.ipynb` |
| `docs/single_compartment/tutorial/channel-en.ipynb` | `docs/single_compartment/tutorial/channel.ipynb` |
| `docs/single_compartment/tutorial/ion-en.ipynb` | `docs/single_compartment/tutorial/ion.ipynb` |
| `docs/single_compartment/advanced_tutorial/rationale-en.ipynb` | `…/rationale.ipynb` |
| `docs/single_compartment/advanced_tutorial/differential_equation-en.ipynb` | `…/differential_equation.ipynb` |
| `docs/single_compartment/advanced_tutorial/more-en.ipynb` | `…/more.ipynb` |
| `docs/single_compartment/advanced_tutorial/examples/sc02-en.ipynb` | `…/sc02.ipynb` |
| `docs/single_compartment/advanced_tutorial/examples/sc03-en.ipynb` | `…/sc03.ipynb` |
| `docs/single_compartment/advanced_tutorial/examples/sc04-en.ipynb` | `…/sc04.ipynb` |
| `docs/single_compartment/advanced_tutorial/examples/sc05-en.ipynb` | `…/sc05.ipynb` |
| `examples/multi_compartment/1.morphology.ipynb` | `docs/multi_compartment/morphology.ipynb` |
| `examples/multi_compartment/2.filter.ipynb` | `docs/multi_compartment/filter.ipynb` |
| `examples/multi_compartment/3.mech.ipynb` | `docs/multi_compartment/mech.ipynb` |
| `examples/multi_compartment/4.cell.ipynb` | `docs/multi_compartment/cell.ipynb` |
| `examples/multi_compartment/5.vis.ipynb` | `docs/multi_compartment/vis.ipynb` |
| `examples/multi_compartment/6.channel.ipynb` | `docs/multi_compartment/channel.ipynb` |
| `examples/multi_compartment/7.ion.ipynb` | `docs/multi_compartment/ion.ipynb` |

## Sphinx wiring

### `docs/index.rst`

Replace the four current toctrees (Quickstart / Tutorials / Advanced Tutorials / API Documentation) with two model-class sections plus the API toctree:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Single-Compartment Modeling
   :hidden:

   single_compartment/index

.. toctree::
   :maxdepth: 2
   :caption: Multi-Compartment Modeling
   :hidden:

   multi_compartment/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Documentation

   apis/braincell.rst
   apis/morphology.rst
   apis/braincell.neuron.rst
   apis/braincell.synapse.rst
   apis/braincell.ion.rst
   apis/braincell.channel.rst
   apis/integration.rst
   apis/vis.rst
   apis/changelog.md
```

### `docs/single_compartment/index.rst` (new)

```rst
Single-Compartment Modeling
===========================

Tutorials and worked examples for single-compartment Hodgkin–Huxley neurons.

.. toctree::
   :maxdepth: 1
   :caption: Quickstart

   quickstart/concepts

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/channel
   tutorial/ion
   tutorial/cell

.. toctree::
   :maxdepth: 1
   :caption: Advanced Tutorials

   advanced_tutorial/rationale
   advanced_tutorial/differential_equation
   advanced_tutorial/more
   advanced_tutorial/examples
```

### `docs/single_compartment/advanced_tutorial/examples.rst`

Reduce the toctree from 8 entries (4 zh + 4 en) to 4:

```rst
Examples
========

.. toctree::
   :maxdepth: 1

   examples/sc02
   examples/sc03
   examples/sc04
   examples/sc05
```

### `docs/multi_compartment/index.rst` (new)

```rst
Multi-Compartment Modeling
==========================

Step-by-step tutorials for building, simulating, and visualizing morphologically detailed cells. Read in order.

.. toctree::
   :maxdepth: 1

   morphology
   filter
   mech
   cell
   vis
   channel
   ion
```

## Path strategy — relative resolution from each file

No new helper API. Each consumer computes its own path to `data/`.

### `braincell/io/` tests (4 files)

Each `parents[N]` currently resolves to repo-root. Since `data/` also lives at repo-root, the `parents` index is unchanged — only the tail (`examples/multi_compartment/morpho_files` → `data/morphology/generic`) flips.

| File | Line | Change |
|---|---|---|
| `braincell/io/asc/test.py` | 33 | `FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "morphology" / "generic"` |
| `braincell/io/checkpoint_test.py` | 47 | `FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "morphology" / "generic"` |
| `braincell/io/swc/test.py` | 30 | `FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "morphology" / "generic"` |
| `braincell/io/neuromorpho/_testing.py` | 38 (docstring), 40–46 | `FIXTURE_SWC = Path(__file__).resolve().parents[3] / "data" / "morphology" / "generic" / "three_points_soma.swc"` plus docstring text update |

### `examples/single_compartment/SC01_fitting_a_hh_neuron.py`

```python
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "neuron_traces"
df_inp_traces = pd.read_csv(DATA_DIR / 'input_traces_hh.csv')
df_out_traces = pd.read_csv(DATA_DIR / 'output_traces_hh.csv')
```

### `docs/multi_compartment/*.ipynb` (promoted tutorials)

Each notebook gets a small setup cell at the top:

```python
from pathlib import Path
DATA = Path.cwd().resolve().parents[1] / "data"   # docs/multi_compartment/ → repo root → /data
```

Then path literals are rewritten:

| Notebook | Old | New |
|---|---|---|
| `morphology.ipynb` | `"./morpho_files/example_tree.swc"` | `str(DATA / "morphology" / "generic" / "example_tree.swc")` |
| `morphology.ipynb` | `"./morpho_files/goc.asc"` | `str(DATA / "morphology" / "generic" / "goc.asc")` |
| `filter.ipynb` | `repo_root / 'examples' / 'multi_compartment' / 'morpho_files'` | `DATA / "morphology" / "generic"` |
| `cell.ipynb` (markdown + code) | `"./morpho_files/example_tree.swc"` | `DATA / "morphology" / "generic" / "example_tree.swc"` |
| `vis.ipynb` | `"./morpho_files/Cerebellum_morph/GoC.asc"` | `DATA / "morphology" / "cerebellum" / "GoC.asc"` |

### `examples/multi_compartment/*.ipynb` (scratch notebooks)

Same `DATA = Path.cwd().resolve().parents[1] / "data"` pattern (parents[1] because `examples/multi_compartment/` is also two levels from repo root, same as `docs/multi_compartment/`).

Affected notebooks: `morphology-checkpoint.ipynb`, `neuromorpho.ipynb`, `neuron_diff.ipynb`, `neuron_test.ipynb`, `vis_old.ipynb`.
- `neuromorpho.ipynb` & `neuron_diff.ipynb` (markdown): replace `examples/multi_compartment/data/neuromorpho/` references with `data/neuromorpho/`.
- `morphology-checkpoint.ipynb`: replace `Path("morpho_files") / "branched_dend.swc"` with `DATA / "morphology" / "generic" / "branched_dend.swc"`.
- `neuron_test.ipynb` line 86: replace the absolute `/home/swl/braincell/...` literal with the relative `DATA / "morphology" / "generic" / "example_tree.swc"` form (drive-by fix — it was already broken on any machine other than the original).
- `vis_old.ipynb` output cells (`out_dir = Path('vis_outputs')`): change to `DATA / "vis_outputs"` so any future re-run lands in the new location.

### `examples/multi_compartment/neuron_diff_test.py`

Lines 259/269/278 (`Path(__file__).resolve().parent / "morpho_files" / ...`) → `Path(__file__).resolve().parents[2] / "data" / "morphology" / ...` (cerebellum subdir for two of them).

### `examples/neuron_compare/cable/` (scope creep — minimal)

Two lines reference `multi_compartment/morpho_files` from inside `neuron_compare/`. Update them; do not restructure anything else inside `neuron_compare/`.

| File | Line | Change |
|---|---|---|
| `examples/neuron_compare/cable/engine/fixtures.py` | 10 | `MORPHO_FILES = Path(__file__).resolve().parents[4] / "data" / "morphology" / "generic"` |
| `examples/neuron_compare/cable/tests/_helpers.py` | 14 | same |

## .gitignore additions

```
# NeuroMorpho cache (populated by neuromorpho.ipynb at runtime)
/data/neuromorpho/
```

The other `data/` subdirs are checked-in fixtures.

## Execution order

Phase 0 — already done: `git merge --ff-only origin/main` (commit 784c8d3).

Phase 1 — File operations (no path edits yet)
1. Delete the 11 zh notebooks.
2. `git mv` directories into their new homes (single_compartment, multi_compartment, data subtrees).
3. `git mv` rename `-en.ipynb` → `.ipynb` on the 11 survivors.
4. `git mv` rename `examples/multi_compartment/N.<topic>.ipynb` → `docs/multi_compartment/<topic>.ipynb`.

Commit checkpoint: "docs: reorganize by single/multi compartment; consolidate data fixtures (paths not yet updated)".

Phase 2 — Path updates
5. Update the 4 `braincell/io/*` test files.
6. Update `examples/single_compartment/SC01_fitting_a_hh_neuron.py`.
7. Add `DATA = …` setup cell to each of the 7 promoted `docs/multi_compartment/*.ipynb` and rewrite path literals.
8. Update the 5 scratch notebooks in `examples/multi_compartment/`.
9. Update `examples/multi_compartment/neuron_diff_test.py` (3 lines).
10. Update `examples/neuron_compare/cable/engine/fixtures.py` + `tests/_helpers.py` (2 lines).

Commit checkpoint: "docs: update data fixture paths after consolidation".

Phase 3 — Sphinx wiring
11. Write `docs/single_compartment/index.rst`.
12. Write `docs/multi_compartment/index.rst`.
13. Update `docs/single_compartment/advanced_tutorial/examples.rst` toctree (8 → 4 entries).
14. Rewrite `docs/index.rst` toctrees.
15. Add `.gitignore` entry for `/data/neuromorpho/`.

Commit checkpoint: "docs: rewire Sphinx toctree for new layout".

Phase 4 — Verification
16. Run `pytest braincell/io/asc/test.py braincell/io/swc/test.py braincell/io/checkpoint_test.py braincell/io/neuromorpho/` to confirm the 4 hardcoded paths still resolve.
17. Smoke test SC01 (`python examples/single_compartment/SC01_fitting_a_hh_neuron.py` — kept short enough to verify the CSV read works; not a full convergence run).
18. Best-effort Sphinx build (`cd docs && make html`); fix any broken cross-references that surface.

## Risks & mitigations

- **`braincell/io` test paths drift in `parents[N]`.** Mitigation: each test recomputed individually; verified by running pytest in Phase 4.
- **Notebook JSON edits fragile.** Mitigation: prefer `git mv` for renames; for path rewrites, use targeted `Edit` calls with exact-match strings rather than bulk `sed` on `.ipynb` (line endings + JSON escaping). Verify notebooks still load as JSON after edits (`python -c "import json; json.load(open(path))"`).
- **`neuron_compare/cable/` left in a half-updated state if step 10 is skipped.** Mitigation: explicit checkbox in Phase 2; the two lines are the only known references and both update to the same form.
- **Sphinx build cannot run in CI here.** Mitigation: best-effort local build in Phase 4; landing PR will exercise CI docs job. `examples.rst` reduction and `index.rst` toctree are the two highest-risk Sphinx edits — review by eye before commit.
- **NeuroMorpho cache previously lived under `examples/multi_compartment/data/neuromorpho/`** (per `neuromorpho.ipynb` docs) but no actual cache exists in the repo. Moving to `/data/neuromorpho/` is forward-only; nothing to migrate today.

## Open questions for execution

None — every ambiguity called out in brainstorming has an explicit answer.
