# Move the `Cell` visualization entry points into `braincell.vis`

## Goal

`braincell.Cell` must stop exposing visualization API. The four public
methods `Cell.vis_topology`, `Cell.vis_node`, `Cell.vis_cv`, and
`Cell.vis_branch` are **deleted**; a single free function
`braincell.vis.plot_cell_topology(cell, *, level=...)` replaces them.

This is a deliberate breaking change. No deprecation shim, no forwarding
method. `braincell.MultiCompartment` is the same class object as `Cell`
(`cell.py:3713`), so the removal applies to both names.

The `Morphology.vis2d/vis3d` and `Branch.vis2d/vis3d` methods are **out of
scope** — the request named `braincell.Cell` only.

## Why now

The four methods are the sole deviation from the package's own layering
rule. Every other model object delegates: `Morphology.vis3d` is a 17-line
forwarder that lazily imports `braincell.vis.plot3d` and adds nothing
(`morph/morphology.py:1200`). The `Cell` methods, by contrast, *are* the
implementation — ~590 lines of entry point plus ~430 lines of private
helpers inside `cell.py` — and two of them reach across packages into a
private name, `braincell.vis.point_topology._plot_discrete_topology_graph`
(`cell.py:2086`, `cell.py:2196`).

## Design

### 1. Public surface: one function

```python
braincell.vis.plot_cell_topology(
    cell,
    *,
    level="node",           # "node" | "cv" | "branch"
    preset="dendrotweaks",
    layout=None,
    layout_scale=1.0,
    region=None,
    locset=None,
    coverage_mode="fraction",
    highlight_color="#ef4444",
    value=None,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    value_label=None,
    show_colorbar=True,
    node_color=None,
    edge_color=None,
    root_color=None,
    ax=None,
) -> Any                    # the rendered Matplotlib axes
```

Migration:

| Before | After |
|---|---|
| `cell.vis_topology(level=L, **kw)` | `bc.vis.plot_cell_topology(cell, level=L, **kw)` |
| `cell.vis_node(**kw)` | `bc.vis.plot_cell_topology(cell, level="node", **kw)` |
| `cell.vis_cv(**kw)` | `bc.vis.plot_cell_topology(cell, level="cv", **kw)` |
| `cell.vis_branch(**kw)` | `bc.vis.plot_cell_topology(cell, level="branch", **kw)` |

`level="node"` is the default, so `cell.vis_node(...)` — the most common
call in the tutorials — becomes `plot_cell_topology(cell, ...)`.

**`show` is dropped.** No function in `braincell.vis` takes it
(`plot2d`, `plot3d`, `plot_point_topology`, `plot_traces`,
`plot_dendrogram`, … all return the artist and leave display to the
caller); `show` belonged to the model-object wrapper, and the wrapper is
gone. Callers that need it write `plt.show()`. Callers that passed
`show=False` drop the argument. In Jupyter the inline backend renders at
cell end regardless, which is how every existing `braincell.vis` call
already behaves in the tutorials.

**Naming.** `plot_cell_topology` does not collide with the two incumbents
it sits beside: `plot_topology(morphology)` is the branch-order schematic
over a `Morphology`, and `plot_point_topology(node_tree)` is the renderer
this function drives. Each new docstring carries a `See Also` pointing at
both and saying what is different.

### 2. Level semantics — three asymmetries to preserve exactly

These are load-bearing, not accidents:

1. **`level="node"` requires `init_state()`; `"cv"` and `"branch"` do
   not.** `vis_node` calls `_raise_if_not_initialized` (`cell.py:1919`)
   and raises `RuntimeError`; the other two are pinned as working
   pre-init by `cell_test.py:1435,1440`.
2. **`level="cv"` demands exactly one root CV** (`cell.py:2071`);
   `level="branch"` takes `morpho.root.index` unconditionally
   (`cell.py:2200`). Do not harmonize.
3. **`level="branch"` is topology-only.** It rejects `locset`, `value`,
   `cmap`, `vmin`, `vmax`, `norm`, `value_label`, and a non-default
   `show_colorbar`.

The branch rejection is currently four separate `ValueError`s
(`cell.py:1779-1786`). Collapse to one check that names every offending
parameter at once:

```
plot_cell_topology(level='branch', ...) does not support: locset, value.
Branch-level rendering is topology-only.
```

### 3. New internal module: `braincell/_multi_compartment/field_resolution.py`

The four methods lean on ~20 private `Cell` helpers that resolve a
user-facing selector — a region, a locset, or a value spec — against the
cell's geometry and runtime, producing a CV-space or point-space vector.
Three of those helpers are **also** used by non-visualization code:

- `_cv_coverage_fractions` / `_resolve_vis_region_intervals` back
  `Cell.on(...)` via `selection.py:120`.
- `_coerce_named_vis_cv_values_object` and
  `_coerce_runtime_point_values_object` back
  `cell.runtime_cvs[i].ions[...]` via `RuntimeIonBinding.get`
  (`cell.py:132,135`).

So the family cannot simply move into `braincell.vis`. It moves into a
new sibling of `cell.py` as free functions taking `cell` — which lets
both `braincell.vis.cell_topology` and the in-package callers import a
*module* instead of reaching for private attributes across a package
boundary.

Moved out of `Cell` (names dropped their `vis` infix — they were never
vis-specific):

| Was | Becomes |
|---|---|
| `_resolve_vis_region_intervals` | `region_intervals(cell, region)` |
| `_cv_coverage_fractions` | `cv_coverage_fractions(cell, region)` |
| `_branch_coverage_fractions` | `branch_coverage_fractions(cell, region)` |
| `_resolve_vis_locset_cv_ids` | `locset_cv_ids(cell, locset)` |
| `_node_highlight_fractions` | `node_highlight_fractions(cell, *, region, locset)` |
| `_cv_highlight_fractions` | `cv_highlight_fractions(cell, *, region, locset)` |
| `_single_population_view` | `single_population_view(cell, values, *, caller, field)` |
| `_vis_cv_voltage` | `cv_voltage(cell, *, caller)` |
| `_resolve_vis_node_values` | `resolve_node_field_values(cell, value, *, caller)` |
| `_resolve_vis_cv_values` | `resolve_cv_field_values(cell, value, *, caller)` |
| `_coerce_vis_node_values_object` | `coerce_node_values(cell, value, *, caller)` |
| `_coerce_vis_cv_values_object` | `coerce_cv_values(cell, value, *, caller)` |
| `_coerce_named_vis_node_values_object` | `coerce_named_node_values(cell, value, *, caller)` |
| `_coerce_named_vis_cv_values_object` | `coerce_named_cv_values(cell, value, *, caller)` |
| `_coerce_runtime_point_values_object` | `coerce_runtime_point_values(cell, value)` |
| `_resolve_unique_layout_by_kind` | `unique_layout_by_kind(cell, kind, *, caller)` |
| `_cv_to_node_values` | `cv_to_node_values(cell, cv_values)` |
| `_mask_non_midpoint_points` | `mask_non_midpoint_points(cell, point_values)` |
| `_layout_field_to_point_values` | `layout_field_to_point_values(cell, layout_id, field, *, caller)` |
| `_layout_field_to_cv_values` | `layout_field_to_cv_values(cell, layout_id, field, *, caller)` |
| `_layout_values_to_point_space` | `layout_values_to_point_space(cell, layout, raw, *, field, caller)` |
| `_layout_values_to_cv_space` | `layout_values_to_cv_space(cell, layout, raw, *, field, caller)` |
| `_split_unit` (module-level, `cell.py:3720`) | `split_unit(value)` |

Stays on `Cell` — genuine runtime bridges and guards with many non-vis
callers, called by `field_resolution` as an in-package private:
`_raise_if_not_initialized`, `_cv_to_point`, `_cv_to_point_unchecked`,
`_point_to_cv`, `_discretization_to_point`.

Rewired callers:

- `selection.py:120` → `field_resolution.cv_coverage_fractions(cell, region)`
- `cell.py:132` → `field_resolution.coerce_named_cv_values(...)`
- `cell.py:135` → `field_resolution.coerce_runtime_point_values(...)`

### 4. New public module: `braincell/vis/cell_topology.py`

Holds `plot_cell_topology` and its three level implementations. It does
**not** fold into `point_topology.py`: that module is a leaf renderer
whose only braincell dependency is `_discretization.base`, and
`docs/design/interface-map.md:39` records it as such.

Module-level imports are cycle-free — verified empirically by probe
(a throwaway module with the exact import set was added as the first
import of `braincell/vis/__init__.py`; `import braincell` and
`import braincell.vis` both succeeded from ten different entry points).
The invariant that makes it safe: `braincell/__init__.py:102` imports
`vis` last, and no module outside `braincell/vis/` imports `braincell.vis`
at module level. Deleting the `Cell` methods *strengthens* this — after
this change `cell.py` has no `braincell.vis` import at all, lazy or
otherwise.

`_plot_discrete_topology_graph` stays private and is imported as a
sibling (`from .point_topology import ...`), matching
`plot3d.py` ← `plot2d._build_value_spec`. Consuming a private sibling
inside `vis` is a strict improvement over today's cross-package reach.

### 5. Error attribution

Roughly 39 message literals name `Cell.vis_*(...)`. All are rewritten to
name the function the caller actually invoked. Two are genuine bugs
surfaced by the audit:

- **`cell.py:2547`** passes `caller="Cell.vis_cv(...)"` from
  `_coerce_named_vis_cv_values_object`, which `RuntimeIonBinding.get`
  reaches for `cell.runtime_cvs[i].ions[...]`. A `pop_size=4` runtime ion
  read today reports *"Cell.vis_cv(...) addresses a single morphology…"*
  on a path with no plotting in it. The `caller=` parameter threaded
  through `field_resolution` fixes this; the runtime path passes
  `"Runtime CV inspection"`.
- **`point_topology.py:280` and `:737`** hardcode
  `"plot_point_topology(...)"` but fire for `vis_cv`/`vis_branch` callers
  that go through `_plot_discrete_topology_graph`. Add a `caller: str`
  parameter so the message names the real entry point.

### 6. Dead code removed on the way

- `vis_node`'s `highlight_point_ids` local (`cell.py:1923`) is assigned
  `None` and never reassigned, making `point_topology.py:280`'s guard
  unreachable from the cell path.
- `_discretization_to_node_values` (`cell.py:2587`) — an alias whose only
  callers are two tests.
- `locate_cv_on_branch` in `cell.py:89` — its sole use is the moved
  `_resolve_vis_locset_cv_ids`; `selection.py` has its own import.
- Duplicate `if __name__ == "__main__"` blocks at `cell_test.py:1557`
  and `1711`/`1715`.

## Tests

| File | Change |
|---|---|
| `braincell/vis/cell_topology_test.py` | **New.** The 23 tests in `cell_test.py:1431-1708` rewritten against `plot_cell_topology`, plus the level-dispatch tests repointed from `mock.patch.object(cell, "vis_cv")` to the module-level implementations. |
| `braincell/_multi_compartment/field_resolution_test.py` | **New.** The coercer-agreement suite (`cell_test.py:1345-1397`) moved and rewritten for the free-function calling convention, plus direct coverage of the region/locset resolvers. |
| `braincell/vis/__init___test.py` | **New.** `__all__` conformance for `braincell.vis`, plus an AST guard asserting no module outside `braincell/vis/` imports `braincell.vis` at load time — the invariant §4 rests on. Follows `braincell/_compute/__init___test.py`. |
| `braincell/vis/_testing.py` | Add `make_soma_dend_tree()` (the `_soma_dend_tree` fixture at `cell_test.py:1422`; *not* a duplicate of `make_length_only_tree` — 1-segment basal vs 2-segment apical). |
| `braincell/_multi_compartment/cell_test.py` | Excise lines 1417-1716; keep the two runtime-inspection coercer cases; drop imports that go unused (`matplotlib.axes`, `RootLocation`, possibly `mock`). |

New coverage the audit found missing:

- `level="node"` before `init_state()` raises `RuntimeError` — the guard
  at `cell.py:1919` has **zero** tests today.
- `level="branch"` rejects each of the eight unsupported parameters, and
  reports all of them together when several are passed.
- `level="cv"` with multiple root CVs raises.
- `plot_cell_topology(not_a_cell)` raises `TypeError`.
- `plot_cell_topology(cell, level="bogus")` raises `ValueError` naming
  the valid levels.
- `cell.runtime_cvs[i].ions[...]` on `pop_size=4` no longer mentions
  `vis_cv` (regression test for §5).

## Docs

- `docs/apis/vis.rst` — new **Cell topology plots** section for
  `plot_cell_topology`. Deliberately not appended to *Morphometry and
  topology plots*, which would render it adjacent to `plot_topology`.
- `docs/design/interface-map.md` — delete `### Cell 可视化入口` (lines
  220-225); add `plot_cell_topology` to the §13 visualization list
  (~line 479); adjust the dependency-direction wording at lines 15, 26,
  28, 34, 39 now that `Cell` no longer reaches into `vis`.
- `changelog.md` — a **Breaking Changes** bullet naming both `Cell` and
  `MultiCompartment` with a before/after block, the dropped `show`, and
  the corrected runtime-inspection message; reword the existing
  `Cell.vis_cv(...)` mention at line 97. Do not touch
  `docs/apis/changelog.md`, which `docs/conf.py:38` generates.
- `docs/apis/braincell.rst` needs no edit —
  `docs/_templates/classtemplate.rst` is `autoclass` + `:members:`, so
  the methods vanish from the rendered page automatically.
- `docs/specs/*` are historical records and are **not** rewritten.

## Notebooks

63 call sites across six notebooks:

| Notebook | Hits |
|---|---|
| `docs/tutorials/vis.ipynb` | 41 |
| `examples/multi_compartment/vis.ipynb` | 10 |
| `docs/tutorials/filter.ipynb` | 5 |
| `examples/neuron_compare/cell/pc_ma2024/tutorial_pc_braincell.ipynb` | 5 |
| `docs/tutorials/single_cell_frontend.ipynb` | 2 |
| `docs/tutorials/cell.ipynb` | 1 |

`docs/conf.py:134` sets `nb_execution_mode = "off"` and no CI job runs
notebooks, so **nothing catches a stale call** — these must be edited by
hand and re-read. Only `source` is touched; a scan confirmed no stored
`outputs` mention the removed names, and the rendered figures stay valid
because the drawing behaviour is unchanged.

`docs/tutorials/filter.ipynb` cells 4/5 build a kwargs dict and call
`cell.vis_topology(**kwargs)` to parameterize the level — that pattern
survives verbatim as `plot_cell_topology(cell, **kwargs)`.

## Verification

1. `pytest braincell/` green.
2. `python -c "import braincell"` and `python -c "import braincell.vis"`
   both succeed (import-order invariant).
3. `git grep -n "vis_topology\|vis_node\|vis_cv\|vis_branch"` returns only
   `docs/specs/` historical records and the `changelog.md` breaking-change
   entry.
4. `python -c "import braincell; assert not [n for n in dir(braincell.Cell) if n.startswith('vis')]"`.
5. Every touched notebook parses as JSON and contains no removed name.
