# Iterative tree walks in `braincell.vis`

## Problem

Every tree walk under `braincell/vis/` recursed once per branch. Python
spends one interpreter frame per recursive call, so a morphology deeper
than roughly 400 branches raised `RecursionError` long before it became
large enough to be slow. Deep chains are not a synthetic edge case:
reconstructed neurites are routinely thousands of branches long, and the
committed benchmarks in `braincell/vis/perf_benchmark_test.py` carried
`xfail(RecursionError)` markers on their 500- and 2000-branch cases to
document exactly this.

The affected call sites:

| File | Walk |
|------|------|
| `vis/layout/_common.py` | `_leaf_counts_by_branch`, `_path_lengths_um_by_branch` |
| `vis/layout/_stem.py` | `_layout_children_stem`, `_layout_children_stem_linear` |
| `vis/layout/_balloon.py` | `_layout_children_balloon` |
| `vis/layout/_fan.py` | `_layout_children_fan` |
| `vis/layout/_radial.py` | `_layout_children_radial_360` |
| `vis/layout/_legacy.py` | `_layout_children_legacy` |
| `vis/morphometry.py` | dendrogram / topology draw walks, `_assign_dendrogram_y`, `_branch_depths`, `_path_segment_distances` |

## Approach

Add `braincell/vis/_traversal.py` with two stack-based orders shared by
everything above:

* `iter_depth_first(root)` — the order a `for child in node.children:
  visit(child)` recursion produces (parents first, left to right).
* `iter_bottom_up(root)` — its reverse, so every node follows all of its
  children. That is all a post-order accumulation needs.

Then convert each walk:

* **Post-order accumulations** (`_leaf_counts_by_branch`,
  `_path_lengths_um_by_branch`, `_assign_dendrogram_y`) become a loop over
  `iter_bottom_up`. `_assign_dendrogram_y` needs two passes, because leaf
  y-positions are assigned in depth-first order before internal branches
  can be centred on them.
* **Pre-order walks** (`_branch_depths`, `_path_segment_distances`, the
  two morphometry draw loops) become a loop over `iter_depth_first`; a
  parent's result is always in the dict before its children are reached.
* **Layout descents** keep their own explicit stack, because each frame
  carries per-parent state (angular interval, inherited angle, stem
  depth). Children are pushed reversed so the visit order is unchanged.
* **The stem family** additionally has to preserve *order-dependent*
  behaviour: side children are placed only after the trunk subtree is
  complete, and each placement scores collisions against a window of the
  most recently placed layouts. A LIFO work stack with two frame kinds
  (`expand` a branch, `place` one side child) reproduces the recursive
  interleaving exactly.

## Equivalence check

`build_layout_branches_2d` was dumped to JSON — branch index, name, total
length, and the first 24 geometry coordinates per branch — for 24
morphologies (5 synthetic, 19 SWC/ASC fixtures under `data/morphology`)
across all six distinct builders reachable through the dispatcher, before
and after the conversion. **144/144 entries byte-identical, 0 differing,
0 error entries.**

## Tests

* `vis/_traversal_test.py` — orders match a recursive oracle; parents
  precede children; the walks survive a chain twice the recursion limit.
* `vis/layout/_dispatch_test.py::DeepMorphologyTest` — all six builders
  lay out a chain twice the recursion limit.
* `vis/morphometry_test.py::DeepMorphologyTest` — dendrogram, topology,
  Sholl, and branch-order histogram on the same chain.
* `vis/_testing.py::make_deep_chain_tree` — the shared fixture.
* `vis/perf_benchmark_test.py` — the four `xfail(RecursionError)` markers
  are removed; those cases now pass.

All the new deep-morphology tests fail with `RecursionError` against the
pre-change source, which is what makes them regression tests rather than
smoke tests.

## Out of scope

`vis/layout/_legacy.py` still contains two recursive helpers (`visit`,
`_leaf_branches_dfs`) inside its "Unused legacy helpers (preserved for
history)" section. Nothing calls them, so they were left alone rather
than rewritten.
