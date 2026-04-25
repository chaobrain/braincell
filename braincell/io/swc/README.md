# SWC Reader Invariants

## Purpose

This note is for maintainers and coding agents, not end users. It records the
parts of `braincell/io/swc/reader.py` that are easiest to break with
"simplifying" refactors.

If you change the behavior described here, at minimum re-run:

- `braincell/io/swc/test.py`
- `braincell/_cv/lower_test.py`
- `examples/neuron_compare/cable/tests/test_mapping.py`
- `examples/neuron_compare/cable/tests/test_runner.py`

## Pipeline Overview

The SWC reader pipeline is:

1. Parse SWC rows.
2. Apply SWC cleanup and normalization rules.
3. Build the node graph.
4. Extract `_SwcBranch` objects.
5. Assemble `Branch` geometry in `_make_branch()`.
6. Attach branches into the final `Morphology`.

The hard parts are not just branch cutting. The reader also carries geometry
semantics that must stay aligned with NEURON `Import3d_SWC_read`.

## Ordinary Branch Formation

`_extract_generic_branches()` handles the non-soma case.

The ordinary rule is simple:

- start from one node
- keep extending the current branch while it has exactly one child
- stop when the child count changes or the child type changes

This part is mode-independent. `mode="neuron"` and `mode="neuromorpho"` do not
change how ordinary non-soma chains are cut into `_SwcBranch` objects.

## Soma Branch Formation

`_extract_soma_branches()` handles the soma-rooted case. This is where the
reader decides how soma sections and soma-attached child branches are cut
before `_make_branch()` turns node ids into geometry.

The current logic distinguishes four soma shapes:

- single-point soma
- special three-point soma
- multi-point non-branching soma
- multi-point branching soma

This is also where child attachments acquire `parent_x = 0`, `0.5`, or `1`.
Those attachment positions drive most of the later copy-or-do-not-copy rules.

## Shared Branch Assembly Rules

`_make_branch()` is the shared node-id-to-geometry step. It takes branch point
ids plus optional attachment metadata and produces the `points` and `radii`
arrays that will be passed into `Branch.from_points(...)`.

### Repeated Points Inside One Branch

Keep these two cases separate:

- `same xyz + same radius`
  - usually just a pure duplicate
  - may be treated as degenerate geometry
- `same xyz + different radius`
  - encodes a radius discontinuity
  - must be preserved
  - `Branch.from_points(...)` turns it into a zero-length jump segment

### Parent/Child Attachments

At a branch boundary, the attach point carries parent-side geometry and the
child's first point carries child-side geometry.

If the attach point and the child's first point have:

- different `xyz`
  - copying the attach point creates the expected first segment
- the same `xyz` and the same `radius`
  - copying is optional and does not add geometry
- the same `xyz` but different `radius`
  - if copying is allowed, the parent point must still be inserted explicitly
  - this creates a zero-length first segment at the branch boundary
  - otherwise the cross-branch radius jump disappears and surface area will no
    longer match NEURON

## Mode Differences

The meaningful difference between `mode="neuron"` and `mode="neuromorpho"` is
mainly soma-midpoint attach copying.

### `mode="neuron"`

- prioritize parity with NEURON `Import3d_SWC_read`
- be more conservative around soma midpoint attachments
- in some `parent_x=0.5` cases, avoid copying the attach point for multi-point
  children
- but once copying is allowed, `same xyz + different radius` still requires
  inserting the parent point so the branch-boundary jump is preserved

### `mode="neuromorpho"`

- lean toward explicit attachment-point copying in the standardized SWC /
  NeuroMorpho style
- copy midpoint soma attach geometry more aggressively than `mode="neuron"`
- keep the same branch-boundary radius-jump rule once copying is allowed

The tables below are summaries of current intent. The code and
`braincell/io/swc/test.py` remain the source of truth.

### NEURON-Oriented Normal Cases

| Soma shape | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| Single-point soma |  | one-point child copies / multi-point child does not |  |
| Special three-point soma |  | one-point child copies / multi-point child does not |  |
| Multi-point non-branching soma | one-point child copies / multi-point child copies | one-point child copies / multi-point child does not | one-point child copies / multi-point child copies |
| Multi-point root-branching soma | one-point child cannot form / multi-point child does not copy | one-point child copies / multi-point child does not | one-point child copies / multi-point child copies |

### NEURON-Oriented `con2prox` Cases

| Soma shape | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| Single-point soma |  | does not trigger |  |
| Special three-point soma |  | triggers |  |
| Multi-point non-branching soma | triggers | triggers | triggers |
| Multi-point root-branching soma | triggers | triggers | triggers |

### BrainCell Current Normal Cases

| Soma shape | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| Single-point soma |  | one-point child copies / multi-point child does not |  |
| Special three-point soma |  | one-point child copies / multi-point child does not |  |
| Multi-point non-branching soma | one-point child copies / multi-point child copies | one-point child copies / multi-point child does not | one-point child copies / multi-point child copies |
| Multi-point root-branching soma | one-point child copies / multi-point child copies | one-point child copies / multi-point child does not | one-point child copies / multi-point child copies |

### BrainCell Current `con2prox` Cases

| Soma shape | 0 | 0.5 | 1 |
| --- | --- | --- | --- |
| Single-point soma |  | triggers |  |
| Special three-point soma |  | triggers |  |
| Multi-point non-branching soma | triggers | triggers | triggers |
| Multi-point root-branching soma | triggers | triggers | triggers |

## Special Handling

### `con2prox`

`con2prox` exists to avoid cutting out tiny artificial sections.

The triggering idea is:

- a point's parent is not soma
- that parent's parent is soma
- NEURON avoids splitting that point into its own short section
- instead, later child branches attach to the proximal end of that child
  branch

The related reader pieces are:

- `_con2prox_attach()`
- `_effective_branch_point_ids()`
- `_section_attach_x()`

### Unknown SWC Type Codes

BrainCell currently collapses unknown SWC type codes to `custom`. This is a
reader choice, not universal SWC truth.

Because of that choice, compare later normalizes NEURON names such as
`dend_<n>` and `minus_<n>` back to `custom`.

### Duplicate `xyzr` Parent/Child Nodes

Exact parent/child duplicates with the same `xyzr` may be merged by the SWC
cleanup rules.

Do not confuse that with `same xyz + different radius`. Exact duplicates can be
collapsed. Radius jumps must be preserved.

## Downstream Coupling

### CV Lowering

`braincell/_cv/lower.py` must preserve zero-length / different-radius segments.
It may only drop zero-length / same-radius pure duplicates.

When a jump lies on a shared CV boundary, the current ownership rule is:

- the left / parent-distal interval owns it
- `x=0` belongs to the first interval

### Compare Mapping

`examples/neuron_compare/cable/templates/mapping.py` is a compare compatibility
layer, not morphology truth.

BrainCell currently collapses unknown SWC type codes to `custom`. NEURON
`Import3d` names unknown positive type codes `dend_<n>` and negative type codes
`minus_<n>`. Compare maps those names back to `custom` only to align with the
current reader behavior.

If the reader later preserves original type codes explicitly, compare-side
normalization must be revised too.

## Tests To Re-Run

- `braincell/io/swc/test.py`
- `braincell/_cv/lower_test.py`
- `examples/neuron_compare/cable/tests/test_mapping.py`
- `examples/neuron_compare/cable/tests/test_runner.py`
