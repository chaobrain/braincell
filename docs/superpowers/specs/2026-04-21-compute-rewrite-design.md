# `braincell.compute` Rewrite — Design

**Date:** 2026-04-21
**Status:** Spec (awaiting plan)
**Scope:** Full rewrite of the compute layer. Package renamed from `braincell/compute/` to `braincell/_compute/` (private). Inner module files drop their leading-underscore prefix (e.g. `_runtime.py` → `runtime.py`) since privacy is already expressed at the package boundary. Public API may change. Downstream call sites in `_multi_compartment/` and `quad/` updated as needed.

---

## 1. Motivation

The current `braincell/compute/` package fuses too many concerns in one 1,450-line module (`_runtime.py`), stores state in `np.ndarray(dtype=object)` of brainunit `Quantity` values (defeats JIT, forces Python loops), and carries a circular import with `_multi_compartment/clamp_table.py`. The package is also exposed as public (`braincell.compute`) despite none of its symbols being part of the intended user surface — they are all consumed only by `_multi_compartment/` and `quad/`. During an audit of the existing module, six correctness bugs and a dozen architectural smells were found. Rewriting (and demoting the package to private `_compute/`) is cheaper than patch-in-place.

### 1.1. Correctness bugs in the current module (must fix)

| # | File / Line | Bug |
|---|---|---|
| C1 | `compute/_point_tree.py:319` | Ternary has identical branches: `half=_entry_half_for_walk(attach_x) if index == 0 else _entry_half_for_walk(attach_x)`. The `else` arm was clearly meant to be `_exit_half_for_walk(attach_x)`. Currently every intra-branch "entry" edge is tagged with entry-half regardless of position. |
| C2 | `compute/_point_tree.py:549-558` (`_compute_peel_levels`) | Walks point ids in reverse numeric order and assumes `parent_id < child_id`. Works today because `build_point_tree` happens to allocate parents first, but invariant is unenforced; any future ordering change silently breaks peel-level correctness. |
| C3 | `compute/_point_tree.py:487-496` (`_locate_branch_cv_by_x`) | Uses `x < cv.dist - epsilon` (exclusive upper), then silently falls back to `ids[-1]` if nothing matches. Off-by-one on branch-interior attachment points can land in the wrong CV without raising. |
| C4 | `compute/_runtime.py:572-587` (`mechanism_signature`) | For `FunctionClamp`, `fn` is compared by dataclass-generated `__eq__` → lambda identity. Two lambdas with identical bodies produce different signatures and distinct layouts. Known footgun; currently just documented. |
| C5 | `compute/_runtime.py:1358-1368` (`_mask_quantity_like`) | Catches bare `Exception` when assigning the zero-of-correct-unit fallback. Swallows unit system errors. |
| C6 | `compute/_runtime.py:1442-1456` (`_is_root_level_runtime_node`) | Swallows `KeyError` from the channel registry and returns `False`. Unknown channel kinds are silently treated as non-root rather than raising. |

### 1.2. Architectural smells (addressed by rewrite)

1. **God module.** `_runtime.py` mixes layout compilation, state storage, ion resolution, channel binding, clamp evaluation, bridging helpers (`scatter_midpoint_values`, `gather_midpoint_values`, `quantity_vector`, `fill_like`), an unrelated `clone_morpho`, and install/uninstall side effects.
2. **Circular import.** `CellRuntimeState.from_cell` imports `braincell._multi_compartment.clamp_table.build_clamp_active_table` at call time. Layering is inverted — clamp table conceptually belongs with runtime state, not with the multi-compartment façade.
3. **Misplaced helpers.** `clone_morpho` lives in `_runtime.py` but is a pure morphology operation.
4. **Misleading `@frozen`.** `CellRuntimeState` is decorated `@dataclass(frozen=True)` yet its `runtime_nodes`, `ions`, `state_buffers`, `layout_mechanisms` fields are mutated after construction. The frozen contract is a lie.
5. **Magic strings as enums.** `target ∈ {"density","point"}`, `layout ∈ {"dense","sparse"}`, `position ∈ {"proximal","mid","distal"}` vs `half ∈ {"prox","dist"}` — vocabulary inconsistent between `CVPoint.position` and `CVEdge.half`.
6. **Channel-specific logic in runtime.** HH1952 temperature→phi translation is hardcoded in `_runtime_constructor_params` and `_sync_runtime_node_param`. Runtime should not know specific channel class names.
7. **Object-dtype state buffers.** `state_buffers: dict[tuple[int,str], np.ndarray(dtype=object)]` stores boxed `Quantity` values. Precludes JIT, forces per-element Python iteration in `_instantiate_runtime_ion_instance`, `_sync_runtime_ion`, `_object_array_from_value`, `_write_state_buffer`, `_extract_point_value`.
8. **Install/uninstall side-effect API.** `install_cell_runtime(cell, runtime)` mutates `cell._in_size`, `cell._out_size`, `cell.ion_channels`, `cell.C`, `cell.V_th` and returns a tuple of attribute names for `uninstall_cell_runtime` to `delattr`. Couples `Cell` lifecycle through a tuple-of-names handshake.
9. **Per-step Python loop over clamps.** `evaluate_point_clamps` loops layouts in Python and performs `.at[point_index].add(...)` once per layout, each step.
10. **Duplicated row-build logic.** `_build_row_to_point_id` and `_build_groups` in `_point_tree.py` both perform the same peel-level grouping and matrix-index sort.

---

## 2. Design decisions (locked)

Locked via brainstorming Q&A before this spec was written:

- **D1.** Full rewrite. Public API may change (e.g. `install_cell_runtime` / `uninstall_cell_runtime` removed). Downstream call sites updated.
- **D2.** State buffers store `u.Quantity(jnp.ndarray, unit)` directly. No `np.ndarray(dtype=object)` intermediate.
- **D3.** Three files in `braincell/_compute/`: `topology.py`, `runtime.py`, `table.py`. Package renamed from `compute/` to `_compute/` to mark it private. Inner files drop the `_` prefix since privacy is expressed once at the package boundary. `runtime.py` trimmed to ~500 lines by moving helpers out. `clamp_active_table` folds back in — kills the circular import.
- **D4.** `Cell` owns runtime directly. No `install_cell_runtime` / `uninstall_cell_runtime` functions. `Cell.init_state()` inlines the setup.
- **D5.** Magic strings replaced with `typing.Literal` on frozen dataclass fields (no runtime enum cost).
- **D6.** HH1952 temperature→phi translation moves out of runtime. The `INa_HH1952` / `IK_HH1952` channel classes accept `T` and compute `phi` internally (or via a `Density` post-init hook). Runtime no longer special-cases channel names.
- **D7.** `FunctionClamp` mechanism signature hashes on `fn.__code__.co_code` + normalized closure cell contents, so two lambdas with identical bodies + closures merge into one layout.
- **D8.** `dhs_static_cache` hook on `CellRuntimeState` preserved (used by `quad/_staggered.py`).

---

## 3. Module layout

```
braincell/_compute/
  __init__.py          internal re-exports (consumed by _multi_compartment/, quad/)
  topology.py          PointTree, PointScheduling, builders
  runtime.py           CellRuntimeState, MechanismLayout, ClampActiveTable, compilation pipeline
  table.py             MechanismObjectTable, MechanismObjectCell, mechanism_cell_key
  topology_test.py     co-located tests (new)
  runtime_test.py      co-located tests (absorb existing)
  table_test.py        co-located tests (new)
```

Naming rationale: the package directory itself carries the leading `_` (mirrors the existing `_multi_compartment/` and `_base.py` convention), so inner modules do not need to repeat it. `runtime.py` reads cleaner than `_compute/_runtime.py`.

### 3.1. Moved out of `_compute/`

| From | To | Reason |
|---|---|---|
| `runtime.py::clone_morpho` | `braincell/morph/morphology.py` | Pure morphology op. |
| `runtime.py::scatter_midpoint_values` | `_multi_compartment/bridge.py` | Sole consumer. |
| `runtime.py::gather_midpoint_values` | `_multi_compartment/bridge.py` | Sole consumer. |
| `runtime.py::cv_value_vector` | `_multi_compartment/bridge.py` | Sole consumer. |
| `runtime.py::fill_like` | `_multi_compartment/bridge.py` | Sole consumer. |
| `runtime.py::quantity_vector` | `_multi_compartment/bridge.py` | Sole consumer. |
| `runtime.py::build_placeholder_ions` | `braincell/ion/__init__.py` | Ion-module construction helper. |
| `runtime.py::install_cell_runtime` | removed | Inlined into `Cell.init_state()`. |
| `runtime.py::uninstall_cell_runtime` | removed | Inlined into `Cell.reset()`. |
| `_multi_compartment/clamp_table.py::ClampActiveTable` | `_compute/runtime.py` | Folds in to kill circular import. |
| `_multi_compartment/clamp_table.py::build_clamp_active_table` | `_compute/runtime.py` | Folds in to kill circular import. |

`_multi_compartment/clamp_table.py` is deleted. `_multi_compartment/clamp_table_test.py` is merged into `_compute/runtime_test.py`.

### 3.2. Package `__init__.py`

```python
from .topology import (
    PointTree, PointScheduling,
    CVPoint, ComputePoint, CVEdge, ComputeEdge,
    build_point_tree, build_point_scheduling,
)
from .runtime import (
    CellRuntimeState,
    MechanismLayout,
    ClampActiveTable,
    Target, Layout, Position, Half,
)
from .table import (
    MechanismObjectTable,
    MechanismObjectCell,
    mechanism_cell_key,
)

__all__ = [
    "PointTree", "PointScheduling",
    "CVPoint", "ComputePoint", "CVEdge", "ComputeEdge",
    "build_point_tree", "build_point_scheduling",
    "CellRuntimeState", "MechanismLayout", "ClampActiveTable",
    "Target", "Layout", "Position", "Half",
    "MechanismObjectTable", "MechanismObjectCell", "mechanism_cell_key",
]
```

`braincell.compute` is removed from the public namespace. Any user code that imported `from braincell.compute import ...` must update to `from braincell._compute import ...` (with the understanding that `_compute` is internal and may break between releases). No such usage is expected outside the repo.

---

## 4. `topology.py` — point tree + scheduling

### 4.1. Vocabulary (unified)

```python
from typing import Literal

Position = Literal["prox", "mid", "dist"]   # was mixed: "proximal" / "mid" / "distal" on CVPoint
Half     = Literal["prox", "dist"]          # unchanged
```

### 4.2. Data classes

Same shape as current module, but:

- `CVPoint.position: Position`
- `CVEdge.half: Half`
- `PointTree` unchanged on field names and types (`points`, `edges`, `point_parent`, `point_children`, `cv_midpoint_point_id`, `branch_terminal_point_id`, `root_point_id`, `point_id_to_matrix_index`, `matrix_index_to_point_id`, `cv_id_to_matrix_index`).
- `PointScheduling` unchanged on field names.

### 4.3. `build_point_tree(morpho, *, cvs) -> PointTree`

Fixes bugs C1, C2, C3:

- **C1 fix.** The intra-branch edge-role loop assigns:
  ```python
  add_edge_role(parent_point_id, midpoint_point_id,
                cv_id=cv_id, half=_entry_half_for_walk(attach_x))
  add_edge_role(midpoint_point_id, child_point_id,
                cv_id=cv_id, half=_exit_half_for_walk(attach_x))
  ```
  i.e. the entry-half tag is on the edge *into* the midpoint, exit-half tag is on the edge *out of* the midpoint. Current module uses entry-half for both.
- **C2 fix.** After finalising `point_parent`, compute peel levels via an explicit leaf-up topological peel that does not rely on numeric ordering of point ids:
  ```python
  def _compute_peel_levels(point_parent, point_children):
      levels = np.full(len(point_parent), -1, dtype=np.int32)
      frontier = [pid for pid, children in enumerate(point_children) if len(children) == 0]
      for pid in frontier:
          levels[pid] = 0
      cursor = 0
      while cursor < len(frontier):
          pid = frontier[cursor]; cursor += 1
          parent = int(point_parent[pid])
          if parent < 0:
              continue
          child_level = int(levels[pid])
          if int(levels[parent]) < child_level + 1:
              levels[parent] = child_level + 1
              if all(int(levels[sib]) >= 0 for sib in point_children[parent]):
                  frontier.append(parent)
      if (levels < 0).any():
          raise ValueError("compute_peel_levels: cycle detected or point unreachable.")
      return levels
  ```
  Invariant-free. Cycle-safe (raises rather than infinite-loops).
- **C3 fix.** `_locate_branch_cv_by_x` uses half-closed `[prox, dist)` on interior CVs but raises `ValueError` with a diagnostic message when `x` lies in no CV (instead of silently returning `ids[-1]`). The boundary cases (`x <= 0 + eps`, `x >= 1 - eps`) still return first/last CV explicitly.

### 4.4. `build_point_scheduling(point_tree, *, max_group_size=32, algorithm="dhs") -> PointScheduling`

Deduplicate the current duplicated sort in `_build_row_to_point_id` / `_build_groups`:

```python
def _order_points_by_peel_then_matrix(point_tree, peel_levels):
    """Return a single permutation of point ids: peel-descending, then matrix-ascending."""
    matrix_idx = np.asarray(point_tree.point_id_to_matrix_index, dtype=np.int32)
    # np.lexsort uses the LAST key as primary: peel-descending primary, matrix-ascending secondary.
    order = np.lexsort((matrix_idx, -np.asarray(peel_levels, dtype=np.int32)))
    return order.astype(np.int32)
```

Used by both row_to_point_id and group building; no re-sort per level.

### 4.5. Edge cases made explicit

- Empty branch (zero CVs) → `ValueError` naming the branch index. (Already raised; kept.)
- Duplicate CV midpoint (shouldn't happen) → `ValueError`. (Already raised; kept.)
- Missing traversal for a branch → `ValueError`. (Already raised; kept.)
- Unknown `algorithm` → `ValueError`. (Already raised; kept.)
- `max_group_size <= 0` or non-int → `TypeError` / `ValueError`. (Already raised; kept.)
- `cvs` empty → `ValueError` (new; currently would trigger the "Root branch has no CVs" path indirectly).

---

## 5. `runtime.py` — runtime state

### 5.1. Types

```python
from typing import Literal
Target = Literal["density", "point"]
Layout = Literal["dense", "sparse"]
```

### 5.2. `MechanismLayout`

Same fields as today, but:
- `target: Target`, `layout: Layout` (Literal).
- `point_index: np.ndarray` (build-time metadata, `int32`, always set; remove `None` branches scattered through the current code).
- `point_mask: np.ndarray | None` (only dense layouts carry it; sparse always `None`).
- `source_cv_ids: tuple[int, ...]` unchanged.
- `source_rule: str | None` kept for provenance.

### 5.3. `CellRuntimeState`

`@dataclass` (not frozen — honest mutable contract).

```python
@dataclass
class CellRuntimeState:
    point_tree: PointTree
    n_point: int
    n_cv: int
    layouts: tuple[MechanismLayout, ...]
    point_to_layout_ids: tuple[tuple[int, ...], ...]
    cv_to_layout_ids: tuple[tuple[int, ...], ...]
    voltage_shape: tuple[int, ...]

    state_shapes: dict[tuple[int, str], tuple[int, ...]]
    state_buffers: dict[tuple[int, str], u.Quantity]   # jnp-backed Quantity; unit attached

    layout_mechanisms: dict[int, object]
    runtime_nodes: dict[int, object]

    ions: dict[str, object]
    ion_aliases: dict[str, str]
    ion_family_candidates: dict[str, tuple[str, ...]]
    ion_class_candidates: dict[str, tuple[str, ...]]
    bound_ion_keys: dict[int, tuple[str, ...]]
    current_owner_keys: dict[int, str | None]

    dhs_static_cache: object | None = None
    clamp_active_table: "ClampActiveTable | None" = None
    cv_area: u.Quantity | None = None
```

Method surface is the same as today: `from_cell`, `get_point_layouts`, `get_cv_layouts`, `expected_state_shape`, `get_state`, `set_state`, `get_point_state`, `get_cv_state`, `get_runtime_node`, `get_layout_mechanism`, `get_ion`, `resolve_ion_key`, `has_layout_value`, `get_layout_value`, `evaluate_point_clamps`.

### 5.4. State buffer storage (D2)

Current: `state_buffers[(lid, var)] : np.ndarray(dtype=object)` of `Quantity` scalars.

New: `state_buffers[(lid, var)] : u.Quantity` where the underlying array is `jnp.ndarray` (or a Python scalar if the mechanism's declared value is scalar). For sequence-valued params (e.g. `CurrentClamp.durations`, `CurrentClamp.amplitudes` with multiple steps) the underlying array is an extra-ragged container — see 5.4.1.

#### 5.4.1. Per-mechanism buffer shapes

| Mechanism | Param kind | Stored as |
|---|---|---|
| `Density` (channel / ion) | scalar `Quantity` per point | `u.Quantity(jnp.ndarray, unit)` with shape `(n_point,)` for dense, `(n_active,)` for sparse |
| `Synapse` / `Junction` | scalar `Quantity` per point | same as Density |
| `CurrentClamp.start` | scalar `Quantity` per point | `u.Quantity(jnp.ndarray, u.ms)` shape `(n_active,)` |
| `CurrentClamp.durations` | **ragged** tuple-per-point | see 5.4.2 |
| `CurrentClamp.amplitudes` | **ragged** tuple-per-point | see 5.4.2 |
| `SineClamp.{amplitude,frequency,phase,offset,start,duration}` | scalar `Quantity` per point | `(n_active,)` |
| `FunctionClamp.fn` | Python callable per point | plain tuple (not JIT-traceable; evaluated Python-side) |
| `FunctionClamp.{start,duration}` | scalar `Quantity` per point | `(n_active,)` |
| Probe layouts (`StateProbe`, `MechanismProbe`, `CurrentProbe`) | — | no state buffer (kept; matches current behavior) |

#### 5.4.2. Ragged per-point step sequences (`CurrentClamp`)

`durations` and `amplitudes` are variable-length tuples per active point. Because each point may declare a different number of steps, a single 2-D `jnp.ndarray` is not always possible. Two sub-cases:

- **Uniform step count across active points in the layout** → rectangular `jnp.ndarray` shape `(n_active, n_steps)`, single `u.Quantity`. This is the common case in practice (one `CurrentClamp` per placement).
- **Ragged step counts** → padded to `max_steps` with zeros, plus a companion mask `(n_active, max_steps): bool` stored alongside as `state_buffers[(lid, "_mask_durations")]` and `state_buffers[(lid, "_mask_amplitudes")]` (leading underscore — not user-visible). Current-clamp evaluation in `_eval_current_clamp` reads both.

This replaces the current object-dtype-per-scalar approach where each point stores its own ragged tuple inside a Python object cell.

#### 5.4.3. `set_state` contract

```python
def set_state(self, layout_id: int, var_name: str, value: u.Quantity | Sequence[u.Quantity]) -> None
```

- Scalar `value` broadcasts to the full buffer shape.
- Sequence of `Quantity` values → converted via `u.Quantity(jnp.stack(...), unit)` and shape-checked.
- Unit mismatch → `brainunit.UnitMismatchError` (or `TypeError` if unitless).
- After writing the buffer, `_sync_runtime_node_param` propagates to the installed runtime node (same behavior as today, but in terms of `Quantity` arithmetic instead of object-array loops).

### 5.5. Removing HH1952 hardcode (D6)

Current: `_runtime_constructor_params` rewrites `raw["T"] → raw["phi"] = 3**((T_c - 36)/10)` only for class names `{"INa_HH1952","IK_HH1952"}`. `_sync_runtime_node_param` has a matching special-case.

New:
1. `INa_HH1952` and `IK_HH1952` classes in `braincell/channel/` accept `T` (temperature `Quantity`) directly in their constructor and compute `phi` internally (or via a cached property).
2. `_runtime_constructor_params` becomes generic: pass declared params through unchanged.
3. `_sync_runtime_node_param` becomes generic: `setattr(node, var_name, new_value)` plus, optionally, call `node._on_param_updated(var_name, new_value)` if the class defines such a hook. HH1952 classes implement that hook to recompute `phi` when `T` changes. Default hook = no-op.

Runtime no longer knows any channel class names.

### 5.6. `FunctionClamp` signature (D7)

Current: `mechanism_signature(m) = (type(m).__qualname__, m)`. For `FunctionClamp` this hashes via dataclass `__eq__`, which compares `fn` by identity.

New helper:

```python
def _fn_fingerprint(fn: Callable) -> tuple:
    code = fn.__code__
    cell_contents = tuple(
        _quantity_fingerprint(cell.cell_contents)
        for cell in (fn.__closure__ or ())
    )
    return (code.co_code, code.co_consts, code.co_varnames, cell_contents)

def mechanism_signature(m) -> tuple:
    if isinstance(m, FunctionClamp):
        return ("FunctionClamp",
                _fn_fingerprint(m.fn),
                m.start, m.duration)
    return (type(m).__qualname__, m)
```

Two lambdas with identical bodies and identical captured values merge into one layout. Caveats documented in the `FunctionClamp` docstring: non-pure closures (reading module-level mutables) still merge on signature but runtime evaluations may diverge — user error, not ours.

### 5.7. Clamp evaluation

Current: Python loop over all sparse point-target clamp layouts, `u.math.zeros` allocation, `.at[point_index].add` per layout.

New: precomputed `ClampActiveTable` groups all clamp layouts of the same kind into single batched arrays, so one JIT-compiled function evaluates all `CurrentClamp` layouts, another all `SineClamp`, another all `FunctionClamp` (still Python, fn-per-point). `evaluate_point_clamps` becomes three calls instead of one-per-layout.

`ClampActiveTable` (the same name, now living in `_runtime.py`) owns:
- `point_index_current: jnp.ndarray (N_c,)` — active points for all current clamps flattened
- `layout_index_current: jnp.ndarray (N_c,)` — which layout each active row belongs to
- same pair for sine / function clamps
- plus the per-kind flattened parameter Quantity arrays

This merges the "precompute active points" work that `_multi_compartment/clamp_table.py::build_clamp_active_table` was doing with the actual evaluation, in one file.

### 5.8. `Cell` lifecycle (D4)

`install_cell_runtime` / `uninstall_cell_runtime` deleted. `Cell.init_state()` absorbs the work:

```python
def init_state(self, *args, **kwargs):
    self._runtime = CellRuntimeState.from_cell(self)
    self._in_size = (self._runtime.n_cv,)
    self._out_size = (self._runtime.n_cv,)

    root_nodes = dict(self._runtime.ions)
    for layout in self._runtime.layouts:
        node = self._runtime.runtime_nodes.get(layout.id)
        if node is not None and _is_root_level_runtime_node(layout.kind):
            root_nodes[f"layout_{layout.id}"] = node

    self.ion_channels = self._format_elements(IonChannel, **root_nodes)
    self.C = bridge.cv_value_vector(self, attr_name="cm")
    self.V_th = bridge.fill_like(self.varshape, self.V_th)
    # ... existing init_state continuation
```

and `Cell.reset()` clears `self._runtime`, and `delattr`s `_in_size`, `_out_size`, `ion_channels`, `C`. `V_th` restoration still uses the property-setter pattern.

The tuple-of-installed-names handshake is gone.

### 5.9. `_is_root_level_runtime_node` (C6 fix)

Current: swallows `KeyError` from registry, returns `False`.

New: if the registry does not know the class name, raise `ValueError(f"Unknown channel class {class_name!r}")`. Unknown classes should have been rejected earlier in `from_cell`; reaching this function with an unknown kind is a bug. Raising surfaces it.

### 5.10. `_mask_quantity_like` (C5 fix)

Not needed in the new storage model. When `state_buffers` are `u.Quantity(jnp.ndarray, unit)` with a known unit, masking off points = `jnp.where(mask[..., None], values, 0.0)` with the same unit. No per-element Python loop, no bare `except Exception`.

### 5.11. Ion system (minor cleanup, not rewritten)

Ion instance resolution logic (`_collect_runtime_ion_instances`, `_build_ion_alias_map`, `_resolve_channel_runtime_bindings`, etc.) is largely correct and stays. Mechanical changes only:

- `ion_class_candidates` typed `dict[str, tuple[str, ...]]` throughout (currently typed as tuple in the return but built as list internally).
- `resolve_ion_key` exception policy documented explicitly: `KeyError` when the selector is not known at all, `ValueError` when it is ambiguous. Both exceptions include the candidate list in the message.
- `_instantiate_runtime_ion_instance` per-point Python loop replaced with vectorised `buffer.at[point_index].set(values[point_index])` on `Quantity`-backed arrays.

---

## 6. `table.py` — inspection helpers

Renamed from `_assignment_table.py` (old) → `table.py` (new, inside `_compute/`). Content unchanged: `MechanismObjectCell`, `MechanismObjectTable`, `mechanism_cell_key`. One cleanup:

- `MechanismObjectCell.__getattr__` currently delegates all unknown attribute lookups to `get_param`, which makes typos look like missing parameters. New behavior: `__getattr__` only resolves names the mechanism or node declares (whitelist from `_mechanism_var_names` plus `declaration.params`); unknown names raise `AttributeError` with a list of valid names.

---

## 7. Downstream call-site updates

| File | Change |
|---|---|
| `_multi_compartment/cell.py` | Drop `install_cell_runtime` / `uninstall_cell_runtime` imports. Inline setup into `init_state()` / teardown into `reset()`. Import `bridge.cv_value_vector` / `bridge.fill_like` (moved). |
| `_multi_compartment/bridge.py` | Absorb `scatter_midpoint_values`, `gather_midpoint_values`, `quantity_vector`, `cv_value_vector`, `fill_like` from old `_runtime.py`. |
| `_multi_compartment/currents.py` | Unchanged API. `runtime.clamp_active_table` still present, same access pattern. |
| `_multi_compartment/probes.py` | Unchanged API. |
| `_multi_compartment/clamp_table.py` | **Deleted.** Functionality moved into `compute/_runtime.py`. |
| `_multi_compartment/clamp_table_test.py` | Merged into `compute/_runtime_test.py`. |
| `quad/_staggered.py` | Unchanged (duck-typed on `target.point_tree()` and `target.point_scheduling(...)`). |
| `morph/morphology.py` | Gains `clone_morpho` (moved from `_runtime.py`). |
| `ion/__init__.py` | Gains `build_placeholder_ions` (moved from `_runtime.py`). |
| `channel/sodium.py` (`INa_HH1952`) | Accepts `T` kwarg; computes `phi` internally; implements `_on_param_updated("T", new_T)` hook. |
| `channel/potassium.py` (`IK_HH1952`) | Same pattern as sodium. |

---

## 8. Error and edge case inventory

### 8.1. Construction-time errors (raised from `CellRuntimeState.from_cell`)

| Condition | Exception | Source |
|---|---|---|
| Root branch has zero CVs | `ValueError` | topology |
| Any branch has zero CVs | `ValueError` | topology |
| Empty `cvs` tuple | `ValueError` (new) | topology |
| Duplicate CV midpoint claim | `ValueError` | topology |
| Missing branch traversal metadata | `ValueError` | topology |
| CV-locator `x` in no CV interval | `ValueError` (was silent fallback, **C3 fix**) | topology |
| Peel-level computation detects cycle | `ValueError` (**C2 fix**) | topology |
| Unsupported point scheduling algorithm | `ValueError` | topology |
| `max_group_size ≤ 0` | `ValueError` | topology |
| `max_group_size` non-int | `TypeError` | topology |
| Unknown channel class in mechanism | `KeyError` from registry | runtime |
| Unknown ion runtime class | `ValueError` | runtime |
| Ion instance name conflicts across families | `ValueError` | runtime |
| Ion instance name reused across families | `ValueError` | runtime |
| Ion instance name conflicts with canonical family key | `ValueError` | runtime |
| Ion family ambiguous (no selector, multiple candidates) | `ValueError` | runtime |
| Ion selector not in family candidates | `ValueError` | runtime |
| Ion selector cannot be resolved | `KeyError` | runtime |
| Mixed-ion channel missing `current_owner_type` | `ValueError` | runtime |
| Mixed-ion channel unknown `ion_names` keys | `ValueError` | runtime |
| Single-ion channel given `ion_names` | `ValueError` | runtime |
| Mixed-ion channel given `ion_name` | `ValueError` | runtime |
| Channel declares `ion_name` but binds no ions | `ValueError` | runtime |
| Sparse layout missing `point_index` | `ValueError` | runtime |
| Unknown layout id | `KeyError` | runtime |
| Unknown state buffer key | `KeyError` | runtime |
| State assignment shape mismatch | `ValueError` | runtime |
| State assignment unit mismatch | `brainunit.UnitMismatchError` | runtime |
| `_is_root_level_runtime_node` unknown class | `ValueError` (**C6 fix**) | runtime |

### 8.2. Runtime evaluation edge cases

| Condition | Behavior |
|---|---|
| `CurrentClamp` with `t < start` for all points | all-zero current (same as today) |
| `CurrentClamp` with `t ≥ sum(durations)` | zero (same as today; documented) |
| `CurrentClamp` with empty `durations` / `amplitudes` | zero; layout allowed |
| `SineClamp` with `t ≥ duration` | zero (masked) |
| `SineClamp` with `frequency == 0` | pure offset current |
| `FunctionClamp` `fn` returns non-Quantity | `TypeError` at eval time |
| `FunctionClamp` `fn` returns non-scalar | `ValueError` at eval time |
| `FunctionClamp` `t < 0` (before `start`) | zero (masked) |
| `evaluate_point_clamps` with no clamp layouts | returns `u.Quantity(zeros(n_point), u.nA)` |
| Multiple clamp layouts targeting same point | additive (same as today) |
| `sample_probe` unknown name | `KeyError` (same as today) |
| `sample_probes` duplicate probe names | `ValueError` (same as today) |

### 8.3. Storage migration edge cases

| Condition | Behavior |
|---|---|
| Ragged-step `CurrentClamp` across points | padded `(n_active, max_steps)` + mask buffer (5.4.2) |
| Mixed-unit sequences in a param list | converted to the first element's unit, shape-checked; mismatched units → `UnitMismatchError` |
| Empty `point_index` (layout has zero active points) | layout skipped during node instantiation; buffer allocated with shape `(0,)` |
| Writing a scalar to a dense `(n_point,)` buffer | broadcasts; unit must match |
| Writing a per-active-point array to a dense buffer | writes only at `point_index`, leaves other positions at their init-time value |
| `set_state` on a sparse buffer with full `(n_point,)` array | `ValueError` — sparse only accepts `(n_active,)` |

---

## 9. Testing strategy

### 9.1. Test files

- `_compute/topology_test.py` (new) — covers `PointTree`, `PointScheduling`, all builders. Moves existing point-tree tests here from wherever they live.
- `_compute/runtime_test.py` (absorbs the current ~900-line file; grows with new cases).
- `_compute/table_test.py` (new) — covers `MechanismObjectTable` / `MechanismObjectCell`.

### 9.2. Must-add test cases (covering the bug fixes)

| ID | Test |
|---|---|
| T1 | **C1 regression.** On a 2-branch morphology with ≥3 CVs per branch, assert `CVEdge.half` values alternate entry/exit correctly along each branch walk, not all "prox". |
| T2 | **C2 regression.** Construct a synthetic `PointTree` where `point_parent[0] = 3` (parent id > child id). Assert peel-level computation terminates and produces correct levels. |
| T3 | **C3 regression.** Place a point at `x = 0.5` on a branch whose CVs tile `[0, 0.3)`, `[0.3, 0.7)`, `[0.7, 1.0)`. Assert attachment lands in CV 1 (middle). Then place at `x = 0.999` — assert last-CV fallback works. Then simulate a broken tiling that covers only `[0, 0.4)` and `[0.6, 1.0)`, assert `ValueError` for `x = 0.5` instead of silent fallback. |
| T4 | **C4 regression.** Two `FunctionClamp` instances with identical lambda bodies (`lambda t: 0.1 * u.nA`) and identical other fields merge into one layout (current: they don't). |
| T5 | **C5 regression.** A `Density` layout with no `point_mask` handles `_mask_quantity_like`-equivalent masking correctly under JIT. |
| T6 | **C6 regression.** `_is_root_level_runtime_node("channel:__does_not_exist__")` raises `ValueError` instead of returning `False`. |

### 9.3. Must-add test cases (storage rewrite)

| ID | Test |
|---|---|
| T7 | `state_buffers[(lid, "g_max")]` is a `u.Quantity` whose mantissa is `jnp.ndarray`, not `np.ndarray(dtype=object)`. |
| T8 | `set_state(lid, "g_max", 4.0 * u.mS/u.cm**2)` broadcasts to the full dense shape; `get_state(...)[0]` returns `4.0 * u.mS/u.cm**2`. |
| T9 | Ragged `CurrentClamp.durations` across 3 points with step counts `[1, 2, 3]` → stored as `(3, 3)` padded array with `(3, 3)` mask. `_eval_current_clamp` returns the correct per-point current. |
| T10 | `set_state` with unit mismatch raises `UnitMismatchError`. |
| T11 | `set_state` with shape mismatch raises `ValueError`. |
| T12 | JIT-compile `evaluate_point_clamps`; confirm compile succeeds and no `np.ndarray(dtype=object)` leaks into the trace (check via `jax.make_jaxpr`). |

### 9.4. Must-add test cases (lifecycle)

| ID | Test |
|---|---|
| T13 | `Cell.init_state()` sets `_in_size`, `_out_size`, `ion_channels`, `C` directly (no `install_cell_runtime` import needed in the test). |
| T14 | `Cell.reset()` clears the above attributes and `self._runtime = None`. |
| T15 | Re-call `init_state()` after `reset()` reproduces equivalent runtime state (topology + layouts identical). |

### 9.5. Must-add test cases (HH1952 decoupling)

| ID | Test |
|---|---|
| T16 | `INa_HH1952(T=u.celsius2kelvin(36.0))` constructs without runtime; `.phi` is populated. |
| T17 | `Cell` with HH1952 channel + `runtime.set_state(lid, "T", new_T)` updates `node.phi` via the `_on_param_updated` hook, with no runtime branching on class names. |

### 9.6. Regression coverage

All ~40 existing tests in `compute/_runtime_test.py` (old location) carry forward to `_compute/runtime_test.py` (new location) and must pass unchanged — except the ones that explicitly import `install_cell_runtime` / `uninstall_cell_runtime`, which get rewritten to test `Cell.init_state()` / `Cell.reset()` directly.

---

## 10. Migration order

Implementation plan will cover this in detail; summarized here so the plan scope is visible:

1. Rename package `braincell/compute/` → `braincell/_compute/`. Within it, rename `_point_tree.py` → `topology.py`, `_runtime.py` → `runtime.py`, `_assignment_table.py` → `table.py`, and co-located test files to match. Update `__init__.py` and every import site in `_multi_compartment/` and `quad/`. This is a pure move — no behavior change. Run full test suite.
2. Apply C1/C2/C3 topology fixes inside `topology.py` + vocabulary unification (`Position`, `Half`). Add T1/T2/T3 tests.
3. Move helpers out of `runtime.py` (`scatter_*`, `gather_*`, `quantity_*`, `fill_like`, `cv_value_vector`, `clone_morpho`, `build_placeholder_ions`) to their new homes. Update imports. Tests stay green.
4. Fold `clamp_table.py` into `runtime.py` (kills circular import). Delete `_multi_compartment/clamp_table.py`; merge its test file into `runtime_test.py`. Run tests.
5. Remove HH1952 hardcode. Update `INa_HH1952` / `IK_HH1952` classes to accept `T`; teach runtime the generic `_on_param_updated` hook. Add T16/T17 tests.
6. Swap state-buffer storage to `u.Quantity(jnp.ndarray, unit)`. Largest single change; touches `_allocate_state_buffer`, `_write_state_buffer`, `_extract_point_value`, `_instantiate_runtime_ion_instance`, `_sync_runtime_ion`, `_runtime_param_value`, `evaluate_point_clamps`. Ragged-step handling (5.4.2) included here. Add T7–T12 tests.
7. Remove `install_cell_runtime` / `uninstall_cell_runtime`; inline setup/teardown into `Cell`. Add T13–T15 tests.
8. Magic strings → `Literal`. Mechanical.
9. `FunctionClamp` signature fingerprinting. Add T4 test.
10. Tighten `MechanismObjectCell.__getattr__` (whitelist only declared vars).
11. Final pass on error messages + edge case coverage (section 8). Full test + JIT-trace regression.

Each step is a separate commit and all tests pass between steps.

---

## 11. Non-goals

Explicit out-of-scope for this rewrite:

- Changing the DHS scheduling algorithm itself (only deduplicating its build code). Any future alternate scheduler is a separate phase.
- Rewriting `quad/_staggered.py` or any `quad/` solver code.
- Rewriting `cv/` or `morph/` beyond receiving `clone_morpho`.
- Changing the public `Cell` API beyond removing `install_cell_runtime` / `uninstall_cell_runtime` and inlining their bodies.
- Any performance work on ion resolution beyond replacing the per-element Python loop in `_instantiate_runtime_ion_instance` with a vectorised write.
- Any new mechanism types. The rewrite is behavior-preserving for all existing mechanism types except the 6 C-bugs, which are behavior changes that fix incorrect results.

---

## 12. Alternatives considered (briefly)

| Alt | Why rejected |
|---|---|
| Bug-fix-only pass, no rewrite | Object-dtype state buffers and god-module layering make the bugs likely to recur. |
| Keep object-dtype storage, JIT only the solver | Probe/clamp evaluation and state sync remain Python-bound; doesn't resolve D2. |
| Split into 5–8 files | User-preferred minimal churn; 3 files hits the readability ceiling without import sprawl. |
| Keep `install_cell_runtime` / `uninstall_cell_runtime` | Tuple-of-names handshake is a code smell with no upside once `Cell` already owns the runtime object. |
| Keep `CellRuntimeState` frozen and push all mutation into a separate installer | More files, same behavior; contract-honesty change is the actual win. |
| Hash `FunctionClamp` by `fn is fn` (identity) | Documented footgun; no reason to keep it given a cheap fingerprint is available. |

---

## 13. Open questions

None at spec-lock time. All design decisions D1–D8 are user-confirmed.
