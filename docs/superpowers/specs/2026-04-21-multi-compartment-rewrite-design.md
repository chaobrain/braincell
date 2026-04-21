# Multi-compartment `Cell` rewrite — design

**Date:** 2026-04-21
**Branch:** `morpho_1`
**File being replaced:** `braincell/_multi_compartment.py` (~1100 LOC)

## Motivation

The current `Cell` class conflates three responsibilities (declaration,
lazy lowering, runtime facade) behind a five-flag dirty-state machine,
and the CV↔point bridging, external-current handling, and runtime
compile/install steps are scattered across several methods. A code
review surfaced 15 issues, the three highest-impact being:

1. **External `I_ext` silently dropped** in
   `compute_membrane_derivative` when no brainstate `current_inputs`
   are registered.
2. **NaN risk** in `_normalize_external_current` when total-current
   input is `(n_point,)`-shaped — division by a point-area array with
   zeros at non-midpoint slots.
3. **Unit-stripping cast** via `jnp.asarray(..., dtype=jnp.float64)` on
   every `compute_axial_derivative` call and no explicit unit check.

The rewrite fixes those bugs while reshaping the module into two
small, focused classes with no dirty flags.

## Architecture

Two classes, sharp boundary:

```
Cell (declaration, mutable)        RunnableCell (runtime, frozen)
----------------------------       -------------------------------
morpho, cv_policy                  cvs, point_tree, runtime
paint_rules, place_rules           V, spike, C, V_th
V_th, V_init, spk_fun, solver      current_time

.paint(region, *mechanisms)        .update(I_ext=0)
.place(locset, *mechanisms)        .run(*, dt, duration) -> RunResult
.build() -> RunnableCell           .compute_derivative(I_ext=0)
.cvs  (preview only)               .compute_voltage_derivative(V, I_ext=0)
                                   .compute_membrane_derivative(V, I_ext=0)
                                   .compute_axial_derivative(V)
                                   .pre_integral / .post_integral
                                   .reset_state / .get_spike
                                   .sample_probe / .sample_probes
                                   .mech_table
                                   .get_state / .set_state
                                   .get_point_state / .get_cv_state
                                   .get_point_layouts / .get_cv_layouts
                                   .get_runtime_node / .get_ion
```

`Cell.build()` runs the full pipeline exactly once:
`build_cv_geo → apply_paint/place_rules → assemble_cv →
build_point_tree → CellRuntimeState.from_cell → install channels →
allocate V, spike, C, V_th → cache jnp axial operator` and returns a
`RunnableCell`. Any post-build mutation on the `Cell` is fine — it
produces a new `RunnableCell` on the next `build()`.

No dirty flags anywhere. No `_ensure_runtime_compiled` /
`_ensure_runtime_ready` guard cascade.

## File layout

Split the single file into one package:

```
braincell/_multi_compartment/
  __init__.py       # re-exports Cell, RunnableCell, RunResult
  cell.py           # Cell (declaration)
  runnable.py       # RunnableCell (runtime facade)
  build.py          # build() pipeline
  currents.py       # total_membrane_current + clamp/ext helpers
  bridge.py         # cv_to_point / point_to_cv (renamed scatter/gather)
  run.py            # run(), time/trace helpers
  probes.py         # sample_probe / sample_probes + helpers
```

Per-file co-located tests under the same directory (CLAUDE.md rule).

`braincell/__init__.py` keeps re-exporting `Cell`, `RunnableCell`,
`RunResult` from the new package.

## `Cell` surface

```python
class Cell:
    def __init__(
        self,
        morpho: Morphology,
        *,
        cv_policy: CVPolicy | None = None,
        V_th: Quantity = -75 * u.mV,
        V_init: Quantity | Callable | None = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        name: str | None = None,
    ): ...

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell": ...
    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell": ...

    cv_policy: CVPolicy      # property + setter (invalidates preview cache)
    V_th: Quantity
    V_init: Quantity | Callable | None
    solver: str | Callable

    @property
    def cvs(self) -> tuple[CV, ...]: ...          # preview only; no runtime install
    @property
    def paint_rules(self) -> tuple[PaintRule, ...]: ...
    @property
    def place_rules(self) -> tuple[PlaceRule, ...]: ...

    def build(self) -> "RunnableCell": ...
```

`.cvs` preview runs geo + rule application but never lowers runtime.
It is memoized with a cache key derived from
`(id(morpho), cv_policy, paint_rules, place_rules)` so repeated
inspection without mutation is cheap.

`place` dedups identical rules via a new `merge_place_rules` helper,
mirroring `merge_paint_rules` (fixes bug #11).

## `RunnableCell` surface

`HHTypedNeuron` subclass, constructed only via `Cell.build()`.
All mutable fields are `brainstate.State` objects; topology/runtime
are plain attributes set in `__init__` and never reassigned.

Signatures identical to current `Cell` methods except:

- no `init_state` — that work is inside `build()`.
- `reset_state(batch_size=None)` reseeds `V.value` + replays
  `channel.reset_state(point_V)`. Cannot re-lower — if topology
  changes, user must rebuild from the `Cell`.
- `current_time` is a read-only property over a `ShortTermState`.
- `run(*, dt, duration)` does **not** mutate `current_time` inside
  the `for_loop`; it only sets the final post-loop value (fixes
  bug #6 clarity).

Dropped entirely:
- `_cached_voltage_linearizer` (unused outside a dead helper).
- `_value_dirty` flag (unused, bug #4).
- All dirty flags and the `_mark_dirty` method.

## Current summation pipeline (fixes bugs #1, #2, #5, #7)

`braincell/_multi_compartment/currents.py`:

```python
def total_membrane_current(
    runtime: CellRuntimeState,
    *,
    V_cv: Quantity,
    I_ext: Quantity | float,
    t: Quantity,
    host: "RunnableCell",
) -> Quantity:
    point_V = bridge.cv_to_point(V_cv, runtime)

    # (1) external -> point-space density; then brainstate current_inputs
    # accumulate on top. SC pattern: external becomes init, not an arg.
    I_ext_density = _normalize_ext_to_point_density(I_ext, runtime)
    I_point = host.sum_current_inputs(I_ext_density, point_V)

    # (2) clamp density from cached active-point table
    I_point = I_point + _clamp_density(runtime, t=t)

    # (3) channel currents
    for key, ch in host.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
        try:
            contrib = ch.current(point_V)
        except Exception as exc:
            raise ValueError(f"Error in channel '{key}':\n{ch}\n{exc}") from exc
        if contrib is not None:
            I_point = I_point + contrib

    return bridge.point_to_cv(I_point, runtime)
```

**Normalization rules** (single source of truth for `I_ext` shapes):

| `I_ext` shape / unit               | Behavior                                                       |
|------------------------------------|----------------------------------------------------------------|
| scalar `0` (python zero)           | return zeros `(n_point,)` in nA/cm²                            |
| scalar density (nA/cm²)            | broadcast to `(n_point,)`                                      |
| scalar total current (nA)          | divide by `(n_cv,)` area → scatter to midpoints                |
| `(n_cv,)` density                  | scatter to midpoints                                           |
| `(n_cv,)` total current            | divide by area → scatter                                       |
| `(n_point,)` density               | pass through                                                   |
| `(n_point,)` total current         | **reject with ValueError**                                     |

The rejected case is where the previous code divided a full point
vector by a midpoint-scatter area and produced NaN. We surface a
clear error instead.

**Clamp active-point table** built at compile time inside
`CellRuntimeState.from_cell`:

```python
@dataclass(frozen=True)
class ClampActiveTable:
    ids: np.ndarray         # unique active point ids
    area: np.ndarray        # membrane area at those ids (cm^2)
```

Positive-area validation raises in the constructor (not in the hot
path). The runtime hot path reads `runtime.clamp_active_table` once
per step — no filter walk per `update()`.

## Axial derivative (fixes bug #9)

`compute_axial_derivative` uses a pre-cached jnp array:

```python
# during build():
self._axial_jax = jnp.asarray(
    build_cv_axial_operator(self, point_tree=..., scheduling=...),
    dtype=jnp.float64,
)

# hot path:
def compute_axial_derivative(self, V):
    V_decimal = V.to_decimal(u.mV).astype(jnp.float64)
    axial = -jnp.matmul(V_decimal, self._axial_jax.T)
    return axial * (u.mV / u.ms)
```

No per-step `jnp.asarray` cast. Unit annotation preserved via the
explicit `mV / ms` return.

## `run()` time handling (bug #6 clarity)

`run()` no longer writes `self._current_time.value` inside the
`for_loop`. It pins the final time once after the loop completes:

```python
with brainstate.environ.context(dt=dt):
    start_t = self.current_time
    rel = u.math.arange(0.0 * u.ms, duration, brainstate.environ.get_dt())
    if int(rel.shape[0]) == 0:
        raise ValueError(...)
    times = start_t + rel

    def _step(t):
        with brainstate.environ.context(t=t):
            self.update()
            return tuple(s for s in self.sample_probes().values())

    traces = brainstate.transform.for_loop(_step, times)
    self._current_time.value = start_t + int(times.shape[0]) * brainstate.environ.get_dt()
```

`t` is propagated via `brainstate.environ['t']` for clamp evaluation
instead of a mutable `_current_time` write inside the scan.

## V init behavior

`RunnableCell.__init__` resolves `V_init` once:

- `V_init is None` → per-CV resting potential from `cv.v`.
- `V_init` is a `Quantity` → broadcast to `(n_cv,)`.
- `V_init` is a callable → called with `(n_cv,)` shape tuple.

`reset_state(batch_size=None)` re-applies the same resolver. No
unused helper branches.

## Migration

**Breaking change.** Existing callers:

```python
cell = Cell(morpho, cv_policy=...)
cell.paint(...); cell.place(...)
cell.init_state()
cell.run(dt=..., duration=...)
```

Migrate to:

```python
cell = Cell(morpho, cv_policy=...)
cell.paint(...); cell.place(...)
rcell = cell.build()
rcell.run(dt=..., duration=...)
```

No compat shim — user accepted breaking API for cleaner final shape.

## Tests

- Rewrite `braincell/_multi_compartment_test.py` and
  `braincell/_multi_compartment_solver_test.py` against the new API.
- Add focused regression tests:
  - `test_external_current_not_dropped` — `rcell.update(I_ext=1*u.nA)`
    with no registered current_inputs must affect V.
  - `test_point_shape_total_current_rejected` — `(n_point,)`-shaped
    nA input raises ValueError.
  - `test_place_dedup` — two identical `place` calls collapse to one
    rule (`len(cell.place_rules) == 1`).
  - `test_clamp_active_table_validates_area` — clamp on zero-area
    point raises at `build()`.
  - `test_rebuild_after_mutation` — `cell.build()` twice with
    different paint rules produces distinct `RunnableCell` topologies.

## Out of scope

- `brainstate`/`braintools` API drift handling. We assume the current
  pinned versions.
- Any `_base.HHTypedNeuron` changes. Left untouched.
- `quad/`, `cv/`, `compute/`, `mech/`, `morph/` are unchanged except
  for the small `CellRuntimeState` extension for `clamp_active_table`
  (backward-compatible: new field with a default of `None`).

## Risk / rollback

- Branch `morpho_1` already has structural work; rewrite lands on
  top. If it regresses, `git revert` the squash merge.
- Keep old `_multi_compartment.py` on a parallel backup branch
  (`morpho_1-multi-compartment-legacy`) for quick diffing during the
  migration.
