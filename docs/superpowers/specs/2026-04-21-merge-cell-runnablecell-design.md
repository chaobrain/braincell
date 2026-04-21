# Merge `Cell` and `RunnableCell` into a Single `Cell` Class

**Date:** 2026-04-21
**Status:** Design approved; ready for implementation plan.
**Scope:** `braincell/_multi_compartment/`, `braincell/__init__.py`, `braincell/compute/_runtime.py`, related tests and examples.

## 1. Goals

Revert the two-class split (`Cell` — mutable declaration frontend / `RunnableCell(HHTypedNeuron)` — frozen runtime facade). Unify into a single `Cell(HHTypedNeuron)` class with an explicit `init_state()` method that transitions the cell from a **DECLARING** phase to an **INITIALIZED** phase. After `init_state()`, `paint` / `place` / all configuration setters raise. `reset()` drops the runtime and per-step state, re-opening the cell for mutation (paint / place rules and configuration values are preserved). `run()` auto-calls `init_state()` for convenience; every other runtime method raises on DECLARING.

### Non-goals

- No change to CV lowering, runtime state layout, channel mechanism registry, integrator API, morphology, I/O, visualization, or filter subsystems.
- No performance changes expected; `init_state()` executes exactly the same lowering that `build.build` does today.
- No deprecation window. `RunnableCell` and `build()` are removed outright. The package is pre-1.0 and users are expected to migrate.

### Success criteria

- `braincell.Cell` is the single importable multi-compartment class. `from braincell import RunnableCell` raises `ImportError`. `braincell._multi_compartment.build` does not exist.
- `cell = Cell(morpho, ...); cell.paint(...); cell.place(...); cell.init_state(); cell.run(dt=..., duration=...)` executes end to end.
- `cell.paint(...)` / `cell.place(...)` / `cell.cv_policy = ...` / `cell.V_th = ...` / `cell.V_init = ...` / `cell.solver = ...` / `cell.spk_fun = ...` after `init_state()` raise `RuntimeError`.
- `cell.init_state()` called twice raises `RuntimeError`.
- `cell.reset()` from DECLARING raises `RuntimeError`.
- `cell.reset(); cell.paint(...); cell.init_state()` works (iterative workflow).
- `cell.run(...)` in DECLARING automatically invokes `init_state()` first, then runs.
- Existing behavior (probes, channel wiring, clamp table, axial operator, membrane-current pipeline, bridge, run loop) is unchanged.
- All tests pass after migration edits. Examples / notebooks updated.

## 2. Lifecycle state machine

Two phases, stored on the instance as `self._initialized: bool`:

```
               Cell(...)  ──►  DECLARING ──init_state()──►  INITIALIZED
                                   ▲                             │
                                   └───────── reset() ───────────┘

  paint / place / cv_policy= / V_th= / V_init= / solver= / spk_fun=
      DECLARING   → mutate
      INITIALIZED → RuntimeError("Cannot <action> after init_state(); call reset() first.")

  init_state()
      DECLARING   → lower CVs + runtime + states; set _initialized=True
      INITIALIZED → RuntimeError("init_state() already called; call reset() first.")

  reset()
      DECLARING   → RuntimeError("reset() requires init_state() first.")
      INITIALIZED → drop runtime+states+installed nodes; set _initialized=False

  run(dt=, duration=)
      DECLARING   → auto-call init_state(), then run
      INITIALIZED → run

  update / pre_integral / compute_derivative / post_integral /
  sample_probe / sample_probes / get_*_state / get_*_layouts /
  set_state / get_runtime_node / get_ion / mech_table /
  point_tree / point_scheduling / runtime / current_time / etc.
      DECLARING   → RuntimeError("<method> requires init_state() first.")
      INITIALIZED → execute
```

### Guard helpers (private)

```python
def _raise_if_initialized(self, action: str) -> None:
    if self._initialized:
        raise RuntimeError(
            f"Cannot {action} after init_state(); call reset() first."
        )

def _raise_if_not_initialized(self, action: str) -> None:
    if not self._initialized:
        raise RuntimeError(f"{action} requires init_state() first.")
```

Every DECLARING mutator calls `_raise_if_initialized`; every INITIALIZED method (except `run`) calls `_raise_if_not_initialized`. `run` inlines an auto-init branch.

### Name collision: `reset()` vs `reset_state()`

`Cell.reset_state(batch_size=None)` is the existing brainstate lifecycle hook — it reseeds `V` / `spike` / `current_time` in place and re-seeds channel states, requires INITIALIZED, and does **not** change the phase.

`Cell.reset()` is the new phase-transition method — it drops the runtime entirely and flips the cell back to DECLARING.

Both docstrings must call out the distinction explicitly.

## 3. `Cell` API surface

```python
class Cell(HHTypedNeuron):
    __module__ = "braincell"

    # ---- Construction ---------------------------------------------------
    def __init__(
        self,
        morpho: Morphology,
        *,
        cv_policy: CVPolicy | None = None,
        V_th = -75 * u.mV,
        V_init = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        name: str | None = None,
    ) -> None:
        HHTypedNeuron.__init__(self, size=(1,), name=name, **build_placeholder_ions())
        # ... validate morpho type.
        # ... _declaration_morpho = morpho  # never mutated; restored on reset()
        # ... _morpho = morpho              # swapped to clone inside init_state()
        # ... store cv_policy, V_th, V_init, spk_fun, ...
        # ... _solver_name, _solver_fn = _resolve_solver(solver)
        # ... _paint_rules = default_paint_rules(); _place_rules = ()
        # ... _cvs_cache = None; _cvs_cache_key = None
        # ... _current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        # ... _point_scheduling_cache = {}
        # ... _runtime = None; _point_tree = None; _axial_jax = None
        # ... _runtime_installed_names = ()
        # ... _initialized = False
        # Eagerly compute self.cvs for early validation of cv_policy.

    # ---- DECLARING-phase mutators (guarded) -----------------------------
    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":

    @cv_policy.setter  # guarded
    @V_th.setter       # guarded
    @V_init.setter     # guarded
    @solver.setter     # guarded
    @spk_fun.setter    # NEW: was immutable on Cell; now guarded setter

    # ---- Preview / phase-agnostic (work in both phases) -----------------
    @property
    def morpho(self) -> Morphology          # returns declaration morpho pre-init;
                                            # cloned morpho post-init (swapped in
                                            # step 6 of init_state, restored in step 7
                                            # of reset).
    @property
    def cv_policy(self) -> CVPolicy
    @property
    def paint_rules / place_rules / V_th / V_init / solver / solver_name / spk_fun / name
    @property
    def cvs                 # lazy preview pre-init; becomes runtime-owned post-init
    @property
    def n_cv: int
    def get_spike(self, last_V, next_V)     # pure: reads V_th + spk_fun only; used
                                            # internally by init_state (step 12).

    # ---- Transition -----------------------------------------------------
    def init_state(self, batch_size=None) -> None
    def reset(self) -> None

    # ---- INITIALIZED-phase API -----------------------------------------
    def run(self, *, dt, duration) -> RunResult   # auto-inits if DECLARING
    def update(self, I_ext=None)
    def pre_integral(self, I_ext=0.0)
    def compute_derivative(self, I_ext=0.0)
    def compute_membrane_derivative(self, V, I_ext=0.0)
    def compute_axial_derivative(self, V)
    def compute_voltage_derivative(self, V, I_ext=0.0)
    def post_integral(self, I_ext=0.0)
    def reset_state(self, batch_size=None)
    def sample_probe(self, name: str) / sample_probes()
    def mech_table() -> MechanismObjectTable
    def point_tree() / point_scheduling(max_group_size=32, algorithm="dhs")
    def get_point_state / get_cv_state / get_state / set_state
    def get_point_layouts / get_cv_layouts / expected_state_shape / get_runtime_node / get_ion

    @property
    def runtime: CellRuntimeState
    @property
    def layouts / voltage_shape / varshape / n_point / n_compartment / pop_size
    @property
    def current_time

    # ---- Internals ------------------------------------------------------
    def _cv_to_point(self, cv_values)
    def _point_to_cv(self, point_values)
    def _raise_if_initialized(self, action: str)
    def _raise_if_not_initialized(self, action: str)
    def _set_current_time(self, value)
```

### `init_state()` body

Linearized from today’s `build.build` plus the flag flip:

1. `self._raise_if_initialized("init_state()")`.
2. `morpho = clone_morpho(self._morpho)` — runtime owns its own topology so subsequent `reset()+init_state()` cycles do not alias the declaration morphology.
3. `cv_geo, cv_ids_by_branch = build_cv_geo(morpho, policy=self.cv_policy, paint_rules=self.paint_rules)`.
4. `cv_mech = init_cv_mech(len(cv_geo)); apply_paint_rules(...); apply_place_rules(...)`.
5. `cvs = tuple(assemble_cv(cv_geo=piece, mech=cv_mech[piece.id]) for piece in cv_geo)`.
6. `self._morpho = morpho` (swap the clone into `_morpho`; `_declaration_morpho` is untouched and will be restored on `reset()`). Assign `self._cvs_cache = cvs` with a matching key so `self.cvs` returns the final tuple post-init.
7. `self._point_tree = build_point_tree(morpho, cvs=cvs)`.
8. `self._runtime = CellRuntimeState.from_cell(self)`.
9. `install_cell_runtime(self, self._runtime)` — registers channel / ion submodules on the cell.
10. `v_initializer = self.V_init if self.V_init is not None else cv_value_vector(self, attr_name="v")`.
11. `self.V = DiffEqState(braintools.init.param(v_initializer, self.varshape, batch_size))`.
12. `self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))`.
13. `self._current_time_state.value = 0.0 * u.ms`.
14. `point_V = self._cv_to_point(self.V.value); for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values(): channel.init_state(point_V, batch_size=batch_size)`.
15. `self._axial_jax = jnp.asarray(build_cv_axial_operator(self, point_tree=self._point_tree, scheduling=self.point_scheduling(algorithm="dhs")), dtype=jnp.float64)`.
16. `self._initialized = True`.

### `reset()` body

1. `self._raise_if_not_initialized("reset()")`.
2. `uninstall_cell_runtime(self)` — remove installed channel / ion submodules from the brainstate Module tree (see §5). `self._runtime_installed_names = ()`.
3. `del self.V`; `del self.spike`. (They were created as attributes during `init_state()`; `delattr` removes them cleanly so re-init can re-create fresh `DiffEqState` / `ShortTermState` instances.)
4. `self._current_time_state.value = 0.0 * u.ms`.
5. `self._runtime = None; self._point_tree = None; self._axial_jax = None`.
6. `self._point_scheduling_cache.clear()`.
7. `self._morpho = self._declaration_morpho` — restore the original declaration morphology so the next `init_state()` clones from the user-supplied tree, not from a previous clone.
8. `self._cvs_cache = None; self._cvs_cache_key = None` — force `self.cvs` to recompute from declaration on next access.
9. `self._initialized = False`.

### `run()` body

```python
def run(self, *, dt, duration):
    if not self._initialized:
        self.init_state()
    return run_module.run(self, dt=dt, duration=duration)
```

All other INITIALIZED-phase methods start with `self._raise_if_not_initialized("<method>()")`.

### Setters — the `spk_fun` change

`spk_fun` is currently an immutable `Cell` attribute (property getter only, no setter). Under the new design all config values must be guarded uniformly; `spk_fun` therefore gains a guarded setter matching `V_th` / `V_init` / `solver`.

## 4. File layout changes

### Delete

- `braincell/_multi_compartment/runnable.py`
- `braincell/_multi_compartment/runnable_test.py`
- `braincell/_multi_compartment/build.py`
- `braincell/_multi_compartment/build_test.py`

### Rewrite

- `braincell/_multi_compartment/cell.py` — absorbs everything previously in `runnable.py` plus the `init_state()` / `reset()` pipeline from `build.py`. `Cell.build` method is removed. `_resolve_solver` remains local.
- `braincell/_multi_compartment/cell_test.py` — gains migrated tests from `runnable_test.py` and `build_test.py`. Adds new tests for freeze semantics, double-init error, reset round-trip, run auto-init.
- `braincell/_multi_compartment/__init__.py`:
  ```python
  from .cell import Cell
  from .run import RunResult
  __all__ = ["Cell", "RunResult"]
  ```
- `braincell/__init__.py` — drop `RunnableCell` re-export. Keep `Cell` / `RunResult`.

### Type-annotation updates

Rename `host: "RunnableCell"` → `host: "Cell"` and adjust the `TYPE_CHECKING` imports in:

- `braincell/_multi_compartment/bridge.py`
- `braincell/_multi_compartment/currents.py`
- `braincell/_multi_compartment/probes.py`
- `braincell/_multi_compartment/clamp_table.py`
- `braincell/_multi_compartment/run.py`

Symbol names inside those modules (`rcell` variables, `RunResult`, helpers) stay unchanged; only the type string and the imported name flip.

### New helper: `uninstall_cell_runtime`

`braincell/compute/_runtime.py` grows a function inverse to `install_cell_runtime`:

```python
def uninstall_cell_runtime(cell: "Cell") -> None:
    """Remove channel / ion submodules previously installed on ``cell``.

    Must undo exactly what ``install_cell_runtime`` did: delattr every
    attribute name it set, and drop cached references (``C``, ``V_th``
    where applicable) so a subsequent ``init_state()`` re-populates
    them cleanly.
    """
```

`install_cell_runtime` must record the list of attribute names it installs (a private `_runtime_installed_names: tuple[str, ...]` attribute on the cell, or return the names from `install_cell_runtime` and have the caller stash them). Simplest shape: have `install_cell_runtime` return `tuple[str, ...]`; `init_state()` stashes it on `self._runtime_installed_names`; `reset()` iterates and `delattr` each, then clears the tuple.

### CLAUDE.md update

Revise the **“Multi-compartment two-class split”** section. Replace with a description of the single-`Cell` lifecycle (DECLARING / INITIALIZED phases, `init_state()`, `reset()`, `run()` auto-init). Remove references to `RunnableCell`, `build()`, the one-way arrow, and the `__new__` + `_preinit` + `_attach_runtime` staging API. Keep the runtime-pipeline notes (membrane-current sum, `CellRuntimeState`, `ClampActiveTable`, `cv_area`) — they are unchanged.

## 5. Migration — existing callers

### Internal callers

- `braincell/compute/_runtime.py` — `install_cell_runtime(rcell, runtime)` signature already takes a generic `cell`-like argument; rename the parameter for clarity but keep the behavior. Add `uninstall_cell_runtime` as described above.
- `braincell/compute/_runtime_test.py` — audit for any `RunnableCell` import or `cell.build()` call, migrate to `Cell` + `init_state()`.
- Any other test that constructed a `RunnableCell` through `RunnableCell.__new__` + `_preinit` + `_attach_runtime` — move to `Cell(...) ... init_state()`.

### Examples / notebooks

Replace the pattern:

```python
cell = Cell(morpho, ...)
cell.paint(...); cell.place(...)
rcell = cell.build()
result = rcell.run(dt=..., duration=...)
```

with:

```python
cell = Cell(morpho, ...)
cell.paint(...); cell.place(...)
result = cell.run(dt=..., duration=...)        # auto-inits
# or, when you want inspection before running:
# cell.init_state(); result = cell.run(dt=..., duration=...)
```

Files to update (non-exhaustive — search for `build(` / `RunnableCell` / `rcell` in):

- `examples/multi_compartment/*.ipynb` — `cell.ipynb`, `morphology.ipynb`, `quad.ipynb`, `vis.ipynb`, `neuron_diff.ipynb`, and any others.
- `examples/multi_compartment/*.py` — any standalone scripts that exercise `build()` / `RunnableCell`.

## 6. Testing strategy

Co-located, `*_test.py` discipline per project convention.

### New / migrated tests in `cell_test.py`

- `test_declaration_phase_mutations` — `paint`, `place`, each setter succeeds before `init_state()`.
- `test_init_state_transitions_to_initialized` — after `init_state()`, `_initialized` is True; runtime / point_tree / axial_jax / V / spike exist.
- `test_init_state_double_call_raises` — second `init_state()` raises `RuntimeError` mentioning `reset()`.
- `test_paint_after_init_raises` / `test_place_after_init_raises` / `test_config_setter_after_init_raises` (parametrized over `cv_policy`, `V_th`, `V_init`, `solver`, `spk_fun`).
- `test_reset_from_declaring_raises`.
- `test_reset_round_trip` — init → reset → `_initialized` False, runtime/point_tree/axial_jax None, V/spike attribute-missing, paint/place/setters allowed again; second `init_state()` produces a working cell.
- `test_reset_preserves_declaration` — paint/place/V_th/V_init/solver/spk_fun all match across reset.
- `test_run_auto_inits_from_declaring` — `cell.run(dt=, duration=)` without an explicit `init_state()` succeeds and leaves `_initialized` True.
- `test_run_after_init_does_not_re_init` — idempotency check (no extra runtime allocation).
- `test_runtime_method_requires_init` — parametrize over `update`, `sample_probes`, `sample_probe`, `get_point_state`, `get_cv_state`, `get_state`, `mech_table`, `point_tree`, `point_scheduling`, `runtime`, `layouts`, `current_time`, `compute_membrane_derivative`, etc. Each raises `RuntimeError` in DECLARING.
- `test_reset_state_vs_reset_distinct` — `reset_state` stays in INITIALIZED and reseeds V; `reset` flips to DECLARING.
- Existing `runnable_test.py` / `build_test.py` coverage (channel nodes wired, probes, membrane currents, axial operator cached, run loop produces traces, clamp table precomputed, `mech_table` shape) — move into `cell_test.py`, adapting imports / construction calls.

### Regression

- `braincell/_multi_compartment/*_test.py` (bridge, currents, probes, run, clamp_table) — run existing suites; they should pass once the `RunnableCell` annotation rename is done, since behavior is identical.
- `braincell/compute/_runtime_test.py` — re-run after `uninstall_cell_runtime` addition.
- Full `pytest braincell/` run as the final gate.

### Non-covered-by-tests migration (manual verification)

- Launch notebooks under headless execution (`jupyter nbconvert --to notebook --execute`) for `examples/multi_compartment/*.ipynb` after migration edits, confirm each still runs end-to-end.

## 7. Risks and mitigations

- **Channel node deregistration completeness.** `install_cell_runtime` currently sets attributes by name via `setattr`. Missing one in `uninstall_cell_runtime` leaves a stale node across `reset()`, and a subsequent `init_state()` will collide on attribute re-assignment. Mitigation: have `install_cell_runtime` **return** the tuple of installed attribute names; `init_state()` stashes that tuple; `reset()` iterates and `delattr`s exactly those names. No guessing.
- **`self.V` / `self.spike` attribute lifecycle.** `brainstate.State` subclasses registered on `Module` instances are tracked in the node tree. `del self.V` should also remove the state from that tree. Plan: verify brainstate’s behavior on attribute deletion in the first implementation step; if `delattr` is insufficient, add explicit deregistration through whatever API brainstate provides. Cover with `test_reset_round_trip` (re-`init_state()` after reset must yield a valid cell with fresh `V`).
- **`cvs` identity drift.** Pre-init `cvs` comes from the declaration morphology; post-init `cvs` is computed against the cloned morphology stored on the cell. Callers holding a reference obtained pre-init will continue to see the pre-init tuple (it is immutable). This is acceptable — they can call `cell.cvs` again post-init. Document explicitly in the `cvs` property docstring.
- **`brainstate.environ['dt']` requirement inside `update`.** Unchanged from today; `run()` sets it before entering the scan, so auto-init + run path is safe.
- **Hidden `RunnableCell` imports.** Grep for `RunnableCell` and `from .runnable` across the repo before deleting. Anything left after migration should be caught by `pytest`.
