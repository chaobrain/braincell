# Hidden-state class by host: `HiddenState` for `SingleCompartment`, `HiddenGroupState` for `Cell`

## Requirement

> 在使用 `SingleCompartment` 时，`V` 等状态变量必须是 `brainstate.HiddenState`；
> 在定义 `Cell` 时，`V` 等状态变量必须是 `brainstate.HiddenGroupState`。

`brainstate.HiddenGroupState` is a `HiddenState` whose **trailing axis
indexes independent hidden states** — `varshape = value.shape[:-1]`,
`num_state = value.shape[-1]`. It exists so eligibility-trace learners
(brainscale) can treat one array as N separately-traced hidden units
instead of one.

That maps exactly onto braincell's two neuron models, because they differ
in precisely this way:

| Model | `varshape` | example `V.shape` | trailing axis is |
|---|---|---|---|
| `SingleCompartment(5)` | `(5,)` | `(5,)` | neurons |
| `Cell(pop_size=4)` | `(4, n_cv)` | `(4, 1)` | **compartments (CV)** |

A `SingleCompartment` has no compartment axis, so its states are single
hidden states. A `Cell` carries one compartment axis at the end, so each
compartment is an independent hidden state. Today both use
`DiffEqState`, which extends plain `HiddenState` — so the `Cell` side is
wrong.

## Current state

`braincell/quad/protocol.py:28` defines the one class both models share:

```python
class DiffEqState(brainstate.HiddenState): ...
```

Every solver in `braincell/quad/` consumes it, and
`braincell/quad/_util.py:45` selects integrable states with
`isinstance(value, DiffEqState)`.

It is not only `V`. A `Cell` subtree owns diffeq states created by shared
channel / ion / synapse code that `SingleCompartment` uses too — measured
on a 3-point soma with HH channels:

```
Cell(pop_size=4) varshape=(4, 1)
  V                          (4, 1)     <- CV space   (n_cv)
  ion_channels/na/…/na/p     (4, 3)     <- point space (n_point)
  ion_channels/na/…/na/q     (4, 3)
  ion_channels/k/…/k/p       (4, 3)
```

`V` lives in **CV space** (`n_cv`) while gates live in **point space**
(`n_point`). Both trailing axes are spatial, so both are correct group
axes — they just have different lengths, which is fine because each state
groups independently.

`spike` and `_current_time_state` are `ShortTermState` and stay unchanged;
the requirement is about hidden states.

### Every creation site

| # | Site | Class today |
|---|---|---|
| 1 | `channel/_base.py:159` | `DiffEqState` — gate channels |
| 2 | `channel/_base.py:340` | `DiffEqState` — Markov channels |
| 3 | `ion/_base.py:295` | `DiffEqState` — dynamic `Ci` |
| 4 | `ion/_base.py:584` | `DiffEqState` — `_Species.init`, diffeq species |
| 5 | `ion/_base.py:586` | `HiddenState` — `_Species.init`, algebraic species |
| 6 | `ion/_base.py:600` | `HiddenState` — `_Species.reset` |
| 7 | `ion/_base.py:664` | `HiddenState` — `_Conserve.writeback` |
| 8 | `synapse/markov.py:44` | `DiffEqState` — `_syn_uS_state` |
| 9 | `synapse/markov.py:49` | `DiffEqState` — `_syn_state` |
| 10 | `_single_compartment/base.py:170` | `DiffEqState` — `V` |
| 11 | `_multi_compartment/cell.py:556` | `DiffEqState` — `V` |

Sites 1–9 are in code shared by both hosts, so the class cannot be chosen
statically at the creation site.

## Prerequisite: `Cell` always carries a population axis

`brainstate.HiddenGroupState._check_value` requires `value.ndim >= 2`.
`Cell`'s current default `pop_size=()`
(`_multi_compartment/cell.py:196`, `:2835`) makes `varshape = (n_cv,)`,
so every state is 1-D and the stock class rejects it.

**Resolution: make the population axis mandatory.** `pop_size` defaults
to `1`, and an explicit rank-0 `pop_size=()` is rejected.

```
Cell(morpho)              -> V.shape == (1, n_cv)
Cell(morpho, pop_size=4)  -> V.shape == (4, n_cv)
Cell(morpho, pop_size=()) -> ValueError
Cell(morpho, pop_size=None) -> (1,)     # None means "unspecified"
```

This is a deliberate breaking change, affordable at version `0.1.0`, and
it buys three things:

* `brainstate.HiddenGroupState` is usable **unmodified**. No braincell
  subclass weakens a parent's validation, so a future brainscale that
  re-validates cannot break on braincell states.
* Rank-0 stops being a special case. `quad/_staggered_test.py:371`
  (`test_midpoint_clamp_is_not_double_counted_with_population_axis`)
  exists precisely because rank-0 and rank-1 populations once disagreed —
  the midpoint clamp was double-counted. Removing rank-0 removes that
  whole bug class.
* It matches `network/core.py:36`, which already rejects a cell whose
  `pop_size` is not one-dimensional.

Rejected alternatives:

* **Change only the default, keep `()` legal.** Strictly dominated: the
  1-D path stays reachable, so the subclass relaxation is still required
  *and* the shape break is still paid.
* **Relax `ndim >= 2` in a braincell subclass, keep `pop_size=()`.** No
  shape break, but braincell then diverges from the brainstate contract at
  exactly the point brainscale depends on.
* **Relax `brainstate.HiddenGroupState` upstream.** Semantically cleanest,
  but needs a brainstate release before braincell CI can pass. Still worth
  doing later; it is orthogonal to this change.

## Design

### 1. Mandatory population axis — `_multi_compartment/cell.py`

* `Cell.__init__` signature default `pop_size: brainstate.typing.Size = 1`.
* `_normalize_pop_size`: `None` → `(1,)`; `()` / `[]` → `ValueError`
  naming the invariant; everything else unchanged.
* `docs/design/cell.md:258` currently states `Cell.V` 的公开尺寸固定为
  `n_cv`; update it to `pop_size + (n_cv,)`.

### 2. One new state class — `braincell/quad/protocol.py`

```python
class DiffEqGroupState(DiffEqState, brainstate.HiddenGroupState):
    """A DiffEqState whose trailing axis indexes compartments."""
    __module__ = 'braincell'
```

Verified on brainstate 0.5.2:

* MRO linearizes to `DiffEqGroupState → DiffEqState → HiddenGroupState →
  HiddenState → ShortTermState → State`, and `DiffEqState.__init__`'s
  cooperative `super().__init__()` reaches `HiddenGroupState.__init__`.
* `isinstance(s, DiffEqState)` stays `True`, so `quad/_util.py:45` and
  every solver need **no change**.
* `derivative` / `diffusion` setters, `.value` writes, and `get_value(i)`
  all work.
* `graph.treefy_split`/`treefy_merge`, `graph.clone`,
  `transform.for_loop`, `transform.jit`, and `to_state_ref()/to_state()`
  round-trip the subclass with `name2index` intact.

Algebraic (non-diffeq) species use `brainstate.HiddenGroupState`
directly — no braincell subclass, so no name shadowing.

### 3. Host-scoped factory — `braincell/quad/protocol.py`

Sites 1–9 need the host's identity without changing ~15 `init_state`
signatures, so the host publishes it for the duration of the call:

```python
_GROUPED = contextvars.ContextVar('braincell_grouped_states', default=False)

@contextlib.contextmanager
def grouped_states(enabled: bool = True): ...

def diffeq_state(value) -> DiffEqState:              # DiffEqGroupState if grouped
def hidden_state(value) -> brainstate.HiddenState:   # HiddenGroupState if grouped
```

`contextvars` (not a bare global) so nesting, exceptions, and threads
restore correctly. Default `False` keeps a standalone
`braincell.channel.*` built outside any host on today's behaviour.

### 4. Hosts opt in explicitly

* `Cell.init_state` / `Cell.reset_state` → `grouped_states(True)`; site 11
  constructs `DiffEqGroupState` directly.
* `SingleCompartment.init_state` / `reset_state` → `grouped_states(False)`;
  site 10 stays `DiffEqState`. Setting it explicitly rather than relying
  on the default is what makes a `Network` holding both model types
  correct regardless of construction order.
* `Network.init_state` sets nothing — each cell scopes itself.

### 5. Route sites 1–9 through the factory

Mechanical substitution: `DiffEqState(x)` → `diffeq_state(x)`,
`brainstate.HiddenState(x)` → `hidden_state(x)`.

Site 7 (`_Conserve.writeback`) runs during simulation, not just
initialization, and only allocates when the attribute is not yet a
`State` — dead in normal flow because `_Species.init` ran first. It is
converted for consistency; the tests assert it does not fire mid-run.

## Tests

* **`braincell/quad/protocol_test.py`** (new) — `DiffEqGroupState`
  construction at 2-D / 3-D; 1-D raises (inherited guard);
  `isinstance` against `DiffEqState`, `brainstate.HiddenState`,
  `brainstate.HiddenGroupState`; `varshape` / `num_state` / `get_value` /
  `set_value`; `derivative` and `diffusion` round-trip; graph and
  transform round-trips; factory dispatch on and off; context nesting and
  restoration after an exception.
* **`_multi_compartment/cell_test.py`** — `pop_size` normalization
  (default `(1,)`, `None` → `(1,)`, `()` → `ValueError`); and for
  `pop_size=1` and `pop_size=4`, with HH channels, a kinetic ion, and a
  placed synapse: *every* `HiddenState` in the `Cell` subtree is a
  `brainstate.HiddenGroupState` whose `num_state` equals the expected
  `n_cv` / `n_point`. Same assertion after `reset_state()`.
* **`_single_compartment/base_test.py`** — every `HiddenState` in the
  subtree is **not** a `HiddenGroupState`, at `batch_size=None` and
  `batch_size=8`.
* **Numerical equivalence** — the state-class change is type-only and the
  `pop_size` change only adds a length-1 axis, so a `Cell` simulation must
  match the pre-change trace after `reshape(-1)`, and `SingleCompartment`
  must match bit-for-bit. `quad/_staggered_test.py:371` already asserts
  rank-0 ≡ rank-1 numerically and is the anchor for this.
* **Full suite** — `pytest braincell/` and `pre-commit run --all`.

## Pre-existing bugs the mandatory axis exposed

Making `pop_size` mandatory moved the population axis onto the *default*
path, where four latent rank assumptions had never been exercised. All
four were already wrong for an explicit `pop_size`; none are caused by
this change, and each is fixed here.

| # | Site | Bug |
|---|---|---|
| 1 | `_multi_compartment/cell.py`, six vis coercion helpers | `_layout_values_to_{point,cv}_space` rejected any field with rank > 1 ("only supports 1-D value fields"), so `Cell(pop_size=4).vis_cv(...)` already failed. Fixed with `_drop_population_axes`, which selects the single-member view and otherwise raises a message naming the offending field and `pop_size`. |
| 2 | `_compute/runtime.py:3001` | `_sync_runtime_ion` broadcast an ion baseline onto a hardcoded `(n_point,)`. Line 2953 already used `runtime.pop_size + (runtime.n_point,)`; the two now agree. |
| 3 | `_compute/runtime.py`, `_instantiate_runtime_node` | The dense-channel size fallback used `layout.point_mask.shape` without the population axis, so a channel whose parameters were all scalars got a rank-1 gate state. Under `jit`/`scan` this surfaced as a carry-signature mismatch (`float32[5]` vs `float32[1,5]`). `pop_size` is now threaded in. |
| 4 | `ion/_base.py`, `_Species.init` / `reset` | `braintools.init.param` passes a bare scalar through unbroadcast, so a species declared as e.g. `0.0 * u.mol / u.cm**2` (`CdpCR_MA2020_GrC.pumpca`) started rank-0 while its siblings were `varshape`. `_Conserve.writeback` then assigned the per-point value, silently growing the state mid-simulation. `_species_value` now broadcasts at allocation. |

A fifth site needed a genuine semantic fix rather than a missing axis:
`_extract_point_value` classified any 2-D buffer as *ragged*
(point × sub-values). With a population axis, 2-D is ambiguous — it can
also be pop × point. The predicate now compares the trailing axis against
the layout's own point-axis length (`point_mask.shape[0]` for dense,
`n_active` for sparse) instead of relying on rank.

## Migration surface

`Cell.V` and every `Cell` trace gain a leading length-1 axis by default.

Handled:

* 8 test files that build a `Cell` (`_compute/runtime_test.py`,
  `_compute/spatial_params_test.py`, `_compute/table_test.py`,
  `_multi_compartment/cell_test.py`,
  `_multi_compartment/currents_test.py`, `network/runtime_test.py`,
  `quad/_staggered_test.py`, plus the new tests above).
* `docs/design/cell.md`.
* The 8 `examples/neuron_compare/cell/*` builders that declared
  `pop_size=()` as their default; each is now `pop_size=1`. Their debug
  runners already `reshape(-1)` every trace, so nothing downstream moves.

Reviewed, no change needed:

* `docs/examples/` and `docs/getting_started/` notebooks — every
  `.V.value` consumer there is a `SingleCompartment`, which is untouched.
* `docs/tutorials/` and `docs/concepts/` notebooks — these build a `Cell`
  and plot `run(...)` probe traces, which go from `(T,)` to `(T, 1)`.
  Matplotlib renders that identically, so the notebooks stay correct;
  they are not executed by the test suite.

One test changed meaning rather than shape:
`quad/_staggered_test.py::test_midpoint_clamp_is_not_double_counted_with_population_axis`
compared `pop_size=()` against `pop_size=(1,)`. Both are `(1,)` now, so
the assertion had become vacuous. It now compares the default against
`pop_size=(3,)` and asserts every member reproduces the single-member
trace — the same invariant, restated for a world without rank-0.

## Out of scope

* brainscale / eligibility-trace integration. brainscale is not a
  braincell dependency; this change only makes the state classes correct
  so that integration is possible later.
* `spike`, `_current_time_state`, and other `ShortTermState`s.
* The upstream brainstate `ndim >= 2` relaxation.
