# `braincell._compute` simplification

Iteration 8 of the module-by-module simplification sweep. Target:
`braincell/_compute/` — the private lowering layer that turns a `Cell`'s
CV/node-tree declarations into the runtime structures the solver executes
against. 4444 production lines across 8 modules, 4247 lines of co-located
tests.

This is the largest package the sweep has touched. It is also the one where
the reviews turned up **three real defects** rather than only untidiness, so
the iteration leads with those.

## Baseline

```
$ pytest braincell/_compute -q
152 passed
```

| module | production lines |
|---|---|
| `bindings.py` | 1188 |
| `layouts.py` | 1037 |
| `state.py` | 757 |
| `ions.py` | 516 |
| `bridge.py` | 364 |
| `scheduling.py` | 229 |
| `table.py` | 220 |
| `__init__.py` | 33 |

## Defects

### 1. The `"no"` ion family is never auto-seeded, so `Kv1p5_MA2020_GrC` cannot be painted

`ions.py:107` seeds placeholder ions for three families:

```python
    for family_key in ("na", "k", "ca"):
```

while the function it calls to build them supplies four:

```
$ python -c "... build_placeholder_ions(size=(3,)).keys()"
placeholder families: ['ca', 'k', 'na', 'no']
```

So `_build_default_ions` constructs the `NonSpecific` placeholder and the loop
throws it away. Any channel that declares a `NonSpecific` owner then fails to
bind, because `_resolve_ion_instance_key` (`bindings.py:920-922`) finds no
candidate for family `"no"`. Reproduced on a two-branch cell with nothing but
the channel painted on it:

```
Kv1p5_MA2020_GrC       KeyError: "No ion candidates are registered for family 'no'."
Na_HH1952              OK   ions=['ca', 'k', 'na']
K_HH1952               OK   ions=['ca', 'k', 'na']
leaky                  OK   ions=['ca', 'k', 'na']
```

**Blast radius: one shipped channel.** A sweep of all 112 registered channel
classes for a `NonSpecific` entry in `root_type.__args__` or in
`current_owner_types` returns exactly `['Kv1p5_MA2020_GrC']`. The workaround —
hand-painting `mech.Ion("NonSpecificFixed", name="no")` — is what
`bindings_test.py:365-400` does, which is why no test catches this.

The family set is stated in **four** places across two modules with **two
different memberships** (`ions.py:107`, `ions.py:154`, `ions.py:214-223`,
`bindings.py:984-991`), and `braincell.ion.build_placeholder_ions` is a fifth.
Only the `ions.py:107` copy is wrong, and nothing forces the five to agree.

### 2. `SineClamp` ignores the per-point delay

`layouts.py:961`, inside the per-point loop of `_evaluate_clamp_layout`:

```python
        if layout.kind == "SineClamp":
            local_t = (t - _scalar_state_value(runtime, layout_id=layout.id, var_name="delay")).in_unit(u.ms)
```

`local_index=local_index` is missing, so `_scalar_state_value` falls back to
its `local_index: int = 0` default and every point in the layout uses point
0's delay. The `CurrentClamp` branch eleven lines above passes it, and so does
every other `_scalar_state_value` call inside `_eval_sine_clamp` itself
(`duration`, `amplitude`, `offset`, `frequency`, `phase`) — this is an omitted
argument, not a design choice.

Reproduced with two identical clamps on one branch, merged into one layout,
per-point delays of `[0, 5] ms`, evaluated at `t = 1 ms`:

```
sine     n_active=2  delay=[0,5]ms  I(t=1ms) = [0.2, 0.2]     <- point 1 should be off
current  n_active=2  delay=[0,5]ms  I(t=1ms) = [0.2, 0.0]     <- correct
```

`layouts_test.py:302` `test_sine_clamp_uses_delay_window` uses a single point,
where index 0 is the only index, so it passes either way.

### 3. The two runtime channel wrappers have complementary gaps

`bindings.py` defines `_BoundIonChannelRuntime` (32 lines) and
`_BoundIonChannelCurrentComponentRuntime` (207 lines). The second is the first
with `component_key` set and an `owns_state` gate; six of its lifecycle
methods are byte-identical to the first's modulo the forwarded name. The
divergence has produced exactly the defect that duplication invites — each
class forwards a method the other does not:

```
  ind_update           W1=own        W2=inherited
  update               W1=inherited  W2=own
```

- **`ind_update` is missing from the component wrapper.** `_base_ion.py:399`
  and `:587` call `node.ind_update(V, ...)` on every child. The component
  wrapper inherits `IonChannel.ind_update` (`_base_channel.py:247-249`), which
  tests `isinstance(self, IndependentIntegration)` on **the wrapper** — never
  true — so sub-solver integration silently no-ops for a multi-owner channel,
  bypassing the `owns_state` gate entirely.
- **`update` on the component wrapper is dead.** `IonChannel` defines no
  `update`, so it overrides nothing, and the only `.update(` call on a channel
  anywhere in the repository is line `bindings.py:447` — inside this method's
  own body. 19 lines.

**This one is latent.** The only multi-owner channel is `Kv1p5_MA2020_GrC`,
and it is not an `IndependentIntegration`, so no shipped model reaches the
`ind_update` gap today. It is fixed here because merging the two classes is
the change that removes the opportunity for the next such divergence — and
because `Kv1p5_MA2020_GrC` is the same channel defect 1 makes unusable.

## Measured performance findings

Every number below was produced by the efficiency review against real cells at
several sizes. Findings that measurement **killed** are listed under
"Considered and declined" rather than quietly dropped.

### 4. `bridge.py:76` — `u.math.asarray` on a Python list of scalars

`quantity_vector` stacks per-CV scalars with `u.math.asarray(decimals)`, which
routes to `jnp.asarray` on a list of Python floats. It is the entire cost of
`attach_runtime_ion_geometry` (6 geometry attributes) and `cv_value_vector`.

```
n_cv=1208, 6 geometry attrs
  u.math.asarray(list-of-scalars)   [current]             89.62 ms
  np.asarray(list-of-scalars)                              0.12 ms      <-- 750x
```

End to end, with values asserted identical:

| n_cv | init current | with `np.asarray` | gain |
|---|---|---|---|
| 168 | 92.7 ms | 80.4 ms | 13.3% |
| 488 | 254.1 ms | 211.3 ms | 16.9% |
| 1208 | 593.6 ms | 513.5 ms | 13.5% |
| 3208 | 1792.8 ms | 1491.2 ms | 16.8% |

A flat 13-17% of total `Cell.init_state()` wall clock at every size — a
constant-factor fix, not a scaling one. `to_decimal` returns Python floats, so
`np.asarray` would default to float64 and make `scatter_midpoint_values` emit
an x64-truncation warning; the dtype is pinned explicitly.

### 5. `state.py:343-350` — synapse parameters unstacked one scalar at a time, then restacked

```python
values = [synapse_store.parameter_value(int(logical_id), var_name) for logical_id in ...]
buffer = _stack_synapse_values(values, parameter=var_name)
```

`parameter_value` → `_take_vector_item` (`_multi_compartment/synapses.py:731`)
does `np.asarray(value.to_decimal(value.unit))[index]` — converting the
**entire** N-element column to an array to take one element, N times. The
column being torn apart was already built as a rectangular `u.Quantity` by
`_build_parameter_columns`, using the same `_stack_synapse_values`.

| N synapses | `init_state` | this block | share | vectorised |
|---|---|---|---|---|
| 200 | 112.9 ms | 3.94 ms | 3.5% | 0.020 ms |
| 1000 | 172.9 ms | 20.63 ms | 11.9% | 0.016 ms |
| 2000 | 253.3 ms | 40.36 ms | 15.9% | 0.016 ms |
| 4000 | 393.8 ms | 83.01 ms | 21.1% | 0.020 ms |

This is the scaling finding of the iteration: 3.5% at n=200 to 21.1% at
n=4000, against a constant-time column gather.

### 6. `layouts.py:944-967` — per-point Python loop over clamp evaluation

`_evaluate_clamp_layout` loops over `n_active` and slices row `local_index`
out of the already-padded 2-D buffers, staging a separate cumsum/where/sum per
point. On the **traced** path, so XLA's CSE removes the duplication and there
is no runtime cost — the entire cost is trace and compile latency:

| n_clamp | eqs now | eqs vectorised | trace now | trace vec | compile now | compile vec | runtime now | runtime vec |
|---|---|---|---|---|---|---|---|---|
| 1 | 19 | 18 | 2.0 ms | 4.2 ms | 29.7 ms | 30.4 ms | 9.8 us | 9.7 us |
| 32 | 422 | 18 | 24.8 ms | 4.1 ms | 62.4 ms | 31.1 ms | 10.8 us | 10.6 us |
| 128 | 1670 | 18 | 104.3 ms | 4.3 ms | 172.0 ms | 32.3 ms | 10.4 us | 10.2 us |
| 512 | 6662 | 18 | 409.8 ms | 4.3 ms | 709.9 ms | 30.7 ms | 11.5 us | 10.9 us |

User-visible as first-call latency: `cell.run(...)` goes 0.59 s with one clamp
to 0.97 s with 128 identical clamps; the second call is 2 ms in both cases.
**No runtime gain is claimed** — the runtime columns above are flat.

Vectorising is also the structural fix for defect 2: gathering `delay` for the
selected indices as a vector removes the site where a scalar default could be
silently taken.

### 7. `state.py:441` / `layouts.py:267` — `u.cm**2` rebuilt per CV, and CV area walked twice

```python
cv_area_decimal = np.asarray([float(np.asarray(cv.area.to_decimal(u.cm**2), ...)) for cv in cell.cvs], ...)
```

`u.cm**2` costs 13.4 us to construct and is rebuilt once per CV:

```
n_cv = 3208
  current: [float(np.asarray(cv.area.to_decimal(u.cm**2)))...]    51.51 ms
  hoisted unit constant                                           10.51 ms      <-- 4.9x
```

Separately, `from_cell` walks `cv.area` twice on a clamped cell —
`build_clamp_routing_table` (`state.py:434`) and then `cv_area`
(`state.py:441`) — and `ClampRoutingTable.midpoint_area` is exactly
`runtime.point_area[midpoint_ids]` (verified equal on a real cell). The second
pass is 0.7% of `init_state` at n_cv=168 and 2.4% at n_cv=3208.

### 8. `ions.py:238` — `inspect.signature` re-run per call

`_supported_ion_runtime_params(cls)` runs `inspect.signature(cls.__init__)`
every call, including from `_sync_runtime_ion` — i.e. on **every `set_state`
write to an ion parameter**.

```
_supported_ion_runtime_params:  22.45 us/call    lru_cached: 0.128 us/call     <-- 175x
```

Honest sizing: 14% of a small-cell `set_state`, 1.2% of a large-cell one, and
~0.03% of `init_state`. Worth a one-line `lru_cache` for parameter-sweep
loops; not an `init_state` win.

## Duplication and dead code

### 9. The ion species ladder is written twice, and both copies re-derive a declared attribute

`ions.py:214-223` (`_runtime_ion_species_key`) and `bindings.py:983-994`
(inside `_root_type_to_family`) are the same four-way `issubclass` chain over
`Sodium`/`Potassium`/`Calcium`/`NonSpecific`, differing only in whether an
unknown class raises or returns `None`. Both reproduce a value
`braincell.ion` already declares as data — `Sodium.ion_symbol = 'Na'`,
`Potassium = 'K'`, `Calcium = 'Ca'`, `NonSpecific = "no"`. Verified:

```
23 ion classes: ion_symbol vs species_key mismatches=0, vs family=0
129 channel root types: ion_symbol vs _root_type_to_family mismatches=0
```

The 129-root-type check includes the non-ion root types, where
`getattr(rt, "ion_symbol", None)` and `_root_type_to_family` both yield
`None` — so the attribute lookup also subsumes the `try/except TypeError`
guard at `bindings.py:992-993`.

`ion_symbol` is currently **dead**: a repo-wide grep finds 4 definitions, 4
docstring entries, and one consumer, an assertion in `nonspecific_test.py:36`.
Routing both functions through it makes the declaration live and collapses
five statements of the family set to one.

### 10. `_runtime_ion_family` is dead

`ions.py:226-235` — a second four-way `issubclass` ladder, over
`KineticIon`/`DynamicNernstIon`/`InitNernstIon`/`FixedIon`. Its single call
site (`ions.py:151`) stores the result as `record["family"]` (`:162`), and a
grep for `["family"]` across `braincell/` finds that write and **no read
anywhere**. 12 lines.

### 11. Two blocks of unreachable code

- `layouts.py:854-857` — a 4-line `raise TypeError` after the unconditional
  `return` at `:852`. The intended non-Quantity rejection never fires; such a
  value falls through to `np.asarray(..., dtype=np.float64)` at `:847` and
  raises a numpy error with no layout id in the message.
- `ions.py:173-177` — `elif _runtime_ion_species_key(record["runtime_cls"]) != species_key:`
  is reached only after `:168` established `record["runtime_cls"] is runtime_cls`,
  and `species_key` was computed from `runtime_cls` by the same pure function
  at `:150`. It compares a value with itself. 5 lines including an error
  message that can never be raised.

### 12. Dead declarations and exports

- `bridge.matches_last_dim` (`bridge.py:251`) — grepped across the whole
  repository including tests, docs and examples: the only two occurrences are
  its own `def` and its `__all__` entry.
- `MechanismLayout.source_rule` (`layouts.py:124`) — two hits repo-wide, the
  declaration and `state.py:318` setting it to `None`. Never read.
- `NodeScheduling.algorithm` (`scheduling.py:35`) and `.level_start` (`:42`) —
  zero reads anywhere, including tests. The `algorithm` *parameter* stays;
  `cell.py:1630` caches on it.
- `state.py:478-481` passes `dhs_static_source_np=None`, `dhs_static_cache=None`,
  `axial_operator_np=None`, `axial_operator_cache=None` — all four are already
  the dataclass defaults at `:157-160`.

### 13. `_compute/_testing.py::_build_tree` duplicates `_discretization`'s fixture

The five body lines of `_build_tree` (`_compute/_testing.py:58-63`) are
identical to `make_two_branch_morpho` (`_discretization/_testing.py:61-67`);
only the `def` line and docstring differ. The package already imports the
canonical one — `scheduling_test.py:24` does
`from braincell._discretization._testing import make_two_branch_morpho` — so
`_compute` currently uses both spellings of the same fixture, with four test
modules on the copy.

### 14. Duck-typed probes with an existing contract

- `ions.py:515` `hasattr(ion, "_update_reversal") and callable(...)`.
  `_update_reversal` is defined once, on `InitNernstIon` (`ion/_base.py:461`),
  which `ions.py:62` already imports and already uses in an `issubclass` test
  at `:230`. Verified equivalent across all 23 registered ion classes, 0
  mismatches. (The probe genuinely fires — `SodiumFixed` and `CalciumDetailed`
  lack the method — so this is a spelling change, not a dead branch.)
- `bindings.py:1085`, `:1095` `getattr(node, "_on_param_updated", None)` plus a
  `callable` check, over a base class that defines `_on_param_updated` as a
  no-op method (`_base_channel.py:251`) inherited by both `Channel` and
  `Synapse`. Ion layouts have already returned at `:1073-1075`, so `node` is
  always one of those.

Both degrade silently rather than failing: rename `_update_reversal` and
Nernst reversals stop refreshing after a parameter write, with
correct-looking output.

## Changes

1. Seed placeholder ions from `build_placeholder_ions`' own key set instead of
   the hardcoded `("na", "k", "ca")`, so the two cannot drift again. Adds the
   `"no"` family. Regression test: paint `Kv1p5_MA2020_GrC` alone and assert
   `init_state()` succeeds.
2. Vectorise `CurrentClamp` and `SineClamp` evaluation over the selected local
   indices, which both removes the per-point Python loop and makes the
   `delay` gather per-point. `FunctionClamp` keeps a loop — it calls a user
   `fn(t)` per point and cannot be vectorised — and its results are stacked as
   before. `_quantity_sequence_to_decimal_vector` already accepts either a
   sequence or a Quantity vector (`layouts.py:980-984`), so the
   `evaluate_point_clamps` consumer is unchanged. Regression test: two clamps
   in one layout with different delays.
3. Merge `_BoundIonChannelRuntime` into
   `_BoundIonChannelCurrentComponentRuntime` as one class with
   `component_key: str | None = None` and `owns_state: bool = True`, routing
   the six lifecycle methods through one helper and forwarding `ind_update`.
   Delete the dead `update`.
4. `bridge.quantity_vector`: `np.asarray(decimals, dtype=...)` with the dtype
   pinned to the default floating type rather than `u.math.asarray`.
5. `state.py:343-350`: gather the synapse parameter column that
   `_build_parameter_columns` already built, instead of tearing it apart per
   row and restacking. The gather is a new `_SynapseStore.parameter_column`
   accessor — the vector form of `parameter_value` — which also replaces the
   identical unstack/restack in `SynapseView.get` (`synapses.py:473`).
6. Hoist the `u.cm**2` constant out of the per-CV comprehension, compute
   `cv_area`/`point_area` once, and pass the resulting point-area vector into
   `build_clamp_routing_table` rather than letting it walk `cv.area` a second
   time. This rests on midpoint nodes reporting their own CV as `roles[0]`,
   verified exactly (`max|old-new| = 0`) at 2, 6 and 18 CVs and pinned by a
   regression test.
7. `functools.lru_cache` on `_supported_ion_runtime_params`.
8. One `ion_species_key(cls)` helper reading `ion_symbol` and returning `None`
   for non-ion classes, replacing both `issubclass` ladders; the raising
   variant wraps it.
9. Delete: `_runtime_ion_family` and `record["family"]`; the unreachable
   blocks at `layouts.py:854-857` and `ions.py:173-177`;
   `bridge.matches_last_dim`; `MechanismLayout.source_rule`;
   `NodeScheduling.algorithm` and `.level_start`; the four redundant
   default arguments at `state.py:478-481`.
10. `_compute/_testing.py` re-exports `make_two_branch_morpho` under the name
    the four test modules already use, rather than holding a second copy.
11. Replace the two duck-typed probes with the contracts they stand in for.

## Breaking changes

`braincell._compute` is a private package — `__init__.py` states "This package
is private. External code should access the same symbols through
:mod:`braincell` re-exports where appropriate." None of the names below are
re-exported from `braincell`, so no public API changes. Listed for
completeness:

1. **`NodeScheduling.algorithm` and `NodeScheduling.level_start` are removed.**
   Zero readers. The `algorithm` argument to `build_node_scheduling` is
   unchanged.
2. **`MechanismLayout.source_rule` is removed.** Only ever written, as `None`.
3. **`braincell._compute.bridge.matches_last_dim` is removed**, along with its
   `__all__` entry. No caller anywhere.
4. **`_BoundIonChannelRuntime` is removed**, merged into
   `_BoundIonChannelCurrentComponentRuntime`. Both are module-private and
   instantiated only within `bindings.py`.
5. **`_runtime_ion_family` is removed**, and the `"family"` key no longer
   appears in the internal ion instance records.
6. **`build_clamp_routing_table` takes `point_area_decimal` and `midpoint_ids`
   instead of `cvs`, `node_tree` and `n_point`.** The only production caller is
   `CellRuntimeState.from_cell`, which now has both to hand.

No deprecation shims, no aliases, no `warnings.warn` bridges, per the sweep's
standing rule.

## Considered and declined

**Memoising `mechanism_signature`.** Called once per (CV, mechanism) pair from
`state.py:212`. Measured: 8000 calls over 1000 objects, 2.87 ms → 2.64 ms;
with a wide paint, 9760 calls over 20 objects, 2.62 → 2.13 ms — **0.5 ms of a
554 ms `init_state`, 0.09%**. cProfile's apparent 18.7 ms was ~6x call
overhead.

**Single-pass `_merged_channel_constructor_params`.** It copies the whole
`full_shape` buffer once per merged layout, which is the exact antipattern
`_apply_density_parameter_overrides` documents avoiding at `state.py:723`.
Measured at 500 merged layouts × 4510 points: 14.93 ms of a 3170 ms
`init_state` (0.47%), and a single-pass in-place variant measured **14.68 ms —
no faster**, outputs asserted equal.

**Vectorising the synapse grouping block** (`state.py:252-271`). 4000
synapses: 1.26 ms → 0.428 ms, **0.2% of `init_state`**. Cosmetic.

**Removing the discarded placeholder ions** from `_build_default_ions`. It
builds all families and keeps a subset, up to 3 times: 0.17 ms vs 0.06 ms,
flat in `n_point`. 0.12 ms total. (The *correctness* half of this — the
discarded `"no"` — is defect 1 and is fixed.)

**The remaining per-`Quantity`-box loops in `ions.py`**
(`_ion_param_broadcast`'s tuple fallback, `_ion_param_scatter`'s object
branch). Instrumented across `PotassiumFixed` and `CalciumDetailed` at
n_point 190 and 1360: 0.05-0.08 ms, flat in `n_point`, and **zero
tuple-fallback hits**. The module docstring's claim that this path is already
rectangular holds.

**Closure-capture leaks.** `_compute` contains one `lambda` (a `sorted` key at
`state.py:279`) and no nested `def` stored on a runtime object; the runtime
wrappers are already classes copying only the fields they need.

**Deduplicating the broadcast-then-scatter algebra** between
`ions._ion_param_broadcast`/`_ion_param_scatter` and
`bindings._initial_merged_channel_param`/`_scatter_active_channel_param`
(~130 lines, 4 functions, 2 call sites each). The unit split/rewrap and shape
broadcast halves are the same code twice, but the mask-vs-index
representations and the `_CONDUCTANCE_PARAM_NAMES` zeroing genuinely differ.
Merging them well needs the `split_unit` move below, which is a different
iteration's change; merging them badly produces a function with two modes.

**Decomposing `CellRuntimeState.from_cell`** (327 lines, six separable
phases). Pure restructuring with no behavioural or performance effect, on the
single most load-bearing function in the package, in an iteration that already
changes three defects underneath it. Not mixed into this diff.

## Deferred to later iterations of the sweep

- **`Cell.mech_table` belongs in `table.py`.** The 90-line builder lives at
  `_multi_compartment/cell.py:2526-2616` while the dataclasses it builds live
  in `_compute/table.py`, and `MechanismObjectCell` reads only
  `CellRuntimeState`. The builder re-derives the signature → layout_id index
  that `from_cell` already computed and discarded, which is the only reason
  `mechanism_signature` is public. Iteration 11.
- **`network/recording.py:430-433` re-derives `MechanismLayout.gather_points`**
  character for character, against the private `population_index` field —
  under a docstring that says "Callers read the rule from here rather than
  repeating the branch." Three other callers obey. Iteration 12.
- **`_multi_compartment/field_resolution.py:465-467` recomputes
  `runtime.midpoint_mask_np`** (verified equal); `cell.py:1734` already reads
  the cached one. Iteration 11.
- **`split_unit` is in the wrong layer.** `field_resolution.py:113-145` names
  the split-unit/rewrap idiom and has 10 call sites; the same shape appears
  inline 8 more times inside `_compute` (`bridge.py` ×4, `ions.py` ×2,
  `bindings.py` ×2). The move is downward, into `bridge.py` or `_misc.py`,
  for ~18 call sites. Iteration 14.
- **Reflection over ion `__init__` signatures should be a declared contract.**
  `ions.py:241`'s hardcoded `excluded` set filters an `inspect.signature` walk,
  and `braincell/ion/_base.py:136` plus five sites in `calcium.py` carry
  comments explaining that their constructors are shaped *around* this
  reflection. A `runtime_params` classattr on the ion base would delete both
  the exclusion set and the apology comments. Cross-package; iteration 14.
- **Calcium-specific knowledge in generic ion lowering.** `ions.py:333-346`
  reaches for `getattr(runtime_ion_instance, "cainull", None)`, an attribute
  only calcium classes have, and `:254-268` special-cases the literal string
  `"Ci_initializer"`. The `cainull` branch was instrumented over the full
  suite and fired **0 times**. A second dynamic-concentration species would
  get no shaped initializer and fail silently. Iteration 14.
- **`CLAMP_KINDS` is enumerated four times** (`layouts.py:197`, the dispatch
  chain at `:947/:959/:963`, the field lists at `:419-424`, and
  `state.py:359/368`). A fourth clamp that misses the `CLAMP_KINDS` edit
  compiles, allocates buffers, and contributes exactly zero current with no
  error. The fix is for the clamp classes in `braincell.mech` to declare their
  own fields and an `evaluate` method. Cross-package; iteration 14.
- **`_CONDUCTANCE_PARAM_NAMES`** (`bindings.py:1044`) decides which merged
  parameters zero-initialize by sniffing for `{"g_max","g","gbar","conductance"}`.
  A channel whose conductance parameter is named `gkbar` or `pmax` is merged
  against a broadcast baseline instead of zeros — silently wrong current
  outside the painted region. The channel class already knows; it should
  declare it. Iteration 14.
- **`layouts.py`'s clamp evaluators take `CellRuntimeState` but touch only
  `runtime.state_buffers`.** Narrowing the parameter would remove
  `layouts.py:72`'s `TYPE_CHECKING` import and make the module's documented
  "imports no other `_compute` module" claim true behaviourally as well as
  syntactically. Partially addressed by change 2; completing it is iteration 14.
- **`__init__.py:28` lists `bridge` after `state`**, but `state.py:78` imports
  `bridge`. The true order is `{scheduling, bridge, layouts} → ions →
  bindings → state → table`. `__init___test.py:105`'s `_LAYER_ORDER` does not
  cover `bridge`, `table`, or `scheduling`, so nothing catches it. Iteration 14.
- **`docs/design/interface-map.md:39` is stale** — it lists `vis.point_topology`
  and `vis.cell_topology` as `_compute` consumers, but `braincell/vis` has no
  coupling to `_compute` at all. `docs/design/module-dependency-map.md:114-208`
  matches the measured import graph exactly and is the authority. Iteration 14.
- **Dense density buffers are O(n_layouts × n_point).** A channel painted
  per-CV allocates a full point-space buffer per layout even at `n_active == 1`:
  1000 layouts × 1360 points = 21.76 MB, 170x the active footprint. Changing
  it means changing the layout contract, not a micro-fix. Architectural.

## Verification

### Tests

Every defect was reproduced by a failing test before its fix, and each
regression test was then shown to be non-vacuous by reintroducing the defect:

| Defect | Test | Module |
| --- | --- | --- |
| 1 (`"no"` never seeded) | `test_nonspecific_placeholder_is_seeded_like_the_other_families`, `test_placeholder_families_match_the_ion_package_exactly` | `ions_test.py` |
| 2 (`SineClamp` delay) | `test_sine_clamp_reads_the_delay_of_each_point_not_of_point_zero` | `layouts_test.py` |
| 3 (wrapper gaps) | `test_component_wrappers_forward_ind_update_to_the_wrapped_channel`, `test_component_wrappers_do_not_carry_a_dead_update_method` | `bindings_test.py` |
| 6 (midpoint area) | `test_clamp_routing_area_matches_the_cv_that_owns_each_midpoint` | `state_test.py` |

The defect-2 test needed a fixture fix of its own: `at(0, 0.25)` and
`at(0, 0.75)` both resolve to point 1 on the two-branch fixture, so the two
clamps scatter-added into a single point and the assertion passed for the wrong
reason. It now places on branches 0 and 1 and asserts
`len(set(layout.point_index.tolist())) == 2` before testing anything else.

Collected-test count went from 2808 to 2814 — the six tests above, no
deletions.

### Performance

`init_state` on a 41-branch cell (`n_cv = 328`, `n_point = 370`, 3 channels,
2 ions, 8 `CurrentClamp` + 8 `SineClamp`), min of 9 runs, baseline extracted
with `git archive` from the pre-iteration commit:

| synapses | before | after | |
| --- | --- | --- | --- |
| 0 | 181.8 ms | 150.7 ms | 1.21x |
| 400 | 206.9 ms | 168.6 ms | 1.23x |
| 2000 | 335.8 ms | 235.4 ms | 1.43x |

Marginal cost per 1000 synapses falls from 77 ms to 42 ms — the column gather
(change 5). The 31 ms saved at zero synapses is the clamp vectorisation, the
`u.cm**2` hoist with the dropped second CV-area pass, and the `bridge` dtype
pin.

Separately, the clamp vectorisation cuts the traced clamp program from 6662
equations to 18, trace time from 410 ms to 4.3 ms, and compile time from
710 ms to 31 ms at 200 clamped points.

### Suite

```
$ pytest braincell/ -q
2799 passed, 15 skipped, 409 warnings, 410 subtests passed in 183.79s (0:03:03)
```
