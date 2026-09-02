# Multi-compartment simplification sweep

Iteration 11 of the module-by-module `/simplify` sweep. Target:
`braincell/_multi_compartment/` — 8,900 lines across nine non-test modules,
of which `cell.py` alone is 3,003.

Baseline on `eddc00c`:

- `pytest braincell/_multi_compartment -q` → 157 passed, 59 subtests passed
- `pytest braincell/ -q` → 2807 passed, 15 skipped, 410 subtests passed (219.29 s)

Four review passes (reuse, simplification, efficiency, altitude) produced 44
findings. This document records which are being applied, which are declined
and why, and which are deferred with the evidence needed to pick them up
later. The applied set groups into three themes.

---

## Theme 1 — Delete what nothing calls

Every item here was confirmed dead by a repo-wide grep over `braincell/`,
`examples/` and `docs/` (`.py`, `.ipynb`, `.md`). None changes behaviour.

### 1.1 `_integrate_selected_ion_channel_states` and `_run_selected_child_channel_hook`

`cell.py:1844-1903` and `cell.py:1933-1947`. The only references to either
name are the definitions and the three calls the first makes to the second,
so both die together:

```
$ grep -rn "_integrate_selected_ion_channel_states\|_run_selected_child_channel_hook" .
cell.py:1844:    def _integrate_selected_ion_channel_states(...)
cell.py:1857/1872/1887:  self._run_selected_child_channel_hook(   # inside the dead method
cell.py:1934:    def _run_selected_child_channel_hook(...)
```

It is also the package's clearest copy-paste: `_pre_integral`,
`_compute_derivative` and `_post_integral` (`:1850`, `:1865`, `:1880`) are
three byte-identical 13-line closures differing only in the hook name —
while the live sibling `_integrate_selected_ion_self_states` (`:1905`,
called from `:1981`) already shows the parameterised form. Deleting removes
104 lines of plausible-looking scheduling code that never executes, sitting
between two paths that do.

### 1.2 `_SynapseStore.parameter_value` and `_SynapseStore.set_parameter`

`synapses.py:207-213` and `:265-266`. `parameter_value` has no caller; the
only other mentions are three references to it in `parameter_column`'s own
`See Also` and Notes, which describe it as the quadratic path to avoid — a
dead accessor kept alive by documentation advertising that it is slow.
`set_parameter` is a two-line wrapper over `set_parameters`, which is what
the one live caller (`synapses.py:574`) uses.

### 1.3 `_CellScope.branch_ids`

`selection.py:62-63`. The one place that wants this reads
`self._scope.exact_branch_ids` directly (`cell.py:521`).

### 1.4 `Cell._discretization_to_point`

`cell.py:1680-1681`, an alias of `_cv_to_point` whose seven callers are all
tests in a *different* package (`_compute/bindings_test.py` ×5,
`_compute/state_test.py` ×2). Its comment ("kept while older inspection
helpers are still in use") no longer describes anything. Deleted; the seven
call sites move to `_cv_to_point`.

### 1.5 `_gather_layout_point_values`

`cell.py:2874-2876`, a module function whose entire body is
`return layout.gather_points(values)`. Its two sibling modules already call
the method directly (`currents.py:120`, `probes.py:227`); only `cell.py:2057`
goes through the wrapper.

### 1.6 `Cell._axial_jax`

Five writes (`cell.py:847`, `:1465`, `:1509`, `:1784`, `:1801`), zero
production reads. The only readers are five assertions in `cell_test.py`
(`:557`, `:604`, `:706`, `:711`, `:729`), four of which already assert on
`runtime.axial_operator_cache` in the adjacent line. Three lifecycle methods
currently have to remember to reset a field nothing reads.

One genuine divergence exists and is preserved rather than papered over: at
`cell.py:1799-1801` the runtime cache is skipped when
`is_traced_value(operator)` while `_axial_jax` is assigned regardless. No
production code observes that difference. The test assertions move to
`cell.runtime.axial_operator_cache`.

### 1.7 `run.py` unreachable cache fallback

`run.py:187-194`. `cache = getattr(rcell, "_run_loop_cache", None)` followed
by an `if cache is None` branch that rebuilds the loop uncached.
`Cell.__init__` sets `self._run_loop_cache = {}` unconditionally
(`cell.py:840`) and the field is only ever `.clear()`ed (`:1225`, `:1375`,
`:1511`) — never reassigned, never `None`. `run()` has exactly one caller
(`cell.py:2635`), always with a real `Cell`. The dead branch also holds a
second copy of the `_make_run_loop(...)` argument list, so a new argument
added to one and missed in the other would be a silent, untested divergence.

### 1.8 Unreachable bool-dtype disjunct

`selection.py:258` and `:276` both read
`values.dtype.kind not in "iu" or values.dtype.kind == "b"`. Since
`"b" not in "iu"`, the first clause already rejects bool arrays and the
second can never decide anything.

### 1.9 `probes._probe_current_ion_info` unused default and unreachable guards

`probes.py:328-351`. Its single caller (`probes.py:261-266`) sits inside
`if declaration.mechanism is not None:` (opened at `:220`) and always passes
`layout_id=layout_id`. So `layout_id: int | None = None` never sees `None`,
the `if layout_id is not None` guard at `:338` is always true, and the
`mechanism_name is None` early return at `:343-345` is unreachable. The
parameter becomes required and the two dead paths go.

---

## Theme 2 — Say it once

### 2.1 Five coercers collapse to one shape ladder

`field_resolution.py:481` `coerce_node_values`, `:519`
`coerce_runtime_point_values`, `:561` `coerce_cv_values`, `:599`
`coerce_named_node_values`, `:638` `coerce_named_cv_values`.

All five open with the same `split_unit(single_population_view(...))` and run
the same five-rung ladder: `ndim == 0` → fill; `ndim != 1` → raise; length
matches the native space → return unchanged; length matches the other space
→ map; otherwise raise. They differ only in a three-cell table:

| | `ndim == 0` | `len == n_point` | `len == n_cv` |
|---|---|---|---|
| `coerce_node_values` | fill `n_point` | identity | `cv_to_node_values` |
| `coerce_runtime_point_values` | fill `n_point` | identity | `cell._cv_to_point` |
| `coerce_named_node_values` | `cv_to_node(fill n_cv)` | `mask_non_midpoint_points` | `cv_to_node_values` |
| `coerce_cv_values` | fill `n_cv` | `cell._point_to_cv` | identity |
| `coerce_named_cv_values` | fill `n_cv` | `_point_to_cv(mask…)` | identity |

A `diff` of the first two bodies (504-516 against 546-558) returns exactly
one differing line. The messages have already drifted between copies — the
named variants say "cannot map the named value into point space" where the
unnamed ones print the two expected lengths — which is what five copies of a
shape rule produce.

Collapses to one `_coerce(cell, value, *, caller, scalar_fill, from_point,
from_cv)`; each public coercer becomes a short call supplying its row.

### 2.2 `resolve_node_field_values` / `resolve_cv_field_values`

`field_resolution.py:935` and `:993`. A `diff` of the two 27-line bodies
(964-990 against 1022-1048) shows five differing lines, each a straight
substitution of a `_node_` helper for its `_cv_` twin. The `"v"`/`"voltage"`
parsing, the 3-tuple shape test, the three `mode` branches, both label
formats and both `ValueError` strings are identical. One core takes the four
slots; the two public functions become thin wrappers.

### 2.3 The probe density-layout lookup

`probes.py:176-184` and `:237-245` are identical apart from indentation:
scan `runtime.get_point_layouts(point_id)`, keep layouts whose mechanism is
a `Density` with a matching `instance_name`, raise the same
"matched multiple mechanisms named …" on more than one. They diverge only
*after* the block — one falls through to an ion lookup, the other raises.
Extracted as `_matched_density_layout(...)`, leaving each caller to state
only its own zero-match policy.

### 2.4 Probe dispatch

`probes.py:112-142` is a 31-line `if/elif/elif` over three probe types whose
branches differ only in which function is called; the argument list is
spelled out identically three times. The same
`(StateProbe, MechanismProbe, CurrentProbe)` tuple is also written out at
`probes.py:50` and `:74`. A module-level `_POINT_SAMPLERS` dict replaces the
dispatch and `tuple(_POINT_SAMPLERS)` replaces both isinstance tuples, so
adding a fourth probe type is a one-site edit instead of four.

### 2.5 `_family_channel_nodes` Ion / MixIons branches

`cell.py:1822-1831`. The `isinstance(node, Ion)` and
`elif isinstance(node, MixIons)` arms are character-for-character identical.
Neither class subclasses the other (`_base_ion.py:204` and `:480` both
derive from `IonChannel, Container`), so this is not an ordering trick.
Merged into `isinstance(node, (Ion, MixIons))`. Today the
`_skip_family_update` opt-out is implemented twice.

### 2.6 One `layout_<id>` parser

`cell.py:2865` `_layout_id_from_runtime_path` and `currents.py:110`
`_layout_id_from_current_key` differ only in a parameter name and one noun
in each of two error strings. Kept once in `cell.py`, imported by
`currents.py`.

### 2.7 One synapse name/type conflict check

`synapses.py:121` `_SynapseStore._validate_name_types` and `cell.py:1347`
`Cell._validate_synapse_names` both build a `name → synapse_type` map with
`setdefault` and raise on mismatch. The message
`"Synapses with the same name {…!r} cannot use different synapse types
({…!r} and {…!r})."` appears exactly twice repo-wide — at `cell.py:1360` and
`synapses.py:127`. The inputs differ (place rules at `place()` time versus
store columns at build time), so the shared piece is a helper over an
iterable of `(name, synapse_type)` pairs, called from both.

### 2.8 `_count_labels`

`synapses.py:696` is a six-line re-implementation of `collections.Counter`,
which `density_views.py:199-201` already uses for the identical repr purpose.

### 2.9 `Cell.mech_table` moves to the layer that owns the table

`cell.py:2526-2621` inlines 96 lines to build a `MechanismObjectTable`. Every
sibling in the same section delegates in one line —
`probes.sample_probe(self, name)`, `probes.sample_probes(self)`,
`run_module.run(self, dt=dt, duration=duration)`.

The table type, its cells, and the key function all live in
`_compute/table.py`; the layout signature lives in `_compute/layouts.py`.
`CellRuntimeState`'s own docstring (`_compute/state.py:129`) advertises
"table views: `mechanism_cv_table`, `mechanism_point_table`" — **neither
method exists**; those names have zero definitions and zero callers
repo-wide. The inline block is what that never-written method turned into.

The builder moves to `_compute/table.py` as
`build_mechanism_object_table(runtime, cvs)`, `Cell.mech_table` becomes a
two-line delegation, and the stale docstring is corrected to name what is
actually there. `cell.py` drops three imports (`MechanismObjectCell`,
`mechanism_cell_key`, `mechanism_signature`), keeping only
`MechanismObjectTable` for the return annotation.

---

## Theme 3 — Make the cached and derived truths honest

### 3.1 `locset_cv_ids` rebuilds a grouping the CV tree already holds

`field_resolution.py:324-327` builds `branch_id → (cv_id, …)` with a
`setdefault` loop over `cell.cvs`. `cell.cv_tree.branch_to_cv_ids` is that
mapping, and `selection.py:229-232` — same package — already uses it. The
existing `.get()` returns `None` for an unknown branch and the loop skips
it; the replacement keeps that skip rather than adopting `selection.py`'s
`IndexError`, since a locset evaluated against `cell.morpho` cannot name an
out-of-range branch and silently changing that to a raise is not this
change's business.

### 3.2 `mask_non_midpoint_points` rebuilds a mask the runtime already stores

`field_resolution.py:463-467`:

```python
midpoint_ids = np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)
midpoint_mask = np.zeros((cell.n_point,), dtype=bool)
midpoint_mask[midpoint_ids] = True
```

is byte-for-byte what `_compute/state.py:440-442` computes and stores as
`runtime.midpoint_mask_np`, and what `cell.py:1733-1734` already reads. No
new precondition is introduced: the function already calls `cell.n_point`,
which raises through `_raise_if_not_initialized`, and `cell.runtime` carries
the same guard.

### 3.3 `CV.midpoint`

`cell.py:1272-1278` recomputes `(cv.prox + cv.dist) * 0.5` inline, and
`cell_test.py:431` recomputes it a third time to assert the result.
`_GeoCV.midpoint` (`_discretization/geometry.py:132-140`) already states the
rule with a docstring explaining it is derived rather than stored so it
cannot drift — but the *user-facing* record `CV`
(`_discretization/base.py:165-277`) carries derived `region` and `diam_mid`
properties and stops short of `midpoint`. Added beside `diam_mid`.

### 3.4 Defect — `Cell.on(region).cv.coverage_fraction` reports a different number than the physics uses

**Confirmed end to end.** On a tapering dendrite (radii 4 µm → 1 µm, one CV,
region `BranchSlice(1, 0.0, 0.5)`):

```
Cell.on(region).cv.coverage_fraction  -> [0.5]
cv1 coverage_area_fraction actually painted = 0.65
```

The same CV, the same region, two numbers 30 % apart. On a branch with no
taper they agree exactly (0.5 and 0.5), which is why this has gone unnoticed.

`field_resolution.cv_coverage_fractions` (`:237-266`) measures overlap in
**normalized branch length**: `overlap_x / (cv.dist - cv.prox)`. The physics
in `_discretization/mechanism.py:359-379` measures overlap in **lateral
membrane area**: `overlap_area_um2 / geo.lateral_area_um2`. The field it
writes is named `coverage_area_fraction` (`mech/_density.py:244`) — the
codebase already states which of the two "coverage" means.

There is no consumer that wants a length fraction here. The value reaches:

- `selection.py:121` `select_region` → `_CellScope.coverage` →
  `CVSelector.coverage_fraction` (`selection.py:182-184`), a public
  introspection property whose only plausible reading is "how much of this
  CV did my region cover", i.e. exactly the number that scaled the
  conductance;
- `field_resolution.cv_highlight_fractions` and `node_highlight_fractions`,
  which shade a `braincell.vis.plot_cell_topology` node by how much of the
  CV a region covers.

`cv_coverage_fractions` is changed to the area measure and renamed
`cv_area_coverage_fractions`, so the two spellings cannot be confused again
at a call site. The area computation is not re-implemented: the frustum
machinery in `_discretization/geometry.py` gains a public
`interval_area_fraction(...)` and both `_discretization/mechanism.py` and
`field_resolution.py` call it.

`branch_coverage_fractions` (`:270-296`) stays length-based and gains a
docstring line saying so explicitly. It is a drawing extent along a branch
line for the branch-level topology plot (`vis/cell_topology.py:454`), not a
membrane fraction, and length is the right measure for it.

**Reviewer disagreement, recorded.** The reuse pass looked at the same two
functions and classified them as "different metrics — not a reuse
candidate", i.e. an intentional split. That reading is defensible from the
code alone. It is overruled by the end-to-end measurement above: a public
property that describes a painted mechanism must report the fraction that
was painted. Nothing documents a length measure, and
`coverage_area_fraction` names the intended one.

**Numeric change to existing test.** `field_resolution_test.py:141-148`
asserts `0.5` on `_soma_dend_tree()`, whose dendrite tapers 2 µm → 1 µm. The
area fraction there is `0.5833333333333334` (verified against the value
`Cell.paint` actually stores). The test is updated with that number and a
comment naming the taper as the reason the two measures differ.

### 3.5 Defect — the discretization cache cannot see a morphology mutated in place

**Confirmed:**

```
n_cv before mutation: 1   revision: 1
morpho n_branches now: 2  revision: 2
n_cv after mutation : 1     <-- wrong
```

`Cell._discretization_key` (`cell.py:1226-1233`) identifies the morphology by
`id(self._morpho)` and nothing else. `Morphology` is mutable —
`cell.morpho`'s own docstring says it is returned "without copying it", and
`Morphology._invalidate_derived_caches` says outright that "``Morphology`` is
mutable, so a stale entry here is a correctness bug rather than a missed
optimization". Attaching a branch to a morphology a `Cell` already holds
changes neither the id nor any other key component, so `cell.cvs`,
`cell.n_cv` and `cell.on(...)` keep answering from the pre-mutation
discretization.

`Morphology` already maintains the counter that detects exactly this:
`_revision`, incremented by `_invalidate_derived_caches`. The in-repo idiom
for consuming it is `filter/cache.py:85-89`:

```python
revision = getattr(morpho, "_revision", None)
if self._morpho is not morpho or self._revision != revision:
```

— identity *and* revision. `Cell` uses neither half correctly. Fix:

- `Morphology` gains a public `revision` property, so the one existing
  consumer stops probing a private attribute with `getattr`;
  `filter/cache.py:85` switches to it.
- `_discretization_key` includes `self._morpho.revision`. The `id()` stays
  as the identity component and is sound here — unlike the temporaries that
  caused the iteration-9 bug, the cell holds `self._morpho` alive for as long
  as the cached key exists, and both reassignment sites (`cell.py:1413`,
  `:1514`) invalidate the cache outright. A comment records that reasoning
  so the next reader does not have to rediscover it.

**Second half of the same defect.** With the key fixed, a key *miss* becomes
reachable for the first time — and the miss path (`cell.py:1235-1246`)
rebuilds only `_discretization_cache`, leaving `_synapse_store_cache`,
`_runtime_cvs_cache`, `_runtime_nodes_cache`, `_root_scope_cache` and
`_run_loop_cache` holding values derived from the old discretization. The
miss path calls `_invalidate_discretization_cache()` before rebuilding.

---

## Breaking changes

1. **`field_resolution.cv_coverage_fractions` → `cv_area_coverage_fractions`,
   and its values change on tapering geometry.** Length-weighted overlap
   becomes lateral-area-weighted overlap, matching what `Cell.paint` stores
   as `coverage_area_fraction`. Affects `Cell.on(region).cv.coverage_fraction`
   and the region shading in `braincell.vis.plot_cell_topology`. On geometry
   with no taper the numbers are unchanged. In-repo callers updated:
   `selection.py:121`, `field_resolution.py:366`, `:398`,
   `field_resolution_test.py:142`.
2. **`Cell._discretization_to_point` deleted.** Use `Cell._cv_to_point`.
   Seven call sites in `_compute/bindings_test.py` and `_compute/state_test.py`
   updated.
3. **`Cell._axial_jax` deleted.** Read
   `cell.runtime.axial_operator_cache.operator`. Five assertions in
   `cell_test.py` updated.
4. **`_SynapseStore.parameter_value` and `_SynapseStore.set_parameter`
   deleted.** Use `parameter_column` and `set_parameters`.
5. **`_CellScope.branch_ids` deleted.** Use `exact_branch_ids`.
6. **`Cell._integrate_selected_ion_channel_states` and
   `Cell._run_selected_child_channel_hook` deleted.** No callers existed.
7. **`probes._probe_current_ion_info(layout_id=...)` is now required.**
8. **`Cell.mech_table()`'s body moves to
   `_compute.table.build_mechanism_object_table`.** The `Cell` method keeps
   its name, signature and return type.

No deprecation shims, aliases or `warnings.warn` bridges are added; the old
spellings are removed outright.

---

## Declined

**Per-row synapse parameter validation** (`synapses.py:131-169`). The
efficiency pass measured it: 5,120 logical synapses × 2 parameters →
`_build_parameter_columns` 0.498 s, of which `spec.validate` is 10,240 calls
and 0.246 s. Moving the check to the stacked column would roughly halve the
build. Declined *for this PR* rather than rejected: `_stack_synapse_values`
(`_compute/layouts.py:451`) documents its input as "one **validated** value
per logical synapse" and its own error paths are weaker than
`_validate_like_default`'s — a list mixing a `Quantity` with a bare float
currently raises `TypeError: Synapse field 'g' requires a quantity
compatible with uS.` and would instead raise `ValueError: … has incompatible
units across instances.` Doing this properly means moving
`_stack_synapse_values` into `synapses.py` (it has zero callers inside
`_compute`) and rewriting its messages, which is a change of a different
kind from the rest of this PR. Recorded below with its measurement.

**`_take_population` unified with `_select_population_value`**
(`density_views.py:288` versus `cell.py:2748`). The two genuinely disagree:
`_select_population_value` recognises both a packed `shape[-2]` and a
leading `shape[0]` population axis and indexes with a vector;
`_take_population` recognises only `shape[0]` and indexes with a scalar.
Substituting would leave a length-1 axis where the caller expects a scalar
*and* start matching the packed case. Reported, not applied — reconciling
the population-axis convention is a decision, not a cleanup.

**`density_views.py:154` routed through `state.get_layout_value`.** Not
equivalent: `get_layout_value` maps a global point id to a packed layout's
local index and handles ragged buffers, whereas `_take_point`'s
`shape[-1] == point_size` guard fails on a packed layout and returns the
whole buffer. Behaviour-changing.

**`_require_name` replaced by `mech/_validate.require_str`**
(`density_views.py:337` plus inline copies at `synapses.py:437`, `:454`,
`selection.py:163`). `require_str` raises `TypeError` for a non-`str` and
`ValueError` only for `""`; `_require_name` raises `ValueError` for both.
Swapping changes the exception type callers see. There are two more
byte-identical copies in `network/` (`recording.py:593`,
`connection.py:756`), so this wants doing once across the package, not
piecemeal here.

**Caching `region.evaluate(cell.morpho)`** (`field_resolution.py:225`,
`:319`). `filter.cache.SelectionCache` exists and `_discretization` uses it.
Declined here because the same measurement that killed a similar idea in
iteration 9 applies: the inner selection cache is already value-keyed, and
no measurement in this pass shows the re-walk mattering. Not applied without
a number.

---

## Deferred, with evidence

These are real and verified; each is a different kind of change from this
PR's three themes, and several belong to iterations already scheduled.

**To iteration 12 (network).**

- Standalone `Connection` event routing is a second implementation of
  `network/delivery.py`, living in `Cell`: `cell.py:2292-2319`,
  `:2321-2355`, `:2367-2394`, plus `_coerce_drive_like` (`:2818`),
  `_scatter_drive_rows` (`:2825`), `_zeros_like_event_template` (`:2879`),
  `_rewrap_event_template` (`:2885`), `_connection_event_weight` (`:2890`).
  `delivery.apply_immediate_events` (`:479-522`) already aggregates weighted
  arrivals per `(post_population, layout_id)` into a buffer from
  `zeros_like_events` (`:267`). Two of these methods reach into
  `ConnectionView._call_views` (`network/connection.py:415`), a private
  whose only other callers are inside `network/`.

**To iteration 14 (whole package).**

- Twelve guard-only forwards to `CellRuntimeState`, `cell.py:2463-2513`.
  Each is `_raise_if_not_initialized` then `return self._runtime.<same
  name>(...)`, and `Cell.runtime` is public and carries the same guard, so
  both spellings work and the callers have already split: `cell.*` in
  `field_resolution.py` and `CellView`; `cell.runtime.*` in `synapses.py`,
  `density_views.py`, `probes.py` and all of `network/`. ~50 lines. Breaking
  public API — belongs with the public-surface pass.
- Five private cross-package imports, each with a cross-package consumer:
  `_base_neuron._zero_spike_like` (`cell.py:56`; also
  `_single_compartment/base.py`), `_compute.bindings._is_root_level_runtime_node`
  (`cell.py:70`), `quad._exp_euler._ind_exp_euler_step_selected`
  (`cell.py:95`), `_compute.ions._runtime_ion_species_key`
  (`density_views.py:28`), `_compute.layouts._stack_synapse_values`
  (`synapses.py:32`). Promoting them touches six other modules.
- `AxialOperatorCache` is defined in `cell.py:108-112` but stored in
  `_compute/state.py:158-159`, which therefore has to type its own field
  `object | None`. The same pattern lives correctly one layer down as
  `DHSStaticCache` (`quad/_staggered.py:172-177`).
- Three declaration-override dicts at three altitudes
  (`cell.py:818-821`, `:849`, `:850`); `_compute/state.py:693` reaches *up*
  into one of them with `getattr(cell, "_density_parameter_overrides", {})`.
- `_CellScope` owns the selection but not the gather, so "apply this scope to
  an array" is written six times (`cell.py:622`, `:636`, `:2748`, `:2764`;
  `density_views.py:281`, `:288`; `synapses.py:833`). The inconsistency is
  load-bearing: `CellView.V` honours the spatially-restricted `varshape` and
  `CellView.spike`, four lines below, does not.
- Three sibling surfaces address a density mechanism by three different keys
  — class name only (`field_resolution.py:978`, `:1036`), instance *or*
  class (`density_views.py:103-113`), instance only (`probes.py:178`,
  `:240`). `field_resolution`'s escape hatch pushes a raw `_compute` layout
  id into the public `plot_cell_topology` API.
- `field_resolution.__all__` lists 24 names; 10 have an importer.
- Three spellings of "index the last axis if there is one"
  (`probes.py:313`, `synapses.py:833`, `density_views.py:281`) and three
  copies of the united last-axis scatter (`cell.py:2825`, `synapses.py:838`,
  `:790`).
- `braincell/vis/cell_topology.py:40`, a public package, imports
  `braincell._multi_compartment.field_resolution` and uses
  `require_initialized`, a wrapper that exists so `vis` need not call
  `cell._raise_if_not_initialized`. The module docstring argues for this
  deliberately and the argument is sound, but it makes `field_resolution` de
  facto public API at a private path. Worth deciding explicitly.

**Own PR — architecture.**

- Ragged `place()` expands the declaration inside `Cell`
  (`cell.py:1033` `_place_per_cell`, helpers at `:2664`, `:2684`, `:2719`,
  `:2725`, `:2737`) because `PlaceRule` cannot express the ragged form,
  even though it already carries `population_indices` and `aligned`
  (`_discretization/mechanism.py:81-99`). The workaround costs a second copy
  of the broadcast grammar (`cell.py:2698-2716` versus `synapses.py:762-774`)
  and an `id()`-keyed undo map `Cell._synapse_origins` (`:851`, `:1051`),
  which in turn forces `network/engine.py:249,271` to snapshot and roll back
  two `Cell` privates instead of one. ~75 lines.
- 236 lines of post-voltage mechanism scheduling in `cell.py:1813-2049` whose
  only consumer is `quad/_staggered.py:158-168`, which re-raises the enum
  error verbatim, duplicating `cell.py:2992-2997`. This is what forced the
  private `_ind_exp_euler_step_selected` entry, whose own docstring
  (`quad/_exp_euler.py:320`) reads "Internal selective variant used by family-
  phased cell scheduling."
- `cell.py` uses both the public `ind_exp_euler_step` (`:1957`, `:2011`,
  `:2432`) and the private `_ind_exp_euler_step_selected` (`:1895`, `:1923`),
  so `cell_test.py:944`'s patch of the public name cannot intercept half the
  calls. Worth checking whether that test still proves what it claims.

**Own PR — performance, with measurements.**

| Site | When | Measured |
|---|---|---|
| `synapses.py:131-169` per-row `spec.validate` | declaration | 5,120 synapses × 2 params: build 0.674 s, `_build_parameter_columns` 0.498 s, `spec.validate` 0.246 s over 10,240 calls |
| `synapses.py:754-775` `_select_declared_parameter_value` | declaration | re-runs `to_decimal` on the whole declared array per row — O(N²) element work for an array-valued parameter |
| `density_views.py:115-163` `_DensityView.get()` | inspection | 0.025–0.036 ms/row, linear; 656 rows = 16–24 ms. `_density_rows` (`:229-258`) materializes the full pop × CV × mechanism cross product before `by_name` filters it |
| `selection.py:51-59`, `:93-143` `_CellScope` pairs | inspection | `cell.soma`: 0.030 ms at 164 pairs, 0.146 ms at 2,624, 1.215 ms at 20,992 — linear in `pop_size × n_cv` |
| `cell.py:1467-1468` eager runtime views in `init_state` | declaration | 1.9 ms at 164 CVs, 6.4 ms at 648 CVs (~1 % of `init_state`); the properties already build lazily |
| `cell.py:2275-2319` connection views per synapse layout | trace | O(n_layouts × n_connect_calls × n_rows) Python per trace |
| `probes.py:113-143`, `:303-310` | trace | rebuilds the whole `SynapseView` per probe point; `_representative_cv_id` scans `cv_to_mid_node_id` per point |
| `currents.py:165` | trace | `evaluate_point_clamps(point_ids=…)` does per-layout Python filtering, then line 166 gathers the midpoints anyway; `point_ids=None` is bit-identical |

Explicitly measured and **not** worth acting on: `brainstate.graph.nodes(...)`
appears at 10 call sites and was expected to be a repeated full-tree walk. A
full `Cell.run` trace makes 10 calls totalling 0.9 ms out of 443 ms, because
`_iter_graph` prunes at `level_ > hi` and `CellRuntimeState` is classified
`STATIC`. Not worth caching. Likewise every Python loop in `Cell.update` /
`compute_derivative` iterates mechanism *types* (O(10)), not CVs, and is
traced once.

---

## Edge cases and tests

New regression tests, each written before the change it covers:

1. `field_resolution_test.py` — a CV coverage fraction on a tapering branch
   equals the `coverage_area_fraction` that `Cell.paint` stores for the same
   CV and region, and on an untapered branch equals the length fraction.
   Non-vacuous: it fails on `eddc00c` with 0.5 against 0.65.
2. `cell_test.py` — attaching a branch to a `Morphology` a `Cell` already
   holds changes `cell.n_cv`. Non-vacuous: it fails on `eddc00c`, reporting
   the pre-mutation CV count.
3. `cell_test.py` — a discretization rebuild triggered by a key miss leaves
   no stale `_root_scope_cache`.
4. `_compute/table_test.py` — `build_mechanism_object_table` produces the
   same table `Cell.mech_table()` did, for a cell carrying both a density
   mechanism and a placed synapse.

Edge cases considered for the coverage change:

- **Zero-area CV.** `interval_area_fraction` returns `0.0` when
  `lateral_area_um2 <= EPS_AREA_UM2`, so such a CV drops out of
  `Cell.on(region)`. That matches the physics, which already skips
  `fraction <= EPS_PARAM` at `_discretization/mechanism.py:653`.
- **Region covering a whole branch.** Both measures give exactly 1.0; the
  clamp to `[0, 1]` is retained.
- **Multiple disjoint intervals on one branch.** Areas add, as lengths did.
- **Zero-length radius jump inside a CV.** Handled by the existing
  `_build_frusta` path, which the physics already uses.

Edge cases for the discretization key:

- Morphology reassigned (`init_state`, `reset`) — both sites already
  invalidate; the revision component is redundant there and harmless.
- Morphology mutated between `paint()` calls — `paint` invalidates, so the
  rebuild happens either way; the key now agrees.
- Morphology mutated after `init_state()` — `init_state` clones, so the
  runtime is insulated; the declaration-time views now track the clone's
  revision rather than the caller's tree.

---

## Verification

Run from the worktree after every item above had landed.

```
$ pytest braincell/_multi_compartment -q
161 passed, 24 warnings, 59 subtests passed in 18.43s

$ pytest braincell/ -q
2814 passed, 15 skipped, 411 warnings, 411 subtests passed in 204.00s (0:03:23)

$ pre-commit run --files <23 changed files>
check for added large files..............................................Passed
check python ast.........................................................Passed
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
```

Against the `eddc00c` baseline at the top of this document: the module suite
gains 4 tests (157 → 161) and the full suite gains 7 tests and one subtest
(2807/410 → 2814/411), with no test lost or skipped. `ruff format` reflowed
two files on its first pass; the numbers above are from the re-run after that
reflow. Net diff: 22 source and test files changed, 916 insertions,
513 deletions, plus this spec.
