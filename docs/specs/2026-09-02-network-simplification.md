# Network simplification sweep

Iteration 12 of the module-by-module `/simplify` sweep. Target:
`braincell/network/` — 5,102 lines across ten non-test modules, plus 1,802
lines of co-located tests.

| Module | Lines | Role |
| --- | --- | --- |
| `__init__.py` | 70 | Package surface, lazy via PEP 562 `__getattr__` |
| `core.py` | 403 | `Population`, `NetworkResult` — the leaf both layers share |
| `event.py` | 775 | `EventSource`, `NetStim`, `EventSequence`, spike plumbing |
| `recording.py` | 613 | `RecordingSpec`, `EventSeries`, `observe` |
| `lowering.py` | 189 | `ConnectionBlock`, delay quantization |
| `delivery.py` | 665 | Ring buffers, arrival writes, backend selection |
| `pairing.py` | 872 | `PairingSpec` grammar and its materialization |
| `connection.py` | 842 | `_ConnectionStore`, `ConnectionView`, `connect()` |
| `engine.py` | 872 | `Network` — populations, run loop, results |
| `_testing.py` | 206 | Shared test doubles |

Baseline on `4fdfd98`:

- `pytest braincell/network -q` → 113 passed, 2 subtests passed (31.38 s)
- `pytest braincell/ -q` → 2814 passed, 15 skipped, 411 subtests passed (219.63 s)

Four review passes (reuse, simplification, efficiency, altitude) produced 61
findings. This document records which are applied, which are declined and
why, and which are deferred with the evidence needed to pick them up later.
The applied set groups into five themes.

## Constraints this package carries

The package is a documented public surface: `docs/design/network/api.md`
(36 KB) spells out the `connect()` and pairing grammar, and
`examples/multi_compartment/connection.ipynb`,
`examples/multi_compartment/network.ipynb`, and
`examples/profiling/connection_sampling_benchmark.py` call it. Any rename
lands in all four in this PR.

`braincell.mech` must stay a leaf — it owns the declaration contracts that
`network.connection`, `_base_channel`, and `_compute.state` all depend on.
`braincell/network/__init___test.py::MechIsALeafTest` guards that, and this
sweep does not touch it.

---

## Theme 1 — The lazy `__init__` guards a cycle that no longer exists

`braincell/network/__init__.py` resolves `Network`, `NetworkConnections`,
and `ConnectionBlock` through a PEP 562 `__getattr__` rather than importing
them. Its docstring (`__init__.py:16-31`) and the guard test that enforces
it (`__init___test.py:16-33`) both assert that an eager import would
"re-enter `braincell._multi_compartment` in the middle of its own
initialization" and that `import braincell` "would die with an
`ImportError`".

**That is measurably false as of `4fdfd98`, and the guard that enforces it
names the wrong modules.**

### 1.1 The claim does not reproduce

`dev/eager_probe.py` adds one eager submodule import to
`braincell/network/__init__.py` at a time and runs `python -c "import
braincell"` in a fresh subprocess:

```
connection  import braincell OK
delivery    import braincell OK
engine      import braincell OK
lowering    import braincell OK
pairing     import braincell OK
recording   import braincell OK
```

`dev/eager_init.py` goes further and replaces the whole file with a fully
eager version — four plain `from .x import Y` lines and an `__all__`, no
`__getattr__`, no `_LAZY_ATTRS`, no `__dir__`:

```
2807 passed, 15 skipped, 411 warnings, 411 subtests passed in 194.78s
```

2807 is the full suite minus exactly the 7 tests in
`braincell/network/__init___test.py`, which the probe excludes because they
assert the lazy mechanism itself.

The reason is visible in `braincell/__init__.py`: line 57 imports
`._multi_compartment`, whose `cell.py:87` imports `braincell.network.event`
and so runs `network/__init__.py` — and none of `engine`, `connection`, or
`pairing` import a *name* from the partially-executed
`braincell/_multi_compartment/__init__.py`. They import its *submodules*
(`.probes`, `.run`, `.synapses`), which Python resolves against a partial
parent without complaint.

The laziness also buys no deferral, because `braincell/__init__.py:66` and
`:70-79` import `Network`, `NetworkConnections`, `ConnectionView`,
`EventSource`, and `observe` thirteen lines later regardless. Every module
the `__getattr__` defers is loaded during `import braincell` anyway.

**Applied:** `braincell/network/__init__.py` becomes three eager import
lines binding four names. `_LAZY_ATTRS`, `__getattr__`, `__dir__`, and the
`TYPE_CHECKING` block go, and the module docstring now states the invariant
that holds instead of the cycle that does not.

### 1.2 `_HEAVY_SUBMODULES` names the wrong three modules

`__init___test.py:46` hardcodes `{".connection", ".engine", ".lowering"}`.
`dev/closure.py` walks each submodule's load-time import closure (skipping
`TYPE_CHECKING` and function bodies) and asks which actually reach
`braincell._multi_compartment`:

```
connection  REACHES  ['braincell._multi_compartment.synapses']
core        clean
delivery    clean
engine      REACHES  ['braincell._multi_compartment', ...cell]
event       clean
lowering    clean
pairing     REACHES  ['braincell._multi_compartment.synapses']
recording   clean
```

The set is wrong in both directions: it misses `.pairing`, which does reach
`_multi_compartment`, and it lists `.lowering`, which does not. A
demonstration that the guard is not merely imprecise but inert — adding
`from .pairing import degree` to `__init__.py` and running the guard:

```
$ python -m pytest braincell/network/__init___test.py -q -k eagerly
1 passed
```

**Applied:** the hardcoded set goes. `LazyInitTest` is replaced by
`ImportGraphTest`, which (a) keeps the empirical
`test_importing_braincell_succeeds`, (b) pins the intra-package import DAG
in the `braincell/_compute/__init___test.py` idiom so any new sibling edge
surfaces in review, and (c) asserts the invariant that actually holds —
that no `braincell/network` module imports a name from
`braincell._multi_compartment`'s package root, only from its submodules.
That is the property the eager import depends on, and it is checkable.

### 1.3 `ConnectionBlock` is not public

No importer outside `braincell/network/` — only prose in
`docs/design/network/*.md`. It is a lowering intermediate consumed solely
by `delivery.py`. **Applied:** dropped from `__all__`.
`docs/design/network/module-layout.md` is updated for 1.1–1.3.

---

## Theme 2 — Delete what nothing reaches

### 2.1 `ConnectionBlock` carries three structurally constant fields

`lowering.py:89-108` is the only construction site in the repository, and
it hardcodes `packed=True`, `source_cv_id=0`, `event_source=source`. Every
downstream branch on those values is therefore unreachable:

| dead | site |
| --- | --- |
| `route_source_event`'s `else` | `delivery.py:143-144` |
| `source_spike` (only caller is that `else`) | `delivery.py:131-135` |
| `population_spike` (test-only) | `delivery.py:108-128` |
| `zeros_like_events` (reachable only via `zero_arrival`'s dead leg) | `delivery.py:267-286` |
| `zero_arrival`'s fork and its `post_size=` parameter | `delivery.py:231-249` |
| `synapse_index if packed else post_index * n_active + synapse_index`, ×2 | `delivery.py:183-185`, `202-204` |
| `(n_active,) if packed else (post_size, n_active)`, ×4 | `delivery.py:457-462`, `494-498`, `504-508`, `514-518` |
| `post_size` parameters threaded to reach those forks | `delivery.py:252`, `303`, `598` |

**Applied:** `packed`, `source_cv_id`, and `buffer_size` (written at
`lowering.py:101`, never read) are deleted; `event_source` becomes
non-optional; every `X if packed else Y` becomes `X`.

### 2.2 `DeliveryBlock.post_index` and `.synapse_index` are write-only

Set at `delivery.py:181-182` and `200-201`. Every read in the module
(`:174`, `:175`, `:193`, `:194`) is on the *source* `ConnectionBlock`, never
on a `DeliveryBlock`. `flat_target_index` is the field the runtime actually
uses. **Applied:** both fields deleted.

### 2.3 `write_arrivals(blocks, ...)` never touches `blocks`

`delivery.py:375-414` iterates `state.queue_keys`. Confirmed by
`ruff check --select ARG`. **Applied:** parameter and its docstring entry
removed; the sole call site at `engine.py:617-621` updated.

### 2.4 `lowering.resolve_source_cv` has no production caller

Only `lowering_test.py:23,54,55,57`. It duplicates
`event._resolve_cell_location_cv` (`event.py:632-636`), differing only in
rejecting an already-resolved `LocsetMask`. **Applied:** deleted, along
with the `LocsetExpr` and `locate_cv_on_branch` imports it alone needed;
the three test assertions move to `event_test.py` against
`_resolve_cell_location_cv`.

### 2.5 Members with no reader

Each verified by grep over `braincell/`, `examples/`, `docs/` (`.py`,
`.ipynb`, `.md`). **Applied:**

- `EventSourceView.ids` (`event.py:135-138`), an alias for `source_id`.
- `_CompiledRecording.is_scheduled` (`recording.py:270-276`) — every
  `.is_scheduled` hit in the repo is `EventSource.is_scheduled`.
- `RecordingSpec.legacy` (`recording.py:143`) — never passed, never read.
- `_RunSetup.delivery_backend` (`engine.py:56`) — written at `:521`; the
  three reads at `:490`, `:493`, `:500` are all of the local variable.
- `times = start_t + relative_times` (`engine.py:402`) — overwritten at
  `:413` before any read.

Four more that the review pass reported as dead survived a second look and
are **declined**; see the Declined section for `ConnectionView.root` /
`EventSourceView.root`, `GroupContext.size` / `.n_groups`, and the
`Score` / `Degree` aliases. "No in-repo reader" is necessary but not
sufficient for deletion on a public surface.

### 2.6 `Population.__init__` re-runs the check `set()` makes three lines later

`core.py:68-70` and `core.py:130-132` are the same three lines with the same
message, and nothing between them consumes `metadata`. **Applied:** the
first copy deleted.

---

## Theme 3 — Stop rescanning every row

Every item here is measured, and every measurement is **runtime**, not
trace time. Scripts are in the gitignored `dev/`; sizes are stated with
each number.

### 3.1 `Network.run()` re-validates the whole topology on every call

`engine.py:302` calls `self._validate_direct_source_ownership()` *above*
the `if self._initialized: return self` guard at `:303`. That validation
walks `_call_views()` (3.2), so a cached-loop 20-step `run()` of a
64-call / 25,600-row network costs **270 ms, ~90% of it in `_call_views`**
(profile: 0.405 s cumulative of 0.449 s). Chunked-run loops pay it per
chunk. **Applied:** the `_initialized` early return moves above the
validation.

### 3.2 `ConnectionView._call_views()` is O(calls × rows)

`connection.py:415` recomputes `self._active_ids` and `self.connect_id` per
connect call, each walking every row through `_ConnectionStore.rows()`
(`connection.py:175`, a Python dict comprehension).

| rows | current | single pass |
| --- | --- | --- |
| 64 calls × 400 | 240 ms | 0.6 ms |
| 256 × 400 | 4.45 s | 3.0 ms |
| 256 × 1600 | **20.7 s** | **9.1 ms** |

Three hot callers: `engine.py:750` (every `run()`), `lowering.py:66`, and
`_multi_compartment/cell.py:2221` — the last once per synapse layout.
**Applied:** `_ConnectionStore.rows()` gets an `int64` id→row lookup array
instead of the dict comprehension, and `_call_views` groups by the already
non-decreasing `connect_id` with one `np.searchsorted` pass.

### 3.3 `_ConnectionStore.weight_for` scans for every row

`connection.py:192` runs `np.flatnonzero(call.row_ids == row_id)` per row
and builds one `Quantity` per row. `view.weight` at 16,000 rows: **191 ms**
(~142 ms Python/`Quantity` churn, ~50 ms quadratic scan). In a real
trace-and-compile profile at 32 calls × 200 rows it is **0.202 s of a
1.11 s network setup — 18%**. `call.row_ids` is a contiguous `np.arange`,
so the row's position is `row_id - call.row_ids[0]`. **Applied:** one
grouped gather per call, concatenated once.

### 3.4 `_ConnectionStore.set_weight` copies the whole array per row

`connection.py:211` → `_set_quantity_or_array` (`:817`) rebuilds the entire
weight array on each iteration. `ConnectionView.set(weight=...)`: 8,000
rows 293 ms, 16,000 rows 661 ms, 32,000 rows **1.63 s** — 36.6 → 51.1 µs
per row, superlinear. **Applied:** one grouped assignment per call, using
the same offset arithmetic as 3.3.

### 3.5 `pairing._score` allocates a dense matrix for the uniform case

`pairing.py:761` builds `np.ones((n_unique, n_candidates))` even when
`score is None`, and `_sample` then rescans it row by row. The shape in
`_conditional_sample` is the product of two population sizes, so it grows
quadratically.

**Applied:** `_score` returns `None` for the uniform case and `_score_row`
carries the sentinel to `_sample`. Measured on a `by_source(6)` connect
over 1000 sources × 5000 synapses (`tracemalloc` around `connect()`):

| | peak allocation | `connect()` |
| --- | --- | --- |
| before | 39.1 MiB | 784 ms |
| after | **1.5 MiB** | 789 ms |

This is a memory result, not a speed one — `rng.choice` dominates the
loop, so wall time is unchanged within noise. It is worth taking anyway
because the 39.1 MiB scales with `n_sources × n_synapses`.

`_sample` passes an explicitly materialized uniform `p` rather than
`p=None`. Two reasons, both measured. `p=None` and `p=uniform` **do not
draw the same values for the same seed** — so `p=None` would silently
change the connectivity of every default-pairing network, which the
"Declined: unify the two seed schemes" entry below rejects for the same
reason. And `p=None` is not even faster: for `replace=False` at
200 × 5000 it took 414 ms against 206 ms for the weighted path.
`np.full(n, 1.0 / n)` is bit-identical to `np.ones(n) / n.sum()`, verified
across nine sizes, so draws are preserved exactly.

### 3.6 `make_delivery_op` builds operators the default backend never calls

Instrumented run (`dev/bench_delivery.py`): `delivery ops built=1, invoked
during trace=0`. On the scatter path `delivery_blocks(group_by_delay=False)`
always produces per-contact `delay_steps` arrays, so
`enqueue_future_events` and `apply_immediate_events` take the vector branch
and never index `state.delivery_ops`. Each unused op still materializes two
device arrays (800 KB per 100k-contact block), closes over the whole
`DeliveryBlock`, and is retained in `_RunSetup` inside `_run_setup_cache`
for the network's lifetime. **Applied:** ops are built only when
`delivery_backend == "brainevent"`.

---

## Theme 4 — Say it once

### 4.1 `NetworkResult.concat` builds throwaway `RunResult`s to reach one combiner

`core.py:340-355` constructs a full `RunResult` per part per recording —
with a `traces=` dict that is immediately discarded — to call
`RunResult.concat(...).samples[name]`, re-running the entire
contiguity/schema validation loop that `NetworkResult.concat` already ran
at `:325-339`. The primitive wanted is `_concat_sample_blocks`
(`_multi_compartment/run.py:283`), a pure function over `SampleBlock`s.

**Applied:** `_concat_sample_blocks` is promoted to public
`concat_sample_blocks` in `network/recording.py`, next to the `SampleBlock`
it operates on; `RunResult.concat` and `NetworkResult.concat` both call it.
This also removes `core.py`'s deferred `_multi_compartment.run` import,
making `core.py` a leaf with no cross-package dependency at all.

`RunResult.concat` also raised `"Recording schema changed for {name!r}."`
on a schema mismatch, which `NetworkResult.concat`'s own loop did not
check. That check is lifted into `core.py` so the diagnostic is not lost.

### 4.2 `_quantity_vector` exists twice, byte-identical

`connection.py:794-807` and `event.py:712-722`, same four error messages,
differing only in the keyword name (`count=` vs `size=`). `connection.py`
already imports from `.event`. **Applied:** the `connection.py` copy is
deleted and the `event.py` one imported.

### 4.3 Three "make this read-only" helpers, diverged on `Quantity`

`core.py:270-273` (`_readonly_array`, ndarray only),
`recording.py:600-613` (`_freeze_array`, also handles `u.Quantity`), and
`engine.py:850-851` (inline, ndarray only). The divergence is visible:
`Population.set(...)` freezes numpy metadata but returns `Quantity`
metadata unfrozen, because `_readonly_array` has no `Quantity` branch.
**Applied:** `_freeze_array` moves to `braincell/_misc.py` as
`freeze_array` — which already hosts `concat_values` and
`same_time_quantity` — and all three sites call it.

### 4.4 `_concat_event_series` re-inlines the `concat_values` its own file imports

`core.py:396-399` spells out `u.Quantity(u.math.concatenate(tuple(
item.time.to_decimal(unit) ...), unit)`, which is `_misc.concat_values`
character for character — imported at `core.py:26` and used two lines away.
**Applied.**

### 4.5 Two different `_stack_values` in sibling modules

`connection.py:828-833` (`np.asarray([...])`, no empty guard) and
`recording.py:548-555` (`u.math.stack(..., axis=-1)`, empty guard). Same
name, same package. The review pass called them "interchangeable in
practice". **Declined** — see the Declined section; they are not, and the
substitution would silently downcast every public `ConnectionView.weight`.

### 4.6 `_require_nonempty_string` written twice

`connection.py:756-758` and `recording.py:595-597` are identical.
**Applied:** one copy, `recording._require_name`, imported by
`connection.py` at its nine call sites.

### 4.7 `ConnectionView._require_single_call` recomputes the store's own answer

`connection.py:481-485` spells out
`tuple(dict.fromkeys(int(i) for i in self.connect_id.tolist()))`, which is
the body of `_ConnectionStore.active_call_ids` (`:188-190`). **Applied:**
it calls the store. This also removes one of the `connect_id`
materializations 3.2 is about.

### 4.8 `_as_source_view` duck-types a class it is allowed to name

`connection.py:702-717` probes `hasattr(source, "model")` and
`hasattr(source, "kind")` under a comment claiming `Population` is imported
"structurally to keep the low-level connection module independent from the
Network package". `connection.py` *is* in the network package, and
`core.py` is a graph leaf that `lowering.py:29` already imports.
**Applied:** `from .core import Population` and `isinstance`. The
`connection → core` edge keeps the package graph a DAG.

### 4.9 `Network.init_state` and `reset_state` duplicate their guard and lifecycle

`engine.py:298-313` and `:335-358`: the `batch_size is not None`
`NotImplementedError` block is character-for-character identical, and both
wrap their loop in the same `_cell_lifecycle_active = True / try /
finally = False`. **Applied:** a module-level `_reject_batch_size` and a
`_cell_lifecycle` context manager.

### 4.10 `RecordingRow(**{**row.__dict__, ...})` at three sites

`recording.py:254`, `:522-529`, `:536-545`. `__dict__` ordering on a frozen
dataclass is an implementation detail; `dataclasses.replace` is the API.
**Applied.**

---

## Theme 5 — Docstrings that describe the branch that cannot run

All are consequences of Theme 2 and land with it:

- `delivery.py:59-61` documents `flat_target_index` as
  `post_index * n_active + synapse_index`. That is the dead branch; it
  always equals `synapse_index`.
- `delivery.py:245-249` says `zero_arrival` returns shape
  `(post_size, n_active)`. It always returns `(n_active,)`.
- `delivery.py:384-386` documents `write_arrivals`'s `blocks` parameter
  (2.3).
- `delivery.py:437-441` describes the packed/broadcast fork that never
  forks.
- `delivery.py:152-170` omits `group_by_delay` from `Parameters` while
  documenting it under `Returns`. NumPy-doc requires the entry, and unlike
  the rest of this list `group_by_delay` is genuinely live —
  `engine.py:493` passes both values.
- `lowering.py:102-103` comments that `source_cv_id` is "only used by raw
  Cell spike blocks", a block kind that no longer exists.
- `core.py:279-288` lists three of `NetworkResult`'s seven fields, omitting
  `samples` — the one `Network.run` actually populates.

---

## Breaking changes

Every in-repo caller is updated in this PR. No deprecation shims, no
aliases, no warnings.

1. **`braincell.network.NetworkRunResult` removed.** It was a bare
   `NetworkRunResult = NetworkResult` alias whose comment claimed it was
   "retained for existing callers"; there are none in the repository, and
   `engine.py` imported the alias rather than the real name. Use
   `braincell.network.NetworkResult`.
2. **`braincell.network.ConnectionBlock` removed from the package
   `__all__`.** Still importable from `braincell.network.lowering`; it is a
   lowering intermediate with no external importer.
3. **`ConnectionBlock.packed`, `.source_cv_id`, and `.buffer_size` removed;
   `.event_source` is now required.** All three were constant at the single
   construction site.
4. **`DeliveryBlock.post_index` and `.synapse_index` removed.** Write-only;
   `flat_target_index` is what the runtime reads.
5. **`delivery.population_spike`, `delivery.source_spike`,
   `delivery.zeros_like_events` removed**, and `delivery.zero_arrival` lost
   its `post_size` keyword. All were reachable only through `packed=False`.
6. **`delivery.write_arrivals` lost its `blocks` parameter.**
7. **`lowering.resolve_source_cv` removed.** Use
   `braincell.network.event._resolve_cell_location_cv`, which it
   duplicated.
8. **`EventSourceView.ids`, `RecordingSpec.legacy`,
   `_CompiledRecording.is_scheduled`, and `_RunSetup.delivery_backend`
   removed.** No reader in the repository. `EventSourceView.ids` was a bare
   alias for `.source_id`, which stays.
9. **`connection._quantity_vector` and `connection._require_nonempty_string`
   removed** in favour of the `event._quantity_vector` and
   `recording._require_name` copies they duplicated.
10. **`braincell._multi_compartment.run._concat_sample_blocks` moved and
    renamed** to `braincell.network.recording.concat_sample_blocks`.
11. **`braincell/network/__init__.py` no longer defines `__getattr__` or
    `__dir__`.** The three names it resolved lazily are now imported
    eagerly; attribute access is unchanged.

---

## Declined

Five of these overrule a review finding. Each is recorded with the check
that overturned it, because "no in-repo reader" and "identical in practice"
are both cheap to assert and expensive to get wrong.

- **Replace `connection._stack_values` with `recording._stack_values`
  (4.5).** They are not interchangeable. Measured on the same input:

  ```
  connection._stack_values -> ndarray   float64
  recording._stack_values  -> ArrayImpl float32
  ```

  `recording`'s goes through `u.math.stack`, which lands on JAX and so
  inherits `jax_enable_x64=False`. `connection._stack_values` is what
  `weight_for` returns, i.e. what `ConnectionView.weight` hands the user.
  Substituting would silently downcast every published weight from float64
  to float32 and change its type. The duplicate name is a real smell; the
  fix is to rename one, not to merge them, and that belongs with the naming
  pass deferred below.
- **Delete `ConnectionView.root` and `EventSourceView.root`.** They have no
  in-repo reader, but they are one instance of a view-family convention:
  `CellView.root` and `IonView.root` are the same accessor on the same kind
  of object and *are* tested (`_multi_compartment/cell_test.py:219`, `:296`,
  `:380`). Deleting two members of a four-member convention makes the API
  less predictable, not smaller.
- **Delete `GroupContext.size` and `.n_groups`.** Nothing inside `pairing.py`
  reads them, which is the point: `GroupContext` is handed to *user* score
  and degree callables as `ctx.group`. "No reader in this repository" does
  not apply to a field whose only intended readers are out of tree.
- **Delete the `Score` and `Degree` type aliases** (`pairing.py:33-34`).
  They are documentary — they name the two callable protocols the pairing
  grammar accepts, and `docs/design/network/api.md` uses that vocabulary.
  AGENTS.md does say shared type expressions belong in `_typing.py`, but
  these are used in one module; moving them there is iteration 14's call.
- **Adopt `mech._validate.require_str` for the network's nine inline
  non-empty-string checks.** `require_str` raises `TypeError` for a
  non-`str`; every network copy raises `ValueError`, and
  `connection_test.py` / `core_test.py` assert `ValueError`. Changing the
  exception type is a breaking change with no benefit beyond the
  deduplication 4.6 already gets within the package.
- **Unify the two deterministic seed schemes** (`pairing._derived_seed`,
  blake2b/`\x1f`/4 bytes; `NetStim._bind_network_seed`, blake2b/`\0`/8
  bytes masked to 31 bits). They answer the same question with different
  bit widths, but merging them changes which random streams existing
  networks draw — silently different connectivity for the same seed. Worth
  doing deliberately, with a note in the changelog, not as a side effect of
  a cleanup sweep.
- **Route `recording._positive_quantity` / `_nonnegative_quantity` through
  `_misc.validate_time_quantity`.** It hardcodes `to_decimal(u.ms)` and the
  `frequency` call site (`recording.py:152`) is in `u.Hz`; the local copies
  also check `np.isfinite`, which the shared helper does not. The honest
  fix adds `unit=` and `require_finite=` parameters to `_misc.py`, which is
  iteration 13's target — deferred there rather than half-done here.
- **Collapse `pairing._materialize_group`'s `_First` arms.** The mirror is
  real and the seed labels match, so RNG streams would be preserved — but
  the adjacent `_ByEndpoint` and `_MatchDegrees` arms look identically
  mirrored and are not: the synapse side passes `positions=synapses` to
  `_group_degree` (`:663`, `:682`) and the source side does not. Collapsing
  one pair of four while its neighbours stay expanded makes the asymmetry
  harder to see, not easier.
- **Drop `EventSource(ABC)`** (ruff `B024` — it declares no abstract
  members, and `event.py:97` duck-types `current_event_count` via
  `getattr`). The `ABC` base and the `NotImplementedError` at `:98-99` are
  the documented extension contract for third-party sources; removing them
  is an API decision, not a cleanup. Folded into the two-ABC deferral
  below.
- **Split `_CONNECT_CALL_WARNING_THRESHOLD`** (`connection.py:66`) into a
  warning threshold and a repr limit. Two policies on one constant is a
  real smell, but both are 256 and the constant is package-private; the
  change is cosmetic and would need a judgement call on the new repr limit
  that belongs with the repr work deferred below.

---

## Deferred, with evidence

### Own PR — architecture

- **`recording.py` is declared a leaf but half of it is a
  `_multi_compartment` module.** `compile_recording` (`:241`) and eight
  helpers below it resolve a live `Cell` against its runtime, staying
  "leaf" only by deferring imports into function bodies (`:321`, `:383`,
  `:421`, `:441`, `:442`). Meanwhile `_multi_compartment/cell.py:88`
  imports `compile_recording` at module scope. Split at the
  declaration/resolution seam: `observe`, `RecordingSpec`, `RecordingRow`,
  `RecordingSchema`, `SampleBlock`, `EventSeries` stay; everything from
  `compile_recording` down moves to `_multi_compartment/`. ~370 lines
  moved plus a test-file split.
- **`event.py` has the same shape.** `VoltageCrossingSource` (`:409`),
  `_CellSpikeSource` (`:559`), `EventOutputCollection` (`:604`), and the
  two `_resolve_cell_location_cv*` helpers (`:632`, `:639`) know
  `cell.morpho`, `cell.cv_tree`, `cell.V_th`, `cell.spike`,
  `cell._event_previous_V` (`:542`), and `cell._raise_if_not_initialized`
  (`:530`, `:596`), and defer `braincell.filter` and
  `braincell._discretization.base` imports. `_multi_compartment/cell.py:87`
  then imports two of them back out — one underscore-private, and neither
  in `event.__all__`. Should land with or after the `recording.py` split;
  same seam.
- **Two independent delayed-event queues.** `ConnectionView`'s
  `prepare_runtime` / `_prepare_live_runtime` / `reset_runtime` /
  `clear_runtime` / `_live_event_count` (`connection.py:433-479`) plus the
  three `_ConnectionCall` slots (`:80-82`) and
  `Cell._apply_direct_live_connection_events`
  (`_multi_compartment/cell.py:2240-2274`) are a second implementation of
  what `DeliveryState` already does (`delivery.py:80-105`, `289-533`). They
  are mutually exclusive at runtime — `prepare_runtime` is called only from
  `_multi_compartment/run.py:123`, so `live_history` stays `None` in
  network mode. They have diverged three ways: the standalone path
  hardwires nearest-step quantization (`connection.py:450`) where lowering
  offers four policies (`lowering.py:170-182`); it uses a shift register
  (`connection.py:478`) where delivery uses a ring and a modular cursor
  (`delivery.py:450-452`); and a delayed event lands *after*
  `_update_dynamics` in standalone (`run.py:224-226`) but *before*
  `_begin_step` in network mode (`engine.py:616-627`). ~90 lines
  duplicated. Unifying is semantics-visible, not a cleanup.

  Note for whoever takes it: the reuse pass's original lead named three
  `Cell` methods here. Two were refuted —
  `_apply_synapse_layout_event_drive` is the *shared sink*
  (`delivery.py:522` calls it), and `_evaluate_contact_inputs` handles
  *scheduled* sources, which `lower_direct_connections` explicitly excludes
  (`lowering.py:66` passes `scheduled=False`). Only the live-delay queue is
  genuinely duplicated.

- **Naming pass.** "target" means the post-population name
  (`connection.py:536`), a `Population`-or-`CellView` (`engine.py:186`), a
  position in the originally-passed `SynapseView` (`connection.py:306`),
  and a slot in a synapse layout (`delivery.py:76`). `DeliveryBlock.source`
  is a `ConnectionBlock`, not an event source, while
  `ConnectionBlock.event_source` is the actual `EventSource` — hence
  `block.source.event_source` at `delivery.py:140`. Renames across five
  modules, and `target_index` is asserted in `connection_test.py:351,368,370`
  and `pairing_test.py:55,331,338`.
- **Two `EventSource` ABCs.** `is_scheduled` (`event.py:72-74`) is doing
  subclass dispatch by hand: scheduled sources override `event_count` and
  expose `.events`; live sources implement `current_event_count`, which the
  base reaches by `getattr`. A `ScheduledEventSource` / `LiveEventSource`
  split makes `is_scheduled` a class fact and turns `connection.py:419`'s
  filter into a type filter. Carries the `execution_owner`-returns-`None`
  ternary at `connection.py:591`, `engine.py:705`, `engine.py:752` and the
  `port = "spike" if kind == "netstim" else "event"` rule written twice
  (`core.py:100`, `engine.py:782`).
- **Retire `Population.kind`.** A stringly-typed enum switched on at
  `core.py:83,94,100,106,113,162`, `connection.py:541,570`,
  `engine.py:172,685,689,716,739,774`, `lowering.py:62` — a hand-rolled
  type test over the `EventSource`/`Cell` polymorphism that already exists.

### Own PR — performance, measured

| site | cost | size |
| --- | --- | --- |
| `event.py:346`, `:697` scheduled `event_count` rebuilds the full `(rows, events)` arrival matrix **inside the compiled step** | NetStim 4000 events / 4000 rows: **4.25 s / 500 steps** (8.49 ms per step); EventSequence 20,000 events: **16.8 s** (33.5 ms per step). The lowered live path over the same sizes is flat at 94–104 ms. | `arrival_step` has no `t` dependence — quantize on the host once and index a per-step count table, or route scheduled sources through the same ring buffer |
| `pairing.py:740` `_conditional_sample` makes one `RandomState` + `rng.choice` per unique endpoint | `by_source(deg=5)`, 800×1600: **1.26 s of a 1.28 s `connect()`**; ~870–1010 µs per endpoint, linear, so 10k sources ≈ 9 s | one batched draw per group; changes RNG streams, so it needs the same deliberate call as the seed unification |
| `pairing.py:312` `_synapse_geometry` rebuilds `MorphologySpatialGeometry` per `connect()` | `build(CA1.swc)` is uncached at **15.4 ms**; the row loop is **196 ms for 2,000 synapses** (98 µs/row). 100 geometry-scored connects on one cell repeat ~20 s | cache on the morphology keyed by `Morphology.revision` (the idiom `filter/cache.py` and `Cell._discretization_key` already use); group rows by `branch_id` |
| `delivery.py:442` blocks sharing one queue run sequentially | same 6400 rows, 200 steps: 1 block 121 ms, 8 blocks 164 ms, 32 blocks 318 ms, 64 blocks **547 ms** — ~33 µs/step per extra block | concatenate blocks sharing `(event source, post_population, layout_id)` in `delivery_blocks()` |
| `connection.py:497`, `:573`, `engine.py:118` repr paths rescan every row per connect name | `repr(ConnectionView)` at 20,000 rows / 400 connects: **1.10 s**; `str(Network)` at 128 connects / 6400 rows: **173 ms** | index a `connect_id → name` array; one grouping pass instead of one scan per name |
| `connection.py:143` `_ConnectionStore.add` re-concatenates seven SoA columns per call | 16,000 rows in 1,600 calls **65.0 ms** vs in 10 calls **4.0 ms** | accumulate in lists, materialize on first read |
| `pairing.py:752` per-row Python scatter in `_conditional_sample` | 30.6 ms at 100,000 rows | `np.argsort(inverse, kind="stable")` + one scatter |
| `delivery.py:190` re-derives `np.asarray(block.delay_steps)` inside the delay-group loop | 30.6 ms at 100,000 contacts × 1,000 delays; only reachable on the `brainevent` backend | hoist the `asarray`; group with one `argsort`+`split` |

### Iteration 13 (root modules)

- `_misc.validate_time_quantity` needs `unit=` and `require_finite=` before
  `recording.py`'s two local copies can go (see Declined).
- Nine copies of `float(np.asarray(x.to_decimal(u.ms), dtype=float).reshape(()))`
  — `engine.py:542,770,771,808`, `connection.py:439`, `lowering.py:162`,
  `recording.py:571` (×2), plus `run.py:266-267,277-278`. One
  `_misc.to_ms_scalar` covers all nine.

### Iteration 14 (whole package)

- **`braincell/network/` imports no alias from `_typing.py`** — zero hits
  across all ten modules, where AGENTS.md makes them mandatory. `T` and
  `DT` cover every `dt`/`duration`/`delay` parameter
  (`engine.py:360-368`, `lowering.py:53`, `:131`).
- **Cross-package private imports**, each a missing public API:
  `connection.py:28` (`_multi_compartment.synapses._cell_label`),
  `engine.py:29` (`run._duration_steps`, `run._recording_time_mask`),
  `recording.py:442` (`density_views._runtime_layout`), `pairing.py:31`
  (`morph._spatial.MorphologySpatialGeometry`, `interpolate_branch` — that
  module declares `__all__` for exactly these two and three packages import
  them, so they should be re-exported from `braincell.morph`). And in
  reverse: `_multi_compartment/cell.py:87` imports `_CellSpikeSource`, and
  `:1352` imports `connection._ConnectionStore`.
- **`Network.run` re-implements `Cell.run`'s sample assembly.**
  `engine.py:423-448` is structurally `run.py:146-159`, and
  `engine._normalize_scan_samples` (`:856`) is `run._normalize_run_traces`
  (`:247`) with different error text. The shared home is
  `network/recording.py`, which already owns `SampleBlock` and
  `RecordingSchema`. Held back from this sweep because 4.1 already moves
  one combiner across that boundary and two moves in one PR would make the
  `_multi_compartment`/`network` seam hard to review.
- **`engine._event_source_metadata` (`:841`) hardcodes the union of every
  source type's attributes** — it probes `("population_index",
  "location_index", "cv_id")` by `getattr` and silently skips misses
  (`:849`). Wants `EventSource.row_metadata(source_ids) -> dict`.
- **`connect()` / `_connect_with_pairing_seed`** (`connection.py:602`,
  `:648`): the public wrapper always passes `pairing_seed_root=0,
  pairing_seed_path=()`, and the only caller passing anything else is
  `Network.connect` (`engine.py:227-236`) — which at its second call site
  (`:266-267`) computes a seed and discards it. `engine.py:31` imports both
  `_UNSET` and `_connect_with_pairing_seed` by their underscore names.
  Resolving the seed in `Network.connect` via
  `dataclasses.replace(pairing, seed=...)` collapses this to one public
  signature. Deferred because it interacts with the seed-unification
  decision above.
- **`Network._run_scheduled_sources_only` (`engine.py:807`) is a second
  `run()`** with its own dt-freeze field (`_scheduled_dt_ms`, `:91`)
  parallel to `_runtime_config` (`:90`), its own time bookkeeping, and its
  own result construction. `_source_current_time` (`:94`) is written by the
  main path at `:456` and read only here (`:814`).
- **`Population._RESERVED_NAMES` (`core.py:47-62`) is a hand-maintained
  frozenset of the class's own attribute and method names** — it will drift
  the first time someone adds a property. Derive it from the class
  namespace.
- **`core_test.py:23` and `recording_test.py:22` import `_cell` from
  `braincell._multi_compartment.selection_test`** — a helper taken from
  another package's *test* module, against the `_testing.py` rule in
  AGENTS.md. `braincell/network/_testing.py` is the right home.
- **`engine.py:163`**: `if callable(model) and not isinstance(model,
  EventSource) and not hasattr(model, "pop_size")` — a three-clause
  structural probe with one concrete-type `isinstance` inside otherwise
  duck-typed code, to decide "is this a factory".
- **`pairing._synapse_geometry` (`:312-345`) is the third builder of the
  same six morphology fields**, alongside `filter/_sampling.py:234-260`
  (`_context` → `SamplingContext`) and `_discretization/context.py:75-108`
  (`CVContext`). All three call `MorphologySpatialGeometry.build`; two call
  `interpolate_branch`.

---

## Edge cases and tests

New regression tests:

1. `__init___test.py` — the import DAG is pinned as a dict literal, and a
   new test asserts that no `braincell/network` module imports a name from
   `braincell._multi_compartment`'s package root. That is the property the
   now-eager `__init__.py` depends on, replacing a `_HEAVY_SUBMODULES` set
   that was wrong in both directions.
2. `connection_test.py::test_multi_call_view_groups_calls_and_round_trips_weights`
   — a view spanning three connect calls in shuffled order. It asserts
   `.weight` per row, that `_call_views()` splits into one view per call in
   first-appearance order with the right IDs, and that a shuffled
   `set(weight=...)` round-trips. This is the case the three rewrites in
   3.2–3.4 share: the offset arithmetic assumes `call.row_ids` is a
   contiguous `arange`, and the grouping assumes `np.unique`'s sorted order
   is re-sorted back to first-appearance order. Existing tests only covered
   single-call views, where both are invisible. It is an equivalence pin,
   not a bug reproducer — it passes against the old implementation too.
3. `pairing_test.py::test_omitted_score_draws_exactly_like_an_explicit_uniform_score`
   — the default `score=None` and an explicit `np.ones(n)` produce identical
   `source_index`/`target_index`. This is the guard on 3.5's sentinel: the
   whole point is that the optimization is not observable.
4. `core_test.py::test_metadata_is_frozen_for_plain_arrays_and_quantities`
   — `Population.set` freezes plain arrays *and* host-backed `Quantity`
   metadata. Non-vacuous, demonstrated by running the pre-change tail
   directly:

   ```
   old path froze Quantity metadata? False
   mutated through the returned metadata -> [999.  20.]
   ```

`concat_sample_blocks` (4.1) needed no new test: it is already driven from
both sides, by `recording_test.py:62` (`braincell.RunResult.concat`) and
`core_test.py:206`/`:412` (`NetworkResult.concat`), which is exactly the
"shared by both concats" property.

Edge cases considered:

- **Empty connection store.** `_call_views` on a store with zero rows must
  return `()`, not a one-element group. Handled by the explicit
  `if ids.size == 0: return ()` guard before the grouping pass, because
  `np.split` on an empty array would otherwise yield one empty group.
- **All rows of a call removed.** `active_ids` filters first, so a call can
  disappear entirely from the grouping; the existing `ConnectionView.remove`
  tests cover it.
- **Row IDs must stay dense.** `_ConnectionStore.rows` is now an array index
  (`_row_of_id[ids]`) rather than a dict lookup, which requires IDs to be a
  gapless `0..n-1` over every row ever added. `add` builds them as one
  `np.arange` from a monotonic `_next_row_id` and `remove` only clears
  `active`, so removed rows keep their slot — pinned by the existing
  `test_connection_view_set_remove_and_nonreused_ids`. A negative index
  cannot reach it: `ConnectionView.__getitem__` selects positionally into
  `_active_ids`, so only real IDs are ever passed down.
- **Single connect call.** The common case, and the one where an off-by-one
  in the boundary computation is invisible in aggregate assertions.
- **`packed=False` removal.** No production constructor ever set it, but
  `delivery_test.py:30` exercised `population_spike` directly. That test is
  deleted with the function rather than rewritten — there is nothing left
  for it to assert.
- **`score=None` in `_score` (3.5).** The obvious spelling — pass
  `p=None` to `rng.choice` — fails this edge case: `p=None` and
  `p=uniform` draw *different* values from the same seed, so it would
  change existing connectivity. `_sample` materializes the uniform row
  instead. End-to-end check on the 1000 × 5000 `by_source` benchmark:
  `identical connectivity for the same seed: True`.
- **An empty candidate pool under a uniform score.** `_score`'s
  `positive support` error had to be preserved for the sentinel branch,
  where there is no matrix to sum. Row sums of the implied all-ones matrix
  are `shape[-1]`, so the equivalent condition is
  `shape[0] > 0 and shape[-1] == 0`. Unreachable through the public API —
  `connect()` rejects an empty source or synapse view first — so it is kept
  for branch symmetry rather than tested.
- **`make_delivery_op` (3.6).** The `brainevent` backend is unavailable in
  this environment (`resolve_event_backend` falls back to `scatter`), so
  the ops path is exercised only by `delivery_test.py`. The guard keys on
  the same `delivery_backend` string that already selects
  `group_by_delay`, so the two cannot disagree.

---

## Verification

```
$ pytest braincell/network -q
117 passed, 5 warnings, 2 subtests passed in 25.01s

$ pytest braincell/ -q
2818 passed, 15 skipped, 411 warnings, 411 subtests passed in 179.27s (0:02:59)

$ pre-commit run --files <18 changed .py files + 2 .md>
check for added large files..............................................Passed
check python ast.........................................................Passed
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
```

Baseline was 113 network tests and 2814 in the full suite. The network
package nets +4: three new regression tests (Edge cases and tests, above)
plus the `__init___test.py` rewrite, which replaces two lazy-import guards
with three that check the invariant that actually holds.
