# Package-wide quality cleanup of `braincell`

Reuse, simplification, efficiency, and altitude cleanup across all 154
source modules of `braincell/` (~77,800 lines). Breaking API changes are
in scope and were explicitly authorised for this pass.

## Scope and method

Four independent reviews swept the whole package, one per angle (reuse,
simplification, efficiency, altitude). Findings were deduplicated — the
same clusters surfaced in two, three, or four of the reviews
independently, which is what promoted them — and the ones with a
concrete, verifiable fix were applied.

Implementation was partitioned across disjoint package boundaries so
edits could not collide: `quad/`; `channel/` + `ion/`; `vis/`; `io/` +
`morph/` + `filter/`; and the cross-cutting core (`braincell/` root,
`_compute/`, `_discretization/`, `_multi_compartment/`, `mech/`,
`network/`).

Three invariants governed every edit:

1. **The test suite stays green.** Baseline before any change:
   2,575 passed, 15 skipped, 309 subtests, 0 failures (353 s).
2. **Every performance claim carries a measurement.** No fix landed on
   the strength of "this looks faster" — each has a before/after number
   and, where output could drift, an equivalence check.
3. **Where two code paths disagreed, the disagreement was preserved,
   not silently unified.** Several duplicated helpers had drifted; the
   shared helper takes the differing input from its caller rather than
   imposing one answer. Divergences worth a decision are listed under
   *Deliberately not changed*.

## Breaking changes

### Six registered integrators deleted

`splitting`, `cn_rk4`, `cn_exp_euler`, `implicit_rk4`,
`implicit_exp_euler`, `exp_exp_euler`.

They could not be called successfully by any code path in the package.
Verified empirically against a real `Cell` — each failed, and for a
*different* reason, showing the module had rotted across several API
generations without a test exercising it:

- `vmap2()` no longer accepts the `in_states` keyword (stale
  `brainstate` API) — 3 of the 6.
- `Cell.pre_integral()` takes 1 positional argument, not 3 — 2 of the 6.
- `UnexpectedTracerError` from a side effect under trace — 1 of the 6.

Underneath them, the shared matrix helpers `construct_A`,
`construct_lhs`, `construct_lhs_sparse`, `construct_lu` and
`construct_lu_sparse` call `target.conductance_matrix()` and read
`target.Gl`. **Neither is defined on any class anywhere in the
package** (repo-wide grep), so those helpers could never have run.
Three of the five had no caller at all outside a commented-out block.

`implicit_euler_step` was **kept**: it works on a plain `DiffEqModule`
and is genuinely exercised by `_implicit_test.py`. Only the
multi-compartment-only steps were removed.

The registry gained a calling-convention field so this class of rot
cannot recur silently — a registered integrator whose signature does
not match the convention `Cell` actually invokes is now detectable
rather than surfacing as a `TypeError` at first use.

`_implicit_test.py`'s module docstring claimed the deleted steps were
"exercised by their dedicated cell tests". No such tests existed. That
claim was the reason the breakage survived, and it is gone; the file
now pins the six names' *absence* from the registry instead.

The registry drops from 25 canonical-plus-alias names to 19. Every
place that listed them was updated: `docs/apis/integration.rst`,
`docs/concepts/integration.ipynb`, `docs/integration/solvers.ipynb`,
`docs/integration/overview.ipynb`, `examples/multi_compartment/quad.ipynb`,
`docs/design/interface-map.md`, and `docs/design/TODO.md`. The notebooks are
published with `nb_execution_mode = "off"`, so their *stored outputs*
are what readers see — the registry-listing cells were re-executed
against this branch rather than left showing solvers that no longer
exist.

### `braincell/_base.py` deleted

The module had become a pure re-export shim. Its three class families
now live in modules named for what they hold, and `braincell/__init__.py`
re-exports all of them, so the supported public path
(`from braincell import Channel, Ion, HHTypedNeuron, …`) is unchanged.
Only the unsupported private path `from braincell._base import …` breaks.

| Symbol | New home |
| --- | --- |
| `Channel`, `IonChannel`, `IonInfo`, `Synapse` | `braincell._base_channel` |
| `Ion`, `MixIons`, `mix_ions` | `braincell._base_ion` |
| `HHTypedNeuron` | `braincell._base_neuron` |

This is the altitude fix for a cycle, not a file move for its own sake.
`_base` and `_base_ion` each carried a *bottom-of-file* import of the
other, and the arrangement worked only because of the exact order of
statements across two files:

```python
# braincell/_base_ion.py, before — at the very bottom of the module
# Late-bound root_type assignment: HHTypedNeuron lives in braincell._base
# and we defer its import until both Ion and MixIons are defined.
from ._base import HHTypedNeuron  # noqa: E402
```

With `HHTypedNeuron` in its own leaf module, `_base_ion` names it with
an ordinary top-of-file import and the cycle is structurally impossible
rather than merely avoided. `_base_neuron_test.py` pins that by parsing
`_base_neuron.py`'s import graph and asserting no edge back to
`_base_ion`.

`_base_test.py` was distributed to the modules that own what it tested:
the `Ion` / `MixIons` / `mix_ions` regressions to `_base_ion_test.py`,
the `HHTypedNeuron.get_spike` tests to a new `_base_neuron_test.py`.
The two "the re-export still resolves" tests were replaced by tests
that the *public* `braincell` namespace exports the same objects — the
back-compat path they guarded no longer exists, but the public one is
worth pinning.

## What was fixed

Grouped by review angle. Each entry names the cost of leaving it.

### Dead code

Symbols verified unreferenced by repo-wide grep across `braincell/`,
`examples/` and `docs/` before deletion. The largest items:

- `braincell/_misc.py`: `deprecation_getattr` (35 lines) and
  `ModuleNotFound` (41 lines) — a module-deprecation shim for a
  migration that had already completed; `Container.add_elem`, a
  four-line alias for `add` with one stale doc reference.
- `braincell/_compute/layouts.py`: the `NetStim` isinstance branch and
  the two functions it guarded, `_evaluate_netstim_layout` (37 lines)
  and `_source_cv_ids_for_point` (8 lines). `NetStim.__mro__` is
  `(NetStim, EventSource, ABC, object)` — it is neither a `Point` nor a
  `Mechanism`, so the branch could never be taken. Deleting it removed
  the only `_compute → network` import edge, which is the real win:
  event sources are a network-layer concept and have no business in the
  layout compiler.
- `braincell/vis/`: `scene2d.build_projected_scene_2d`, four
  dendrogram-only helpers in `_legacy.py`, `export._PNG_SUFFIXES`.
- `braincell/_multi_compartment/cell.py`: four unreachable methods.
- `braincell/channel/potassium.py`: `_linoid_stable` and its test.

### Reuse — one home per concept

Helpers that existed in two to five verbatim copies were collapsed to a
single definition, with the duplicates importing it. The recurring
hazard is that a fix applied to one copy silently misses the others;
several of these had already drifted apart before this pass.

- `braincell/_misc.py` became the shared-helper home for the core:
  `concat_values`, `same_time_quantity`, `validate_time_quantity`, and
  the four profiler-name helpers (`profiler_safe_name`,
  `profiler_scope_name`, `profiler_call_name`, `profile_barrier_current`),
  each of which had two to four copies across `_base_ion.py`,
  `_multi_compartment/`, and `network/`.
- `network/`: `_cell_label`, `_concat_values`, `_same_time`, and
  `zero_ring_buffer` were deduplicated across `connection.py`,
  `core.py`, `delivery.py`, and `lowering.py`.
- `channel/_base.py` gained `q10_factor(q10, temp, temp_ref)`; 15 call
  sites across `sodium.py`, `calcium.py`, `potassium.py`,
  `potassium_calcium.py`, and `hyperpolarization_activated.py` now
  route through it, and the module-local `_q10_factor` copy is gone.
- `ion/_base.py` gained a `_RadialShellGeometry` mixin, replacing 47
  duplicated method definitions across the five `Cdp*` classes in
  `ion/calcium.py` (−252 lines in that file alone).
- 20 byte-identical channel classes became docstring-only subclasses.
  Each collapse was verified with both AST-normalised and raw-text
  diffs; every docstring, citation, and `@register_channel` key is
  preserved verbatim.
- `vis/_arclength.py` is new: `cumulative_arclength_um`, `ordered_span`,
  `segment_index_at`, `interpolate_at`, and a dimension-agnostic
  `ArcPolyline` replace five parallel implementations across
  `scene3d.py`, `scene2d.py`, and `layout/_geometry.py`.
- `vis/layout/_common.walk_layout_top_down` now drives `_radial`,
  `_legacy`, `_fan`, and `_balloon`; each family keeps only a local
  `_place` closure.
- `io/_geometry.py` is new: `MIN_SYNTHETIC_LENGTH_UM`,
  `synthetic_soma_geometry`, `should_copy_attach_point`, and the soma
  contour math lifted out of `asc/reader.py`. The contour helpers took
  `self` and never used it; they now operate on `(N, 3)` float arrays,
  so `_geometry_test.py` can exercise the eigenvector-based
  principal-axis conversion on circles and ellipses directly instead of
  only through a parsed file.
- `Morphology.naming_state()` / `restore_naming_state()` replace two
  sites where `io/checkpoint.py` reached into `_type_name_counters`.
- Both copies of `_x_over_one_minus_exp_neg_stable` were replaced with
  `u.math.exprel`. The two disagreed with each other by up to 4.9e-2
  relative at float32 — but measured against an exact reference the
  *old helper* was the inaccurate side (407,489 eps versus `exprel`'s
  2.44 eps), so this is an accuracy fix as well as a dedup.

### Simplification

- `mech/_density.py` and `mech/_params.py` carried a
  `# imported lazily to break a cycle` comment on a function-local
  `from ._registry import get_registry`. There is no cycle:
  `_registry.py` imports only `difflib`, `dataclasses`, and `typing`.
  The imports are now at module level and the false comment is gone.
- `Markov.reset_to_steady_state: ClassVar[bool] = False` replaces ten
  delegating `reset_state` overrides.
- `KineticIon.Ci_initializer` is now a property over
  `species_initializers["Ci"]` rather than a second copy of it; the two
  were written independently by every subclass constructor and could
  drift apart.
- `vis/`: the two disagreeing 2D-layout name sets are unified on a
  single `LAYOUT_2D_FAMILIES` registry in `config.py`.

### Efficiency — measured, untraced host code

The per-step device path was already in good shape: `Cell.run` and
`Network.run` both use `brainstate.transform.jit` + `for_loop` with a
compile cache, and there are no bare Python timestep loops anywhere, so
AGENTS.md rule 11 is respected throughout. The waste was concentrated
in **untraced** host code — model construction, morphology and filter
queries, visualisation layout, and I/O — where several accessors were
quadratic.

- `vis/layout/_cache._metric_key` did an O(n²) `.index` lookup on every
  cache hit: **0.228 s → 0.028 s** at 1,600 branches.
- `vis/movie.py` hoisted a `ValueLayout` out of the per-frame callback:
  per-frame value resolve **0.193 s → 0.001 s**, so a 500-frame movie
  sheds roughly 96 s.
- `network/engine.py` built its probe-name map by calling
  `sample_probes()` — running real gathers outside `jit` only to
  discard everything but the names. It now reads the same ordering off
  the layouts via `probe_names()`.
- `morph/morphology.py` memoises `_ordered_node_ids_by`,
  `_branch_index_map`, `branch_by_order`, and `has_full_point_geometry`,
  invalidated from `_register_node` — the only place `_nodes` grows
  (repo-wide grep; there is no detach or rename path). Whole-morphology
  `.metric`: **1.67 s → 0.62 s at 1,600 branches** (2.70×), 1.54× at
  400. All nine metric fields identical at every size. Removing the
  invalidation call fails five of the new cache tests, which is the
  negative control.
- `filter/helper._resolve_branch_property` took a branch index and
  re-derived the branch; it now takes the `MorphoBranch` and falls
  through to `getattr` against three new properties (`branch_id`,
  `branch_order`, `n_tapers`). Property filters:
  **`branch_order` 0.195 s → 0.007 s (29.7×)**, `parent_id` 32.2×,
  `type` 9.5×, `n_children` 5.9×. Selected intervals identical for all
  seven properties.
- `morph/_spatial.build()` walked `morpho.edges` twice and rebuilt an
  invariant base array per query; it now walks once and precomputes
  `root_bases_um` / `soma_bases_um`: **0.116 s → 0.022 s (5.38×)** for
  an 800-branch build, distances identical to 6 dp.
- `io/swc/rules.rule_duplicate_xyzr_parent_child` was 21% of
  `SwcReader.read` on a 38k-row reconstruction: a restart-from-scratch
  scan became a single pass over a min-heap of pending candidates with
  lazy validation, reproducing the old "lowest surviving row" order.
  A 27-entry digest over every SWC fixture plus `CA1.swc` (node ids,
  parents, coordinates, radii, types) is identical.
- `filter/cache.SelectionCache` was allocated and never populated. It
  now memoises composite operands at all six sites in `region.py` /
  `locset.py`, invalidating on a different morphology object or a
  bumped `_revision`: 56.3 ms → 30.2 ms (1.86×) for an operand repeated
  three times over 1,600 branches, identical 1,089-interval mask.
- `io/neuromorpho`: a warm cache went from **2 HTTP requests to 0**
  (`load_neuromorpho` returned only after `download()` had already
  issued `get_neuron()` + `get_measurement()`), and
  `fetch --load` from 6 to 3 (it re-ran the whole download path instead
  of parsing the file it had just written). The old
  `test_respects_existing_cache` pinned the wasteful behaviour and was
  replaced.

One efficiency finding was **reverted**: a memo on `mechanism_signature`
was reported as a 4,900× win, but an A/B benchmark measured 0.726 s
versus 0.719 s — noise. Per invariant 2 it did not land.

### Altitude — behaviour moved to the layer that owns it

- The `_base` split above.
- `network/lowering.py` validated `dt` and `delay` through one helper
  that branched on `name == "dt"`. That string test encodes two real
  rules — a delay may legitimately be a per-contact vector, and zero
  means immediate delivery — so it was replaced by explicit
  `require_scalar` / `require_positive` parameters on the shared
  `validate_time_quantity`, preserving behaviour exactly while making
  the intent readable. `lowering_test.py` gained four tests pinning
  both halves of the split and the per-layer message prefix.
- `network/engine.py`'s time validation borrowed `Cell.run`'s error
  wording via an imported wrapper; it now reports
  `Network.run(...)`.
- `_compute/layouts.py` is a leaf again (see *Dead code*).

### Lint debt

`pyproject.toml` silenced seven rules with counts recorded as debt.
Five are now zero and their entries are deleted:

| Rule | Before | After |
| --- | --- | --- |
| `F401` unused import | 42 | 0 outside `__init__.py` |
| `F403` / `F405` star import | 12 star imports | 0 |
| `F811` redefinition | 2 | 0 |
| `F841` unused local | 7 | 0 |
| `E402` import not at top | 28 | 0 |
| `E731` lambda assignment | 144 | 145 (still silenced) |
| `E741` ambiguous name `I` | 8 | 2 (still silenced) |

`F401` is now silenced only by `lint.per-file-ignores` for
`**/__init__.py`, where every import *is* the package's public surface.
Anywhere else an unused import is an error again.

## Verification

- Full suite, before: 2,575 passed / 15 skipped / 309 subtests / 0 failed
  in 353.58 s.
- Full suite, after: **2,723 passed / 15 skipped / 334 subtests / 0
  failed in 219.32 s**. 148 more tests and 25 more subtests, all green,
  and the suite itself runs 38% faster — most of that is the morphology
  and filter caching below, which the tests exercise heavily.
- `ruff check braincell/` and `ruff format --check braincell/` clean
  under the tightened configuration (277 files formatted).
- Performance fixes carry before/after measurements (above).
- Output-preserving fixes carry equivalence evidence: a 297-entry
  snapshot across 5 morphologies × 2 geometry modes × 4 layout families
  × 2 type-split settings confirms the `vis` refactor is **bit-identical**
  (`np.array_equal` on shape, dtype, and values for every entry).

## Deliberately not changed

Recorded so the next reader does not re-litigate them.

- **`E741` ambiguous name `I`.** `pyproject.toml` documents that `I`
  means *current* here and is domain-meaningful; blanket-renaming would
  hurt readability. Left silenced.
- **Per-channel gating equations.** Kept explicit and written out, per
  the standing convention. Only mechanical scaffolding was collapsed.
  The `_kdc` / `_kdm` / `_ss_pv_*` parvalbumin-equilibrium helpers are
  identical across three `Cdp` classes but are biophysics, not
  scaffolding, so they stay explicit.
- **`dhs_voltage_step`.** Cannot be selected via `solver="dhs_voltage"`
  because `Cell` calls `self.solver(target)` while the function is
  `(target, t, dt)` — but it *works* when called directly, so it is a
  call-convention mismatch, not dead code. Left in place and recorded
  by the new registry field.
- **`_round_half_up_steps`.** Three copies existed: one using `u.math`
  in `network/event.py` and two identical NumPy ones. Only the NumPy
  pair was collapsed. The `u.math` version pushes float64 step ratios
  through `u.math.asarray`, truncating them to float32, and therefore
  returns a *different* answer for ratios within a few ulp of a half
  tie. `event.py` now exports both, with a Notes section stating why
  they are not interchangeable.
- **`vis` weighted child allocators.** The three look alike but differ
  in fallback policy (clamp-to-zero vs equal-width vs `linspace`
  centres), weight defaults, gap comparison (`<` vs `<=`), and float
  association (`available * (w/t)` vs `available * w / t`, which are
  not bit-equal). Only `_child_weight` was extracted; a NOTE at
  `vis/layout/_fan.py` documents the divergence.
- **`_compute/ions.py::_restore_shaped_species_initializers`.**
  Reported as dead. It is not: neutralising it fails
  `_compute/ions_test.py::RuntimeIonTest::test_constant_quantity_ci_initializer_stays_quantity_with_population_shape`,
  because it still unwraps `braintools.init.Constant(shaped Quantity)`
  back to a raw `Quantity` for the five `Cdp*` classes. Only its
  now-redundant second write was removed.
- **`_on_param_updated`.** Reported as a base definition that "is never
  consulted". `getattr(node, "_on_param_updated", None)` does find an
  inherited method, so the base definition is load-bearing. Skipped.
- **`MorphoBranch.parent_id` vs the filter's `parent_id`.** They mean
  different things: the branch property is the parent's *node id* or
  `None` (pinned by `morphology_test.py` and relied on by `edges`,
  `_root_attach_distances_um`, `_path_node_ids`,
  `_max_path_distance_um`), while the filter wants the parent *branch
  index* with `-1` at the root. The filter keeps that one special case,
  with a comment.
- **`should_copy_attach_point(keep_radius_jump=...)`.** SWC passes
  `True` — same-xyz with a different radius still copies, keeping the
  boundary radius jump as a zero-length first segment, which is the
  invariant in `docs/design/io-swc-reader-invariants.md`. ASC passes
  `False` — coincident xyz suppresses the copy unconditionally, matching
  `read_nlcda3.hoc`. The shared helper takes the flag from its caller
  rather than picking a winner; whether ASC should adopt the SWC rule is
  a NEURON-parity question, not a refactoring one.
- **`SamplingContext` laziness.** Making its fields lazy would have paid
  for the residual `_context` cost, but the class is in
  `braincell.filter.__all__` and its field table is documented in
  `docs/design/network/api.md`. Left eager.
- **Unit-carrying filter bounds are never memoised.**
  `brainunit.Quantity` is unhashable, so e.g.
  `branch_range("length", ...)` falls through `SelectionCache` rather
  than being cached. That is inherent to the type, not fixable by
  normalising `bounds` to a tuple, and the fall-through is tested.

## One pre-existing bug fixed in passing

`io/neuromorpho/cli.py` read `b.n_points`, which does not exist on
`Branch` — `braincell-neuromorpho fetch --load` raised `AttributeError`
on every invocation. It was untested because the existing CLI tests
mocked `NeuroMorphoClient` wholesale. Now `b.n_segments + 1`, with a
test that drives the real parse path.
