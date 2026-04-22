# braincell Triage Review — Report

**Date:** 2026-04-22
**Scope:** read-only triage per `2026-04-22-braincell-triage-review.md`.
**Depth:** full read of `_base.py`, `_misc.py`, `mech/*`, `_cv/*`, `_compute/*`,
`_single_compartment/base.py`, `_multi_compartment/*`, `quad/__init__.py`,
`quad/protocol.py`, `quad/_registry.py`, `quad/_staggered.py`, `quad/_exp_euler.py`
(first ~250 lines); surface scan of `channel/`, `ion/`, `synapse/`, `filter/`,
`io/`, `vis/`, `morph/`.

## Executive summary

Top five risks, ordered by urgency:

1. **`MixIons.update` passes the wrong receiver to `brainstate.graph.nodes`**
   (`_base.py:1194`). Missing `self` argument; iterates a default/module-level
   graph instead of this MixIons container. Silent: channels never update.
2. **`Ion.current(include_external=True)` crashes when `nodes` empty but
   external currents are registered** (`_base.py:798-806`). `current` is `None`
   on entry to the external loop; `None + Quantity` raises `TypeError`.
3. **`SingleCompartment.compute_derivative` loses exception chain**
   (`_single_compartment/base.py:229-234`). `raise ValueError(...)` without
   `from e` drops traceback. Already fixed in
   `_multi_compartment/currents.py:52`, inconsistency is the bug.
4. **`Ion.add` accepts any object as a "channel"** (`_base.py:926`). Uses
   `object` as the expected type for `_format_elements`, bypassing hierarchy
   checks that the `Ion.__init__` path enforces with `Channel`.
5. **Arg-validation via `assert` in integrators** (`quad/_exp_euler.py:188`,
   `quad/_staggered.py:135`, and kernel contract checks in
   `quad/_staggered.py:510-525, 545-558`). `python -O` elides asserts and
   bypasses the contract. Low likelihood in production but real.

One architectural note: **CLAUDE.md paths drift from the code**. `cv/` is
`_cv/`, `compute/` is `_compute/`, `morpho/` is `morph/`, `_multi_compartment/`
appears as `multi_compartment/` in places. Documented in Appendix A.

## Findings table of contents

| Severity | IDs |
|----------|-----|
| Critical | CRIT-01, CRIT-02 |
| High     | HIGH-01, HIGH-02, HIGH-03, HIGH-04 |
| Medium   | MED-01 … MED-10 |
| Low      | LOW-01 … LOW-09 |
| Arch     | ARCH-01 … ARCH-09 |

Every finding schema: `file:line | issue | risk | fix | confidence | behavior
preserved`. Confidence **H** means "I would defend this as a bug". **L** is
always marked `speculative` in the body.

---

## Critical

### CRIT-01 `MixIons.update` passes wrong receiver to `brainstate.graph.nodes`

- **Location:** `braincell/_base.py:1193-1196`
- **Issue:** The method does
  ```python
  def update(self, V, *args, **kwargs):
      for key, node in brainstate.graph.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
          ...
  ```
  The first positional argument of `brainstate.graph.nodes` is the graph
  root. Compare `Ion.update` (`_base.py:847`), which correctly passes
  `self`: `brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1))`.
  As written, `MixIons.update` resolves against whatever default the
  function has when receiver is omitted (e.g. a module-level or empty
  graph), so *none of this MixIons's child channels are updated*.
- **Risk:** Silent wrong output. Any simulation that relies on
  `MixIons.update` (e.g. a two-ion channel group whose gating variables
  depend on both ion concentrations) will not advance its channel states.
  Integrators that drive the update through this code path will produce
  incorrect voltage traces with no exception raised.
- **Smallest fix:** Insert `self` as first argument.
  ```python
  def update(self, V, *args, **kwargs):
      for key, node in brainstate.graph.nodes(self, IonChannel, allowed_hierarchy=(1, 1)).items():
          infos = tuple([self._get_ion(root).pack_info() for root in node.root_type.__args__])
          node.update(V, *infos)
  ```
- **Confidence:** H — direct compare-against-sibling confirms.
- **Behavior preserved?:** no; intentionally changes broken behavior.

### CRIT-02 `Ion.current(include_external=True)` crashes on empty channels

- **Location:** `braincell/_base.py:781-807`
- **Issue:**
  ```python
  current = None
  if len(nodes) > 0:
      for node in nodes:
          new_current = node.current(V, ion_info)
          current = new_current if current is None else (current + new_current)
  if include_external:
      for key, node in self._external_currents.items():
          current = current + node(V, ion_info)
  ```
  When `nodes` is empty and `include_external=True`, `current` stays `None`
  and the `current + node(...)` in the external loop raises `TypeError`.
- **Risk:** Real for mixed-ion channel setups where the registrant ion has
  no native channels but external channels drive its current. `MixIons.add`
  (`_base.py:1235-1239`) explicitly registers external currents on every
  participating ion, so this is reachable. Symptom is a `TypeError` at
  simulation time, not at compile time.
- **Smallest fix:** Initialise `current` to zero of the right unit when
  entering the external loop:
  ```python
  if include_external:
      if current is None:
          current = 0.0 * (u.nA / u.cm ** 2)  # or the canonical density unit
      for key, node in self._external_currents.items():
          current = current + node(V, ion_info)
  ```
  Better: refactor to accumulate into a zero Quantity up front so the loops
  are uniform.
- **Confidence:** H.
- **Behavior preserved?:** no; fixes broken case.

---

## High

### HIGH-01 `SingleCompartment.compute_derivative` drops exception chain

- **Location:** `braincell/_single_compartment/base.py:229-234`
- **Issue:**
  ```python
  except Exception as e:
      raise ValueError(
          f"Error in computing current for ion channel '{key}': \n"
          f"{ch}\n"
          f"Error: {e}"
      )
  ```
  Missing `from e`. Compare `braincell/_multi_compartment/currents.py:52`,
  which does the same thing correctly (`from exc`). The dropped chain hides
  the original exception type and traceback.
- **Risk:** Debugging pain when a channel raises. Users see a generic
  `ValueError` whose message is an arbitrary-format render of the inner
  exception instead of the traceback pointing to the failing line in the
  channel. No data corruption.
- **Smallest fix:** Add `from e`.
- **Confidence:** H.
- **Behavior preserved?:** no (improves diagnostics).

### HIGH-02 `Ion.add` accepts any object as "channel"

- **Location:** `braincell/_base.py:924-926`
- **Issue:**
  ```python
  def add(self, **elements):
      self.check_hierarchies(type(self), **elements)
      self.channels.update(self._format_elements(object, **elements))
  ```
  `_format_elements(object, ...)` always succeeds — `isinstance(x, object)`
  is vacuous. Compare `Ion.__init__` (line 719): `self._format_elements(Channel, ...)`.
- **Risk:** User calls `ion.add(foo=NotAChannel())`, the object gets stored
  into `self.channels` without complaint. `check_hierarchies(type(self),
  **elements)` at line 925 runs *before* format, so it still enforces
  `root_type` on each element — but only for objects with a `root_type`
  attribute. Non-Channel objects that happen to satisfy that single check
  slip through.
- **Smallest fix:** Change `object` → `Channel`:
  ```python
  self.channels.update(self._format_elements(Channel, **elements))
  ```
- **Confidence:** H (compare with sibling constructor in same class).
- **Behavior preserved?:** no (tightens validation).

### HIGH-03 Integrator arg validation via `assert`

- **Location:**
  - `braincell/quad/_exp_euler.py:188-191` (`exp_euler_step`)
  - `braincell/quad/_staggered.py:135-138` (`staggered_step`)
- **Issue:** `assert isinstance(target, HHTypedNeuron)` and equivalent.
  Under `python -O` asserts are stripped; a wrong `target` would pass this
  gate and fail deeper with a less helpful error (e.g.
  `AttributeError: 'int' object has no attribute 'point_tree'`).
- **Risk:** Not high in day-to-day use (few sites run `-O`), but production
  deployments occasionally do. Docstrings even advertise the
  `AssertionError`, so callers may be `except AssertionError:`-guarding;
  switching to `TypeError` would be a breaking change for them — worth
  noting when fixing.
- **Smallest fix:** Replace with `if not isinstance(target, HHTypedNeuron):
  raise TypeError(...)`. Update docstrings to say `TypeError`.
- **Confidence:** H.
- **Behavior preserved?:** no (exception type changes); justified.

### HIGH-04 DHS kernel contract checks via `assert`

- **Location:** `braincell/quad/_staggered.py:510-525, 545-558`
- **Issue:** `_check_comp_triang` and `_check_comp_backsub` use `assert`
  statements to enforce kernel input contracts (unit-lessness, shapes,
  etc.). Under `-O` these checks evaporate; a bad caller would silently
  run a kernel with wrong-shaped inputs, producing garbage.
- **Risk:** Same category as HIGH-03 but lower blast radius because
  internal callers are stable. Still: contract checks are defensive for a
  reason.
- **Smallest fix:** Replace with explicit `if ...: raise ValueError(...)`
  blocks.
- **Confidence:** H.
- **Behavior preserved?:** no (surfaces raises under `-O`).

---

## Medium

### MED-01 `Density.__eq__` / `__hash__` may fail with array-valued `Params`

- **Location:** `braincell/mech/_density.py:181-206`, `_params.py:173-180`
- **Issue:** `Density.__eq__` calls through to `self.params == other.params`,
  which reduces to `Params.__eq__ → dict(self._items) == dict(other)`.
  When values are `numpy`/`jax` arrays, dict element-wise equality returns
  an array and dict comparison raises `ValueError: The truth value of an
  array with more than one element is ambiguous`. Likewise `__hash__`
  requires all values hashable — arrays are not.
- **Risk:** Cell-level `_cvs_cache_key` comparison
  (`_multi_compartment/cell.py:295`) and `merge_paint_rules` dedup key
  (`_cv/lower.py:219`) both rely on equality over declarations. Today all
  params are scalar brainunit quantities, so this is latent. A user who
  ever passes an array-valued parameter will get opaque failures.
- **Smallest fix:** Document the contract ("`Params` values must be hashable
  and support scalar equality") in `Params` and `Density` docstrings.
  Prefer: normalise Quantity mantissas to hashable tuples inside `Params`
  at construction time, or raise a clear `TypeError` at that point.
- **Confidence:** M (depends on whether users ever pass arrays, which the
  declarative layer documentation says they don't, but nothing in the code
  enforces it).
- **Behavior preserved?:** yes (fix is pure error-quality).

### MED-02 `SineClamp` / `FunctionClamp` / `ProbeMechanism` skip validation

- **Location:** `braincell/mech/_point.py:206-262, 325-340`
- **Issue:** These point-mechanism dataclasses have no `__post_init__`,
  unlike `CurrentClamp`. `SineClamp` will accept `duration=-1.0*u.ms`,
  `frequency=0*u.Hz`, unit-less `phase`, etc. Downstream code handles some
  of these (`_eval_sine_clamp` masks by `active` when `local_t < duration`
  fails), but negative frequency or zero duration silently produce
  surprising currents.
- **Risk:** User-facing correctness. Invalid inputs survive the declaration
  phase and fail mysteriously at simulation time or produce wrong output.
- **Smallest fix:** Add `__post_init__` to each with cheap range / unit
  checks mirroring `CurrentClamp.__post_init__`.
- **Confidence:** H for the omission; M for how often it bites users.
- **Behavior preserved?:** no (fails earlier for invalid input).

### MED-03 `_coverage_fraction` recomputes frusta per (CV × rule)

- **Location:** `braincell/_cv/lower.py:746-763`
- **Issue:** For every painting rule and every CV whose interval overlaps
  that rule, `_coverage_fraction` calls `_build_frusta(branch, prox=start,
  dist=end)` anew. Frustum build is not cheap: it walks all branch
  segments and interpolates. Large morphologies with many fine-grained
  paint regions pay `O(n_rules · n_cvs · n_segments)`.
- **Risk:** Build-time performance, not correctness. For a 5000-CV
  morphology with 20 paint rules, the per-build overhead is meaningful.
- **Smallest fix:** Cache `_build_frusta(branch, prox, dist)` on
  `(branch_id, prox, dist)` inside `_build_mech`, reusing across rules.
  Alternatively, pre-build per-CV frusta once in `_build_geo` and pass
  them through.
- **Confidence:** M (profiling needed to quantify).
- **Behavior preserved?:** yes.

### MED-04 Broad `except Exception` in channel current paths

- **Location:**
  - `braincell/_multi_compartment/currents.py:49-54`
  - `braincell/_single_compartment/base.py:229-234`
- **Issue:** `except Exception` re-raises as `ValueError`. Swallows the
  original exception type — a `ShapeMismatchError`, `FloatingPointError`,
  or `jax.core.InconclusiveDimensionOperation` all become "`ValueError:
  Error in computing current...`". Callers doing targeted `except` cannot
  distinguish.
- **Risk:** Debug pain, test brittleness. Users who catch specific
  numerical errors upstream cannot.
- **Smallest fix:** Limit the catch (e.g. `except (TypeError, ValueError,
  RuntimeError)`), or replace entirely with a top-level decorator that
  logs the failing channel name and re-raises. Ensure `from exc` is used
  (`from exc` in `_multi_compartment/currents.py:54` is correct; the
  single-compartment counterpart is missing it — see HIGH-01).
- **Confidence:** M.
- **Behavior preserved?:** partially (error quality changes; exception
  type narrows).

### MED-05 `RegionExpr` operators skip type-checking

- **Location:** `braincell/filter/region.py:57-67`
- **Issue:** `__or__`, `__and__`, `__sub__` on `RegionExpr` wrap any value
  into a `RegionSetOp` without checking that the right-hand side is also
  a `RegionExpr`. `LocsetExpr` at `braincell/filter/locset.py:59-72`
  correctly returns `NotImplemented` in the same position. Inconsistency.
- **Risk:** `my_region | 5` silently constructs a malformed
  `RegionSetOp(union, (my_region, 5))` that detonates at `evaluate` time
  with a harder-to-read trace. Also inhibits reflected-operator dispatch.
- **Smallest fix:** Mirror the `LocsetExpr` pattern: `if not
  isinstance(other, RegionExpr): return NotImplemented` in each operator.
- **Confidence:** H.
- **Behavior preserved?:** no (tighter validation).

### MED-06 `_exponential_euler` uses `A⁻¹(e^(hA) − I)f`

- **Location:** `braincell/quad/_exp_euler.py:60-90`
- **Issue:** Computes the exponential-Euler update as
  `inv(A) @ (expm(dt*A) − I) @ df`. Standard formulation, but numerically
  unstable when `A` is near-singular — which happens routinely for gating
  variables whose rate becomes very small. The φ₁ form
  `φ₁(z) = (e^z − 1)/z` would be the stable replacement when
  `abs(z)` is near zero; here we do the matrix version via
  `jnp.linalg.solve(A, expm(dt*A) − I)` which is solvent only when `A` is
  full-rank.
- **Risk:** Accuracy degradation or NaN near quiescent neurons. The
  `ind_exp_euler_step` variant (state-by-state) uses `u.math.exprel` which
  *is* the φ₁ function, so it does not share this issue; `exp_euler_step`
  and the coupled path do.
- **Smallest fix:** Replace with a matrix-function solver that uses the
  augmented-matrix trick:
  `phi1(hA) = [[I, 0], [0, I]] ← upper-right block of
  expm([[hA, I], [0, 0]])`. Then `update = phi1(hA) · hdf`. Numerically
  robust for singular `A`.
- **Confidence:** M (known result from exp-integrator literature).
- **Behavior preserved?:** no (accuracy improves); justified.

### MED-07 `_get_dhs_static_source` / `_get_dhs_static_cache` probe two runtime attr names

- **Location:** `braincell/quad/_staggered.py:372-402`
- **Issue:** `getattr(target, "_runtime", getattr(target, "_compiled_runtime",
  None))`. `_compiled_runtime` looks vestigial; every current call-site
  uses `_runtime`. Dual lookup masks rename errors.
- **Risk:** Low directly. Medium for future refactors where someone
  removes `_runtime` and the silent `_compiled_runtime` fallback hides
  the break.
- **Smallest fix:** Grep all uses. If no live owner of
  `_compiled_runtime`, drop the fallback and read `target._runtime`
  directly (raising a clear `AttributeError` instead of returning
  `None`).
- **Confidence:** M (needs grep to confirm).
- **Behavior preserved?:** yes, assuming no live caller.

### MED-08 `FunctionClamp.fn` is compared by identity

- **Location:** `braincell/mech/_point.py:236-262`, layout de-dup at
  `_compute/runtime.py:621-628`.
- **Issue:** Dataclass `__eq__` falls back to identity on lambdas. The
  runtime’s `_fn_fingerprint` tries to normalise by bytecode + closure
  cells, but two functionally identical `lambda t: 0 * u.nA` definitions
  at different source locations still hash differently if closure content
  diverges. The fingerprint fallback `("id", id(v))` inside closure cells
  for opaque objects permanently splits otherwise-identical functions.
- **Risk:** Two place rules with textually identical lambdas still
  produce two separate layouts, doubling storage and state-buffer work.
  Correctness is preserved, but it silently undermines dedup.
- **Smallest fix:** Document that users should hoist shared lambdas to
  module level, and add a warning when a layout is created from a lambda
  whose closure contains opaque cells. Longer-term: require users to
  supply a named callable for dedup.
- **Confidence:** M.
- **Behavior preserved?:** yes.

### MED-09 `build_placeholder_ions` allocated twice at `Cell.__init__`

- **Location:** `braincell/_multi_compartment/cell.py:126`
- **Issue:** `HHTypedNeuron.__init__(self, size=(1,), name=name,
  **build_placeholder_ions())` creates a throwaway set of ion containers
  that `init_state` then overwrites at line 355 (`self.ion_channels =
  self._format_elements(IonChannel, **root_nodes)`).
- **Risk:** Wasted memory and import-time work for every `Cell`. Also
  confuses stack traces — exceptions raised during placeholder
  construction look unrelated to the real simulation.
- **Smallest fix:** Pass empty `**{}` to the parent and let `init_state`
  populate `ion_channels`. Or move placeholder allocation behind a lazy
  property.
- **Confidence:** M (needs verification that `HHTypedNeuron` accepts an
  empty mapping without error).
- **Behavior preserved?:** yes (placeholder objects are never observed).

### MED-10 `assert` in kernel `_check_comp_*` paths under `-O`

Already described in HIGH-04. Grouped separately here because the fix
pattern is shared and should be batched.

---

## Low

### LOW-01 `_misc.py` indentation quirk

- **Location:** `braincell/_misc.py:35`
- **Issue:** `try:` block body is indented 7 spaces, `except` block is
  indented 4. Works (Python accepts any consistent indentation inside
  `try`), but inconsistent with surrounding code.
- **Fix:** Reindent the return line.
- **Confidence:** H (style).
- **Behavior preserved?:** yes.

### LOW-02 `normalize_param(..., None)` raises `TypeError` not `ValueError`

- **Location:** `braincell/_misc.py:123`
- **Issue:** "`{name} cannot be None`" is a value-shape problem, not a
  type problem. Speculative — current type is defensible.
- **Fix:** Optional: `ValueError` instead of `TypeError`. Or leave.
- **Confidence:** L (speculative; matter of taste).
- **Behavior preserved?:** no if changed.

### LOW-03 `braincell/__init__.py` eagerly imports neuromorpho

- **Location:** `braincell/__init__.py:52`
- **Issue:** `from .io.neuromorpho import load_neuromorpho` pulls in the
  HTTP client, cache, and Solr query machinery at package import time.
  Users who never touch NeuroMorpho still pay the import cost (and the
  transitive `requests` dependency).
- **Fix:** Move `load_neuromorpho` to a lazy accessor via
  `deprecation_getattr`-style `__getattr__`, or drop the root-level
  re-export.
- **Confidence:** M.
- **Behavior preserved?:** yes (import lazy-loads).

### LOW-04 `mix_ions` arity is inconsistent with `MixIons` arity

- **Location:** `braincell/_base.py:991, 1302`
- **Issue:** `MixIons.__init__` asserts `len(ions) >= 2`; `mix_ions`
  asserts `len(ions) > 0`. Single-ion call to `mix_ions` reaches MixIons
  and fails there instead.
- **Fix:** Harmonise at `mix_ions`: `assert len(ions) >= 2`.
- **Confidence:** H.
- **Behavior preserved?:** no (fails earlier with a clearer message).

### LOW-05 `CLAMP_KINDS` defined but duplicated literally

- **Location:** `braincell/_compute/runtime.py:104, 531`
- **Issue:** Line 104 defines `CLAMP_KINDS = frozenset({"CurrentClamp",
  "SineClamp", "FunctionClamp"})`. Line 531 re-inlines the same literal
  set.
- **Fix:** Use `CLAMP_KINDS` at line 531.
- **Confidence:** H (style).

### LOW-06 Magic constant `1e5` in d-lambda formula

- **Location:** `braincell/_cv/policy.py:282`
- **Issue:** `lambda_f_um = 1.0e5 * np.sqrt(diam_um / (4.0 * np.pi *
  frequency_hz * ra_ohm_cm * cm_uF_per_cm2))`. The `1e5` is the
  unit-conversion factor that yields μm; worth a comment referencing the
  NEURON `d_lambda` derivation.
- **Fix:** Comment plus a named constant (`_CM_TO_UM_FACTOR`).
- **Confidence:** H (readability).

### LOW-07 `Container.__getattr__` complexity

- **Location:** `braincell/_misc.py:300-310`
- **Issue:** Custom `__getattr__` combines container-name resolution,
  child lookup, and fallback to `super().__getattribute__`. Hard to
  reason about; any call during `__init__` before `_container_name` is
  set raises during attribute resolution.
- **Fix:** Guard `_container_name` absence more explicitly; add a
  docstring with the lookup order.
- **Confidence:** L (speculative — depends on how often subclasses rely
  on unusual access patterns).

### LOW-08 Bool treated as numeric zero in `_is_python_zero`

- **Location:** `braincell/_multi_compartment/currents.py:143`,
  `_multi_compartment/bridge.py:89-91`
- **Issue:** `isinstance(value, (int, float)) and value == 0`. `bool` is
  a subclass of `int`; `_is_python_zero(False)` returns `True`. Passing
  `False` as `I_ext` silently yields zero current.
- **Fix:** `isinstance(value, (int, float)) and not isinstance(value,
  bool) and value == 0`.
- **Confidence:** H (bug), L (severity).

### LOW-09 `CompositeByTypePolicy` computes full bounds per inner policy

- **Location:** `braincell/_cv/policy.py:221-242`
- **Issue:** For each policy, the composite calls
  `policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)` once (cached
  per policy) and then picks a single branch's bounds from the result.
  For a composite with many policies, each must compute full
  per-branch bounds even if only one branch uses it.
- **Fix:** Extend inner policies with a per-branch entry point that
  returns bounds for one branch only. Or accept the tradeoff and add a
  comment; the cache already makes this `O(n_policies × n_branches)`
  rather than `O(n_rules × n_branches)`.
- **Confidence:** L (optimisation, not a bug).

---

## Arch

### ARCH-01 CLAUDE.md ↔ code path drift

Reported in Appendix A.

### ARCH-02 `_compute/runtime.py` is 1676 lines and holds ≥4 responsibilities

- **Location:** `braincell/_compute/runtime.py`
- **Issue:** The single file defines `MechanismLayout`,
  `CellRuntimeState`, `ClampActiveTable`, clamp evaluation, ion runtime
  construction, channel-ion binding resolution, state-buffer allocation,
  and runtime-node instantiation.
- **Fix:** Split into:
  - `_compute/layouts.py` — `MechanismLayout`, `ClampActiveTable`,
    `build_clamp_active_table`, layout grouping.
  - `_compute/state.py` — `CellRuntimeState` and its methods.
  - `_compute/ions.py` — ion-instance resolution + `_sync_runtime_ion`.
  - `_compute/bindings.py` — channel-ion binding resolution
    (`_resolve_channel_runtime_bindings`, family/class helpers).
- **Confidence:** M (architectural; needs coordinated test changes).

### ARCH-03 `_base.py` packs six public classes in 1337 lines

- **Location:** `braincell/_base.py`
- **Issue:** `HHTypedNeuron`, `IonChannel`, `IonInfo`, `Ion`, `MixIons`,
  `Channel`, `Synapse` all in one file.
- **Fix:** Split by class family — `_base_ion.py` (`Ion`, `MixIons`,
  `IonInfo`), `_base_channel.py` (`Channel`, `IonChannel`), root
  `_base.py` keeps `HHTypedNeuron` plus re-exports.
- **Confidence:** M.

### ARCH-04 `get_spike` duplicated across SC and MC

- **Location:** `braincell/_single_compartment/base.py:285-309`,
  `_multi_compartment/cell.py:609-615`
- **Issue:** Identical surrogate-gradient spike logic in both. Includes
  the same magic `20.0 * u.mV` denominator and `_cast_like` helper.
- **Fix:** Promote `_cast_like` and `get_spike` to `_base.py` as methods
  on `HHTypedNeuron`, parameterised by the scale.
- **Confidence:** H.

### ARCH-05 Public CV API lives in `_cv/base.py` despite the underscore-prefix convention

- **Location:** `braincell/_cv/`
- **Issue:** The directory is `_cv` (private) but the public re-exports
  live in `_cv/base.py`, `_cv/policy.py`, `_cv/lower.py` without
  underscore prefixes. User-visible `CV`, `CVPolicy`, `build_cvs` live
  in `_cv/base.py`. The mix of "package-private" (`_cv`) and "module-
  public" names is confusing.
- **Fix:** Either promote to `cv/` (public package) or rename internal
  modules with `_` prefix (`_cv/_lower.py`, etc.) keeping `base.py`,
  `policy.py` as the public surface of a private package. CLAUDE.md
  already documents this as `cv/` so the rename-to-public route aligns
  docs with code.
- **Confidence:** M.

### ARCH-06 `_is_traced_value` duplicated

- **Location:** `braincell/_multi_compartment/cell.py:83-86`,
  `braincell/quad/_staggered.py:47-50`
- **Issue:** Same helper defined in two places.
- **Fix:** Promote to `braincell/_misc.py`.
- **Confidence:** H.

### ARCH-07 `object.__setattr__(runtime, ...)` pattern is inconsistent

- **Location:** multiple in `_multi_compartment/cell.py`,
  `quad/_staggered.py`
- **Issue:** `CellRuntimeState` is a regular `@dataclass`, not frozen.
  `object.__setattr__` is only needed on frozen dataclasses. Using it on
  a mutable dataclass signals frozenness falsely to readers.
- **Fix:** Use plain `setattr` (or direct attribute assignment). If the
  intent is to enforce "this is cache-only", make `CellRuntimeState`
  `frozen=True` and explicitly use `object.__setattr__` for caches.
- **Confidence:** M.

### ARCH-08 `quad/__init__.py` hard-imports every integrator module

- **Location:** `braincell/quad/__init__.py:20-72`
- **Issue:** Side-effect imports for registration. Works, but means even
  users who never touch `diffrax_*` pay the import cost — `diffrax`
  itself is lazy per its module docstring, but the `_diffrax.py` module
  is eagerly pulled in here. The pattern is deliberate (registration
  must run) but worth documenting more explicitly.
- **Fix:** Add a docstring comment near the top of `__init__.py`
  explaining "side-effect imports populate the integrator registry; do
  not remove."
- **Confidence:** H (docs only).

### ARCH-09 `IndependentIntegration.__init__` drops kwargs silently

- **Location:** `braincell/quad/protocol.py:288-290`
- **Issue:** Accepts `**kwargs` and discards them without forwarding to
  any cooperative-mixin `super().__init__`. In an MRO chain where
  another mixin expects kwargs, they quietly vanish.
- **Fix:** `super().__init__(**kwargs)` inside the body, or remove the
  `**kwargs` parameter.
- **Confidence:** M (depends on real MRO usage).

---

## Appendix A — CLAUDE.md ↔ code drift

| CLAUDE.md path | Actual path |
|----------------|-------------|
| `braincell/cv/` | `braincell/_cv/` |
| `braincell/cv/_cv.py` | `braincell/_cv/base.py` |
| `braincell/cv/_geo.py` | (folded into `braincell/_cv/lower.py`) |
| `braincell/cv/_mech.py` | (folded into `braincell/_cv/lower.py`) |
| `braincell/cv/_policy.py` | `braincell/_cv/policy.py` |
| `braincell/compute/` | `braincell/_compute/` |
| `braincell/compute/_point_tree.py` | `braincell/_compute/topology.py` |
| `braincell/compute/_runtime.py` | `braincell/_compute/runtime.py` |
| `braincell/compute/_assignment_table.py` | `braincell/_compute/table.py` |
| `braincell/morpho/` | `braincell/morph/` |
| `braincell/morpho/branch.py` | `braincell/morph/branch.py` |
| `braincell/morpho/morpho.py` | `braincell/morph/morphology.py` |
| `braincell/_multi_compartment/cell.py` → re-export `Cell, RunnableCell` | only `Cell` exported today; `RunnableCell` / `RunResult` split differs |
| `braincell/_multi_compartment/build.py` | not present (build inlined into `cell.py`) |
| `braincell/_multi_compartment/clamp_table.py` | lives in `_compute/runtime.py` (`ClampActiveTable`) |
| `braincell/quad/_diffrax.py` | present, but `__init__.py` eager-imports it (see ARCH-08) |

None of these are *bugs* — code is authoritative. But documentation out
of sync with code corrodes trust in the docs, and several of these
drifts pointed me at the wrong files during review. Suggest either
updating CLAUDE.md wholesale or running a pre-commit check that compares
the layout section against `ls braincell/`.

## Appendix B — modules not deeply read

The triage spec tiered these as surface-scan; they were not read line
by line. Noting them here so the user knows where additional review
would uncover findings not in this report.

- `braincell/channel/*` — 9 files, ~250 KB. Surface-confirmed that all
  modules register via `@register_channel`; did not verify per-channel
  current or derivative math.
- `braincell/ion/*` — 3 species + `_template.py`. Surface-confirmed
  that `DynamicNernstIon`, `InitNernstIon`, `FixedIon` match the family
  assumptions in `_compute/runtime.py`.
- `braincell/synapse/markov.py` — surface-scanned.
- `braincell/io/*` — neuromorpho / swc / asc / neuroml2 / checkpoint all
  surface-scanned. Notable: `checkpoint.py` is 19 KB and lives at
  `io/`, not inside a subpackage; consider a `checkpoint/` subpackage.
- `braincell/vis/*` — not read at all beyond `__init__.py`-level
  inventory.
- `braincell/filter/helper.py` — not read (24 KB). Owns interval
  arithmetic that I assumed correct when inspecting `locset.py` /
  `region.py`; spot-check recommended.
- `braincell/morph/morphology.py` — not read (70 KB). Used via imports
  in `_cv/lower.py` and `_compute/topology.py`; observed that expected
  methods (`branches`, `edges`, `root`, `branch(index=…)`) exist.
- `braincell/_compute/runtime.py` — read lines 1-600 and 1000-1680;
  middle 400 lines (600-1000) read only for symbols, not line-by-line.
- `braincell/quad/_runge_kutta.py`, `_implicit.py`, `_backward_euler.py`,
  `_diffrax.py`, `_util.py` — not read; same validation concerns as
  the integrators that were read likely apply.
- `braincell/_multi_compartment/probes.py`, `run.py` — not read.
- `braincell/_cv/lower.py` — read in full.
- All `*_test.py` files — only consulted when a finding pointed to one.
