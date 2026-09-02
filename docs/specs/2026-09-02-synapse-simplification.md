# `braincell.synapse` simplification

Iteration 7 of the module-by-module simplification sweep. Target:
`braincell/synapse/` — 184 production lines across two files, plus its
53-line test module.

This is the smallest package in the sweep and the one whose *published*
description is furthest from its code. The package documents three classes
that raise `NotImplementedError` and omits both classes that work; its only
model file is named after a modelling formalism it does not use; it declares
a schema field nothing reads; and its one cross-parameter validator cannot
run inside a JAX trace.

## Baseline

```
$ pytest braincell/synapse braincell/mech -q
182 passed
```

Production lines (excluding `*_test.py`): 184 (99 non-blank, non-comment).
Test lines: 53 (40 code).

## What is wrong

### 1. `markov.py` holds no Markov model

`git show 1568ef3:braincell/synapse/markov.py` (2025-06-02) shows the name was
once accurate: the file held `AMPA`/`GABAa`/`NMDA` as two-state kinetic
schemes with `alpha`/`beta` rate constants. Those models are gone. What
remains is `ExpSyn` (single-exponential conductance decay) and `Exp2Syn`
(difference of two exponentials) — neither has a transition matrix, a state
graph, or a `Transition`. The real `Markov` template lives in
`braincell/channel/_base.py` and is unrelated.

The stale name propagated outward. `docs/apis/braincell.synapse.rst` files
the package's API reference under a heading `Markov Models`, and the
`autosummary` under that heading lists `AMPA`, `GABAa`, `NMDA` — the exact
inverse of the working surface. `ExpSyn` and `Exp2Syn` appear in no `.rst` in
the repository, despite being used throughout `docs/design/network/api.md`
and `docs/tutorials/single_cell_frontend.ipynb`.

### 2. `AMPA` / `GABAa` / `NMDA` are three copies of a raising stub with no caller

```python
class AMPA(Synapse):
    """Unavailable legacy receptor model pending an event-model redesign."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_UNAVAILABLE_RECEPTOR_MESSAGE.format(model="AMPA"))
```

`GABAa` and `NMDA` are byte-identical apart from the `model=` argument, and
`_UNAVAILABLE_RECEPTOR_MESSAGE` is defined *below* all three of its uses.

Nothing in the repository constructs them. The only references are the
definitions, the two `__all__` blocks, the docs listings above, and
`markov_test.py:46-49`, which asserts that they raise. The `AMPA` hits under
`examples/single_compartment/` are `brainpy.state.AMPA` and a locally defined
`class GABAa(brainpy.state.Synapse)` — a different package.

The same policy is enforced a second time, one layer up, with wording that
has already drifted:

| site | message tail |
|---|---|
| `braincell/synapse/markov.py:153` | `...Use ExpSyn or Exp2Syn for the current event runtime.` |
| `braincell/mech/_point.py:358-359` | `...Use ExpSyn or Exp2Syn for now.` |

`mech/_point.py` hard-codes the set `{"AMPA", "GABAa", "NMDA"}` and rejects it
eagerly in `Synapse.__init__`. That blocklist defends against keys that could
not resolve anyway — `_registry_test.py:318` already pins
`_REGISTRY.contains("synapse", "AMPA") is False`. Meanwhile
`mech.Synapse("ExpSynn")` **constructs successfully** and only fails much
later, at `_multi_compartment/synapses.py:135`. So name validation is
currently eager for three hard-coded strings and deferred for every other
wrong name, which is backwards: the deferred path produces the *better*
error, `_missing_mechanism_message` with `difflib` suggestions.

### 3. `Exp2Syn` cannot be constructed inside a JAX trace

```python
    @classmethod
    def validate_parameter_values(cls, parameters) -> None:
        if u.math.any(parameters["tau1"] >= parameters["tau2"]):
            raise ValueError("Exp2Syn requires tau1 < tau2.")
```

This is the `is_disabled` bug from iteration 6 in a second location. Under an
open trace every jax operation returns a tracer *even when all its inputs are
concrete*, so `bool()` on the reduction raises. Reproduced:

```
$ python -c "... jax.make_jaxpr(lambda: (Exp2Syn(4, tau1=0.5*u.ms, tau2=5.0*u.ms), ...))()"
ExpSyn under trace: OK
Exp2Syn under trace: TracerBoolConversionError Attempted boolean conversion of traced array with shape bool[].
```

`ExpSyn` survives because its `positive` validator already uses numpy
(`_synapse_schema.py:61`: `np.any(np.asarray(u.get_mantissa(value)) <= 0.0)`).
This `u.math.any` is the only one in a validator anywhere in `braincell/`;
every one of its ~35 siblings uses `np.any`.

No shipped code path constructs a synapse under a trace today, so the bug is
latent — it blocks any jitted setup function, and it is inconsistent with the
validator sitting three lines above it. It is also 15-18x slower: on a
100 000-element parameter column the `u.math` form costs 0.198 ms against
0.013 ms for the numpy form, and it sits on `_build_parameter_columns`, whose
column width is the number of placed synapses.

### 4. `derived` / `DerivedSpec` is declared, re-exported, tested — and read by nothing

`Synapse.derived` (`_base_channel.py:351`) has exactly three assignments and
one read in the entire repository:

```
braincell/synapse/markov.py:58:    derived = {}                      # restates the base default
braincell/synapse/markov.py:97:    derived = {"g": DerivedSpec()}
braincell/synapse/markov_test.py:17:  self.assertEqual(model.derived, {})
```

No consumer — not `_compute/bindings.py`, not `_multi_compartment/synapses.py`,
not `network/`, not the recorder — ever reads `cls.derived`. `DerivedSpec`
itself is an empty frozen dataclass with no fields and no methods.

What actually makes `Exp2Syn.g` publicly readable is duck typing:
`_multi_compartment/synapses.py:479-482` does `hasattr(node, field)` then
`getattr(node, field)`. That is why `network/recording_test.py:114`
(`observe.synapse().state("g")`) works uniformly across an `ExpSyn`, where
`g` is a `DiffEqSingleState`, and an `Exp2Syn`, where it is a `@property`.
Deleting the declaration changes nothing observable.

The cost of keeping it is that it reads as authoritative. A future synapse
author who declares `derived = {"i"}` and forgets the `@property` gets an
`AttributeError` from deep inside the store rather than a schema error at
class definition.

### 5. `braincell/synapse/__init__.py` re-exports three names nobody imports

```python
from braincell.mech import DerivedSpec, ParameterSpec, StateSpec
```

There is no `from braincell.synapse import ParameterSpec` anywhere in `.py`,
`.md`, `.rst`, or `.ipynb`. Even `markov.py` reaches them through
`braincell.mech`. And `positive` — the one schema name `markov.py` actually
uses, and the one a third-party synapse would need — is *not* re-exported, so
the single path this block was meant to serve is broken anyway.

`docs/design/network/module-layout.md:42-46` argues these types live in `mech`
precisely so they are not reached through `braincell/synapse/`. The
re-export reintroduces the second import site that design note exists to
prevent.

### 6. Three license-header violations, all in this package

`braincell/synapse/markov.py:16` and `braincell/synapse/__init__.py:16` place
`# -*- coding: utf-8 -*-` *below* the Apache header. CPython honours PEP 263
only on lines 1-2, so both are inert comments as well as misplaced. A
repository-wide scan for an encoding line below line 2 returns exactly four
files: these two and two under `examples/single_compartment/`.

`braincell/synapse/markov_test.py` carries only the `# Copyright` line;
twelve of the thirteen header lines are missing. A scan of every tracked
`.py` under `braincell/` for `"Licensed under the Apache License"` finds
exactly one file without it — this one. No pre-commit hook or guard test
enforces headers, which is why it stayed broken.

### 7. No test ever successfully constructs an `Exp2Syn`

`markov_test.py:42-44` exercises only the *raising* branch
(`tau1=2.0, tau2=1.0`), which aborts inside `validate_parameter_values`
before `__init__` returns. Consequently the following are executed by no test
in this file:

| location | untested behaviour |
|---|---|
| `markov.py:68` | `ExpSyn.compute_derivative` — the `-g/tau` decay law, the class's only dynamical equation |
| `markov.py:105-108` | `Exp2Syn._compute_factor` — the peak-normalisation constant, the hardest line in the file |
| `markov.py:110-112` | `Exp2Syn.g` — the `B - A` identity |
| `markov.py:114-119` | `Exp2Syn.apply_events` — that `A` and `B` receive the *same* `payload * factor` |
| `markov.py:121-124` | `Exp2Syn.compute_derivative` — that `A` decays with `tau1` and `B` with `tau2`, not swapped |
| `markov.py:126-127` | `Exp2Syn.current` |
| `markov.py:102` (False branch) | a valid `tau1 < tau2` being accepted |
| `markov.py:63,116` | `event_input.validate_payload` rejecting a wrong-unit payload |
| `_base_channel.py:382-386` | `reset_state` for either class |

All 13 repository-wide `Exp2Syn` test references were checked for indirect
coverage. `network/recording_test.py:107` and `network/core_test.py:77` do run
an `Exp2Syn` through a real simulation, so `compute_derivative` and
`apply_events` are *executed* — but those tests assert only `schema.size` and
finiteness. **No test anywhere asserts a numeric conductance or current
produced by `Exp2Syn`.** A `tau1`/`tau2` transposition in `compute_derivative`
would leave the entire suite green.

## Changes

### Rename the module

`braincell/synapse/markov.py` → `braincell/synapse/exponential.py`, and
`markov_test.py` → `exponential_test.py` to keep the AGENTS.md rule 10
sibling relation. The new name states what both classes are: exponential
conductance-decay synapses. It matches the `braincell/channel/` convention
(`sodium.py`, `leaky.py`, `hyperpolarization_activated.py` — named for what
they model).

No Python code imports the module path: live code reaches both classes
through the registry (`mech.Synapse("ExpSyn")`) or through
`braincell.synapse.ExpSyn`. The only code reference is
`braincell/synapse/__init__.py:19`. Prose references updated:
`braincell/mech/_synapse_schema_test.py:19` (a Sphinx `:mod:` role that would
otherwise dangle), `docs/design/network/module-layout.md:43-44`,
`docs/apis/braincell.synapse.rst`, `docs/design/interface-map.md:416-420`,
`docs/developer/extending.ipynb`, `TODO.md:163` and `TODO.md:1219-1220`.

`docs/specs/*` and `changelog.md` keep their `markov.py` references: they are
dated records of a state that was true when written.

### Delete the three receptor stubs and the duplicated blocklist

`AMPA`, `GABAa`, `NMDA`, `_UNAVAILABLE_RECEPTOR_MESSAGE`, their `__all__`
entries, and their `__init__.py` re-exports are removed. They are not
collapsed onto a shared base: a shared base preserves the whole maintenance
surface to buy back eight lines, when the correct line count is zero.

`braincell/mech/_point.py`'s `{"AMPA", "GABAa", "NMDA"}` blocklist goes with
them, making `mech.Synapse` resolve names uniformly: every unknown
`synapse_type`, including these three, now produces the registry's
`_missing_mechanism_message` at Cell-build time —

```
No 'synapse' mechanism registered as 'AMPA'. Registered synapse names: 'Exp2Syn', 'ExpSyn'...
```

— which is more actionable than the string it replaces, and covers every
typo rather than three hard-coded names. This also removes the asymmetry
against `mech.Channel` / `mech.Ion`, which already defer to the registry
(`mech/_density.py:491-511` `_resolve_class_name` is explicit about not
verifying names at declaration time).

`_point_test.py:303-306` (which tested the blocklist) is replaced by nothing;
`_compute/state_test.py:438-441` is rewritten to assert the deferred registry
error for an unregistered synapse type, which no test currently covers.
`_registry_test.py:318` is unaffected and still pins that the three names are
not registered.

### Fix the tracing hazard in `Exp2Syn.validate_parameter_values`

```python
        tau1 = np.asarray(u.get_mantissa(parameters["tau1"]))
        tau2 = np.asarray(u.get_mantissa(parameters["tau2"]))
        if np.any(tau1 >= tau2):
            raise ValueError("Exp2Syn requires tau1 < tau2.")
```

Matching `positive` three lines up. A regression test that constructs an
`Exp2Syn` inside `jax.make_jaxpr` is added first, per AGENTS.md rule 4.

The two mantissas must be extracted separately rather than comparing the
quantities directly, because `tau1` and `tau2` may arrive in different units;
`u.get_mantissa` on each is only correct when both are already in the
canonical unit. They are: `_build_parameter_columns` stacks each column
through `spec.validate` → `_validate_like_default`, which converts to
`spec.default.unit`, and `Synapse.__init__` does the same. The comparison is
therefore between two `ms` mantissas in both construction paths. A unit
conversion is applied explicitly rather than relying on that invariant.

### Delete the `derived` schema slot

Removed: `derived` from both synapse classes; `Synapse.derived` and the
`DerivedSpec` import in `_base_channel.py`; `DerivedSpec` from
`mech/_synapse_schema.py`, its `__all__`, `mech/__init__.py`, that module's
docstring, and `DerivedSpecTest`; the `synapse/__init__.py` re-export; and the
assertion in the synapse test. `Exp2Syn.g` remains a plain `@property`,
which is what the runtime has always used.

### Delete the dead schema re-export from `braincell/synapse/__init__.py`

`DerivedSpec` / `ParameterSpec` / `StateSpec` and their `__all__` entries.
`braincell.mech` remains the single import site, as
`docs/design/network/module-layout.md` intends.

### License headers

Both inert `# -*- coding: utf-8 -*-` lines deleted. The full Apache-2.0
block restored in `exponential_test.py`, year 2026 — the file was created
2026-08-27 in `78b3720`, so this is the correct year and not a renumbering.

### `_compute_factor` micro-cleanup

`u.math.asarray(self.tau2 / self.tau1)` → `self.tau2 / self.tau1`. The
division already yields a bare mantissa on all three construction paths
(`float`, `jax.Array`, `ndarray`), and `u.math.log` accepts each directly;
the wrapper costs 17 us of a ~300 us call and reads as if a unit were being
stripped.

### Tests

`exponential_test.py` gains coverage for every row of the table in §7:

- `Exp2Syn` construction with a valid `tau1 < tau2` (the never-taken branch).
- `_compute_factor` against its defining property: a unit event drives
  `max_t g(t)` to exactly the payload. Checked against the closed-form peak
  time `tp = tau1*tau2/(tau2-tau1) * ln(tau2/tau1)` and against a numerically
  integrated trajectory.
- `g == B - A`, and that one event moves `A` and `B` by the same amount.
- `compute_derivative` decaying `A` with `tau1` and `B` with `tau2` — the
  transposition guard.
- `Exp2Syn.current` sign and magnitude.
- `ExpSyn.compute_derivative` (`-g/tau`).
- `reset_state` returning both classes' states to their declared initials.
- `validate_payload` rejecting a wrong-unit payload.
- The tracing regression from §3.

## Breaking changes

1. **`braincell/synapse/markov.py` is renamed to `braincell/synapse/exponential.py`**
   (and `markov_test.py` to `exponential_test.py`). `import braincell.synapse.markov`
   and `from braincell.synapse.markov import ...` stop working. `braincell.synapse.ExpSyn`
   and `braincell.synapse.Exp2Syn` are unaffected. No shim.
2. **`braincell.synapse.AMPA`, `braincell.synapse.GABAa`, `braincell.synapse.NMDA` are deleted.**
   They raised `NotImplementedError` on construction, so no working code depended on
   them. Attribute access now raises `AttributeError`.
3. **`braincell.mech.Synapse("AMPA" | "GABAa" | "NMDA")` no longer raises at declaration time.**
   It constructs like any other unknown synapse type and raises at Cell build with the
   registry's suggestion message.
4. **`braincell.mech.DerivedSpec` is deleted**, along with the `derived` class attribute on
   `braincell.Synapse` and on both concrete synapses. Nothing read it. Declared derived
   values continue to work as plain properties.
5. **`braincell.synapse.DerivedSpec` / `ParameterSpec` / `StateSpec` re-exports are deleted.**
   Import them from `braincell.mech`, which is where every existing caller already
   imports them from.

Per the sweep's standing rule and the precedent in
`docs/specs/2026-09-02-mech-synapse-rename.md`, there are no deprecation
shims, no aliases, and no `warnings.warn` bridges. The package is at 0.1.0.

## Considered and declined

**Memoise `Exp2Syn._compute_factor`.** Three of the four reviews proposed
caching it via `cached_q10_factor`'s identity-keyed pattern; the fourth
measured it. On the real run path `Cell.place` materialises runtime
parameters as **numpy** arrays, so the seven-op chain never stages at all —
the jaxpr of a full traced step with a placed, event-bound `Exp2Syn` is 293
equations with `has log: False, has exp: False`. When it does stage (direct
construction) XLA constant-folds it: the optimised HLO of the whole chain is
`%constant.0 = f32[] constant(1.43505526); ROOT %mul.0 = multiply(param, broadcast(constant.0))`.
Instrumented call counts on an 800-step `Cell.run` and a 20-cell
`Network.run` are `{'_compute_factor': 1}` — once per trace, 0.078% of total
wall time. The `cached_q10_factor` precedent is justified by its own
docstring at 26 calls per `compute_derivative`; this is 1 call per compile.
The invalidation contract *would* hold (`_compute/bindings.py:1093-1097`
`_sync_runtime_node_param` rebinds by `setattr` and fires `_on_param_updated`
for synapse layouts), so the memo would be correct — it just buys nothing.

**A shared ohmic point-current template.** `ExpSyn.current` and
`Exp2Syn.current` do both spell `<g> * (self.e - V_post)`, and
`channel/_base.py`'s `_OhmicCurrent` is the exact shape. But `_OhmicCurrent`
is amortised over 47 concrete classes across two gating families that could
not otherwise share a `current`; here it would be amortised over 2 classes in
one family, and after the stub deletion it is permanently 2. Copying the form
without the fan-out that justifies it is cargo-culting.

**A template-method `apply_events` on the `Synapse` base.** Both overrides
open with `self.event_input.validate_payload(payload)`, and moving that into
the base would make "every synapse validates its payload" an enforced
invariant rather than a convention. Same fan-out objection: two classes, and
it adds a `_apply_events` indirection between the class and its own event
law. Revisit when a fifth synapse lands.

**Hoisting `root_type = HHTypedNeuron` off the two classes.** One review
called it dead. It is not: `_base_neuron.py:295-296` collects
`self.nodes(IonChannel, allowed_hierarchy=(1, 1))` — which includes any
`Synapse`, since `Synapse(IonChannel)` — and passes them to
`TreeNode.check_hierarchies`, whose `_root_leaf_pair_check`
(`_misc.py:481-486`) raises *"Child class should define root_type"* when the
attribute is absent. The runtime synapse path does bypass this, but
`cell.some_syn = ExpSyn(4)` does not. The only existing hoist site is
`_base_ion.py:706-707`, which exists to break an import cycle; adding
`Synapse.root_type = HHTypedNeuron` there would put a synapse declaration in
the ion module for no reason. Two explicit declarations, matching what every
channel writes, is the clearer form.

**A schema-level ordered-pair validator.** `ParameterSpec.validate(value, name)`
structurally cannot express a cross-field rule — the validator never sees
siblings — so this means adding a whole new relation concept, not reusing
`validator=`. A repository sweep finds exactly one `validate_parameter_values`
override, one orderable parameter pair, and zero mechanisms with such a pair
and no check. One caller does not justify the concept.

## Deferred to later iterations of the sweep

- `_multi_compartment/synapses.py:165` calls `spec.validate` **inside** the
  per-synapse-row Python loop, before the rows are stacked at line 167.
  Measured: 3005 calls costing 16.5 ms, 7.2% of `init_state`, on a
  1000-synapse cell; validating the stacked column once instead would make it
  5 calls. Iteration 11 (`_multi_compartment`).
- `braincell/network/_testing.py:97-138` — `make_post_cell` and
  `make_post_cell_with_synapse_pool` pass `weight=1.0 * u.uS` to
  `mech.Synapse("ExpSyn", ...)`, a parameter `ExpSyn` has not accepted since
  the weight-removal refactor; `_build_parameter_columns` would raise
  `TypeError`. Both are in `__all__` and called from nowhere. Iteration 12
  (`network`).
- `braincell.Synapse` (runtime) and `braincell.mech.Synapse` (declaration)
  collide; four sites import the former `as RuntimeSynapse`. Iteration 14.
- `mech.Synapse` accepts only a string while `mech.Channel` / `mech.Ion`
  accept a class object; `_resolve_class_name` already generalises this.
  Iteration 14.
- `mech/_point.py:419-425` `_raise_if_nonpositive_duration` re-implements
  `positive` plus `_validate_like_default`'s non-empty check, differing only
  in the message prefix. Iteration 14.
- `_base_channel_test.py` is 44 lines and tests only re-export identity and
  `issubclass`; the ~50 lines of `Synapse.__init__` / `init_state` /
  `reset_state` / `validate_parameters` are exercised only through
  `braincell/synapse/`. Iteration 13 (root modules).
- `examples/neuron_compare/synapse/engine/braincell_runner.py:168-171`
  recomputes `g = B - A` in numpy from separately probed `A` and `B`, where
  `MechanismProbe(field="g")` works today. Left alone: it is the NEURON
  comparison harness, and the numpy path is what is being compared.

## Verification

```
$ pytest braincell/synapse braincell/mech braincell/_compute -q
344 passed

$ pytest braincell/ -q
2793 passed, 15 skipped, 411 warnings, 410 subtests passed in 186.82s (0:03:06)

$ pre-commit run --files <17 changed files>
check for added large files..............................................Passed
check python ast.........................................................Passed
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
```

Collected-test delta, measured against a `git archive HEAD` extraction of the
baseline tree rather than inferred: 2798 → 2808. That is +11 in the synapse
test module (5 → 16) and −1 from the deleted `DerivedSpecTest`; 2793 passed
plus 15 skipped accounts for all 2808.

Package size, production lines: 184 → 166. Test lines: 53 → 194 — the one
place in this iteration where the count goes up, and deliberately so: §7 is
the finding with the largest consequence, and eight of the nine untested rows
now have a numeric assertion behind them.

The tracing bug was reproduced before the fix and is now pinned by
`Exp2SynTest.test_construction_survives_an_open_jax_trace`:

```
# before
ExpSyn under trace: OK
Exp2Syn under trace: TracerBoolConversionError Attempted boolean conversion of traced array with shape bool[].
# after
ExpSyn under trace: OK
Exp2Syn under trace: OK
```

No numerical behaviour changed. `_compute_factor` produces the same constant
(`1.43505526` for `tau1=0.5 ms, tau2=5 ms`, both before and after dropping the
`u.math.asarray` wrapper), and the new peak-normalisation test confirms the
constant satisfies its defining property to float32 precision.
