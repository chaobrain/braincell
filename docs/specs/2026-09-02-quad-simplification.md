# Quality cleanup of `braincell.quad`

Reuse, simplification, efficiency, and altitude cleanup of the integrator
package (`braincell/quad`, 9 source modules, ~4,170 source lines plus
~2,090 lines of co-located tests). Breaking API changes are in scope and
were explicitly authorised for this pass.

This is iteration 1 of a module-by-module sweep. The package-wide pass in
PR #137 touched `quad` only twice — it deleted six unreachable integrators
and added a calling-convention field to the registry — so almost everything
below is new ground.

## Scope and method

Four independent reviews swept the package, one per angle (reuse,
simplification, efficiency, altitude). Findings were deduplicated; the
clusters that surfaced in two or more reviews independently are what
promoted them. Every claim below was re-verified against the code before it
was acted on.

Three invariants govern every edit, carried over from PR #137:

1. **The test suite stays green.** Baseline on `main` @ `a10f461` before any
   change: 2,723 passed, 15 skipped, 334 subtests, 0 failed (236.32 s).
2. **Every performance claim carries a measurement.** No fix lands on the
   strength of "this looks faster".
3. **Where two code paths disagree, the disagreement is preserved, not
   silently unified.** Divergences worth a decision are listed under
   *Deliberately not changed*.

## Breaking changes

### `implicit_euler` and `dhs_voltage` become selectable by name

Both are registered, both appear in `all_integrators`, and neither could be
selected via `solver="..."`. The model hosts call `self.solver(self)`
(`Cell._update_dynamics`) and `self.solver(self, I_ext)`
(`SingleCompartment.update`), while `implicit_euler_step(target, t, dt, *args)`
and `dhs_voltage_step(target, t, dt, *args)` demand explicit time arguments.
PR #137 recorded the mismatch as `IntegratorEntry.requires_time_args` plus a
test allowlist and two Notes paragraphs, rather than fixing it.

The explicit `t`/`dt` carry no information: both in-repo call sites already
read them from `brainstate.environ` immediately before calling. Both
signatures become `(target, *args, t=None, dt=None)`, where `None` means
"read from `environ`" — the convention the other 15 steps already use.

Consequently `IntegratorEntry.requires_time_args` and the private
`_requires_time_args` helper are **deleted**. The field had no production
consumer anywhere in `braincell/`, `examples/`, or `docs/` — it was read only
by its own test, which already re-derives the same answer from
`inspect.signature(...).bind`. `_registry_test.py::CallConventionTest` is
rewritten to assert the strictly stronger property that *every* registered
entry binds against the host convention.

`examples/multi_compartment/quad.ipynb` stores an `IntegratorEntry` repr in a
cell output, and a markdown cell describes the two steps as taking positional
`t`/`dt`. Both are corrected in place — the stored repr loses the field, the
prose describes the keyword-only override — rather than by re-executing the
notebook, which would churn every unrelated output and memory address in the
file for two textual fixes.

### `braincell.quad.dhs_voltage_step` is exported

`dhs_voltage` was advertised by `all_integrators` and by
`get_integrator`'s "Available: ..." error message, but
`braincell.quad.dhs_voltage_step` raised `AttributeError` — the symbol was
registered and never exported. Its own tests import it from the private
`braincell.quad._staggered`. It joins `__all__`.

### `power_iteration_expm` is deleted

A one-line wrapper around `jax.scipy.linalg.expm` behind a dead branch. Its
only caller passed `method='scipy'` explicitly, so the hand-rolled Taylor
series (`method='approx'`, `num_steps=20`) — which its own docstring calls
"not numerically stable or efficient" — had never run. The call site uses the
already-imported `expm` directly.

### `_to_jax_quantity` and `_array_dtype` are deleted

Both re-implement `brainunit` helpers: `u.math.asarray(value, unit=unit)` and
`u.math.get_dtype(value)` respectively. `u.math.get_dtype` was already in use
elsewhere in the same package. `_to_jax_quantity` had a dedicated test pinning
that converting a `float32` quantity under `precision=64` does *not* promote to
`float64`; the DHS numeric state still depends on that, so the test is
retargeted at `u.math.asarray` rather than dropped. It now reads as a contract
test against `brainunit`: if that promotion behaviour ever changes, it surfaces
here instead of as silent numerical drift in the voltage solve.

### `DiffEqModule.diffeq_state_merging` replaces an isinstance chain

`exp_euler_step` imported `HHTypedNeuron`, `Cell`, and `SingleCompartment`
inside its body — the **only** place any `quad` module imports a model class —
purely to choose between `merging='stack'` and `merging='concat'`. That is a
property of the state layout the host allocated, which the host already knows.
`DiffEqModule` gains a documented `ClassVar[str]`, defaulting to `'stack'`;
`Cell` overrides it to `'concat'`. Behaviour is identical for both existing
hosts, `quad` loses its last upward import edge, and any third-party
`DiffEqModule` can now use `exp_euler_step` instead of hitting a `TypeError`.

`backward_euler_step`'s hardcoded `merging='stack'` is deliberately **not**
changed — switching it would alter `Cell` numerics (stack gives a
per-compartment block-diagonal Jacobian, concat a fully coupled one), which is
a modelling decision, not a refactor.

### The `DiffEqModule` type guard raises `TypeError`, not `AssertionError`

`apply_standard_solver_step` guarded its contract with `assert`, which
evaporates under `python -O`, while two hand-written duplicates raised
`TypeError`. In-file comments show the conversion was started and left
half-done. The guard is now `TypeError` in the one shared place; the
duplicates in `_exp_euler.py` and `_staggered.py` cover only the paths that
bypass it.

### `IntegratorRegistry.items()` is deleted

Zero callers anywhere, including its own test file, and undocumented.
`as_dict(include_aliases=False)` already returns the same mapping.

## What was fixed

### Reuse — one home per concept

- **The `t`/`dt` environ prologue existed in 14 copies and had already
  drifted.** Thirteen sites read
  `environ.get('t', getattr(target, 'current_time', 0.0 * u.ms))`; the
  fourteenth, `_staggered.py:130`, read `environ.get('t', 0.0)` — a **bare
  unitless default**, against the mandatory-units convention, which then flowed
  into `runtime.evaluate_point_clamps(t=...)`. A single
  `environ_time(target)` in `_util.py` replaces all of them, and
  `_general_rk_step` now reads the pair itself, collapsing eleven three-line
  step bodies to one line each.
- **The DHS forward-elimination kernel existed twice.**
  `comp_triang_raw`'s inline loop body is byte-identical arithmetic to
  `_comp_triang_level`, which runs only under
  `BRAINCELL_PROFILE_DHS_LEVELS=1`. A numerics fix applied to the default path
  would silently miss the profiled one, and CI never exercises it — the
  divergence would surface as "the profiler build gives different voltages".
  `comp_triang_raw` now calls the helper.
- **`_check_comp_triang` / `_check_comp_backsub` shared six verbatim
  checks** and had already partly diverged (only one checks
  `diags.shape == solves.shape`). The shared checks move to
  `_check_dhs_operands`; each caller keeps only its own extras.
- **`_dict_state_to_arr` / `_dict_derivative_to_arr`** differed by one
  attribute name; they collapse to one helper.
- **The "append one sentinel row" expression appeared three times**, with the
  one difference that matters (`ones_like` for `diags`, `zeros_like` for
  `lowers`/`uppers`) buried in identical noise. `_with_sentinel(base, fill)`
  makes the intent readable.
- **`quad/_testing.py` is new** — the package was the only one in `braincell/`
  without the shared-test-helper module AGENTS.md prescribes. It absorbs the
  HH fixture (byte-identical in two files, one-line different in a third),
  `_LinearDecay` (five definitions that had forked into two incompatible
  shapes), `_FLOAT_DTYPE` (six copies), and `_drive` (two copies with
  *different return types*).
- `_registry_test.py` enumerated the canonical integrator names twice; the
  bare set is now derived from the metadata table, so registering an
  integrator means editing one list.

### Simplification

- **Thirteen tests that assert nothing** were deleted. Eleven in
  `TestRungeKutta` plus one each in `_exp_euler_test.py` and
  `_backward_euler_test.py` ran twelve 10 ms HH simulations apiece and then
  called `plt.close()`. Measured cost: **31.0 s of `_runge_kutta_test.py`'s
  52.5 s**. `RungeKuttaConvergenceTest` in the same file already covers all
  eleven methods *with* assertions on the analytic solution and the observed
  convergence order. Two of the three copies also carried the stale class name
  `TestRungeKutta` — in `_exp_euler_test.py`, a class called `TestRungeKutta`
  with a method called `test_euler_step` was testing `ind_exp_euler`.
- **`_newton_method` was mostly unreachable.** Its only caller invokes it
  positionally through `apply_standard_solver_step`, so `modified`, `tol`,
  `max_iter`, and `order` always take their defaults. That makes
  `body_fun_modified`, the `order == 1` branch, and the `else: raise` branch
  statically dead, alongside a triple-quoted `jax.lax.while_loop` alternative
  whose carry arity does not match the live code. The docstring documented the
  dead branch (backward Euler) rather than the live one (trapezoidal /
  Crank-Nicolson) and is rewritten to describe what actually runs.
- **Seven `jax.named_scope` + `jax.named_call` pairs collapse to one label
  each.** `jax.named_call` is documented as returning "a version of `fun` that
  is wrapped in a `named_scope`" — its implementation is
  `source_info_util.extend_name_stack(name)(fun)` — so wrapping it inside an
  explicit `named_scope` pushes two entries for one call and every DHS region
  appeared twice in a profile. `staggered_step` in the same file already used
  `named_scope` alone. `examples/profiling/README.md` documents only the outer
  names, so the `_call` suffixes were never part of the published workflow.
- `_edge_point_current` took a `static_source` parameter it never read.
- `_edge_conductance` hand-rolled two accumulations and guarded them with a
  `# pragma: no cover` branch that cannot fire (the caller already raises on
  an empty role list). Both become `functools.reduce(operator.add, ...)`;
  `sum()` is deliberately avoided because its `0` start breaks unit
  arithmetic. `_rk_update` gets the same treatment.
- `_ind_exp_euler_step_selected` normalised `excluded_paths` and then handed
  it to `split_diffeq_states`, which normalised it again; the callee owns the
  contract.
- A hand-rolled loop counter used only to detect the first pass becomes
  `enumerate`.
- The "concat or stack" test was spelled in four places, one of them **inside
  a per-leaf loop**; it is validated once at the boundary.
- `_RegistryDictView.__getitem__` caught `KeyError` only to re-raise
  `KeyError(name)`, which is what the callee already raises.

### Efficiency — measured, or not landed

Invariant 2 applies here: each item below states what was measured.

- **`_build_backsub_indices` built more rows than the tree can need.** The
  recursive-doubling jump table was grown until `np.all(k_step_parent ==
  n_nodes)`, i.e. until every node had walked past the root. A node tree of
  `n` nodes needs at most `ceil(log2(n))` doubling steps, so the loop is now
  bounded by `max(1, int(n_nodes)).bit_length()` and stops as soon as either
  condition holds. Verified across seven tree shapes: output **bit-identical**
  in every case, with row counts falling 11→5, 10→5, 9→2, and 10→3. Chain
  topologies are correctly unchanged, since a chain genuinely needs every
  level. The saving is in `jnp` work per step, because `comp_backsub_raw`
  iterates one gather-and-combine pass per row.

  This one needed a correction: the first verification harness reported a
  mismatch on a wide star topology. The harness was wrong, not the change — it
  generated random sentinel values, violating the invariant that
  `diags[:, n] == 1`, `solves[:, n] == 0`, and `lowers[n] == 0`. Re-run against
  the real sentinel contract, all seven shapes agree bit-for-bit.

- **`_newton_method` re-evaluated `f(t, y0)` on every iteration.** It is
  loop-invariant, and XLA does not hoist loop-invariant code out of a
  `while_loop` body, so each Newton iteration paid for a second full
  vector-field evaluation. It is hoisted above the loop. The hoist sits
  *after* the `u.get_magnitude` calls so `f0` is evaluated at exactly the
  stripped `t` the inline expression used.

- **Quad suite wall clock: 71.22 s → 31.13 s**, almost entirely from deleting
  the thirteen assert-free tests described above.

Two efficiency findings were **not** landed, on the same evidence standard PR
#137 applied:

- **Pruning zero Butcher-tableau coefficients** provably removes multiply-add
  pairs from the compiled HLO, and A/B benchmarking showed a consistent win on
  large arrays. But on a full HH-style rollout — the shape that actually
  matters — repeated runs did not show a stable difference. An optimisation
  that only shows up in a microbenchmark does not justify hand-unrolling
  eleven tableaux.
- **Rebuilding row capacitance per step** happens at trace time, not per
  simulated step, so the cost is paid once per compilation and does not appear
  in a rollout.

### Altitude — behaviour moved to the layer that owns it

- **Registry logic left `__init__.py`.** `get_integrator`, `all_integrators`,
  and `_RegistryDictView` are registry behaviour that lived in the package
  `__init__`, which forced `protocol.py` to carry a function-local
  `from . import get_integrator` — a genuine import cycle, since
  `quad/__init__` imports every backend and every backend imports `protocol`.
  They move to `_registry.py`, which imports only stdlib, making the cycle
  structurally impossible rather than merely deferred. This is the same fix
  shape PR #137 applied to `_base`/`_base_ion`. The public path
  `braincell.quad.get_integrator` is unchanged.
- The calling-convention unification and the `diffeq_state_merging` ClassVar,
  both described under *Breaking changes*.

### Convention compliance

- `_backward_euler.py`, `_exp_euler.py`, and `_staggered.py` now annotate
  their step functions with the `braincell._typing` aliases
  (`VectorField`, `Y0`, `Y1`, `T`, `DT`, `Args`, `Aux`) that exist for exactly
  this package; three of six modules were already using them, so the alias
  table was only half load-bearing.
- Google-style `Args:` / `Returns:` docstring sections on `jacrev_last_dim`,
  `_newton_method`, and `_backward_euler` become NumPy-doc.
- `split_diffeq_states` documented only one of its two parameters;
  `excluded_paths` is the one `staggered_step` depends on.
- Two inert `# -*- coding: utf-8 -*-` lines sat on line 16 of two test files,
  *below* the licence header. PEP 263 only honours an encoding line on line 1
  or 2, and AGENTS.md forbids anything but a shebang or encoding line above
  the header, so they were dead either way.

## Deliberately not changed

Recorded so the next reader does not re-litigate them, and so the later
iterations of this sweep know what is waiting for them.

- **`staggered_step`'s post-voltage scheduling.** It reaches into five
  underscore-private `Cell` members (`_cv_to_point`,
  `_integrate_runtime_synapse_dynamics`, `_update_ion_channel_families`,
  `_update_ion_channels_by_integration`, plus `ion_channel_update_order`) to
  decide the order in which mechanisms advance — scheduling, not numerics, and
  `Cell` already owns it. The deep fix is a `Cell.advance_mechanisms(...)`
  hook declared on the host protocol. **Deferred to iteration 11
  (`_multi_compartment`)**, because the change is a `Cell` method extraction
  and two `_staggered_test.py` schedule tests move to `cell_test.py`.
- **DHS static assembly lives in `quad`.** ~250 lines of pure NumPy topology
  work over `node_tree`/`scheduling`/`cvs`, plus `build_cv_axial_operator`,
  which has zero callers inside `quad` — its only consumer is
  `Cell._get_axial_operator`. `_compute/state.py` already owns the cache slots
  and types them `object` because it cannot name `DHSStaticSource` without
  importing `quad`. **Deferred to iteration 8 (`_compute`)**, where the
  receiving module lives.
- **`excluded_paths` is not part of the step protocol.** Only
  `backward_euler_step` and `ind_exp_euler_step` accept it, so
  `ion/_base.py:731-738` probes for support by calling the solver and
  **parsing the `TypeError` message**. That runs under trace inside a
  `for_loop`, so a genuine `TypeError` whose message happens to contain the
  substring is swallowed and the step re-executed after partial state writes.
  The fix is to make the keyword uniform across every registered step.
  **Deferred to iteration 5 (`ion`)**, which owns the offending code.
- **The registry cannot express which host an integrator supports.**
  `SingleCompartment(size=1, solver="staggered")` resolves fine and then dies
  on `target._cv_to_point`. This is the same class of rot that killed the six
  integrators deleted in PR #137. The structural guard depends on a declared
  host protocol, so it **follows the deferred item above**.
- **`implicit_euler` runs Crank-Nicolson, not implicit Euler.** The registry
  describes it as "Implicit Euler via Newton iteration" with `order=1`, but
  `_newton_method`'s `order` parameter defaults to `2` and its own docstring
  and `_implicit_test.py` both say trapezoidal. Correcting the *name* or the
  *scheme* changes either a public solver identifier or the numerics, so it is
  a decision for the maintainer, not a refactor. Recorded here; the dead
  `order == 1` branch is removed regardless, which makes the live scheme
  unambiguous in the source.
- **Module naming inside `quad`.** Eight of nine modules carry a leading
  underscore and `protocol.py` does not, but the underscore tracks nothing:
  `_registry.py` exports two documented public classes, while
  `braincell.quad.protocol` is a path nobody needs. Sibling public packages
  (`filter/`, `network/`) use unprefixed inner modules. Renaming touches 14
  import sites plus two Jinja channel templates, so it belongs to the
  cross-module pass. **Deferred to iteration 14.**
- **`ralston2_tableau` duplicates `rk2_tableau`'s values.** Both are
  conventional names for the same second-order method and `ralston2_step`'s
  docstring already says so. Aliasing the two would couple them through a
  shared mutable dataclass instance, so the literals stay.

## Verification

### Test count is reconciled, not just compared

The suite drops from 2,723 to 2,713 passing tests, so "still green" is not
sufficient evidence on its own — a deleted test also stays green. Collected
test IDs were diffed between `main` and this branch (`pytest braincell/quad
--collect-only -q`), giving 20 removed and 10 added. Every removal is
accounted for:

| Removed | Why |
|---|---|
| `_runge_kutta_test.py::TestRungeKutta` (11 tests) | Assert-free; `RungeKuttaConvergenceTest` covers the same eleven methods with assertions. |
| `_exp_euler_test.py::TestRungeKutta::test_euler_step` | Assert-free, and misnamed — it tested `ind_exp_euler`. Replaced by `IndExpEulerHHTest::test_drives_hodgkin_huxley_to_a_spiking_trace`. |
| `_backward_euler_test.py::TestBackwardEulerHH::test_backward_euler_step` | Assert-free. Replaced by `BackwardEulerHHTest::test_drives_hodgkin_huxley_to_a_spiking_trace`. |
| `ExpEulerTypeGuardTest::test_rejects_minimal_diffeq_module` | The behaviour it pinned is intentionally inverted by `diffeq_state_merging`; replaced by `ExpEulerTargetContractTest::test_accepts_minimal_diffeq_module` plus `test_default_merging_is_stack`. |
| `ExpEulerTypeGuardTest::test_rejects_plain_object` | Class renamed to `ExpEulerTargetContractTest`; test kept. |
| `CallConventionTest` (4 tests) | `requires_time_args` is deleted; replaced by 2 tests asserting the strictly stronger property over *every* entry. |
| `test_to_jax_quantity_preserves_existing_dtype` | Retargeted at `u.math.asarray` as `test_numeric_operands_preserve_an_existing_dtype`. |

Net −10, which is exactly the full-suite delta. No test was silently lost.

### Commands

Run from the worktree, on the final tree (after `ruff format`):

```
$ pytest braincell/quad -q
126 passed, 1 warning, 92 subtests passed in 31.13s

$ pytest braincell/ -q
2713 passed, 15 skipped, 411 warnings, 338 subtests passed in 181.94s (0:03:01)

$ pre-commit run --files <changed>
check for added large files..............................................Passed
check python ast.........................................................Passed
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
fix end of files.........................................................Passed
trim trailing whitespace.................................................Passed
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
```

Baseline on `main` @ `a10f461` for comparison: 2,723 passed, 15 skipped, 334
subtests, 236.32 s. Quad alone was 71.22 s.
