# `braincell/_single_compartment` simplification

Iteration 10 of the module-by-module simplification sweep. Written before
any code changed; the Verification section at the end is filled in
afterwards.

## Baseline

`braincell/_single_compartment` is the smallest package in the sweep. It
holds one class, `SingleCompartment`, the point-neuron counterpart to
`_multi_compartment.Cell`.

| module | lines | role |
| --- | --- | --- |
| `base_test.py` | 709 | tests |
| `base.py` | 284 | the `SingleCompartment` class |
| `__init__.py` | 19 | export surface |

Tests: `pytest braincell/_single_compartment -q` -> **48 passed, 22
subtests passed in 12.80s**.

## Findings

Every claim below was verified against this worktree.

### 1. The ion-channel graph walk is written four times and run seven times per step

`self.nodes(IonChannel, allowed_hierarchy=(1, 1))` appears at `base.py`
lines 209, 226, 232, 249 and 271 — four methods, and `compute_derivative`
does it **twice** in a row over the same unchanged graph.

One `update()` under the default rk2 solver evaluates
`compute_derivative` twice, so the walk count per step is
`1 (ind_update) + 2 x 2 (compute_derivative) + 1 (pre) + 1 (post)` = **7**,
measured by instrumenting the call:

```
compute_derivative() evaluations per update(): 2  (rk2 has 2 stages)
graph walks per update(): 7
```

Each walk costs, on a 3-channel 200-neuron `SingleCompartment`:

```
nodes(IonChannel, (1,1)) : 95.98 us/call  (3 top-level channels)
```

**This is not a runtime cost.** `update()` is meant to run inside
`brainstate.transform.jit` / `for_loop`, which trace the Python body once,
so the ~0.7 ms is paid per *trace*, not per step. Measured eagerly it is
0.9% of a step, and eager stepping is not the supported way to drive the
model. The finding is duplication, not speed — recorded here with the
number so that nobody re-litigates it as an optimization.

### 2. The `IndependentIntegration` predicate is written three times and explained zero times

```python
if not isinstance(node, IndependentIntegration):
```

appears verbatim in `pre_integral`, `compute_derivative` and
`post_integral`, while `update` deliberately calls `ind_update` on *every*
channel including those. That asymmetry is the whole contract of the mixin
— a channel that integrates itself must not also be stepped by its
neuron — and nothing in this module says so.

The same predicate appears six more times in `_base_ion.py` and several
more in `_multi_compartment/cell.py`. That is a package-wide concern, not a
`_single_compartment` one; see the deferral below.

### 3. `key` is bound and never used in three of the four loops

`for key, node in ...` in `pre_integral`, `post_integral` and `update`
never reads `key`. Only `compute_derivative` uses it, in its error message.

### 4. A package rename corrupted the class docstring's central formula

`base.py:51` reads

```
{braincell \over dt} = \phi_x {x_\infty (V) - x \over \tau_x(V)}
```

The numerator should be `d x` — the surrounding prose says "where
:math:`x \in [M, N]`", and the next display renders the same law in
alpha/beta form. A find-and-replace of the old package name rewrote the
`dx`. `git grep` finds this as the only surviving instance in the
repository, so it is a one-off, and it renders into the published API docs.

### 5. Nothing tests the dispatch split that finding 2 describes

`grep -c IndependentIntegration braincell/_single_compartment/base_test.py`
-> **0**. The 48 tests cover defaults, geometry, lifecycle, spikes, solver
resolution and state grouping, but no test asserts that an
`IndependentIntegration` channel is skipped by the three hook methods and
still driven by `update`. Refactoring that rule into a named helper without
first pinning it would be moving an untested invariant.

## Changes

- Add `_ion_channels()` and `_neuron_driven()` to
  `SingleCompartment`. The first is the one spelling of the graph walk; the
  second applies the `IndependentIntegration` filter and carries the
  docstring explaining why it exists. The four call sites use them.
- `compute_derivative` walks once instead of twice, reusing the dict it
  already has for its second loop. Seven walks per `update()` become five.
- Drop the unused `key` binding from the three loops that ignore it.
- Fix the corrupted `{braincell \over dt}` to `{d x \over dt}`.
- Add `SingleCompartmentIndependentIntegrationTest`, pinning both halves of
  the dispatch split, written before the refactor.

## Breaking changes

None. `_ion_channels` and `_neuron_driven` are new private helpers;
no public name, signature or behaviour changes.

## Considered and declined

**Caching the channel walk on the instance.** It would turn five walks per
`update()` into one, but `self.nodes(...)` must observe channels assigned as
attributes after construction — `HHTypedNeuron.__init__` only records those
passed as keyword arguments — so a cache would need an invalidation hook
that does not exist today. At 96 us per trace the walk is not worth
inventing one for.

**Hoisting `_neuron_driven` onto `HHTypedNeuron`.** The right home
for a predicate shared by `_single_compartment`, `_base_ion` and
`_multi_compartment`, but moving it is a three-package change; see below.

## Deferred to later iterations of the sweep

- **Iteration 14 (whole package).** `not isinstance(node,
  IndependentIntegration)` appears three times here, six times in
  `_base_ion.py` (lines 275, 292, 309, 531, 538, 545) and repeatedly in
  `_multi_compartment/cell.py` (1700, 1709, 1973, ...). One shared helper
  on `HHTypedNeuron` or in `quad.protocol` should replace all of them, and
  that decision spans the packages rather than belonging to any one of
  them.

## Verification

Run from the worktree with `PYTHONPATH=$PWD JAX_PLATFORMS=cpu`.

The three invariant tests were written first and pass against unmodified
code — they document existing correct behaviour rather than reproduce a
bug, which is the point: they had to exist before the predicate behind them
could be moved.

```
$ pytest braincell/_single_compartment/base_test.py -q -k IndependentIntegration
3 passed, 48 deselected in 5.41s
```

After the refactor:

```
$ pytest braincell/_single_compartment -q
51 passed, 22 subtests passed in 10.13s

$ pytest braincell/ -q
2807 passed, 15 skipped, 408 warnings, 410 subtests passed in 213.99s
```

Baseline was 48; the delta is the three added tests.

Structural check — the walk and the predicate now have one site each:

```
$ grep -c "self.nodes(IonChannel" braincell/_single_compartment/base.py
1
$ grep -c "isinstance(node, IndependentIntegration)" braincell/_single_compartment/base.py
0
```

Walk count per `update()`, measured by instrumenting the call:

```
compute_derivative() evaluations per update(): 2  (rk2 has 2 stages)
graph walks per update(), after : 5
graph walks per update(), before: 7
```
