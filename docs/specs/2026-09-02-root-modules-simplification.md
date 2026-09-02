# Root-module simplification sweep

Iteration 13 of the module-by-module `/simplify` sweep. Target: the
top-level modules of `braincell/`, and explicitly **not** any subpackage.

| File | Lines | Role |
| --- | --- | --- |
| `__init__.py` | 178 | The public surface — 74 names in `__all__` |
| `_base_neuron.py` | 363 | `HHTypedNeuron`, the neuron base class |
| `_base_channel.py` | 402 | `IonChannel`, `Channel`, `Synapse`, `IonInfo` |
| `_base_ion.py` | 707 | `Ion`, `MixIons`, `mix_ions` |
| `_misc.py` | 570 | Shared utilities, `Container`, `TreeNode` |
| `_testing.py` | 108 | Docstring-conformance test mixin |
| `_typing.py` | 40 | The shared type-alias table |
| `_version.py` | 18 | `__version__`, `__version_info__` |
| `_base_neuron_test.py` | 78 | |
| `_base_channel_test.py` | 43 | |
| `_base_ion_test.py` | 355 | |
| `_misc_test.py` | 100 | |

Baseline on `921a255`:

- `pytest braincell/_base_channel_test.py braincell/_base_ion_test.py
  braincell/_base_neuron_test.py braincell/_misc_test.py -q` → 23 passed,
  6 subtests passed (5.75 s)
- `pytest braincell/ -q` → 2818 passed, 15 skipped, 411 subtests passed
  (184.06 s)

## Note on the file list

The sweep goal names `_base.py`; the file is `braincell/_base_neuron.py`.
It was renamed before this iteration, and the module docstring
(`_base_neuron.py:16-26`) explains why the class was given its own module:
`_base_ion` names `HHTypedNeuron` as `root_type`, and the two once reached
each other through bottom-of-file imports that worked only because of the
exact order of statements across two files.

Two design documents still say `_base`:
`docs/design/interface-map.md:17` and `:359`.

## Constraints this scope carries

These four modules are the root of the dependency graph. `_base_neuron`,
`_base_channel`, and `_base_ion` are the base classes every mechanism
subpackage inherits from, `_misc.py` is the shared utility bag every
package imports, `_typing.py` is the alias table AGENTS.md makes
mandatory, and `__init__.py` is the entire public API. A layering mistake
here is the most expensive kind, and a rename in `__init__.py` reaches
every notebook and doc page.

Everything below the root is out of scope for *edits*, but in scope for
*evidence*: a duplicate in a subpackage is the reason a root helper should
exist, and a subpackage caller is what makes a root symbol live.

### The lint bar is deliberately low, and that is not an invitation

`pyproject.toml:140` selects only `E4`, `E7`, `E9`, `F`, with a comment
recording that import sorting and pyupgrade are "deliberately NOT enabled
yet" and that each ignored code is listed with its violation count "so the
debt is auditable". `format.quote-style = "preserve"` (`:131`) likewise
keeps the existing single/double-quote mix on purpose.

Running the scope under a wider ruleset (`ARG,B,SIM,RUF,PIE,C4,PERF`)
reports 29 findings. They are *not* violations of this project's stated
standard, so this sweep does not mass-fix them. Each one adopted below is
adopted because it is a defect on its own terms, and is named individually.
The rest are left, deliberately, rather than churning eight files against a
bar the repository has not set.

## Defects fixed

Four, each reproduced before it was fixed and pinned by a new test.

### 1. `MixIons(...)` silently dropped a constructor-passed channel

`MixIons.__init__` wrote `self.channels.update(self._format_elements(...))`
directly. `add()` is the only door into `channels` that also registers each
channel's current with the ions that own it, so a channel passed to the
constructor contributed **no current at all** — `ion.current(V,
include_external=True)` returned `None` where the same channel passed to
`add()` produced a current.

`__init__` now calls `self.add(**channels)`. That made `ion_types` load-
bearing one line earlier than it used to be, so `_ion_types` was deleted and
`ion_types` derives from `self.ions`.

Pinned by `MixIonsConstructorChannelTest`, which builds the same pool both
ways and asserts they agree.

### 2. `IonChannel(size=...)` accepted a tuple with non-int members

Three inlined copies of the size check — two of them with the same missing-
space typo in the message (`int.But`) — checked only `size[0]`, so
`(4, "x")` was accepted and failed much later somewhere unrelated. One
`_normalize_size` now type-checks every element.

### 3. An empty ion pool returned two different "nothing" values

`Ion.current` returned `None`; `MixIons.current` returned a bare, unitless
`0.0`. Neither is a valid addend for a current density, and the ordinary way
to write a calcium-dependent potassium current — K and Ca pools carrying
only a mixed channel — crashed before a single step ran:

```
UnitMismatchError: Cannot calculate 0. nA / cm^2 + None,
because units do not match: nA / cm^2 != 1
```

`MixIons.current` now returns `None` too, so there is one sentinel, and
`_single_compartment/base.py` skips a pool that has nothing to contribute
rather than adding it.

### 4. Two of four dynamic-ion `derivative` implementations ignored their own default

`derivative(self, Ci, V, total_current=None)` declares `None` valid.
`CdpHVA_SU2015_DCN` and `CdpLVA_SU2015_DCN` guarded against it;
`CalciumDetailed` and `CalciumFirstOrder` did not, and raised `TypeError:
unsupported operand type(s) for /: 'NoneType' and 'float'` on exactly the
model fixed above. The guard now lives once, on `DynamicNernstIon`, as
`_drive_current`, and all four route through it.

This one file is outside the nominal root-module scope. It is here because
fixing only defect 3 moves the crash one frame later and delivers nothing a
user can see.

## Themes

### One helper where there were several near-copies

| Was | Is | Copies |
| --- | --- | --- |
| `_mask_inactive_current` + `_safe_inactive_voltage` | `_where_active(value, mask, fill)` | 2 → 1 |
| three inlined `size` checks in `_base_channel.py` | `_normalize_size` | 3 → 1 |
| `MixIons.pre_integral` / `compute_derivative` / `post_integral` | `_run_on_dependent_children` | 3 → 1 |
| `Ion`'s three lifecycle child loops | `_run_on_dependent_children` | 3 → 1 |
| `Ion.init_state` / `reset_state` | `_run_state_lifecycle` | 2 → 1 |
| seven `graph.nodes(...)` spellings in `MixIons` | `_channels()` | 7 → 1 |
| `float(np.asarray(q.to_decimal(unit), ...).reshape(()))` | `_misc.scalar_decimal` | 16 → 1 |
| four hand-copied `__all__` guards | `_testing.ReExportTests` | 4 → 1 |
| `_BOUND_OPERATORS` + a per-call `comparators` dict | `_COMPARATORS` | 2 → 1 |
| a `None` guard in 2 of 4 `derivative` bodies | `DynamicNernstIon._drive_current` | 2 → 1 (+2 fixed) |

`scalar_decimal` deserves a note: the sixteen sites spelled it two ways, and
the two are **not** equivalent. Without `reshape(())`, NumPy 2.4 raises
`TypeError: only 0-dimensional arrays can be converted to Python scalars`
for a length-1 quantity that the other form accepts — the exact shape
`validate_time_quantity` documents as scalar. The shared helper is the
accepting form.

### Dead code and unreachable branches

- `Ion.current`'s `if len(nodes) > 0:` wrapper (the loop is a no-op on an
  empty tuple) and its `and self._external_currents` conjunct (the `for`
  over an empty dict is likewise a no-op).
- `Container.__getattr__`'s `if item == '_container_name'` branch.
  `__getattr__` only runs when normal lookup already failed, so either the
  class attribute resolves and this method is never entered for that name,
  or it does not and the `super().__getattribute__` on the line above raises
  first.
- `validate_time_quantity`'s second shape check, which existed only so the
  positivity test could call `reshape(())`. Comparing elementwise instead
  lets `require_scalar` and `require_positive` be chosen independently,
  which the signature already promised and the docstring already claimed.
- `IonChannel.post_integral`'s `Raises: NotImplementedError`, on a body that
  is `pass`.
- Two bare `node: Channel` annotation statements in `Ion` that annotate a
  loop variable and do nothing.
- The trailing `Ion.root_type = HHTypedNeuron` / `MixIons.root_type = ...`
  patch, left from when the two classes shared a module and reached each
  other through bottom-of-file imports. Both are plain class attributes now.
- `_base_channel._make_synapse_state`, a one-call wrapper whose only content
  was a function-body import. Line 38 of the same file already imports from
  `quad.protocol`; there was no cycle to dodge.

### Docstrings that described something else

`_base_neuron.HHTypedNeuron` is a neuron, but ~90 lines of its docstrings
described ion channels — `current` opened "Generate ion channel current",
`compute_derivative` "Compute the derivative of the state variables for the
ion channel". They also documented an `AssertionError` that no code path
raises. `IonChannel` documented `in_size` / `out_size` attributes the class
never sets.

Nineteen Google-style sections (`Args:`, `Returns:`, `Note:`) across
`_base_ion.py` and `_base_channel.py` are now NumPy-doc, which AGENTS.md
requires and which Sphinx actually renders.

## Efficiency

Two measured changes; everything else measured and left alone.

### `MixIons` packed each ion once per channel

`_infos_for(node)` called `ion.pack_info()` for each of the node's roots,
once per node. For a `DynamicNernstIon` such as `CalciumDetailed`,
`pack_info()` reads `E`, a **computed Nernst property**, so the whole
expression was re-emitted into the jaxpr for every channel. Each lifecycle
method now packs once, into an identity-keyed map, and selects from it.

Measured on a `SingleCompartment(1000)` with `mix_ions(PotassiumFixed,
CalciumDetailed)` carrying 32 `AHP_De1994` plus one `CaT_HM1992` and one
`K_HH1952` (`dev/verify_mixions.py`, best of 3):

| | step-body eqns | `make_jaxpr` |
| --- | --- | --- |
| before | 2261 | 0.232 s |
| after | 1461 (**−35 %**) | 0.139 s (**−40 %**) |

Output checksum bit-identical (`-64999.019531`). XLA
common-subexpression-eliminates the duplicates, so the *compiled* step is
unchanged — this buys trace and compile time and a smaller HLO, not a
faster run. Say so plainly rather than claiming a speedup that is not there.

### The external-current callback captured the whole `MixIons`

`_get_ion_fun`'s closure ran `isinstance(ion, root)` and a linear
`_get_ion` scan on every call, and captured `self`. The callback is stored
in `Ion._external_currents` for the life of the model, so it pinned the
`MixIons` and, through it, every child channel. Which root each argument
position comes from is now resolved once, at registration — `self.ions` is
fixed at construction, so the answer cannot change — and `self` is no longer
captured.

Verified: the closure cells are now `(None, AHP_De1994, tuple)` with no
`MixIons` among them, and with `gc` disabled both the `MixIons` and its
child channel are released by refcounting alone after the owning cell is
dropped. Both are pinned by tests.

## Public surface

`braincell/__init__.py` advertised 73 names in an unsorted `__all__` and
was incoherent about its own domain packages: `filter` and `morph` resolved
only as a side effect of some unrelated module importing them, and
`braincell.io` did not resolve at all, while `braincell.vis` did. All ten
non-underscored packages are now imported and listed, and `__all__` is
ASCII-sorted — the convention `braincell.vis.__all__` already followed.

Adding `io` to the eager imports costs a measured 15.8 ms of a ~2.3 s
import (0.7 %); `vis`, already eager, costs 19.8 ms. Deferring both was
measured and rejected: 72 % of `import braincell`'s 3.04 s is third-party
(`import brainpy, braintools` alone is 2.30 s) and runs before any
BrainCell module body.

`Container.__module__` and `TreeNode.__module__` both claimed `braincell`,
where neither is exported. `pickle.dumps` failed on both with *"attribute
lookup Container on braincell failed"*, and `help()` pointed at a path that
does not resolve. They report `braincell._misc` now. A new
`braincell/__init___test.py` pins all of this, including that every
exported class is picklable by reference.

## Breaking changes

1. **`braincell.MixIons.current` returns `None`, not `0.0`, for an empty
   pool.** It now agrees with `Ion.current`. Code that added the result
   unconditionally must skip `None`; the one in-repo site
   (`_single_compartment/base.py`) does.
2. **`braincell.MixIons.ion_types` is a read-only property.** The
   `_ion_types` attribute is gone; the property derives from `self.ions`.
3. **`IonChannel(size=(4, "x"))` now raises.** It used to be accepted.
4. **`braincell.__all__` gained `filter`, `io`, `morph` and is sorted.**
   Additive for `import *`; a test comparing `__all__` to a literal list
   would need updating.
5. **`Container.__module__` / `TreeNode.__module__` are `braincell._misc`.**
   Both are private; nothing outside the package could have depended on the
   old value except to fail at pickling.
6. **`DynamicNernstIon.derivative` implementations must call
   `_drive_current`.** An out-of-tree subclass that already guards against
   `None` keeps working; one that does not is fixed by adding the call.

No deprecation shims, no aliases, no warnings — the old spellings are gone.

## Declined, with the check that overturned them

- **Cache `MixIons._channels()`.** 37 `brainstate.graph.nodes` calls
  totalling 8.10 ms in one traced step of the 32-channel model — 2 % of that
  model's 0.42 s trace. `graph.nodes` is linear, not superlinear (37.6 µs at
  n=1, 492 µs at n=64). A cache needs invalidation on `add()`; 2 % of trace
  time does not buy that risk.
- **Cache the converted bound in `_check_bounds`.** 5.44 µs → 1.72 µs per
  call (3.2×), but that is ≈7.6 ms of a ~600 ms `SwcReader.read` of
  `CA1.swc` (1.3 %), and the end-to-end A/B is inside run-to-run noise.
- **Rewrite `cast_like` to read `mant.dtype` instead of materialising an
  array.** 1.4–1.8× on the microbenchmark, identical jaxpr, end-to-end
  inside noise.
- **Memoize `profiler_safe_name` (60×) and hoist `profile_barrier_current`'s
  `os.environ.get`.** 0.26 ms and 0.047 ms per trace respectively.
- **Drop `jax.named_call` from `Ion.current`.** Measured *worse*:
  trace+compile 2.43 s → 2.72 s, warm run 537 → 550 ms.
- **A lazy `__getattr__` for the domain packages.** Would fix the `io` gap
  and defer `vis`, but it is new machinery in a simplification sweep and it
  changes when import side effects run. The eager import costs 15.8 ms.
  Iteration 14 owns the public surface.
- **Sort every subpackage's `__all__`.** `channel` (123 names), `ion`,
  `filter`, `quad`, `mech`, `morph` are unsorted. The `ReExportTests` mixin
  makes the check one flag away, but flipping it is a package-wide
  convention change, not a root-module one. `require_sorted_all` is opt-in.
- **Replace `_misc.Container` / `TreeNode` with brainpy's.** A deliberate
  fork: different `_container_name` mechanism, `TypeError` where brainpy
  raises `ValueError`, and a `check_fun` hook `MixIons` depends on.
- **Replace `network`'s two `_stack_values` copies with one.** They are not
  the same function: `connection.py` stacks along axis 0 and returns
  `ndarray float64`; `recording.py` stacks along axis −1 with `u.math.stack`
  and returns `ArrayImpl float32`. Substituting either would silently change
  the dtype of every published value.
- **The 29 findings from the wider ruff ruleset.** See above — the
  repository has not set that bar, and each one adopted here was adopted on
  its own merits and named.

## Measured and found to be nothing

- Total `tottime` across all four root modules in a full trace+compile of a
  33-channel `SingleCompartment(1000)`: **0.004 s of 3.358 s**. `Ion.current`
  is called 8× per *trace* (4 RK stages × 2 ions), not per step. Python cost
  in these modules is not where time goes.
- `freeze_array` across all 117 `braincell/network/` tests: 193 host-array
  calls copying **4 KB total**, plus 75 device-array fast-path hits. No copy
  amplification.
- The `DocstringConformanceTests` guards re-parse each docstring in four
  methods. Both consumers run 25 tests in 3.29 s, every duration < 0.01 s.

## Verification

```
$ pytest braincell/ -q
2854 passed, 20 skipped, 411 warnings, 417 subtests passed in 181.01s
```

Baseline on `921a255` was 2818 passed / 15 skipped / 411 subtests. The +36
tests are the new guards; the +5 skips are `ReExportTests`' opt-out
branches (`require_sorted_all` off for `channel` and `ion`, no
`reexport_sources` for `braincell`, `vis`, `network`).
