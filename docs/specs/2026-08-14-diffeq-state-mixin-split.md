# `DiffEqState` as a mixin: split the integrability marker from the storage layout

## Requirement

> In `quad/protocol.py`, define `DiffEqState` as a Mixin, and define two
> classes: one for `SingleCompartment`, inheriting from `HiddenState`; one
> for `MultiCompartment`, inheriting from `HiddenGroupState`.

Plus, decided during design:

- The rename is a **hard break**. `DiffEqState(value)` stops working.
- The ungrouped class is named `DiffEqSingleState`; `DiffEqGroupState`
  keeps its name.
- `braincell.diffeq_state` is renamed to `braincell.state`.

## Current state

`braincell/quad/protocol.py:50` and `:179` define:

```python
class DiffEqState(brainstate.HiddenState): ...              # marker AND ungrouped class
class DiffEqGroupState(DiffEqState, brainstate.HiddenGroupState): ...
```

`DiffEqState` carries the two solver-facing slots — `derivative` and
`diffusion` — as properties whose setters call
`brainstate._state.record_state_value_write`, plus an `__init__` that
seeds `self._derivative = self._diffusion = None`.

Three problems follow from `DiffEqState` being both the marker and a
concrete class.

**1. One name, two meanings.** `isinstance(x, DiffEqState)` asks "is this
integrable?" — that is how `split_diffeq_states` (`quad/_util.py:97`) and
the channel collision guard (`channel/_base.py:142`) use it. But
`type(x) is DiffEqState` means "not grouped". A reader has to know which
sense is meant at each site.

**2. A live MRO hazard.** Because `brainstate.HiddenGroupState` derives
from `brainstate.HiddenState`, today's diamond linearises as:

```
[DiffEqGroupState, DiffEqState, HiddenGroupState, HiddenState, ShortTermState, State, ...]
```

`DiffEqState` sits **ahead of** `HiddenGroupState`. Any method
`DiffEqState` ever adds silently shadows the grouped implementation of
the same name. It works today only because `DiffEqState.__init__`
happens to cooperate via `super().__init__(*args, **kwargs)` — an
accident of style, not an invariant anyone stated.

**3. The type graph lies.** `derivative`/`diffusion` are orthogonal to
whether the trailing axis indexes independent states. The hierarchy
claims they are not.

## Design

### Hierarchy

```python
class DiffEqState(brainstate.mixin.Mixin):
    """Marker + derivative/diffusion. Not a State. Not instantiable."""
    _derivative = None
    _diffusion = None
    # derivative / diffusion properties and __pretty_repr_item__, bodies unchanged

class DiffEqSingleState(DiffEqState, brainstate.HiddenState): ...
class DiffEqGroupState(DiffEqState, brainstate.HiddenGroupState): ...
```

`__init__` is **removed**, not moved. brainstate documents that a
`Mixin` must not define one, and removing it is what makes `DiffEqState`
a legal mixin. The two `None` seeds become class-level defaults; the
property setters continue to write `self._derivative`, which creates an
instance attribute on first assignment.

Verified in a throwaway REPL against the installed brainstate:

| Check | Result |
|---|---|
| `DiffEqSingleState.__mro__` | `[DiffEqSingleState, DiffEqState, Mixin, HiddenState, ShortTermState, State, ...]` |
| `DiffEqGroupState.__mro__` | `[DiffEqGroupState, DiffEqState, Mixin, HiddenGroupState, HiddenState, ...]` |
| Diamond | gone — neither concrete class inherits the other |
| `Mixin` position | precedes the storage class in both |
| Metaclass | no `ABCMeta` / `type` conflict |
| `HiddenGroupState` `ndim >= 2` validation | still fires (`ValueError` on a 1-D value) |
| `varshape` / `num_state` on the grouped class | unchanged |
| `repr` | byte-identical to today |
| `DiffEqState(v)` | `TypeError: DiffEqState() takes no arguments` |

### Naming

| Role | Name | Rationale |
|---|---|---|
| marker | `DiffEqState` | "Participates in a differential equation" is exactly the orthogonal property being extracted. Keeping the name means every existing `isinstance` site keeps working and keeps meaning what it always meant. |
| ungrouped | `DiffEqSingleState` | Contrasts crisply with `Group`: one state per `varshape` element, versus a trailing axis of `num_state` independently-traced ones. Echoes `SingleCompartment` without hard-coding a host into the type — a `Network` mixing both hosts stays coherent. |
| grouped | `DiffEqGroupState` | Unchanged. It is a literal mirror of `brainstate.HiddenGroupState`, which is the entire reason it exists, and it shipped in PR #124 (`6a832d6`). Renaming it would be a second break in the same release for no gain. |

`DiffEqPointState` was considered and **rejected**: "point" is already
taken in braincell. A `Cell`'s mechanism arrays are shaped
`pop_size + (n_point,)` (`quad/protocol.py:185`), where a *point* is one
entry of the **grouped** trailing axis. Naming the ungrouped class after
the unit of the grouped one inverts the vocabulary.

### Factory rename

```python
def state(value, **kwargs) -> DiffEqState:          # was diffeq_state
    cls = DiffEqGroupState if _STATE_GROUPING.get() else DiffEqSingleState
    return cls(value, **kwargs)
```

`braincell.state` is free — no module, attribute, or export collides.

`hidden_state` keeps its name and body. Noted for the record: `state()`
and `hidden_state()` both allocate hidden states, and the difference
between them is integrable versus not, which the names do not convey.
Renaming the pair together was raised at design time and deliberately
deferred.

`state_grouping`, `DiffEqModule`, and `IndependentIntegration` are
untouched.

### Why the solvers do not care

`split_diffeq_states` (`quad/_util.py:97`) filters through a **callable**
(`functools.partial(_filter_diffeq, ...)`), not a State-subclass filter,
so a non-`State` marker is fine. Every `isinstance(x, DiffEqState)` site
in the codebase keeps returning what it returns today:

- `braincell/channel/_base.py:142` — the gate/Markov collision guard
- `braincell/ion/calcium_test.py` (14 sites), `ion/_base_test.py:305-307`,
  `channel/_base_test.py:476`

None of them are edited by this change. Only *constructions* change.

## Migration

`DiffEqState(...)` construction sites, by file:

| File | Sites | Becomes |
|---|---|---|
| `braincell/_single_compartment/base.py:174` | 1 | `braincell.state(...)` |
| `braincell/channel/hyperpolarization_activated.py:235-237` | 3 | commented-out code, updated in place |
| `braincell/quad/protocol_test.py` | 4 | rewritten wholesale (see Tests) |
| `braincell/quad/_util_test.py` | 4 | `DiffEqSingleState` |
| `braincell/_multi_compartment/cell_test.py` | 4 | `DiffEqSingleState` |
| `braincell/quad/{_backward_euler,_exp_euler,_implicit,_runge_kutta}_test.py` | 1 each | `DiffEqSingleState` |
| `braincell/_base_ion_test.py` | 1 | `DiffEqSingleState` |
| `docs/tutorials/channel.ipynb` | 8 | `braincell.state(...)` |
| `examples/multi_compartment/quad.ipynb` | 5 | `braincell.state(...)` |
| `docs/integration/{diffeq,advanced,solvers}.ipynb` | 3 / 2 / 1 | `braincell.state(...)` |
| `examples/single_compartment/SC01_fitting_a_hh_neuron.py` | 3 | `braincell.state(...)` |
| `examples/convert_mod/nmodl/templates/{one_ion_hh_ohmic,density_channel}.py` | 1 each | `braincell.state(...)` |
| `examples/convert_mod/nmodl/examples/artifacts/kv/rendered_channel.py` | 1 | regenerated or edited to match its template |
| `docs/apis/integration.rst` | 1 | `DiffEqSingleState` |

`diffeq_state` → `state` call sites: `braincell/channel/_base.py` (3),
`braincell/ion/_base.py` (3), `braincell/synapse/markov.py` (3),
`braincell/quad/protocol.py` (7), `braincell/quad/__init__.py` (2),
`braincell/__init__.py` (2), `docs/apis/integration.rst` (2), plus tests
in `quad/protocol_test.py` (11), `quad/_util_test.py` (1),
`_single_compartment/base_test.py` (1), `_misc_test.py` (1).

Exports to update: `braincell/quad/__init__.py:50-52,115-116`,
`braincell/quad/protocol.py:39-47` (`__all__`), `braincell/__init__.py`,
`docs/apis/integration.rst`.

Prose to update: the `quad/protocol.py` module docstring (`:16-28`),
the `DiffEqGroupState` docstring (`:194`, `:209`), and the `diffeq_state`
and `hidden_state` docstrings. Under `docs/design/`:
`cell.md:264`, `channel-template-invariants.md:32-33`, and
`interface-map.md:52,477`. The `channel-template-invariants.md` wording
stays accurate as-is under the marker reading — `_bind_state` still
refuses to overwrite an attribute that is not a `DiffEqState` — so that
file needs only the class-list refresh, not a rewrite.

Deliberately **not** touched:

- `dev/legacy/` — legacy tree, out of scope.
- `docs/specs/2026-08-13-*.md` — historical records of what shipped then.
  Rewriting them would falsify the record.

### One correctness tightening

`braincell/_single_compartment/base.py:174` builds `self.V` *outside* the
`state_grouping(False)` scope opened two lines later at `:176`. It is
correct today only because `False` is the contextvar default. Moving the
assignment inside the scope removes the dependence on that default. In
scope because the line is being edited anyway.

## Tests

Rewrite `braincell/quad/protocol_test.py` to cover:

1. **Class-attribute leakage.** `_derivative`/`_diffusion` are shared
   class attributes on the mixin. Writing `a.derivative = x` must create
   an instance attribute and leave `b.derivative is None`. Same for
   `diffusion`. This is the one genuinely new failure mode the design
   introduces.
2. `DiffEqState(value)` raises `TypeError`.
3. `DiffEqState` is not a subclass of `brainstate.State`;
   `DiffEqSingleState` and `DiffEqGroupState` both are.
4. Neither concrete class is a subclass of the other; both are subclasses
   of `DiffEqState`; `brainstate.mixin.Mixin` precedes the storage class
   in both MROs.
5. `state()` returns `DiffEqSingleState` outside `state_grouping`,
   `DiffEqGroupState` inside `state_grouping(True)`, and restores
   correctly across nesting and across an exception raised in the body.
6. `record_state_value_write` still fires on both setters under an active
   state trace — assert via `brainstate` trace machinery, matching how
   the existing suite checks it.
7. `DiffEqGroupState` still rejects `ndim < 2`; `varshape`/`num_state`
   are as documented.
8. `repr` omits `derivative`/`diffusion` while unset. The mechanism
   changes (the key is now absent from `__dict__` rather than present and
   `None`), so the assertion guards rendered output, not the mechanism.

Existing `isinstance(..., DiffEqState)` assertions across
`ion/calcium_test.py`, `ion/_base_test.py`, and `channel/_base_test.py`
stay as they are — that they still pass unmodified is the regression
signal that the marker split preserved solver-visible behaviour.

Full suite must stay green: `pytest braincell/` (2229 passed, 19 skipped
as of `6a832d6`).

## Documentation and changelog

- `changelog.md` — a **breaking change** entry under Unreleased covering
  both the `DiffEqState`-is-now-a-mixin split and the
  `diffeq_state` → `state` rename, with the before/after migration
  snippet.
- `docs/design/cell.md` — update the hidden-state class table.
- `quad/protocol.py` module docstring — describe the three-class shape.
- `DiffEqState` class docstring — state plainly that it is a mixin, that
  it is not a `State`, and that subclassing it alone yields a non-`State`
  object. `braincell.state()` is the recommended allocation path;
  `DiffEqSingleState`/`DiffEqGroupState` remain public for explicit
  construction and for `isinstance` narrowing in user code.

## Out of scope

- Renaming `hidden_state`, or introducing a grouped/ungrouped split in
  its return annotation.
- Any deprecation shim for `DiffEqState(value)`. The break is
  intentional and documented.
- Touching `dev/legacy/`.
