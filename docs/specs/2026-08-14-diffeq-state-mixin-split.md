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

---

# Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `DiffEqState` a marker mixin with two sibling concrete
state classes, and rename the `braincell.diffeq_state` factory to
`braincell.state`.

**Architecture:** `DiffEqState` becomes a `brainstate.mixin.Mixin`
carrying only `derivative`/`diffusion`. `DiffEqSingleState` mixes it into
`brainstate.HiddenState`; `DiffEqGroupState` mixes it into
`brainstate.HiddenGroupState`. Neither concrete class inherits the other,
so the old diamond disappears while every existing
`isinstance(x, DiffEqState)` check keeps its meaning.

**Tech Stack:** Python 3.10+, brainstate, brainunit, JAX >= 0.8.0,
pytest + `unittest.TestCase`.

## Global Constraints

- Branch: `worktree-diffeq-state-mixin-split`. Never commit to `main`
  (AGENTS.md rule 6).
- All physical quantities carry explicit `brainunit` units; bare numbers
  are rejected by `normalize_param()`.
- NumPy-style docstrings on every public class and function, with
  `Examples` wrapped in `.. code-block:: python` and `>>>` prefixes.
- Tests are co-located `*_test.py` siblings. Never `test_*.py`, never a
  `tests/` directory.
- Commit messages must **not** carry a `Co-Authored-By` trailer.
- Full suite baseline at `6a832d6`: 2229 passed, 19 skipped.
- Do not touch `dev/legacy/` or `docs/specs/2026-08-13-*.md`.

## Corrected site counts

A grep during planning found the spec's migration table over-counts by one
file. `braincell/_single_compartment/base_test.py:197,692` and
`braincell/ion/calcium_test.py:207,1783` match `diffeq_state` only inside
**test method names** (`test_init_state_creates_diffeq_state_V`), not
call sites. They need no edit. Renaming those methods is optional polish
and is **not** part of this plan.

Real `diffeq_state` factory sites — 35 across 10 files:
`quad/protocol.py` (7), `quad/protocol_test.py` (11), `channel/_base.py`
(3), `ion/_base.py` (3), `synapse/markov.py` (3), `quad/__init__.py` (2),
`braincell/__init__.py` (2), `docs/apis/integration.rst` (2),
`_misc_test.py` (1), `quad/_util_test.py` (1).

---

### Task 1: Split `DiffEqState` into a mixin plus two sibling classes

Keeps the `diffeq_state` factory name untouched — that rename is Task 2,
so a reviewer can reject one without the other. The suite must be green
at this task's commit.

**Files:**
- Modify: `braincell/quad/protocol.py:16-28` (module docstring),
  `:39-47` (`__all__`), `:50-176` (class), `:179-228` (group class),
  `:288-311` (factory body only)
- Modify: `braincell/quad/__init__.py:49-57`, `:114-120`
- Modify: `braincell/__init__.py:20-27`, `:87-93`
- Modify: `braincell/_single_compartment/base.py:174-177`
- Modify: `braincell/channel/hyperpolarization_activated.py:235-237`
- Test: `braincell/quad/protocol_test.py` (add a class, edit two)
- Test: `braincell/quad/_util_test.py`,
  `braincell/quad/_backward_euler_test.py:54`,
  `braincell/quad/_exp_euler_test.py:106`,
  `braincell/quad/_implicit_test.py:59`,
  `braincell/quad/_runge_kutta_test.py`,
  `braincell/_base_ion_test.py`,
  `braincell/_multi_compartment/cell_test.py`

**Interfaces:**
- Produces: `DiffEqState` (a `brainstate.mixin.Mixin`, **not**
  instantiable, not a `State`); `DiffEqSingleState(DiffEqState,
  brainstate.HiddenState)`; `DiffEqGroupState(DiffEqState,
  brainstate.HiddenGroupState)`. Both concrete classes take
  `(value, **kwargs)` exactly as before. `diffeq_state(value, **kwargs)`
  keeps its name and now returns `DiffEqSingleState` when ungrouped.

- [ ] **Step 1: Write the failing tests**

Add this class to `braincell/quad/protocol_test.py`, immediately after
the imports and before `class DiffEqStateTest`. Add
`DiffEqSingleState` to the `from braincell.quad.protocol import (...)`
block at `:36-44`.

```python
class DiffEqStateMixinTest(unittest.TestCase):
    """``DiffEqState`` is the integrability marker, not a state class."""

    def test_the_marker_is_not_instantiable(self):
        with self.assertRaises(TypeError):
            DiffEqState(jnp.zeros(3) * u.mV)

    def test_the_marker_is_a_mixin_and_not_a_state(self):
        self.assertTrue(issubclass(DiffEqState, brainstate.mixin.Mixin))
        self.assertFalse(issubclass(DiffEqState, brainstate.State))

    def test_both_concrete_classes_are_states_carrying_the_marker(self):
        for cls in (DiffEqSingleState, DiffEqGroupState):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, DiffEqState))
                self.assertTrue(issubclass(cls, brainstate.State))

    def test_the_concrete_classes_are_siblings_not_a_chain(self):
        # The old hierarchy had DiffEqGroupState inherit DiffEqState, which
        # placed the ungrouped class ahead of HiddenGroupState in the MRO
        # so any method added to it would silently shadow the grouped one.
        self.assertFalse(issubclass(DiffEqGroupState, DiffEqSingleState))
        self.assertFalse(issubclass(DiffEqSingleState, DiffEqGroupState))

    def test_the_marker_precedes_the_storage_class_in_both_mros(self):
        for cls, storage in (
            (DiffEqSingleState, "HiddenState"),
            (DiffEqGroupState, "HiddenGroupState"),
        ):
            with self.subTest(cls=cls.__name__):
                names = [c.__name__ for c in cls.__mro__]
                self.assertLess(names.index("DiffEqState"), names.index(storage))

    def test_derivative_defaults_do_not_leak_between_instances(self):
        # ``_derivative``/``_diffusion`` are class attributes now that the
        # mixin has no ``__init__``, so the setters must shadow them per
        # instance rather than mutate the shared class attribute.
        a = DiffEqSingleState(jnp.zeros(3) * u.mV)
        b = DiffEqSingleState(jnp.zeros(3) * u.mV)
        a.derivative = jnp.ones(3) * (u.mV / u.ms)
        a.diffusion = jnp.ones(3) * (u.mV / u.ms)
        self.assertIsNone(b.derivative)
        self.assertIsNone(b.diffusion)
        self.assertIsNone(DiffEqState._derivative)
        self.assertIsNone(DiffEqState._diffusion)

    def test_setters_still_record_a_state_write(self):
        # The exponential-Euler and Runge-Kutta drivers discover which
        # states participate by watching this trace.
        st = DiffEqSingleState(jnp.zeros(3) * u.mV)
        with brainstate.StateTraceStack() as trace:
            st.derivative = jnp.ones(3) * (u.mV / u.ms)
        self.assertTrue(any(s is st for s in trace.get_write_states()))

    def test_repr_hides_derivative_until_it_is_written(self):
        st = DiffEqSingleState(jnp.zeros(3) * u.mV)
        self.assertNotIn("derivative", repr(st))
        st.derivative = jnp.ones(3) * (u.mV / u.ms)
        self.assertIn("derivative", repr(st))
```

Then retarget the three existing `DiffEqStateTest` methods at
`protocol_test.py:49-66` and the `IndependentIntegrationTest` helper at
`:217` from `DiffEqState(...)` to `DiffEqSingleState(...)`, and rename
the class `DiffEqStateTest` → `DiffEqSingleStateTest`.

Finally, tighten `StateFactoryTest` at `:144-170`. It currently asserts
the ungrouped case negatively (`assertNotIsInstance(..., DiffEqGroupState)`),
which would still pass if the factory returned any non-grouped object.
Now that a named ungrouped class exists, assert it positively:

```python
    def test_default_scope_is_not_grouped(self):
        self.assertIsInstance(diffeq_state(jnp.zeros(3) * u.mV), DiffEqSingleState)
        self.assertNotIsInstance(hidden_state(jnp.zeros(3) * u.mV), brainstate.HiddenGroupState)

    def test_scopes_nest_and_restore(self):
        with state_grouping(True):
            self.assertIsInstance(diffeq_state(jnp.zeros((1, 3)) * u.mV), DiffEqGroupState)
            with state_grouping(False):
                self.assertIsInstance(diffeq_state(jnp.zeros(3) * u.mV), DiffEqSingleState)
            self.assertIsInstance(diffeq_state(jnp.zeros((1, 3)) * u.mV), DiffEqGroupState)
        self.assertIsInstance(diffeq_state(jnp.zeros(3) * u.mV), DiffEqSingleState)

    def test_scope_is_restored_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with state_grouping(True):
                raise RuntimeError("boom")
        self.assertIsInstance(diffeq_state(jnp.zeros(3) * u.mV), DiffEqSingleState)
```

Leave `test_scope_does_not_leak_across_threads` and
`test_kwargs_are_forwarded_to_the_state_constructor` as they are.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest braincell/quad/protocol_test.py::DiffEqStateMixinTest -v`
Expected: FAIL — `ImportError: cannot import name 'DiffEqSingleState'`.

- [ ] **Step 3: Rewrite the classes in `braincell/quad/protocol.py`**

Replace `class DiffEqState(brainstate.HiddenState):` at `:50` through the
end of `class DiffEqGroupState` at `:228` with the following. The
`derivative`/`diffusion` property bodies and `__pretty_repr_item__` are
carried over verbatim; `__init__` is **deleted**.

```python
class DiffEqState(brainstate.mixin.Mixin):
    """Marker mixin for a state that participates in numerical integration.

    A :class:`DiffEqState` is the unit of work consumed by every solver in
    :mod:`braincell.quad`. It contributes two slots — ``derivative`` and
    ``diffusion`` — that the surrounding solver writes during one ODE/SDE
    step:

    - ``derivative`` is the right-hand side :math:`f(t, y)` for an ODE
      :math:`\\dot y = f(t, y)`, or the *drift* term for an SDE
      :math:`dy = f(t, y)\\,dt + g(t, y)\\,dW`.
    - ``diffusion`` is the SDE noise coefficient :math:`g(t, y)`. It
      stays ``None`` for plain ODE systems.

    This class is a :class:`brainstate.mixin.Mixin`, **not** a
    :class:`brainstate.State`. It carries no storage and cannot be
    instantiated; ``DiffEqState(value)`` raises :exc:`TypeError`. Mixing
    it into something that is not a :class:`brainstate.State` yields an
    object no solver will accept. Use :func:`diffeq_state` to allocate,
    or name :class:`DiffEqSingleState` / :class:`DiffEqGroupState`
    explicitly.

    Separating the marker from the storage layout is what lets the two
    concrete classes be siblings. Solver state selection goes through
    ``isinstance(value, DiffEqState)``
    (see :func:`braincell.quad._util.split_diffeq_states`), which stays
    ``True`` for both.

    Attributes
    ----------
    derivative : brainstate.typing.PyTree
        Time derivative (or SDE drift) of the state. Set inside
        :meth:`DiffEqModule.compute_derivative`. Must carry units that
        satisfy ``unit(derivative) * unit(dt) == unit(value)``.
    diffusion : brainstate.typing.PyTree
        Optional SDE diffusion coefficient. ``None`` denotes a
        deterministic ODE system.

    See Also
    --------
    DiffEqSingleState : Concrete ungrouped state, used by
        ``SingleCompartment``.
    DiffEqGroupState : Concrete grouped state, used by ``Cell``.
    DiffEqModule : Container that owns and updates the states.
    """

    __module__ = 'braincell'

    #: Class-level defaults in place of ``__init__``: a brainstate
    #: ``Mixin`` must not define one. The property setters below shadow
    #: these per instance on first write.
    _derivative = None
    _diffusion = None

    @property
    def derivative(self):
        """
        Get the derivative of the state.

        Returns
        -------
        brainstate.typing.PyTree
            The derivative of the state, used to compute the derivative of the ODE system
            or the drift of the SDE system.
        """
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        """
        Set the derivative of the state.

        Parameters
        ----------
        value : brainstate.typing.PyTree
            The new value for the derivative of the state.
        """
        record_state_value_write(self)
        self._derivative = value

    @property
    def diffusion(self):
        """
        Get the diffusion of the state.

        Returns
        -------
        brainstate.typing.PyTree
            The diffusion of the state, used to compute the diffusion of the SDE system.
            If it is None, the system is considered as an ODE system.
        """
        return self._diffusion

    @diffusion.setter
    def diffusion(self, value):
        """
        Set the diffusion of the state.

        Parameters
        ----------
        value : brainstate.typing.PyTree
            The new value for the diffusion of the state.
        """
        record_state_value_write(self)
        self._diffusion = value

    def __pretty_repr_item__(self, k, v):
        if k == '_derivative':
            if self._derivative is not None:
                return 'derivative', self._derivative
            else:
                return None
        if k == '_diffusion':
            if self._diffusion is not None:
                return 'diffusion', self._diffusion
            else:
                return None
        return super().__pretty_repr_item__(k, v)


class DiffEqSingleState(DiffEqState, brainstate.HiddenState):
    """An integrable hidden state with no trailing state axis.

    This is the state class used by every hidden variable owned by a
    :class:`braincell.SingleCompartment`, which has no spatial axis: one
    value per element of ``varshape``, and nothing to group.

    See Also
    --------
    DiffEqGroupState : The grouped counterpart used by ``Cell``.
    diffeq_state : Host-scoped factory that picks between the two.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> import braincell
        >>> state = braincell.DiffEqSingleState(np.zeros(4) * u.mV)
        >>> state.varshape
        (4,)
        >>> isinstance(state, braincell.DiffEqState)
        True
    """

    __module__ = 'braincell'


class DiffEqGroupState(DiffEqState, brainstate.HiddenGroupState):
    """An integrable hidden state whose trailing axis indexes independent states.

    This is the state class used by every hidden variable owned by a
    :class:`braincell.Cell`. A ``Cell`` is a *spatial* model: its runtime
    arrays are shaped ``pop_size + (n_cv,)`` for voltage and
    ``pop_size + (n_point,)`` for mechanism variables, so the trailing
    axis enumerates compartments (or points) that evolve independently.
    That is exactly the contract of
    :class:`brainstate.HiddenGroupState` — ``varshape`` is everything but
    the last axis and ``num_state`` is the last axis — which lets an
    eligibility-trace learner treat one array as ``num_state`` separately
    traced hidden units.

    Notes
    -----
    :class:`brainstate.HiddenGroupState` requires ``value.ndim >= 2``.
    This is why :class:`braincell.Cell` makes its population axis
    mandatory (``pop_size`` defaults to ``1`` and may not be empty) — the
    validation is inherited unmodified rather than relaxed.

    See Also
    --------
    DiffEqSingleState : The ungrouped counterpart used by
        ``SingleCompartment``.
    diffeq_state : Host-scoped factory that picks between the two.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> import braincell
        >>> state = braincell.DiffEqGroupState(np.zeros((1, 4)) * u.mV)
        >>> state.varshape
        (1,)
        >>> state.num_state
        4
        >>> isinstance(state, braincell.DiffEqState)
        True
    """

    __module__ = 'braincell'
```

- [ ] **Step 4: Point the factory at the new ungrouped class**

In `braincell/quad/protocol.py`, change the body of `diffeq_state` at
`:310` from:

```python
    cls = DiffEqGroupState if _STATE_GROUPING.get() else DiffEqState
```

to:

```python
    cls = DiffEqGroupState if _STATE_GROUPING.get() else DiffEqSingleState
```

and update its `Returns` block at `:299-303` to read:

```
    Returns
    -------
    DiffEqState
        A :class:`DiffEqGroupState` inside :func:`state_grouping`
        (i.e. within a :class:`braincell.Cell`), otherwise a
        :class:`DiffEqSingleState`.
```

- [ ] **Step 5: Update `__all__` and the module docstring**

In `braincell/quad/protocol.py`, add `'DiffEqSingleState',` to `__all__`
at `:39-47`, immediately after `'DiffEqState',`. Replace the module
docstring paragraph at `:18-22` with:

```
Defines the state classes solvers consume — the :class:`DiffEqState`
marker mixin and its two concrete carriers :class:`DiffEqSingleState`
and :class:`DiffEqGroupState` — the :class:`DiffEqModule` mixin that
declares a module integrable, and the host-scoped factory that chooses
between the grouped and ungrouped classes.
```

- [ ] **Step 6: Re-export the new class**

In `braincell/quad/__init__.py`, add `DiffEqSingleState,` to the
`from .protocol import (...)` block at `:49-57` and
`'DiffEqSingleState',` to `__all__` at `:114-120`. Make the same two
edits in `braincell/__init__.py` at `:20-27` and `:87-93`.

- [ ] **Step 7: Migrate the two library construction sites**

In `braincell/_single_compartment/base.py`, replace `:174-177`:

```python
        self.V = DiffEqState(braintools.init.param(self.V_initializer, self.varshape, batch_size))
        self.spike = brainstate.ShortTermState(_zero_spike_like(self.V.value))
        with state_grouping(False):
            super().init_state(batch_size)
```

with:

```python
        with state_grouping(False):
            self.V = diffeq_state(braintools.init.param(self.V_initializer, self.varshape, batch_size))
            self.spike = brainstate.ShortTermState(_zero_spike_like(self.V.value))
            super().init_state(batch_size)
```

This also fixes a latent fragility: `self.V` was built *outside* the
scope and was correct only because `False` is the contextvar default.
Update the import at `:26` from `DiffEqState` to `diffeq_state`.

In `braincell/channel/hyperpolarization_activated.py:235-237`, the three
hits are commented-out code; change `DiffEqState(` to `diffeq_state(` in
each so the comment does not teach a broken call.

- [ ] **Step 8: Migrate the remaining test construction sites**

These files construct an ungrouped state directly. Change the import and
the call in each:

```bash
sed -i 's/\bDiffEqState(/DiffEqSingleState(/g; s/^\(\s*\)DiffEqState,$/\1DiffEqSingleState,/' \
  braincell/quad/_util_test.py \
  braincell/quad/_backward_euler_test.py \
  braincell/quad/_exp_euler_test.py \
  braincell/quad/_implicit_test.py \
  braincell/quad/_runge_kutta_test.py \
  braincell/_base_ion_test.py \
  braincell/_multi_compartment/cell_test.py
```

A blanket `sed` is safe across exactly these seven files: a planning grep
confirmed **none** of them uses `DiffEqState` inside `isinstance`,
`assertIsInstance`, or `assertNotIsInstance`, so every occurrence is a
construction. Confirm that assumption still holds before trusting it:

Run: `grep -rn "sinstance(.*DiffEqState" braincell/quad/_util_test.py braincell/quad/_backward_euler_test.py braincell/quad/_exp_euler_test.py braincell/quad/_implicit_test.py braincell/quad/_runge_kutta_test.py braincell/_base_ion_test.py braincell/_multi_compartment/cell_test.py`
Expected: no output (exit 1). If it prints anything, restore the marker
name at those lines by hand and import both symbols.

The files that *do* assert on the marker — `ion/calcium_test.py`,
`ion/_base_test.py`, `channel/_base_test.py` — are deliberately absent
from the `sed` list and must stay unedited.

- [ ] **Step 9: Run the new tests**

Run: `pytest braincell/quad/protocol_test.py -v`
Expected: PASS, including all eight `DiffEqStateMixinTest` cases.

- [ ] **Step 10: Run the full suite**

Run: `pytest braincell/ -q`
Expected: 2229+ passed, 19 skipped, 0 failed. The ~20 untouched
`assertIsInstance(..., DiffEqState)` assertions in `ion/calcium_test.py`,
`ion/_base_test.py`, and `channel/_base_test.py` passing unmodified is
the regression signal that the marker split preserved solver-visible
behaviour.

- [ ] **Step 11: Commit**

```bash
git add braincell/
git commit -m "refactor(quad): make DiffEqState a marker mixin

DiffEqState carried two jobs: the isinstance marker every solver uses to
select integrable states, and the concrete ungrouped state class. That
forced DiffEqGroupState to inherit it, putting the ungrouped class ahead
of HiddenGroupState in the MRO where any added method would silently
shadow the grouped one.

Split them: DiffEqState is now a brainstate Mixin holding only
derivative/diffusion, with DiffEqSingleState and DiffEqGroupState as
siblings over HiddenState and HiddenGroupState. Every existing
isinstance(x, DiffEqState) check keeps its meaning and is unedited.

BREAKING: DiffEqState(value) now raises TypeError."
```

---

### Task 2: Rename `diffeq_state` to `state`

**Files:**
- Modify: `braincell/quad/protocol.py` (7 sites), `braincell/quad/__init__.py` (2),
  `braincell/__init__.py` (2), `braincell/channel/_base.py` (3),
  `braincell/ion/_base.py` (3), `braincell/synapse/markov.py` (3)
- Test: `braincell/quad/protocol_test.py` (11),
  `braincell/quad/_util_test.py` (1), `braincell/_misc_test.py:77` (1)

**Interfaces:**
- Consumes: `DiffEqSingleState` / `DiffEqGroupState` from Task 1.
- Produces: `braincell.state(value, **kwargs) -> DiffEqState`. The name
  `diffeq_state` no longer exists anywhere — no alias, no shim.

- [ ] **Step 1: Update the name-registration test first**

`braincell/_misc_test.py:77` asserts the public factory names literally.
Change:

```python
        for name in ("state_grouping", "diffeq_state", "hidden_state"):
```

to:

```python
        for name in ("state_grouping", "state", "hidden_state"):
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest braincell/_misc_test.py -v -k module`
Expected: FAIL with `AttributeError: module 'braincell' has no attribute 'state'`.

- [ ] **Step 3: Rename the function and every call site**

The token `diffeq_state` also appears as a substring inside
`diffeq_states`, `split_diffeq_states`, `_check_diffeq_state_derivative`,
`diffeq_state_key`, `diffeq_state_val`, and several test *method* names.
A word-boundary match alone is not enough — `diffeq_states` would match
`\bdiffeq_state` — so anchor on the trailing boundary too:

```bash
sed -i 's/\bdiffeq_state\b/state/g' \
  braincell/quad/protocol.py \
  braincell/quad/__init__.py \
  braincell/__init__.py \
  braincell/channel/_base.py \
  braincell/ion/_base.py \
  braincell/synapse/markov.py \
  braincell/quad/protocol_test.py \
  braincell/quad/_util_test.py
```

- [ ] **Step 4: Verify no collateral damage**

Run: `grep -rn "\bstates\b\|split_state\|_check_state_derivative" braincell/quad/_util.py | head`
Expected: `split_diffeq_states` and `_check_diffeq_state_derivative` are
**unchanged** — `_util.py` was not in the sed list and those identifiers
keep their names.

Run: `grep -rn "diffeq_state" braincell/ | grep -v "diffeq_states\|split_diffeq\|_check_diffeq_state\|diffeq_state_key\|diffeq_state_val\|def test_"`
Expected: no output.

- [ ] **Step 5: Check the rename did not shadow a local variable**

`sed` renames the imported symbol to `state`, which is a common local
name. Verify no function now shadows it:

Run: `grep -n "state = \|state=" braincell/channel/_base.py braincell/ion/_base.py braincell/synapse/markov.py`
Expected: review each hit; if a local named `state` exists in a function
that also calls `state(...)`, rename that local to `st`.

- [ ] **Step 6: Fix the docstrings the rename touched**

In `braincell/quad/protocol.py`, the `state_grouping` and `hidden_state`
docstrings cross-reference the factory. Confirm the `See Also` entries
now read `state : Allocate an integrable hidden state under this scope.`
and that the `state_grouping` doctest at `:275-278` reads:

```
        >>> with braincell.state_grouping(True):
        ...     st = braincell.state(np.zeros((1, 4)) * u.mV)
        >>> type(st).__name__
        'DiffEqGroupState'
```

The example variable is renamed `state` → `st` because `state` is now the
function being called.

- [ ] **Step 7: Run the full suite**

Run: `pytest braincell/ -q`
Expected: 2229+ passed, 19 skipped, 0 failed.

- [ ] **Step 8: Commit**

```bash
git add braincell/
git commit -m "refactor(quad)!: rename braincell.diffeq_state to braincell.state

BREAKING: braincell.diffeq_state is gone with no alias. Call
braincell.state(value) instead."
```

---

### Task 3: Migrate documentation, notebooks, and examples

**Files:**
- Modify: `docs/apis/integration.rst`, `docs/design/cell.md:264`,
  `docs/design/channel-template-invariants.md:32-33`,
  `docs/design/interface-map.md:52,477`
- Modify: `docs/tutorials/channel.ipynb` (8),
  `docs/integration/diffeq.ipynb` (3),
  `docs/integration/advanced.ipynb` (2),
  `docs/integration/solvers.ipynb` (1),
  `examples/multi_compartment/quad.ipynb` (5)
- Modify: `examples/single_compartment/SC01_fitting_a_hh_neuron.py` (3),
  `examples/convert_mod/nmodl/templates/one_ion_hh_ohmic.py` (1),
  `examples/convert_mod/nmodl/templates/density_channel.py` (1),
  `examples/convert_mod/nmodl/examples/artifacts/kv/rendered_channel.py` (1)
- Modify: `changelog.md`

**Interfaces:**
- Consumes: `braincell.state`, `braincell.DiffEqSingleState`,
  `braincell.DiffEqGroupState` from Tasks 1 and 2.

- [ ] **Step 1: Rewrite user-facing construction calls**

Every `DiffEqState(...)` in a notebook, example, or tutorial is user-facing
code that should demonstrate the recommended path, so it becomes
`braincell.state(...)` — **not** `DiffEqSingleState(...)`. Match the
surrounding import style in each file (`braincell.state` vs a bare
`state` from an existing `from braincell import ...`).

```bash
grep -rln "DiffEqState(\|diffeq_state" docs/ examples/ \
  --include=*.ipynb --include=*.py --include=*.rst \
  | grep -v "^docs/specs/"
```

Work through each file from that list.

- [ ] **Step 2: Update the API reference**

In `docs/apis/integration.rst`, add `DiffEqSingleState` to the class
listing next to `DiffEqState` and `DiffEqGroupState`, and rename the two
`diffeq_state` entries to `state`.

- [ ] **Step 3: Update the design docs**

- `docs/design/cell.md:264` — the line currently ends
  "`SingleCompartment` 无空间轴，仍用普通 `DiffEqState`。" Change the
  trailing class name to `DiffEqSingleState`.
- `docs/design/interface-map.md:52` and `:477` — add `DiffEqSingleState`
  to the protocol class lists.
- `docs/design/channel-template-invariants.md:32-33` — the wording
  ("`_bind_state` refuses to overwrite an attribute that is not already a
  `DiffEqState`") stays **accurate** under the marker reading. Leave the
  prose; only confirm no code sample there constructs a `DiffEqState`.

- [ ] **Step 4: Write the changelog entry**

Add to the `Unreleased` section of `changelog.md`, under a
`### Breaking changes` heading (create it if absent):

```markdown
- `DiffEqState` is now a marker mixin rather than a concrete state class.
  `DiffEqState(value)` raises `TypeError`. The two concrete classes are
  `DiffEqSingleState` (over `brainstate.HiddenState`, used by
  `SingleCompartment`) and `DiffEqGroupState` (over
  `brainstate.HiddenGroupState`, used by `Cell`). Every
  `isinstance(x, DiffEqState)` check keeps working unchanged.
- `braincell.diffeq_state` is renamed to `braincell.state`. No alias is
  provided.

  ```python
  # before
  self.m = braincell.DiffEqState(init)

  # after (preferred — picks the right class for the host)
  self.m = braincell.state(init)

  # after (explicit)
  self.m = braincell.DiffEqSingleState(init)
  ```
```

- [ ] **Step 5: Verify nothing user-facing still names the old API**

Run:

```bash
grep -rn "DiffEqState(\|diffeq_state" docs/ examples/ \
  --include=*.ipynb --include=*.py --include=*.rst --include=*.md \
  | grep -v "^docs/specs/2026-08-13"
```

Expected: only `docs/specs/2026-08-14-diffeq-state-mixin-split.md` (this
file, which documents the old names on purpose) and the `changelog.md`
migration snippet.

- [ ] **Step 6: Verify the executable examples still run**

Run: `python examples/single_compartment/SC01_fitting_a_hh_neuron.py`
Expected: runs to completion without `AttributeError` or `TypeError`.
If it is long-running, interrupt once past model construction — the
construction path is what this change touches.

- [ ] **Step 7: Run the full suite and pre-commit**

Run: `pytest braincell/ -q && pre-commit run --all-files`
Expected: 2229+ passed, 19 skipped, 0 failed; pre-commit clean.

- [ ] **Step 8: Commit**

```bash
git add docs/ examples/ changelog.md
git commit -m "docs: migrate to the DiffEqState mixin split and braincell.state

Updates the API reference, design docs, tutorials, notebooks, and
examples to the three-class protocol and the renamed factory, and records
both breaking changes in the changelog."
```
