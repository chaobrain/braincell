---
myst:
  heading_anchors: 2
---

# Code Conventions

The rules that every change to `braincell/` follows, with the details behind the one-line versions in the repository's `AGENTS.md`. Checks that pre-commit enforces are marked *(checked)*.

## Physical units

Every physical quantity carries an explicit [`brainunit`](https://github.com/chaobrain/brainunit) unit: a Python number, NumPy array or JAX array multiplied by a unit. `normalize_param()` (in `braincell/_misc.py`) rejects a bare number with `TypeError`, so never pass a unitless value where a quantity is expected.

`brainunit` provides SI units with the usual prefixes: m (milli), u (micro), n (nano), p (pico), k (kilo), M (mega).

| Category | Quantity | Units |
| --- | --- | --- |
| Electrical | Voltage | `u.V`, `u.mV` |
| Electrical | Current | `u.A`, `u.mA`, `u.uA`, `u.nA`, `u.pA` |
| Electrical | Conductance | `u.S`, `u.mS`, `u.uS`, `u.nS`, `u.pS` |
| Electrical | Resistance | `u.ohm`, `u.kohm`, `u.Mohm` |
| Electrical | Capacitance | `u.F`, `u.uF`, `u.nF`, `u.pF` |
| Space and time | Length | `u.m`, `u.cm`, `u.mm`, `u.um` |
| Space and time | Time | `u.s`, `u.ms` |
| Substance and temperature | Molar concentration | `u.M`, `u.mM` |
| Substance and temperature | Temperature | `u.kelvin`, `u.celsius` |

```python
import numpy as np
import jax.numpy as jnp
import brainunit as u

v_rest = -65.0 * u.mV                               # scalar
branch_lengths = np.array([10.5, 20.0, 15.3]) * u.um  # array
xyz = jnp.zeros((10, 3)) * u.um

v_total = -60.0 * u.mV + 0.01 * u.V                 # -50 mV: units align on addition
area = 2 * np.pi * (1.0 * u.um) * (10.0 * u.um)     # um**2
cm = 1.0 * u.uF / u.cm**2                           # a density
current = (50 * u.nS / u.cm**2) * (-65 * u.mV)      # the current dimension is derived
```

To leave the unit system, use `u.get_unit(a)` and `u.get_mantissa(a)`, `a.to_decimal(u.V)` (a raw float in the target unit), `a / u.mV` (divide the unit out), or `a.to(u.V)` (convert and keep the unit).

## Random numbers

Use `brainstate.random`, never `jax.random` directly, so that seeding and key handling stay in one place.

## Compiled loops

Never drive a model that runs repeatedly with a bare Python `for` or `while` loop. A Python loop executes op by op, with dispatch overhead and no fusion, and traces again every step. The `brainstate.transform` primitives lower the whole loop into one compiled XLA program and trace the body once:

| Shape of the work | Use |
| --- | --- |
| One step or a one-shot call | `brainstate.transform.jit` |
| Many steps, collect the outputs | `brainstate.transform.for_loop`: repeats a step `length` times or maps over `xs`; `State` is carried automatically and outputs are stacked |
| Many steps with an explicit carry | `brainstate.transform.scan`: `f(carry, x) -> (carry, y)` alongside `State` |
| A long rollout under reverse-mode autograd | `brainstate.transform.checkpointed_for_loop` / `checkpointed_scan`: same semantics, but activations are recomputed on the backward pass (tune `base`) to bound peak memory |

Compose them freely, for example a `jit` driver that calls a `for_loop`. Use the checkpointed variants only when gradients through a long simulation would otherwise exhaust memory.

## JAX versions

Support every JAX release from the floor in `pyproject.toml` (`jax>=…`) to the latest. CI tests the floor and the latest release (the `jax-version` matrix in `.github/workflows/CI.yml`). Prefer feature or shape detection to version checks.

## Imports and package layout

- Internal modules use absolute imports: `from braincell.morph import MorphoBranch`, not `from .morph import MorphoBranch`.
- Public names are re-exported through `braincell/__init__.py`; `set_module_as('braincell')` sets the module a public function reports.
- Internal packages (`_compute`, `_discretization`, `_multi_compartment`, `_single_compartment`) and internal top-level modules (`_base.py`, `_base_channel.py`, `_base_ion.py`, `_misc.py`, `_typing.py`) carry a leading underscore because their paths are not public API. The modules inside those packages (`cell.py`, `runtime.py`, `geometry.py`, …) carry none: sibling internal code imports them.
- The public subpackages carry no underscore: `channel`, `filter`, `io`, `ion`, `mech`, `morph`, `network`, `quad`, `reduction`, `synapse`, `trainable`, `vis`. Reusable experimental interfaces live in `braincell.experimental`; usage workflows, numerical validation and performance drivers stay outside the installed package (`examples/`, `validation/`, `benchmarks/`).

## Type aliases

Annotate with the shared aliases in `braincell/_typing.py`: write `g_max: Initializer`, never `g_max: Union[brainstate.typing.ArrayLike, Callable]`.

- Import them absolutely with the other `braincell.*` imports: `from braincell._typing import Initializer, Size`.
- Optional parameters use `Optional[Initializer]`, not a hand-rolled `Union[..., None]`.
- Never re-import a type that an alias already covers (`from brainstate.typing import ArrayLike`); go through `_typing.py`.
- A type expression repeated across modules belongs in `_typing.py`.
- Substitute an alias only for the same concept: a `Hashable` that is not a section name, or a `tuple[str, ...]` that is not a state path, stays raw.
- Aliases are for annotations only. `_typing.py` is not public API, so docstrings keep the importable qualified name (`size : brainstate.typing.Size`) that Sphinx can cross-reference.

| Alias | Stands for | Use for |
| --- | --- | --- |
| `Initializer` | `Union[ArrayLike, Callable]` | a parameter accepting a value or a callable that produces one |
| `ArrayLike` | `brainstate.typing.ArrayLike` | scalar and array parameters |
| `Size` | `brainstate.typing.Size` | a shape parameter: `size`, `pop_size`, and similar |
| `PyTree` | `brainstate.typing.PyTree` | nested state passed through an integrator |
| `SectionName` | `Hashable` | a morphology section identifier |
| `T`, `DT` | `u.Quantity[u.second]` | simulation time and time step |
| `Path` | `Tuple[str, ...]` | a state path within a model tree |
| `VectorField`, `Y0`, `Y1`, `Jacobian`, `Args`, `Aux` | see `_typing.py` | integrator and solver signatures in `braincell/quad` |

## Docstrings

Public classes, methods and functions use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html), with sections in this order:

1. **Short summary**: one imperative line.
2. **Extended summary** (optional), after a blank line.
3. **Parameters**: `name : type` on its own line, the description indented below.
4. **Returns** / **Yields**: the same format.
5. **Raises**: the exception and when it is raised.
6. **See Also**.
7. **Notes**: implementation details, equations.
8. **References**: numbered entries (`.. [1]`), including the sources of a model's equations.
9. **Examples**.

Examples are runnable and doctestable:

- wrap them in a `.. code-block:: python` directive so that Sphinx highlights them;
- prefix every input line with `>>>` (continuation lines with `...`), and put the expected output on the next line without a prefix;
- separate scenarios with a blank `>>>` line;
- start with every import the block needs (`import brainunit as u`, …).

## Deprecation

A change that removes or renames a public name, parameter or accepted value, or changes what an existing call returns or raises, first ships a deprecation for at least one minor release; the next minor release may then remove the old form ([contributing](contributing.md#breaking-changes)). Names with a leading underscore change without one.

### Declare it

Each deprecation is a module-level `Deprecation` constant from the private module `braincell._deprecation`, beside the code it deprecates; delete it with that code. A function, method or class calls `warn()` first thing in its body (a class in the `__init__` each instance runs once, a dataclass in `__post_init__`):

```python
from braincell._deprecation import Deprecation, deprecated_attributes

# A class (a function or method is the same).
_OLD_MODEL = Deprecation("braincell.OldModel", since="0.4", removed_in="0.5", replacement="braincell.NewModel")

class OldModel(NewModel):
    def __init__(self, size, **kwargs):
        _OLD_MODEL.warn()
        super().__init__(size, **kwargs)

# A module attribute, served by the module's __getattr__ and kept out of __all__.
__getattr__ = deprecated_attributes(__name__, {
    "old_name": (
        Deprecation("braincell.old_name", since="0.4", removed_in="0.5", replacement="braincell.new_name"),
        new_name,
    ),
})

# A parameter or an accepted value.
_OLD_PARAMETER = Deprecation("Model(old_name=...)", since="0.4", removed_in="0.5",
                             replacement="Model(new_name=...)")

def __init__(self, *, new_name=None, old_name=None):
    if old_name is not None:
        _OLD_PARAMETER.warn()
        new_name = old_name
```

- `name` is what the user wrote, as they would search for it (`braincell.OldModel`, `braincell.Cell.old_method()`, `Model(old_name=...)`); the changelog entry quotes it verbatim.
- `since` is the `"MAJOR.MINOR"` release whose changelog first announces the deprecation, normally the next minor release; `removed_in` is a later one, by default the minor release after `since`.
- `replacement` is required; pass `None` only with a `note` that says what to do instead.
- Construction raises on a malformed version, a `removed_in` not after `since`, a missing replacement, another warning category, or a `name` registered twice with different fields, so a mistake fails `import braincell`.

### Category and message

- `DeprecationWarning` (the default): a name, parameter or value is removed or renamed.
- `FutureWarning`: the same call will return something else or raise. Pass the details as `warn(message)`; the message must name `braincell <removed_in>`, as the MECH-26 checks do: `... This will raise ValueError at declaration from braincell 0.4.`

The standard message is

```text
<name> is deprecated since braincell <since> and will be removed in braincell <removed_in>. Use <replacement> instead.[ <note>]
```

Python shows a `DeprecationWarning` by default only when it is attributed to `__main__` (a script or a notebook cell), and pytest shows it in its summary; a warning attributed to a line inside braincell is never seen. So never pass `stacklevel`: the helper finds the line.

- `warn()` (`at="caller"`) attributes the warning to the caller of the function whose body calls it. When that caller is braincell's own code (not a `*_test.py`), nothing is issued: braincell rebuilding a deprecated dataclass with `dataclasses.replace`, or constructing a private subclass of a deprecated class, is not the user's to fix. A braincell wrapper between the user and the deprecated callable counts as braincell's own code too; use `at="user"` there.
- `warn(at="user")` attributes it to the nearest frame outside braincell, for a check that runs deep inside braincell or a value that may arrive through other braincell calls.
- Both skip the frames that call code on someone else's behalf: brainstate's node metaclass `__call__`, a dataclass-generated `__init__`, `dataclasses` (`replace`) and the import machinery. `from braincell import X` for a `deprecated_attributes` name warns once, at the import line.
- Every call warns; nothing records "already warned". The warning filters decide what is shown (the default filter shows each line once; `always` and `error` see every call). Each `warn()` costs about 1.3 µs, so keep deprecated paths out of per-step code.

### `__all__` and the API reference

- A name served by `deprecated_attributes` leaves `__all__` and the API reference when it is deprecated; `from module import *` then no longer binds it, which the changelog entry says.
- A class or function still defined in its module stays in `__all__` and the API reference. Its docstring gets `.. deprecated:: <since>` after the summary line, naming the replacement and the removal release.
- A removed module attribute keeps its name as a tombstone that says what to use instead: `__getattr__ = removed_attributes(__name__, {"old_name": "braincell.old_name was removed in braincell 0.5. Use braincell.new_name."})`. Access raises `AttributeError` with the message, and `from braincell import old_name` raises `ImportError` with it (Python would turn a plain `AttributeError` there into its generic "cannot import name"). A removed method does the same in its class body: `old_method = removed_member("braincell.Cell.old_method() was removed in braincell 0.5. Use ...")`.

### Tests

- Each deprecation has a test in its module's `*_test.py` that asserts the category; that the message contains the replacement and `removed in braincell <removed_in>`; that the warning is attributed to the test (`cm.filename == __file__`); and that the old form still gives the same result.
- Any other test that uses a deprecated API on purpose wraps the call in `assertWarns`, or silences it locally with `warnings.catch_warnings()` and `warnings.simplefilter("ignore", DeprecationWarning)`.
- `filterwarnings` in `[tool.pytest]` (`pyproject.toml`) turns braincell's `DeprecationWarning`s (`... is deprecated since braincell ...`) and `FutureWarning`s that name a `braincell X.Y` release into errors, so a test that uses a deprecated API without declaring it fails. A command-line `-W` cannot stand in for these rules: pytest escapes its message regex.
- A test that constructs a `Deprecation` restores the registry afterwards, e.g. with `mock.patch.dict(braincell._deprecation._REGISTRY)`.

### Changelog

- **Deprecating:** under `### Deprecations` in `## Unreleased`, ``- **`<name>`** (<ID>) is deprecated since <since>; removal planned for <removed_in>. Use `<replacement>`.``, with a migration example when the replacement is not one-to-one. The entry must contain `name` verbatim *(checked)*.
- **Postponing:** an entry under `### Deprecations` in `## Unreleased` with the new `removed_in` and the reason.
- **Removing:** under `### Breaking Changes`, ``- `<name>` (deprecated since <since>) is removed; use `<replacement>`.``

### Expiry

`braincell/_deprecation/__init___test.py` imports every braincell module and fails when fewer deprecations register than it knows of, when a `removed_in` is not later than the minor release of `__version__`, when a `since` is later than the next minor release, or when a `name` is missing from the `### Deprecations` of its `since` release's changelog section (`## Unreleased` while that release is the next one) *(checked)*. A release PR bumps `__version__`, so it fails while a due deprecation remains.

Before opening a release PR, run `python -m braincell._deprecation X.Y` with the new version: it lists the deprecations due by that release and exits with 1 if there are any. Remove or postpone each one in its own PR first ([releasing](releasing.md#checklist)).

## License notice

braincell is Apache-2.0. Every tracked `.py` file, including tests, examples, `conftest.py`, `docs/conf.py`, and the Jinja templates that render to Python, opens with this notice, verbatim *(checked)*. The rule line is `#`, a space, and exactly 78 `=`:

```python
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

- The year is the year the file was created, never the year it was edited; a 2024 file stays 2024.
- The notice comes first, then one blank line, then the module docstring or the first import. Only a `#!` shebang (line 1) and a PEP 263 encoding line (line 1 or 2) may precede it.
- The year is the only field that varies.
- The Jinja templates under `examples/convert_mod/nmodl/templates/` keep their own creation year: Jinja copies the notice into every rendered channel.
