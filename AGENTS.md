# BrainCell – Developer Guide

Biologically detailed brain cell modeling in BrainX.

## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main.
7. Use `brainstate.random` instead of `jax.random` directly for all random number generation.
8. **All prose lives under `docs/`; never leave a stray `.md` inside `braincell/`.** Two homes, each with a filename rule:
    - `docs/specs/YYYY-MM-DD-<slug>.md` — the spec and plan for one change, written *before* implementation. The date prefix is the creation date, so the directory reads chronologically.
    - `docs/design/<topic>.md` — durable design notes, invariants, and architecture maps that outlive any single change. Group a multi-document topic in its own subdirectory (`docs/design/network/`).

    Name a file for what it documents, not where the code happens to sit: `io-swc-reader-invariants.md`, never `README.md` or `notes.md`. Give it an `# H1` that matches.
9. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines.
10. Co-locate tests with the code under test: each module `foo.py` has its tests in a sibling `foo_test.py` (suffix style — never a separate `tests/` directory, never the `test_*.py` prefix). See [Testing](#testing) for the full rule.
11. **Never drive a model with a bare Python `for`/`while` loop when it runs repeatedly.** Python loops execute op-by-op (dispatch overhead, no fusion) and trace fresh each step; the `brainstate.transform` primitives lower the whole loop into one compiled XLA program, tracing the body only once. Pick by shape of the work:
    - **Single step or one-shot call** → `brainstate.transform.jit` — wrap the step/model call so it compiles once and reuses the trace.
    - **Many steps, collect outputs** → `brainstate.transform.for_loop` — repeat a step `length` times or map over `xs`; `State` is carried automatically and stacked outputs are returned.
    - **Many steps with an explicit carry** → `brainstate.transform.scan` — when threading a carry value alongside `State` (`f(carry, x) -> (carry, y)`).
    - **Long rollout under autograd (backprop through time)** → `brainstate.transform.checkpointed_for_loop` / `brainstate.transform.checkpointed_scan` — same semantics as above but rematerialize activations on the backward pass (tune `base`) to bound peak memory at the cost of recomputation.

    Compose them freely (e.g. `jit` an outer driver that calls a `for_loop`/`scan`). Reach for the checkpointed variants only when reverse-mode gradients through a long simulation would otherwise exhaust memory — otherwise prefer plain `for_loop`/`scan`.
12. Maintain compatibility with JAX versions >= 0.8.0 — `0.8.0` is the floor pinned by the `jax-version` matrix in `.github/workflows/CI-daily.yml`, which also runs latest. Prefer feature/shape detection over hard version checks.
13. **Every tracked `.py` file opens with the Apache-2.0 license header — add it when you create the file.** It goes at the very top, above the module docstring, below only a shebang or PEP 263 encoding line. See [License header](#license-header) for the verbatim block.


## Quick Reference

```bash
# Install (dev)
pip install -e ".[testing]"

# Run tests (tests are co-located with source code)
pytest braincell/

# Pre-commit
pre-commit install
pre-commit run --all
```


### Note on package naming

Internal packages (`_compute`, `_discretization`, `_multi_compartment`,
`_single_compartment`) and internal top-level modules (`_base.py`,
`_base_channel.py`, `_base_ion.py`, `_misc.py`, `_typing.py`) carry a
leading underscore because their *paths* are not part of BrainCell's
supported public API — all public re-exports flow through
`braincell/__init__.py`. Inner modules inside those packages (`base.py`,
`cell.py`, `runtime.py`, `geometry.py`, …) are unprefixed because they
are import targets for sibling internal code within the same package.
Domain packages that are part of the public surface (`channel`, `filter`,
`io`, `ion`, `mech`, `morph`, `network`, `quad`, `synapse`, `vis`) carry
no underscore. This is deliberate and matches the rest of the codebase.

## Critical Conventions

### License header

`braincell` is Apache-2.0 (`LICENSE`, `license = "Apache-2.0"` in `pyproject.toml`). **Every tracked `.py` file carries the notice below** — package modules, co-located `*_test.py` files, everything under `examples/`, `conftest.py`, `docs/conf.py`, and the Jinja templates that render to Python (so generated channels inherit it). Untracked scratch under `dev/` is exempt only because it is gitignored; give it the header too if it is ever promoted into the repository. Paste the block verbatim; the rule line is `#`, a space, then exactly 78 `=`:

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

Placement rules:

- **Year is the year the file is created**, not the year it is edited. Never renumber an existing header — a 2024 file stays 2024.
- **The header comes first**, then one blank line, then the module docstring or the first import.
- **Only two things may sit above it**: a `#!` shebang (must be line 1) and a PEP 263 encoding line (must be line 1 or 2). Nothing else — not a docstring, not an import, not a comment.
- **Copy the notice, do not reword it.** The `Copyright` year is the only field that varies.
- **Jinja templates under `examples/convert_mod/nmodl/templates/` are an exception to the year rule.** Their header is literal text that Jinja copies into every rendered channel, so the year is fixed at the template's own creation year rather than the generated file's. Leave it alone; generated output under `artifacts/` inherits whatever the template says.

### Units are mandatory

All physical quantities must carry explicit `brainunit` units. Bare numeric values **rejected with TypeError** by `normalize_param()` (in `braincell/_misc.py`). Never pass unitless numbers where quantity expected.

Data format: **python number / numpy array / jax array + brainunit unit**

#### Unit reference

`brainunit` provide SI units with standard prefixes: m (milli), u (micro), n (nano), p (pico), k (kilo), M (mega).

| Category | Quantity | Units |
|----------|----------|-------|
| Electrical | Voltage | `u.V`, `u.mV` |
| Electrical | Current | `u.A`, `u.mA`, `u.uA`, `u.nA`, `u.pA` |
| Electrical | Conductance | `u.S`, `u.mS`, `u.uS`, `u.nS`, `u.pS` |
| Electrical | Resistance | `u.ohm`, `u.kohm`, `u.Mohm` |
| Electrical | Capacitance | `u.F`, `u.uF`, `u.nF`, `u.pF` |
| Space/Time | Length | `u.m`, `u.cm`, `u.mm`, `u.um` |
| Space/Time | Time | `u.s`, `u.ms` |
| Substance/Temp | Molar concentration | `u.M`, `u.mM` |
| Substance/Temp | Temperature | `u.kelvin`, `u.celsius` |

#### Creating quantities

```python
import numpy as np
import brainunit as u
import jax.numpy as jnp

# Scalars
v_rest = -65.0 * u.mV
dt     = 0.1 * u.ms

# Arrays (numpy / JAX / list)
branch_lengths = np.array([10.5, 20.0, 15.3]) * u.um
xyz_coords     = jnp.zeros((10, 3)) * u.um
radii          = [2.0, 3.0, 4.0] * u.um
```

Quantities support same math functions and attributes as numpy arrays.

#### Arithmetic and dimensional analysis

```python
# Addition: automatic unit alignment
v1 = -60.0 * u.mV
v2 = 0.01 * u.V
v_total = v1 + v2              # → -50 mV

# Multiplication: compound dimensions
radius = 1.0 * u.um
length = 10.0 * u.um
area = 2 * np.pi * radius * length  # → u.um**2

# Division: density units
cm = 1.0 * u.uF / u.cm**2

# Ohm's law: conductance × voltage → current
g = 50 * u.nS / u.cm**2
I = g * (-65 * u.mV)          # auto-derived current dimension
```

#### Extracting units and raw values

```python
from brainunit import get_unit, get_mantissa

a = jnp.array([1, 2, 3]) * u.mV

get_unit(a)          # u.mV
get_mantissa(a)      # array([1, 2, 3])

a.to_decimal(u.V)    # array([0.001, 0.002, 0.003])  – raw float in target unit
a / u.mV             # array([1., 2., 3.])            – divide out the unit
a.to(u.V)            # array([0.001, 0.002, 0.003]) * u.V  – convert keeping unit
```

### Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Canonical section order:

1. **Short summary** – one-line imperative description (no blank line before).
2. **Extended summary** – optional, follow blank line after short summary.
3. **Parameters** – each entry: `name : type` on own line, description indented below.
4. **Returns** / **Yields** – same format as Parameters.
5. **Raises** – exception type and when raised.
6. **See Also** – related functions / classes.
7. **Notes** – implementation details, math, references.
8. **References** – numbered bibliography entries (`.. [1]`).
9. **Examples** – runnable, doctestable code snippets.

#### Rules for the Examples section

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

### Import style

- Internal modules use **absolute** imports: `from braincell.morph import MorphoBranch`
- Private modules prefix with `_` (e.g., `_base.py`, `_misc.py`)
- `set_module_as('braincell')` decorator marks functions for public namespace

### Type aliases

- **Annotate with the shared aliases in `braincell/_typing.py`** — write `g_max: Initializer`, never `g_max: Union[brainstate.typing.ArrayLike, Callable]`
- Import absolutely, alongside the other `braincell.*` imports: `from braincell._typing import Initializer, Size`
- Optional parameters use `Optional[Initializer]`, not a hand-rolled `Union[..., None]`
- Never re-import a type the aliases already cover (`from brainstate.typing import ArrayLike`) — go through `_typing.py`
- A type expression that repeats across modules belongs in `_typing.py` — add it there rather than duplicating it inline
- Only substitute an alias when it is genuinely the same concept — a bare `Hashable` that is not a section name, or a `tuple[str, ...]` that is not a state path, stays raw
- **Aliases are annotations only.** `_typing.py` is not public API, so docstrings keep the resolvable qualified name (`size : brainstate.typing.Size`) that a reader can import and Sphinx can cross-reference

| Alias | Stands for | Use for |
|-------|------------|---------|
| `Initializer` | `Union[ArrayLike, Callable]` | any parameter accepting a value or a callable that produces one |
| `ArrayLike` | `brainstate.typing.ArrayLike` | scalar / array parameters |
| `Size` | `brainstate.typing.Size` | a shape parameter — `size`, `pop_size`, and similar |
| `PyTree` | `brainstate.typing.PyTree` | nested state passed through an integrator |
| `SectionName` | `Hashable` | a morphology section identifier |
| `T`, `DT` | `u.Quantity[u.second]` | simulation time and time step |
| `Path` | `Tuple[str, ...]` | a state path within a model tree |
| `VectorField`, `Y0`, `Y1`, `Jacobian`, `Args`, `Aux` | see `_typing.py` | integrator / solver signatures in `braincell/quad` |


## Testing

- Framework: **pytest** with `unittest.TestCase`
- Config: `pyproject.toml` → `[tool.pytest]` (`ini_options.testpaths = ["braincell"]`, `ini_options.python_files = ["*_test.py", "test_*.py"]`). There is no `pytest.ini`.
- **Test file naming — mandatory.** Every test module **must** be named `*_test.py` and **co-located** with source it covers (e.g. `braincell/io/neuromorpho/client.py` → `braincell/io/neuromorpho/client_test.py`). Do **not** use bare `test.py`, `test_*.py`, or `tests/` subdirectories. When splitting large module across several files, give each file its own sibling `*_test.py`.
  - A bare `test.py` matches neither collection pattern and is silently never collected — this already cost the repo 72 uncollected SWC/ASC tests.
  - `test_*.py` is enabled in `python_files` **only** for the out-of-package NEURON comparison suite at `examples/neuron_compare/cable/tests/`, which predates this rule and is run directly by CI. That exception is documented at the `python_files` entry in `pyproject.toml`; it is not licence to use the prefix, or a `tests/` directory, anywhere new.
- **Shared test helpers** not themselves tests go into private `_testing.py` (or similar leading-underscore name) inside same package, so pytest does not discover them as test modules. Example: `braincell/io/neuromorpho/_testing.py` provides `FakeResponse` / `FakeSession` doubles consumed by every `*_test.py` in that package.
- JAX forced to CPU via `conftest.py` at project root (`JAX_PLATFORMS=cpu`)
- Matplotlib headless via `MPLBACKEND=Agg` in `conftest.py`
- Test morphology fixtures (SWC + ASC) live in `data/morphology/` at the repository root. Resolve them relative to the test file, counting parents up to the root — `parents[2]` from a module one level deep (`braincell/io/checkpoint_test.py`, `braincell/vis/*_test.py`), `parents[3]` from two levels deep (`braincell/io/swc/swc_test.py`, `braincell/io/asc/asc_test.py`):

    ```python
    FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "morphology"
    ```

    `data/` is pruned from the source distribution by `MANIFEST.in`, so these tests only run from a repository checkout, never from an installed package. They have no skip guard and will error rather than skip if the fixtures are missing.
