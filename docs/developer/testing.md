# Testing

`braincell` uses **pytest** with `unittest.TestCase` classes. The test suite is
the safety net for a numerically delicate library, so the conventions below are
project rules, not suggestions.

## Running the tests

```bash
pytest braincell/
```

`pytest.ini` sets `testpaths = braincell` and excludes `legacy` and `develop`.
Two environment defaults are applied automatically by the root `conftest.py`:

- `JAX_PLATFORMS=cpu` — JAX is forced onto CPU so tests are deterministic and
  don't depend on a GPU.
- `MPLBACKEND=Agg` — matplotlib runs headless so visualization tests don't open
  windows.

## Test file naming — mandatory

Every test module **must**:

- be named `*_test.py` (not `test_*.py`, not bare `test.py`), and
- sit **next to the source file it covers**.

```text
braincell/io/neuromorpho/client.py
braincell/io/neuromorpho/client_test.py   ← its test, co-located
```

When a module is split across several source files, give each its own sibling
`*_test.py`. The `tests/` directory at the repo root is an empty placeholder —
do not put tests there.

## Shared test helpers

Helpers that are **not themselves tests** go in a private, leading-underscore
file (e.g. `_testing.py`) inside the same package, so pytest does not collect
them. For example, `braincell/io/neuromorpho/_testing.py` provides fake
HTTP doubles used across that package's tests.

## The bug-fixing workflow

The project rule for bugs is:

> **Write a test that reproduces the bug, then fix until the test passes.**

This guarantees the bug stays fixed and documents the expected behavior.

## Test fixtures

Morphology fixtures (SWC + ASC) live in
`examples/multi_compartment/morpho_files/`. IO tests locate them relative to the
test file:

```python
from pathlib import Path
MORPHO_DIR = Path(__file__).resolve().parents[2] / "examples" / "multi_compartment" / "morpho_files"
```

## See also

- {doc}`contributing` — the overall development workflow.
- {doc}`extending` — testing a new channel or integrator.
