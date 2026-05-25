# Contributing

Contributions of all kinds are welcome — bug reports, fixes, new channels,
documentation, and examples. `braincell` is developed on GitHub at
[chaobrain/braincell](https://github.com/chaobrain/braincell).

## Development setup

Clone the repository and install in editable mode with the testing extras, then
install the pre-commit hooks:

```bash
git clone https://github.com/chaobrain/braincell.git
cd braincell
pip install -e ".[testing]"
pre-commit install
```

## The workflow

1. **Open an issue** (or find one) describing the bug or feature.
2. **Create a branch** for your work.
3. **Write a test first.** The project convention is: *reproduce a bug with a
   failing test, then fix until it passes.* See {doc}`testing`.
4. **Implement** the change, keeping code simple and intuitive.
5. **Run the checks** locally:
   ```bash
   pytest braincell/
   pre-commit run --all
   ```
6. **Open a pull request** referencing the issue.

## Coding conventions

These conventions come from the project's developer guide and keep the codebase
coherent:

- **Units are mandatory.** Every physical quantity must carry an explicit
  `brainunit` unit; bare numbers are rejected. See {doc}`../concepts/units`.
- **One canonical namespace per API.** Public symbols are re-exported through
  the top-level `braincell` namespace; internal packages are underscore-prefixed
  (`_cv`, `_compute`, …). See {doc}`project_layout`.
- **Ask when requirements are ambiguous** rather than guessing.
- **NumPy-style docstrings** for every public class, method, and function, with
  a runnable, doctestable Examples section where it helps.
- **Absolute imports** for internal modules
  (`from braincell.morph import Morphology`).

## Documentation

Documentation lives in `docs/` and is built with Sphinx (`sphinx_book_theme`,
MyST, and `myst-nb` for notebooks). To build locally:

```bash
cd docs
make html        # output in docs/_build/html
```

Narrative pages are Markdown (MyST) or reStructuredText; tutorials and examples
are Jupyter notebooks. The API reference is generated from docstrings via
`autosummary`, so improving a docstring improves the reference automatically.

## See also

- {doc}`testing` — the testing conventions in detail.
- {doc}`extending` — adding channels, ions, and integrators.
- {doc}`project_layout` — where everything lives.
