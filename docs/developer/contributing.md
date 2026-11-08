---
myst:
  heading_anchors: 2
---

# Contributing

Start with a problem that can be described and verified. Bug reports should include a minimal reproduction; new models should explain their sources and purpose. Before changing a public interface, open an issue to discuss it. See [Code Navigation](project_layout.md) for entry points.

## Set up a development environment

Clone the repository in a separate Python environment. External contributors can first fork the project on GitHub and clone their fork. The commands below clone the main repository and install development dependencies and commit checks:

```bash
git clone https://github.com/chaobrain/braincell.git
cd braincell
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
```

In Windows PowerShell, replace the activation command with `.venv\Scripts\Activate.ps1`.
If you already have a separate environment, install directly. The Python version requirement is defined by `requires-python` in `pyproject.toml`.

The `dev` extra includes testing, documentation, and pre-commit dependencies, as defined in
[pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml).
See the [Installation Guide](../getting_started/installation.ipynb) for CPU/GPU installation options and [Troubleshooting](troubleshooting.md) for environment issues. Run the commands below from the repository root.

Add new dependencies to `pyproject.toml`; `requirements*.txt` files only reference these extras for CI and other tools.

## Make and validate changes

1. **Locate the task:** roadmap tasks have an ID (for example `CELL-3a`) and are assigned by the maintainers; the task's acceptance criteria define done. For anything else, open an [issue](https://github.com/chaobrain/braincell/issues) that describes the problem before starting on a feature.
2. **Create a working branch:** for example, `git switch -c fix/swc-validation`. Keep each PR focused on one problem.
3. **Make verifiable changes:** reproduce a bug with a failing test before fixing it. Validate new model dynamics and integration behavior using the [Extension Guide](extending.md).
4. **Run relevant checks:** start with tests for the changed modules and affected examples. Broaden coverage for shared behavior changes; see the [Testing Guide](testing.md).
5. **Review before committing:** update the docstrings, user documentation, examples, and `changelog.md` that the change affects, in the same PR.

Conventions for physical units, random numbers, compiled loops, imports, licenses, and docstrings are maintained in the
[repository rules](https://github.com/chaobrain/braincell/blob/main/AGENTS.md).

## Change documentation

Contribution steps belong in the Developer Guide; signatures, parameters, equations, and runnable examples belong in docstrings, which the [API reference](../apis/braincell.rst) renders.
Maintain existing examples at their actual locations.

Build the website:

```bash
python -m sphinx -b html docs docs/_build/html
```

The output is `docs/_build/html/index.html`. Website builds currently do not execute notebooks; after changing runnable code in a notebook, run that notebook or its companion script separately. API references are generated from docstrings.

## Submit a pull request

Before submitting, run formatting and static checks and check for generated outputs in the diff:

```bash
pre-commit run --all-files
git diff --check
git status --short
```

Formatting and linting use Ruff as configured in `pyproject.toml`, with a line length of 120 and existing quote styles preserved.
Pre-commit may modify files; review its results before committing. Push your working branch and open a PR using the
[PR template](https://github.com/chaobrain/braincell/blob/main/.github/PULL_REQUEST_TEMPLATE.md). Describe:

- The problem or use case, with the related issue, proposal, or model source.
- The resulting behavior, with a before/after example when useful.
- The tests and examples actually run, their results, and the reasons for any verification gaps.
- Changes affecting existing calls and any migration steps.

For numerical models, also state the reference, units, temperature, initial state, solver, time step, and comparison tolerances.
Support performance claims with measurement configurations and reproduction entry points.
