---
myst:
  heading_anchors: 2
---

# Contributing

Start with a problem that can be described and verified. Bug reports should include a minimal reproduction; new models should explain their sources and purpose. Before changing a public interface or starting a feature, open an [issue](https://github.com/chaobrain/braincell/issues) to discuss it. See [Repository Layout](project_layout.md) for entry points.

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

## Find or claim a task

Roadmap tasks have an ID made of a milestone code and a number, for example `CELL-3`; `CELL-3a` is one PR of `CELL-3`. The maintainers plan and assign them, and each task's acceptance criteria define when it is done. For anything without a task, open an issue first.

## Make and validate changes

1. **Create a branch** off the latest `main`, one task or one problem per branch: `git switch -c fix/swc-validation origin/main`. Nothing reaches `main` except through a pull request.
2. **Agree on the approach** for a non-trivial change before writing it: in the issue, or in a draft PR opened on the first push.
3. **Reproduce a bug with a failing test** before fixing it, and keep the test.
4. **Follow the [code conventions](conventions.md)**: units, random numbers, compiled loops, imports, type aliases, docstrings, and the license notice.
5. **Run the relevant checks**: tests for the changed modules and affected examples first, more for shared behavior; see the [Testing Guide](testing.md). Validate new model dynamics and integration behavior with the [Extension Guide](extending.md).
6. **Update in the same PR** the docstrings, user documentation, examples, and `changelog.md` (under `## Unreleased`) that the change affects.

## Change documentation

Contribution steps belong in the Developer Guide; signatures, parameters, equations, and runnable examples belong in docstrings, which the [API reference](../apis/braincell.rst) renders.
Maintain existing examples at their actual locations, and use only implemented interfaces in executable examples.

Build the website:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

CI runs this same command, so any warning (a broken link, cross-reference, or toctree entry) fails the PR; `--keep-going` lists every warning in one run.
The build downloads the page header from brainx.chaobrain.com into `docs/_brand/`; offline, place that cache first and set `BRAINX_HEADER_OFFLINE=1`.
Do not add `-j`: parallel builds fork a process in which JAX already runs threads, and can hang.
The output is `docs/_build/html/index.html`. Website builds currently do not execute notebooks; after changing runnable code in a notebook, run that notebook or its companion script separately.

## Breaking changes

A change that removes or renames a public name, or changes its behavior in a way existing calls notice, first ships a deprecation warning for at least one minor release; the next minor release may then remove the old form. Record both steps in `changelog.md`, with the migration. Declare the deprecation with the `braincell._deprecation` helper, which sets the message, the line the warning points at, the tests and the changelog entries ([conventions](conventions.md#deprecation)).

## Submit a pull request

Before pushing, run the commit checks and look for generated outputs in the diff:

```bash
pre-commit run --all-files
git diff --check
git status --short
```

Formatting and linting use Ruff as configured in `pyproject.toml`, with a line length of 120 and existing quote styles preserved.
Pre-commit may modify files: review what it changed before committing, and keep rewrites outside your change out of the PR.

Push your branch and open a PR with the
[PR template](https://github.com/chaobrain/braincell/blob/main/.github/PULL_REQUEST_TEMPLATE.md):

- **Title**: the task ID (`CELL-3a: add the point-process sampler`) or a conventional prefix (`fix(io): …`, `docs: …`). PRs are squash-merged, and the title becomes the commit subject on `main`.
- **Size**: split a diff over about 1500 lines, deletions and generated files aside, into PRs that can be reviewed one at a time.
- **Body**: the problem and the related issue or task; the resulting behavior, with a before/after example when useful; the commands actually run and their results, and any verification gaps; breaking changes and the migration.

For numerical models, also state the reference, units, temperature, initial state, solver, time step, and comparison tolerances.
Support performance claims with measurement configurations and reproduction entry points.

## Review and merge

A PR can merge when:

- the `ci-ok` check passes: it requires every pull-request CI job (pre-commit, the package build and the documentation build; the package suite with coverage; the NEURON cable and single-channel comparisons; the validation and benchmark tests) to succeed or to be skipped because the PR's files cannot affect it ([what CI runs](testing.md#what-ci-runs-on-a-pull-request)); a draft PR runs no tests, so mark it ready for review first;
- a code owner other than the author approves it ([.github/CODEOWNERS](https://github.com/chaobrain/braincell/blob/main/.github/CODEOWNERS)).

PRs merge through the merge queue: once both hold, choose *Merge when ready* (or `gh pr merge <N> --squash`). The queue tests `main` plus the queued PRs, runs `ci-ok` again on that combination, and squash-merges only if it passes, so a PR no longer needs to be rebased each time `main` moves.
