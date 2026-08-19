# Contributing to `braincell`

Thank you for contributing to `braincell`.

This project provides biologically detailed brain cell modeling tools built on top of JAX and the BrainX ecosystem. Contributions are welcome across code, tests, documentation, examples, bug reports, and design discussions.

By participating in this project, you agree to follow the guidance in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Project conventions](#project-conventions)
- [Building documentation](#building-documentation)
- [Pull requests](#pull-requests)
- [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
- [License](#license)

## Ways to contribute

You can help by:

- reporting bugs or unclear behavior
- proposing new features or API improvements
- improving documentation and tutorials
- adding tests for uncovered behavior
- contributing bug fixes or new functionality
- improving examples under `examples/`

If you are planning a larger change, open an issue first so the scope and API impact can be discussed before implementation.

`TODO.md` is the project design document. It tracks the architectural intent and the current implementation state of every subsystem, using `[x]` shipped / `[~]` partial / `[ ]` planned markers. Read the relevant section before starting substantial work, and update it when your change moves a subsystem forward.

## Development setup

`braincell` requires Python 3.11 or newer. Continuous integration currently tests Python 3.13 on Linux, macOS, and Windows, and a nightly matrix exercises JAX 0.8.0 through the latest release.

Create an isolated environment and install the package in editable mode with the development extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Dependency groups are declared in `pyproject.toml` under `[project.optional-dependencies]`:

| Extra | Contents |
| --- | --- |
| `vis` | `matplotlib`, `networkx`, `pyvista`, `plotly` — 2D and 3D visualization backends |
| `io` | `requests` — the NeuroMorpho.Org client |
| `all` | `vis` + `io`; everything a user-facing install might need |
| `testing` | `all` + `pytest`, `pytest-benchmark`, `hypothesis`, `absl-py` |
| `doc` | `all` + the Sphinx toolchain |
| `dev` | `testing` + `doc` + `pre-commit` |
| `cpu` / `cuda12` / `cuda13` / `tpu` | the matching JAX backend build; pick exactly one |

The `requirements*.txt` files are thin pointers to these extras and exist only for tooling that expects them (CI and Read the Docs). Add new dependencies to `pyproject.toml`, never to the `requirements*.txt` files.

Finally, install the git hooks:

```bash
pre-commit install
```

## Running tests

Run the test suite from the repository root:

```bash
pytest braincell/
```

Test discovery is configured in `pyproject.toml` under `[tool.pytest]`; it points at the `braincell` package and collects only `*_test.py` modules.

To run a single module or a single test:

```bash
pytest braincell/io/swc/reader_test.py
pytest braincell/io/swc/reader_test.py::SwcReaderTest::test_single_point_soma_expands_to_three_points_and_connects_at_midpoint
```

With coverage (configuration lives in `[tool.coverage]`):

```bash
pytest braincell/ --cov=braincell --cov-report=term-missing
```

On Windows, CI disables the fault handler. If you hit platform-specific issues locally, this is the closest CI-equivalent command:

```bash
pytest braincell/ -p no:faulthandler
```

Some suites skip themselves when an optional dependency is absent — `pytest-benchmark` for the performance baselines, `hypothesis` for the layout property tests, and `pyvista` / `plotly` for the 3D backends. Install the `testing` extra to run them.

When you change behavior, add or update tests in the same area of the codebase.

## Code style

Formatting and linting are handled by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml` under `[tool.ruff]` and run through pre-commit:

```bash
pre-commit run --all-files
```

The line length is 120. Quote style is set to `preserve`, so the existing mix of single and double quotes is left alone — do not reformat strings gratuitously.

The lint rule set is currently minimal on purpose. `[tool.ruff]`'s `lint.ignore` list enumerates the rules the tree still violates, each with its violation count and the reason it is deferred rather than fixed. If your change cleans up one of those categories, delete the corresponding entry in the same pull request.

## Project conventions

These are the conventions that reviewers will look for. `AGENTS.md` holds the full version; the essentials are:

**Units are mandatory.** Every public API that takes a physical quantity routes through `normalize_param()` in `braincell/_misc.py`, which rejects bare numerics with `TypeError`. Accept `python number / numpy array / jax array * brainunit unit`, store canonical SI internally, and hand values back with units attached — never raw floats.

```python
import brainunit as u

v_rest = -65.0 * u.mV      # correct
v_rest = -65.0             # rejected with TypeError
```

**Tests are co-located and suffix-named.** Every module `foo.py` has its tests in a sibling `foo_test.py`. Never a separate `tests/` directory, and never the `test_*.py` prefix — `python_files` is set to `*_test.py` only, so a misnamed file is silently never run. Shared test helpers that are not themselves tests go in a leading-underscore module such as `_testing.py`.

**Docstrings are NumPy-style.** See `AGENTS.md` for the canonical section order. Examples must be `.. code-block:: python` blocks that are doctest-compatible and self-contained.

**Use `brainstate.random`,** not `jax.random` directly.

**Do not drive a model with a bare Python loop.** Use `brainstate.transform.for_loop` / `scan` (or the `checkpointed_` variants under autograd) so the loop lowers into a single compiled XLA program.

**Keep optional dependencies lazy.** `matplotlib`, `pyvista`, and `plotly` must be imported inside the backend that uses them, gated on `importlib.util.find_spec`, so that `import braincell` stays cheap.

**Support JAX >= 0.8.0.** Prefer feature or shape detection over hard version checks.

Write a spec under `docs/specs` before implementing a substantial change, so the design is available for reference during review.

## Building documentation

The documentation lives in `docs/` and uses Sphinx.

```bash
python -m pip install -e ".[doc]"
cd docs
make html
```

On Windows:

```powershell
cd docs
.\make.bat html
```

If you regenerate notebooks that contain interactive `vis3d(...)` output, also install the PyVista HTML export dependencies before running and saving those notebooks:

```bash
python -m pip install ipywidgets trame trame-vtk trame-vuetify "jupyterlab>=3"
```

Documentation includes Markdown, reStructuredText, and notebooks. Notebook execution is disabled in the Sphinx configuration, so documentation changes should focus on content correctness and importability. For static interactive PyVista output, use `vis3d(notebook=True, jupyter_backend="html")`; the PyVista backend exports raw iframe HTML for this backend so the published docs page does not need to load the Jupyter widget manager.

## Pull requests

Before opening a pull request:

1. make sure your branch is based on the latest target branch state
2. run `pre-commit run --all-files`
3. run the relevant tests locally
4. update documentation, examples, `TODO.md`, or `changelog.md` if your change is user-facing
5. review the pull request template in `.github/PULL_REQUEST_TEMPLATE.md`

When opening a pull request, include:

- a clear description of the problem and the change
- links to any related issues
- the local test commands you ran
- notes about API changes, breaking changes, or follow-up work

Keep pull requests small and focused. Please keep the following in mind:

- keep public APIs and examples stable unless the change intentionally updates them
- avoid adding new dependencies unless they are clearly justified
- preserve compatibility with the supported Python versions declared in `pyproject.toml`

Draft pull requests are welcome for early feedback.

## Reporting bugs and security issues

- For general bugs, feature requests, and usability issues, open a GitHub issue: <https://github.com/chaobrain/braincell/issues>
- For security-related concerns, **do not open a public issue** — see [SECURITY.md](SECURITY.md) for the private reporting channels.

Please include enough detail to reproduce the problem: operating system, Python version, package versions, a minimal example, and the observed error or unexpected behavior.

## License

By contributing to this repository, you agree that your contributions will be distributed under the same license as the project — Apache License 2.0. See [LICENSE](LICENSE).
