# Contributing to `braincell`

Thank you for contributing to `braincell`.

This project provides biologically detailed brain cell modeling tools built on top of JAX and the BrainX ecosystem. Contributions are welcome across code, tests, documentation, examples, bug reports, and design discussions.

By participating in this project, you agree to follow the guidance in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Ways to contribute

You can help by:

- reporting bugs or unclear behavior
- proposing new features or API improvements
- improving documentation and tutorials
- adding tests for uncovered behavior
- contributing bug fixes or new functionality
- improving examples under `examples/`

If you are planning a larger change, open an issue first so the scope and API impact can be discussed before implementation.

## Development setup

`braincell` requires Python 3.11 or newer. Continuous integration currently tests Python 3.13 on Linux, macOS, and Windows.

Create an isolated environment, install the development requirements, and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If you only need the base package instead of the full development environment, `pip install -e .` is sufficient for local package edits.

## Running tests

Run the test suite from the repository root:

```bash
pytest braincell/
```

The repository-level `pytest.ini` already points test discovery at the `braincell` package and skips `legacy/` and development notebooks.

On Windows, CI disables the fault handler. If you hit platform-specific issues locally, this is the closest CI-equivalent command:

```bash
pytest braincell/ -p no:faulthandler
```

When you change behavior, add or update tests in the same area of the codebase whenever practical.

## Building documentation

The documentation lives in `docs/` and uses Sphinx.

Install the documentation dependencies:

```bash
python -m pip install -r requirements-doc.txt
python -m pip install -e .
```

Then build the HTML docs:

```bash
cd docs
make html
```

On Windows:

```powershell
cd docs
.\make.bat html
```

Documentation includes Markdown, reStructuredText, and notebooks. Notebook execution is disabled in the Sphinx configuration, so documentation changes should focus on content correctness and importability.

## Code style and contribution expectations

This repository does not currently define a dedicated formatter or linter configuration. Follow the existing style in the surrounding files and keep changes consistent with the current codebase.

Please keep the following in mind:

- prefer small, focused pull requests
- keep public APIs and examples stable unless the change intentionally updates them
- update documentation when user-facing behavior changes
- update tests when logic changes
- avoid adding new dependencies unless they are clearly justified
- preserve compatibility with the supported Python versions declared in `pyproject.toml`

Type hints and concise docstrings are encouraged when they improve clarity.

## Working with examples and legacy code

Current examples live under `examples/`.

Legacy multi-compartment code and older examples are kept under `legacy/`. Unless your change is specifically about legacy support, prefer making updates in the main `braincell/` package and the current examples/docs.

## Pull requests

Before opening a pull request:

1. make sure your branch is based on the latest target branch state
2. run the relevant tests locally
3. update documentation, examples, or `changelog.md` if your change is user-facing
4. review the pull request template in `.github/PULL_REQUEST_TEMPLATE.md`

When opening a pull request, include:

- a clear description of the problem and the change
- links to any related issues
- the local test commands you ran
- notes about API changes, breaking changes, or follow-up work

Draft pull requests are welcome for early feedback.

## Reporting bugs and security issues

- For general bugs, feature requests, and usability issues, open a GitHub issue: <https://github.com/chaobrain/braincell/issues>
- For security-related concerns, see [SECURITY.md](SECURITY.md)

Please include enough detail to reproduce the problem: operating system, Python version, package versions, a minimal example, and the observed error or unexpected behavior.

## License

By contributing to this repository, you agree that your contributions will be distributed under the same license as the project. See [LICENSE](LICENSE).
