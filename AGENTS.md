# braincell

Biophysically detailed brain cell modeling in BrainX. This file holds the rules for changing code, tests and docs in this repository; the [Developer Guide](docs/developer/index.rst) has the details and the reasons. Keep it short: one rule per line; edit or delete a rule before adding one.

## Tasks and specs

- Roadmap tasks are planned by the maintainers in a private spec repository, checked out beside this one. If your work has a task ID (e.g. `CELL-3a`) and you have that checkout, read its `AGENTS.md` and the task's section first; the acceptance criteria there define done.
- Nothing private here: name a task by its ID and state the decision you rely on in the text itself; never cite spec paths or file names (pre-commit checks).

## Workflow

1. Before coding a non-trivial change, describe the approach and wait for approval; ask when requirements are ambiguous.
2. One task, one PR, from a branch off `origin/main` (agents: a git worktree); `main` accepts changes only through pull requests.
3. PR titles start with the task ID (`CELL-3a: …`) or a conventional prefix (`fix(io): …`); the squash commit on `main` takes the PR title.
4. Split a diff over ~1500 lines (deletions and generated files aside) into PRs that can be reviewed alone.
5. A bug fix starts with a test that reproduces it.
6. Docs, examples and `changelog.md` change in the same PR as the code they describe.
7. Breaking a public API needs a deprecation warning that ships in at least one minor release first.
8. Before pushing, run `pre-commit run --all-files` and the tests your change affects, the latter with `--cov=braincell --cov-report=xml` followed by `diff-cover coverage.xml --compare-branch=origin/main --fail-under=90`; the PR body lists the commands and their results.
9. A PR merges when `ci-ok` passes and a code owner other than the author approves (`.github/CODEOWNERS`).
10. After writing code, list the edge cases it handles and the tests that cover them.
11. When corrected, state the mistake and how you will avoid repeating it.

Details: [contributing](docs/developer/contributing.md).

## Code

- Units are mandatory: every physical quantity carries a `brainunit` unit; `normalize_param()` rejects bare numbers with `TypeError`.
- Random numbers come from `brainstate.random`, never from `jax.random` directly.
- Never drive a model repeatedly with a Python `for`/`while` loop: use `brainstate.transform.jit` (one step), `for_loop` (many steps, stacked outputs), `scan` (explicit carry), or `checkpointed_for_loop` / `checkpointed_scan` (backprop through a long rollout).
- Support every JAX release from the floor in `pyproject.toml` to the latest; prefer feature detection to version checks.
- Internal imports are absolute (`from braincell.morph import MorphoBranch`); public names are re-exported through `braincell/__init__.py`.
- Annotate with the aliases in `braincell/_typing.py` (`Initializer`, `Size`, …), never with hand-rolled `Union`s.
- Public classes, methods and functions have NumPy-style docstrings with runnable `>>>` examples.
- Every tracked `.py` file starts with the Apache-2.0 notice, verbatim, dated the year the file was created (pre-commit checks).

Details and examples: [conventions](docs/developer/conventions.md).

## Tests

- Run `pytest braincell/`; tests use `unittest.TestCase` and run on CPU (`conftest.py`).
- A module `foo.py` has its tests in a sibling `foo_test.py`, never in `test_*.py` or a `tests/` directory (pre-commit checks); package-wide guards go in `<package>/__init___test.py`.
- Coverage gates (CI `coverage` job): total coverage of `braincell/` must not fall below `fail_under` in `[tool.coverage]`, which is raised to the current total before each minor release; lines a PR adds or changes need at least 90 % (`diff-cover`). Cover edge cases and critical paths, not trivial lines.
- Import fixture paths from `braincell/io/_testing.py`; never recompute them.

Details: [testing](docs/developer/testing.md).

## Benchmarks

- Before running a benchmark, list its rounds, processes, warm-ups, timed repeats and extra runs and have them confirmed; script defaults are not an authorization.
- What goes in a benchmark's README, `results/` and `artifacts/`: [benchmarks/README.md](benchmarks/README.md).
