---
myst:
  heading_anchors: 2
---

# Testing

Start with tests for the changed modules, then check integration behavior according to the scope of the change.
BrainCell uses pytest with tests next to the source they cover; collection is configured in
[pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml) (`[tool.pytest.ini_options]`; there is no `pytest.ini`).
After [setting up the development environment](contributing.md#set-up-a-development-environment), run the following commands from the repository root.

## Run relevant tests

For the SWC reader, run one module, inspect its collected tests, or run the entire IO package:

```bash
python -m pytest braincell/io/swc/reader_test.py -q
python -m pytest braincell/io/swc/reader_test.py --collect-only -q
python -m pytest braincell/io/ -q
```

For shared behavior spanning several modules, expand testing to the relevant packages. Run the full core suite with:

```bash
python -m pytest braincell/ -q
python -m pytest braincell/ -q -n auto --dist loadfile   # in parallel (pytest-xdist), as CI runs it
```

The root `conftest.py` forces CPU JAX (`JAX_PLATFORMS=cpu`) and headless Matplotlib (`MPLBACKEND=Agg`).
Tests at both precisions share one process: a float64 test may follow a float32 one, or run while float32 models are alive. `run_float64` in `braincell/ion/_testing.py` computes a report at float64 in the same process.
A test must not hand one module-level NumPy array to JAX at both precisions (see {ref}`Troubleshooting <precision-switch>`); use a Python float, a NumPy scalar, or a JAX array made at the precision in use.
A test that needs float64 sets it with `brainstate.environ.context(precision=64)` or an autouse fixture. A module-level `brainstate.environ.set` is process-wide and would change the precision of every test collected with it, so the root `conftest.py` restores the session's precision once collection ends; the NEURON suites under `validation/neuron`, which set float64 when imported, get it while their tests run from `validation/neuron/conftest.py`.
GPU performance and large numerical comparisons require separate workflows; unit tests do not establish those results.

## Validation and performance workflows

```bash
python -m pytest braincell/experimental/optim -q
python -m pytest validation/optim benchmarks -m "not slow" -q   # every pull request
python -m pytest validation/optim benchmarks -m slow -q         # daily CI run
python -m pytest validation/neuron/cable/tests -q
```

CI runs `validation/optim` and the benchmark scripts' tests in two tiers: modules marked `pytestmark = pytest.mark.slow` (more than about 25 s on their own on a CI runner, roughly 10 s on a workstation) run only in the daily workflow, the rest on every pull request. A new test module there runs on every pull request unless you mark it slow.

## What CI runs on a pull request

A draft pull request runs no tests: its `ci-ok` fails until you mark it ready for review, which starts the run. On a ready pull request, the `changes` job selects the test jobs from the files it changes ([.github/ci_changes.py](https://github.com/chaobrain/braincell/blob/main/.github/ci_changes.py)), and `ci-ok` accepts a job it skipped:

| Files the pull request changes | Jobs that run |
| --- | --- |
| `.github/`, `pyproject.toml`, `requirements*`, `setup.*`, `MANIFEST.in`, a `conftest.py`, `data/` | every job |
| `braincell/` | every job |
| `validation/neuron/` | `checks`, the NEURON steps of `validation` |
| `benchmarks/` or another directory of `validation/` | `checks`, the `validation/optim` and benchmarks step of `validation` |
| anything else (`docs/`, `examples/`, `changelog.md`, ...) | `checks` |

`checks` (pre-commit, the package build, the documentation build and the docstring examples, `pytest --doctest-modules braincell --ignore-glob='*_test.py'`) always runs. `coverage` runs `pytest braincell/` on the latest JAX, in parallel with pytest-xdist. Every job uses Python 3.13; the JAX floor (0.11.0), macOS and Windows run only in the daily workflow, which runs every job and also tests `main` (there is no run on `main` after a merge). To add a rule, edit `RUN_ALL` or `jobs()` in `ci_changes.py` and its `CASES`; the `changes` job runs `ci_changes.py --self-test` first.

The NEURON comparison suites are tiered the same way, each suite in its own pytest process (one `validation` job on a pull request):

| Tier | CI job | What runs |
| --- | --- | --- |
| Every pull request (in `ci-ok`) | `validation` | `validation/neuron/cable/tests` |
| Every pull request (in `ci-ok`) | `validation` | `validation/neuron/channel_no_conc/tests` and the shared `_clamp_test.py`, `_tolerance_test.py`, `_convergence_test.py`, `_mechanisms_test.py`, `_paths_test.py` under `validation/neuron/`, with `-m "not slow"`; then `validation/neuron/ion` with `-m "not slow"` in a separate process |
| Every pull request (in `ci-ok`) | `validation` | `validation/neuron/morph` (SWC and ASC morphology metrics and soma layout against NEURON's import3d), `AscNeuronParityTest` in `braincell/io/asc/reader_test.py`, and `validation/neuron/synapse` (`ExpSyn`, `Exp2Syn`, `NetCon` delays and spike times), each in its own process; these tests skip without NEURON, so the job fails if any of them or a fast ion test is skipped (`.github/junit_counts.py` reads each step's JUnit report) |
| Daily run only | `test_linux` | `pytest braincell/` on the JAX floor (0.11.0) |
| Daily run only | `test_macos`, `test_windows` | `pytest braincell/` on macOS and Windows (Python 3.13); kept out of pull requests to stay within the concurrent-job limit |
| Daily run only | `slow_tests_neuron` | `-m slow` of `validation/neuron/channel_no_conc/tests` (the per-family catalogue and `dt` convergence tests, and `test_neuron_agreement.py`), then of `validation/neuron/ion` (the pool comparisons, and the chloride, GABA_A, sodium and potassium comparisons against the ModelDB sources in `data/modeldb/`) in a separate process, which fails if a test skipped |
| Daily run only | `examples` (in `CI-daily.yml`, not `CI.yml`) | every example script under `examples/` except those in `examples/nightly_skip.toml`, then `pytest examples`; see {ref}`the examples job <the-daily-examples-job>` |

Both tiers compile the eight channel models first; the morphology, ASC and synapse comparisons use only NEURON's built-in mechanisms and need no build. To reproduce them locally:

```bash
for model in bc_ma2025 dcn_su2015 goc_ma2020 grc_ma2020 io_zh2019 pc_ma2024 sc_ma2021 testing; do
    python -m validation.neuron._mechanisms "$model" --kind channel
done
# Pull-request tier.
python -m pytest validation/neuron/channel_no_conc/tests \
    validation/neuron/_clamp_test.py validation/neuron/_tolerance_test.py \
    validation/neuron/_convergence_test.py validation/neuron/_mechanisms_test.py \
    validation/neuron/_paths_test.py -m "not slow" -q
python -m pytest validation/neuron/ion -m "not slow" -q
python -m pytest validation/neuron/morph -q
python -m pytest braincell/io/asc/reader_test.py -k AscNeuronParityTest -q
python -m pytest validation/neuron/synapse -q
# Daily tier.
python -m pytest validation/neuron/channel_no_conc/tests -m slow -q
python -m pytest validation/neuron/ion -m slow -q
```

A NEURON comparison module that takes more than about 25 s on its own on a CI runner gets `pytestmark = pytest.mark.slow` and so moves to the daily run; the daily job selects by marker, so it needs no workflow change. A red daily run opens a `main is red` issue, and fixing it is the first task of the day. To run the daily jobs on a branch, dispatch the daily workflow on it (`gh workflow run CI-daily.yml --ref <branch>`); a red run there opens an issue titled `<branch> is red`.

(the-daily-examples-job)=
### The daily examples job

The daily workflow's `examples` job (not in `ci-ok`; a red run opens the `main is red` issue) runs `python .github/run_examples.py`, with the test dependencies only (`requirements-dev.txt`) and on CPU:

- **Scripts**: every tracked `examples/**/*.py` except `*_test.py`, `__init__.py` and `conftest.py`, each in its own process from the repository root, with `MPLBACKEND=Agg`. A script inside a package runs as `python -m examples.<dotted.name>`, any other as `python <path>`. A script fails on a non-zero exit or after 600 s. Scripts run in path order, so one may read what an earlier script in its folder wrote (`stimulus_design/robust_oed.py` reads `dataset.py`'s output).
- **Tests**: `pytest examples`. The step fails when a test is skipped, so a test that needs an optional dependency is deselected through the list, not skipped silently.
- **Notebooks**: none run here. Re-execute the ones a change affects, as the PR checklist says.

`examples/nightly_skip.toml` is the one list of what does not run. A `[[skip]]` entry (`path`, `reason`) is for a script that cannot run in that job: it needs a package the test dependencies do not install (numba, pandas, nmodl), a GPU, a long experiment run or command-line arguments, or it is not Python. A `[[deselect]]` entry (`nodeid`, `reason`) is for such a test. A `[[timeout]]` entry (`path`, `seconds`) is for a script that is merely slow. Each reason says what is missing. An entry for an untracked script, or for a test pytest does not collect, fails the job, so renaming or deleting an example means editing the list in the same PR. A new example joins the job without editing the workflow. To run the job locally, use a fresh environment with `pip install -r requirements-dev.txt` (a development environment with numba or NEURON hides missing dependencies) and run:

```bash
python .github/run_examples.py --self-test
python .github/run_examples.py                 # or --only examples/<path>.py
python .github/run_examples.py --pytest /tmp/examples.xml
```

See the [Validation Guide](https://github.com/chaobrain/braincell/blob/main/validation/neuron/README.md) for entry points and dependencies.
The benchmark tests check the scripts on small configurations and synthetic records; nothing they time is a benchmark result. Running a benchmark itself is different:
list and confirm the run counts first, as the
[benchmark rules](https://github.com/chaobrain/braincell/blob/main/benchmarks/README.md#before-running-a-benchmark) describe.

## Name and place test files

Tests use `unittest.TestCase` classes collected by pytest.

- **A module `foo.py` has its tests in a sibling `foo_test.py`** *(checked by pre-commit)*: `braincell/io/neuromorpho/client.py` is tested by `braincell/io/neuromorpho/client_test.py`. Never use `test_*.py`, a bare `test.py` (it matches no collection pattern and is silently skipped), or a `tests/` directory.
- The only `test_*.py` files are the NEURON comparison suites under `validation/neuron/cable/tests/` and `validation/neuron/channel_no_conc/tests/`; `pyproject.toml` documents that exception.
- **The `foo` in `foo_test.py` names a real sibling module.** When a test file grows to cover several modules, split it, one file per module whose API the tests call. A test that drives a higher-level entry point (`SwcReader`, `Cell`, `Network.run`) belongs with that entry point's module, even when the behavior under test originates deeper.
- **Package-wide guards go in `<package>/__init___test.py`**: an AST scan over a package's import graph, or a docstring check with one allowlist for the package. `braincell/channel/__init___test.py` and `braincell/_compute/__init___test.py` are examples.
- **Shared helpers that are not tests** go in a private `_testing.py` in the same package, so that pytest does not collect them. `braincell/io/neuromorpho/_testing.py`, for example, provides the `FakeResponse` and `FakeSession` doubles that keep the NeuroMorpho tests offline.
- **A new `Morphology` edit adds one row to `EDIT_CASES`** in `braincell/morph/_testing.py`. The morph and Cell tests then check, for every row, that each cached read after the edit (metrics, branch orders, ranges, plotting data, the Cell's CVs and placements) equals the same read on `cold_rebuild(morpho)`, a copy built from the raw node tables. `SABOTAGE_CASES` holds an edit that skips the edit gate; each of those tests asserts that it is reported, which shows the check can fail.
- **Optional-dependency skips travel with the tests they guard.** When you split a file guarded by a module-level `pytestmark = pytest.mark.skipif(...)`, the destination files already hold unguarded tests, so re-express the guard per class (`@unittest.skipUnless`) or per function, and request any autouse fixture explicitly. Put shared autouse fixtures in a `conftest.py`, because pytest resolves fixtures by name and cannot import them from `_testing.py`.

## Write reproduction and regression tests

For bug fixes, add a minimal failing reproduction first and retain it as a regression test after the fix.
For new features, choose observable outcomes: return shapes, units, state changes, event timing, or numerical error.
Shared behavior changes also need validation through the Cell or Network path that uses them.

## Meet the coverage gates

The `coverage` job in CI runs `pytest braincell/ -n auto --dist loadfile --cov=braincell` on the latest JAX as three shards in parallel. With `BRAINCELL_TEST_SHARD=<index>/<count>` set, the root `conftest.py` keeps only the tests of the files that `.github/ci_shards.py` assigns to that shard. Every shard collects the whole session, so each test file runs in exactly one shard, including new ones. The `coverage-report` job then combines the three shards' data and enforces two tiers:

1. **Total floor.** Total coverage of `braincell/` (branch coverage on; `*_test.py` and `braincell/_version.py` omitted, see `[tool.coverage]` in `pyproject.toml`) must not fall below `report.fail_under`. The floor started at the total measured on `main` when the gate was introduced, rounded down; before each minor release it is raised to the current total, rounded down.
2. **Changed lines.** On a pull request, the lines it adds or changes must be at least 90 % covered, checked by `diff-cover coverage.xml --compare-branch=origin/main --fail-under=90`.

The shards are balanced by the measured seconds per test file in `.github/test_durations.json`. A file missing from that table is estimated from its size. When the shards drift apart (`python .github/ci_shards.py --show 3` prints the estimates), download the `junit-shard-*` artifacts of a recent `coverage` run, run `python .github/ci_shards.py --update junit-shard-*.xml`, and commit the table. Each shard also prints its 30 slowest tests (`--durations=30`).

Reach coverage by testing edge cases and critical paths, not trivial lines. To check a change locally before pushing:

```bash
pip install pytest-cov pytest-xdist diff-cover
pytest braincell/ -n auto --dist loadfile --cov=braincell --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main --fail-under=90
```

## Reuse morphology fixtures

Repository morphology data lives in `data/morphology/`; cerebellar variants live in `data/cerebellum/`. Import their paths from `braincell/io/_testing.py`, which defines `FIXTURE_DIR`, `VALID_SWC_FIXTURES`, `ALLOWED_TYPES` and `CEREBELLUM_FIXTURES`. Never recompute a path such as `Path(__file__).resolve().parents[N] / "data" / "morphology"`: `N` changes whenever a test file moves.

```python
import braincell as bc
from braincell.io._testing import FIXTURE_DIR

morpho, report = bc.io.SwcReader().read(
    FIXTURE_DIR / "three_points_soma.swc", return_report=True,
)
assert morpho.n_branches > 0
assert not report.has_errors
```

`MANIFEST.in` leaves `data/` out of the source distribution, so these tests run only from a repository checkout. They have no skip guard and fail rather than skip when the fixtures are missing.

## Accept numerical results

Declare the reference and tolerances before comparing results. Fix model parameters, units, temperature, initial state, solver, and dt.
Check dynamics against an analytical solution or trusted reference, vary dt to check convergence, and distinguish implementation errors from discretization errors.
Drive repeated model stepping with the compiled loops described in [Code Conventions](conventions.md#compiled-loops).
See the [Extension Guide](extending.md) for model-specific checks.

In the PR, list commands actually run, passes and skips, and the evidence supporting the results.
Run executable documentation examples separately; Sphinx website builds currently disable notebook execution.
