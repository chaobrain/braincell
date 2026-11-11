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
```

The root `conftest.py` forces CPU JAX (`JAX_PLATFORMS=cpu`) and headless Matplotlib (`MPLBACKEND=Agg`).
GPU performance and large numerical comparisons require separate workflows; unit tests do not establish those results.

## Validation and performance workflows

```bash
python -m pytest braincell/experimental/optim -q
python -m pytest validation/optim benchmarks -m "not slow" -q   # every pull request
python -m pytest validation/optim benchmarks -m slow -q         # daily CI run
python -m pytest validation/neuron/cable/tests -q
```

CI runs `validation/optim` and the benchmark scripts' tests in two tiers: modules marked `pytestmark = pytest.mark.slow` (more than about 25 s on their own) run only in the daily workflow, the rest on every pull request. A new test module there runs on every pull request unless you mark it slow.

The NEURON comparison suites are tiered the same way, each suite in its own job (and so its own process):

| Tier | CI job | What runs |
| --- | --- | --- |
| Every pull request (in `ci-ok`) | `test_linux_neuron_compare_cable` | `validation/neuron/cable/tests` |
| Every pull request (in `ci-ok`) | `test_linux_neuron_compare_channel` | `validation/neuron/channel_no_conc/tests` and the shared `_clamp_test.py`, `_tolerance_test.py`, `_convergence_test.py`, `_mechanisms_test.py` under `validation/neuron/`, with `-m "not slow"`; then `validation/neuron/ion` with `-m "not slow"` in a separate process |
| Daily run only | `test_macos`, `test_windows` | `pytest braincell/` on macOS and Windows (Python 3.13); kept out of pull requests to stay within the concurrent-job limit |
| Daily run only | `slow_tests_neuron` | `-m slow` of `validation/neuron/channel_no_conc/tests` (the per-family `dt` convergence tests), then of `validation/neuron/ion` (the pool comparisons) in a separate process |

Both tiers compile the eight channel models first. To reproduce them locally:

```bash
for model in bc_ma2025 dcn_su2015 goc_ma2020 grc_ma2020 io_zh2019 pc_ma2024 sc_ma2021 testing; do
    python -m validation.neuron._mechanisms "$model" --kind channel
done
# Pull-request tier.
python -m pytest validation/neuron/channel_no_conc/tests \
    validation/neuron/_clamp_test.py validation/neuron/_tolerance_test.py \
    validation/neuron/_convergence_test.py validation/neuron/_mechanisms_test.py -m "not slow" -q
python -m pytest validation/neuron/ion -m "not slow" -q
# Daily tier.
python -m pytest validation/neuron/channel_no_conc/tests -m slow -q
python -m pytest validation/neuron/ion -m slow -q
```

A NEURON comparison module that takes more than about 25 s on its own gets `pytestmark = pytest.mark.slow` and so moves to the daily run; the daily job selects by marker, so it needs no workflow change. A red daily run opens a `main is red` issue, and fixing it is the first task of the day. To run the daily jobs on a branch, dispatch the daily workflow on it (`gh workflow run CI-daily.yml --ref <branch>`); a red run there opens an issue titled `<branch> is red`.

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
- **Optional-dependency skips travel with the tests they guard.** When you split a file guarded by a module-level `pytestmark = pytest.mark.skipif(...)`, the destination files already hold unguarded tests, so re-express the guard per class (`@unittest.skipUnless`) or per function, and request any autouse fixture explicitly. Put shared autouse fixtures in a `conftest.py`, because pytest resolves fixtures by name and cannot import them from `_testing.py`.

## Write reproduction and regression tests

For bug fixes, add a minimal failing reproduction first and retain it as a regression test after the fix.
For new features, choose observable outcomes: return shapes, units, state changes, event timing, or numerical error.
Shared behavior changes also need validation through the Cell or Network path that uses them.

## Meet the coverage gates

The `coverage` job in CI runs `pytest braincell/ --cov=braincell --cov-report=xml` on the latest JAX and enforces two tiers:

1. **Total floor.** Total coverage of `braincell/` (branch coverage on; `*_test.py` and `braincell/_version.py` omitted, see `[tool.coverage]` in `pyproject.toml`) must not fall below `report.fail_under`. The floor started at the total measured on `main` when the gate was introduced, rounded down; before each minor release it is raised to the current total, rounded down.
2. **Changed lines.** On a pull request, the lines it adds or changes must be at least 90 % covered, checked by `diff-cover coverage.xml --compare-branch=origin/main --fail-under=90`.

Reach coverage by testing edge cases and critical paths, not trivial lines. To check a change locally before pushing:

```bash
pip install pytest-cov diff-cover
pytest braincell/ --cov=braincell --cov-report=xml
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
