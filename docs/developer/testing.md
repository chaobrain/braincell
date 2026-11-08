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
python -m pytest validation/neuron/channel_no_conc/tests -q   # after compiling its mechanisms
```

CI runs `validation/optim` and the benchmark scripts' tests in two tiers: modules marked `pytestmark = pytest.mark.slow` (more than about 25 s on their own) run only in the daily workflow, the rest on every pull request. A new test module there runs on every pull request unless you mark it slow.

Both NEURON suites run on every pull request, each in its own job (and so its own process). NEURON mechanism comparisons require compiling the relevant models first. See the
[Validation Guide](https://github.com/chaobrain/braincell/blob/main/validation/neuron/README.md) for entry points and dependencies.
The benchmark tests check the scripts on small configurations and synthetic records; nothing they time is a benchmark result. Running a benchmark itself is different:
list and confirm the run counts first, as the
[benchmark rules](https://github.com/chaobrain/braincell/blob/main/benchmarks/README.md#before-running-a-benchmark) describe.

## Name and place test files

Tests use `unittest.TestCase` classes collected by pytest.

- **A module `foo.py` has its tests in a sibling `foo_test.py`** *(checked by pre-commit)*: `braincell/io/neuromorpho/client.py` is tested by `braincell/io/neuromorpho/client_test.py`. Never use `test_*.py`, a bare `test.py` (it matches no collection pattern and is silently skipped), or a `tests/` directory.
- The only `test_*.py` files are the NEURON comparison suites under `validation/neuron/cable/tests/` and `validation/neuron/channel_no_conc/tests/`; `pyproject.toml` documents that exception.
- **The `foo` in `foo_test.py` names a real sibling module.** When a test file grows to cover several modules, split it, one file per module whose API the tests call. A test that drives a higher-level entry point (`SwcReader`, `Cell`, `Network.run`) belongs with that entry point's module, even when the behavior under test originates deeper.
- **Package-wide guards go in `<package>/__init___test.py`**: an AST scan over a package's import graph, or a docstring check with one allowlist for the package. `braincell/channel/__init___test.py` and `braincell/_compute/__init___test.py` are examples. This and `braincell/vis_baseline_test.py` (the pixel baselines, which draw through several modules' adapters into braintools) are the only test files without a sibling module.
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

## Compare figures with pixel baselines

Plotting lives in `braintools.visualize.morphology`. `braincell/vis_baseline_test.py` draws a few representative figures from braincell objects, through `Morphology.vis2d`, `Cell.topology_data` and `NodeTree.to_topology_data` (the `plot2d` layouts, value colouring with a colorbar, the dendrogram, the Sholl plot and the two topology graphs), and compares them pixel by pixel with PNGs in `braincell/vis_baselines/`; braintools compares the same images in its own tests. A test draws on `self.new_axes()` and calls `self.assert_baseline(fig, name)`. Rendering is pinned (Agg, default `rcParams`, the bundled DejaVu Sans, fixed size and dpi), and the comparison uses `matplotlib.testing.compare.compare_images` with `tol=0`. Pixels are compared on Linux only; other platforms skip, because font rasterisation differs.

When a deliberate change, or a Matplotlib upgrade, changes a figure, regenerate the baselines on Linux and look at the new images before committing them:

```bash
BRAINCELL_UPDATE_BASELINES=1 pytest braincell/vis_baseline_test.py
git diff --stat braincell/vis_baselines
```

Say in the PR why the images changed; an image that changes here because of braintools changes there too. When a comparison fails, the message names a temporary directory that holds the rendered image and a `-failed-diff.png`. `MANIFEST.in` keeps the baselines out of the source distribution, and the wheel does not include them.

## Accept numerical results

Declare the reference and tolerances before comparing results. Fix model parameters, units, temperature, initial state, solver, and dt.
Check dynamics against an analytical solution or trusted reference, vary dt to check convergence, and distinguish implementation errors from discretization errors.
Drive repeated model stepping with the compiled loops described in [Code Conventions](conventions.md#compiled-loops).
See the [Extension Guide](extending.md) for model-specific checks.

In the PR, list commands actually run, passes and skips, and the evidence supporting the results.
Run executable documentation examples separately; Sphinx website builds currently disable notebook execution.
