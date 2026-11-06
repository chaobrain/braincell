---
myst:
  heading_anchors: 2
---

# Testing

Start with tests for the changed modules, then check integration behavior according to the scope of the change.
BrainCell uses pytest with tests next to the source they cover; collection is configured in
[pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml).
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

Tests use CPU JAX and headless Matplotlib as configured by the root conftest.
GPU performance and large numerical comparisons require separate workflows; unit tests do not establish those results.

## Validation and performance workflows

```bash
python -m pytest braincell/experimental/optim -q
python -m pytest validation/optim -q
python -m pytest benchmarks/performance/optim_gradient_scaling/analysis/common/report_test.py -q
python -m pytest validation/neuron/cable/tests -q
```

NEURON mechanism comparisons require compiling the relevant models first. See the
[Validation Guide](https://github.com/chaobrain/braincell/blob/main/validation/neuron/README.md) for entry points and dependencies.
The report tests above use synthetic records only. Other benchmark tests may run real measurements or profiling, so do not treat pytest over the entire benchmark directory as an ordinary regression check.
Before execution, specify and confirm rounds, warmup, timing repetitions, and extra runs according to the
[Benchmark rules](https://github.com/chaobrain/braincell/blob/main/benchmarks/AGENTS.md).

## Write reproduction and regression tests

For bug fixes, add a minimal failing reproduction first and retain it as a regression test after the fix.
For new features, choose observable outcomes: return shapes, units, state changes, event timing, or numerical error.
Shared behavior changes also need validation through the Cell or Network path that uses them.

Name new tests `*_test.py` to match the adjacent source module. See the
[repository testing rules](https://github.com/chaobrain/braincell/blob/main/AGENTS.md#testing) for splitting tests, package export checks, shared helpers, and optional-dependency skips.

## Reuse morphology fixtures

Repository morphology data lives in `data/morphology/`. Access it through shared constants:

```python
import braincell as bc
from braincell.io._testing import FIXTURE_DIR

morpho, report = bc.io.SwcReader().read(
    FIXTURE_DIR / "three_points_soma.swc", return_report=True,
)
assert morpho.n_branches > 0
assert not report.has_errors
```

The same helper module provides `VALID_SWC_FIXTURES`, `ALLOWED_TYPES`, and `CEREBELLUM_FIXTURES`, which points to the original test variants under `data/cerebellum/`.
Use existing test doubles for HTTP and other external services, such as `braincell/io/neuromorpho/_testing.py`, so ordinary regression tests can run offline.

## Accept numerical results

Declare the reference and tolerances before comparing results. Fix model parameters, units, temperature, initial state, solver, and dt.
Check dynamics against an analytical solution or trusted reference, vary dt to check convergence, and distinguish implementation errors from discretization errors.
Use the repository's brainstate compiled-loop conventions for repeated model stepping.
See the [Extension Guide](extending.md) for model-specific checks.

In the PR, list commands actually run, passes and skips, and the evidence supporting the results.
Run executable documentation examples separately; Sphinx website builds currently disable notebook execution.
