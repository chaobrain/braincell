# Development Troubleshooting

First identify the Python environment, source location, and failing stage, then narrow the problem to a reproducible command.
When reporting an issue, include the command, versions, complete exception, and minimal input.

## Source changes have no effect

Check that the interpreter imports the current checkout:

```bash
python -m pip show braincell
python -c "import sys, braincell; print(sys.executable); print(braincell.__file__)"
```

If the path points to another environment or installed copy, return to the intended environment and run
`python -m pip install -e ".[dev]"` from the repository root. Then restart any running Python process or notebook kernel.

## Missing test, plotting, or documentation dependencies

Use `.[dev]` for contribution development. For individual tasks, use `.[testing]` for tests, `.[doc]` for documentation,
`.[vis]` for plotting (it installs `braintools[morphology]`), and `.[io]` for the NeuroMorpho client.
The extras are defined in [pyproject.toml](https://github.com/chaobrain/braincell/blob/main/pyproject.toml).

If a message reports a fallback from GPU to CPU, check the active devices:

```bash
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

The root conftest sets a CPU environment for ordinary tests. See the
[Installation Guide](../getting_started/installation.ipynb) for GPU installation and device selection.

## Tests are not collected or input files are missing

Run `python -m pytest` from the repository root, using `--collect-only` to inspect the target test file first.
Test files follow the naming rule in [Testing](testing.md#name-and-place-test-files); collection is configured in `pyproject.toml`.
Import morphology fixtures from `braincell.io._testing`; see [Testing](testing.md#reuse-morphology-fixtures) for an example.
Prepare other workflow data and optional dependencies according to their local instructions.

## Documentation builds fail or links point to missing pages

Install documentation dependencies, then build from the repository root:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Check new warnings using the source filenames and line numbers in the log. If the build hangs, check that it was not started with `-j`: a parallel build forks a process in which JAX already runs threads.
Internal references in Markdown, RST, and notebooks use the same document names. When converting formats, preserve the document name and remove the old file to avoid duplicate names.
A successful website build does not establish that notebook code has run.

Regenerating notebooks containing interactive PyVista plots needs `trame-pyvista` for the HTML export (pyvista 0.49 and later export through it). The `doc` extra installs it; for an interactive session add JupyterLab and ipywidgets:

```bash
python -m pip install -e ".[doc]"
python -m pip install ipywidgets "jupyterlab>=3"
```

Use `vis3d(notebook=True, jupyter_backend="html")` and save the executed outputs so the published page includes the exported interactive HTML.
See the [Vis API](../apis/vis.rst) for call parameters.

## Modeling and numerical issues

| Symptom | Where to look |
| --- | --- |
| Physical units or parameter shapes do not match | [Units](../concepts/units.ipynb), parameter descriptions in the [API reference](../apis/braincell.rst) |
| Problems with dt/duration, recorded results, or continued runs | [Cells](../concepts/cells.ipynb), the `Cell.run` docstring |
| Solver name or supported model does not match | [Integration](../concepts/integration.ipynb), [Integration API](../apis/integration.rst) |
| Numerical divergence or disagreement with a reference | [Numerical acceptance](testing.md#accept-numerical-results) |
| SWC/ASC reading fails | [IO API](../apis/io.rst) |

### `RuntimeProgramInputMismatch` after switching precision

jaxlib 0.12 caches the device copy of a host NumPy array per array object, and hands that copy back for as long as
something holds it, whatever `jax_enable_x64` is by then. A compiled program that captured the array holds it. So a
module-level NumPy array used by a float64 model that is still alive reaches a float32 operation as a float64 buffer
(or the reverse), and JAX raises `JaxRuntimeError: INVALID_ARGUMENT: ... RuntimeProgramInputMismatch: ... expected
parameter 0 of size 4 (f32[]) but got buffer with incompatible size 8 (f64[])`. `jax.clear_caches()` does not help
while the model holds its program.

braincell's own code does not pass such shared arrays to JAX: it holds the gas and Faraday constants as NumPy scalars,
which jaxlib does not cache per object (before OPT-17, brainunit's `u.gas_constant` and `u.faraday_constant`
triggered this in every Nernst ion and GHK channel). Your own mechanisms can still trigger it, for example one that
multiplies by `u.faraday_constant` when models at both precisions live in one process. Use
`u.Quantity(np.float64(u.faraday_constant.mantissa), unit=u.faraday_constant.unit)`, a Python float or an array made at
the current precision instead, or delete the models of the other precision and call `gc.collect()` before switching.

If the problem remains unresolved, open an [issue](https://github.com/chaobrain/braincell/issues) with a runnable reproduction and environment details.
