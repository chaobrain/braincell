# Troubleshooting & FAQ

Common errors, what they mean, and how to fix them.

## "TypeError: expected a quantity" (or a bare-number rejection)

**Cause.** You passed a plain `float`/`int` where `braincell` expects a united
quantity. Units are mandatory everywhere (see {doc}`concepts/units`).

**Fix.** Attach a unit:

```python
# wrong
g_max = 0.03
# right
import brainunit as u
g_max = 0.03 * (u.mS / u.cm**2)
```

## "An NVIDIA GPU may be present … falling back to cpu"

**Cause.** JAX found a GPU but no CUDA-enabled `jaxlib` is installed, so it runs
on CPU. This is a *warning*, not an error — everything still works.

**Fix (to use the GPU).** Install the matching CUDA build:

```bash
pip install -U braincell[cuda12]   # or braincell[cuda13]
```

If you *intend* to run on CPU, you can ignore the message.

## "Cell.run(...) requires at least one placed probe"

**Cause.** A multi-compartment {class}`~braincell.Cell` simulation returns probe
traces, so it needs at least one probe to know what to record.

**Fix.** `place` a probe before running:

```python
import braincell.mech as mech
from braincell.filter import RootLocation

cell.place(RootLocation(0.5), mech.StateProbe("V"))
```

See {doc}`concepts/mechanisms`.

## "dt"/"duration" rejected as zero, negative, or unitless

**Cause.** `Cell.run(dt=..., duration=...)` validates its time arguments: they
must be positive `brainunit` time quantities.

**Fix.**

```python
result = cell.run(dt=0.1 * u.ms, duration=100 * u.ms)
```

## `ImportError` / `ModuleNotFoundError` for matplotlib, pyvista, plotly

**Cause.** Visualization backends are optional dependencies, imported lazily.

**Fix.** Install the visualization extra:

```bash
pip install -U braincell[vis]
```

Likewise, the NeuroMorpho client needs `braincell[io]`.

## A solver name isn't recognized

**Cause.** The `solver=` string doesn't match a registered integrator.

**Fix.** List the available names and pick one:

```python
import braincell.quad as quad
sorted(quad.all_integrators)
```

See {doc}`concepts/integration`.

## My simulation is unstable / blows up

**Cause.** Biophysical models are **stiff**; an explicit solver (e.g. `euler`,
`rk4`) needs a very small `dt` to stay stable.

**Fix.** Reduce `dt`, or switch to a solver designed for stiff systems —
`exp_euler` / `ind_exp_euler` for single-compartment channel kinetics, or the
staggered / Crank–Nicolson solvers for cable equations. See
{doc}`integration/index`.

## An SWC/ASC file fails to load

**Cause.** Reconstructions often have irregularities (unknown structure ids,
non-soma roots, disconnected points).

**Fix.** Read with a report and/or relaxed options to see and handle the issue:

```python
morpho, report = braincell.Morphology.from_swc("neuron.swc", return_report=True)
print(report)
```

See {doc}`file_formats/swc`.

## Still stuck?

- Re-read the relevant {doc}`concept page <concepts/index>`.
- Check the {doc}`API reference <apis/braincell>` for exact signatures.
- Search or open an issue at
  [github.com/chaobrain/braincell/issues](https://github.com/chaobrain/braincell/issues).
