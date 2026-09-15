# Optimization examples

These workflows demonstrate parameter fitting, initialization and stimulus design
using `braincell.trainable` and `braincell.experimental.optim`.

Declare mechanisms and connections with valid initial parameters, call
`cell.init_state()` (or `network.init_state()`), then select fresh Views and
register all trainable parameters before creating the optimizer and gradient
engine. `reset_state()` retains current parameters; full `reset()` discards
runtime overrides and training registrations. See the
[lifecycle contract](../../docs/design/optim/current/api.md#lifecycle).

- [Parameter learning](parameter_learning/README.md): channel, ion, synapse and HH training tutorials.
- [Parameter fitting](parameter_fitting/README.md): composable models, datasets,
  losses, optimizer stages, presets and run commands.
- [Initialization](initialization/README.md): random, Sobol and derivative-free starts.
- [Stimulus design](stimulus_design/README.md): protocol generation, OED and identifiability.

The reusable [gradient implementation](../../braincell/experimental/optim/README.md)
is installed with BrainCell. [Accuracy validation](../../validation/optim/README.md)
and [scaling benchmarks](../../benchmarks/performance/optim_gradient_scaling/README.md)
provide separate verification and measurement workflows.

Run from the repository root:

```bash
pytest -q examples/optim
python -m examples.optim.parameter_fitting.run --help
```

Each workflow writes generated outputs to its ignored `artifacts/` directory.
Contracts, proposals and measured results live in [Optimization Design](../../docs/design/optim/TODO.md).

Existing local outputs moved with their workflows. Older ignored `online_learning` and `parameter_learning` files are preserved under `artifacts/legacy_experimental/`.
