# Experimental Optimization Work

Optimization prototypes and studies are organized by responsibility. The
candidate core is deliberately separate from the experiments that validate or
apply it.

| Directory | Responsibility |
| --- | --- |
| [`optim/`](optim/) | Model-independent BPTT/RTRL gradient interfaces being evaluated for `braincell.optim` |
| [`optim_gradient_correctness/`](optim_gradient_correctness/) | One-CV and multicompartment numerical correctness |
| [`optim_gradient_scaling/`](optim_gradient_scaling/) | State, parameter, time, batch, and seed scaling |
| [`optim_training_comparison/`](optim_training_comparison/) | Matched end-to-end BPTT/RTRL training |
| [`optim_parameter_fitting/`](optim_parameter_fitting/) | Composable parameter-fitting framework and one-CV presets |
| [`optim_initialization/`](optim_initialization/) | Random, Sobol, and derivative-free initialization |
| [`optim_stimulus_design/`](optim_stimulus_design/) | Stimulus design and parameter identifiability |

The dependency direction is strict: experiment directories may import
`examples.experimental.optim`, while the core must not import an experiment.
Parameter selection, sharing, transforms, and materialization remain in the
public `braincell.trainable` package.

Generated data, traces, figures, and reports are stored in each experiment's
ignored `artifacts/` directory.

```bash
pytest -q \
  examples/experimental/optim \
  examples/experimental/optim_gradient_correctness \
  examples/experimental/optim_gradient_scaling \
  examples/experimental/optim_training_comparison \
  examples/experimental/optim_parameter_fitting \
  examples/experimental/optim_initialization \
  examples/experimental/optim_stimulus_design
```
