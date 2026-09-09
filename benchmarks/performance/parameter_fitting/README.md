# Parameter Fitting Performance Evidence

This directory preserves historical batch-capacity, throughput and training-cost comparisons.
It has no separate benchmark runner. Supporting models and training scripts are maintained
in the [parameter fitting workflow](../../../examples/optim/parameter_fitting/README.md);
its README documents dependencies, scripts and supported arguments. Follow those interfaces
when preparing a new experiment; the historical measurements are not a new runnable suite.

[Batch size and throughput](results/batch-size-and-throughput.md) records the measured
environment, numerical results, training quality and missing historical metadata.
Raw outputs may exist only in the original user's ignored artifacts. Any future local
performance outputs belong in this directory's ignored `artifacts/`, while reviewed
summaries belong in `results/`. Reusable interfaces and design rationale are described
in the [optimization architecture](../../../docs/design/optim/current/architecture.md).

Before measuring, follow the [execution rules](../../AGENTS.md): confirm the full
configuration matrix, independent rounds, first executions, warmups, timed repeats,
extra validation and time budget. Commands below are templates, not authorization.
Replace angle-bracket placeholders before running from the repository root.
