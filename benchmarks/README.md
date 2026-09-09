# Performance benchmarks

Performance workflows measure compilation, steady execution, memory and scaling.
Keep numerical protocols fixed and synchronize device execution before recording time.

Before launching a benchmark or profiling run, follow the
[benchmark execution rules](AGENTS.md): list all independent rounds, processes,
first executions, warmups, timed repeats, extra validation runs and time budgets,
and obtain user confirmation. Script defaults and commands below are not run authorization.

- [Gradient scaling](performance/optim_gradient_scaling/README.md): BPTT/RTRL
  state, parameter, time, batch and seed sweeps.
- [Parameter fitting](performance/parameter_fitting/README.md): historical capacity and training-throughput evidence.
- [Simulator comparison](performance/simulator_compare/README.md): BrainCell,
  NEURON and Jaxley under the same model and stimulus.
- [Sampling](performance/sampling/README.md): connection and continuous-region sampling.
- [Solvers](performance/solvers/README.md): isolated DHS solver measurements.
- [Synapse events](performance/synapse_events/README.md): NetStim scaling,
  schedule lookup, synapse aggregation and CPU/GPU timing.
- [Profiling](profiling/README.md): construction phases, XPlane and device traces.

```bash
python -m benchmarks.performance.optim_gradient_scaling.benchmark --help
python -m benchmarks.profiling.profile_simulation --help
```

GPU and simulator requirements are documented with each workflow. Generated
outputs belong to ignored `artifacts/`; reviewed measurement summaries and their
environments belong to Git-tracked `results/` beside each experiment. Design
cites those summaries within the relevant proposal or current architecture document. Accuracy studies live in [validation](../validation/README.md).
