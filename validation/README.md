# Numerical validation

Validation compares numerical results under explicit protocols and tolerances.

- [NEURON comparisons](neuron/README.md): morphology, channels, ions, synapses,
  cables and whole cerebellar cells.
- [Optimization](optim/README.md): BPTT/RTRL gradients, sensitivities and matched training.

Tests of reusable package code remain co-located under `braincell/`.
Timing and scale measurements live in [benchmarks](../benchmarks/README.md).
Shared reference sources live in [data](../data/README.md).
Run commands below are relative to the repository root; each workflow records its
optional dependencies and outputs. Generated outputs go to ignored `artifacts/`.

```bash
pytest -q validation/optim
pytest -q validation/neuron/cable/tests
```

NEURON comparisons require NEURON and use double precision. Mechanism-based
comparisons additionally require compiling their reference MOD sources;
see [NEURON setup](neuron/README.md).
