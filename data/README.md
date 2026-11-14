# Shared reference data

- [Cerebellum](cerebellum/README.md): morphology, MOD sources, parameter tables
  and original reference files grouped by cell model.
- `morphology/`: generic morphology and IO fixtures, including simulator benchmark `n144.swc`,
  [`neuromorpho/`](morphology/neuromorpho/README.md), seven NeuroMorpho.Org neurons (CC BY 4.0)
  for the offline NeuroMorpho regression, and [`neuroml2/`](morphology/neuroml2/README.md),
  NeuroML2 cells for the NeuroML2 reader (one from libNeuroML, BSD-3-Clause).
- `mechanisms/testing/`: synthetic MOD fixtures for numerical validation.
- [`modeldb/`](modeldb/README.md): unmodified ModelDB MOD files without a license statement
  (MECH-48, D13), one folder per accession, with their source and SHA-256, for the ion-pool
  comparisons against NEURON.
- `reference_traces/`: fixed reference input/output traces.

Data is read by examples, validation and benchmarks. Python adapters, compiler
commands and comparisons live with their workflows. Compiled libraries and
new run outputs are written to workflow `artifacts/`, preserving these sources.

See [Repository Layout](../docs/developer/project_layout.md) for top-level directory responsibilities.
