# Shared reference data

- [Cerebellum](cerebellum/README.md): morphology, MOD sources, parameter tables
  and original reference files grouped by cell model.
- `morphology/`: generic morphology and IO fixtures, including simulator benchmark `n144.swc`.
- `mechanisms/testing/`: synthetic MOD fixtures for numerical validation.
- `reference_traces/`: fixed reference input/output traces.

Data is read by examples, validation and benchmarks. Python adapters, compiler
commands and comparisons live with their workflows. Compiled libraries and
new run outputs are written to workflow `artifacts/`, preserving these sources.

See [repository organization](../docs/repository.md) for top-level directory responsibilities.
