# IO examples

- [Morphology checkpoints](morphology-checkpoint.ipynb): save and restore branches and morphologies; generated files go to `artifacts/checkpoint/`.
- [NeuroMorpho](neuromorpho.ipynb): query and download neurons, inspect cached metadata and compare morphology measurements.

The checkpoint tutorial uses the shared `data/morphology/branched_dend.swc` fixture.
NeuroMorpho downloads require network access and are cached in `data/neuromorpho/`.
See [IO Design](../../docs/design/io/TODO.md) for interface contracts.

Follow-up: extract NeuroMorpho metric comparisons into validation with a fixed
dataset and explicit tolerances; keep query, download and cache usage here.
