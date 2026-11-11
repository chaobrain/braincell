# IO examples

- [Morphology checkpoints](morphology-checkpoint.ipynb): save and restore branches and morphologies; generated files go to `artifacts/checkpoint/`.
- [NeuroMorpho](neuromorpho.ipynb): compare `braincell`'s morphology metrics with NeuroMorpho's published measurements; optionally query and download neurons and inspect cached metadata.

The checkpoint tutorial uses the shared `data/morphology/branched_dend.swc` fixture.
The NeuroMorpho notebook runs offline by default, on the seven neurons in
`data/morphology/neuromorpho/` that the regression test
`braincell/io/neuromorpho/regression_test.py` also uses. With `DOWNLOAD = True`
it queries NeuroMorpho, which needs network access, and caches the downloads in
`data/neuromorpho/` (ignored by git).
See the [IO API reference](../../docs/apis/io.rst) for interface contracts.
