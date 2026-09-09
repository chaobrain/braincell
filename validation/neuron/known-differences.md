# Known numerical differences

The following failures were reproduced on both the pre-migration commit
`bd8e2f9` and the reorganized checkout on 2026-09-08, using NEURON 8.2.6,
JAX 0.10.1 and double precision. The original assertions remain in place.

- **DCN SK initialization:** the
  [BrainCell channel smoke test](channel_no_conc/tests/test_braincell_runner.py)
  changes a scan carry from scalar `float64[]` to `float64[1,1]`. Resolve the
  ion/channel initialization shape before using this case as an accuracy baseline.
- **Sinusoidal channel stimulus:** the
  [comparison test](channel_no_conc/tests/test_compare.py) reports voltage MAE
  `0.015916517057767133 mV`, above its `2e-5 mV` limit. Investigate stimulus
  sampling and voltage alignment under the existing protocol.
- **SC morphology geometry:** the
  [morphology comparison](morph/neuron_diff_test.py) reports maximum Euclidean
  distance difference `8.559982695066495 um`, above its `1e-4 um` limit.
  This uses the preserved IO fixture variant, not the whole-cell morphology.

Run each workflow in a fresh process. Combining cell and channel suites in one
pytest process can attempt to register the same NEURON mechanism from both the
cell and channel libraries; NEURON then raises a duplicate-name error. The
separate build directories preserve each workflow's mechanism selection.

```bash
pytest -q validation/neuron/cable/tests
pytest -q validation/neuron/cell
pytest -q validation/neuron/channel_no_conc/tests
pytest -q validation/neuron/morph
```

Model-specific import and comparison status is maintained in
[import progress](cerebellum-import-progress.md). These discrepancies need their
own numerical fixes; moving files preserves the model inputs and tolerance values.
