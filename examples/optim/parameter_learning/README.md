# Parameter learning examples

- [Channel learning](channel_learning.ipynb): fit individual channel parameters to voltage traces.
- [Ion learning](ion_learning.ipynb): fit ion parameters using voltage or concentration targets.
- [Synapse learning](synapse_learning.ipynb): fit synaptic parameters, connection weights and detector thresholds; helper functions live in [synapse_learning.py](synapse_learning.py).
- [HH multistart training](train.ipynb): fit three conductance scales from parallel initial points; [trainable_hh_multistart.py](trainable_hh_multistart.py) provides the script entry point.

Run the example tests from the repository root:

```bash
pytest -q examples/optim/parameter_learning
```

The training examples use [fitting diagnostics](../parameter_fitting/README.md).
Pass an `artifact_dir` under this directory's `artifacts/` when saving training results.
See [Optimization Design](../../../docs/design/optim/TODO.md) for parameter contracts.

Follow-up: move gradient-comparison sections from the Synapse notebook into
validation tutorials, reusing the existing [gradient checks](../../../validation/optim/gradient_correctness/README.md).
