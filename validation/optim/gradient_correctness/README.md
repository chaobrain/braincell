# Gradient Correctness Experiments

This directory validates exact RTRL against BPTT and finite differences without
mixing those checks with performance conclusions.

- `single_cv_sensitivity.ipynb` develops the one-CV sensitivity example.
- `multicv_hh.py` compares compact RTRL, full RTRL, BPTT, and directional finite
  differences on a branched multicompartment HH cell.
- `gradient_diagnostics.ipynb` inspects sensitivity, learning-signal, direct,
  and eligibility-gradient decompositions.
- `autapse.py` compares full-state RTRL and BPTT through single-Cell feedback
  and two-Cell event delivery. Scope and results:
  [Synapse and Network Learning Results](../../../docs/design/optim/current/results/synapse-network-learning.md).
- `bidirectional.py` checks independent populations of sizes two and three,
  trainable parameters on both sides, twelve bidirectional contacts, and fits
  with both gradient methods. Current results:
  [Bidirectional Population Learning](../../../docs/design/optim/current/results/synapse-network-learning.md#双向-population).

```bash
pytest -q validation/optim/gradient_correctness
```
