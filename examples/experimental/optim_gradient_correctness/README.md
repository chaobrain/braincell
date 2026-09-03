# Gradient Correctness Experiments

This directory validates exact RTRL against BPTT and finite differences without
mixing those checks with performance conclusions.

- `single_cv_sensitivity.ipynb` develops the one-CV sensitivity example.
- `multicv_hh.py` compares compact RTRL, full RTRL, BPTT, and directional finite
  differences on a branched multicompartment HH cell.
- `gradient_diagnostics.ipynb` inspects sensitivity, learning-signal, direct,
  and eligibility-gradient decompositions.

```bash
pytest -q examples/experimental/optim_gradient_correctness
```
