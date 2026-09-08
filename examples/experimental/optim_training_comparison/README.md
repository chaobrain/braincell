# Training Comparison Experiments

`rtrl_bptt.py` runs matched Adam training with BPTT and exact RTRL, retaining
loss histories, parameter trajectories, timing, and comparison plots. This is
an end-to-end experiment, not part of the candidate optimization API.

Generated outputs live under ignored `artifacts/`.

```bash
pytest -q examples/experimental/optim_training_comparison
```
