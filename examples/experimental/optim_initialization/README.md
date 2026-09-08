# Initialization Experiments

This directory compares bounded random, Sobol, and optional Nevergrad candidate
generation before exact-gradient training.

- `dc_protocol_dataset.py` defines the shared DC protocol dataset.
- `hybrid_initialization.py` implements candidate search, selection, and the
  RTRL/Adam handoff experiment.
- Generated results live under ignored `artifacts/`.

```bash
pytest -q examples/experimental/optim_initialization
```
