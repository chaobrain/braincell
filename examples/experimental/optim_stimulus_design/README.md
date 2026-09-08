# Stimulus Design Experiments

This directory contains the PRMLS dataset, local robust optimal experimental
design, and global parameter-ensemble identifiability studies.

- `dataset.py` builds the morphology, shared parameter roots, and protocol set.
- `robust_oed.py` evaluates local online information across a prior ensemble.
- `global_ensemble.py` evaluates global loss geometry and weak directions.
- Generated results live under ignored `artifacts/`.

```bash
pytest -q examples/experimental/optim_stimulus_design
```
