# NEURON accuracy comparisons

BrainCell and NEURON run matched morphology, mechanisms, initial conditions,
stimuli and integration settings. Each workflow defines its measured quantities
and tolerances; existing comparison gaps remain documented with that workflow.

- `morph/`, `channel_no_conc/`, `ion/`, `synapse/`, `cable/`: component comparisons.
- [Whole-cell comparisons](cell/README.md): BC, DCN, GoC, GrC, IO, PC and SC.
- [Import progress](cerebellum-import-progress.md): model-specific status and remaining differences.
- [Reference sources](../../data/cerebellum/README.md): shared model bundles.

## Setup and execution

Use a repository checkout with BrainCell, NEURON and a working `nrnivmodl`
compiler installed. Run comparisons in double precision. For NEURON 8.2's Python
compiler wrapper, install `setuptools<81` to provide `pkg_resources`.

```bash
python -m validation.neuron._mechanisms bc_ma2025 --kind cell
pytest -q validation/neuron/cable/tests

# Compile all sources used by the channel suite before its first run.
for model in bc_ma2025 dcn_su2015 goc_ma2020 grc_ma2020 io_zh2019 pc_ma2024 sc_ma2021 testing; do
    python -m validation.neuron._mechanisms "$model" --kind channel
done
pytest -q validation/neuron/channel_no_conc/tests
```

Compile the selected model before opening its `cell/<model>/run.ipynb`.
Channel configurations point to the corresponding `--kind channel` build.
Ion notebooks build their explicit mechanism subsets. Builds live under
`artifacts/mechanisms/`, independently of the read-only data sources. Restart
the Python process when changing a loaded mechanism version.

`_paths.py` owns repository source and build locations; `_mechanisms.py` owns
compilation. Model builders and comparison configuration remain in this tree,
so performance workflows can reuse them without duplicating model parameters.

Historical comparison reports remain with their workflow. Newly generated
traces, figures and reports use ignored `artifacts/` directories.

[Known differences](known-differences.md) records numerical failures reproduced before and after the repository migration, and the process isolation needed for mechanism libraries.
