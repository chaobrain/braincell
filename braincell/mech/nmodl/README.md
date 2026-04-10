# BrainCell NMODL Pipeline

This directory contains a BrainCell-only conversion pipeline for one focused target:

- `one_ion_hh_ohmic`

That target means:

- exactly one `USEION`
- Hodgkin-Huxley gate dynamics
- ohmic current
- BrainCell current sign convention: `g_max * gates * (Ion.E - V)`

## Pipeline

```text
.mod
-> AST
-> RawBlocks
-> CanonicalBlocks
-> one_ion_hh_ohmic IR
-> templates/one_ion_hh_ohmic.py
-> generated BrainCell channel
```

The stable flow is always three steps:

1. Step 1: parse and normalize
2. Step 2: build a BrainCell IR family
3. Step 3: render with a matching template

What changes over time is not the number of steps, but the implementation variant selected inside each step.

## Layout

- `pipeline/`
  - `inspect_ast/`: Step 1 code and variants
  - `inspect_ir/`: Step 2 code and variants
  - `render/`: Step 3 code and variants
  - `flow.py`: fixed three-step orchestration
  - `registry.py`: allowed step combinations
- `templates/`
  - `one_ion_hh_ohmic.py`: the only supported code-generation template
- `steps/`
  - `inspect_ast.py`: stable Step 1 entry point
  - `inspect_ir.py`: stable Step 2 entry point
  - `render_preview.py`: stable Step 3 entry point
- `examples/`
  - `generate_braincell.py`: direct `.mod -> generated Python` conversion
  - `kv.mod`: positive single-ion `inf/tau` example
  - `na_alpha_beta.mod`: positive single-ion `alpha/beta` example
  - `hh.mod`: rejection example; intentionally unsupported because it is multi-ion
- `notebooks/`
  - `braincell_pipeline_walkthrough.ipynb`: step-by-step walkthrough
- `docs/`
  - `pipeline_overview.md`: architecture and data boundaries
  - `braincell_template_recommendations.md`: current template contract and limits

## Environment

The scripts need one of these NMODL Python API backends:

- `neuron.nmodl`
- `nmodl.dsl` from Blue Brain's `nmodl` package

Quick backend check:

```bash
python - <<'PY'
import importlib.util
for name in ("neuron.nmodl", "nmodl.dsl"):
    spec = importlib.util.find_spec(name)
    print(name, "->", spec.origin if spec else None)
PY
```

If neither backend is available, install the external `nmodl` package in the current environment or switch to an environment that already provides `neuron.nmodl`.

If Jinja2 is missing:

```bash
python -m pip install jinja2
```

## Commands

Current supported combination:

```text
canonical_default
-> one_ion_hh_ohmic
-> braincell_one_ion_hh_ohmic
```

Direct conversion:

```bash
python examples/generate_braincell.py examples/kv.mod
python examples/generate_braincell.py examples/na_alpha_beta.mod
python examples/generate_braincell.py examples/kv.mod --pipeline canonical_default__one_ion_hh_ohmic__braincell_one_ion_hh_ohmic
```

Inspection:

```bash
python steps/inspect_ast.py examples/kv.mod
python steps/inspect_ir.py examples/kv.mod
python steps/render_preview.py examples/kv.mod
python steps/render_preview.py examples/kv.mod -o /tmp/kv_channel.py
```

You can also select the step variants explicitly:

```bash
python steps/inspect_ir.py examples/kv.mod \
  --step1 canonical_default \
  --step2 one_ion_hh_ohmic

python steps/render_preview.py examples/kv.mod \
  --step1 canonical_default \
  --step2 one_ion_hh_ohmic \
  --step3 braincell_one_ion_hh_ohmic
```

Use `examples/hh.mod` to inspect a clean rejection:

```bash
python steps/inspect_ir.py examples/hh.mod
```

## Current Coverage

The current BrainCell target normalizes:

- single-ion type to a fixed base class:
  - `k -> PotassiumChannel`
  - `na -> SodiumChannel`
  - `ca -> CalciumChannel`
- gate kinetics to `inf/tau`
  - if source is `alpha/beta`, it is converted first
- gate names to original state names
- gate powers from the ohmic current expression
- current to BrainCell sign convention
- `Q10` to an explicit field per gate, default `1.0`
- `temp` to a single runtime entry, default `u.celsius2kelvin(23)`
- `Tref` to a fixed Kelvin constant
- unit stripping to `/ unit`, not `.to_decimal(...)`

The generator currently rejects:

- multi-ion channels
- non-ohmic currents
- gates that cannot be reduced to `inf/tau`
- advanced blocks such as `KINETIC`, `NET_RECEIVE`, `LINEAR`, `NONLINEAR`

## Notes

- `examples/kv.mod` is the best positive example for the first template.
- `examples/na_alpha_beta.mod` demonstrates `alpha/beta -> inf/tau`.
- `examples/hh.mod` remains useful as a rejection case.
- `examples/generate_braincell.py` is the main user-facing entry point.
- Future growth should happen by adding variants under `pipeline/inspect_ast`, `pipeline/inspect_ir`, and `pipeline/render`, then registering valid combinations in `pipeline/registry.py`.
