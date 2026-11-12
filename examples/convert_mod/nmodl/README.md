# BrainCell NMODL Pipeline

This directory provides an inspectable `.mod -> BrainCell density channel` conversion chain. The point is not to "fill in a template directly" but to lower NMODL layer by layer into BrainCell's own intermediate representations.

The first version stably supports target mechanisms with:

- a single `USEION`
- Hodgkin-Huxley-style gating dynamics
- an ohmic current
- generated BrainCell Python density-channel source

Representative samples:

- `mod_files/kv.mod`
  - positive case, `inf/tau`
- `mod_files/na_alpha_beta.mod`
  - positive case, `alpha/beta`
- `mod_files/hh.mod`
  - negative case: kept in the semantic layer, but rejected in target lowering because it uses several ions and `NONSPECIFIC_CURRENT`

If you are mainly filling in NMODL fundamentals rather than changing the pipeline, start with:

- `docs/nmodl_mental_model.md`
  - explains `PARAMETER` / `ASSIGNED` / `STATE` and `INITIAL` / `BREAKPOINT` / `DERIVATIVE` at three levels: a minimal leak mechanism, `hh.mod`, and `NET_RECEIVE`
  - also summarizes the key differences between `NEURON 8.2.x`, `9.0` and Arbor

## Current pipeline

```text
.mod
-> parser AST
-> raw_blocks
-> canonical_blocks
-> bc_ast
-> semantic_ir
-> target_ir
-> rendered Python
-> render validation
```

The external CLI keeps three entry steps, but the internal boundaries are four layers:

1. Step 1
   - parse and canonicalize
   - produce `raw_blocks`, `canonical_blocks`, `bc_ast`
2. Step 2
   - build `semantic_ir`
   - then lower to `target_ir`
3. Step 3
   - render the Python source
   - validate with `compile + exec`

## Directory layout

- `mod_files/`
  - sample `.mod` inputs
- `examples/`
  - `generate_braincell.py`
    - runs the full pipeline in one command and writes the final `.py`
  - `walktrough.ipynb`
    - layer-by-layer walkthrough notebook
- `steps/`
  - `inspect_ast/`
    - Step 1
  - `inspect_ir/`
    - Step 2
  - `render/`
    - Step 3
  - `model.py`
    - dataclass boundaries of `bc_ast` / `semantic_ir` / `target_ir`
  - `semantic_ir.py`
    - semantic recovery
  - `target_ir.py`
    - BrainCell density-channel lowering
- `templates/`
  - `density_channel.py`
    - the template the renderer currently uses
- `docs/`
  - architecture notes

## Dependencies

Step 1 AST parsing depends on an NMODL Python backend. The code tries, in order:

- `neuron.nmodl`
- `nmodl.dsl`

Quick check:

```bash
python - <<'PY'
import importlib.util
for name in ("neuron.nmodl", "nmodl.dsl"):
    spec = importlib.util.find_spec(name)
    print(name, "->", spec.origin if spec else None)
PY
```

If neither is available, install Blue Brain's `nmodl` package, or switch to an environment that already provides `neuron.nmodl`.

The rendering stage also needs `jinja2`:

```bash
python -m pip install jinja2
```

## Entry points

### Step 1: AST / blocks / bc_ast

```bash
python examples/convert_mod/nmodl/steps/inspect_ast/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

Main outputs:

- `ast_root_type`
- `block_counts`
- `raw_blocks`
- `canonical_blocks`
- `bc_ast`
- `ast_json`

### Step 2: semantic_ir / target_ir

```bash
python examples/convert_mod/nmodl/steps/inspect_ir/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

Main outputs:

- `summary`
- `semantic_ir`
- `target_ir`
- `braincell_ir`

`braincell_ir` is currently the same as the target IR payload; it is kept for compatibility with the old entry point.

### Step 3: render preview / validation

```bash
python examples/convert_mod/nmodl/steps/render/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

Main outputs:

- `summary`
- `render_preview`
- `validation`

`validation` reports whether the generated source can:

- `compile`
- `exec`
- yield the target class

### Generate the source in one step

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

The pipeline can be given explicitly:

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod \
  --pipeline canonical_default__one_ion_hh_ohmic__braincell_one_ion_hh_ohmic
```

Default output:

- `examples/generated_<mod_stem>_one_ion_hh_ohmic.py`

## Supported scope

The first version of target lowering accepts:

- exactly one `USEION`
- a single recognizable ohmic current
- gate powers that can be extracted from the current expression
- gates in the current that reduce to
  - `inf/tau`
  - or `alpha/beta`

Current `target_family` values:

- `hh_ohmic_inf_tau`
- `hh_ohmic_alpha_beta`

## Not supported yet

The parser and Step 1 still keep many advanced blocks, but the BrainCell density-channel path does not generate them yet:

- `KINETIC`
- `DISCRETE`
- `NET_RECEIVE`
- `BEFORE`
- `AFTER`
- `INDEPENDENT`
- `CONSTANT`
- `LINEAR`
- `NONLINEAR`
- `FUNCTION_TABLE`
- `CVODE`
- `LONGITUDINAL_DIFFUSION`

In addition, target lowering currently rejects these mechanisms:

- several `USEION`
- `NONSPECIFIC_CURRENT`
- non-ohmic currents
- mechanisms whose gate powers or gating dynamics cannot be recognized

## Walkthrough

Start with:

- `examples/walktrough.ipynb`

The notebook shows, in the order of the current implementation:

1. the input `.mod`
2. the parser AST summary
3. `raw_blocks`
4. `canonical_blocks`
5. `bc_ast`
6. `semantic_ir`
7. `target_ir`
8. `rendered Python`
9. `validation`
10. positive and negative cases side by side

It also saves the core artifacts under `examples/artifacts/<mod_stem>/` through `steps.save_pipeline_artifacts()` for manual inspection.

The `kv` artifacts are tracked. They name the source `.mod` relative to the repository, so running the notebook again leaves them unchanged unless the pipeline's output changed; a source outside the repository is named by its absolute path.

## Recommended order

When debugging or extending the converter:

1. look at the Step 1 `bc_ast` first
2. then the Step 2 `semantic_ir`
3. confirm that `target_ir["supported"]` is `true`
4. finally look at the Step 3 render preview and validation

Quick smoke test:

1. run `kv.mod` first
2. then `na_alpha_beta.mod`
3. finally confirm that `hh.mod` is rejected in target lowering
