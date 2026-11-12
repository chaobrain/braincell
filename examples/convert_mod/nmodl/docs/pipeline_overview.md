# BrainCell NMODL Pipeline Overview

## Summary

This BrainCell-oriented pipeline is built around "layered lowering" rather than handing the parser AST straight to a template:

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

The outer entry points are still three steps, but the internal boundaries are split into:

- Step 1: parser AST -> `raw_blocks` / `canonical_blocks` / `bc_ast`
- Step 2: `bc_ast` -> `semantic_ir` -> `target_ir`
- Step 3: `target_ir` -> rendered Python -> validation

## Entry Points

| Stage | Output | Entry point |
| --- | --- | --- |
| Parse and normalize | `raw_blocks`, `canonical_blocks`, `bc_ast` | `steps/inspect_ast/cli.py` |
| IR inspection | `semantic_ir`, `target_ir`, support summary | `steps/inspect_ir/cli.py` |
| Render inspection | rendered Python preview, validation, optional file write | `steps/render/cli.py` |
| Direct conversion | generated BrainCell `.py` file | `examples/generate_braincell.py` |
| Walkthrough | notebook inspection of the full flow | `examples/walktrough.ipynb` |

## Layout

- `steps/inspect_ast/`
  - Step 1 parser-facing normalization
- `steps/inspect_ir/`
  - Step 2 orchestration
- `steps/model.py`
  - shared dataclasses for `bc_ast`, `semantic_ir`, `target_ir`
- `steps/semantic_ir.py`
  - semantic recovery from `bc_ast`
- `steps/target_ir.py`
  - BrainCell density-channel lowering
- `steps/render/`
  - Step 3 render helpers and render validation
- `steps/flow.py`
  - fixed step runner
- `steps/registry.py`
  - legal variant combinations

## Step Variants

The registered combination is still:

```text
step1: canonical_default
step2: one_ion_hh_ohmic
step3: braincell_one_ion_hh_ohmic
```

Their responsibilities are:

- `canonical_default`
  - produces `raw_blocks`, `canonical_blocks`, `bc_ast`
- `one_ion_hh_ohmic`
  - builds `semantic_ir` from `bc_ast`
  - then lowers it to the current density-channel `target_ir`
- `braincell_one_ion_hh_ohmic`
  - renders the target IR
  - validates the generated source with `compile + exec`

## Data Boundaries

### Step 1: parse and normalize

Step 1 outputs:

- parser AST
- `raw_blocks`
- `canonical_blocks`
- `bc_ast`

`bc_ast` is BrainCell's own typed AST boundary. It keeps:

- `NEURON`/`USEION`/`RANGE`/`GLOBAL`
- `PARAMETER`/`ASSIGNED`/`STATE`
- `INITIAL`/`BREAKPOINT`/`SOLVE`
- `FUNCTION`/`PROCEDURE`
- statement- and expression-level structure and source spans

### Step 2: semantic recovery and lowering

Step 2 builds `semantic_ir` first, then `target_ir`.

Main fields of `semantic_ir`:

- `symbols`
- `procedure_env`
- `breakpoint_assignments`
- `currents`
- `gate_kinetics`
- `ion_dependencies`
- `unsupported_features`

Main fields of `target_ir`:

- `target_family`
- `class_name`
- `base_class_name`
- `root_type`
- `g_max_param`
- `gates`
- `current_model`
- `supported`
- `rejection_reasons`

### Step 3: render and validation

Step 3 outputs:

- `rendered_text`
- `summary`
- `validation`

`validation` records:

- whether `compile` succeeds
- whether `exec` succeeds
- whether the target class is obtained

## Current Coverage

The density-channel lowering accepts:

- exactly one `USEION`
- a single recognizable ohmic current
- recognizable gate powers
- gates in the current that reduce to
  - `hh_ohmic_inf_tau`
  - `hh_ohmic_alpha_beta`

Positive cases:

- `examples/convert_mod/nmodl/mod_files/kv.mod`
- `examples/convert_mod/nmodl/mod_files/na_alpha_beta.mod`

Negative case:

- `examples/convert_mod/nmodl/mod_files/hh.mod`

`hh.mod` does not fail in the parser; instead:

- `semantic_ir` builds completely
- but `target_ir` is rejected because of `multi_useion` and `nonspecific_current`

## Current Unsupported Areas

Step 1 can still keep the raw block text, but the BrainCell path does not generate these advanced constructs yet:

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

## Extension Direction

Further extensions keep the same layered approach, with the boundaries of the current implementation as the reference:

1. first extend the `bc_ast` canonicalization of Step 1
2. then extend the semantic and symbol recovery of `semantic_ir`
3. then add new `target_ir` families
4. finally add the matching renderer / template / validation

This keeps:

- parser-facing logic in `inspect_ast/`
- semantics in `semantic_ir.py`
- BrainCell lowering in `target_ir.py`
- rendering and validation in `render/`
- orchestration in `flow.py` and `registry.py`
