# BrainCell NMODL Pipeline Overview

## Summary

This directory now exposes a single BrainCell-oriented path with a fixed three-step shape:

```text
.mod
-> AST
-> RawBlocks
-> CanonicalBlocks
-> one_ion_hh_ohmic IR
-> templates/one_ion_hh_ohmic.py
-> generated Python
```

There is no toy backend anymore. The only supported generation target is `one_ion_hh_ohmic`.
The number of steps is fixed; extensibility comes from adding variants inside each step directory.

## Entry Points

| Stage | Output | Entry point |
| --- | --- | --- |
| Parse and normalize | AST, RawBlocks, CanonicalBlocks | `steps/inspect_ast.py` |
| IR inspection | support summary and `one_ion_hh_ohmic IR` | `steps/inspect_ir.py` |
| Render inspection | rendered Python preview, optional file write | `steps/render_preview.py` |
| Direct conversion | generated BrainCell `.py` file | `examples/generate_braincell.py` |
| Walkthrough | notebook inspection of the full flow | `notebooks/braincell_pipeline_walkthrough.ipynb` |

## Layout

- `pipeline/inspect_ast/`
  - Step 1 parsing, RawBlocks, CanonicalBlocks, and Step 1 variants
- `pipeline/inspect_ir/`
  - Step 2 IR extraction and Step 2 variants
- `pipeline/render/`
  - Step 3 render helpers and Step 3 variants
- `pipeline/flow.py`
  - fixed three-step runner
- `pipeline/registry.py`
  - legal variant combinations

## Step Variants

The current registered combination is:

```text
step1: canonical_default
step2: one_ion_hh_ohmic
step3: braincell_one_ion_hh_ohmic
```

Conceptually, future pipelines are composed as:

```text
step1:<variant> + step2:<variant> + step3:<variant>
```

Not every combination is valid. Allowed combinations are declared in `pipeline/registry.py`.

## Data Boundaries

### Step 1: parse and normalize

- Purpose: parsed NMODL syntax tree from the backend parser
- Current variant: `canonical_default`
- Outputs:
  - AST
  - RawBlocks
  - CanonicalBlocks

### Step 2: build BrainCell IR

- Purpose: convert Step 1 output into a target BrainCell IR family
- Current variant: `one_ion_hh_ohmic`
- Outputs:
  - support summary
  - rejection reasons
  - IR payload

### Step 3: render

- Purpose: convert a supported Step 2 IR into generated Python
- Current variant: `braincell_one_ion_hh_ohmic`
- Outputs:
  - rendered text
  - optional file write

## Step 1 Coverage

`canonical_default` currently provides:

- AST parsing
- RawBlocks extraction
- CanonicalBlocks normalization

CanonicalBlocks currently covers:

- `TITLE`
- `COMMENT`
- `NEURON`
- `UNITS`
- `PARAMETER`
- `ASSIGNED`
- `STATE`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- `FUNCTION`
- `PROCEDURE`

RawBlocks can still preserve many advanced blocks even when they are not yet normalized.

## What The Current Template Understands

The current template family accepts only mechanisms that satisfy all of these:

- exactly one `USEION`
- one ohmic current
- conductance parameter identifiable from the current expression
- gate powers identifiable from the current expression
- every participating gate reducible to `inf/tau`
  - either directly
  - or by conversion from `alpha/beta`

Positive examples:

- `examples/kv.mod`
- `examples/na_alpha_beta.mod`

Negative example:

- `examples/hh.mod`

## Current Unsupported Areas

The parser can still preserve many advanced blocks in RawBlocks, but the BrainCell path does not yet normalize or generate them:

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

The clean extension path is:

1. add new Step 1 variants when canonicalization itself needs different strategies
2. add new Step 2 variants for new BrainCell IR families
3. add new Step 3 variants for matching renderers/templates
4. register only valid combinations in `pipeline/registry.py`

This keeps:

- Step 1 internals inside `inspect_ast/`
- Step 2 internals inside `inspect_ir/`
- Step 3 internals inside `render/`
- orchestration rules in `pipeline/flow.py` and `pipeline/registry.py`
