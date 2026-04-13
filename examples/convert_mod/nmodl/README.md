# BrainCell NMODL Pipeline

This directory is a step-by-step `.mod -> BrainCell channel` conversion toolchain.

Current end-to-end target:

- `one_ion_hh_ohmic`

That means:

- exactly one `USEION`
- Hodgkin-Huxley style gate dynamics
- one ohmic current
- BrainCell current sign convention: `g_max * gates * (Ion.E - V)`

## Directory Layout

- `mod_files/`
  - sample input `.mod` files
  - `kv.mod`: positive single-ion `inf/tau` example
  - `na_alpha_beta.mod`: positive single-ion `alpha/beta` example
  - `hh.mod`: rejection example; intentionally unsupported because it is multi-ion
- `examples/`
  - `generate_braincell.py`: one command that runs all three steps and writes the final generated Python
  - `walktrough.ipynb`: notebook walkthrough for step-by-step inspection, including saving the final rendered result
- `steps/`
  - `inspect_ast/`: Step 1 implementation, variants, and `cli.py`
  - `inspect_ir/`: Step 2 implementation, variants, and `cli.py`
  - `render/`: Step 3 implementation, variants, and `cli.py`
  - `flow.py`: fixed three-step runner
  - `registry.py`: legal step combinations
- `templates/`
  - `one_ion_hh_ohmic.py`: current Jinja template used by Step 3
- `docs/`
  - architecture and template notes

## Pipeline Shape

```text
.mod
-> AST
-> RawBlocks
-> CanonicalBlocks
-> one_ion_hh_ohmic IR
-> templates/one_ion_hh_ohmic.py
-> generated BrainCell channel
```

The step count is fixed:

1. Step 1: parse and normalize
2. Step 2: build BrainCell IR
3. Step 3: render Python from IR

What can vary is the variant selected inside each step.

## Environment

补充说明：第一步 AST 解析依赖额外的 NMODL Python backend，不一定已经包含在当前环境里。当前代码会优先尝试 `neuron.nmodl`，如果没有，则尝试 `nmodl.dsl`；如果两者都没有，通常需要在当前环境里额外安装 Blue Brain 的 `nmodl` Python 包。

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

If neither backend is available, install `nmodl` or switch to an environment that already provides `neuron.nmodl`.

If Jinja2 is missing:

```bash
python -m pip install jinja2
```

确认这些依赖没问题后，下一步直接参考 `examples/walktrough.ipynb`。

## Current Variants

Step 1 variants:

- `canonical_default`
  - status: implemented and registered
  - purpose: parse `.mod`, reconstruct source, collect RawBlocks, and build CanonicalBlocks

Step 2 variants:

- `one_ion_hh_ohmic`
  - status: implemented and registered
  - purpose: convert CanonicalBlocks into a BrainCell IR for a single-ion HH ohmic channel

Step 3 variants:

- `braincell_one_ion_hh_ohmic`
  - status: implemented and registered
  - purpose: render the Step 2 IR with the `one_ion_hh_ohmic.py` template

Current legal combinations:

- `canonical_default -> one_ion_hh_ohmic -> braincell_one_ion_hh_ohmic`

Current named pipeline string:

```text
canonical_default__one_ion_hh_ohmic__braincell_one_ion_hh_ohmic
```

No other combinations are currently registered in `steps/registry.py`.

## Step 1 Support

Entry point:

```bash
python examples/convert_mod/nmodl/steps/inspect_ast/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

What Step 1 does:

- parse `.mod` into an AST
- reconstruct the normalized source text
- extract RawBlocks
- build CanonicalBlocks
- report block counts and the AST JSON payload

What `canonical_default` currently canonicalizes:

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

What Step 1 can still preserve only as raw, not normalized:

- advanced blocks that appear in the parser output but are not yet part of CanonicalBlocks
- examples include `KINETIC`, `NET_RECEIVE`, `LINEAR`, `NONLINEAR`, and other advanced constructs

What to expect as output:

- `ast_root_type`
- `block_counts`
- `reconstructed_nmodl`
- `raw_blocks`
- `canonical_blocks`
- `ast_json`

## Step 2 Support

Entry point:

```bash
python examples/convert_mod/nmodl/steps/inspect_ir/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

What Step 2 does:

- take Step 1 canonical output
- decide whether the mechanism matches a supported BrainCell family
- build a typed IR payload
- report support or rejection reasons

What `one_ion_hh_ohmic` currently supports:

- exactly one `USEION`
- ion types `k`, `na`, `ca`
- one ohmic current
- gate product extractable from the current expression
- gate kinetics reducible to `inf/tau`
- direct `inf/tau` forms
- `alpha/beta -> inf/tau` conversion
- BrainCell sign convention rewrite to `(Ion.E - V)`
- temperature fields normalized to `temp`, `Tref`, and per-gate `Q10`

What Step 2 currently rejects:

- multi-ion mechanisms
- non-ohmic currents
- gates that cannot be reduced to `inf/tau`
- mechanisms whose current expression does not cleanly expose conductance and gate powers

What to expect as output:

- `summary`
- `braincell_ir`
- `supported: true/false`
- `rejection_reasons`

Use this rejection example:

```bash
python examples/convert_mod/nmodl/steps/inspect_ir/cli.py \
  examples/convert_mod/nmodl/mod_files/hh.mod
```

## Step 3 Support

Entry point:

```bash
python examples/convert_mod/nmodl/steps/render/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

What Step 3 does:

- run the full 1 -> 2 -> 3 chain
- render the supported IR into Python source
- print a preview
- optionally write the rendered file

Current render variant:

- `braincell_one_ion_hh_ohmic`

Current Jinja template:

- `templates/one_ion_hh_ohmic.py`

What the current template generates:

- a BrainCell channel class inheriting from the ion-specific base class
- `init_state`
- `reset_state`
- `compute_derivative`
- `current`
- one `f_<gate>_inf`
- one `f_<gate>_tau`

What the generated class no longer exposes as class attributes:

- `source_file`
- `mechanism_name`
- `manual_fix_required`

Current Step 3 limitation:

- it only renders IR that Step 2 marked as supported
- unsupported IR exits with rejection reasons instead of generating fallback code

Preview only:

```bash
python examples/convert_mod/nmodl/steps/render/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

Preview and write file:

```bash
python examples/convert_mod/nmodl/steps/render/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod \
  -o /tmp/kv_channel.py
```

## One-Step Run

If you do not want to inspect each stage separately, use the combined entry:

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

This command:

1. resolves the registered step combination
2. runs Step 1
3. runs Step 2
4. runs Step 3
5. writes the final generated Python file
6. prints a JSON summary and preview

You can also specify the pipeline explicitly:

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod \
  --pipeline canonical_default__one_ion_hh_ohmic__braincell_one_ion_hh_ohmic
```

Default output file:

- `examples/generated_<mod_stem>_one_ion_hh_ohmic.py`

Custom output file:

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod \
  -o /tmp/kv_channel.py
```

## Notebook Use

Walkthrough notebook:

- `examples/walktrough.ipynb`

This notebook is for manual inspection of:

- parsed AST
- RawBlocks
- CanonicalBlocks
- Step 2 IR
- rendered Python preview
- writing the core step artifacts and final rendered Python into one artifact directory

It is now grouped with the runnable examples instead of living in a separate `notebooks/` directory.
Unlike `generate_braincell.py`, it is also the only entry that saves the core step artifacts into one folder.

## Recommended Usage Order

For development or debugging:

1. run Step 1 on a `.mod` file and inspect `canonical_blocks`
2. run Step 2 and inspect `summary` plus `rejection_reasons`
3. run Step 3 only after Step 2 reports `supported: true`
4. use `generate_braincell.py` when the step-by-step inspection already looks correct

For quick smoke tests:

1. start with `mod_files/kv.mod`
2. verify `mod_files/hh.mod` is rejected at Step 2
3. verify `mod_files/na_alpha_beta.mod` can pass the `alpha/beta -> inf/tau` path
