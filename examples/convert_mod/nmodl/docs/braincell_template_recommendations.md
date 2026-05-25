# `one_ion_hh_ohmic` Template Recommendations

## Summary

The BrainCell path is intentionally focused on one target:

```text
.mod
-> AST
-> RawBlocks
-> CanonicalBlocks
-> one_ion_hh_ohmic IR
-> templates/one_ion_hh_ohmic.py
```

This target means:

- one ion only
- HH gate dynamics
- ohmic current
- BrainCell sign convention: `g_max * gates * (Ion.E - V)`

Current implementation entry points:

- `steps/inspect_ir/variants/one_ion_hh_ohmic.py`
- `steps/render/variants/braincell_one_ion_hh_ohmic.py`
- `examples/generate_braincell.py`
- `steps/inspect_ir/cli.py`
- `steps/render/cli.py`

Current registered combination:

```text
canonical_default
-> one_ion_hh_ohmic
-> braincell_one_ion_hh_ohmic
```

## What Is Standardized

### Ion and base class

The template normalizes a single `USEION` into a `braincell.Channel` subclass with a
`root_type` and an ion argument name:

- `k -> root_type=braincell.ion.Potassium / K`
- `na -> root_type=braincell.ion.Sodium / Na`
- `ca -> root_type=braincell.ion.Calcium / Ca`

### Gate representation

Every gate is normalized to:

- original gate name
- gate power from the current expression
- `inf_expr`
- `tau_expr`
- `q10`

The IR always stores `inf/tau`, even if the source derivative is `alpha/beta`.

### `alpha/beta -> inf/tau`

If the source form is:

```text
x' = alpha_x * (1 - x) - beta_x * x
```

the IR rewrites it to:

```text
x_inf = alpha_x / (alpha_x + beta_x)
x_tau = 1 / (alpha_x + beta_x)
```

### Current form

The current is normalized to:

- one conductance parameter
- gate product
- one reversal variable
- BrainCell sign convention

Accepted source orientations include both `(v - ek)` and `(ek - v)`, but generated code always uses:

```python
g_max * gate_product * (K.E - V)
```

### Temperature form

The generated class standardizes temperature as:

- runtime `temp=u.celsius2kelvin(23)`
- fixed `temp_ref=u.celsius2kelvin(23)`
- per-gate `Q10_gate`, default `1.0`

## Current IR Shape

Important fields:

- `class_name`
- `base_class_name`
- `root_type`
- `ion_name`
- `ion_arg_name`
- `g_max_param`
- `v_shift_param`
- `temperature_param`
- `tref_expression`
- `extra_parameters`
- `gates`
- `current_model`
- `supported`
- `rejection_reasons`
- `manual_fix_required`

Each gate includes:

- `name`
- `safe_name`
- `power`
- `q10_expression`
- `source_form`
- `helper_alias_lines`
- `inf_expr_python`
- `tau_expr_python`

This is enough for the template to render:

- `init_state`
- `reset_state`
- `compute_derivative`
- `current`
- `f_gate_inf`
- `f_gate_tau`

## What Is Accepted

The first template accepts only mechanisms that satisfy all of these:

- exactly one `USEION`
- the ion current reduces to one ohmic current
- gate powers can be extracted from the current expression
- every current-participating state can be reduced to `inf/tau`

Positive examples:

- `examples/convert_mod/nmodl/mod_files/kv.mod`
- `examples/convert_mod/nmodl/mod_files/na_alpha_beta.mod`

## What Is Rejected

The template currently rejects:

- multi-ion mechanisms such as `examples/convert_mod/nmodl/mod_files/hh.mod`
- GHK currents
- extra multiplicative current factors outside conductance and gate powers
- gates that cannot be recognized as `inf/tau` or `alpha/beta`
- advanced blocks still only preserved in RawBlocks

Current unsupported advanced blocks:

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

## Good Next Extensions

The most coherent future work is:

1. extend Step 1 normalization in `steps/inspect_ast/`
2. add new Step 2 IR families in `steps/inspect_ir/`
3. add matching Step 3 render variants in `steps/render/`
4. keep each legal combination explicit in `steps/registry.py`
