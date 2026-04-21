# BrainCell NMODL Pipeline Overview

## Summary

当前这条 BrainCell-oriented 流水线的重点是“分层降级”，而不是把 parser AST 直接交给模板：

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

外层入口仍然是三个 step，但内部边界已经拆成：

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

当前注册组合仍然是：

```text
step1: canonical_default
step2: one_ion_hh_ohmic
step3: braincell_one_ion_hh_ohmic
```

它们现在的职责是：

- `canonical_default`
  - 产出 `raw_blocks`、`canonical_blocks`、`bc_ast`
- `one_ion_hh_ohmic`
  - 从 `bc_ast` 构建 `semantic_ir`
  - 再 lowering 到当前 density-channel `target_ir`
- `braincell_one_ion_hh_ohmic`
  - 渲染 target IR
  - 对生成源码做 `compile + exec` 验证

## Data Boundaries

### Step 1: parse and normalize

当前 Step 1 输出：

- parser AST
- `raw_blocks`
- `canonical_blocks`
- `bc_ast`

`bc_ast` 是 BrainCell 自己的 typed AST 边界。它保留：

- `NEURON`/`USEION`/`RANGE`/`GLOBAL`
- `PARAMETER`/`ASSIGNED`/`STATE`
- `INITIAL`/`BREAKPOINT`/`SOLVE`
- `FUNCTION`/`PROCEDURE`
- statement / expression 级别的结构和 source span

### Step 2: semantic recovery and lowering

Step 2 先构建 `semantic_ir`，再构建 `target_ir`。

`semantic_ir` 当前重点字段：

- `symbols`
- `procedure_env`
- `breakpoint_assignments`
- `currents`
- `gate_kinetics`
- `ion_dependencies`
- `unsupported_features`

`target_ir` 当前重点字段：

- `target_family`
- `class_name`
- `base_class_name`
- `g_max_param`
- `gates`
- `current_model`
- `supported`
- `rejection_reasons`

### Step 3: render and validation

Step 3 当前输出：

- `rendered_text`
- `summary`
- `validation`

`validation` 现在会记录：

- 是否 `compile` 成功
- 是否 `exec` 成功
- 是否成功取得目标 class

## Current Coverage

当前 density-channel lowering 接受：

- 恰好一个 `USEION`
- 可识别的单 ohmic current
- 可识别 gate power
- 参与电流的 gate 可规约到
  - `hh_ohmic_inf_tau`
  - `hh_ohmic_alpha_beta`

正例：

- `examples/convert_mod/nmodl/mod_files/kv.mod`
- `examples/convert_mod/nmodl/mod_files/na_alpha_beta.mod`

反例：

- `examples/convert_mod/nmodl/mod_files/hh.mod`

`hh.mod` 不是 parser 失败，而是：

- `semantic_ir` 能完整构建
- 但 `target_ir` 因 `multi_useion` 和 `nonspecific_current` 被拒绝

## Current Unsupported Areas

虽然 Step 1 仍可保留原始块文本，但当前 BrainCell path 不会生成这些高级结构：

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

后续扩展仍然遵循原来的分层思路，但边界应以当前实现为准：

1. 先扩展 Step 1 的 `bc_ast` 规范化能力
2. 再扩展 `semantic_ir` 对语义和符号的恢复
3. 再增加新的 `target_ir` 家族
4. 最后增加匹配的 renderer / template / validation

这样可以保持：

- parser-facing 逻辑在 `inspect_ast/`
- semantics 在 `semantic_ir.py`
- BrainCell lowering 在 `target_ir.py`
- 渲染和验证在 `render/`
- orchestration 继续留在 `flow.py` 和 `registry.py`
