# BrainCell NMODL Pipeline

这个目录现在提供的是一条可检查的 `.mod -> BrainCell density channel` 转换链，重点不是“直接套模板”，而是把 NMODL 逐层降到 BrainCell 自己的中间表示。

当前首版稳定支持的目标机制是：

- 单 `USEION`
- Hodgkin-Huxley 风格门控动力学
- ohmic current
- 生成 BrainCell Python density channel 源码

当前代表性样例：

- `mod_files/kv.mod`
  - 正例，`inf/tau`
- `mod_files/na_alpha_beta.mod`
  - 正例，`alpha/beta`
- `mod_files/hh.mod`
  - 反例，语义层保留，但因多离子和 `NONSPECIFIC_CURRENT` 在 target lowering 阶段拒绝

如果你现在主要是在补 NMODL 基础，而不是直接改 pipeline，建议先看：

- `docs/nmodl_mental_model_zh.md`
  - 从最小 leak 机制、`hh.mod`、`NET_RECEIVE` 三个层次理解 `PARAMETER` / `ASSIGNED` / `STATE` 与 `INITIAL` / `BREAKPOINT` / `DERIVATIVE`
  - 顺带整理了 `NEURON 8.2.x`、`9.0` 与 Arbor 的关键差异

## 当前流水线

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

外部 CLI 仍然保持三步入口，但内部边界已经是四层：

1. Step 1
   - 解析并规范化
   - 产出 `raw_blocks`、`canonical_blocks`、`bc_ast`
2. Step 2
   - 构建 `semantic_ir`
   - 再 lowering 到 `target_ir`
3. Step 3
   - 渲染 Python 源码
   - 执行 `compile + exec` 验证

## 目录结构

- `mod_files/`
  - 示例 `.mod` 输入
- `examples/`
  - `generate_braincell.py`
    - 一条命令跑完整流水线并写出最终 `.py`
  - `walktrough.ipynb`
    - 逐层 walkthrough notebook
- `steps/`
  - `inspect_ast/`
    - Step 1
  - `inspect_ir/`
    - Step 2
  - `render/`
    - Step 3
  - `model.py`
    - `bc_ast` / `semantic_ir` / `target_ir` 的 dataclass 边界
  - `semantic_ir.py`
    - 语义恢复
  - `target_ir.py`
    - BrainCell density-channel lowering
- `templates/`
  - `density_channel.py`
    - 当前 renderer 使用的模板
- `docs/`
  - 架构说明

## 环境依赖

第一步 AST 解析依赖 NMODL Python backend。当前代码会优先尝试：

- `neuron.nmodl`
- `nmodl.dsl`

快速检查：

```bash
python - <<'PY'
import importlib.util
for name in ("neuron.nmodl", "nmodl.dsl"):
    spec = importlib.util.find_spec(name)
    print(name, "->", spec.origin if spec else None)
PY
```

若两者都不可用，需要安装 Blue Brain 的 `nmodl` 包，或切换到已经提供 `neuron.nmodl` 的环境。

渲染阶段还需要 `jinja2`：

```bash
python -m pip install jinja2
```

## 当前入口

### Step 1: AST / blocks / bc_ast

```bash
python examples/convert_mod/nmodl/steps/inspect_ast/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

当前输出重点：

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

当前输出重点：

- `summary`
- `semantic_ir`
- `target_ir`
- `braincell_ir`

这里的 `braincell_ir` 目前等同于当前 target IR payload，保留是为了兼容旧入口。

### Step 3: render preview / validation

```bash
python examples/convert_mod/nmodl/steps/render/cli.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

当前输出重点：

- `summary`
- `render_preview`
- `validation`

`validation` 现在会报告生成源码是否能：

- `compile`
- `exec`
- 取到目标 class

### 一步生成源码

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod
```

可显式指定 pipeline：

```bash
python examples/convert_mod/nmodl/examples/generate_braincell.py \
  examples/convert_mod/nmodl/mod_files/kv.mod \
  --pipeline canonical_default__one_ion_hh_ohmic__braincell_one_ion_hh_ohmic
```

默认输出：

- `examples/generated_<mod_stem>_one_ion_hh_ohmic.py`

## 当前支持范围

首版 target lowering 接受：

- 恰好一个 `USEION`
- 可识别的单 ohmic current
- 可从 current expression 中提取 gate powers
- 参与电流的 gate 可规约为
  - `inf/tau`
  - 或 `alpha/beta`

当前 `target_family`：

- `hh_ohmic_inf_tau`
- `hh_ohmic_alpha_beta`

## 当前不支持

虽然 parser 和 Step 1 还能保留很多高级块，但当前 BrainCell density-channel 路径还不会生成它们：

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

此外，下列机制目前会在 target lowering 阶段拒绝：

- 多 `USEION`
- `NONSPECIFIC_CURRENT`
- 非 ohmic current
- 无法识别 gate power 或门控动力学的机制

## Walkthrough

推荐先打开：

- `examples/walktrough.ipynb`

这个 notebook 会按当前实现顺序展示：

1. 输入 `.mod`
2. parser AST 摘要
3. `raw_blocks`
4. `canonical_blocks`
5. `bc_ast`
6. `semantic_ir`
7. `target_ir`
8. `rendered Python`
9. `validation`
10. 正例 / 反例对照

它也会通过 `steps.save_pipeline_artifacts()` 把核心产物保存到 `examples/artifacts/<mod_stem>/` 下面，便于人工检查。

## 推荐使用顺序

调试或扩展转换器时：

1. 先看 Step 1 的 `bc_ast`
2. 再看 Step 2 的 `semantic_ir`
3. 确认 `target_ir["supported"]` 为 `true`
4. 最后再看 Step 3 的 render preview 和 validation

快速 smoke test：

1. 先跑 `kv.mod`
2. 再跑 `na_alpha_beta.mod`
3. 最后确认 `hh.mod` 在 target lowering 阶段被拒绝
