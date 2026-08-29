# Parameter Learning 历史实验与训练诊断

## 当前状态

本目录是早期 BrainCell parameter-learning 实验目录。它不是公共 API，也不再包含当时完整的
九参数训练、dataset、ablation 和 demo 源码。当前仍活跃并有调用方的代码只有：

```text
training_diagnostics.py
training_diagnostics_test.py
```

因此本目录应理解为“历史实验已收缩，但训练诊断 helper 仍在使用”，不能整体标记为
deprecated，也不能把残留 `__pycache__/*.pyc` 当作可恢复源码。

## 当前活跃代码

`training_diagnostics.py` 提供 experiment-local、optimizer-agnostic 的 multistart 训练诊断：

- optimizer-space 与 physical parameter history；
- total/component loss 和 evaluator metrics；
- gradient norm、gradient cosine 和 optimizer/physical step norm；
- parameter bound position；
- per-start continuous-best 与 spike-feasible best archives；
- failure summary、plotting 和 artifact serialization。

它不负责：

- 模型参数声明和 runtime materialization；这些由 `braincell.trainable` 负责；
- BPTT/RTRL 或 solver；这些由 BrainState/JAX 和对应实验模块负责；
- optimizer；当前继续使用 `braintools.optim`；
- 通用 Trainer、Dataset 或 checkpoint 公共 API。

当前直接调用方：

```text
examples/multi_compartment/trainable_hh_multistart.py
examples/multi_compartment/trainable_hh_multistart_test.py
examples/multi_compartment/train.ipynb
```

这个 helper 同时适用于 BPTT、RTRL 或其他能返回相同命名 gradient mapping 的实验，因此保留在
`parameter_learning/`，不并入只研究 exact online gradient 的 `online_learning/`。

## 历史源码

仓库历史曾包含下列实验方向，但当前 worktree 已没有对应 `.py` 源文件：

- `conductance_learning_core.py`；
- `heterogeneous_nine_parameter_training.py`；
- `heterogeneous_protocol_dataset.py`；
- `heterogeneous_nine_parameter_composite_ablation.py`；
- `parameter_learning_demo.py`；
- `sine_expanded_batch_comparison.py`。

局部机器可能残留同名 `.pyc`，这些只是 Python cache：

- 不能作为代码依赖；
- 不进入版本控制；
- 不应反编译后恢复为当前实现；
- 应由测试或维护过程直接清理。

相关设计文档引用这些名字时，表示历史快照中的实验依据，不表示文件仍存在于当前分支。

## 相关文档

- [Modular Training Diagnostics](../../../docs/design/optim/references/modular-training-diagnostics.md)：当前 `training_diagnostics.py` 的角色、history alignment 和 archive contract。
- [电压轨迹与 Spike-Aware 参数训练](../../../docs/design/optim/references/voltage-and-spike-parameter-fitting.md)：早期九参数、多 protocol 和 loss 设计证据；其中部分本地实现路径属于历史快照。
- [Optimization Design Overview](../../../docs/design/optim/design-overview.md)：公共 `braincell.trainable` 与实验训练代码的边界。
- [Online Learning 实验](../online_learning/README.md)：exact forward sensitivity、RTRL/BPTT 与 scaling benchmark。

## 最小使用方式

当前训练示例采用以下数据流：

```text
rollout
  -> voltage_mse_objective
  -> evaluate_voltage_protocols
  -> capture_state / capture_update
  -> optimizer.update
  -> finalize_history
  -> extract_best_archives / summarize_history
```

新增实验应尽量只替换其中一个角色，例如 objective 或 evaluator，不要复制整套 history/archive
实现。

## 验证

```bash
pytest -q examples/experimental/parameter_learning/training_diagnostics_test.py
pytest -q examples/multi_compartment/trainable_hh_multistart_test.py
```

如果未来该 helper 有多个稳定调用方并形成明确公共合同，再决定是否移动到正式模块；本 README
不承诺当前位置或类型名长期稳定。
