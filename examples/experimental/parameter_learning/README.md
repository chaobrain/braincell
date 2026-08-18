# Experimental Parameter Learning

本目录用于验证 BrainCell 参数学习的新功能，不属于稳定公共 API。当前实验使用一个
三 compartment cell（`soma`、`dend_a`、`dend_b`），通过 BrainState 和
BrainTools 对带物理单位且有边界约束的 conductance 参数求梯度并优化。

## 文件说明

### `conductance_learning_core.py`

提供其余实验复用的底层组件：

- trainable/frozen Leak、HH Na 和 HH K channel adapter；
- 三个 compartment 共享三个 conductance 的小型训练问题；
- bounded parameter transform、state reset 和 compiled rollout；
- voltage、derivative、multiscale、event、count 和 peak loss helper。

`conductance_learning_core_test.py` 验证 parameter ownership、梯度、reset、参数边界和
短训练过程。

### `heterogeneous_protocol_dataset.py`

生成九参数实验使用的 synthetic dataset：

- 三个注入位置和三个 voltage probe；
- DC、paired-pulse 和 sine 三类 stimulus；
- 144 条 protocol，划分为 108 train、18 validation、18 test；
- 每条记录包含 100 ms current 和 voltage trace。

默认数据保存在 `plot/heterogeneous_protocol_dataset/dataset.npz`。重新生成完整数据：

```bash
JAX_PLATFORMS=cpu python examples/experimental/parameter_learning/heterogeneous_protocol_dataset.py
```

### `heterogeneous_nine_parameter_training.py`

当前训练基线。它分别学习 soma、dend_a 和 dend_b 的 Leak、Na、K `g_max`，共九个
参数。默认设置为八个初值、batch size 18、每 epoch 六个 batch、30 epochs，使用
target-spike-window-masked voltage Huber loss 和 validation-best checkpoint。

完整运行还会计算 validation landscape、one-dimensional profiles、trace atlas、硬
spike 指标和参数误差图：

```bash
JAX_PLATFORMS=cpu python examples/experimental/parameter_learning/heterogeneous_nine_parameter_training.py
```

只基于已有训练 archive 重绘质量诊断：

```bash
JAX_PLATFORMS=cpu python examples/experimental/parameter_learning/heterogeneous_nine_parameter_training.py \
  --diagnostics-only
```

注意：`--epochs 1` 仍会在训练后计算完整 landscape 和诊断，因此不适合作为快速
smoke test。

### `heterogeneous_nine_parameter_composite_ablation.py`

在完全相同的数据、初值、optimizer 和 checkpoint 规则下比较三组 loss：

- `voltage_count`
- `without_count_composite`
- `full_composite`

例如运行完整 Composite：

```bash
JAX_PLATFORMS=cpu python examples/experimental/parameter_learning/heterogeneous_nine_parameter_composite_ablation.py \
  --configuration full_composite
```

## 最小 Huber Smoke Test

下面的测试只运行一个真实的 18-protocol batch。它会构造九参数 cell、计算 masked
Huber loss、反向传播九个梯度，并执行一次 Adam update；不会运行 multistart、完整
epoch、landscape 或绘图。

```bash
JAX_PLATFORMS=cpu pytest -q \
  examples/experimental/parameter_learning/heterogeneous_nine_parameter_training_test.py \
  -k one_batch_huber
```

通过条件：loss 和九参数梯度全部有限、梯度范数非零，并且一次 optimizer update
改变全部九个 physical parameters。

运行本目录全部测试：

```bash
JAX_PLATFORMS=cpu pytest -q examples/experimental/parameter_learning
```

## 结果目录

所有 dataset、训练 archive 和图片位于本目录的 `plot/` 下。该目录被 Git 忽略；
移动或清理前应确认是否仍需要其中耗时生成的结果。

正式实验合同和长期设计分别记录在：

- `docs/specs/2026-08-17-heterogeneous-protocol-dataset.md`
- `docs/specs/2026-08-17-heterogeneous-nine-parameter-training.md`
- `docs/specs/2026-08-18-nine-parameter-composite-loss-ablation.md`
- `docs/design/training/voltage-and-spike-parameter-fitting.md`
