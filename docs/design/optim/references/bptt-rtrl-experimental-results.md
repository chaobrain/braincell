# BPTT/RTRL 实验结果

## Reference 状态

本文是 BrainCell exact RTRL 与 full BPTT 的 tracked 实验结果快照，记录正确性、内存、性能和
训练一致性。数学推导见 [BPTT/RTRL 通用理论](./bptt-to-rtrl-neuron-derivation.md)，实验程序、
CLI 和 notebook 导航见
[Optimization Experiments](../../../../examples/experimental/README.md)。

本文记录的 A100 scaling 数据生成于 2026-08-29；CPU prototype 数据来自同一实现阶段的本机
x64 测量。原始 CSV/NPZ、trial JSON、manifest 和 worker log 位于 Git ignored artifacts，仍是
具体测量的权威来源。本文只固定可复查的结果与保守结论，不把单台硬件的墙钟外推为一般
渐近规律。

## 1. 实验对象与维度

主 scaling workload 是 multi-CV HH cell。每个 CV 包含 Leak、Na 和 K 三个独立 row parameter：

```text
active state per simulation = v[C] + Na.m[C] + Na.h[C] + K.n[C] = 4C
parameter DOF per seed = leak[C] + Na[C] + K[C] = 3C
```

Batch 中的 $B$ 个样本共享同一 seed 的参数；$S$ 个 seed 分别拥有独立的 $3C$ 参数。一次
benchmark 因而并行仿真 $S\times B$ 个 cell instances，但 sensitivity 按 seed block 传播，
不保存已知为零的 cross-seed block。

对 block-exact RTRL：

```text
N_state_active = 4 * B * C
N_parameter = 3 * C
minimal x64 carry = 96 * S * B * C^2 bytes
N_state_full = (6B + 6) * C + 10
measured full carry = S * N_parameter * N_state_full * 8 bytes
```

RTRL logical carry 不含 rollout 长度 $T$。BPTT temporary 包含时间 tape，随 $T$ 增长。

## 2. 梯度正确性

### 2.1 小模型 feasibility

| 模型 | 配置 | Forward gradient | BPTT gradient | 误差 |
| --- | --- | ---: | ---: | ---: |
| 2-compartment Leak | 1 scale、8 steps、voltage MSE | -25.068812955953636 | -25.068812955953646 | abs $1.1\times10^{-14}$ |
| 2-compartment HH | Na/K scales、21 captured states、4 steps | - | - | max abs $1.6\times10^{-15}$ |

自动化验证还覆盖：

- 每个 prefix gradient 与相同 prefix loss 的 reverse-mode gradient；
- central finite-difference directional derivative；
- 参数相关 reset 对 $S_0$ 的贡献；
- 两个固定 delay 的外源输入；
- carry shape 不含 $T$ 轴；
- x64 下 forward/reverse `rtol <= 1e-8`。

### 2.2 Multi-CV 与 row parameter

三 CV 模型具有 12 个 active state 和 9 个 row parameter，理论 sensitivity
$S_t\in\mathbb R^{12\times9}$；实验 core 将 batched tangent carry 存成 parameter-major
`(9, 12)`。五 CV 对应 $S_t\in\mathbb R^{20\times15}$，实现 carry 为 `(15, 20)`。

三 CV 与五 CV 的测试共同验证：

- parameter coordinate basis 与 row runtime axis 一致；
- compact、full RTRL、BPTT 与 directional finite difference 一致；
- 两条 distal dendrite arm 之间存在非零 cross-CV sensitivity；
- exact sensitivity 不能按 compartment 截成互不相干的局部 trace。

### 2.3 A100 全套数值一致性

Stored A100 suites 中，BPTT/RTRL 配对结果为：

| Metric | Worst observed value |
| --- | ---: |
| Relative gradient error | `4.540e-08` |
| Absolute loss error | `6.776e-10` |
| Ordinary/recursive gradient relative difference | 约 `1.27e-08` |

这些误差包含不同 AD 求值顺序和 GPU 浮点归约差异，不表示 objective 不同。

## 3. CPU prototype 结果

以下是 CPU-only JAX、x64、2000 steps、`dt=0.025 ms` 的稳态测量。3.1 和 3.2 是
density runtime 迁移到 CV space 之前的基线；3.3 单独记录迁移前后 A/B。

### 3.1 三 CV

| Method | Steady median | XLA temporary | Sensitivity carry |
| --- | ---: | ---: | ---: |
| Reverse BPTT | 91.4 ms | 6.81 MB | temporal tape |
| Full-state-tree RTRL | 50.1 ms | 8.8 KB | 5,040 bytes |
| Compact $12\times9$ RTRL | 51.0 ms | 9.6 KB | 864 bytes |

Compact/full/BPTT 最大绝对梯度误差约为 `2.7e-12` 和 `7.1e-11`。Compact projection 将
logical carry 减少 5.83 倍，但每步仍嵌回 full functional state 执行 whole-cell JVP，因此没有
减少实际 tangent work，墙钟与 full RTRL 接近。

### 3.2 五 CV

| Method | Steady median | XLA temporary | Sensitivity carry |
| --- | ---: | ---: | ---: |
| Reverse BPTT | 96.8 ms | 11.19 MB | temporal tape |
| Full-state-tree RTRL | 54.0 ms | 20.0 KB | 12,720 bytes |
| Compact $20\times15$ RTRL | 61.3 ms | 22.6 KB | 2,400 bytes |

Compact/full/BPTT 最大绝对梯度误差约为 `1.8e-12` 和 `3.9e-12`。

### 3.3 Density runtime 迁移到 CV space

五 CV painted density state 从 point-tree dense storage 迁移到 CV dense storage：

| Object | Before | After |
| --- | ---: | ---: |
| `n_cv` | 5 | 5 |
| `n_point` DHS workspace | 11 | 11 |
| Na.m / Na.h / K.n shape | `(1, 11)` | `(1, 5)` |
| Captured state bytes | 848 | 560 |
| Full RTRL tangent carry | 12,720 | 8,400 |
| Compact carry | 2,400 | 2,400 |

迁移前后的 16-step target voltage 与 loss 位级一致；compact/full/BPTT gradient 最大绝对变化
不超过 `3.5e-18`。

十次稳态中位数：

| Kernel | Before | After | Before temporary | After temporary |
| --- | ---: | ---: | ---: | ---: |
| Primal terminal rollout | 9.43 ms | 6.89 ms | 3.07 KB | 2.22 KB |
| Reverse BPTT | 91.86 ms | 104.16 ms | 11.19 MB | 7.14 MB |
| Full-state-tree RTRL | 57.29 ms | 30.90 ms | 20.01 KB | 13.15 KB |
| Compact-adapter RTRL | 71.24 ms | 36.55 ms | 22.64 KB | 15.28 KB |

迁移稳定降低 state 和 temporary。RTRL 墙钟明显下降；BPTT temporary 下降约 36%，但该次
CPU 墙钟反而升高，因此不能据此声称 BPTT 加速或减速具有一般性。

### 3.4 统一 rollout engine

五 CV、x64 CPU、2000 steps、十次稳态测量：

| Method | Steady median | XLA temporary | Output |
| --- | ---: | ---: | ---: |
| Full RTRL | 29.35 ms | 14,968 bytes | 16,168 bytes |
| BPTT | 102.33 ms | 7,125,584 bytes | 16,168 bytes |

两者返回相同的 `(2000,)` local losses、scalar total loss 和三个 `(5,)` parameter gradients；
最大相对梯度差为 `5.4e-12`。正常 RTRL 结果不输出 sensitivity history。

## 4. A100 scaling

### 4.1 环境与协议

- NVIDIA A100-SXM4-80GB；
- JAX 0.10.1，x64；
- `dt=0.025 ms`；
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`；
- 每个 method/config 使用独立 worker process；
- 十次 synchronized steady executions；
- gradient kernel timing 不包含 Adam、target generation 和 worker startup。

### 4.2 Reference points

| Configuration | Method | Compile | Steady | XLA temporary | Process peak | RTRL carry |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `C1/T40/B16/S16` | BPTT | 4.664 s | 0.321 s | 275.50 MB | 1.02 GB | - |
| `C1/T40/B16/S16` | RTRL | 3.293 s | 0.082 s | 298.66 KB | 486.54 MB | 43.01 KB |
| `C5/T40/B16/S16` | BPTT | 34.582 s | 0.715 s | 1.40 GB | 2.67 GB | - |
| `C5/T40/B16/S16` | RTRL | 16.140 s | 0.230 s | 1.95 MB | 490.73 MB | 998.40 KB |
| `C9/T40/B16/S16` | BPTT | 101.684 s | 1.276 s | 2.65 GB | 4.83 GB | - |
| `C9/T40/B16/S16` | RTRL | 30.494 s | 0.459 s | 6.88 MB | 492.83 MB | 3.21 MB |
| `C9/T80/B32/S32` | BPTT | 435.058 s | 4.893 s | 21.42 GB | 35.95 GB | - |
| `C9/T80/B32/S32` | RTRL | 56.682 s | 1.389 s | 27.24 MB | 505.41 MB | 12.39 MB |

### 4.3 CV axis

固定 workload 的 large-CV 测量：

| CV | BPTT steady | RTRL steady | BPTT temporary | RTRL temporary | RTRL carry |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.321 s | 0.082 s | 275.50 MB | 298.66 KB | 43.01 KB |
| 3 | 0.484 s | 0.157 s | 799.96 MB | 833.21 KB | 364.03 KB |
| 5 | 0.715 s | 0.230 s | 1.40 GB | 1.95 MB | 998.40 KB |
| 7 | 0.938 s | 0.282 s | 1.93 GB | 3.38 MB | 1.95 MB |
| 9 | 1.276 s | 0.459 s | 2.65 GB | 6.88 MB | 3.21 MB |
| 13 | 1.943 s | 0.727 s | 3.78 GB | 13.68 MB | 6.67 MB |
| 17 | 2.830 s | 0.995 s | 5.26 GB | 22.86 MB | 11.38 MB |
| 25 | 9.517 s | 1.539 s | 7.72 GB | 48.38 MB | 24.58 MB |
| 33 | 13.780 s | 2.634 s | 10.85 GB | 78.67 MB | 42.78 MB |

在已测到的 33 CV 内没有出现 RTRL 墙钟反超 BPTT 的 crossover。该结论只描述当前
parameterization、GPU、batch/seed parallelism 和静态 shape。RTRL carry 按预期接近 CV
平方增长；更多参数、更少可并行 tangent direction 或不同 solver 都可能改变 crossover。

BPTT 在 17 到 25 CV 之间进入新的执行区间，runtime 增长明显快于 temporary。后面的
backsub A/B 表明 recursive doubling 不是这一变化的唯一原因。

### 4.4 Ordinary 与 recursive backsub

| CV | Method | Recursive | Ordinary | Ordinary/recursive | Recursive temporary | Ordinary temporary |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 9 | BPTT | 1.276 s | 1.375 s | 1.08x | 2.65 GB | 1.91 GB |
| 9 | RTRL | 0.459 s | 0.627 s | 1.37x | 6.88 MB | 6.93 MB |
| 17 | BPTT | 2.830 s | 3.106 s | 1.10x | 5.26 GB | 3.59 GB |
| 17 | RTRL | 0.995 s | 1.353 s | 1.36x | 22.86 MB | 23.01 MB |
| 25 | BPTT | 9.517 s | 10.058 s | 1.06x | 7.72 GB | 5.32 GB |
| 25 | RTRL | 1.539 s | 2.428 s | 1.58x | 48.38 MB | 48.51 MB |
| 33 | BPTT | 13.780 s | 14.376 s | 1.04x | 10.85 GB | 7.04 GB |
| 33 | RTRL | 2.634 s | 4.388 s | 1.67x | 78.67 MB | 83.46 MB |

Ordinary backsub 将 BPTT temporary 降低约 28--35%，代价是 4--10% 墙钟增加；对 RTRL
几乎不降低 temporary，却慢 37--67%。当前默认 recursive doubling 在这些配置上更快。

## 5. Adam 训练一致性

固定 `C5/T40/B16/S32` workload，BPTT 与 full RTRL 在独立 CUDA process 中从相同参数
开始，各运行 10 次 gradient + Adam update：

| Method | Compile | Gradient median | XLA temporary | Final mean loss |
| --- | ---: | ---: | ---: | ---: |
| BPTT | 35.85 s | 0.710 s | 2.806 GB | 65.5525765 |
| Full RTRL | 16.27 s | 0.245 s | 4.625 MB | 65.5525765 |

该固定 shape 下，RTRL gradient kernel 快 2.89x，temporary 小 606.8x。首轮与末轮 raw
gradient 最大相对差分别为 `8.82e-11` 和 `3.65e-9`；10 次 Adam update 后参数最大绝对差
为 `1.33e-10`。

这说明两种方法在该离散 objective 上产生了数值一致的优化轨迹，不表示所有参数规模下
RTRL 都更快。

## 6. 可复现入口

原始数据目录：

```text
examples/experimental/optim_gradient_scaling/artifacts/rtrl_bptt_scaling/
  pilot_block_exact/
  full_block_exact/
  large_cv_block_exact/
  backsub_ordinary_block_exact/
  RESULTS.md

examples/experimental/optim_training_comparison/artifacts/rtrl_bptt_training/
  c5_t40_b16_s32_e10/
```

重新聚合已有 scaling artifact，不运行 JAX：

```bash
python examples/experimental/optim_gradient_scaling/report.py
```

重新运行 A100 full suite：

```bash
python examples/experimental/optim_gradient_scaling/benchmark.py run \
  --suite full --gpu 7 --repeats 10 \
  --python /home/swl/anaconda3/envs/braincell_311/bin/python \
  --output-dir examples/experimental/optim_gradient_scaling/artifacts/rtrl_bptt_scaling/full_block_exact \
  --resume
```

分析 notebook：

1. `optim_gradient_correctness/gradient_diagnostics.ipynb`：同一 global loss 的总梯度；
2. `optim_gradient_correctness/single_cv_sensitivity.ipynb`：单 CV sensitivity、learning signal 和 prefix gradient；
3. `optim_gradient_scaling/analysis.ipynb`：性能、内存、GPU 和 backsub 数据。

## 7. 稳定结论

- Full BPTT 与 exact RTRL 在相同 fixed-parameter 离散 objective 上数值一致。
- RTRL carry 与 $N_sN_\theta$ 一致且不含时间轴；BPTT workspace 包含 temporal tape。
- Compact state projection 只有在同时减少实际 JVP state/directions 时才会转化为墙钟收益。
- GPU wall time 受并行填充、kernel span、memory traffic 和静态 shape regime 影响，不能只由
  大 O 推断。
- 当前 A100 workload 在 33 CV 以内未观察到 RTRL/BPTT 墙钟 crossover；这不是对更大参数
  维度或其他 solver 的保证。
