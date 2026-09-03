# Online Learning 实验

## 状态与范围

本目录包含 BrainCell 的 exact forward sensitivity、RTRL/BPTT 对照、梯度诊断和性能
scaling 实验。这里的 Python 类型和调用方式仍是 experimental contract，不属于 BrainCell
公共 API；实现会直接暴露 JAX/BrainState 的 state trace、parameter coordinate 和 tangent
PyTree，以便验证数学等价性和工程成本。

当前代码按职责分层：

```text
../optim/_forward_sensitivity.py
  StatefulFunction、parameter coordinates、full/compact sensitivity recurrence
            |
            +--> ../optim/gradients.py
            |      正常 BPTT/RTRL engine 与独立 diagnostics
            |
            +--> ../optim_gradient_correctness/multicv_hh.py
                   compact/full/BPTT 的模型级正确性实验

../optim/gradients.py
  +--> rtrl_bptt_scaling_benchmark.py
         seed-block、batch-shared GPU benchmark 与 artifacts
  +--> rtrl_bptt_training_comparison.py
         固定 workload 的完整 Adam 训练对照
```

这些层不应仅为了减少文件数而合并。Core 不依赖具体模型；rollout engine 定义训练调用；
multi-CV 和 scaling 模块分别提供 compact projection 与性能证据。

## 文件导航

| 文件 | 责任 | 适用场景 |
| --- | --- | --- |
| `../optim/_forward_sensitivity.py` | Stateful step functionization、parameter coordinates、full/compact recurrence、参数相关初始化 | 新的 exact forward-sensitivity 算法和底层正确性验证 |
| `../optim/_forward_sensitivity_test.py` | Prefix gradient、finite difference、reset、固定 delay 和 shape 测试 | Core 回归 |
| `../optim/gradients.py` | local one-pass 与 trajectory two-pass engine，自动参数发现和独立 diagnostics | 常规训练实验 |
| `../optim/gradients_test.py` | BPTT/RTRL 等价、optimizer mapping、reset、global trace loss 和 `S/L/D` 分解 | Engine 回归 |
| `../optim_gradient_correctness/multicv_hh.py` | 可配置 HH Cell 的 compact/full/BPTT 比较 | Compact RTRL 参考，不是首选训练 API |
| `../optim_gradient_correctness/multicv_hh_test.py` | Directional finite difference、projection identity、distal coupling、multi-CV shape | Compact/full 正确性 oracle |
| `rtrl_bptt_scaling_benchmark.py` | Block-exact seed `vmap`、batch-shared Cell、独立 worker、GPU monitor、可恢复 suites | 性能实验 |
| `rtrl_bptt_scaling_benchmark_test.py` | Suite、seed isolation、CPU 小配置、monitor 和 artifact 协议 | Benchmark harness 回归 |
| `rtrl_bptt_scaling_report.py` | 读取已有 CSV，生成本地 `RESULTS.md` | 整理结果，不运行 JAX/GPU |
| `rtrl_bptt_scaling_report_test.py` | Report 缺失数据、recursive/ordinary 合并和写入测试 | Report 回归 |
| `rtrl_bptt_training_comparison.py` | `C5/T40/B16/S32` 的独立 BPTT/RTRL Adam worker、history 和对照图 | 完整训练效率/一致性检查 |
| `rtrl_bptt_training_comparison_test.py` | 小规模两 epoch 训练与 artifact 协议 | 训练 harness 回归 |
| `../optim_gradient_correctness/gradient_diagnostics.ipynb` | 同一个 global trajectory loss 下的 BPTT/two-pass RTRL 总梯度对照 | 方法入门 |
| `../optim_gradient_correctness/single_cv_sensitivity.ipynb` | 单 CV sensitivity、learning signal、prefix gradient、Adam moments 和 epoch slider | 理解 online quantity |
| `rtrl_bptt_scaling_analysis.ipynb` | 只读 CSV/NPZ，绘制 full、large-CV、GPU monitoring 和 backsub A/B | 性能分析；不会启动 benchmark |

测试遵守同目录 `*_test.py` 规则。Notebook 保存已执行输出，因此查看结论不要求重新编译
耗时模型。

## 正常梯度调用

Rollout engine 默认从目标 Cell 发现 trainable roots，在 reset 和每步执行前 materialize，返回
可直接交给同一 optimizer state mapping 的梯度：

```python
from examples.experimental.optim.gradients import (
    build_rollout_value_and_grad,
)


def rollout_step(step_data):
    time_ms, target_mv = step_data
    voltage_mv = cell.V.value.to_decimal(u.mV)
    local_loss = jnp.mean((voltage_mv - target_mv) ** 2) / num_steps
    with brainstate.environ.context(t=time_ms * u.ms):
        cell.update()
    return local_loss


engine = build_rollout_value_and_grad(
    cell,
    step=rollout_step,
    method="rtrl",  # 或 "bptt"
)
engine.prepare((times_ms[0], target_voltage_mv[0]))
result = engine((times_ms, target_voltage_mv))
optimizer.update(result.gradients)
```

Normal mode 只返回逐步 losses、总 loss 和总参数梯度，不保存 sensitivity history。完整 rollout
期间参数固定，结束后只做一次 optimizer update，因此 exact RTRL gradient 可与 full BPTT
直接比较。

这是默认且最快的路径，要求 objective 可写成逐步 scalar loss 的和。MSE、MAE、多个局部 loss
加权都可以直接放进 `rollout_step`。

## Global Trajectory Loss

当 loss 依赖整条输出轨迹，使用 opt-in two-pass engine：

```python
from examples.experimental.optim.gradients import (
    build_trajectory_value_and_grad,
)


def observe_step(data):
    time_ms, _target_mv = data
    voltage_mv = cell.V.value.to_decimal(u.mV)
    with brainstate.environ.context(t=time_ms * u.ms):
        cell.update()
    return voltage_mv


def trajectory_loss(voltage_mv, data):
    target_mv = data[1]
    mse = jnp.mean((voltage_mv - target_mv) ** 2)
    smoothness = jnp.mean(jnp.diff(voltage_mv, axis=0) ** 2)
    return mse + 1e-3 * smoothness


engine = build_trajectory_value_and_grad(
    cell,
    step=observe_step,
    loss=trajectory_loss,
    method="rtrl",  # 或 "bptt"
)
result = engine(step_data)
optimizer.update(result.gradients)
```

`step` 可以返回 floating-array PyTree，`loss(observations, step_data)` 可以执行任意可微的跨时间
运算，包括全 trace mean、相邻时间差和多个 observation 的加权组合。RTRL 第一遍生成
observation trace，并自动反传得到每个时刻的 learning signal；第二遍重放 transition，在线传播
full sensitivity 并立即收缩。它比 local one-pass 多一次仿真，并保存
`O(T * N_observation)` 的 observation/learning-signal trace，但不保存
`O(T * N_state * N_parameter)` 的 sensitivity history。正常结果仍只有 scalar loss 和命名总梯度。

当前 trajectory loss 必须通过 `observations` 依赖可训练参数；参数本身的显式 regularization
应在 optimizer/controller 层另加，或扩展该 experimental contract 后再放入 loss。

## Diagnostics

Diagnostics 使用单独的编译路径：

```python
sampled = engine.diagnose(step_data, at=(0, 100, 399))
all_steps = engine.diagnose(step_data)  # 只用于分析
```

结果包含 full-state sensitivity、learning signal、direct parameter term、eligibility
contraction、local gradient、prefix gradient 和 decomposition residual，并验证：

```text
local_gradient_t = S_t @ L_t + D_t
total_gradient   = sum_t local_gradient_t
```

记录所有时间点会产生 `O(T * N_state * N_parameter)` 输出内存，不能用该路径衡量 normal
RTRL 的内存和速度。

## Compact RTRL 参考

`../optim_gradient_correctness/multicv_hh.py` 保留的原因，是它回答了与正常 full RTRL 不同的问题：只传播经过证明的
active-state projection 时，可以减少多少跨时间 carry。当前 compact adapter 每步仍把
tangent 嵌回 full functional state 做 whole-step JVP，所以 carry 下降不保证 temporary 或
墙钟等比例下降。

常规 full RTRL 训练使用 `../optim/gradients.py`；修改 projection 语义或比较
compact/full/BPTT 时使用 `../optim_gradient_correctness/multicv_hh.py`。

## Scaling Benchmark

只列出 suite，不运行：

```bash
python examples/experimental/online_learning/rtrl_bptt_scaling_benchmark.py run \
  --suite full --dry-run
```

使用保存 A100 结果时的 CUDA 环境：

```bash
python examples/experimental/online_learning/rtrl_bptt_scaling_benchmark.py run \
  --suite full --gpu 7 --repeats 10 \
  --python /home/swl/anaconda3/envs/braincell_311/bin/python \
  --output-dir examples/experimental/online_learning/artifacts/rtrl_bptt_scaling/full_block_exact \
  --resume
```

当前 suites：

| Suite | 用途 |
| --- | --- |
| `pilot` | 九个 endpoint/baseline 配置，用于验证协议 |
| `full` | CV、duration、batch、seed 单轴 sweep，加四个 interaction corners |
| `large_cv` | 固定 `T40/B16/S16`，测试 `C=13,17,25,33` |
| `backsub_ab` | `C=9,17,25,33` 的普通 Hines backsub；需传 `--backsub ordinary` |

每个 method/config 在新 subprocess 中运行。成功 trial 写入 JSON metadata 和压缩后的
loss/gradient NPZ；`results.csv` 聚合配对误差。`--resume` 会跳过成功 trial，并保留结构化
失败结果。

Benchmark 只测 gradient kernel。Adam 对 BPTT/RTRL 相同，因此不计时；target generation 和
worker startup 也不进入 steady time。

完整 Adam 对照使用固定的 `C5/T40/B16/S32` workload：

```bash
python examples/experimental/online_learning/rtrl_bptt_training_comparison.py run \
  --gpu 7 --epochs 10 \
  --python /home/swl/anaconda3/envs/braincell_311/bin/python \
  --resume
```

两个方法在独立 CUDA subprocess 中从相同参数出发，各自执行 gradient 和 Adam update。输出
保留每个 epoch、seed、channel、CV 的 loss、raw gradient 和更新前后参数，而不保存 state trace。

当前 A100 x64 实测（10 epochs）如下；这是该固定 shape 的结果，不外推为渐近结论：

| Method | Compile | Gradient median | XLA temporary | Final mean loss |
| --- | ---: | ---: | ---: | ---: |
| BPTT | 35.85 s | 0.710 s | 2.806 GB | 65.5525765 |
| full RTRL | 16.27 s | 0.245 s | 4.625 MB | 65.5525765 |

RTRL 的 gradient kernel 在这里快 `2.89x`，temporary 小 `606.8x`。首轮/末轮 raw gradient
最大相对差分别为 `8.82e-11` 和 `3.65e-9`；10 次 Adam 更新后的参数最大绝对差为
`1.33e-10`。更大 CV、seed 或参数维度下，RTRL 的 `N_state * N_parameter` carry 与 tangent
计算最终会成为主导，需结合 scaling suites 判断 crossover。

## Result Artifacts

生成数据被 Git 忽略：

```text
examples/experimental/online_learning/artifacts/rtrl_bptt_scaling/
  pilot_block_exact/
  full_block_exact/
  large_cv_block_exact/
  backsub_ordinary_block_exact/
  RESULTS.md

examples/experimental/online_learning/artifacts/rtrl_bptt_training/
  c5_t40_b16_s32_e10/
    bptt.json / bptt.npz
    rtrl.json / rtrl.npz
    comparison.json / comparison.png
```

每个 run 目录包含：

```text
manifest.json
results.csv
trials/*.json
trials/*.npz
logs/*.log
```

不运行 JAX，直接重新生成本地总报告：

```bash
python examples/experimental/online_learning/rtrl_bptt_scaling_report.py
```

`RESULTS.md` 是本机硬件测量报告，CSV/NPZ 是权威原始数据。稳定的数学与 API 分析见
[BPTT/RTRL 通用理论](../../../docs/design/optim/references/bptt-to-rtrl-neuron-derivation.md)。

## Notebook 阅读顺序

1. `../optim_gradient_correctness/gradient_diagnostics.ipynb`：先确认同一个 global loss 的总梯度相等。
2. `../optim_gradient_correctness/single_cv_sensitivity.ipynb`：理解单参数、时间和 optimizer 行为。
3. `rtrl_bptt_scaling_analysis.ipynb`：查看性能、内存、GPU 和 backsub 数据。

不要把长时间 benchmark 执行单元加入分析 notebook；长任务必须走可恢复 CLI。

## 验证

运行全部实验测试：

```bash
pytest -q examples/experimental/online_learning examples/experimental/optim examples/experimental/optim_gradient_correctness
```

修改 DHS/backsub 时还要运行：

```bash
pytest -q braincell/quad/_staggered_test.py
```

默认 DHS backsub 始终是 recursive doubling。只有设置
`BRAINCELL_DHS_BACKSUB=ordinary` 或 benchmark 对应选项时才使用普通 Hines 路径。
