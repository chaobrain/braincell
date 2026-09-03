# Batch Size、数据规模与 GPU 吞吐

## 文档定位

本文记录九参数 conductance-fitting 实验的硬件测量，不定义 BrainCell API 或通用性能承诺。
必须区分 protocol batch、并行 candidate lanes 和 recurrent time length；硬件宽度近似为
`protocol batch * candidate lanes`，但更宽不等于更高优化效率。

## A100 容量测量

条件为 A100-SXM4-80GB、8 starts、float32 JAX、pure-voltage MSE、4,000 simulation steps。
超过 108-row train set 的 batch 会重复输入，只测容量，不是训练建议。

| Protocol batch | Warm seconds | Protocol-start lanes/s | Peak GiB |
| ---: | ---: | ---: | ---: |
| 72 | 11.05 | 52.1 | 5.8 |
| 108 | 10.84 | 79.7 | 8.7 |
| 144 | 11.29 | 102.0 | 11.6 |
| 216 | 11.45 | 151.0 | 17.3 |
| 288 | 11.40 | 202.1 | 23.0 |
| 360 | 11.24 | 256.3 | 28.8 |
| 432 | 11.25 | 307.3 | 34.6 |

batch 432 仍在提高吞吐，当前 workload 先碰到 allocator 限制而非明显 compute plateau。GPU
utilization 接近 100% 只说明 kernel 活跃。batch 不必是 8 的倍数；固定 shape、protocol 平衡和
避免 padded tail 更重要。108-row split 自然适配 18/36/54/108；batch 72 需要 72+36 两个 shape
或 25% padding。

## 训练质量测量

### 108-row、30 Epoch

| 配置 | Updates | Seconds | Mean test loss | Median test loss |
| --- | ---: | ---: | ---: | ---: |
| batch 36, Adam 0.02 | 90 | 169.3 | 0.2556 | 0.2743 |
| batch 54, Adam 0.03 | 60 | 127.8 | 0.2742 | 0.3234 |

batch 54 延长到 40 epochs/80 updates 时为 157.8 s，mean/median `0.2678/0.2949`；LR 降到
0.02 后 mean 恶化为 `0.3326`。线性 LR scaling 有帮助，但同等 wall-time 下 batch 36 更稳。

### 144 vs 216 Protocol、相同 90 Updates

两组都用 8 个相同 starts、30 epochs 和 pure-voltage MSE；baseline 为 batch36/LR0.02，
sine-expanded 为 batch54/LR0.03。

| Metric | Baseline | Sine-expanded |
| --- | ---: | ---: |
| Mean/median best validation | `0.1694 / 0.1220` | `0.2342 / 0.2281` |
| Mean/median test normalized MSE | `0.2260 / 0.2272` | `0.3488 / 0.4101` |
| Mean test trace RMSE | `10.59 mV` | `10.93 mV` |
| Exact soma spike fraction | `68.1%` | `52.3%` |
| Mean/median parameter relative RMS | `29.0% / 35.4%` | `30.6% / 34.9%` |
| Two-lane wall time | `600.3 s` | `633.3 s` |

normalized loss 因 held-out protocol 和 per-protocol normalizer 不同，不能严格横比；RMSE、hard
spike 和 parameter distance 更直接。当前预算下 sine diversity 未改善结果，但这不证明其有害：
30 epoch 时轨迹仍不规则，test 更接近 threshold，LR0.03 也未充分调参。

该旧路径只并行 2 starts，profile 约分配 62 GB、报告 99% utilization，但 recurrent backward
阶段仅约 67 W；这是脚本测量，不是 A100 最优吞吐估计。

### 同一 216-Protocol 数据：Batch 27 vs 54

固定 162 train、27 validation、27 test、30 passes，并将 8 starts 放入同一 `vmap`：

| Metric | Batch 27, LR 0.015 | Batch 54, LR 0.03 |
| --- | ---: | ---: |
| Updates | 180 | 90 |
| Mean/median best validation | `0.1907 / 0.1795` | `0.2313 / 0.2220` |
| Mean/median test normalized MSE | `0.2870 / 0.3241` | `0.3574 / 0.4055` |
| Mean test trace RMSE | `9.83 mV` | `11.17 mV` |
| Exact soma spike fraction | `59.7%` | `50.9%` |
| Mean/median parameter relative RMS | `26.6% / 30.0%` | `30.2% / 34.4%` |
| Eight-lane wall time | `298.6 s` | `154.3 s` |

batch 27 的全部聚合质量更好，但 updates 加倍、耗时约 1.94 倍；这不能区分 gradient noise 与
update count，因为固定的是 data exposure。将 2-start chunks 改为单次 8-start lane，使同一
batch54/LR0.03 配置从 633.3 s 降至 154.3 s，candidate-lane width 必须作为执行参数记录。

## 统计解释与选择

不放回抽样时 mini-batch gradient covariance 含 finite-population factor：

```text
(N - B) / (B * (N - 1))
```

`N=108` 时 batch36 使用 1/3 数据，batch54 使用 1/2 数据，后者采样方差约减半但每 epoch 仅
两次更新。扩展到 162 条独有 protocol 后，batch54 恢复为 1/3。重复 trace 只扩大硬件宽度，
不增加 identifiability。

选择顺序为：构造平衡固定 shape，测 compile/throughput/memory，为每个 batch 独立调 LR，再
比较达到 held-out 目标的时间；分别报告 equal-epoch、equal-update、equal-wall-clock，并在多个
训练阶段测 per-protocol gradient dispersion。剩余 GPU 宽度优先用于独立 starts 或超参数。

当前建议：108-row baseline 用 batch36/LR0.02；162-row 质量优先用 batch27/LR0.015，速度优先
用 batch54/LR0.03；A100 的固定八初值实验使用 8 candidate lanes，CPU 保留低内存 2 lanes。
validation/test 使用 native shape；padding 必须有 validity mask，指标保存前去除 padded traces。
冻结已收敛 lane 不会自动减少统一 `vmap` 的 forward/backward 成本，必须作为独立性能消融。

## References

1. Hoffer et al. [Train longer, generalize better](https://arxiv.org/abs/1705.08741), 2017.
2. Goyal et al. [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677), 2017.
3. McCandlish et al. [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162), 2018.
4. Shallue et al. [Effects of Data Parallelism](https://www.jmlr.org/papers/v20/18-789.html), 2019.
5. Smith et al. [Generalization Benefit of Noise in SGD](https://proceedings.mlr.press/v119/smith20a.html), 2020.
6. Wu et al. [Noisy Gradient Descent that Generalizes as SGD](https://proceedings.mlr.press/v119/wu20c.html), 2020.
7. Golmant et al. [Computational Inefficiency of Large Batch Sizes](https://openreview.net/forum?id=S1en0sRqKm), 2019.
