# Batch Size, Dataset Scale, and GPU Throughput

## Reference 状态

本文记录特定参数拟合实验的硬件测量和 batch-size 方法分析，是非规范性 reference，
不定义 BrainCell 公共 API 或通用性能承诺。规范性范围见
[Design Overview](../design-overview.md)。

## Purpose

This note records the batch-size discussion and measurements from the
heterogeneous nine-parameter conductance-fitting experiment.  It distinguishes
hardware throughput from optimization efficiency so later experiments do not
select a batch solely because it fills a GPU.

## Three Batch Axes

The experiment has three independent scales:

- **protocol batch**: stimulus traces contributing to one gradient;
- **candidate lanes**: starts, learning rates, or schedule seeds trained in
  parallel; and
- **time length**: 4000 recurrent simulation steps per trace.

The useful hardware width is approximately `protocol batch * candidate lanes`.
A protocol batch does not need to be a multiple of eight.  Static shapes,
balanced protocol composition, and avoiding a large padded tail matter more
for these elementwise dynamics and reductions than tensor-core alignment.

## Card 6 Measurements

Measurements used one NVIDIA A100-SXM4-80GB, eight start lanes, float32 JAX,
the pure-voltage MSE objective, and 4000 simulation steps.  Synthetic batches
larger than the 108-row training set repeat inputs only to measure hardware
capacity; they are not valid training recommendations.

| Protocol batch | Warm seconds | Protocol-start lanes/s | Peak GiB |
| ---: | ---: | ---: | ---: |
| 72 | 11.05 | 52.1 | 5.8 |
| 108 | 10.84 | 79.7 | 8.7 |
| 144 | 11.29 | 102.0 | 11.6 |
| 216 | 11.45 | 151.0 | 17.3 |
| 288 | 11.40 | 202.1 | 23.0 |
| 360 | 11.24 | 256.3 | 28.8 |
| 432 | 11.25 | 307.3 | 34.6 |

Throughput was still increasing at batch 432.  The current fitting workload
therefore reaches an allocation limit before a clear compute-throughput
plateau.  GPU utilization near 100% only means kernels are active; it does not
prove maximum examples per second.

## End-to-End Batch Comparison

The 108-row training split naturally supports batches 18, 36, 54, and 108.
Batch 72 is awkward: an epoch is 72 plus 36, requiring two compiled shapes or
25% padded computation.

On the same GPU allocator and 30-epoch budget:

| Configuration | Updates | Train seconds | Mean test loss | Median test loss |
| --- | ---: | ---: | ---: | ---: |
| batch 36, Adam 0.02 | 90 | 169.3 | 0.2556 | 0.2743 |
| batch 54, Adam 0.03 | 60 | 127.8 | 0.2742 | 0.3234 |

Extending batch 54 to 40 epochs gave 80 updates in 157.8 seconds, mean test
loss 0.2678, and median 0.2949.  Lowering its learning rate to 0.02 worsened
mean test loss to 0.3326.  Thus linear learning-rate scaling helped, but batch
36 remained more robust at a similar wall-clock budget.

## Sine-Expanded Equal-Update Comparison

The response-calibrated 216-protocol dataset was compared with the original
144-protocol dataset on card 6.  Both used the pure-voltage MSE component,
eight identical initial points, 30 epochs, and exactly 90 Adam updates.  The
baseline used batch 36 and learning rate 0.02; the expanded train split used
batch 54 and learning rate 0.03.  Validation and test used their native sizes
of 18 and 27 respectively.

| Metric | Baseline | Sine-expanded |
| --- | ---: | ---: |
| Mean best validation loss | 0.1694 | 0.2342 |
| Median best validation loss | 0.1220 | 0.2281 |
| Mean test normalized MSE | 0.2260 | 0.3488 |
| Median test normalized MSE | 0.2272 | 0.4101 |
| Mean test trace RMSE (mV) | 10.59 | 10.93 |
| Exact soma spike-count fraction | 68.1% | 52.3% |
| Mean nine-parameter relative RMS | 29.0% | 30.6% |
| Median nine-parameter relative RMS | 35.4% | 34.9% |
| Current two-lane wall time (s) | 600.3 | 633.3 |

The normalized losses are not a strict cross-dataset metric: the held-out
protocols differ and each protocol is divided by its own canonical-error
normalizer.  Trace RMSE, hard spike accuracy, and parameter distance are more
directly comparable.  At this budget the additional sine diversity did not
improve held-out waveform or spike accuracy.  This does not establish that
the expanded data are harmful: several expanded trajectories remained
irregular at epoch 30, the expanded test set deliberately contains more
threshold-sensitive sine responses, and learning rate 0.03 has not been tuned
beyond linear scaling.

The wall times above use the existing two-start chunk path.  GPU profiling
showed approximately 62 GB allocated, 99% reported utilization, but only about
67 W during the long recurrent backward pass.  These are valid measurements
of the current script, not an estimate of optimal A100 throughput.  A separate
optimization should make candidate-lane width configurable and test all eight
starts in one compiled call.

## Same-Dataset Batch 27 versus Batch 54

The corrected batch comparison holds the 216-protocol sine-expanded dataset
fixed.  Both configurations see the 162-row train split for 30 complete data
passes and use the same 27 validation and 27 test protocols.  Batch 27 uses
learning rate 0.015 and 180 updates; batch 54 uses learning rate 0.03 and 90
updates.  Both train all eight initial points in one GPU vmap lane group.

| Metric | Batch 27 | Batch 54 |
| --- | ---: | ---: |
| Mean best validation loss | 0.1907 | 0.2313 |
| Median best validation loss | 0.1795 | 0.2220 |
| Mean test normalized MSE | 0.2870 | 0.3574 |
| Median test normalized MSE | 0.3241 | 0.4055 |
| Mean test trace RMSE (mV) | 9.83 | 11.17 |
| Exact soma spike-count fraction | 59.7% | 50.9% |
| Mean nine-parameter relative RMS | 26.6% | 30.2% |
| Median nine-parameter relative RMS | 30.0% | 34.4% |
| Updates per start | 180 | 90 |
| Eight-lane wall time (s) | 298.6 | 154.3 |

Batch 27 is better on every aggregate quality metric in this run, but costs
twice as many optimizer updates and approximately 1.94 times the wall time.
The result supports the hypothesis that more frequent updates help follow the
additional directions introduced by the expanded protocols.  It does not by
itself separate batch-gradient noise from update count because the comparison
holds data exposure, not optimizer steps, constant.  A later equal-update
ablation can compare the first 90 B27 updates against B54 if that distinction
becomes important.

Moving from four serial two-start chunks to one eight-start lane reduced the
measured wall time substantially: the prior batch-54 run took 633.3 seconds,
whereas the same batch and learning-rate configuration took 154.3 seconds in
the eight-lane comparison.  Candidate-lane width should therefore remain an
explicit experimental execution parameter.

## Dataset Fraction and Gradient Noise

For a dataset of size `N` sampled without replacement with batch `B`, the
mini-batch gradient covariance contains the finite-population factor

```text
(N - B) / (B * (N - 1)).
```

With 108 training protocols, batch 36 consumes one third of the data while
batch 54 consumes one half.  The latter has approximately half the sampling
variance and only two updates per epoch.  Expanding the train split to 162
unique protocols makes batch 54 one third of the data and restores three
updates per epoch.

Increasing `N` helps only when the new protocols add stimulus or response
diversity.  Duplicating existing traces improves apparent hardware width but
does not add identifiability or gradient information.

## How to Choose a Batch

Use the following order:

1. construct balanced, fixed-shape candidates that cover the train split;
2. measure cold compile time, warm throughput, and peak memory;
3. tune learning rate for each candidate rather than sharing one value;
4. compare time to a held-out loss target, not epoch throughput alone;
5. report equal-epoch, equal-update, and equal-wall-clock views separately;
6. measure gradient noise or per-protocol gradient dispersion near several
   stages of training to estimate the useful critical batch;
7. spend spare GPU width on independent starts or hyperparameters instead of
   duplicating protocols.

Validation and test shapes are independent of the training batch.  Small held-
out splits should use their native shape; a non-divisible final evaluation
batch requires a validity mask and removal of padded traces before metrics are
saved.

## Current Recommendation

- Keep batch 36 and learning rate 0.02 for the 108-row baseline.
- Prefer batch 27 and learning rate 0.015 when fitting quality is the priority
  on the 162-row sine-expanded train split.
- Retain batch 54 and learning rate 0.03 as the faster option when approximately
  halving optimizer updates and wall time matters more than endpoint quality.
- Use eight candidate lanes on the A100 for the fixed eight-start experiment;
  retain two lanes as the lower-memory CPU default.
- Do not conflate best-checkpoint retention with inactive lanes.  The current
  early-stop implementation can freeze a lane while still paying its complete
  forward/backward cost; that issue requires a separate ablation.

## References

1. Hoffer, Hubara, and Soudry. [Train longer, generalize better](https://arxiv.org/abs/1705.08741), 2017.
2. Goyal et al. [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677), 2017.
3. McCandlish et al. [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162), 2018.
4. Shallue et al. [Measuring the Effects of Data Parallelism on Neural Network Training](https://www.jmlr.org/papers/v20/18-789.html), 2019.
5. Smith, Elsen, and De. [On the Generalization Benefit of Noise in Stochastic Gradient Descent](https://proceedings.mlr.press/v119/smith20a.html), 2020.
6. Wu et al. [On the Noisy Gradient Descent that Generalizes as SGD](https://proceedings.mlr.press/v119/wu20c.html), 2020.
7. Golmant et al. [On the Computational Inefficiency of Large Batch Sizes for Stochastic Gradient Descent](https://openreview.net/forum?id=S1en0sRqKm), 2019.
