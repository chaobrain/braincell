# 电压轨迹与 Spike-Aware 参数训练

## 文档定位

本文是 loss、curriculum 和评价方法的非规范性 reference，不定义参数 API、梯度算法或
optimizer controller。相关主文档为：

| 主题 | 主文档 |
| --- | --- |
| 参数选择、映射与单位 | [API](../api.md) 与 [Architecture](../architecture.md) |
| 训练诊断、archive、spike region 与恢复 | [模块化训练诊断](modular-training-diagnostics.md) |
| BPTT/RTRL | [通用理论](bptt-to-rtrl-neuron-derivation.md) |
| batch 与 GPU | [Batch Size 与 GPU 吞吐](batch-size-and-gpu-throughput.md) |

本文区分 subthreshold trace、包含动作电位的 spiking trace，以及由阈值检测得到的 event
trace。三者不能使用同一条未经归一化的 raw MSE 作为唯一目标。

## 核心选择

| 数据 | 早期目标 | 后期目标 | 不应单独使用 |
| --- | --- | --- | --- |
| 无 spike | resting、steady state、time constant、低频波形 | 多尺度 trace、导数、held-out 泛化 | raw MSE |
| 含 spike | 正确兴奋性区域、count、大致 timing | alignment、AP shape、ISI、AHP、background | 未对齐 MSE、永久 clipping |

所有 loss component 必须先用固定 data-derived、noise-derived 或 initialization-derived
尺度无量纲化，再显式加权。target preprocessing、mask、normalizer 和 protocol split 在
训练前确定并 stop-gradient；prediction-derived mask 会改变目标，不使用。

BrainCell 的 `get_spike` 以 `V_th=0 mV` 上穿和默认
`braintools.surrogate.ReluGrad()` 产生 surrogate gradient。它在接近阈值时可支持 event
loss，但完全静默、远离阈值时可能没有足够梯度，因此仍需连续 voltage margin。

## 无 Spike 路线

长稳态区会按样本数压过短暂 onset，MSE 也容易受 outlier、protocol 幅度和 probe 数影响。
推荐目标为：

```text
L_no_spike = w_v    * L_voltage
           + w_dv   * L_derivative
           + w_ms   * L_multiscale
           + w_feat * L_features
           + w_reg  * L_regularization

L_hat_k = L_k / stop_gradient(max(L_k_at_initialization, epsilon))
L_total = sum_k w_k * L_hat_k
```

| 分量 | 建议定义 |
| --- | --- |
| `L_voltage` | 以 mV 表示误差后的 Huber 或 MAE |
| `L_derivative` | 平滑 `dV/dt` 的 Huber，强调 onset/offset |
| `L_multiscale` | 原分辨率及多个低通/降采样尺度的 trace loss |
| `L_features` | resting、steady deflection、input resistance、tau、sag、rebound |
| `L_regularization` | parameter prior、空间平滑、hierarchical shrinkage |

有实验噪声估计时优先按观测标准差归一化。normalizer 必须固定，不能随 batch 改变目标。

训练顺序为：resting/steady feature，onset/offset 与 `dV/dt`，多尺度完整 trace，逐步扩大
horizon，最后降低 LR 并以 held-out current sweep 选模型。探索范围而非默认值可从 Adam
LR `{1e-3, 3e-3, 1e-2, 3e-2}`、clip norm `1.0`、Huber `delta=1--3 mV`、归一化后
regularization `1e-3--1e-2` 开始；快速实验至少 16 starts，正式可辨识性分析建议
32--64 个以上。

评价至少包括 train/held-out voltage、resting/steady/tau/sag feature、gradient norm、bound
position 和多 start 终点；synthetic 数据增加 parameter recovery，等价低损失解按集合报告。

## 含 Spike 路线

spike 是稀疏但信息密度高的事件。raw MSE 同时存在：

| 问题 | 表现 | 后果 |
| --- | --- | --- |
| 时间稀释 | `100 ms` 中 `5 ms` spike 被 `95 ms` background 稀释 | 静默预测也可能得到较低 loss |
| 相位敏感 | 相同波形仅有小时间偏移 | 逐点误差形成两个大峰 |
| count boundary | 小参数变化使 spike 产生/消失 | 狭谷、台阶与梯度骤变 |

把 MSE 换成 Huber 只处理 outlier，不能解决以上三项。Jaxley 使用 50-step sliding maximum、
30-step stride、soft-DTW 和共同 unit-interval scaling（原始 `dt=0.025 ms`）；这是参考配置，
不是 BrainCell 默认值。

### 组合目标

```text
L_spike = w_bg     * L_background
        + w_win    * L_spike_window
        + w_event  * L_filtered_events
        + w_count  * L_soft_count
        + w_shape  * L_AP_shape
        + w_align  * L_local_alignment
        + w_reg    * L_regularization
```

| 分量 | 定义与边界 |
| --- | --- |
| background | target-fixed spike window 外的 Huber |
| spike window | 每个 target spike 附近的正权重 waveform；首轮可试 `[-1,+4] ms` |
| filtered events | event train 经宽/窄 kernel 后比较，如 `tau=5 ms` 与 `2 ms` |
| soft count | smooth crossing 总数差，提供 count 错误时的连续信号 |
| AP shape | 对齐后的 amplitude、half-width、max `dV/dt`、repolarization、AHP |
| local alignment | 局部 window 或降采样 envelope 上的 soft-DTW |
| regularization | 与无 spike 路线相同 |

background 与 spike window 各自按有效样本数平均，再通过 `w_bg/w_win` 聚合；不能统一除以
完整 time-step 数。target-fixed window 只定义区域，额外 predicted spike 仍由全时程 event、
count 和 no-extra-event 项捕获。

soft-DTW 时间与存储近似二次增长，也可能以过度 warping 隐藏错误模式，因此只用于局部
window、降采样 envelope 或有限 Sakoe-Chiba band，并显式惩罚偏移。Victor--Purpura 更适合
hard 评价；phase-plane `(V,dV/dt)` density 可作为后续 shape-objective 消融。

### Mask 与 Alignment

| 方法 | 优点 | 风险 | 用途 |
| --- | --- | --- | --- |
| hard clipping | 简单抑制 peak | clipped 区域零梯度，丢失 width/amplitude | 仅早期 subthreshold stage |
| target-fixed mask | 区域稳定 | 没有独立 spike loss 时会丢信息 | background/window 共同预处理 |
| prediction-derived mask | 看似自适应 | 模型可通过改变 mask 规避错误 | 禁止 |
| soft clipping / envelope | 梯度较平滑 | 增加温度/窗口超参数 | alignment preprocessing |
| raw AP shape | 约束 `gNa/gK` | 对 timing shift 极敏感 | 对齐后使用 |

学习 `gNa`、`gK` 或 kinetics 时，后期必须恢复 amplitude、width、upstroke 和 repolarization
监督；“尖峰不学”只能是早期 curriculum。

### 从静默到放电

```text
V_peak = smooth_max(V_in_target_spike_window)
L_margin = softplus((V_threshold + margin - V_peak) / temperature)
```

target 应放电但 prediction 静默时使用上述方向；target 静息时反转方向。多 spike protocol
按每个 target spike 的局部窗口计算，不能用单个全局 maximum 掩盖缺失事件。count 正确后
降低 margin，提高 timing 和 AP-shape 权重。

### Curriculum 与选择

| Stage | 新增重点 |
| ---: | --- |
| 1 | background/spike-window 分别归一化；resting、steady、smooth peak、宽 margin |
| 2 | soft count 和宽时间尺度 filtered event，修正缺失/额外 spike |
| 3 | 窄 event filter，降低 surrogate temperature、收紧 timing tolerance |
| 4 | local waveform、amplitude、half-width、`dV/dt`、AHP |
| 5 | timing 仍主导时加入局部 envelope soft-DTW |
| 6 | 全 trace/protocol 低 LR 精修，以 held-out 选模型 |

不要一次打开全部分量后手调权重。保存每项的 normalizer、gradient norm 和 gradient cosine；
GradNorm 只能作为有消融证据的实验候选。

checkpoint 字典序为 hard spike-count exact protocol fraction、unmatched/timing/event distance、
spike-window waveform 与 background error。hard count 和 Victor--Purpura 用于选择与报告，
不强求梯度。具体双 archive 和 region 规则见[模块化训练诊断](modular-training-diagnostics.md)。

当前主线之外的方法保留以下边界：

| 方法 | 适用位置 |
| --- | --- |
| Polyak-style `step = gamma * grad(L) / norm(grad(L))^beta` | Jaxley 使用过的 optimizer 候选；必须与 Adam 在同 seed/预算下消融 |
| EventProp | 适合 continuous-time hard event/root-finding；当前 fixed-`dt` conductance solver 不直接具备其 jump/adjoint 语义 |
| point-process likelihood | 只有 spike time、没有可信连续 voltage 时；当前 synthetic 数据会因此丢失 AP shape 和 subthreshold 信息 |
| full-trace soft-DTW | 成本近二次且可能过度 warping；仅保留局部或降采样版本 |

## 当前实现证据

历史九参数实验的七个可微分量及缺口为：

| 分量 | 历史定义 | 当前判断 |
| --- | --- | --- |
| `voltage` | 全 compartment masked Huber；target window `[-1,+3] ms` 权重 `0.1` | background 近似 |
| `derivative` | masked `dV` Huber | AP shape 的一部分，window 也被降权 |
| `multiscale` | 20-step block-mean Huber | 低频 trace/envelope |
| `event` | smooth crossing 后 `tau=2 ms` 指数滤波 MSE | 缺少宽时间尺度 |
| `count` | smooth crossing 总数差平方 | soft count |
| `peak` | `20--100 ms` soma smooth maximum 差平方 | 粗粒度 spike birth |
| `mse` | 全 protocol/time/compartment MSE | 时间稀释且相位敏感 |

尚未证明的能力包括独立归一化的正权重 spike-window loss、宽窄双 event filter 和显式
silent-to-spike margin，不能写成现行 API。

被当前接口取代的本地历史实现只保留为证据索引：

| 来源 | 证明了什么 | 当前替代 |
| --- | --- | --- |
| `55871ea`: `heterogeneous_nine_parameter_training.py` | 多参数梯度训练和 composite ablation 可行 | 当前 trainable API 与 diagnostics |
| `55871ea`: `heterogeneous_protocol_dataset.py` | 多 protocol、probe 和目标预处理可行 | 当前 parameter-learning experiment modules |
| [`SC01_fitting_a_hh_neuron.py`](../../../../examples/single_compartment/SC01_fitting_a_hh_neuron.py) | 单 compartment HH gradient-free fitting | 保留为非梯度 baseline |
| [`get_spike`](../../../../braincell/_base_neuron.py) | surrogate crossing 已存在 | 继续作为 event 信号，不承担完整 loss |

详细旧实现由 Git 历史追溯；当前入口见
[Parameter Learning README](../../../../examples/experimental/parameter_learning/README.md)。

## 四个实验模板

| 模板 | 模型/数据 | 训练重点 | 验证重点 |
| --- | --- | --- | --- |
| dendritic leak | target `0.6`、initial `0.15`、bounds `[0.01,1.0] mS/cm^2`；多 subthreshold sweeps | feature 后加原分辨率、4x/16x low-pass；LR `{0.003,0.01,0.03}` | held-out amplitude、1D landscape、parameter error |
| soma `gNa/gK` | 静默、单 spike、重复放电 sweep | margin -> event/count/latency -> alignment -> AP shape | `gNa-gK` landscape、count map、held-out compensation |
| dendritic density | 3--4 个系数生成 apical-CV density，多 probe | positive amplitude/width、bounded transition、spatial prior | density profile、coefficient landscape、跨 CV discretization |
| mixed five sweeps | 无 spike、单 spike、多 spike；记录 stimulus/`dt`/location/windows/noise | 按 sweep 应用组件并固定归一化后求均值 | leave-one-amplitude-out；MSE/masked/soft-DTW/composite 对照 |

## 记录与边界

每个实验保存 data split、`dt`/duration/units、morphology/CV/solver、parameter transform/bounds/
seed、differentiation window、optimizer budget、loss 定义/normalizer/mask、diagnostics 和 traces/
trajectory/landscape artifacts。

| 失败模式 | 处理 |
| --- | --- |
| silent-to-spike 零梯度 | voltage margin + curriculum |
| extra/missing spike | count + unmatched-event，不能只靠 soft-DTW |
| burst | ISI distribution、burst duration、adaptation |
| spike 位于窗口边缘 | clip 索引，禁止 wrap-around |
| 不同 `dt` | 在有单位时间轴上重采样，禁止按 sample index 比较 |
| dropout / NaN | target-fixed validity mask，按有效样本重归一化 |
| parameter 贴界 | 检查 transform saturation、bounds 和缺失机制 |
| non-finite simulation | 不得覆盖最后 finite archive |
| component 初值为零 | normalizer 加 `epsilon` 或禁用无信息项 |
| probe 数不平衡 | per-probe 归一化后聚合 |
| 参数补偿 | 多 protocol、held-out、prior 和 identifiability 分析 |

## References

1. Deistler, M. et al. *Jaxley: differentiable simulation enables large-scale
   training of detailed biophysical models of neural dynamics*. Nature Methods
   22 (2025). [doi](https://doi.org/10.1038/s41592-025-02895-w).
2. Jaxley documentation. [Training tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/07_gradient_descent.html),
   [L5PC example](https://jaxley.readthedocs.io/en/latest/examples/00_l5pc_gradient_descent.html).
3. Cuturi, M. & Blondel, M. *Soft-DTW*. PMLR 70 (2017). [Paper](https://proceedings.mlr.press/v70/cuturi17a.html).
4. van Rossum, M. C. W. *A Novel Spike Distance*. Neural Computation 13 (2001). [doi](https://doi.org/10.1162/089976601300014321).
5. Van Geit, W. et al. *BluePyOpt*. Frontiers in Neuroinformatics 10 (2016). [doi](https://doi.org/10.3389/fninf.2016.00017).
6. Marder, E. & Taylor, A. L. *Multiple models to capture variability*. Nature Neuroscience 14 (2011). [doi](https://doi.org/10.1038/nn.2735).
7. Neftci, E. O., Mostafa, H. & Zenke, F. *Surrogate Gradient Learning in SNNs*. IEEE SPM 36 (2019). [doi](https://doi.org/10.1109/MSP.2019.2931595).
8. Wunderlich, T. C. & Pehle, C. *Event-based backpropagation*. Scientific Reports 11 (2021). [doi](https://doi.org/10.1038/s41598-021-91786-z).
9. Van Geit, W., Achard, P. & De Schutter, E. *Neurofitter*. Frontiers in Neuroinformatics 1 (2007). [doi](https://doi.org/10.3389/neuro.11.001.2007).
10. Druckmann, S. et al. *Multiple objective optimization for conductance-based neurons*. Frontiers in Neuroscience 1 (2007). [doi](https://doi.org/10.3389/neuro.01.1.1.001.2007).
11. Victor, J. D. & Purpura, K. P. *A metric-space analysis*. Journal of Neurophysiology 76 (1996). [doi](https://doi.org/10.1152/jn.1996.76.2.1310).
12. Zenke, F. & Ganguli, S. *SuperSpike*. Neural Computation 30 (2018). [doi](https://doi.org/10.1162/neco_a_01086).
13. Chen, Z. et al. *GradNorm*. PMLR 80 (2018). [Paper](https://proceedings.mlr.press/v80/chen18a.html).
14. Paninski, L. *Maximum likelihood estimation of cascade point-process neural encoding models*. Network 15 (2004). [doi](https://doi.org/10.1088/0954-898X/15/4/002).
