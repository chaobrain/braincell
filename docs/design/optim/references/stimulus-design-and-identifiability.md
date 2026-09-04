# 刺激设计、Persistent Excitation 与参数可辨识性

## 文档定位

本文是 BrainCell 参数拟合的 research reference，讨论如何选择 current-clamp stimulus，使
voltage observation 提供互补参数信息；不定义 Dataset、Trainer、loss 或 optimizer API。
实验实现位于：

```text
examples/experimental/optim_stimulus_design/dataset.py
examples/experimental/optim_stimulus_design/robust_oed.py
```

## 问题模型

均匀扫描 current amplitude 不等于均匀覆盖参数信息。刺激设计必须同时控制：

| 维度 | 例子 | 主要作用 |
| --- | --- | --- |
| waveform | step、PRMLS、sine、noise | temporal spectrum 与 transition |
| operating regime | hyperpolarized、subthreshold、near-rheobase、spiking | 激活不同 nonlinear state |
| location | soma、proximal/distal dendrite | cable transfer 与 regional conductance |

small-signal linearization 下可写为 `delta V = h * delta I`，此时 impulse、multisine、chirp 或
PRBS 近似估计同一 transfer function；完整 HH dynamics 则满足
`response(I1+I2) != response(I1)+response(I2)`。因此输入“基”只表示 local sensitivity
basis，不是可任意叠加的 waveform basis。nonlinear amplitude excitation 更适合考虑 PRMLS [1]。

| 概念 | 问题 | 当前风险 |
| --- | --- | --- |
| Structural identifiability | 理想无噪声、无限精度下参数是否理论唯一 | gating 与 maximal conductance 可能只能识别组合 [2,3] |
| Practical identifiability | 有限时长、刺激、噪声下 uncertainty 是否足够小 | soma/dend、Na/K compensation 和 phase-sensitive minima |

当前 synthetic task 只学习 kinetics 已知的六个 maximal-conductance scales，虽然比未知 kinetics
简单，但 current-clamp aggregate voltage 仍不保证参数唯一。

## 研究证据压缩表

| 研究 | 方法 | 主要结果 | 对 BrainCell 的直接启示 | 局限 |
| --- | --- | --- | --- | --- |
| Pant 2018 [11] | HH 的 time-resolved voltage sensitivity、information gain、sampling frequency | AP 区域含大量 active-conductance 信息；低采样率损害 `gNa`；相关性依 protocol 而变 | 保留 upstroke/peak/repolarization/AHP 分辨率；按时间窗检查 sensitivity | local/reference-dependent |
| Foster 1993 [12] | 搜索大量满足行为 tolerance 的 acceptable parameter sets | firing behavior 可由宽但有界的补偿区域支持；增加 AP height/timing 才缩小区域 | 保存所有 low-loss starts，分析 covariance/manifold，而非只看 best | 不是 posterior |
| Daly 2018 [13] | FIM/SVD、inverse sensitivity、MCMC/ABC | rank deficiency 对应无约束方向；ill-conditioning 对应 elongated/curved region | FIM 后仍需 ensemble/posterior 验证 global compensation | Bayesian 成本更高 |
| Prinz 2004 [14] | 大规模 circuit parameter database | 差异很大的机制参数可产生相似 network activity | functional success 不要求机制唯一 | 网络模型，不是当前单细胞任务 |
| Migliore 2018 [15] | 详细 morphology 下的 conductance ensemble | 更多 morphology/CV/feature 仍不能自动消除 regional correlation | 将输出重要性与参数唯一性分开报告 | 模型与协议特定 |

方法在当前实验中的对应关系：

| 证据层 | 输出 | 当前工具 | 回答的问题 |
| ---: | --- | --- | --- |
| 1 | time/protocol sensitivity | exact RTRL observation sensitivity | 哪段数据约束哪个参数？ |
| 2 | rank/eigenvalue/condition | per-protocol FIM、robust OED | 最弱局部方向是什么？ |
| 3 | acceptable set/covariance | Sobol/DE、multi-start endpoints | 全局有多少低-loss 解？ |
| 4 | held-out behavior | frozen validation/test protocols | 训练等价解是否泛化？ |
| 5 | posterior uncertainty | 后续 MCMC/SMC/manifold method | 噪声下还剩多少不确定性？ |

Sensitivity/FIM 不能替代实际训练。完整证据链必须是 `design -> fit -> held-out validation`；
即使提高 `lambda_min(F)`，spike boundary、Adam coordinate、预算和 loss weighting 仍可能使
success rate 不改善 [9,10,16,17]。

## 零号训练基线

| 类别 | 固定合同 |
| --- | --- |
| 模型/参数 | 1 soma CV；三个 bounded direct `g_max` |
| target | classical HH `(Leak,Na,K)=(0.3,120,36) mS/cm^2` |
| parameterization | `theta=lower+(upper-lower)*sigmoid(z)`；无frozen scale |
| data | Step-only train/validation/test=`5/2/1`；test final-only |
| loss | protocol/time/CV 等权 raw voltage MSE |
| optimizer | exact RTRL + Adam `lr=0.01`；无 clip/schedule/screening/early stopping |
| starts | 一个seed生成64个physical starts；一次进入同一个kernel和optimizer |
| budget | 180 full-batch epochs，每 epoch 对5条train protocols更新一次 |
| primary success | epoch-180 validation RMSE `<=5 mV` 且每条 validation spike count 正确 |
| secondary | 三参数 relative RMS `<=10%` 与 joint success |

Validation每10轮记录但不改变trajectory；test只在最终状态评价。A100 x64基线为：

| 指标 | 结果 |
| --- | ---: |
| trace success | `8/64 = 12.5%` |
| Wilson 95% interval | `[6.47%,22.77%]` |
| parameter / joint success | `3/64 / 1/64` |
| median train MSE | `61.2255 mV^2` |
| median validation / test RMSE | `10.6116 / 17.2393 mV` |
| median parameter relative RMS | `0.2260` |
| compile / stage / end-to-end | `2.85 / 48.88 / 82.50 s` |
| XLA temporary / monitored GPU peak | `2.10 MiB / 1166 MiB` |

64个endpoint均finite；validation count全对`22/64`，RMSE通过`8/64`，交集`8/64`；test
count全对`35/64`，同时通过5 mV与count为`5/64`。train MSE中位数从`178.5024`降至
`61.2255 mV^2`。后续改动复用相同initial candidates与预算，不能只展示更好的best case。

Python stage pipeline以physical `CandidateSet`在方法间交接：gradient stage在`z`空间工作，
derivative-free stage在bounded normalized coordinate工作；非梯度改变参数后重建Adam moments。
第一阶段依次单独改变dataset、loss、initialization/search和optimizer。单变量有效后使用
`baseline / A / B / A+B`：

```text
interaction = improvement(A+B) - improvement(A) - improvement(B)
```

同时报告 paired per-start transition、连续 RMSE、Wilson interval、parameter error、wall time 和
额外 forward budget；除初始化研究外复用同一64 starts。旧7CV/6-scale、四cohort且holdout混入
PRMLS的`6/64`结果保留为legacy，不与本基线直接比较。

只把Adam预算从180延长到300轮后，trace success从`8/64`增至`10/64`、parameter success从
`3/64`增至`6/64`，joint仍为`1/64`；validation/test RMSE中位数分别从`10.6116/17.2393`
变为`10.1949/17.1670 mV`。配对迁移为trace `6保持/4新增/2丢失`，joint则丢失原start 37并
新增start 59。因而增加budget有小幅总体收益，但不能当作lane-wise monotonic recovery；后续方法
仍需保存best archive并报告固定epoch endpoint。

300轮下同时使用Adam `lr=0.02`与`0.1--2.0 x target`宽bounds，并保持64个physical初值逐位
不变，parameter success从`6/64`增至`14/64`，validation/test RMSE中位数降至
`8.7019/16.4772 mV`；但trace success从`10/64`降至`5/64`。全部endpoint远离新bounds，
而`3-spike` train count exact仅`1/64`。该双变量组合改善continuous fit和parameter recovery，
却损害spike-region保持；不能据此区分收益来自宽bounds还是高LR。

补充bounds-only对照后，`lr=0.01`的wide bounds得到trace/parameter/joint=`9/8/3`，validation/test
RMSE=`8.3472/16.8468 mV`。在相同wide bounds下将LR升到0.02后变为`5/14/1`。因此宽bounds
主要改善continuous fit和parameter recovery；高LR会进一步增加parameter success，但降低trace与
joint success，表现为更不稳定的spike-region跨越。

只替换optimizer为Rprop后，final trace/parameter/joint从Adam的`9/8/3`提高到`35/13/11`，
validation/test RMSE从`8.3472/16.8468`降至`4.4684/9.4214 mV`。Validation-feasible archive
得到validation/test trace success=`38/12`，高于Adam的`22/7`。这支持按gradient符号反转自适应
缩步比固定moment-based Adam更适合当前deterministic spike-region landscape。

BrainTools wrapper把Rprop LR应用两次，使名义`0.01`成为实际initial step `1e-4`，min step成为
`1e-8`。Single-scale Optax Rprop保持initial `1e-4`但恢复min `1e-6`后，K后50轮median step提高
约80倍；final trace/parameter/joint为`36/11/10`，archive validation/test trace为`39/12`，与
wrapper的`35/13/11`和`38/12`接近。重复LR是实现bug且解释了冻结量级，但解除它没有产生额外的
整体性能跃升，sign-flip零更新仍是后期停滞的主要机制。

使用target-std protocol-balanced MSE后，final trace/parameter/joint提高到`43/15/12`，validation/
test RMSE为`3.8424/9.1728 mV`。Validation archive trace达到`47/64`，但对应test trace为`9/64`，
低于raw-loss archive的`12/64`。权重平衡改善了多数连续指标和validation basin覆盖，但不能替代
phase-robust loss来保证unseen high-spike protocol泛化。

将balanced MSE替换为`delta=5 mV`的MSE-normalized Huber后，final trace/parameter/joint为
`38/17/15`，archive validation/test trace为`39/13`。相比balanced MSE的`43/15/12`与`47/9`，
Huber牺牲部分validation覆盖，却提高parameter/joint和严格test success，符合其降低大spike-phase
残差主导性的设计目的。它同时把`3-spike` median MSE从`13.995`降到`8.930`，但small-positive
从`7.217`升到`96.045`，不是所有regime同时改善；下一步需避免让线性尾部忽略subthreshold错误。

Balanced Huber下的vanilla SGD `lr=1e-4`得到final trace/parameter/joint=`0/2/0`，validation/test
RMSE=`15.1178/21.5266 mV`，显著弱于Rprop。SGD没有贴边或数值发散，但300轮没有形成正确
validation spike signature；固定幅值gradient descent会稳定下降Huber objective，却缺少Rprop在
连续同号阶段快速增大per-coordinate step、跨入目标spike basin的能力。

加入`momentum=0.9`或Nesterov后，final Huber objective中位数从vanilla SGD的`21.12`降到
`10.69/11.42`，parameter success提高到`8/11`；但两者final trace/joint仍为0，archive
validation/test trace仅`6/2`与`5/6`。Momentum提高移动速度却不能替代Rprop的per-coordinate
sign adaptation；Nesterov在此任务上也没有稳定优于普通Momentum。

## 六参数 Identifiability 结果

### Local / Sampled-Prior FIM

target + 16 Sobol references、33 条 train candidates 得到：

| 诊断 | 结果 | 解释 |
| --- | ---: | --- |
| relative numerical rank | `6` | 六个 log-conductance 方向均非零可见 |
| worst condition number | 约 `1e7` | 最强/最弱 sensitivity 相差数千倍 |
| worst column correlation | 约 `0.99997` | 严重 regional/Na-K compensation |

最弱 eigenvector 主要为 soma conductance 减小、dend conductance 增大。满秩不表示 practical
identifiability 良好；加入 33 条 protocol 后仍高度病态。

### Forward-Only Global Ensemble

在 `[log(0.5), log(1.5)]^6` 中评估 16,384 个 scrambled Sobol 参数点，只运行 forward，
保存 per-protocol voltage MSE/hard count 与 raw、normalized train/validation/test score。
normalizer 来自 target + 16 fixed Sobol prior points 的 per-protocol MSE median，下限
`1 mV^2`，不随 candidate 或 optimizer 改变。

| 比较 | 结果 |
| --- | ---: |
| raw/normalized Top256 intersection | `130` |
| Top256 Jaccard | `0.340` |
| raw Top256 PCA 与 target-FIM weakest direction cosine | `0.871` |
| normalized Top256 cosine | `0.790` |
| raw / normalized Top256 median parameter relative RMS | `0.258 / 0.246` |

结果验证了 FIM 的 soma/dend compensation direction 会影响全局 candidate ordering，也说明 loss
weighting 会改变“好解”集合。但 16,384 点在六维仍很稀疏，这不是 posterior 或完整 uncertainty
quantification，只是 low-loss candidate pool 和 local weak direction 的非梯度验证。

当前评价顺序是：loss 能否下降，unseen voltage/spike 是否泛化，最后再报告 parameter recovery
或 equivalent-model ensemble；参数不接近 synthetic target 不自动等于 functional failure。

## Observation Sensitivity 与 OED

令无量纲参数为 `phi = log(theta/theta_target)`，对 protocol `p`：

```text
J_p[t, cv, k] = partial V_p[t, cv] / partial phi[k]
F_p = mean_(t,cv)(J_p.T @ J_p)
F(S) = sum_(p in S) F_p
```

| 指标 | 含义 |
| --- | --- |
| `rank(F)` | 局部可见方向数 |
| `lambda_min(F)` | 最弱方向信息量；E-optimal |
| `condition(F)` | compensation 严重度 |
| normalized off-diagonal | sensitivity-column correlation |
| `logdet(F+epsilon I)` | 总 uncertainty volume；D-optimal |
| `trace(F^-1)` | 平均 variance；A-optimal |

只在 target 计算会过度局部化，因此使用：

```text
target + 16 scrambled Sobol points in [log(0.5), log(1.5)]^6
= 17 reference points

greedy score = min_reference logdet(F(prefix + candidate) + 1e-8 I)
```

这是 sampled-prior robust D-optimal heuristic，输出 deterministic ordering 与每个 prefix 的
worst-reference rank/condition。它不证明连续参数盒全局 identifiable，也不自动决定 protocol 数；
ordering 和 conditioning curve 必须人工审阅后冻结。

Exact RTRL 每步传播 sensitivity，但 OED 只累计
`FIM[protocol, parameter, parameter]`，不保存
`sensitivity[time, parameter, protocol, CV]`。artifact 保存 per-protocol FIM、prior scales、
ordering 和 prefix spectrum，大小不随完整 time history 增长。

## Waveform 与空间合同

| Family | 当前配置 | 信息作用 | 本轮决策 |
| --- | --- | --- | --- |
| Feature Step | `0--20 ms` baseline、`20--80` stimulus、`80--100` recovery | passive、threshold、f-I、ISI | train candidate |
| PRMLS | levels `{-A,-A/3,+A/3,+A}`；clock `2/5/10 ms` | 多幅度与多 transition；不同 split 用不同 seed | train candidate |
| sine/chirp | 多周期、频率受时长约束 | frequency generalization | held-out 候选 |
| frozen colored noise | broadband excitation | richer spectrum，但 timing mismatch 难处理 [7] | 本轮不加入 |

PRMLS 对三个位置使用相同全局 amplitude `A`，并缩小到所有 target trace 均 subthreshold；
10-ms clock 在 60-ms window 只有 6 symbols，应称 slow random square pattern。极端负电流不是
覆盖目标，因为当前模型没有 Ih 或 T-type Ca。

| Region | CV | Conductance target `(Leak, Na, K) mS/cm^2` |
| --- | ---: | --- |
| soma | 1 | `(0.60, 120, 36)` |
| dend_a | 3 | shared dend target |
| dend_b | 3 | shared dend target |
| all dend | 6 | `(0.45, 90, 27)` |

每条 location-independent waveform 与 soma midpoint、distal dend_a、distal dend_b 做 Cartesian
product。禁止按位置校准到相同 soma response，因为差异正是 cable-transfer 信息。全部 7 CV
voltage 进入 observation，当前对 time/CV 等权；area/region weighting 必须单独消融。

## 数据隔离与检查

| 阶段 | 必须检查 |
| --- | --- |
| 生成前 | 参数自由度、waveform/regime/location 覆盖、AP 时间分辨率；先冻结 split |
| OED | 只读取 33 条 train candidates；validation/test PRMLS 使用 unseen seeds |
| 训练前 | per-protocol FIM 的 rank、`lambda_min`、condition、correlation、weak direction |
| 训练后 | voltage/parameter/held-out、全部 low-loss starts、PCA/eigenvector projection |

validation/test 不参与 amplitude、FIM ordering 或 prefix count。人工看过并据此修改设计的
diagnostic 已属于开发数据，不能继续作为 final test。

## References

1. Toker, O. *Pseudo-random multilevel sequences*. IMA J. Math. Control 21 (2004). [doi](https://doi.org/10.1093/imamci/21.2.183)
2. Walch, O. J. & Eisenberg, M. C. *Identifiable combinations in generalized HH models*. Neurocomputing 199 (2016). [doi](https://doi.org/10.1016/j.neucom.2016.03.027)
3. Csercsik, D. et al. *Identifiability of a single HH-type channel*. Neurocomputing 77 (2012). [doi](https://doi.org/10.1016/j.neucom.2011.09.006)
4. Meliza, C. D. et al. *Estimation of neuron parameters from imperfect observations*. PLoS CB 16 (2020). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7386621/)
5. De Cock, A. et al. *D-optimal input design for nonlinear FIR systems*. Automatica 73 (2016). [doi](https://doi.org/10.1016/j.automatica.2016.04.052)
6. Maidens, J. N. et al. *Input Design via Convex Relaxation* (2010). [arXiv](https://arxiv.org/abs/1009.5614)
7. Brookings, T. et al. *Parameter estimation of multicompartmental neuron models*. J. Neurophysiology 112 (2014). [doi](https://doi.org/10.1152/jn.00007.2014)
8. Jauberthie, C. et al. *Input design for persistency of excitation*. IFAC 35 (2002). [doi](https://doi.org/10.3182/20020721-6-ES-1901.00434)
9. Lei, C. L. et al. *Model-driven OED for cardiac electrophysiology*. CMPB 240 (2023). [doi](https://doi.org/10.1016/j.cmpb.2023.107690)
10. Beattie, K. A. et al. *Sinusoidal protocols for ion-channel kinetics*. J. Physiology 596 (2018). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978315/)
11. Pant, S. *Information sensitivity functions*. JRS Interface 15 (2018). [doi](https://doi.org/10.1098/rsif.2017.0871)
12. Foster, W. R. et al. *Significance of conductances in HH models*. J. Neurophysiology 70 (1993). [doi](https://doi.org/10.1152/jn.1993.70.6.2502)
13. Daly, A. C. et al. *Inference-based parameter identifiability*. JRS Interface 15 (2018). [doi](https://doi.org/10.1098/rsif.2018.0318)
14. Prinz, A. A. et al. *Similar network activity from disparate parameters*. Nature Neuroscience 7 (2004). [doi](https://doi.org/10.1038/nn1352)
15. Migliore, R. et al. *Physiological variability of channel density*. PLoS CB 14 (2018). [doi](https://doi.org/10.1371/journal.pcbi.1006423)
16. Banks, H. T. et al. *Comparison of optimal design methods*. Inverse Problems 27 (2011). [doi](https://doi.org/10.1088/0266-5611/27/7/075002)
17. Clerx, M. et al. *Four ways to fit an ion channel model*. JGP 151 (2019). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6990153/)
