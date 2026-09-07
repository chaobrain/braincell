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

## 历史实验结果

零号训练基线、优化器对照、六参数 FIM 和 global ensemble 的完整数字已迁至
[拟合与可辨识性结果](../current/results/fitting-and-identifiability.md)。下面保留方法定义与协议，
这些定义不等于公共训练 API。

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
