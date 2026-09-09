# Fitting and Identifiability Results

## 来源与测量范围

本页汇集原刺激设计、训练诊断与消融参考文档中的历史实验数据。于 2026-09-07 迁移，
不是当天重新运行。未记载的实际运行日期、软件版本、硬件或 artifact 标为未记录，
不补造。各小节的 starts、模型、预算和成功标准不同，不能跨表直接比较。
原始解释和数字保留；方法原理见 [刺激设计](../../references/stimulus-design-and-identifiability.md)
与 [诊断参考](../../references/modular-training-diagnostics.md)。

复查入口：[参数拟合](../../../../../examples/experimental/optim_parameter_fitting/README.md)、
[刺激设计](../../../../../examples/experimental/optim_stimulus_design/README.md)、
[训练恢复提案](../../proposals/training-recovery.md)。旧配置不一定能由当前默认命令原样重现，
应先核对历史配置；本次未新建原始测量 artifact。

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


## 诊断基线

原诊断记录未完整注明 backend/precision、运行日期和 artifact。
下述 RandomState 是历史初始化来源描述，不是新示例推荐的随机 API。

默认 1-CV HH 基线以 `RandomState(seed=123)` 产生 32 个 initialization starts，并在固定
完整数据上运行 Adam；这些 lane 不是独立随机实验。一次 100-update 运行得到：

| 观测 | 结果 |
| --- | ---: |
| 终点 loss 低于初值 | `32/32` |
| 三个电导 scale 平均相对误差 `<10%` | `10/32` |
| best loss 出现在最终 update 以前 | `19/32` |
| final loss 比 best 高 `>10%` | `5/32` |
| 接近 `[0.1, 2.0]` transform bounds | `0/32` |
| final MSE | 中位数 `4.7627 mV^2`，范围 `0.0923--44.4395 mV^2` |

因此不能用 final loss 或 bound saturation 单独解释失败。至少要区分 spike basin、低梯度
平原、高梯度震荡、慢速移动、spike phase 误差和 `gNa/gK` 补偿。


固定 protocol 下，spike count 在参数区域内是整数常量，跨兴奋性边界时跳变。当前四协议
目标及已观察失败示例为：

| 类型 | Signature |
| --- | --- |
| target | `(1, 2, 3, 4)` |
| low excitability | `(1, 1, 1, 2)` |
| mixed mismatch | `(1, 1, 3, 4)` |
| high excitability | `(2, 3, 4, 5)` |


## 消融的历史参考点

以下来自恢复消融计划，不是已经执行了完整 A-F 消融；正式基线仍须按提案重新测量。

现有 fixed Adam `lr=0.02`、180-update 结果只作为 promotion reference，不是测试常量：

| 指标 | 当前值 |
| --- | ---: |
| trace success | `3/8` |
| parameter success | `4/8` |
| median common loss | `0.235153` |
| median aggregate RMSE | `7.8201 mV` |
| median mean parameter error | `0.1562` |
| best common loss | `0.0153285` |


## Scheduler 的历史缺陷观察

以下只描述特定历史依赖版本；不宣称当前依赖仍有同一错误。正式恢复策略尚未实现，
必须执行 effective-LR 测试，不能将 reported LR 当作实际参数步长。

`braintools 0.1.9` 的 `CosineAnnealingWarmRestarts` 曾出现 reported LR 更新但实际参数增量
仍固定的问题（`base_lr=0.1, T_0=2, eta_min=0.01` 时报告 `0.1, 0.055, ...`，实际 delta
始终 `-0.1`）。正式使用前必须以常梯度回归测试验证 effective LR；临时 controller 只放
在 example 内。


## 解释边界

FIM 满秩、loss 下降、spike signature 正确、held-out 通过和恢复真实参数是不同验收。
本页保存的是有限配置的证据，不是通用 optimizer 排名或全局可辨识性证明。
