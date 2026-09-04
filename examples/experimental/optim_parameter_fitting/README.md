# Parameter Learning 历史实验与训练诊断

## 当前状态

本目录保留 parameter-fitting 训练诊断和 1-CV 组合式实验框架。它不是公共 API，也不再包含
早期完整的九参数训练、dataset、ablation 和 demo 源码。初始化和刺激设计已经拆到独立实验
目录，当前本目录代码为：

```text
diagnostics.py
diagnostics_test.py
config.py / models.py / datasets.py / losses.py
optimizers.py / search.py / training.py / reporting.py / run.py
configs/
```

相关的 DC/Sobol/Nevergrad 实验见 [`optim_initialization`](../optim_initialization/)，PRMLS、OED
和 global ensemble 见 [`optim_stimulus_design`](../optim_stimulus_design/)。

## 当前活跃代码

`diagnostics.py` 提供 experiment-local、optimizer-agnostic 的 multistart 训练诊断：

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
该框架保留在 `optim_parameter_fitting/`，不并入只提供模型无关梯度能力的 `optim/`。

## Hybrid Initialization 实验

`dc_protocol_dataset.py` 构建 3 CV、9 conductance-scale 的 synthetic target。每条 protocol
是完整 50 ms 的恒定 DC；电流按 target response 在三个注入位置独立校准。固定 split 为：

| Split | 每个位置 | 总数 |
| --- | --- | ---: |
| train | `-80/-110 mV`、`0/1/3 spike` | 15 |
| validation | `-90 mV`、`2 spike` | 6 |
| test | `-100 mV`、`4 spike` | 6 |

`../optim_initialization/hybrid_initialization.py` 固定 raw voltage MSE 和 exact RTRL + Adam，只比较
初始化入口：

| Method | Forward screening budget | Adam starts |
| --- | ---: | ---: |
| direct random | 16（不筛选） | 16 |
| random screen | 1024 | 16 |
| scrambled Sobol | 1024 | 16 |
| Nevergrad TwoPointsDE | `64 x 16 = 1024` | 16 |

所有方法在九维 relative physical scale 的 `[0.5, 1.5]` 中工作，并统一在 log-space
生成候选。Nevergrad 是可选依赖：

```bash
pip install -e '.[optim]'
```

先运行单 search seed pilot：

```bash
python examples/experimental/optim_initialization/hybrid_initialization.py run \
  --stage pilot --gpu 7 \
  --python /home/swl/anaconda3/envs/braincell_311/bin/python \
  --resume
```

pilot 通过后运行 search seeds `0/1/2`：

```bash
python examples/experimental/optim_initialization/hybrid_initialization.py run \
  --stage formal --gpu 7 \
  --python /home/swl/anaconda3/envs/braincell_311/bin/python \
  --resume
```

每个 update 保存 loss、参数、raw gradient、gradient norm 和 optimizer delta；每 10 updates
保存逐 protocol MSE、spike count 和 timing。完整 voltage 只为 initial、validation-best 和
final 保存。Test split 从不参与候选排序或 checkpoint 选择。

生成结果位于被 Git 忽略的：

```text
examples/experimental/optim_initialization/artifacts/hybrid_initialization/
```

当前 A100 x64 正式结果使用 search seeds `0/1/2`，每种方法共训练 48 个 starts：

| Method | Trace success | Parameter success | Joint success | Median validation MSE | Median parameter relative RMS |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct random | 14/48 | 0/48 | 0/48 | 33.519 | 0.248 |
| random-1024 | 19/48 | 1/48 | 0/48 | 24.321 | 0.260 |
| Sobol-1024 | 21/48 | 0/48 | 0/48 | 24.097 | 0.251 |
| TwoPointsDE-1024 | 19/48 | 0/48 | 0/48 | 24.216 | 0.248 |

这里的 trace success 要求所有 validation spike counts 正确且 aggregate RMSE 不超过 5 mV；
parameter success 要求九参数 relative RMS 不超过 10%。三种 1024-evaluation screening 都提高了
trace success，Sobol 在本轮最高；但没有方法产生 joint success。说明 initialization screening
已经改善进入可拟合轨迹区域的概率，而 DC-only/raw-MSE 数据仍不足以稳定识别九个 conductance。
下一轮应先分析 observation Jacobian conditioning，再决定增加 protocol 还是改变 loss。

## Stimulus Design v2

`../optim_stimulus_design/dataset.py` 保留旧 morphology 几何，但使用 `1/3/3` CV 和六个参数：soma
Leak/Na/K，以及所有 dend CV 共享的 Leak/Na/K。它生成统一100 ms current tensor：

```text
0--20 ms baseline / 20--80 ms stimulus / 80--100 ms recovery
```

候选输入包括 feature-based Step 和 `2/5/10 ms` 四电平 PRMLS，并与 soma、distal dend_a、
distal dend_b 三个注入位置组合。总计 `train/validation/test = 33/15/12` 条，全部保存7 CV voltage。

`../optim_stimulus_design/robust_oed.py` 在 target 和16个 log-space Sobol parameter points 上用 exact RTRL
在线累计 observation Fisher information，只对33条 train candidates生成 robust D-opt greedy
ordering。它不运行参数优化，也不自动冻结最终 train subset。

```bash
python examples/experimental/optim_stimulus_design/dataset.py
python examples/experimental/optim_stimulus_design/robust_oed.py
```

详细理论、文献和证据边界见
[刺激设计、Persistent Excitation 与参数可辨识性](../../../docs/design/optim/references/stimulus-design-and-identifiability.md)。

当前 A100 x64 dataset/OED 结果：

```text
target dataset: current (60, 4000, 3), voltage (60, 4000, 7)
PRMLS: all locations subthreshold, global A = 0.05931144 nA
OED references: target + 16 Sobol prior points
OED candidates: 33/33 finite
compile: 24.92 s
FIM execution: 0.78 s
XLA temporary: 2.99 MB
```

Greedy prefix 第一条在相对 rank 阈值下已经达到 rank 6，但 worst condition 仍为 `1.36e7`，
全部33条后仍约 `1.89e7`，worst parameter-column correlation 约 `0.99997`。因此这些 stimuli
数值上能看到六个方向，却仍存在很强的 practical compensation；不能因为“满 rank”就直接冻结
最小训练集。当前 ordering 前十项由1-spike、mild-negative、3-spike三个位置和 dend_b
small-positive 构成，PRMLS 从第11项开始进入。最终 subset 需先审阅完整 conditioning curve。

`../optim_stimulus_design/global_ensemble.py` 在六维 log-scale parameter box 中评估16384个 forward-only Sobol
candidates，同时保存全60条 protocol MSE 和 hard spike count。它使用两种 train score：

```text
raw MSE
prior-median protocol-normalized MSE
```

```bash
python examples/experimental/optim_stimulus_design/global_ensemble.py
```

当前 A100 x64 结果：

```text
candidates: 16384
compile: 21.84 s
ensemble/prior evaluation: 83.86 s
weak/strong/plane profiles: 7.16 s
XLA temporary: 875.73 MB
raw/normalized Top256 overlap: 130, Jaccard 0.340
raw Top PC1 vs FIM weak cosine: 0.871
normalized Top PC1 vs FIM weak cosine: 0.790
```

Raw Top256 的 validation/test spike-signature exact 数为 `12/51`，normalized Top256 为 `4/29`；
低 voltage score 不自动保证所有 hard spike counts 正确。两种 Top256 的 parameter relative RMS
中位数仍约 `0.25`，说明当前最好的稀疏全局 samples 远离 synthetic target。Target-centered
weak-direction profile 则显示 target 附近存在明显浅方向，因此16384点 Sobol ensemble主要用于
验证几何趋势，不能冒充完整posterior或证明已经找到全部 equivalent solutions。

### Python 组合式零号基线

本目录是当前固定实验入口，不是 BrainCell 公共 API。Python preset 直接组合真实组件，
不通过 JSON 字符串 registry：

```python
CONFIG = ExperimentConfig(
    model=hh_1cv_classic_bounded_direct(),
    dataset=feature_step_dataset(),
    loss=raw_voltage_mse(),
    initialization=InitializationConfig(seed=0, num_candidates=64),
    stages=(adam(epochs=180, learning_rate=0.01, gradient_method="rtrl"),),
)
```

组件职责：

| 模块 | 可替换内容 | 固定合同 |
| --- | --- | --- |
| `models.py` | morphology、channel、parameter space | physical / bounded `u` / optimizer `z` 往返 |
| `datasets.py` | waveform、split、target generation | train/validation/test metadata |
| `losses.py` | local/trajectory objective | 训练与split评价同义 |
| `optimizers.py` | Adam等gradient stage | optimizer只接收`z` gradient |
| `search.py` | forward-only/non-gradient stage | 通过physical `CandidateSet`交接 |
| `training.py` | stage pipeline、RTRL/BPTT | 不隐式拆candidate |
| `reporting.py` | success、CSV/JSON、plots | test final-only |
| `run.py` | worker、resume、GPU monitor | 一个固定CLI |

增加普通配置字段时在dataclass中给默认值；行为不兼容时增加版本化factory。非梯度stage只使用train
objective，DFO后重新进入gradient stage时从physical参数反解`z`并重建optimizer moments。可以组合：

```python
stages = (sobol_screen(...), adam(...))
stages = (adam(...), differential_evolution(...))
stages = (adam(...), local_search(...), adam(...))
```

当前只正式运行plain Adam；`ForwardSelectionStage`用于验证stage交接合同，未冒充实际搜索算法。

零号模型是一个25 um soma CV，直接学习有界的经典HH conductance：

| Parameter | Target | Bounds | Runtime mapping |
| --- | ---: | ---: | --- |
| `leak.g_max` | `0.3` | `[0.15,0.45] mS/cm^2` | direct parameter |
| `na.g_max` | `120` | `[60,180] mS/cm^2` | direct parameter |
| `k.g_max` | `36` | `[18,54] mS/cm^2` | direct parameter |

Cell runtime使用`theta=lower+(upper-lower)*sigmoid(z)`，不存在frozen baseline scale。数据全部为
`0--20/20--80/80--100 ms`的DC Step，split为`5/2/1`；PRMLS不进入零号实验。64个start由一个
seed生成并一次进入同一个RTRL kernel和Adam optimizer，不拆cohort。Test只在epoch 180评价。

```bash
python examples/experimental/optim_parameter_fitting/run.py run \
  --config examples/experimental/optim_parameter_fitting/configs/basic_1cv_bounded_direct_adam.py \
  --gpu 1 --resume
```

当前A100 x64正式结果：

| 指标 | 结果 |
| --- | ---: |
| trace success | `8/64 = 12.5%` |
| trace-success Wilson 95% CI | `[6.47%,22.77%]` |
| parameter / joint success | `3/64 / 1/64` |
| median train MSE | `61.2255 mV^2` |
| median validation / test RMSE | `10.6116 / 17.2393 mV` |
| median parameter relative RMS | `0.2260` |
| compile / 180-update stage / end-to-end | `2.85 / 48.88 / 82.50 s` |
| XLA temporary / monitored peak GPU memory | `2.10 MiB / 1166 MiB` |

64个endpoint全部finite；validation count全对`22/64`，RMSE不超过5 mV为`8/64`，交集为
`8/64`。Test count全对`35/64`，同时满足5 mV与count为`5/64`。结果位于被Git忽略的：

```text
examples/experimental/optim_parameter_fitting/artifacts/parameter_experiments/
  20260831-210204_adam-narrow-e180_db936968/
```

旧7CV/6-scale/四cohort且holdout混入PRMLS的`6/64`结果保留为legacy，不再作为零号reference。

#### Epoch预算：180 vs 300

`basic_1cv_bounded_direct_adam_e300.py`使用`dataclasses.replace()`派生，resolved config唯一变化为
`stages[0].epochs: 180 -> 300`。初始physical/normalized/`z`逐位相同，前180轮optimizer `z`、
physical参数和gradient逐位相同；epoch-180 validation完全相同。

| 指标 | Epoch 180 | Epoch 300 |
| --- | ---: | ---: |
| trace success | `8/64` | `10/64` |
| parameter success | `3/64` | `6/64` |
| joint success | `1/64` | `1/64` |
| median train MSE | `61.2255` | `55.3886 mV^2` |
| median validation RMSE | `10.6116` | `10.1949 mV` |
| median test RMSE | `17.2393` | `17.1670 mV` |
| end-to-end | `82.50` | `112.95 s` |

同start配对显示trace success为`6`个保持、`4`个新增、`2`个丢失；parameter success新增3个且无
丢失；joint success总数不变，但从start 37迁移到start 59。继续训练总体降低loss，却不保证每条
lane的held-out成功单调保持。正式比较artifact位于300轮结果的
`comparisons/20260831-210204_adam-narrow-e180_db936968/`。

#### LR 0.02 + Wide Bounds

在300轮配置上同时把Adam LR从`0.01`改为`0.02`，把transform bounds从`0.5--1.5`扩为
`0.1--2.0 x target`；64个physical初值与`c00763f960b0`逐位相同，仍位于`0.5--1.5 x target`。

| 指标 | 300轮基准 | LR 0.02 + wide bounds |
| --- | ---: | ---: |
| trace success | `10/64` | `5/64` |
| parameter success | `6/64` | `14/64` |
| joint success | `1/64` | `1/64` |
| median train MSE | `55.3886` | `49.1545 mV^2` |
| median validation RMSE | `10.1949` | `8.7019 mV` |
| median test RMSE | `17.1670` | `16.4772 mV` |
| median parameter relative RMS | `0.2261` | `0.1706` |

全部final参数均离新bounds至少10%；parameter success迁移为`4保持/10新增/2丢失`，trace为
`2保持/3新增/8丢失`。`3-spike` train protocol的count exact只有`1/64`，说明组合设置改善连续
误差和parameter recovery，却更容易错过目标spike region。由于LR与bounds同时改变，不能把结果
单独归因给其中一项。结果digest为`682d461a9318`。

为拆开两个变量，另运行wide bounds但保持`lr=0.01`，digest为`3692d0969bca`：

| 指标 | 原bounds LR 0.01 | Wide LR 0.01 | Wide LR 0.02 |
| --- | ---: | ---: | ---: |
| trace success | `10/64` | `9/64` | `5/64` |
| parameter success | `6/64` | `8/64` | `14/64` |
| joint success | `1/64` | `3/64` | `1/64` |
| validation RMSE | `10.1949` | `8.3472` | `8.7019 mV` |
| test RMSE | `17.1670` | `16.8468` | `16.4772 mV` |
| parameter relative RMS | `0.2261` | `0.1956` | `0.1706` |

Bounds-only使42/64 validation MSE和45/64 parameter error改善，joint success增至3/64；trace总数
只减少1，但集合为`3保持/6新增/7丢失`。在相同wide bounds下把LR升到0.02，trace从9降至5、joint
从3降至1，而per-start train MSE median delta仅`+0.063 mV^2`。因此wide bounds总体有益于连续
fit/parameter recovery；高LR进一步推动参数恢复，却明显损害spike-region稳定性。

#### Rprop Optimizer

在wide bounds、300轮和完全相同初值下，只把Adam替换为
`Rprop(lr=0.01, etas=(0.5,1.2), step_sizes=(1e-6,50))`：

| 指标 | Adam | Rprop |
| --- | ---: | ---: |
| trace success | `9/64` | `35/64` |
| parameter success | `8/64` | `13/64` |
| joint success | `3/64` | `11/64` |
| validation RMSE | `8.3472` | `4.4684 mV` |
| test RMSE | `16.8468` | `9.4214 mV` |
| stage time | `78.06` | `89.53 s` |

Optimizer-only配对中，Rprop新增28个trace成功、丢失2个；49/64 validation MSE和47/64 test
MSE改善。Validation-feasible archive进一步得到validation trace `38/64`、test trace `12/64`，
而Adam archive为`22/64`与`7/64`。Rprop对当前deterministic low-dimensional问题明显优于Adam，
且多数收益已保留到final endpoint。结果digest为`8ae34281f8a6`。

BrainTools Rprop wrapper会在`scale_by_rprop(base_lr)`后再次应用同一LR scheduler，因此名义
`lr=0.01`的实际initial/min step约为`1e-4/1e-8`。使用custom Optax tx只应用一次LR，并保持
initial step=`1e-4`、min step=`1e-6`后：

| 指标 | Wrapper Rprop | Single-scale Rprop |
| --- | ---: | ---: |
| trace / parameter / joint | `35/13/11` | `36/11/10` |
| validation / test RMSE | `4.4684/9.4214` | `4.5119/9.4196 mV` |
| archive validation / test trace | `38/12` | `39/12` |
| K最后50轮median `abs(delta z)` | `1.5e-8` | `1.2e-6` |

解除`1e-8`冻结使K的physical movement提高约80倍，但final/held-out只小幅变化；约24%的gradient
sign flip仍会让Rprop在反号轮执行零更新。因此wrapper重复LR是实际bug，但不是Rprop优于Adam或
后期停滞的唯一原因。Single-scale结果digest为`81eff6e7aa2e`。

#### Protocol-Balanced MSE

在single-scale Rprop上只把protocol权重改为inverse target-voltage std（`5 mV` floor），final
trace/parameter/joint从`36/11/10`提高到`43/15/12`，validation/test RMSE从`4.5119/9.4196`
降至`3.8424/9.1728 mV`。配对中trace `34保持/9新增/2丢失`，47/64 validation、42/64 test、
46/64 parameter error改善。Validation archive得到validation trace `47/64`，但test trace为
`9/64`，低于raw-loss archive的`12/64`；因此balanced loss改善总体与validation拟合，但严格test
成功没有同比提高。结果digest为`cdc6f22c0f24`。

#### Balanced Huber Delta 5 mV

从balanced MSE只替换逐点惩罚为MSE-normalized Huber，`abs(error)<=5 mV`时与MSE完全一致，
大误差进入线性尾部：

| 指标 | Balanced MSE | Balanced Huber |
| --- | ---: | ---: |
| final trace / parameter / joint | `43/15/12` | `38/17/15` |
| validation / test RMSE | `3.8424/9.1728` | `3.3781/9.1748 mV` |
| archive validation / test trace | `47/9` | `39/13` |
| archive median test RMSE | `8.9297` | `9.0319 mV` |

Huber减少final和archive validation trace覆盖，但parameter、joint以及严格archive test success提高。
它不是全面降低所有raw MSE：`3-spike` median MSE从`13.995`降到`8.930`，但small-positive从
`7.217`升到`96.045`，说明线性尾部允许部分lane牺牲subthreshold轨迹以改善spiking/parameter
方向。结果目录为：

```text
20260901-152320_balanced-huber-d5_3ee11164/
```

`checkpoint_every=10`是历史命名，实际只表示每10轮执行一次validation forward；不保存optimizer
checkpoint，也不改变trajectory。新配置可写`validation_every=10`，旧名称继续兼容。所有完整run
可通过`plot/parameter_experiments/runs.csv`查询。

#### Vanilla SGD

在balanced Huber配置上只替换为vanilla SGD `lr=1e-4`，不使用momentum、Nesterov、weight decay、
clip或scheduler。结果显著差于single-scale Rprop：

| 指标 | Rprop | SGD |
| --- | ---: | ---: |
| final trace / parameter / joint | `38/17/15` | `0/2/0` |
| validation / test RMSE | `3.3781/9.1748` | `15.1178/21.5266 mV` |
| archive validation / test trace | `39/13` | `0/3` |
| median final objective | `6.5554` | `21.1214 mV^2` |

SGD没有发散或贴边，gradient norm中位数从`22.7`降到`5.38`，但300轮内没有形成正确validation
spike signature。它稳定降低Huber objective，却沿错误连续basin缓慢移动；Rprop同号gradient时指数
扩大step的机制对跨入目标spike region更有效。结果目录为：

```text
20260902-143337_sgd-lr1e-4-balanced-huber_94649c2f/
```

#### Momentum 与 Nesterov SGD

在相同`lr=1e-4`下并行测试`momentum=0.9`和Nesterov。两者明显加速Huber objective下降，但没有
恢复final trace success：

| 指标 | Vanilla | Momentum | Nesterov | Rprop |
| --- | ---: | ---: | ---: | ---: |
| final trace | `0/64` | `0/64` | `0/64` | `38/64` |
| parameter | `2/64` | `8/64` | `11/64` | `17/64` |
| joint | `0/64` | `0/64` | `0/64` | `15/64` |
| validation RMSE | `15.12` | `10.78` | `12.54` | `3.38 mV` |
| test RMSE | `21.53` | `18.87` | `19.94` | `9.17 mV` |
| final Huber objective | `21.12` | `10.69` | `11.42` | `6.56` |
| archive validation/test trace | `0/3` | `6/2` | `5/6` | `39/13` |

Momentum/Nesterov解决了vanilla移动慢的问题，但5/7条lane最终靠近transform边缘，且仍落在错误
spike basin；Nesterov没有稳定优于普通Momentum。这进一步说明Rprop的per-coordinate sign-step
机制比统一velocity累积更适合当前任务。两个并行run分别为：

```text
20260902-145308_momentum09-lr1e-4-balanced-huber_1404e6e8/
20260902-145308_nesterov09-lr1e-4-balanced-huber_2025b7f6/
```

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

- [模块化训练诊断与优化恢复](../../../docs/design/optim/references/modular-training-diagnostics.md)：当前 `diagnostics.py` 的角色、history alignment、双 archive、spike-region 和恢复策略。
- [电压轨迹与 Spike-Aware 参数训练](../../../docs/design/optim/references/voltage-and-spike-parameter-fitting.md)：subthreshold/spike loss、mask、curriculum 和历史实验索引。
- [Optimization Design Overview](../../../docs/design/optim/design-overview.md)：公共 `braincell.trainable` 与实验训练代码的边界。
- [Experimental Optimization Work](../README.md)：exact forward sensitivity、RTRL/BPTT、正确性与 scaling 实验导航。

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
pytest -q examples/experimental/optim_parameter_fitting/diagnostics_test.py
pytest -q examples/multi_compartment/trainable_hh_multistart_test.py
```

如果未来该 helper 有多个稳定调用方并形成明确公共合同，再决定是否移动到正式模块；本 README
不承诺当前位置或类型名长期稳定。
