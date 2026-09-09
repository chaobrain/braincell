# Parameter Learning Results

## 来源与口径

本页收录 2026-09-07 Channel、Ion、Synapse 学习记录及已执行 Notebook 的教学结果，
本轮文档整理没有重跑。初值、目标与拟合值属于各自实验，不是模型默认参数推荐。
每个示例只学习一个 scalar root；拟合成功不等于任意参数可辨识。

历史来源：[Channel](../../../../specs/2026-09-07-channel-learning.md)、
[Ion](../../../../specs/2026-09-07-ion-learning.md)、
[Synapse/Network](../../../../specs/2026-09-07-synapse-network-learning.md)。
没有新增原始 artifact；环境缺项不从其他实验推断。

## Channel

Python 3.11、JAX 0.8.0、CPU；一 CV、一目标 spike、20 ms waveform、100 次 Adam。
拟合后也保留一个 spike，MSE 单位为 mV squared。精度以 Notebook 环境配置为准，
原历史结果摘要没有独立列出精度值。本次不补作跨精度比较。

| Parameter | Initial | Target | Fitted | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| g_max (mS/cm^2) | 108 | 120 | 119.966 | 55.9342 | 0.0000709192 |
| V_sh (mV) | -44 | -45 | -45.0055 | 258.324 | 0.00785799 |
| temp (K) | 308.15 | 309.15 | 309.145 | 24.2734 | 0.000379125 |

三项拟合当次合计 17.48 s，包含编译，不是预热后的梯度内核耗时。
历史回归 1,051 passed；兼容回归 392 passed、2 skipped；112 类的 375 个 numeric defaults
已检查。改动可执行行覆盖 135/140 = 96.43%，不是整仓库覆盖率或正确概率。
gateCurrent 的固定电压探查前向会变但开关梯度为零；既有 enabled current 幅值未校准，
没有用来生成教学训练目标。GPU 未测。

示例：[channel_learning.ipynb](../../../../../examples/multi_compartment/channel_learning.ipynb)。

## Ion

Python 3.11、JAX 0.8.0、CPU；原记录说明教学与集成检查亦使用默认 float32。
每项一 CV、一个 scalar scale root、800 步、dt=0.025 ms、100 次 Adam、lr=0.03。
前两项训练带 spike 的电压，后三项训练浓度。电压 MSE 用 mV squared，浓度 MSE 用 uM squared。

| Parameter | Initial | Target | Fitted | Initial MSE | Final MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| E (mV) | 45 | 50 | 49.97645 | 14.18368 | 8.10408e-5 |
| temp (K) | 300 | 309.15 | 309.10382 | 4.34521 | 3.91733e-4 |
| Ci_initializer (mM) | 0.0008 | 0.001 | 0.0009999672 | 0.00501708 | 1.32332e-10 |
| tau (ms) | 7 | 5 | 4.998580 | 0.00462772 | 2.85074e-9 |
| kf (1/(mM ms)) | 1.6 | 2 | 2.000439 | 17.27890 | 1.51110e-5 |

五项拟合当次合计 23.75 s，包含编译。历史回归 1,482 passed、2 skipped，加一个字典顺序
回归；23 类、310 个 numeric defaults 已检查。改动可执行行覆盖 209/212 = 98.58%。
未覆盖分支是非均匀 scalar 配置拒绝、默认解析的空间 callback、独立 kinetic 显式 Ci 合并。
历史覆盖率 artifact 位于临时目录，未新增持久报告。
CalciumFirstOrder 原有 alpha/beta 默认单位问题未修复，不包含在成功教学例子中。
原 Ion 记录未测 GPU/其他 JAX；后续 JAX 0.10.1 上下文敏感问题另见网络结果页，不能混成全绿。

示例：[ion_learning.ipynb](../../../../../examples/multi_compartment/ion_learning.ipynb)。

## Synapse 与 Connection

来自单 CV、每例一个 scalar root、每个目标含一个 spike 的既有 Notebook。
以下是 JAX 0.8.0 CPU 已执行结果；JAX 0.10.1 的三个拟合验收也通过。
helper 使用 dt=0.025 ms、240 步/6 ms、100 次 Adam；tau/weight 的 lr=0.03，threshold 的
lr=0.5。三项都是 voltage MSE（mV squared），但前两项用固定 NetStim，threshold 用 autapse，
不是相同输入配置。scale 初值为 0.8；threshold 初始为 -30 mV、目标 -20 mV。

| Target | Initial MSE | Final MSE | 解释 |
| --- | ---: | ---: | --- |
| Synapse tau | 0.0564968 | 2.48711e-7 | 连续突触动力学参数 |
| Connection weight | 3.11666 | 9.70201e-7 | 独立乘权路径 |
| Detector threshold | 2.5993e-5 | 0 | 离散事件网格等价，不是唯一阈值恢复 |

示例：[synapse_learning.ipynb](../../../../../examples/multi_compartment/synapse_learning.ipynb)，
实现：[synapse_learning.py](../../../../../examples/multi_compartment/synapse_learning.py)。
网络梯度、双向联合训练及环境限制见 [网络结果](synapse-network-learning.md)。

## 复查方式

三个 Notebook 均可在选择好依赖环境后用 nbconvert 从新 kernel 执行，旧 spec 中保留原命令。
本次不执行它们，不把文档检查当作新的一轮数值验收。支持范围与错误语义以
[参数支持度](../parameter-support.md) 为准，不由这些成功例子构造白名单。
