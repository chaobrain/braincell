# 仓库组织指南

状态：目标布局，目录迁移待确认。

仓库按内容的用途组织：核心实现、使用示例、基准评测、研究实验、数值验证、公共资产和文档各有归属。
本页说明目录职责、内容如何分类，以及研究成果如何进入长期维护的位置。

## 目录职责

| 目录 | 用途 | 典型内容 |
| --- | --- | --- |
| `braincell/` | 核心实现及行为回归测试 | 公共接口、内部计算模块、与源码相邻的 `*_test.py` |
| `examples/` | 展示用户如何完成任务 | 构造 Cell、组网、记录、参数学习、绘图 |
| `benchmarks/` | 在可重复的条件下评价性能或任务表现 | 编译时间、吞吐量、内存、规模扫描、序列学习基准 |
| `experiments/` | 探索 proposal 中的方法与候选实现 | 新模型、学习算法原型、消融实验、方案验证 |
| `validation/` | 验证数值结果和模型行为 | 与 NEURON 对照电压、机制状态、事件时序及误差 |
| `data/` | 保存可共享的输入和参考资产 | 形态、参考机制 `.mod` 源文件、参考轨迹 |
| `docs/` | 解释用法、设计和结论 | 使用文档、模块接口与架构、提案、研究依据、历史记录 |

目标目录结构如下，子目录按实际内容创建：

```text
braincell/
docs/
  repository.md
  design/
  specs/
examples/
  cell/
  network/
  optim/
  vis/
  ...
benchmarks/
  performance/
  profiling/
  tasks/
    sequence_learning/
experiments/
  <topic>/
validation/
  neuron/
    morph/
    channel/
    ion/
    synapse/
    cable/
    cell/
data/
  morphology/
  mechanisms/
  reference_traces/
```

核心代码的内部模块关系见 {download}`系统总览 <design/architecture/current/system-overview.md>`，
开发与测试约定见根目录 {download}`AGENTS.md <../AGENTS.md>`。

## 示例按模块组织

`examples/` 与 `docs/design/` 使用相同的模块主题名称，例如 `cell`、`network`、`optim`、`vis`。
对应的是主题；示例目录直接保存可运行脚本、notebook 及必要的辅助文件。

一个示例按主要教学目的归属。网络构建放 `network`，学习通道或突触参数放 `optim`，
形态绘图放 `vis`。单室和多室模型的基本用法都归 `cell`，文件较多时再按模型类型细分。
示例可以使用多个模块，并链接到各模块的接口说明。

脚本与讲解它的 notebook 放在一起。多个工作流使用同一个模型时，复用已有的模型构造函数；
共享需求稳定后，再提取共同实现，避免各处复制模型和参数。

## 基准、实验与验证如何区分

分类依据是代码要回答的问题。同样是训练序列模型，可以有三种用途：

| 问题 | 归属 | 应保留的内容 |
| --- | --- | --- |
| 如何构建并训练一个网络？ | `examples/network/` 或 `examples/optim/` | 最小完整流程、参数设置、结果读取 |
| 不同模型或算法在相同任务上表现如何？ | `benchmarks/tasks/sequence_learning/` | 固定任务协议、模型配置、评价指标和结果汇总 |
| 新的学习规则或模型结构是否有效？ | `experiments/<topic>/` | 候选实现、实验配置、分析过程及关联 proposal |

### Benchmarks

| 分类 | 评价内容 |
| --- | --- |
| `performance/` | 构建与编译时间、稳定运行耗时、吞吐量、内存占用，以及 neuron/CV/synapse/时间步规模变化 |
| `profiling/` | 定位计算热点、设备利用率、内存与数据传输开销 |
| `tasks/` | 序列学习等固定任务上的学习效果和资源成本 |

任务目录按任务命名，具体循环网络模型和训练算法作为配置。同一任务可以比较不同模型，
也可以比较同一个模型的 BPTT、RTRL 等训练方式。

任务基准应明确数据划分或生成规则、评价指标、训练预算、随机种子、模型与算法配置。
结果同时记录软件版本和硬件环境。按任务需要报告最终准确率或损失、训练时间、峰值内存，
以及达到目标表现所需的训练步数。

性能基准分别记录编译与稳定运行时间，并明确设备同步和数据传输是否计入耗时。
规模扫描保留各规模的结果、资源需求和失败原因。性能诊断中的额外同步或 profiler 配置会改变执行开销，
因此诊断结果应注明采集条件。

### Experiments

`experiments/` 按 proposal 或研究主题组织，允许自由调整实现和实验结构。
一个主题可以同时包含原型代码、正确性检查、规模实验和任务测试，保持研究过程完整。

本地 README 给出研究问题、运行入口和对应 proposal 的链接。接口还在变化的原型留在实验中；
获得稳定接口、明确用途和验证依据后，再将适合长期维护的部分迁出。

### Validation

`validation/neuron/` 组织 BrainCell 与 NEURON 的数值对照，按形态、通道、离子、突触、
cable 和整细胞等对象分类。每个工作流明确共同输入、刺激、温度、初态、离散与求解配置，
并记录比较量、容差和差异来源。

NEURON 对比若主要评价电压或机制状态的一致性，归 validation；若主要评价速度和内存，
归 benchmarks。性能评测仍需检查数值正确性，避免把不同计算结果当作速度提升。

发现具体实现错误后，将适合日常运行的小型复现用例补到核心代码旁。
大型对照与规模扫描保留独立运行入口，其调度和依赖要求随工作流记录。

## 文档分工

| 位置 | 维护内容 |
| --- | --- |
| 本页 | 仓库目录职责、内容归属、公共资产与成果流转 |
| `docs/` 中的使用文档和 API 参考 | 安装、建模流程、接口查询及用户排错 |
| 根目录 `CONTRIBUTING.md` 与 `docs/developer/` | 贡献入口、开发环境、代码与设计导航、测试和 PR 流程；模块契约链接 Design |
| `docs/design/<module>/current/` | 已实现的接口契约、架构、专题说明和可复现实测结论 |
| `docs/design/<module>/proposals/` | 具体问题、候选方案、取舍和待决定事项 |
| `docs/design/<module>/references/` | 外部方法、文献、推导和研究依据 |
| `docs/design/<module>/TODO.md` | 模块事项、状态、下一步与详情入口 |
| `docs/design/TODO.md` | 宏观目标、模块里程碑、主要依赖和阻塞 |
| `docs/specs/` | 按日期保存的历史决策与变更记录 |
| 工作流旁的 README | 本地文件导航、运行命令、输入输出和环境要求 |

完整的接口、架构、提案与结果记录规格由 {download}`Design 文档规范 <design/AGENTS.md>` 维护；
开发协作规则由根目录 AGENTS.md 维护。

设计论证和长期引用的实验分析保留一份权威说明，工作流 README 链接到它，说明中反向链接复现入口。
根目录 `examples/` 保存实际示例；文档网站中的 `docs/examples/` 负责展示与导航，优先引用已有示例。

## 公共资产与运行产物

| 内容 | 存放方式 |
| --- | --- |
| 多个工作流复用的形态和参考输入 | `data/` 下按资产类型、来源或模型家族组织 |
| NEURON `.mod` 等参考机制源文件 | `data/mechanisms/`，保留来源、版本、使用许可和本地修改说明 |
| 工作流专用的小型输入与配置 | 随对应工作流维护，便于独立复现 |
| 编译动态库、缓存、训练 checkpoint、原始 trace 和临时图表 | 工作流的 `artifacts/`，默认忽略提交 |
| 长期使用的参考轨迹或基准摘要 | 选择必要的小型资产纳入版本控制，记录生成配置与来源 |
| 大型数据和结果 | 保存获取位置、版本或校验信息及复现命令，按需下载或生成 |

`.mod` 是可编译的参考机制源文件。原始模型与移除 TABLE、修改积分方法等本地版本应能明确区分，
比较结果注明使用的版本。编译脚本、加载器和模型运行脚本随工作流维护，编译输出进入 artifacts。

公共输入保持一个维护位置，各工作流引用它。读取路径的公共辅助函数在存在共享需求时集中维护，
避免多个脚本分别假设目录深度。

## 从实验到长期维护

| 成果 | 后续归属 |
| --- | --- |
| 接口稳定、适合复用的模型或算法 | `braincell/`，配套模块 API、架构说明和相邻测试 |
| 固定协议、可长期重复比较的评测 | `benchmarks/` |
| 成熟的数值对照或模型验证流程 | `validation/` |
| 能清楚展示已实现接口的用法 | `examples/` |
| 设计决定、研究结论及其证据 | 对应模块的设计文档，必要历史进入 specs |

一个研究主题可以产出多类成果。迁出公共实现后，实验代码引用新的实现，保留复现实验所需的配置和入口。

## 现有内容的整理方向

| 当前内容 | 目标归属 |
| --- | --- |
| `examples/single_compartment/` 与 `examples/multi_compartment/` | 按教学主题分入 `examples/cell/`、`network/`、`optim/`、`vis/` 等 |
| `benchmarks/profiling/` | 诊断工具进入 `benchmarks/profiling/`，性能与规模评测进入 `benchmarks/performance/` |
| `benchmarks/performance/simulator_compare/` | `benchmarks/performance/simulator_compare/` |
| `examples/experimental/` | `experiments/`，按研究主题保持原型与实验的关联 |
| `validation/neuron/` | 对照工作流进入 `validation/neuron/`，公共参考资产进入 `data/` |
| `validation/neuron/Cerebellum_mod/` | 按来源和模型家族整理到 `data/mechanisms/`、`data/morphology/`，编译产物移入 artifacts |
| `docs/developer/` | 保留为贡献者指南，维护贡献步骤和阅读导航；完整接口、公式与架构由 Design 维护 |

后续目录迁移需要同步处理 Python 导入、notebook 路径、数据定位、运行命令、CI、
打包排除规则、产物忽略规则和文档引用。迁移按工作流验证运行入口及测试收集，核对共享模型与数据的引用关系。
贡献者指南已按上述分工更新；仓库入口为根目录 CONTRIBUTING.md，网站入口为 [Developer](developer/index.rst)。
