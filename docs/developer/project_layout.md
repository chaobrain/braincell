# 代码与设计导航

先按要修改的功能找到源码，再阅读该模块的 TODO 和 Current。
目录职责与运行入口见 [仓库组织指南](../repository.md)，跨模块数据流见
[系统总览](https://github.com/chaobrain/braincell/blob/main/docs/design/architecture/current/system-overview.md)。

## 按任务定位

源码路径相对仓库根目录。每个模块的 TODO 提供当前事项及 API、架构、proposal 的入口。

| 要修改什么 | 源码入口 | 设计入口 |
| --- | --- | --- |
| 形态构造、连接、统计 | `braincell/morph/` | [Morph](https://github.com/chaobrain/braincell/blob/main/docs/design/morph/TODO.md) |
| SWC/ASC、checkpoint、在线形态读取 | `braincell/io/` | [IO](https://github.com/chaobrain/braincell/blob/main/docs/design/io/TODO.md) |
| 区域、位点与连续采样 | `braincell/filter/` | [Filter](https://github.com/chaobrain/braincell/blob/main/docs/design/filter/TODO.md) |
| 机制声明与注册 | `braincell/mech/` | [Mech](https://github.com/chaobrain/braincell/blob/main/docs/design/mech/TODO.md) |
| 通道与离子动力学 | `braincell/channel/`、`braincell/ion/` | [Channel](https://github.com/chaobrain/braincell/blob/main/docs/design/channel/TODO.md)、[Ion](https://github.com/chaobrain/braincell/blob/main/docs/design/ion/TODO.md) |
| Cell、离散与运行时绑定 | `braincell/_multi_compartment/`、`braincell/_discretization/`、`braincell/_compute/` | [Cell](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/TODO.md) |
| 单室模型 | `braincell/_single_compartment/` | [Single/Cell 统一讨论](https://github.com/chaobrain/braincell/blob/main/docs/design/cell/proposals/single-multi-compartment-unification.md) |
| 突触动力学 | `braincell/synapse/` | [Synapse](https://github.com/chaobrain/braincell/blob/main/docs/design/synapse/TODO.md) |
| 网络、事件源与连接 | `braincell/network/` | [Network](https://github.com/chaobrain/braincell/blob/main/docs/design/network/TODO.md) |
| 数值积分与电压求解 | `braincell/quad/` | [Quad](https://github.com/chaobrain/braincell/blob/main/docs/design/quad/TODO.md) |
| 参数选择与学习映射 | `braincell/trainable/` | [Optim](https://github.com/chaobrain/braincell/blob/main/docs/design/optim/TODO.md) |
| 图形展示 | `braincell/vis/` | [Vis](https://github.com/chaobrain/braincell/blob/main/docs/design/vis/TODO.md) |

内部源码路径与公共导入名不同。例如 Branch、Morphology 从 `braincell` 顶层导入，
机制声明从 `braincell.mech` 导入；实际调用以对应模块 Current 的公开入口为准。

## 实验、验证和性能工作流

通用实验梯度接口在 `braincell.experimental.optim`，使用流程在 `examples/optim/`。
数值精度对照在 `validation/`，性能与规模测试在 `benchmarks/`，共享参考资料在 `data/`。
小脑模型按 `data/cerebellum/<model>/` 管理形态、MOD 和参数；编译输出写入 validation 的 artifacts。
具体命令见 [仓库组织指南](../repository.md)。

## 阅读与更新顺序

1. 从模块 TODO 找到事项；Current 描述当前行为，proposals 保存尚需讨论或实施的方案。
2. 对照源码和相邻 `*_test.py`，确认现有约束和实际使用方式。
3. 查看对应示例或数值对照，确定修改后要验证的结果。
4. 实现后更新受影响的 Current、示例和事项状态；必要的历史决定按日期保存在 specs。

新增机制的步骤见 [扩展指南](extending.md)，测试定位与 fixture 用法见 [测试指南](testing.md)。
Design 的文档分工和状态定义由
[Design 规范](https://github.com/chaobrain/braincell/blob/main/docs/design/AGENTS.md) 维护。
