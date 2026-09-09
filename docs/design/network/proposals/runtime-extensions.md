# Network 运行时扩展

状态：待讨论。保留原 issues 的问题编号，并汇总原 implementation-plan 的延后事项。
进度见 [Network TODO](../TODO.md)，已有契约见 [架构](../current/architecture.md) 和 [API](../current/api.md)。
以下方向尚未形成完整实施与验收合同，不提供占位公共 API。

## I-09 Sparse delay slots

当前每个 target layout 使用 dense time ring，成本与最大 delay 和 layout width 相关。后续评估只保存
实际 event rows 的 sparse slots，并比较 JIT 静态 shape、scatter 成本和事件密度阈值。
需要先确定 sparse/dense 自动选择规则及性能基准，不能仅凭理论稀疏度宣称性能提升。

## I-10 Trainable topology

当前只允许初始化前结构编辑和初始化后 shape-preserving 参数更新。可学习连接存在性、位置或新增/
删除 rows 会改变 JAX shapes，需要独立的 masked/padded 或重编译协议，不能复用普通参数训练接口。
讨论需覆盖初始化后的结构 mutation、状态 owner、reset 及已有连接身份的兼容性。

## I-11 Scalable endpoint generators

当前通用 pairing 会按实际候选矩阵计算 conditional score；语义已经锁定，但大 N 下仍需增加不改变
结果的 score chunking、Bernoulli/all-to-all specialized generator，并记录 host peak-memory contract。

## Network batch runtime

这是原计划中的延后方向，不等于已支持的同构 Cell population。先确定 Network 层 batch 的含义、
拓扑是否共享、事件与 recording 的轴约定，再确定实现范围和验收。

## 研究依据

- [平台调研](../references/platform-survey-2026-06.md)：作为扩展比较背景，不表示引入了任何平台的完整执行模型。
- [Connection 与 Synapse 语义](../references/bmtk-netpyne-synapse-sharing.md)：用于检查扩展能否保持既有 owner 边界；其中候选 recipe 不自动成为已批准接口。
