# Optimization 后续方向

状态：待讨论；具体候选方案见 [可塑性](connection-plasticity.md) 和 [训练恢复](training-recovery.md)。
进度唯一入口是 [Optimization TODO](../TODO.md)，本文只保留讨论边界和验收方向，不重复状态表。
现有能力见 [支持度](../current/parameter-support.md) 与 [实验工作流](../current/experimental-workflows.md)。
已完成 P0 的历史明细见 [归档](../../../specs/2026-08-29-trainable-parameter-implementation.md)。

## Cell 初值与 cable 参数

Cell 初值和 cable 参数目前没有 trainable owner 接入。先梳理构造转换、几何缓存、runtime buffer
与初始化依赖，再确定注册和重置契约。构造时可以配置不等于运行时可以训练。

## 公共训练协议与分组

数据、loss、result、resume 的公共协议需先验证实验组合接口能否通用，不提前占用公共类型名。
稳定 branch/region/custom grouping 需先解决 row fingerprint、持久化和 ownership 合同。

## 验证缺口

- 完整 RTRL 的多 CV × 多 population 组合：固定参数下验证全网梯度、跨 CV/Cell 敏感度和资源扩展。
- 双向事件网络 GPU 性能：独立进程、公平 warm timing、相同 surrogate、记录峰值内存。
- checkpoint BPTT 与 RTRL 对照：分别报告重计算成本、临时内存和进程峰值。

已有单项结果不能代替上述组合验收。更换 solver 或加入自定义可微机制时，按
[solver 分析](../references/staggered-solver-gradient-analysis.md) 重新检查初始化、状态捕获和程序导数，
不能沿用旧误差数字作为新验收结果。

## Rollout 内更新参数

先定义梯度对应哪个参数历史，不从固定参数等价性外推。这是独立研究方向。

## 关联提案与边界

- [Connection 可塑性训练](connection-plasticity.md)：先依据 [Network 可塑性提案](../../network/proposals/connection-plasticity.md)
  确定运行时契约，再讨论初始权重、规则参数的注册和梯度验收；冻结讨论边界不代表发布占位 API。
- [训练恢复](training-recovery.md)：有 observer/archive 不等于有 controller；先完成 effective-LR 与恢复状态测试，保留原有策略和消融协议，正式比较仍未执行。

delay 训练保持不支持，不因本路线图而自动进入开发范围。
proposal 确认并实际实现后才更新当前 API/实验工作流；实测进入 current/results，理论继续留在 references。
设计可以领先代码，但示例不能展示不存在的调用；历史 specs 不反向覆盖现行合同。
维护方式见 [文档对齐](../TODO.md#文档对齐)。
