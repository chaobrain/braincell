# 突触事件实验结果索引

本索引链接随 Git 维护的完整实测总结。每份总结保存测试环境、协议、数据与证据范围；原始 artifacts 可能仅存在于运行者本地，未随 Git 提供。公共运行入口见 [README](README.md)。

| 实验主题 | 完整实测总结 | 阅读重点 |
| --- | --- | --- |
| 生产路径基线 | [NetStim 基线](results/netstim-baseline.md) | 规模、布局、声明分组与未来时间表；包含数值不匹配记录 |
| 预定事件查询 | [查询比较](results/scheduled-event-queries.md) | scan、cursor、bucket 的查询/消费者成本，窗口与边界限制 |
| 直接投递 | [结果解释](results/scheduled-event-delivery.md)、[数据表](results/scheduled-event-delivery-tables.md) | direct/padded 的完整模型收益、存储与首次成本 |
| 完整模型四层对照 | [结果解释](results/scheduled-event-controls.md)、[数据表](results/scheduled-event-controls-tables.md) | 无突触、无连接、静默、正常输入的完整运行对照 |

这些实验回答不同问题，模型与计时图并不相同，不能跨实验相减为内核独占耗时。[设计方案](../../../docs/design/synapse/proposals/event-delivery-optimization.md) 维护采用方向与尚未验收的生产接入合同。
