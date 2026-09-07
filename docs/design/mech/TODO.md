# Mech TODO

机制声明、注册和输入契约的协作入口。宏观依赖见 [全局 TODO](../TODO.md)，文档规则见 [Design 规范](../AGENTS.md)。

| 事项 | 状态 | 下一步 | 详情 |
| --- | --- | --- | --- |
| 参数单位诊断 | 待讨论 | 对照当前构造签名与运行时校验，设计指向 paint 声明的错误信息 | [声明 API](current/api.md)、[Channel TODO](../channel/TODO.md) |
| Junction 运行时接线 | 待讨论 | 确定 partner 身份、对称连接与电压方程贡献 | [运行时缺口](proposals/runtime-extensions.md) |
| 旧 Probe 字段校验 | 待讨论 | 核对废弃 Probe 与现行 observe 的迁移方式，避免新增重复 taxonomy | [Recording](../network/current/recording.md) |
| MOD 数值验证框架 | 待讨论 | 从现有比较例子提取可复用的电压钳与电流钳对照 | [Channel TODO](../channel/TODO.md) |
| NMODL 生成器 | 待讨论 | 需要时以 registry 为生成目标，确定最小语法集 | [扩展方向](proposals/runtime-extensions.md) |

当前实现：[API](current/api.md)、[架构](current/architecture.md)。
