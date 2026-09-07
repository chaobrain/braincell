# Mech 运行时扩展

Junction 已有占位声明，但没有连接另一端的身份，也没有在电压装配中贡献跨 Cell 电流。
NMODL 生成与统一 MOD 对照框架则需要从现有模型导入和比较脚本提取共同部分。

## Junction

两个位置 a、b 的电耦合应给出成对电流 `I_a=g*(V_b-V_a)`、`I_b=-I_a`。
下一步需决定 partner 用逻辑 placement ID 还是显式端点对表示，并确认跨 Cell 求解的同步顺序。
只补一个 params 字段不能完成耦合，验收需包含电流守恒和两个 Cell 的同步电压对照。

## 机制生成与验证

生成器的目标是现有 registry 注册类、参数构造签名和 Channel/Ion/Synapse 模板。
待讨论的最小范围包括单位、状态方程、事件输入以及特殊求解语句；
以一个有公开 MOD 参照的机制完成“生成、注册、放入 Cell、电压钳对照”后再扩展语法。
比较框架从 [现有 NEURON 示例](../../../../examples/neuron_compare/) 提取，逐模型来源继续由
[共享文献表](../../ion/references/ion-channel-bibliography.md) 维护。

事项状态见 [Mech TODO](../TODO.md)。
