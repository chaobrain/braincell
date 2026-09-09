# Trainable Parameter Support

## 口径与共同能力

公共入口见 [API](api.md)，
实验梯度入口见 [实验工作流](experimental-workflows.md)，实现与发布状态见
[总览](../TODO.md#状态口径)。下面的参数是例子，不是新的可训白名单。

Channel、Ion、Synapse 通过构造签名发现候选参数，保留默认 get/set、单位、区域选择、
row/population/cv/all 分组、共享根，以及 parameter/scale/parameterized。
声明在初始化前完成。注册通过不保证非零梯度；实际类型转换、单位、shape、静态控制流
仍可能自然报错。训练发生在原 root 上，reset 清动态状态但不回滚 root。

| 对象 | 当前入口与能力 | 关键边界 |
| --- | --- | --- |
| Channel | ChannelView.trainable；签名参数 | 内部常数和动态 gate 不自动暴露 |
| Ion | IonView.trainable；签名参数与具名初始参数 | 初始参数与动态浓度分离 |
| Synapse | SynapseView.trainable；签名参数 | 事件输入不妨碍 tau/e 等连续参数求导 |
| Connection | ConnectionView.trainable(weight=...) | delay 明确拒绝；可塑性未实现 |
| 电压检测器 | event output/source trainable(threshold=...) | 硬事件前向，代理梯度反向 |
| Network | trainables 聚合、prepare_run/update | 路由固定；共享根去重；全网状态一起求导 |

## Channel

典型候选包括 g_max、V_sh、temp、q10、temp_ref，以及显式独立 phi。
签名中的继承/转发参数可以发现；未暴露到签名的内部速率常数仍由自定义模型负责。

温度派生的 sodium phi 在属性访问时根据当前参数求值；独立传入的 phi 不绑定温度。
速率函数本身每次调用会读取当前参数，不必因为依赖其他参数再改成 property。
开关或比较产生零梯度是合法结果；不能仅凭整数默认值判断某个数值参数不可学。

已验证省略默认值、区域隔离、共享根、dtype 提升、重复 JIT/reset、温度依赖、
有限差分及自然错误。教学拟合覆盖 g_max、V_sh、temp；不是所有 Channel 的训练穷举。
示例：[channel_learning.ipynb](../../../../examples/optim/parameter_learning/channel_learning.ipynb)。
测试：[manager](../../../../braincell/trainable/_manager_test.py)、
[base Channel](../../../../braincell/_base_channel_test.py)。
实测：[参数学习结果](results/parameter-learning.md#channel)。

## Ion

区分三个层次：固定物理参数（如固定 Ion 的 E/Ci）、动力学参数（如 tau/kf），
以及动态状态的初始参数（如 Ci_initializer）。Ci(t) 本身不是用初始参数替代的常量。
每轮在求导范围内 reset，使 Ci(0) 读取更新后的初始 root。

复杂 Ion 的默认平衡初值在 reset 按当前依赖参数计算；显式初值保持独立，未选择区域
保留默认依赖。Fixed E 保持固定语义；InitNernst 在初始化/reset/参数同步刷新；
DynamicNernst/KineticIon 在读取时计算。不是所有派生量都具有相同刷新时机。
species_initializers 字典的覆盖功能保留，但不增加内部字典键的训练入口；
已有具名 BC_initializer 等签名参数可选择。

已验证初值依赖、区域覆盖、共享 Channel/Ion 根、Nernst、shell 因子、重复编译和梯度。
CalciumFirstOrder 既有 alpha/beta 默认单位不一致，不能作为已通过的训练例子。
示例：[ion_learning.ipynb](../../../../examples/optim/parameter_learning/ion_learning.ipynb)。
测试：[Ion 参数集成](../../../../braincell/trainable/_manager_test.py)、
[Ion 基类](../../../../braincell/ion/_base_test.py)。
结果与已知精度边界：[参数学习](results/parameter-learning.md#ion)、
[网络验证限制](results/synapse-network-learning.md#验证与限制)。

## Synapse

Synapse 使用构造签名 metadata，不再通过手写字典决定训练资格。
ExpSyn 的 tau/e、Exp2Syn 的 tau1/tau2/e 是典型候选；实际模型仍要求有效时间常数，
训练时由 transform 或参数化函数维持正值及 tau1 < tau2。
Exp2Syn 的归一化因子读取当前时间常数；reset 清突触动态状态，不清参数根。

固定 NetStim/event 输入下，学习 tau 或 weight 不需要对事件时间求导；
跨 presynaptic Cell 或 threshold 反传才需要代理梯度路径。没有输入或 loss 不敏感时，
零梯度并非接口故障。同一 CV 上的多个逻辑突触仍是独立 row，除非显式按 cv 分组。

示例：[synapse_learning.ipynb](../../../../examples/optim/parameter_learning/synapse_learning.ipynb)。
测试：[ExpSyn/Exp2Syn](../../../../braincell/synapse/exponential_test.py)、
[点目标参数](../../../../braincell/trainable/_targets_test.py)。
结果：[单参数拟合](results/parameter-learning.md#synapse-与-connection)。

## Connection 与检测器

weight 由接收 Cell 持有，按逻辑 contact 分组；投递读取当前物理 State，不冻结编译前权重。
默认 spike output 的 threshold 绑定 Cell.V_th；显式 detector threshold 可独立持有参数。
升沿为 last < threshold <= next，降沿为 last > threshold >= next。
事件值必须保持浮点，从检测器经过乘权、scatter/稀疏投递到 delay queue 保留梯度。

delay 明确拒绝训练，routing/source index 等拓扑也不是当前训练入口。
weight_initial 与动态可塑性 weight 尚未接入，见 [候选方案](../proposals/connection-plasticity.md)。
边界与 NEURON 差异由 [事件架构](architecture.md#event-derivatives) 统一定义。
测试：[事件](../../../../braincell/network/event_test.py)、
[投递](../../../../braincell/network/delivery_test.py)、
[参数目标](../../../../braincell/trainable/_targets_test.py)。
示例仍使用上面的 synapse_learning Notebook。

## Network 与梯度方法

Network.trainables 聚合原始 Cell roots，以 population.local_name 命名，并按对象身份去重。
host 上先 prepare_run 固定 dt、路由和队列形状，再以 BrainState 编译循环调用 update。
run 的 host 记录转换不是可微 rollout 接口。BPTT/full RTRL 使用同一个完整状态转移；
不能在有连接时把跨 Cell 敏感度截成独立块。

已实测 A(2) 与 B(3)、每成员 1 CV、12 个双向 contacts、45 个独立坐标，含单侧损失、
共享根、事件截断对照、前缀和队列敏感度。另有单 Cell 多 CV 证据；不意味着任意
多 CV × 多 population 组合已经验证。rollout 内更新参数、GPU 上的新双向网络组合未测。

测试：[Network roots](../../../../braincell/trainable/_network_test.py)、
[双向网络](../../../../validation/optim/gradient_correctness/bidirectional_test.py)。
示例与计时：[网络结果](results/synapse-network-learning.md)。

## 暂不支持的 Owner

Cell.V_init、cable、morphology/topology 和初始化后改变 trainable ownership 尚未接入。
构造时能够设置这些值不等于当前 View.trainable 支持它们。后续入口见 [路线图](../proposals/roadmap.md)。
