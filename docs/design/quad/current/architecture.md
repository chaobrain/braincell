# Quad Architecture

积分步骤消费目标的导数/阶段协议，电压后端消费已装配的电缆系统。几何和机制布局由 Cell 提供，Quad 不拥有 morphology。

```text
general step: state -> stage derivative -> weighted stage combination -> new state
staggered: current snapshot -> mechanism/voltage stages -> constrained cable solve -> new state
```

通用显式 RK 在每个局部 stage 重新计算导数，以加权组合更新 DiffEqState。
独立积分机制使用自己的 solver/substeps，外层 Cell 步长由 brainstate.environ.dt 决定。
staggered 的离子电流快照和 family/integration 更新顺序由
[Cell 调度](../../cell/current/architecture.md#离子电流快照与调度) 控制。

电压求解中，CV 节点有膜电容，边界/分叉 point 提供代数约束；DHS 用树结构求解装配后的线性系统。
CV 电压与 point 电压不能仅按数组长度互换。完整装配、两条推进路径及方程见
[Cell 架构](../../cell/current/architecture.md)。

当前显式 axial operator 消去边界时没有同步端点输入的等效贡献，这是方程完整性问题，
提高 RK 阶数不能恢复缺项。对应改进由 [Cell proposal](../../cell/proposals/explicit-solver-boundary-inputs.md) 管理。
求解精度与梯度结论应对应具体积分路径和 dt，已有分析见
[solver 梯度](../../optim/references/staggered-solver-gradient-analysis.md)。

实现入口：[quad 导出](../../../../braincell/quad/__init__.py)、
[积分协议](../../../../braincell/quad/protocol.py)。选择与注册用法见 [API](api.md)。
