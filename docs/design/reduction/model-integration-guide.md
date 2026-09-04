# 约化模型接入指南

本文面向实现 DBNN、DLIF 或其他 Cell 约化模型的开发者。约化模型不是新的
Network population 类型，而是挂载在一个已经声明好 morphology、population 和
synapse 的 `Cell` 上，并替代该 Cell 的 detailed dynamics 执行。

运行时的权威行为定义见
[`cell-reduction-runtime.md`](../../specs/2026-09-04-cell-reduction-runtime.md)。本文只说明如何实现
一个模型，以及哪些行为属于公共契约、哪些细节由具体模型自行决定。

## 1. 最短接入流程

推荐按以下顺序构建 Cell：

1. 创建 detailed Cell，完成 morphology、CV policy 和 population 声明。
2. 放置约化模型需要看到的全部 synapse，并设置其类型和参数。
3. 创建并注册一个或多个 `ReductionModel`。
4. 如有需要，通过 reduction view 设置不同 population member 的参数。
5. 用 `cell.use_model(name)` 选择本次仿真使用的模型。
6. 声明 raw output recording，把原 Cell 加入 Network 并正常连接、运行。

```python
import braincell
import brainunit as u

# morphology、paint 和 synapse placement 已经完成。
cell.add_reduction("dbnn", DBNNReduction.load("dbnn-model"))
cell.add_reduction("dlif", DLIFReduction(...))

# 可选；只有模型实现 get()/set() 时才可使用。
cell.reductions["dlif"].set(threshold=-50.0 * u.mV)

cell.use_model("dbnn")
cell.record("soma_voltage", braincell.observe.output("voltage"))

network = braincell.Network()
reduced = network.add_population("reduced", cell)
# 使用普通 connection API 将上游连接到 cell 已经声明的 SynapseView。
result = network.run(duration=100.0 * u.ms, dt=0.025 * u.ms)
```

一个 Cell 可以注册多个约化模型，但一次初始化只能选择一个。选择作用于整个 root Cell，
不能给同一个 Cell 的不同 `CellView` 分别选择执行模型。`"detailed"` 是保留名称；执行
`cell.use_model("detailed")` 可在下一次初始化恢复 detailed dynamics。

synapse 不要求在 `add_reduction()` 之前放置，但必须在 Cell 初始化之前完成。运行时只在
初始化时根据最终 synapse 声明创建 `ReductionContext`。对于依赖固定 synapse 分布的训练模型，
应采用上面的推荐顺序，避免注册时误以为 context 已经固定。

## 2. 必须实现的公共接口

所有模型继承 `braincell.ReductionModel`，并实现以下三个方法：

```python
class ReductionModel:
    def init_state(self, context, batch_size=None) -> ReductionOutput: ...
    def update(self, inputs) -> ReductionOutput: ...
    def reset_state(self, batch_size=None) -> ReductionOutput: ...
```

| 方法 | 由谁调用 | 必须完成的工作 |
| --- | --- | --- |
| `init_state()` | Cell 初始化 | 校验 context，编译静态输入映射，创建动态状态并返回初始输出 |
| `update()` | 每个 Cell update | 消费本步 payload，将模型推进一步并返回新输出 |
| `reset_state()` | 原位重置 | 清空动态状态，保留当前 context、学习参数和已编译映射，并返回初始输出 |
| `reset()` | Cell 完全反初始化 | 可选地删除依赖当前 context 的缓存；必须保留模型配置和学习参数 |
| `get()` / `set()` | reduction view | 可选的逐 population 参数接口；不支持时保留基类默认报错即可 |

`reset()` 在基类中是空实现。无 context 缓存的模型不需要覆盖它；DBNN、DLIF 这类保存了
输入映射或动态 state 的模型通常应该覆盖。

### 2.1 Context 是静态声明

`init_state()` 收到的 `ReductionContext` 包含：

- `pop_size`：Cell 的 population shape。
- `population_size`：展平后的 population member 数量。
- `synapses`：每个逻辑 synapse 的静态记录。
- `input_groups`：按 synapse runtime type 和 event-input contract 分组的输入 schema。
- `fingerprint`：当前 synapse 布局的运行时指纹。
- `cell`：对原 Cell 的弱引用，用于读取仍然存在的静态声明。

每个 `ReductionSynapse` 提供 `id`、member-local `synapse_index`、`population_index`、
`placement_id`、`point_id`、`cv_id`、`branch_id`、`branch_x`、`name`、`synapse_type` 和只读
`parameters`。模型可以使用全部、部分或完全忽略这些信息。

每个 `ReductionInputGroupSchema` 提供稳定的 `layout_id`、`synapse_type`、`event_input`，以及
同长度的 `synapse_id`、`synapse_index`、`population_index` 数组。不要假设：

- 所有 population member 拥有相同数量或相同顺序的 synapse；
- 不同 synapse type 会进入同一个 group；
- `synapse_id`、group 行号和 member-local `synapse_index` 相同；
- 一个模型只能遇到一种 payload 单位或 event-input contract。

需要 channelized input 的模型应在 `init_state()` 中根据这些显式 id 编译 scatter/gather 映射，
并在 `reset()` 中丢弃该映射。

### 2.2 Inputs 是已经交付的 payload

`update()` 收到 `ReductionInputs`。遍历它得到若干 `ReductionInputGroup`，每组包含静态
`group.schema` 和当前的 `group.payload`。

payload 已经完成以下网络语义：

- connection delay；
- connection weight；
- 同一时刻、同一目标 synapse row 的聚合。

模型不得再次应用 delay 或 connection weight。聚合前的 event 个数不会保留：trigger 类输入
只能判断聚合后的 slot 是否非零；scalar payload 可以读取聚合后的幅值。每次输入被包装后，
底层 event buffer 会被清零，因此模型若需要历史信息，必须保存在自己的动态 state 中。

输入 group 可以为空。一个具体模型可以支持空输入，也可以在 `init_state()` 中以明确错误拒绝
不支持的 synapse type、event-input contract、单位或布局。

### 2.3 Output 必须保持稳定

三个生命周期方法都必须返回真正的 `ReductionOutput`：

```python
output = braincell.ReductionOutput(
    values={"voltage": voltage, "latent": latent},
    event=spike,
)
```

设运行时前缀为：

```text
runtime_prefix = ([batch_size] if batched else []) + list(cell.pop_size)
```

必须满足：

- `event.shape` 必须严格等于 `runtime_prefix`，不能带 feature 轴。
- 每个 raw output 的 shape 必须以 `runtime_prefix` 开头，之后可以有任意 feature 轴。
- output 名称必须是非空字符串。
- 从 `init_state()` 到 Cell 完全 `reset()` 之间，名称及顺序、完整 shape、普通数组或 Quantity
  类型、dtype 和单位必须稳定。
- `init_state()` 与 `reset_state()` 返回的初始输出必须遵守相同结构。

`event` 会成为原 Cell 的 canonical `event_outputs["spike"]`，因此约化 Cell 仍可作为普通
connection 的上游。`values` 是模型自己的 raw outputs，可通过 `cell.outputs` 查看，也可用
`braincell.observe.output(name)` 记录。recording 取得的是当前 update 返回的状态。

当前 Cell-level reduction contract 支持 batch 前缀；当前 `Network` 在线执行仍不接受
`batch_size`。DBNN 的离线批量前向可以复用同一数学实现，但不能把离线 batch runner 当成
Network 接口。

## 3. 可复制的模型骨架

下面的骨架把公共生命周期和模型内部函数分开。实现者只需要替换标有“模型内部”的部分。

```python
import brainstate
import jax.numpy as jnp

import braincell


class MyReduction(braincell.ReductionModel):
    def __init__(self, parameters):
        # 模型内部：学习参数、超参数和资产元数据。
        self.parameters = parameters
        self._context = None
        self._compiled_inputs = None
        self._state = None
        self._batch_size = None

    def init_state(self, context, batch_size=None):
        self._validate_context(context)  # 模型内部
        self._context = context
        self._batch_size = batch_size
        self._compiled_inputs = self._compile_inputs(context)  # 模型内部

        prefix = ((int(batch_size),) if batch_size is not None else ()) + context.pop_size
        self._state = brainstate.ShortTermState(self._initial_state(prefix))
        return self._initial_output(prefix)

    def update(self, inputs):
        if self._state is None:
            raise RuntimeError("MyReduction requires init_state() first.")

        drive = self._project_inputs(inputs, self._compiled_inputs)  # 模型内部
        next_state, raw_values, spike = self._step(  # 模型内部
            self._state.value, drive, self.parameters
        )
        self._state.value = next_state
        return braincell.ReductionOutput(values=raw_values, event=spike)

    def reset_state(self, batch_size=None):
        if self._context is None:
            raise RuntimeError("MyReduction requires init_state() first.")
        if batch_size != self._batch_size:
            raise ValueError("reset_state() must preserve the initialized batch size.")

        prefix = ((int(batch_size),) if batch_size is not None else ()) + self._context.pop_size
        self._state.value = self._initial_state(prefix)
        return self._initial_output(prefix)

    def reset(self):
        # 保留 parameters；删除只对当前 Cell/synapse schema 有效的内容。
        self._context = None
        self._compiled_inputs = None
        self._state = None
        self._batch_size = None

    def _initial_output(self, prefix):
        # 名称、类型、单位和完整 shape 必须与 update() 返回值一致。
        voltage = jnp.zeros(prefix)
        spike = jnp.zeros(prefix, dtype=jnp.int32)
        return braincell.ReductionOutput(values={"voltage": voltage}, event=spike)
```

骨架中的 `_validate_context()`、`_compile_inputs()`、`_initial_state()`、`_project_inputs()` 和
`_step()` 都不是公共 API，只是建议的模型内部职责划分。它们可以改名、合并或完全替换。

如果模型需要提供逐 member 参数，额外实现：

```python
def get(self, field: str, population_indices: tuple[int, ...]): ...
def set(self, population_indices: tuple[int, ...], **parameters) -> None: ...
```

`set()` 只会在 Cell 未初始化时通过 `ReductionView` 调用。模型负责检查字段名、值的单位和 shape，
以及将 scalar 或选中 member 对应的值保存为声明期参数。

## 4. DBNN 和 DLIF 如何落到这个接口

### 4.1 DBNN

DBNN adapter 的公共部分仍然只有上述生命周期。模型内部通常负责：

- 从模型资产读取学习参数、训练 schema、dt、单位和版本。
- 在 `init_state()` 对照 `context.synapses`、`input_groups` 和资产 manifest，拒绝错误的 Cell、
  point、synapse prototype、参数或输入单位。
- 编译 `synapse_id -> DBNN channel` 映射；不能依赖运行时 group 的偶然行顺序。
- 在 `update()` 将 payload scatter 到 channel，执行一次在线递推，输出胞体 voltage，并从阈值
  上穿生成 canonical spike。
- 在 `reset_state()` 清空卷积核、双线性层或其他递归历史，但保留学习参数和 channel mapping。
- 在 `reset()` 删除 channel mapping 和 context 缓存，但保留已加载的模型资产。

`context.fingerprint` 可以用于快速拒绝完全不同的 synapse 声明，但 DBNN 资产仍应保存自己可读、
可版本化的 manifest。不要只保存一个不透明 hash，否则无法向用户说明是 point、prototype、参数、
dt 还是单位不兼容。

### 4.2 DLIF

DLIF adapter 可以用同一公共接口，但内部通常是：

- 在 `init_state()` 根据 synapse metadata 编译输入权重或 E/I channel 映射。
- 用 `brainstate.ShortTermState` 保存每个 population member 的 membrane/recurrent state。
- 在 `update()` 聚合本步 payload，执行 decay、积分、threshold 和 reset，返回 voltage 与 spike。
- 自行决定是否消费 synapse type、位置和参数；公共运行时不要求 DLIF 模拟真实 synapse dynamics。
- 若提供 threshold、decay 等逐 member 参数，再实现 `get()` / `set()`；否则保持为构造参数或模型
  资产的一部分。

DBNN 和 DLIF 都不应创建或推进 detailed `Cell.V`、ion、channel 或 synapse state。在 reduced
mode 下这些 runtime 根本不会分配。`context.cell` 只应用于读取声明和静态元数据，不能作为
detailed dynamics 的旁路。

## 5. 哪些细节完全属于模型自己

以下内容不应加入 `ReductionModel` 公共基类，除非将来至少两个正式模型出现相同、稳定的需求：

- 网络结构、状态方程、threshold/reset 规则和数值积分方式；
- 学习参数、训练循环、loss、optimizer 和 offline sequence forward；
- 输入 channel 定义，以及是否消费 payload 幅值、synapse type、参数或形态位置；
- checkpoint、部署资产、版本迁移和训练来源追踪格式；
- context 兼容范围，以及允许重建、拒绝或要求重新训练的策略；
- raw output 的名字、单位和 feature 维度；
- 是否支持可编辑的逐 member 参数；
- JIT、gradient、缓存和性能优化方式。

约束这些内部选择的只有公共输入、生命周期和输出契约。例如，模型可以输出 `voltage`、latent
features 或多个诊断量，但一旦初始化完成，本次 runtime 中不能动态增删这些字段。

## 6. 最低测试清单

一个准备接入主库的约化模型至少应验证：

- 非 batch Cell 下，初始、单步和 reset 输出 shape、dtype、单位一致。
- 若宣称支持 batch，batch 前缀正确且不同 member、不同 batch 不共享动态状态。
- 多个 population member 的 synapse 数量不相等时，输入仍按显式 schema 映射正确。
- 多个 synapse type/group、空 group 或完全无输入时，得到预期结果或明确的初始化错误。
- connection weight、delay 和同目标聚合只由运行时应用一次。
- `reset_state()` 与全新初始化得到相同动态初态，同时保留参数和 context 映射。
- `reset()` 后修改 synapse 并再次初始化时，模型能重新编译或报告需要重新训练。
- `values` 名称、shape、Quantity 类型和单位在每步保持稳定。
- 约化 Cell 作为上游和下游连接时，canonical spike 与输入 delay 语义正确。
- `observe.output()` 和 spike recording 取得的是预期的 post-update 输出。
- 对训练模型，资产 round trip 后预测、schema 校验和错误信息保持一致。
- 若模型承诺可训练或可 JIT，分别验证 gradient、JIT 后的单步结果与离线 sequence forward 一致。

现有 [`toy.py`](../../../braincell/reduction/toy.py) 提供三种较小的参考：
`EventAccumulatorReduction` 只看 slot 是否活动，`PayloadAccumulatorReduction` 消费幅值，
`SynapticKernelAccumulatorReduction` 还会读取 synapse type 和参数，但不运行真实 synapse dynamics。

## 7. 常见错误

- 在构造模型时读取 Cell synapse：此时 placement 可能尚未完成；应在 `init_state(context)` 读取。
- 按 group 行号固定 channel：异构 population 或重新声明后行顺序可能不再符合模型假设。
- 在模型内再次乘 connection weight：`group.payload` 已经包含它。
- 希望从 trigger payload 恢复同一时刻的原始 event 数：聚合后该信息已经不存在。
- 只在 `update()` 返回某个诊断字段：所有 output 必须从 `init_state()` 起存在并保持结构稳定。
- 用 `reset_state()` 重新加载资产或改变 batch：它只负责原位重置动态状态。
- 在 reduced mode 访问详细膜电压或真实 synapse state：这些 runtime 不会被创建。
- 用一个 fingerprint 代替可解释的资产 manifest：hash 适合快速比较，不足以诊断或迁移模型。
