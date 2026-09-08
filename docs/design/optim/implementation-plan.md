# Trainable Parameter Implementation Plan

## 文档状态

本文定义 [Trainable Parameter API](api.md) 的实现顺序和验收边界。它不是代码状态报告；
在相应代码和测试合入前，所有阶段均视为未实现。

更宽的模型优化能力地图见 [Design overview](design-overview.md)，内部模型见
[Architecture](architecture.md)。

## 实现边界

P0 交付 `IL`、`Na_HH1952`、`K_HH1952` 的 Cell-local 可训参数闭环：

```text
View.trainable(source)
  -> Cell.trainables registry
  -> graph-discoverable ParamState roots
  -> binding materialization
  -> existing runtime simulation
  -> grad/update
```

P0 不同时实现 Dataset、loss abstraction、optimizer、search、history、checkpoint、Ion、
Synapse、Connection 或 Network trainable 聚合。

## 候选代码布局

```text
braincell/
  _parameter_schema.py
  trainable/
    __init__.py
    _sources.py
    _binding.py
    _manager.py
    _parameters.py
```

职责：

| Module | Responsibility |
| --- | --- |
| `_parameter_schema.py` | 公共 ParameterSpec、field roles、schema validation。 |
| `trainable._sources` | `parameter`、`scale`、`parameterized` immutable specs。 |
| `trainable._binding` | target metadata、group gather、JAX evaluator 和 scatter。 |
| `trainable._manager` | Cell registry、Network aggregation、事务注册和 materialize。 |
| `trainable._parameters` | ParameterSet、stable trees、atomic setters。 |

View 只负责把 logical row selection 和 target field 传给 manager。不得在 ChannelView、
IonView 或未来 ConnectionView 中复制 grouping、root ownership 或 optimizer logic。

## P0-A：Parameter Schema

1. 将 Synapse 已有 `ParameterSpec` 的通用部分提升到公共 schema，并保留旧 import 的
   compatibility re-export。
2. 定义 `parameter/state/derived/static` 角色；trainability 由 binding policy 决定。
3. 为 `IL`、`Na_HH1952`、`K_HH1952` 标注完整字段。
4. 在 ChannelView 提供只读 `parameter_info()`。
5. 未标注 mechanism 明确报告 unsupported，不回退到 constructor introspection。

验收：

- schema 可以验证单位、单字段 validator 和 cross-field validation；
- gate、dynamic concentration、derived Nernst `E`、valence、solver 和 topology 被拒绝；
- discovery 在 init 前后返回相同的字段角色。

## P0-B：Source 与 Root Registry

1. 实现 immutable direct、scale 和 parameterized source specs。
2. 实现四种 group key 和 ragged-safe `row -> root` gather index。
3. 实现 `TrainableManager`，作为 Cell graph 子 module 持有 roots。
4. 实现 root stable naming、对象身份去重和名称冲突检测。
5. 实现 `View.trainable(**fields)` 的预验证、原子提交和 overlap ownership map。
6. 实现 `cell.trainables`；Network 聚合延后。

验收：

- `graph.states(cell, ParamState)` 只返回 roots；
- 共享一个 factor 的多个 bindings 只产生一个 ParamState；
- 多字段调用中任意字段失败时 manager 状态不变；
- 4 x 10 rows 对四种 group 分别产生 40/4/10/1 个自由度；
- grouped direct current values 不一致时不平均并明确报错。

## P0-C：ParameterSet

1. 实现 stable name -> root 的 ParameterSet facade。
2. 暴露原 ParamState tree、physical value tree 和 optimizer value tree。
3. 实现完整 tree 的 physical/raw atomic setter。
4. 验证 key、shape、dtype、单位、finite 状态后再提交。
5. setter 使用现有 Param/State 更新路径，保留对象身份并正确失效 transform cache。

验收：

- optimizer 注册的 ParamState 与模型实际读取的 root 是同一对象；
- setter 失败不产生部分更新；
- physical -> raw -> physical round trip 遵守 BrainState transform 精度；
- Quantity leaf 保留单位，dimensionless root 不被包装成错误物理单位。

## P0-D：Binding 与 Materialization

1. 为 direct source 实现 group gather 和 runtime row scatter。
2. 为 scale source 捕获 frozen baseline，创建或复用 dimensionless factor。
3. 为 parameterized source 使用函数签名绑定 trainable/fixed arguments。
4. 将 materialization 接入 Cell init/reset/run。
5. 保留显式 `manager.materialize()` 给直接 update 流程。
6. 对已绑定 target 的普通 set 建立明确拒绝语义，避免断开 binding。

验收：

- materialization 全程 JAX-traceable，不出现 NumPy/Python scalar conversion；
- 参数影响 mechanism reset 时，在 reset hook 前已经刷新；
- 连续手写 update 不会每 step 重复执行空间函数；
- reset_state 不改变 roots 或 scale baseline；
- Cell.reset 后重新初始化能根据稳定 row metadata 重建 runtime target。

## P0-E：真实梯度闭环

完成以下集成场景：

1. 两 CV Leak 分别用 direct `g_max` 和 `baseline * theta` 表示，在等价初值下 forward 和
   链式梯度一致。
2. Na/K 共享同一个 factor，只有一个 root，但 materialized values 使用各自 baseline。
3. `distance(ctx) * a + b` 在 scalar roots 时为 2 DOF，在四个 population members 上按
   population 分组时为 8 DOF。
4. scalar 和 vector transform bounds 按 root shape 广播，不按 runtime row shape 广播。
5. 使用 BrainState grad 和 BrainTools Adam 完成一个端到端参数恢复示例，BrainCell 不提供
   Trainer wrapper。

测试同时覆盖 eager、JIT、grad、重复 rollout 和 full reset/re-init。

## P1：Owner 扩展

P0 稳定后按现有 storage owner 接入：

1. SynapseView 连续 `ParameterSpec` fields；
2. ConnectionView continuous weight；
3. Network manager 跨 Cell roots、bindings 和 materialization；
4. cable/Cell parameter owner 只有在字段角色与 runtime buffer 明确后再评估。

Connection weight 接入前必须先证明：

- ConnectionStore 到 runtime delivery block 存在 shape-preserving buffer；
- optimizer 更新不会触发 topology 或 delay queue 重建；
- Network cached run 读取当前 weight State，而不是 trace 外冻结值；
- target Cell ownership 和跨 Cell stable name 不发生碰撞。

delay、routing、source index、synapse ID 和 active mask 不进入连续 trainable tree。

## 后续独立设计

以下能力只有完成独立 API 讨论后才加入 [API](api.md)：

- unit-aware batch/data adapter；
- loss component 协议；
- candidate tree 和 multi-start helper；
- history、result 和 success aggregation；
- convergence/plateau/profile diagnostics；
- checkpoint/resume 和 fitted-model export；
- backend-aware timing/memory instrumentation。

[References](design-overview.md#references) 提供需求证据，但其中的实验结构和候选接口
不自动成为公共类型。

## 错误与事务测试

至少覆盖：

- empty/multi-owner selection；
- unknown/non-parameter target；
- post-init registration；
- overlap binding；
- duplicate/conflicting stable names；
- incompatible units、shape、dtype 和 transform bounds；
- non-finite root candidate；
- nested direct source missing initial；
- invalid function signature、fixed output shape change、non-JAX callable；
- ParameterSet incomplete/extra tree keys；
- manager materialize before runtime exists；
- one binding failure while other targets already have valid materialized values。

所有验证需证明失败是原子的，而不只是抛出异常。

## 文档与发布门槛

- API doc 的签名、默认值和异常与实现一致；
- architecture 的 ownership 和 lifecycle 由集成测试覆盖；
- `braincell.trainable` 有最小公共 export，不额外发布同名 `optim` 或 `training` 公共模块；
- 示例直接显示 BrainState grad/ParamState 和 BrainTools optimizer；
- 未完成 owner 和后续协议明确标为 deferred，不发布空壳类型；
- 依赖版本范围和 CI 环境均验证 transform、Quantity 和 graph state 行为。

## 冻结不变量

- root 才是 ParamState；runtime target 是物理 materialized value。
- Cell TrainableManager 是 Cell-owned bindings 的唯一状态 owner。
- Network manager 聚合但不复制 Cell roots。
- direct、scale、parameterized 共用 binding/materialization 主链。
- reset dynamic state 不回滚 roots。
- 不从 final target bounds 反推 latent bounds。
- 不平均不一致的 grouped direct initial values。
- 不序列化 arbitrary Python callable。
- 不在 BrainCell 中重复实现 optimizer 或通用搜索算法。
