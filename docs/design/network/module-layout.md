# Network Module Layout

本文记录 `braincell/network/` 与 `braincell/mech/` 之间的分层约束，以及
`braincell/network/__init__.py` 为什么必须保持"轻"。改动这两个包之前先读这一页。

## 分层

```mermaid
flowchart TD
    MECH["`braincell.mech`<br/>纯 leaf：声明 + 契约<br/>_event_contract / _synapse_schema"]
    BASE["`_base_channel` / `_base_ion`<br/>Channel / Synapse 基类"]
    COMPUTE["`_compute`<br/>layouts / state / bindings"]
    MC["`_multi_compartment`<br/>Cell / run"]
    NETLEAF["`network.core`<br/>`network.event`<br/>`network.recording`<br/>三个 leaf 模块"]
    NETHEAVY["`network.connection` / `network.pairing`<br/>`network.engine` / `network.lowering` / `network.delivery`"]

    BASE --> MECH
    COMPUTE --> MECH
    COMPUTE --> NETLEAF
    MC --> MECH
    MC --> NETLEAF
    NETHEAVY --> MC
    NETHEAVY --> MECH
    NETHEAVY --> NETLEAF
```

关键事实：**箭头没有回边**。`mech` 不 import 任何其它 `braincell` 包，
`network` 的三个 leaf 模块顶层也不 import 任何 `braincell` 包。

## 为什么契约在 `mech` 而不在 `network`

`EventInput` / `NoEventInput` / `TriggerEventInput` / `ScalarEventInput` 是
*目标机制声明自己能消费什么事件* 的契约，不是事件源。它的调用方是
`_base_channel.Synapse`（类属性 `event_input`）、`_compute.state`（分配
runtime event buffer）和 `network.connection`（校验 `connect()` 的 payload）。

前两个位于栈底。若契约留在 `network` 包内，`_base_channel` 就要 import
`braincell.network.*`，而 Python 会先执行 `network/__init__.py` →
`connection.py` → `_multi_compartment.synapses` → 回到 `_base_channel`，
形成硬 `ImportError`。

`_synapse_schema`（`ParameterSpec` / `StateSpec` / `DerivedSpec` / `positive`）
同理：它被 `_base_channel` 和 `synapse.markov` 共同使用，放进
`braincell/synapse/` 会经 `synapse/__init__.py` → `markov.py` →
`braincell._base` 成环。

两者都是纯声明、无 runtime state，放在 `mech` 使 `mech` 成为一个真正的 leaf，
栈上任何一层都可以安全依赖它。

## `network/__init__.py` 必须保持轻

`_multi_compartment.cell`、`_multi_compartment.run` 和 `_compute.layouts` 在
**模块顶层** import `braincell.network.event` / `braincell.network.recording`。
Python 在导入任何子模块前先执行包的 `__init__`，因此：

> `network/__init__.py` 一旦在模块作用域 import `.connection`、`.engine` 或
> `.lowering`，就会在 `braincell._multi_compartment` 初始化过程中重新进入它，
> `import braincell` 直接失败。

所以 `__init__.py` 只 eager import leaf 的 `.core`，三个重名字
（`Network`、`NetworkConnections`、`ConnectionBlock`）通过 PEP 562
`__getattr__` 延迟解析。对外行为完全不变：`braincell.network.Network` 照常可用，
`__all__` 未变，只是解析时机推后到首次属性访问。

`braincell/network/__init___test.py` 守住这条不变式：一个 AST 检查禁止
`__init__.py` 在模块作用域 import 那三个模块，另一个 AST 检查禁止
`braincell/mech/` 下任何模块 import 其它 `braincell` 包。`import braincell`
本身是经验性守卫——上述任一环回归都会让它直接崩溃。

## 命名

| 模块 | 职责 |
|---|---|
| `network/core.py` | `Population`、`NetworkResult`、`NetworkRunResult` |
| `network/event.py` | 事件*源*：`EventSource`、`EventTable`、`EventSequence`、`NetStim`、`VoltageCrossingSource` |
| `network/recording.py` | `RecordingSpec`、`RecordingSchema`、`SampleBlock`、`EventSeries`、`observe` |
| `network/connection.py` | `connect()`、`ConnectionView`、`NetworkConnections` |
| `network/pairing.py` | 端点配对规则：`PairingSpec`、`independent`、`by_source`、`degree` … |
| `network/engine.py` | `Network` |
| `network/lowering.py` | 声明 → `ConnectionBlock` |
| `network/delivery.py` | delay queue / scatter |
| `mech/_event_contract.py` | 事件输入契约 |
| `mech/_synapse_schema.py` | runtime synapse 字段 schema |

`pairing.py` 取自模块自身导出的词汇（`PairingSpec` / `PairingContext` /
`materialize_pairing`）；它此前叫 `braincell/_connection_sampling.py`，
在一个本就以 connection 为主题的包里，`connection_` 前缀是冗余的。
