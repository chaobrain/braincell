# Network Module Relocation

把 `braincell/` 包根下的五个模块搬到它们真正所属的包，并顺手清掉这次改动
覆盖到的 AGENTS.md 违规。持久化的分层约束见
[`docs/design/network/module-layout.md`](../design/network/module-layout.md)。

## 动机

包根混住了三个不同层次的模块：

| 文件 | 真实层次 |
|---|---|
| `_synapse_schema.py` | leaf，无任何 `braincell` import |
| `event.py` | leaf，但同时装了*契约*和*事件源*两种东西 |
| `recording.py` | leaf，跨 `_multi_compartment` 与 `network` |
| `_connection_sampling.py` | 在 `_multi_compartment` 之上 |
| `connection.py` | 在 `_multi_compartment` 之上、`network` 之下 |

`event.py` 尤其名不副实：第 41–88 行的 `EventInput` 家族是*目标机制的声明契约*，
调用方是栈底的 `_base_channel` 和 `_compute.state`；第 90 行往后是网络层的
事件源。两者互不引用，被同一个文件名捆在一起纯属历史遗留。

## 决策

1. **布局**：网络相关的四个模块进 `braincell/network/`，两个契约模块进
   `braincell/mech/`，`mech` 因此成为纯 leaf。
2. **打破环**：`network/__init__.py` 只 eager import leaf 的 `.core`，
   `Network` / `NetworkConnections` / `ConnectionBlock` 走 PEP 562 `__getattr__`。
3. **兼容性**：干净切断。`braincell.connection` 从 `braincell.__all__` 移除，
   所有调用点改写。版本仍是 0.1.0，且没有任何 `docs/apis/*.rst` 引用这些路径。
4. **命名**：`_connection_sampling.py` → `network/pairing.py`。
5. **顺带修复**：补 7 个缺失的 license header，修 3 个孤儿测试文件名。

### 搬迁表

| 原路径 | 新路径 |
|---|---|
| `braincell/connection.py` | `braincell/network/connection.py` |
| `braincell/_connection_sampling.py` | `braincell/network/pairing.py` |
| `braincell/recording.py` | `braincell/network/recording.py` |
| `braincell/event.py` (L90+) | `braincell/network/event.py` |
| `braincell/event.py` (L41–88) | `braincell/mech/_event_contract.py` |
| `braincell/_synapse_schema.py` | `braincell/mech/_synapse_schema.py` |

包根的非测试模块从 12 个降到 7 个：`__init__`、`_base`、`_base_channel`、
`_base_ion`、`_misc`、`_typing`、`_testing`、`_version`。

## 被打破的三个环

搬迁把三处顶层 import 变成了环，全部由轻量 `network/__init__.py` 解决：

| 站点 | 环路径 |
|---|---|
| `mech/__init__.py:70`（`NetStim` re-export） | → `network/__init__` → `engine` → `braincell.mech` |
| `_multi_compartment/cell.py:97,98` | → `network/__init__` → `connection` → `_multi_compartment/__init__` → `cell`（初始化中） |
| `_multi_compartment/run.py:33` | 同上 |

`TYPE_CHECKING` 救不了：`cell.py:1328,1332,1386` 和 `run.py:146,316` 在运行期
真的构造 `EventOutputCollection`、`_CellSpikeSource`、`compile_recording`、
`SampleBlock`。

实施中发现第四处设计阶段漏掉的站点：`_compute/layouts.py:61` 从
`braincell.mech` import `NetStim`（而非从 `braincell.event`），所以最初的
grep 没有命中。它改为直接 import `braincell.network.event`，无环。

`braincell.mech.NetStim` 这个 re-export 被删除；两个 example 调用点改用
`braincell.NetStim`。

## 测试

三个孤儿测试文件按 AGENTS.md 规则 10 归位：

- `network/population_test.py` → `network/core_test.py`
- `network/runtime_test.py` → 按类拆到 `core_test.py`（`PopulationTest` 与
  原 `core_test.py` 重名，改为 `PopulationRuntimeTest`）、`lowering_test.py`、
  `engine_test.py`；共享 fixture 进 `network/_testing.py`
- `filter/filter_sampling_test.py` → `filter/_sampling_test.py`

新增覆盖此前完全没有直接测试的两个契约模块：
`mech/_event_contract_test.py`（aggregation 白名单、`validate_payload` 的
单位校验、jit 下不强制 host materialization）与
`mech/_synapse_schema_test.py`（量纲/空数组/非有限值/非数值分支、`positive`）。

`network/__init___test.py` 守住惰性 `__init__` 不变式与 `mech` 的 leaf 性质。

## 验证

`pytest braincell/` — 2537 passed / 19 skipped（与改动前基线一致），加上新增
的 34 个测试。
