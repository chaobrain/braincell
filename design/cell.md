# Cell 前端层规范（重写版）

## 目标与边界

`cell` 现在既做前端建模，也作为运行时主对象。

- 入口固定为：`Cell(morpho, cv_policy=CVPerBranch())`
- 自动离散得到 `CV` 集合
- 支持 `paint` / `place` 规则积累与查询
- 支持懒重建（改了 `cv_policy`、`paint`、`place` 后置脏）

本层明确不做：

- 不单独再暴露 `MultiCompartment` / `CellExecution` 这类中间壳对象
- 不要求用户手动再包装一层执行对象

---

## 核心类列表（必须保留的主干）

### `class Cell`

主对象，持有：

- 形态快照：`morpho`
- 离散策略：`cv_policy`
- 离散结果：`cvs`
- 原始规则：`paint_rules`、`place_rules`
- 编译缓存：layout / runtime node / ion / state buffer
- 运行时状态：`V`、`spike`

对外接口：

- `paint(region, *mechanisms)`
- `place(locset, *point_mech)`
- `n_cv`
- `cvs[i] -> CV`
- `init_state()`
- `reset_state()`
- `pre_integral() / compute_derivative() / post_integral() / update()`
- `layouts`
- `get_state()` / `set_state()`
- `get_point_state()` / `get_cv_state()`
- `get_ion()`

### `class CV`

一个 CV 对应一个 branch 区间：

- `region = (branch_id, prox, dist)`

拓扑信息：

- `parent_cv`
- `children_cv`

包含信息：

- 本 CV 覆盖的圆台切片列表（可能含截断插值点）
- 可选 `as_branch()`：返回一个由本 CV 切片构造出的新 `Branch`

中点属性（CV 的核心属性都定义在中点）：

- `cm`
- `ra`
- `v`
- `temp`

派生属性：

- `length`
- `area`
- `r_axial_prox`
- `r_axial_dist`
- `r_axial`

机制容器：

- `density_mech`（按机制类型索引）
- `point_mech`（按位点存放，可落在 `mid` 或边界 node，并在 `CV` 上保留位置角色）

### `class CVPolicy`

离散策略基类。

- 默认：`CVPerBranch()`，即每个 branch 1 个 CV
- 支持 `CVPerBranch(cv_per_branch=...)`：每个 branch 使用统一 `cv_per_branch`
- 支持 `MaxCVLen(max_cv_len=..., keep_odd=True)`：按 `max_cv_len` 计算每个 branch 的 CV 数量
  - 规则：先算 `n = max(1, ceil(branch_total_length / max_cv_len))`
  - 若 `keep_odd=True` 且 `n` 为偶数，则提升为 `n + 1`
  - 语义：默认对齐 NEURON 风格偏好奇数分段；若要严格长度上限可用 `keep_odd=False`
- 支持 `DLambda(d_lambda=..., frequency=100 * u.Hz, keep_odd=True)`：按 branch 级电长度决定 CV 数量
  - `d_lambda` 必须显式给出
  - `frequency` 默认 `100 Hz`
  - `keep_odd=True` 时，自动分段数若为偶数则提升到下一个奇数
  - `Ra/cm` 不从 policy 参数读取，而是从默认 cable 和 `paint(CableProperties)` 推导
  - 支持不同 branch 使用不同 `Ra/cm`
  - 若同一 branch 内 `Ra/cm` 不一致，则直接报错，要求统一该 branch 或改用其他 `cv_policy`
  - `v_rest` / `temperature` 不参与 `DLambda` 的 branch 内一致性检查

`PaintRule` / `PlaceRule` 仍存在于实现内部，用于保存标准化后的声明，但不再作为公开主接口导出或文档重点强调。

---

## 当前代码组织（实现约定）

- `cell/cell.py`：`Cell` 对外接口与重建编排（总控）
- `cell/cv.py`：`CV` 与 `CV` 组装逻辑
- `cell/cv_policy.py`：`CVPolicy` 基类和各类离散策略
- `cell/cv_geo.py`：`CVGeo` + `CVFrustum`，负责离散、几何、拓扑映射
- `cell/cv_mech.py`：内部规则与 CV 机制应用
- `_discretization/topology.py`：`NodeTree` 与 node graph builder
- `_compute/scheduling.py`：`NodeScheduling` 与 scheduler builder
- `cell/runtime.py`：内部编译 helper，负责 mechanism grouping、layout lowering、runtime node 与 state query

---

## CV 与离散点语义

每个 CV 有三个几何参考点：

- 近端点 `prox_node`
- 中点 `mid_node`
- 远端点 `dist_node`

说明：

- 这三个点用于后续矩阵组装视角
- 单树中 `n` 个 CV 共有 `2n+1` 个唯一计算点（边点可共享）
- 但本层状态与膜相关计算只在 `mid_node`

硬规则：

- `cm/ra/v/temp` 只在中点定义
- 膜电流、离子电流、电极电流、突触电流都归中点
- 边点只用于后续 KCL 矩阵的一行约束，不在本层存独立膜状态

---

## 几何与属性计算规则

### CV 几何切片

`CV` 覆盖的是 branch 的一个区间 `(prox, dist)`，需要得到该区间包含的圆台切片。

- `lengths + radii` 输入：在 `prox` / `dist` 最多插值两次 `radius`
- `xyz + radii` 输入：在 `prox` / `dist` 最多插值两次 `xyz` 与 `radius`

### 派生属性

- `length = (dist - prox) * branch.total_length`
- `area = CV 内所有圆台侧面积之和`

### 轴向电阻

`Branch` 本身没有 `ra`，所以轴向电阻在 `CV` 上计算。

- `r_axial_prox`：CV 长度前半段轴阻
- `r_axial_dist`：CV 长度后半段轴阻
- `r_axial = prox + dist`

---

## paint / place 语义

### paint

支持覆盖：

- 电缆属性：`cm`、`ra`、`v`、`temp`
- 密度机制：`ion`、`channel`

规则：

- 默认进来有 1 条全区域 `CableProperties` paint 规则
- 规则按声明顺序覆盖（后写覆盖前写）
- 同一 `region` 的 `CableProperties` 规则只保留最后一次声明
- `channel` 部分覆盖按膜面积比例缩放 `gmax`
- `ion` 不按体积/面积比例缩放，只做机制挂载与参数覆盖

### place

点机制吸附到 CV 中点。

映射规则：

- 点位于某 CV 内部：归该 CV
- 点恰好位于 CV 边界：归后一个 CV（右侧 CV）
  - 例：`[0,0.5]` 与 `[0.5,1]`，`x=0.5` 归后者
- 特殊规则：`(branch0, 1.0)` 仍归 `branch0` 的末端 CV
  - 即使 `branch0` 在该点连接了 child branch，也不跳到 child 的 CV

---

## 查询、编译与懒重建

`Cell` 层查询包含两类内容：

- 原始规则：`paint_rules`、`place_rules`
- 离散结果：`cvs[i]` 的属性与机制视图
- 编译结果：`layouts`、`get_state()`、`get_ion()` 等

懒重建触发条件：

- 修改 `cv_policy`
- 新增或修改 `paint`
- 新增或修改 `place`

触发后行为：

---

## 当前实现进展补充

- `Cell.update(I_ext)` 现在按旧版 `MultiCompartment` 语义接收外部总电流，默认单位是 `u.nA`
- 若传入的是总电流，`Cell` 内部会按 CV 面积换成电流密度后再参与膜方程
- 若传入的已经是电流密度 `u.nA / u.cm**2`，当前实现也兼容
- `staggered` 电压求解器的热路径已尽量改为 `jnp` / `u.math`，`numpy` 仅保留在静态拓扑与编译期数据整理中

- 仅标记 `dirty`
- 下次查询 `n_discretization/cvs` 时重建前端离散结果
- 下次 `init_state()` 时重新编译 runtime，并重建真实 state

运行时约束：

- `paint/place/cv_policy` 修改后，必须重新 `init_state()`
- `reset_state()` 可以在需要时隐式补编译
- `update()` / `compute_derivative()` 不会隐式编译，未初始化会直接报错

---

## 不兼容与迁移约束（必须执行）

本规范不兼容旧别名。旧名字要么改名，要么删除，不保留兼容壳。

需要退出主模型语义的旧对象：

- `CVRecord`
- `CellAssembly`
- `ControlVolume`
- `Discretizer`（可作为内部 helper，但不作为用户主接口）
- 旧 `DiscretizationPolicy` 命名（迁移到 `CVPolicy`）

文档标准以 `Cell + CV` 为唯一主干，其他均为内部实现细节。

---

## 当前进展

### 已完成

- `Cell` 已直接继承 `HHTypedNeuron`
- `Cell.init_state()` 负责 runtime 编译与 state 创建
- `Cell` 已直接接管 `reset_state/pre_integral/compute_derivative/post_integral/update`
- `CellRuntimeState` 退化为内部编译缓存，不再作为公开主接口
- `braincell.mech.Channel("IL")` 与 `braincell.mech.Channel("INa_HH1952")` 已能创建真实 runtime channel，并绑定到默认 `na/k/ca`
- `Cell` 已可直接查询 `layouts/get_state/get_point_state/get_cv_state/get_runtime_node/get_ion`
- `Cell.V` 的公开尺寸现在固定为 `n_cv`
- runtime channel / ion 仍按 `node_tree` 的 `n_point = n_cv + n_branch + 1` 创建
- `braincell.quad._voltage_solver.dhs_voltage_step()` 已改为从 `node_tree` 中的调度视图提取树结构
- `Cell(solver="staggered")` 已可直接走新的 node-tree DHS 电压求解

### 当前约束

- `density_mech` 当前仍先映射到 `cv_to_mid_node_id`
- 默认离子先固定为全局 `na/k/ca`
- `dense/sparse` 自动阈值切换接口已保留，但阈值策略尚未实装
- solver 内部会把 `n_cv` 的 `V` scatter 到 `n_point`，只在 midpoint row 写入/读回
- 当前 buffer 以 bridge view 为主，参数/状态先用 object array 承载，不做真实数值积分

### 下一步

- 已接入 `braincell.mech.Channel("IL", ...)` 这类简短 spec，`Cell.paint(...)` 可直接接受，旧 `DensityMechanism` 仍兼容
- 已打通 `Channel("IL", ...) -> CellRuntimeState.runtime_nodes -> braincell.channel.IL(size=(n_point,))`
- 当前 `set_state(layout_id, var_name, value)` 会同步更新 bridge buffer 和已注册的 `IL` runtime node 参数
- 已增加默认全局固定 ion 容器：`runtime.ions["na" | "k" | "ca"]`
- 已打通 `Channel("INa_HH1952", ...) -> runtime.get_runtime_node(layout_id) -> runtime.get_ion("na").channels["INa"]`
- 当前 `INa_HH1952` spec 里的 `temp` 会在 runtime bridge 中转换成底层构造参数 `phi`
- 当前仍只支持 dense runtime；`k/ca` 容器已创建但还未绑定新的真实 channel
- 下一步优先做更多 ion-bound channel 映射，或者设计 `cell.ion[...]` / `cell.soma.channel[...]` 这种更直接的 facade

### 公开面收口

- `braincell.cv`、`braincell.compute` 与顶层 `braincell` 只稳定导出：`Cell`、`CV`、`CVPolicy*`、`NodeTree`、`NodeScheduling`、`CellProfileReport`
- `PaintRule`、`PlaceRule`、`MechanismLayout`、`CellExecution` 不再作为稳定公开 API
