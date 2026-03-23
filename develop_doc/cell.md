# Cell 前端层规范（重写版）

## 目标与边界

`cell` 当前只做前端建模与可检查数据，不进入计算执行层。

- 入口固定为：`Cell(morpho, cv_policy=CVPolicy())`
- 自动离散得到 `CV` 集合
- 支持 `paint` / `place` 规则积累与查询
- 支持懒重建（改了 `cv_policy`、`paint`、`place` 后置脏）

本层明确不做：

- 不提供 `run()`
- 不组装 `HHTypedNeuron`
- 不执行积分与矩阵求解

---

## 核心类列表（必须保留的主干）

### `class Cell`

前端主对象，持有：

- 形态快照：`morpho`
- 离散策略：`cv_policy`
- 离散结果：`cvs`
- 原始规则：`paint_rules`、`place_rules`
- 缓存状态：`dirty`

对外接口：

- `paint(region, *mechanisms)`
- `place(locset, *point_mech)`
- `n_cv`
- `cv(i) -> CV`
- `summary()`

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
- `lateral_area`
- `r_axial_prox`
- `r_axial_dist`
- `r_axial`

机制容器：

- `density_mech`（按机制类型索引）
- `point_mech`（按位点存放，本规范只允许 `mid`）

### `class CVPolicy`

离散策略对象。

- 默认：每个 branch 1 个 CV
- 支持 `mode="cv_per_branch"`：每个 branch 使用统一 `cv_per_branch`
- 支持 `mode="max_cv_len"`：按 `max_cv_len` 计算每个 branch 的 CV 数量
  - 规则：`n = max(1, ceil(branch_total_length / max_cv_len))`
  - 语义：保证每段 CV 长度不超过 `max_cv_len`（浮点容差内）

### `class PaintRule`

保存原始 `paint` 声明：

- 哪个 `region`
- paint 了哪些机制/属性

### `class PlaceRule`

保存原始 `place` 声明：

- 哪个 `locset`
- 放了哪些点机制对象

---

## 当前代码组织（实现约定）

- `cell/cell.py`：`Cell` 对外接口与重建编排（总控）
- `cell/cv.py`：`CV`、`CVPolicy` 与 `CV` 组装逻辑
- `cell/cv_geo.py`：`CVGeo` + `CVFrustum`，负责离散、几何、拓扑映射
- `cell/cv_mech.py`：`CVMech` + `PaintRule/PlaceRule`，负责规则归一化与应用

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
- `lateral_area = CV 内所有圆台侧面积之和`

`Branch` 层建议补齐：

- `Branch.lateral_areas()`
- `Branch.total_lateral_area()`

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

## 查询与懒重建

`Cell` 层查询的是抽象声明与离散结果：

- 原始规则：`paint_rules`、`place_rules`
- 离散结果：`cv(i)` 的属性与机制视图

懒重建触发条件：

- 修改 `cv_policy`
- 新增或修改 `paint`
- 新增或修改 `place`

触发后行为：

- 仅标记 `dirty`
- 下次查询 `n_cv/cv()/summary()` 时重建

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
