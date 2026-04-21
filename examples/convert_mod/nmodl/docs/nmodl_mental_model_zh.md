# 从零理解 NMODL：以 `hh.mod` 为主线看 NEURON 8.2.x、9.0 与 Arbor

这篇说明不是语法手册，而是一份“先建立心智模型，再看细节”的学习笔记。目标是把你最容易混淆的几件事拆开：

- `mod` 文件到底在描述什么
- 各个块为什么存在
- `PARAMETER`、`ASSIGNED`、`STATE` 到底怎么区分
- `INITIAL`、`BREAKPOINT`、`DERIVATIVE` 在运行时分别什么时候执行
- `NEURON 8.2.x`、`9.0`、`Arbor` 的差异到底是“文档写法不同”，还是“语义真的不同”

## 1. 先只记一个总图

把一个 `mod` 文件先看成四层：

1. 对外接口
   - 这个机制叫什么
   - 它是密度机制还是点机制
   - 它读写哪些离子、电流
2. 变量分类
   - 哪些量由用户设定
   - 哪些量是运行时直接算出来的
   - 哪些量是需要积分更新的状态
3. 初始化
   - 仿真开始时，状态先设成什么
4. 每步更新
   - 每个时间步先算什么、再积分什么、最后输出什么电流

对应到 NMODL 里，最常见的是：

- `NEURON` / `UNITS`
- `PARAMETER` / `ASSIGNED` / `STATE`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- 可选的 `PROCEDURE` / `FUNCTION`
- 点机制额外常见 `NET_RECEIVE`

如果你先把这些块理解成“职责分工”，就不会觉得它们像一堆互不相关的语法。

## 2. 从最简单的开始：一个没有状态的 leak 机制

很多人一上来就看 `hh.mod`，容易被 `m/h/n`、`rates()`、`SOLVE` 吓到。其实最小的通道可以没有 `STATE`。

```mod
NEURON {
    SUFFIX pas_simple
    NONSPECIFIC_CURRENT i
    RANGE g, e
}

PARAMETER {
    g = 0.001 (S/cm2)
    e = -65 (mV)
}

ASSIGNED {
    v (mV)
    i (mA/cm2)
}

BREAKPOINT {
    i = g * (v - e)
}
```

这个例子只做一件事：按欧姆律计算漏电流。

先按块理解：

- `NEURON`
  - `SUFFIX pas_simple` 表示这是一个密度机制，插入 section 后名字是 `pas_simple`
  - `NONSPECIFIC_CURRENT i` 表示这个机制贡献一个非特异跨膜电流 `i`
  - `RANGE g, e` 表示每个 segment 都可以有自己的 `g` 和 `e`
- `PARAMETER`
  - `g` 和 `e` 由用户提供或修改
- `ASSIGNED`
  - `v` 是膜电位，来自模拟器
  - `i` 是当前时刻算出来的电流，不需要积分
- `BREAKPOINT`
  - 每个时间步评估当前电流

这里没有 `STATE`，因为：

- `g` 和 `e` 不是动态演化的状态
- `i` 只是当前时刻由 `v` 直接计算出来
- 没有任何变量满足“需要根据微分方程随时间更新”

这就是你理解 NMODL 的第一个关键：

`STATE` 不是“重要变量”，而是“要被积分器推进的变量”。

## 3. 第二步：为什么 `hh.mod` 需要更多块

仓库里的示例文件 [`hh.mod`](/home/swl/braincell/examples/convert_mod/nmodl/mod_files/hh.mod) 是一个简化版 Hodgkin-Huxley 机制。它比 leak 多出来的核心复杂度只有一个：

`m`、`h`、`n` 不是常数，而是随时间变化的门控变量。

### 3.1 先看完整结构

```mod
TITLE Simplified Hodgkin-Huxley example for AST and IR extraction

NEURON {
    SUFFIX hh
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    RANGE gnabar, gkbar, gl, el
    GLOBAL minf, hinf, ninf, mtau, htau, ntau
    NONSPECIFIC_CURRENT il
}

UNITS {
    (S) = (siemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
}

PARAMETER {
    gnabar = 0.12 (S/cm2)
    gkbar = 0.036 (S/cm2)
    gl = 0.0003 (S/cm2)
    el = -54.3 (mV)
}

STATE {
    m
    h
    n
}

ASSIGNED {
    v (mV)
    ena (mV)
    ek (mV)
    ina (mA/cm2)
    ik (mA/cm2)
    il (mA/cm2)
    minf
    hinf
    ninf
    mtau (ms)
    htau (ms)
    ntau (ms)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
    ik = gkbar * n*n*n*n * (v - ek)
    il = gl * (v - el)
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v + 40) / 10))
    hinf = 1 / (1 + exp((v + 62) / 10))
    ninf = 1 / (1 + exp(-(v + 53) / 16))
    mtau = 0.2
    htau = 1
    ntau = 1
}
```

### 3.2 每个块到底在干什么

#### `NEURON`

这是“对模拟器声明接口”的地方。

- `SUFFIX hh`
  - 说明这是一个密度机制
- `USEION na READ ena WRITE ina`
  - 这个机制读取钠反转电位 `ena`
  - 写出钠电流 `ina`
- `USEION k READ ek WRITE ik`
  - 读取钾反转电位 `ek`
  - 写出钾电流 `ik`
- `RANGE gnabar, gkbar, gl, el`
  - 这些量可以沿空间变化
- `GLOBAL minf, hinf, ninf, mtau, htau, ntau`
  - 这些量在 NEURON 里按机制共享
- `NONSPECIFIC_CURRENT il`
  - `il` 不是某个特定离子的电流

理解口诀：

- `SUFFIX`/`POINT_PROCESS` 决定“机制类型”
- `USEION`/`NONSPECIFIC_CURRENT` 决定“这个机制与电流/离子怎么接线”
- `RANGE`/`GLOBAL` 决定“变量对外暴露时按段存还是按机制共享”

#### `UNITS`

这是单位声明和单位检查辅助块。

在 NEURON 中，它主要用于可读性和单位一致性检查。  
在 Arbor 中，单位会被解析，但大部分换算语义并不执行，所以 Arbor 里更接近“注释性说明 + 约定单位”。

#### `PARAMETER`

这里放“用户设定的参数”：

- `gnabar`
- `gkbar`
- `gl`
- `el`

它们是模型的旋钮，而不是状态。

一个好用的判断法：

- 想象你在做参数扫描
- 会主动调的量，通常优先考虑 `PARAMETER`

#### `STATE`

这里放“会被积分器推进的动态变量”。

在 HH 里就是：

- `m`
- `h`
- `n`

为什么它们是 `STATE`？

因为你不是直接用代数式每步重算 `m/h/n`，而是要根据微分方程：

```text
m' = ...
h' = ...
n' = ...
```

让它们从旧值更新到新值。

这是第二个关键：

- `PARAMETER` 是你给它值
- `ASSIGNED` 是你算它值
- `STATE` 是求解器推进它的值

#### `ASSIGNED`

这里放“运行时通过赋值计算出来，但不需要积分”的量。

在这个 `hh.mod` 里包括：

- 来自模拟器的量：`v`、`ena`、`ek`
- 输出电流：`ina`、`ik`、`il`
- 中间结果：`minf`、`hinf`、`ninf`、`mtau`、`htau`、`ntau`

为什么 `minf` 和 `mtau` 不是 `STATE`？

因为它们是当前电压 `v` 的函数。只要知道当前 `v`，就能直接算出来，不需要保存为“随时间积分推进的未知量”。

#### `INITIAL`

这个块在仿真初始化时执行一次。

这里做了两件事：

```mod
INITIAL {
    rates(v)
    m = minf
    h = hinf
    n = ninf
}
```

先根据初始电压算出稳态门控值，再把状态初始化到这些稳态值。

这一步很重要，因为如果你不初始化：

- `m/h/n` 可能默认是 0
- 模型刚开始会出现不合理的瞬态

理解口诀：

- `INITIAL` 负责“仿真开始时把系统摆到合理初始状态”

#### `BREAKPOINT`

这是最容易被误解的块。你可以把它先理解成：

“每个时间步，模拟器需要这个机制给出当前电流时，会执行这里。”

```mod
BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m * h * (v - ena)
    ik = gkbar * n*n*n*n * (v - ek)
    il = gl * (v - el)
}
```

这里分两部分：

- `SOLVE states METHOD cnexp`
  - 调用下面那个 `DERIVATIVE states` 块，用 `cnexp` 方法推进 `m/h/n`
- 三个电流表达式
  - 用当前状态和电压计算 `ina/ik/il`

初学者最容易犯的错是把 `BREAKPOINT` 当成“随便写所有逻辑的主函数”。  
更安全的理解是：

- `BREAKPOINT` 主要负责“当前时刻的可观测量”，尤其是电流
- 真正定义状态怎么变的，是 `DERIVATIVE` 或 `KINETIC`

#### `DERIVATIVE`

这个块写状态的微分方程。

```mod
DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    n' = (ninf - n) / ntau
}
```

这里它表达的是经典 HH 形式：

```text
x' = (x_inf(v) - x) / tau_x(v)
```

也就是说：

- 先根据 `v` 算出当前目标稳态 `x_inf`
- 再按时间常数 `tau_x` 向那个目标靠近

所以：

- `DERIVATIVE` 定义“状态如何演化”
- `SOLVE` 决定“用什么数值方法去演化”

#### `PROCEDURE rates(...)`

这个块不是机制生命周期的一部分，而是“辅助计算子程序”。

它只是把重复逻辑提出来：

- 给定 `v`
- 计算 `minf/hinf/ninf`
- 计算 `mtau/htau/ntau`

这样 `INITIAL` 和 `DERIVATIVE` 都可以复用它。

所以第三个关键是：

- `INITIAL`、`BREAKPOINT`、`DERIVATIVE` 是“调度块”
- `PROCEDURE`/`FUNCTION` 是“辅助计算块”

## 4. 把 `hh.mod` 再压缩成一句话

如果你只能记住一句，那么记这个：

`hh.mod` 做的是：先声明它会产生钠电流、钾电流和漏电流；再把门控变量 `m/h/n` 作为状态初始化；然后每步根据电压推进这些状态，并据此计算三个电流。

一旦你这样理解，块就不会显得零散。

## 5. 运行时顺序怎么理解

对初学者，先用这个简化顺序就够了：

1. `finitialize()` 时执行 `INITIAL`
2. 每个时间步执行 `BREAKPOINT`
3. `BREAKPOINT` 里的 `SOLVE` 会调用对应的 `DERIVATIVE` 或 `KINETIC`
4. 点机制收到事件时执行 `NET_RECEIVE`

更精细地说，NEURON 在一个时间步里对 `BREAKPOINT` 的调用时机和次数是有数值算法背景的，尤其与 `fadvance()`、`secondorder`、电导求导有关。对学习阶段，不必先钻这层实现细节，只需记住：

- `BREAKPOINT` 不是“只会执行一次的普通函数”
- 不要在 `BREAKPOINT` 里写依赖调用次数的副作用逻辑
- 电流和电导表达式应尽量保持“给当前状态和电压就能纯计算”的风格

## 6. 为什么点突触会多一个 `NET_RECEIVE`

HH 通道没有事件输入，所以不需要 `NET_RECEIVE`。  
突触常常是“收到 spike 事件后，导通度瞬间跳一下，再指数衰减”，因此会多一个事件块。

最小指数突触可以写成：

```mod
NEURON {
    POINT_PROCESS expsyn_simple
    RANGE tau, e
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    tau = 2 (ms)
    e = 0 (mV)
}

STATE {
    g (uS)
}

ASSIGNED {
    v (mV)
    i (nA)
}

INITIAL {
    g = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g * (v - e)
}

DERIVATIVE state {
    g' = -g / tau
}

NET_RECEIVE(weight) {
    g = g + weight
}
```

理解方式：

- `POINT_PROCESS` 表示它不是沿 cable 分布的密度机制，而是一个放在某个位置的点对象
- `g` 是状态，平时指数衰减
- 每收到一个事件，`NET_RECEIVE(weight)` 把 `g` 瞬间加一截

所以：

- 通道机制常见 `BREAKPOINT + DERIVATIVE`
- 事件突触常见 `BREAKPOINT + DERIVATIVE + NET_RECEIVE`

## 7. `NEURON 8.2.x` 和 `9.0` 到底哪里不一样

如果你现在很混乱，先抓住一句最重要的话：

`PARAMETER`、`ASSIGNED`、`STATE`、`INITIAL`、`BREAKPOINT`、`DERIVATIVE` 这些核心块的基本学习语义，在 NEURON 8.2.x 和 9.0 之间没有根本变化。

真正容易让人误以为“语法变了”的，主要是下面几件事。

### 7.1 变化最大的地方：9.0 开始 NMODL 生成 C++

从 `NEURON >= 9.0` 开始，MOD 文件翻译后生成的是 C++，不再是旧的 C。  
官方迁移说明也明确说了：

- 数值方法本身没有因此改变
- 大多数普通 MOD 文件不需要改
- 真正容易出问题的是含 `VERBATIM` 的文件

所以如果你看的 HH 风格教学例子没有 `VERBATIM`，通常不需要把 8.2.x 和 9.0 当成两门语言。

### 7.2 文档位置和措辞变了，但基础概念没变

你会看到：

- 8.2.x 文档在 `python/modelspec/programmatic/mechanisms`
- 9.0 文档拆到了 `nmodl/language`

而且很多老文档带明显历史包袱，比如：

- `PARAMETER` 和 `ASSIGNED` 被说成“几乎是同义”
- `LOCAL`、`POINTER`、`VERBATIM` 这些老机制扩展在新文档里被更谨慎地讨论

这些会让人以为“9.0 的块语义全变了”，但对你学习基础块不是主要矛盾。

### 7.3 对初学者最实用的版本判断

可以直接这样记：

- 如果你写的是普通 HH/ohmic/channel/synapse 机制
  - 学习语义按同一套理解即可
- 如果你的 MOD 文件用了 `VERBATIM`
  - 必须额外检查 9.0 的 C++ 兼容
- 如果用了很老、很技巧化的写法
  - 要更谨慎地核对文档和编译器行为

## 8. Arbor 什么时候“相同”，什么时候“不同”

Arbor 最适合这样理解：

- 对常见 HH 风格机制，它和 NEURON 足够像，足以共用学习框架
- 但 Arbor 不是“完全兼容的 NEURON 替身”
- 它支持一个更严格的子集，并加入少量 Arbor 自己的扩展

### 8.1 对学习基础块时，哪些几乎可以当成一样

下面这些在学习层面可以近似按同一职责理解：

- `NEURON { SUFFIX ... }`
- `PARAMETER`
- `ASSIGNED`
- `STATE`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- `POINT_PROCESS`
- `NET_RECEIVE`

也就是说，如果你是在理解：

- leak 机制
- HH 通道
- 指数突触

那么 Arbor 和 NEURON 的心智模型基本一致。

### 8.2 真正需要单独记住的 Arbor 差异

#### 单位

- NEURON 的 `UNITS` 真参与单位检查和单位数据库
- Arbor 只解析单位，大部分换算语义忽略

所以：

- 在 Arbor 里，`UNITS` 更接近“给人看 + 约束自己按约定单位写”

#### 特殊变量作用域

Arbor 要求特殊变量更显式：

- `v`、`diam`、`area`、`celsius` 要放进 `PARAMETER` 才可访问
- 在 `PROCEDURE` / `FUNCTION` 里，像 `v` 这样的非 `LOCAL` 变量要显式传参

这和很多 NEURON 旧例子“直接在过程里用 `v`”的写法不一样。

#### 离子语义更严格

Arbor 对 `USEION` 的约束比 NEURON 更明确：

- `Xi/Xo/eX/iX` 会自动暴露
- 不应再手动放进 `PARAMETER`、`ASSIGNED`、`CONSTANT`
- 如果机制要写 `Xi/Xo`，它们应作为 `STATE` 并在 `INITIAL` 里初始化

#### 不支持的一批老特性

Arbor 官方明确列出不支持：

- `TABLE`
- `VERBATIM`
- `INDEPENDENT`
- 块外 `LOCAL`
- loops / arrays / pointers
- `derivimplicit`

这就是为什么“一个老 NEURON 模型能编译”不等于“它能直接拿去 Arbor 编译”。

### 8.3 Arbor 还有一些 NEURON 没有的扩展

Arbor 额外支持例如：

- `POST_EVENT`
- `VOLTAGE_PROCESS`
- `WHITE_NOISE`
- `v_peer`

所以 Arbor 既不是“缩水版 NEURON”，也不是“完全同语义实现”，而是“常用核心相似、边角语义更严格、并带少量自家扩展”的方言。

## 9. 一个实用的三列表

| 主题 | NEURON 8.2.x | NEURON 9.0 | Arbor |
|---|---|---|---|
| `PARAMETER/ASSIGNED/STATE` 学习语义 | 与传统教程一致 | 基本一致 | 基本一致，但特殊变量/离子变量更严格 |
| `INITIAL/BREAKPOINT/DERIVATIVE` 学习语义 | 基本一致 | 基本一致 | 基本一致 |
| `VERBATIM` | 可用 | 需要注意 C++ 兼容 | 不支持 |
| `TABLE` | 可用 | 可用 | 不支持 |
| `POINTER` | 可用 | 可用，但更要小心兼容与实现细节 | 不支持 |
| `UNITS` | 真实参与单位体系 | 真实参与单位体系 | 解析但大多不做换算 |
| `PROCEDURE/FUNCTION` 里直接用 `v` | 老例子常见 | 仍常见 | 通常应显式传参 |
| `USEION` 自动暴露离子变量后的约束 | 有历史兼容逻辑 | 有历史兼容逻辑 | 更严格 |
| `hh` / `pas` / `expsyn` 这种基础机制 | 可直接理解 | 可直接理解 | 也可直接理解 |

## 10. 你现在最应该怎么学

如果你现在很懵，不建议从“所有块的全集”开始，而建议按这个顺序：

1. 先吃透最小 leak
   - 为什么没有 `STATE`
   - 为什么只有 `BREAKPOINT`
2. 再吃透 `hh.mod`
   - 为什么 `m/h/n` 是 `STATE`
   - 为什么 `minf/mtau` 是 `ASSIGNED`
   - 为什么 `INITIAL` 要先把状态设到稳态
   - 为什么 `BREAKPOINT` 里要 `SOLVE`
3. 再看指数突触
   - 为什么点机制用 `POINT_PROCESS`
   - 为什么会多出 `NET_RECEIVE`
4. 最后再去碰高级块
   - `KINETIC`
   - `POINTER`
   - `TABLE`
   - `VERBATIM`
   - `BEFORE/AFTER`
   - `WATCH`

也就是说，你现在不需要“把所有块全背下来”，你只需要先把下面这张最小表背住：

| 块 | 你可以先这样理解 |
|---|---|
| `NEURON` | 对模拟器声明接口 |
| `PARAMETER` | 用户给定的参数 |
| `ASSIGNED` | 直接算出来的量 |
| `STATE` | 需要积分推进的状态 |
| `INITIAL` | 初始化时执行一次 |
| `BREAKPOINT` | 每步计算当前输出，尤其是电流 |
| `DERIVATIVE` | 定义状态怎么变 |
| `PROCEDURE/FUNCTION` | 辅助计算 |
| `NET_RECEIVE` | 收到事件时执行 |

## 11. 对这个仓库里的 `hh.mod` 再补一句

[`hh.mod`](/home/swl/braincell/examples/convert_mod/nmodl/mod_files/hh.mod) 是一个适合教学和 AST/IR 提取的简化版 HH。

它不是为了完整复刻 NEURON 内置 `hh` 的所有历史细节，而是为了把最核心的块关系保留下来：

- 多离子 `USEION`
- `STATE m/h/n`
- `INITIAL`
- `BREAKPOINT`
- `DERIVATIVE`
- `PROCEDURE rates`

正因为它是简化版，拿来建立第一层理解比直接啃复杂旧模型更合适。

## 12. 参考资料

- 仓库内示例 `hh.mod`：
  - [`examples/convert_mod/nmodl/mod_files/hh.mod`](/home/swl/braincell/examples/convert_mod/nmodl/mod_files/hh.mod)
- Arbor NMODL 规则：
  - <https://docs.arbor-sim.org/en/latest/fileformat/nmodl.html>
- Arbor NMODL 教程：
  - <https://docs.arbor-sim.org/en/latest/tutorial/nmodl.html>
- NEURON 9.0 NMODL 文档：
  - <https://nrn.readthedocs.io/en/9.0.1/nmodl/language/nmodl.html>
- NEURON 8.2.x NMODL 文档：
  - <https://nrn.readthedocs.io/en/8.2.7/python/modelspec/programmatic/mechanisms/nmodl.html>
- NEURON 9.0 的 MOD 迁移说明：
  - <https://nrn.readthedocs.io/en/latest/guide/porting_mechanisms_to_cpp.html>

## 13. 一句话结论

先不要把 NMODL 当“很多块的语法题”，而要把它当：

“用几个固定块，把一个机制的接口、状态、初始化和每步演化写清楚。”

当你先吃透 leak、再吃透 `hh.mod`、再看 `NET_RECEIVE`，大部分困惑都会自然消掉。
