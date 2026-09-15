# H200 优化前后对比

本页比较 H200 上同一 HH scaling workload 的优化前后结果。优化前数据来自
2026-09-10 coarse scan，优化后数据来自 2026-09-12 optimized scan。

## 比较范围

共同配置为 `C = 1, 21, 41, 81`，每个 `C` 使用 `Nθ = C, 2C, 3C`，
`B=S=16`、`T=1600`、`dt=0.025 ms`、`float64`、staggered solver 和
recursive backsub。两批数据在不同日期运行，未锁定 GPU 时钟或 CPU affinity，
因此前后差异用于说明本 workload 下的工程收益，不作为跨环境基准。

## 优化内容

优化后包含批量 `materialize`、identity gather 消除和 rollout 入口
materialization。相关实现结论见当前 optimizer architecture；具体 compile、steady
runtime、temporary memory 和 correctness 数值见下方统一结果页中的历史对照表。

- 优化前：[H200 coarse scan](before/hh-crossover-h200-20260910.md)
- 优化前补测：[H200 C=81 supplement](before/hh-crossover-h200-20260910-supplement.md)
- 优化后：[H200 完整扫描与历史对照](after/optimized.md)：C=1/21/41/51/61/81，四批互不重叠的配置合并为一个结果入口。

## 判读

优化前后共同点可用于比较 compile time、steady runtime 和 XLA temporary memory。
优化后新增的 `C=51/61` 只用于缩小 crossover 搜索范围，没有对应的优化前测量，
因此不做伪造的前后配对。

优化前后的图分别保存在 [legacy coarse scan](figures/before/) 与
[optimized](figures/after/)，每套图展示对应阶段内的 BPTT/RTRL。
优化后的完整扫描图、runtime ratio 与 memory/compile 图也直接展示在 [结果总结](after/optimized.md)。图表只连接实际测量点，
不把线段解释为未测量位置的 crossover 证据。
