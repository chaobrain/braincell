# 编译时间调查：批量 materialize 的小规模前后对照

已确认逐行参数 materialize 是本次小规模 HH 编译开销的重要来源，并完成向量化修复。**C=5 时，BPTT 总编译时间从 33.36 s 降至 2.89 s（11.55 倍），RTRL 从 13.62 s 降至 2.12 s（6.42 倍）**。修改前后的 loss/梯度通过校验；未重新测量 C=21、41、81，因此不外推这些规模的具体加速倍数。

本轮墙钟 **5.55 分钟**，包含一次诊断导出修复造成的调度暂停。8 个 worker 都完成了规定的模型调用：5 个正常退出，3 个在模型调用完成后的 Jaxpr 统计导出阶段报错。后三者的计时样本、首次输出与 StableHLO 已保存，离线恢复统计并验证；原始错误记录保留。没有重试或增加模型调用。

## 编译分段

下表单位均为秒，箭头表示 baseline → optimized。总编译时间为 tracing、lowering、XLA compile 三段之和，阶段间保存文件不计入。两种路径复用同一次 trace/lower 的产物，没有为了诊断重复编译。

| C | Nx | Ntheta | 方法 | tracing | lowering | XLA compile | 总编译 | 编译加速倍数 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | 3 | BPTT | 1.19491 → 0.596007 | 0.17834 → 0.117407 | 3.62259 → 1.88678 | 4.99583 → 2.60019 | 1.92133 |
| 1 | 4 | 3 | RTRL | 0.955763 → 0.512189 | 0.155676 → 0.120382 | 2.09063 → 1.24808 | 3.20206 → 1.88065 | 1.70264 |
| 5 | 20 | 15 | BPTT | 2.66428 → 0.720622 | 0.40994 → 0.145885 | 30.287 → 2.02087 | 33.3612 → 2.88738 | 11.5542 |
| 5 | 20 | 15 | RTRL | 1.96126 → 0.59012 | 0.291295 → 0.146781 | 11.3698 → 1.3865 | 13.6223 → 2.1234 | 6.41532 |

在 C=5 的 BPTT 基线中，XLA compile 占总编译约 90.8%；优化后该段由 30.29 s 降至 2.02 s。RTRL 对应由 11.37 s 降至 1.39 s。tracing 也变快，但主要节省来自 XLA 编译阶段。尚未对单个 XLA pass 做内部计时，不能进一步断言是其中哪个优化 pass 主导。

## 图膨胀来源与修改

`braincell/trainable/_manager.py` 的 `TrainableManager.materialize()` 原先对每个 binding 的每个 population/CV 行单独调用 `.at[population, cv].set(value)`。全 HH、B=16、C=81、三个可训练通道时，一次 materialize 在源码层面对应 3888 次逐行写入；这不是该大模型整个编译图的实测 scatter 数量。

`RolloutGradientEngine.prepare()` 在 initializer 和 rollout 单步中都调用 materialize，求导会进一步变换这些写入操作。修复将同一 layout、同一字段的写入分组：部分覆盖采用向量 scatter，完整覆盖直接按行顺序重排/reshape。保留单位转换、dtype promotion、重复位置最后写入、跨 binding 合并和原子提交语义。神经元方程、BPTT/RTRL 算法及求解器均未修改。

以下是实际 C=5 梯度图的嵌套 Jaxpr 统计；每个静态子图按调用位置展开计数，scan 的 1600 次迭代不展开。这些是编译器优化前的图操作数量，不是运行时 GPU kernel 次数。

| C=5 方法 | 图操作数：前 → 后 | scatter：前 → 后 | StableHLO 文件字节：前 → 后 |
|---|---:|---:|---:|
| BPTT | 10282 → 1588 | 1442 → 5 | 4886555 → 2430395 |
| RTRL | 8799 → 2307 | 1452 → 12 | 3562532 → 2400423 |

在不运行神经元的局部 fixture 中，B=2、单字段完整覆盖时，C=1 的 materialize 顶层操作由 13 减至 1、scatter 由 2 减至 0；C=5 对应 61 → 1、10 → 0。真实 HH 对照进一步证明这项图结构改动带来了编译时间下降，而不只是缩小源码行数。

## 运行时间与原始样本

每个 worker 只有首次 1 次、额外预热 0 次、正式计时 2 次。下面的 median 与 IQR 仅来自这两个正式样本，适用于粗诊断，不能代表充分预热或跨进程稳定性。单位为秒。

| C | 方法 | median：前 → 后 | IQR：前 → 后 | 运行加速倍数 |
|---:|---|---:|---:|---:|
| 1 | BPTT | 0.205873 → 0.188317 | 0.000255849 → 8.3185e-05 | 1.09323 |
| 1 | RTRL | 0.078192 → 0.0622846 | 0.000474643 → 3.4755e-05 | 1.2554 |
| 5 | BPTT | 0.488288 → 0.377341 | 4.521e-05 → 7.45396e-06 | 1.29402 |
| 5 | RTRL | 0.191481 → 0.157937 | 9.39609e-05 → 5.5643e-05 | 1.21239 |

| C | 方法 | 版本 | 准备 s | 首次 s | 正式样本 s | min s | p90 s |
|---:|---|---|---:|---:|---|---:|---:|
| 1 | BPTT | baseline | 8.68953 | 0.218882 | 0.206129, 0.205617 | 0.205617 | 0.206078 |
| 1 | BPTT | optimized | 7.50375 | 0.200673 | 0.1884, 0.188233 | 0.188233 | 0.188383 |
| 1 | RTRL | baseline | 12.753 | 0.0800559 | 0.0786666, 0.0777174 | 0.0777174 | 0.0785717 |
| 1 | RTRL | optimized | 10.9848 | 0.0634376 | 0.0623193, 0.0622498 | 0.0622498 | 0.0623124 |
| 5 | BPTT | baseline | 11.6419 | 0.530876 | 0.488243, 0.488333 | 0.488243 | 0.488324 |
| 5 | BPTT | optimized | 10.0746 | 0.415439 | 0.377333, 0.377348 | 0.377333 | 0.377347 |
| 5 | RTRL | baseline | 17.3189 | 0.195629 | 0.191387, 0.191575 | 0.191387 | 0.191556 |
| 5 | RTRL | optimized | 13.2728 | 0.160692 | 0.157993, 0.157882 | 0.157882 | 0.157982 |

运行时间也有所变化，因此后续搜索 BPTT/RTRL 反转应使用同一修改版本的两种方法；本轮数据不覆盖或混入旧的大规模粗扫描。

## 内存记录

内存仍为 XLA compiler analysis 或 RTRL logical carry，不是实测进程峰值显存。前三个 worker 在诊断导出错误前尚未将 XLA memory_analysis 字段写入 JSON，这些字段保持缺失，不重新编译补取。其余完整字段保留在 trial JSON；下表为 temporary 与 logical carry（MiB）。

| C | 方法 | temporary：前 → 后 MiB | logical carry：前 → 后 MiB |
|---:|---|---:|---:|
| 1 | BPTT | — → — | — → — |
| 1 | RTRL | — → 0.310913 | 0.117188 → 0.117188 |
| 5 | BPTT | 1375.64 → 1328.74 | — → — |
| 5 | RTRL | 2.39643 → 2.39199 | 2.50488 → 2.50488 |

## 正确性与次数

- 比较保存的首次输出，新增校验 rollout 为 0。C=1 的两种方法各自修改前后 loss、逐步 losses 和 gradient 完全相同。
- C=5 的 BPTT/RTRL 修改前后 gradient relative-L2 error 分别为 `1.90469948e-13` / `1.78937418e-13`；最大梯度绝对误差 `1.13914211e-09`。
- 4 组方法内前后比较全部通过；4 组同版本 BPTT/RTRL 配对也全部通过，后者最大 gradient relative-L2 error `3.00548033e-13`。
- 验收条件：输出有限；总 loss 和逐步 losses 的 rtol=1e-7、atol=1e-8；梯度 relative-L2 <=1e-6 或 max-abs <=1e-8。

```json
{
  "analysis_model_executions": 0,
  "completed_workload_measurements": 8,
  "diagnostic_export_errors": 3,
  "extra_warmups_completed": 0,
  "first_executions_completed": 8,
  "gradient_calls_completed": 24,
  "retries": 0,
  "target_rollouts_completed": 8,
  "timed_executions_completed": 16,
  "total_workload_calls_completed": 32,
  "usable_measurements": 8,
  "worker_exit_ok": 5,
  "workers_launched": 8
}
```

## 环境与协议

- 开始/结束 UTC：`2026-09-10T15:01:22.524908+00:00` / `2026-09-10T15:06:55.300320+00:00`。墙钟 332.775 秒，包含调度暂停；未触及 600 秒/worker 和 1800 秒/整轮的预算。
- GPU 0：`NVIDIA H200 NVL, GPU-2c98f8c9-32c8-3bb9-2b43-7c8f13d4ef6f, 590.48.01, 143771 MiB`；运行前未发现 GPU 计算进程占用。
- 主机：双路 AMD EPYC 9754，每路 128 核、每核 2 线程，512 逻辑 CPU，约 1 TiB RAM。未固定 CPU affinity 或 GPU 时钟。
- Python `3.12.14 (main, Sep  2 2026, 23:27:36) [GCC 15.3.0]`；JAX `0.11.1`、jaxlib `0.11.1`、BrainState `0.5.4`、BrainUnit `0.5.2`、NumPy `2.5.2`。
- C=1、5，对应 Nx=4、20 和 Ntheta=3、15。全 CV paint Leak/K/Na，并训练全部三个 g_max scale。B=S=16，每次梯度计算 256 条轨迹；每个 seed 的参数跨 batch 共享。
- 40 ms，dt=0.025 ms，1600 步，float64；staggered solver、recursive backsub，完整 BPTT 和 full-state exact RTRL，随机种子 20260828。
- 每个 variant 一个独立轮次；8 个独立 worker 串行执行，按 C=1/5、BPTT/RTRL、baseline/optimized 排序。每 worker 单独生成 target、编译和执行。没有自动重试、优化器更新、GPU 轮询或 GPU profiling。
- Worker 环境固定 `CUDA_VISIBLE_DEVICES=0`、`JAX_PLATFORMS=cuda`、`JAX_ENABLE_X64=1`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 正式计时包含 reset 与完整 loss＋gradient，设备同步后停止。不含 target、准备、编译、文件保存。准备包括 target 前向及初始化相关构建；每个 target 只生成一次。

## 诊断导出异常与恢复

前三个 worker（C=1 的 BPTT baseline/optimized、RTRL baseline）在模型调用、NPZ 保存、StableHLO 保存完成后，遍历 Jaxpr 时触发 RecursionError。当前 JAX 的 Jaxpr.jaxpr 属性可引用自身，原统计函数先沿 .jaxpr 递归，未先读取 .eqns。

暂停父进程调度后，添加回归测试复现问题，再修复遍历顺序。另将核心测量结果的落盘移到可选诊断之前，并单独记录 diagnostics_status，避免辅助导出失败使已完成的测量丢失。剩余 5 个 worker 使用这一诊断修复；模型、materialize 候选、求导和计时阶段的实现没有在运行期间改变。

原始 trials/*.json 仍保留 3 条 error 和 5 条 ok，原始日志未覆盖。analysis/trials/ 中的派生条目保留 worker_status、source_trial、source_trial_sha256 和 recovered_from_saved_samples；从已有正式样本重算 median/IQR/min/p90，复用首次 NPZ 验证后将核心测量标为可用。没有伪造缺失内存或 Jaxpr 操作统计。第三个 worker 的父进程计墙钟包含了其退出后的调度暂停，不用于性能比较。

## 版本、复现与检查

- 基础提交 `c70a8b3cdcc867f42d85510f9fe2d5044a05d081`；`bench/hh-crossover-coarse` worktree，未提交修改。
- 启动时 tracked tree SHA-256：`4e19f5485257d34ae79cd987c6f3af9bbdf5840dea622ddc3b97cc0c9f87d2c9`；diff SHA-256：`376d8ed90755144113dd2b104c10370f86a96d37c0869844cd0711fa495d6c58`。
- baseline manager 源码 SHA-256：`864cc3d5f14c6769458c7bd0263ac06846694b5a020995bb183f002a424b2070`，已核对与基础提交完全一致。baseline worker 只恢复旧 materialize 方法及其原始 helper globals；其余实现与 optimized 相同。
- 诊断导出修复后的 benchmark.py SHA-256：`73c321eb41f44442e11aae5ae0bdc51649bd1984592d9c8f94c67d0224c224ca`。变化在编译/执行计时完成后的保存与图统计路径。已核对 materialize、rollout gradient engine、workload 的源码哈希在本轮期间不变。
- 原始命令、日志、JSON/NPZ、StableHLO、源码快照和图统计在本地 ignored `artifacts/compile_materialize_diagnostic_20260910/`；不随 Git 提供。`analysis/` 为离线派生结果，`actual_counts.json` 为实际调用计数；`diagnostic_export_incident.json` 记录异常和恢复方式。
- 完成修复后，14 个不执行神经元的测试及 6 个 subtest 通过，覆盖 materialize 的图规模、单位/精度、部分/重复/多 layout 写入、原子提交、正反向梯度，以及诊断自引用和导出失败时保留结果。未运行会增加模型 rollout 的整目录测试。

实际执行命令（工作目录 `/home/swl/studio/braincell-hh-crossover`）：

```bash
/home/swl/miniforge3/envs/braincell/bin/python benchmarks/performance/optim_gradient_scaling/artifacts/compile_materialize_diagnostic_20260910/launch.py --execute
```

该 launcher 对已存在 manifest 拒绝重复启动；重测须使用新的实验目录和明确执行计划。绘图和[旧粗扫描结果](../hh_crossover/before/hh-crossover-h200-20260910-supplement.md)保留原版本，未因本次优化重新解释为新测量。

本轮证据支持保留这项 materialize 向量化改动。C=21/41/81 的实际编译收益、其他机制/选区的完整集成覆盖以及跨进程稳定性仍未由本轮证明；不从 C=5 的 11.55 倍直接换算 C=81 的预期分钟数。
