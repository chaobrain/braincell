# HH 粗扫描补测：补齐 C=81 的两组 BPTT

本次补齐 **C=81、Nx=324、Ntheta=162/243** 的 BPTT，两个 worker 均成功。补测墙钟 **68.96 分钟**。与先前同配置 RTRL 配对后，12/12 组均通过正确性检查；RTRL 仍全部更快，速度比 RTRL/BPTT 范围 **0.2054–0.4762**，没有进入 [0.9, 1.1] 的接近区间，也没有观察到相邻测点胜负变化。

本次只补齐两个缺失方法结果，复用第一轮的 RTRL 测量，没有扩大 C 范围或增加独立复测。两次原始编译超时保留在[第一轮报告](hh-crossover-h200-20260910.md)及其原始目录中。

现有数据的 [Nx–Ntheta 切片、速度、内存和编译结果图](figures-notes.md)另附，绘图没有新增模型执行。

## 固定条件与环境

- 全 CV paint Leak、K、Na，Nx=4C；每 CV 解冻 Leak+K 或 Leak+K+Na 的 g_max scale，Ntheta=2C 或 3C。B=16、S=16，参数跨 batch 共享、各 seed 独立。单次完整梯度调用包含 256 条轨迹。
- 40 ms，dt=0.025 ms，1600 步，float64；沿用 soma/双 dendrite arm 构造、staggered solver、recursive backsub、完整 BPTT，无 checkpoint。配对基线为 full-state exact RTRL。随机种子 20260828。
- 每个补测 worker：target 前向 1 次、首次梯度执行 1 次、额外预热 2 次、正式计时 5 次；不额外运行模型做验证。首次执行与预热单列，5 个正式样本来自同一进程。
- 稳态时间包含 reset 与完整 loss＋gradient；同步设备后停止计时，不含准备、target、编译、文件保存或优化器更新。梯度 kernel 编译时间包含 tracing/lowering/compile，target 编译在 preparation 中另计。
- 按 Ntheta=162、243 串行启动两个独立进程；按用户确认取消单 worker 及整轮时间上限。未再重试；GPU 利用率、功耗和显存轮询全部关闭，无 profiling。
- 开始/结束 UTC：`2026-09-10T12:38:37.154917+00:00` / `2026-09-10T13:47:35.264380+00:00`。
- GPU 0：`NVIDIA H200 NVL, GPU-2c98f8c9-32c8-3bb9-2b43-7c8f13d4ef6f, 590.48.01, 143771 MiB`；与第一轮同一张卡，补测开始前无该 GPU 的计算进程。
- 主机：双路 AMD EPYC 9754，每路 128 核、每核 2 线程，512 逻辑 CPU，约 1 TiB RAM；未设置 CPU affinity 或锁定 GPU 时钟。
- Python `3.12.14 (main, Sep  2 2026, 23:27:36) [GCC 15.3.0]`；JAX `0.11.1`、jaxlib `0.11.1`、BrainState `0.5.4`、BrainUnit `0.5.2`、NumPy `2.5.2`。
- Worker 环境：`CUDA_VISIBLE_DEVICES=0`、`JAX_PLATFORMS=cuda`、`JAX_ENABLE_X64=1`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。

## 完整 12 组配对

时间单位为秒，IQR 是四分位距，不是置信区间；内存为 XLA temporary 分析值，MiB=2^20 bytes。最后两行 BPTT 来自本次补测，其余结果来自第一轮。

| C | Nx | Ntheta | BPTT median ± IQR (s) | RTRL median ± IQR (s) | RTRL/BPTT | BPTT temporary MiB | RTRL temporary MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 1 | 0.183171 ± 0.000418602 | 0.0484968 ± 0.000175787 | 0.264762 | 262.759 | 0.258423 |
| 1 | 4 | 2 | 0.193615 ± 0.000224708 | 0.0690595 ± 0.000229568 | 0.356685 | 269.01 | 0.304596 |
| 1 | 4 | 3 | 0.206179 ± 0.000659265 | 0.0774856 ± 0.000241678 | 0.375817 | 275.261 | 0.310226 |
| 21 | 84 | 21 | 1.28026 ± 0.000233884 | 0.56993 ± 0.000691691 | 0.445167 | 6045.61 | 12.8461 |
| 21 | 84 | 42 | 1.7208 ± 0.000291257 | 0.67668 ± 0.00124139 | 0.393236 | 6177.89 | 25.3459 |
| 21 | 84 | 63 | 2.16103 ± 0.000344762 | 1.02917 ± 0.000484648 | 0.476242 | 6310.16 | 37.8225 |
| 41 | 164 | 41 | 4.19389 ± 0.00402678 | 1.22065 ± 0.000606368 | 0.291055 | 12500.2 | 47.488 |
| 41 | 164 | 82 | 7.20601 ± 0.00898875 | 1.67968 ± 0.000405516 | 0.233095 | 12805.8 | 94.1796 |
| 41 | 164 | 123 | 9.96703 ± 0.00241556 | 2.04701 ± 0.0060126 | 0.205378 | 13111.4 | 140.956 |
| 81 | 324 | 81 | 8.9497 ± 0.0043196 | 2.86095 ± 0.00254305 | 0.31967 | 26182 | 184.332 |
| 81 | 324 | 162 | 15.2023 ± 0.0020399 | 4.34762 ± 0.00519242 | 0.285985 | 26880.8 | 363.388 |
| 81 | 324 | 243 | 21.413 ± 7.82199e-05 | 6.45925 ± 0.0112309 | 0.301651 | 27579.7 | 544.356 |

全部 24 条成功方法结果的最大 IQR/median 为 **0.362%**，仅说明进程内波动。

## 补测的完整计时记录

| Ntheta | preparation s | 梯度编译 s | 首次 s | 额外预热 s | 5 次正式样本 s | min s | p90 s | worker 墙钟 s |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 162 | 32.6123 | 987.216 | 16.3314 | 15.2041, 15.2061 | 15.2023, 15.2019, 15.2019, 15.2098, 15.204 | 15.2019 | 15.2074 | 1150.87 |
| 243 | 38.2877 | 2765.93 | 22.8227 | 21.4603, 21.4231 | 21.413, 21.4129, 21.413, 21.4133, 21.413 | 21.4129 | 21.4132 | 2986.97 |

## 补齐点的编译与内存

B/R 分别表示 BPTT/RTRL；内存单位为 MiB。argument、output、temporary、alias 来自 XLA 编译器分析，carry 为 RTRL 逻辑大小，均不代表实测进程峰值显存。

| Ntheta | 编译 B/R s | argument B/R | output B/R | temporary B/R | alias B/R | RTRL logical carry |
|---:|---:|---:|---:|---:|---:|---:|
| 162 | 987.216/212.635 | 0.0197754/0.0197754 | 0.215233/0.215233 | 26880.8/363.388 | 0/0 | 417.636 |
| 243 | 2765.93/337.6 | 0.0296631/0.0296631 | 0.225121/0.225121 | 27579.7/544.356 | 0/0 | 631.23 |

## 正确性与实际次数

- 使用保存的首次输出配对比较，未增加完整 rollout。12/12 对输出有限并通过验收；最大梯度 relative-L2 error `4.0572756e-12`，最大梯度绝对误差 `2.34986146e-08`，最大总 loss 绝对误差 `8.22666379e-10`。
- 验收条件：总 loss 和逐步 losses 的 rtol=1e-7、atol=1e-8；梯度 relative-L2 <=1e-6 或 max-abs <=1e-8。
- 本次新增：2 个 worker，2 次 target、2 次首次梯度、4 次额外预热、10 次正式计时，共 **16 次梯度计算＋2 次 target=18 次完整 workload 调用**。
- 连同第一轮所有尝试：26 个 worker、24 条成功方法结果；26 次 target、24 次首次梯度、48 次额外预热、120 次正式计时，共 **192 次梯度计算＋26 次 target=218 次完整 workload 调用**。原先失败的两个 worker 各执行过 1 次 target，因此累计总量比最初无失败计划的 216 次多 2 次。
- 汇总目录的 selected_trial_counts.json 仅计入选中的 24 条成功结果（216 次），不应作为历史所有尝试的实际次数；累计权威计数为 cumulative_attempt_counts.json（218 次）。

## 来源与复现

- 基础提交 `c70a8b3cdcc867f42d85510f9fe2d5044a05d081`；分支 `bench/hh-crossover-coarse`，包含未提交 benchmark 修改。
- 补测启动时 tracked tree SHA-256：`a2f3a5bd0f983a2f9ec2ebc4f87e83e681532c4ddfb2a6e6a594042e9bcacdcd`；diff SHA-256：`2180524fc43a1a93d988bce7e7ebeceb502ce86e5e818267b2c3e4a3546f805a`。
- 已核对 benchmark.py、validation/optim/_workload.py、braincell/experimental/optim/gradients.py 与第一轮及补测完成时的 SHA-256 一致；测量 kernel 未变更。整个 tracked tree 的差异来自测试替身修正和文档更新。
- 本轮通过 artifact 内的 launch.py 直接调用已有 worker，subprocess 不传 timeout；实际两个 worker 命令逐项保存在 supplement/manifest.json 的 execution_order 中。启动前与汇总脚本做了语法/静态检查，没有再运行会启动模型的测试。
- 本地原始材料位于本实验目录下的 ignored artifacts，不随 Git 提供：`hh_crossover_20260910/` 保留第一轮全部成功和超时；`hh_crossover_20260910_supplement/` 保存两个补测 worker 的原始 JSON/NPZ、日志、命令和代码指纹；`hh_crossover_20260910_completed/` 仅组装已保存数据并重算配对，不运行模型。
- completed 的 manifest 和 trial JSON 记录每条结果的来源及原始 JSON SHA-256，区分初测与补测；supplement 的 actual_counts.json 记录本次实际次数，completed 的 cumulative_attempt_counts.json 记录所有尝试累计次数。

实际启动命令（工作目录 `/home/swl/studio/braincell-hh-crossover`）：

```bash
/home/swl/miniforge3/envs/braincell/bin/python benchmarks/performance/optim_gradient_scaling/artifacts/hh_crossover_20260910_supplement/launch.py
```

## 边界判读与局限

三个切片在 C=1、21、41、81 的所有已测点均由 RTRL 获胜，没有找到可供二分的异号区间。速度比随 C 并不单调，不能从这些点排除未测区域的局部反转，也不把插值当作已验证边界。

补测与 RTRL 基线发生在不同时间，每个配置/方法仍只有一个成功进程测量，不能据此断言跨进程稳定性。此结论限于当前 full HH、B=S=16、T=1600、x64、solver 和 H200 配置。下一轮范围与复测次数另行确定；本次到此完成。
