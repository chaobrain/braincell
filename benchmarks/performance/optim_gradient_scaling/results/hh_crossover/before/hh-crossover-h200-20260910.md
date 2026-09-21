# HH 三条参数切片：H200 第一轮粗扫描

后续已取消时间限制并补齐两个缺失 BPTT 点，见[补测及完整 12 组结果](hh-crossover-h200-20260910-supplement.md)。本页保留第一轮结束时的原始结果、超时与次数记录。

本轮用时 **111.77 分钟**，24 个 worker 中 **22 个完成测量、2 个在编译阶段超时**。10 组有效配对均为 RTRL 更快（约 **2.10–4.87 倍**），没有观察到速度接近或反转；C=81、Ntheta=162/243 的 BPTT 缺失，不能判断这两个末端测点。

没有根据结果追加 CV、二分、独立复测或 profiling。

## 环境与来源

- 开始/结束 UTC：`2026-09-10T07:12:33.965095+00:00` / `2026-09-10T09:04:20.453870+00:00`；墙钟 111.77 分钟。
- GPU 0：`NVIDIA H200 NVL, GPU-2c98f8c9-32c8-3bb9-2b43-7c8f13d4ef6f, 590.48.01, 143771 MiB`；测量前未发现占用该 GPU 的计算进程。
- 主机：双路 AMD EPYC 9754（每路 128 核、2 线程/核），512 逻辑 CPU，约 1 TiB RAM；未设置 CPU affinity 或锁定 GPU 时钟。
- Python：`3.12.14 (main, Sep  2 2026, 23:27:36) [GCC 15.3.0]`；JAX `0.11.1`、jaxlib `0.11.1`、BrainState `0.5.4`、BrainUnit `0.5.2`、NumPy `2.5.2`。
- 基础提交：`c70a8b3cdcc867f42d85510f9fe2d5044a05d081`；在 `bench/hh-crossover-coarse` worktree 上使用未提交的 benchmark 修改。
- 测量时 tracked tree SHA-256：`ac0855415716a2f360861d5d2466e655fbcdf91d6781ad796d0b9bd52222d4ef`。
- 测量时 diff SHA-256：`6a2ba730472ef92785fe3f4d4b6900ac09fb02370fbe1ca0f5125d07c8edbec1`。
- 原始 JSON/NPZ、日志、manifest、源码 diff 和逐文件哈希存于本地 ignored `artifacts/hh_crossover_20260910/`，不随 Git 提供。
- 启动后仅修正了不执行模型的测试替身，使其声明的 Leak-only 配置与返回参数维度一致；所有 worker 使用相同的测量代码。运行后的汇总/文档修改不重新解释为新的测量版本。

## 模型与计时口径

- CV 数 C=1、21、41、81；始终全 paint Leak、K、Na，因此每条轨迹 active Nx=4C。仅训练每 CV 的 g_max scale，解冻组合为 Leak、Leak+K、Leak+K+Na，对应每 seed Ntheta=C、2C、3C。
- B=16 个电流协议共享一个 seed 的参数；S=16 个 seed 独立，一次梯度调用包含 256 条轨迹。40 ms，dt=0.025 ms，共 1600 步，float64。
- 延续现有 soma/双 dendrite arm 构造；C=1 是只有 soma 的特殊构型。staggered solver、recursive backsub、完整 BPTT 与 full-state exact RTRL；BPTT 不使用 checkpoint。
- 同一配置内两种方法使用相同输入与随机参数，固定随机种子 20260828。不同解冻组合会改变哪些参数取随机 scale、哪些固定为 1，因此跨组合不保证相同轨迹。
- 计划每个 worker：1 次 target 前向 rollout，1 次首次梯度执行，2 次额外预热，5 次正式计时。每次梯度调用从 reset 初态开始，返回逐步 losses、总 loss 和全部梯度；设备同步后停止计时。
- 正式耗时不含 target 生成、编译、磁盘保存、Adam 或 worker 启动。逐次 JSON 保存发生在计时区间外；原始计时样本全部保留。
- 编译时间是外层梯度 JIT 的 tracing/lowering/compile；模型准备与 target 编译另在 preparation_seconds 中。
- Worker 环境固定 `CUDA_VISIBLE_DEVICES=0`、`JAX_PLATFORMS=cuda`、`JAX_ENABLE_X64=1`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 关闭 GPU 活动、功耗、时钟及进程显存轮询；内存仅报告 XLA 分析与 RTRL logical carry，不声称测得进程峰值。
- 一个独立轮次，24 个串行独立 worker；按 CV 升序，同配置的方法相邻，配置间交替 BPTT/RTRL 顺序。单 worker 上限 1200 s，总预算 7200 s；无自动重试。

## 稳态时间与临时内存

已完成方法的最大 IQR/median 为 **0.362%**，最大样本极差/median 为 **1.181%**；这些仅描述进程内波动。

下表时间单位为秒。`median ± IQR` 中的 IQR 表示样本四分位距，不是置信区间。MiB=2^20 bytes。

| C | Nx | Ntheta | BPTT s ± IQR | RTRL s ± IQR | RTRL/BPTT | BPTT MiB | RTRL MiB | Classification / status |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 1 | 0.1832 ± 0.0004186 | 0.0485 ± 0.0001758 | 0.2648 | 262.8 | 0.2584 | rtrl_faster |
| 1 | 4 | 2 | 0.1936 ± 0.0002247 | 0.06906 ± 0.0002296 | 0.3567 | 269 | 0.3046 | rtrl_faster |
| 1 | 4 | 3 | 0.2062 ± 0.0006593 | 0.07749 ± 0.0002417 | 0.3758 | 275.3 | 0.3102 | rtrl_faster |
| 21 | 84 | 21 | 1.28 ± 0.0002339 | 0.5699 ± 0.0006917 | 0.4452 | 6046 | 12.85 | rtrl_faster |
| 21 | 84 | 42 | 1.721 ± 0.0002913 | 0.6767 ± 0.001241 | 0.3932 | 6178 | 25.35 | rtrl_faster |
| 21 | 84 | 63 | 2.161 ± 0.0003448 | 1.029 ± 0.0004846 | 0.4762 | 6310 | 37.82 | rtrl_faster |
| 41 | 164 | 41 | 4.194 ± 0.004027 | 1.221 ± 0.0006064 | 0.2911 | 1.25e+04 | 47.49 | rtrl_faster |
| 41 | 164 | 82 | 7.206 ± 0.008989 | 1.68 ± 0.0004055 | 0.2331 | 1.281e+04 | 94.18 | rtrl_faster |
| 41 | 164 | 123 | 9.967 ± 0.002416 | 2.047 ± 0.006013 | 0.2054 | 1.311e+04 | 141 | rtrl_faster |
| 81 | 324 | 81 | 8.95 ± 0.00432 | 2.861 ± 0.002543 | 0.3197 | 2.618e+04 | 184.3 | rtrl_faster |
| 81 | 324 | 162 | — ± — | 4.348 ± 0.005192 | — | — | 363.4 | unavailable (timeout/ok) |
| 81 | 324 | 243 | — ± — | 6.459 ± 0.01123 | — | — | 544.4 | unavailable (timeout/ok) |

## 编译与其他内存

B/R 分别为 BPTT/RTRL。内存字段为 MiB；完整字节值在 trial JSON 和配对 CSV 中。

| C | Ntheta | 编译 B/R (s) | argument B/R | output B/R | alias B/R | RTRL carry |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3.06212/2.1154 | 0.00012207/0.00012207 | 0.19558/0.19558 | 0/0 | 0.0388184 |
| 1 | 2 | 3.91207/2.50278 | 0.000244141/0.000244141 | 0.195702/0.195702 | 0/0 | 0.0778809 |
| 1 | 3 | 4.77555/2.95829 | 0.000366211/0.000366211 | 0.195824/0.195824 | 0/0 | 0.117188 |
| 21 | 21 | 192.732/23.612 | 0.00256348/0.00256348 | 0.198021/0.198021 | 0/0 | 14.0427 |
| 21 | 42 | 389.098/46.0878 | 0.00512695/0.00512695 | 0.200584/0.200584 | 0/0 | 28.2957 |
| 21 | 63 | 600.458/73.6008 | 0.00769043/0.00769043 | 0.203148/0.203148 | 0/0 | 42.7588 |
| 41 | 41 | 79.6066/50.7257 | 0.00500488/0.00500488 | 0.200462/0.200462 | 0/0 | 53.2419 |
| 41 | 82 | 208.337/112.84 | 0.0100098/0.0100098 | 0.205467/0.205467 | 0/0 | 107.295 |
| 41 | 123 | 410.228/177.081 | 0.0150146/0.0150146 | 0.210472/0.210472 | 0/0 | 162.158 |
| 81 | 81 | 213.097/96.222 | 0.0098877/0.0098877 | 0.205345/0.205345 | 0/0 | 207.226 |
| 81 | 162 | —/212.635 | —/0.0197754 | —/0.215233 | —/0 | 417.636 |
| 81 | 243 | —/337.6 | —/0.0296631 | —/0.225121 | —/0 | 631.23 |

## 次数与正确性

实际完成的执行次数：

```json
{
  "extra_warmups_completed": 44,
  "first_executions_completed": 22,
  "gradient_calls_completed": 176,
  "incomplete_counts_are_lower_bounds": true,
  "methods_ok": 22,
  "target_rollouts_completed": 24,
  "timed_executions_completed": 110,
  "total_workload_calls_completed": 200,
  "workers_launched": 24
}
```

- 全量目标：24 次 target rollout、24 次首次梯度执行、48 次额外预热、120 次正式计时，即 192 次梯度调用、216 次完整 workload 调用。BPTT 的反向传播包含在梯度调用内。
- 正确性检查直接使用首次输出，额外完整仿真次数为 0。初始化/reset/单步 tracing 不计为完整 rollout。
- 配对验收：输出有限；总 loss 和逐步 losses 的 rtol=1e-7、atol=1e-8；梯度 relative-L2 <=1e-6 或 max-abs <=1e-8。近零梯度的逐元素相对误差另存，不单独决定验收。
- 10/12 组配对通过正确性验收。最大梯度 relative-L2 error：`4.05728e-12`；最大梯度绝对误差：`2.34986e-08`；最大总 loss 绝对误差：`6.78241e-10`。
- 实际共 176 次梯度调用、24 次 target rollout，即 200 次完整 workload 调用。两次超时均停在 `compile` 阶段：各完成 1 次 target rollout，首次梯度/额外预热/正式计时均为 0 次。相对计划少了 16 次梯度调用；没有重试。
- 通用计数器对异常 worker 保守设置 `incomplete_counts_are_lower_bounds=true`；本轮已额外核对两份阶段记录，未发生进入首次梯度执行后再中断的情形。
- 失败/未完成项：c81_t40_b16_s16__lkn_fit_lkn bptt: timeout；c81_t40_b16_s16__lkn_fit_lk bptt: timeout

## 边界判读与局限

10 组有效配对的 RTRL/BPTT 均小于 0.48，没有进入 [0.9, 1.1] 的接近区间，也没有相邻有效测点的速度比跨过 1。三条线上的速度比并不随 C 单调变化，因此当前没有可直接二分的交叉区间。

C=81 的 2C、3C 参数组缺少 BPTT 稳态时间。这是编译预算限制，不能视为 BPTT 运行更慢的证据。下一轮应优先讨论是否增加这两个点的编译预算并补齐测量，再决定是否向更大 C 扩展；本轮不执行这些后续测量。

- 仅在四个离散 CV 上进行了一个独立轮次，不能排除未测区间存在局部反转，也不能据此宣称跨进程稳定性。
- 三条线同时改变 CV 与参数数；结果代表这组 HH/解冻/硬件配置，不是只由 Nx 或 Ntheta 决定的普遍边界。
- 此处固定 T，不做内存等值边界搜索，也不从本轮数据推断完整 T 扩展规律。

## 实际命令与实现检查

```bash
/home/swl/miniforge3/envs/braincell/bin/python -m benchmarks.performance.optim_gradient_scaling.runner.hh_crossover.runner run \
  --suite hh_crossover --gpu 0 --repeats 5 --warmups 2 --no-gpu-monitor \
  --worker-timeout-seconds 1200 --budget-seconds 7200 \
  --output-dir benchmarks/performance/optim_gradient_scaling/artifacts/hh_crossover_20260910 \
  --python /home/swl/miniforge3/envs/braincell/bin/python
```

工作目录为 `/home/swl/studio/braincell-hh-crossover`。14 项不执行 neuron 模型的驱动测试及 4 个 subtest 已验证；最后增加维度断言后，相关测试替身曾失败并已修正，失败项单独复测通过。正式扫描在该失败返回时已启动，记录这一检查顺序偏差；未增加或重跑模型调用。

后续命令与脚本说明见 [README](../../README.md)。历史 A100 数据见 [历史 scaling](../a100/scaling.md)，本轮不与其混合统计。
