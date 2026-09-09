# 预定事件直接投递完整数据表

本表为已审阅的历史实测快照；完整硬件、软件、代码版本及证据范围见 [对应实测总结](scheduled-event-delivery.md)。报告脚本默认在输入 artifacts 中生成新表，不自动覆盖本文件。

原始运行材料仅保存在运行者本地的 artifacts，未随 Git 提供；本页的正文与表格随仓库维护。原始目录、日志或图像链接仅为可选核查入口，缺失时不影响读取已保存的数据。

第二轮比较 current、预计算 scan、原 bucket、紧凑 direct 和固定形状 padded，分别评估 CPU/GPU。完整模型由实验 Cell 子类替换预定事件输入计算，复用真实 Network/Cell solver；production 为未安装适配器的对照。

默认 100 ms、dt=.025 ms、4000 步、float32、种子 7；每 worker 预热 2 次、重复 5 次、同步等待。表中为独立进程中位数的中位数。profile 运行不进入计时汇总；本轮 float64 仅做正确性测试。

## 稳态运行时间

单位 ms，越小越好；— 表示本组合没有成功计时，原因见排除记录。synapse 包含查询、权重归约和 ExpSyn 更新；full 包含真实膜电位积分。

| 模式 | 设备 | 场景 | production | current | scan | bucket | direct | padded |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | cpu | large | 475.952 | — | 451.752 | — | 332.706 | 319.981 |
| full | cpu | long | 1208.438 | 1192.592 | 1196.379 | 60.528 | 11.247 | — |
| full | cpu | shared | 224.799 | — | 200.495 | — | 18.452 | 16.981 |
| full | cpu | small | 3.803 | 3.660 | 3.922 | 35.516 | 35.317 | — |
| full | gpu | large | 143.188 | — | 142.133 | — | 305.728 | 138.376 |
| full | gpu | long | 115.306 | 114.147 | 117.697 | 194.761 | 211.104 | — |
| full | gpu | shared | 129.210 | — | 129.812 | — | 273.685 | 104.663 |
| full | gpu | small | 91.899 | 99.762 | 97.050 | 164.625 | 188.154 | — |
| synapse | cpu | burst | — | 743.859 | 715.048 | 9.692 | 5.247 | — |
| synapse | cpu | dense | — | 701.376 | 659.111 | 10.104 | 6.850 | 5.599 |
| synapse | cpu | large | — | 176.466 | 152.355 | 64.133 | 28.114 | 25.300 |
| synapse | cpu | long | — | 1139.356 | 1023.474 | 8.529 | 6.153 | — |
| synapse | cpu | small | — | 3.109 | 2.792 | 4.602 | 5.112 | — |
| synapse | cpu | sparse | — | 20.792 | 17.945 | 9.731 | 5.843 | — |
| synapse | gpu | burst | — | 41.015 | 35.469 | 187.946 | 117.626 | — |
| synapse | gpu | dense | — | 40.594 | 35.406 | 216.376 | 203.571 | 54.634 |
| synapse | gpu | large | — | 37.929 | 56.919 | 232.144 | 217.546 | 58.497 |
| synapse | gpu | long | — | 43.333 | 39.556 | 147.488 | 123.460 | — |
| synapse | gpu | small | — | 43.331 | 43.191 | 126.188 | 111.711 | — |
| synapse | gpu | sparse | — | 34.250 | 53.644 | 158.101 | 133.648 | — |

## 稳定收益与复用成本

稳定收益要求至少三个独立配对进程均更快，配对加速比中位数 ≥1.10。此规则只支持所测场景；不等于统计置信区间或全局自动选择阈值。

| 模式 | 设备 | 场景 | 方法 | 独立进程 | 稳态加速 | 稳定收益 | 首次使用 ms | 100 次估算 ms | 数组 KiB |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| full | cpu | large | direct | 3 | 1.43× | 是 | 15357.7 | 48363.1 | 836.86 |
| full | cpu | large | padded | 3 | 1.41× | 是 | 15552.8 | 47234.5 | 1351.56 |
| full | cpu | large | production | 7 | 1.00× | 否 | 14108.7 | 61216.1 | 0.00 |
| full | cpu | large | scan | 1 | 1.06× | 否 | 15523.9 | 60247.4 | 390.62 |
| full | cpu | long | bucket | 1 | 20.81× | 否 | 4871.0 | 10863.3 | 20.04 |
| full | cpu | long | current | 1 | 1.06× | 否 | 4866.5 | 122933.1 | 0.00 |
| full | cpu | long | direct | 3 | 107.44× | 是 | 4800.0 | 5913.5 | 24.83 |
| full | cpu | long | production | 3 | 1.00× | 否 | 4656.4 | 124262.8 | 0.00 |
| full | cpu | long | scan | 1 | 1.05× | 否 | 5974.5 | 124416.0 | 3906.25 |
| full | cpu | shared | direct | 3 | 12.22× | 是 | 13711.6 | 15538.4 | 836.86 |
| full | cpu | shared | padded | 3 | 13.30× | 是 | 13756.1 | 15437.2 | 1351.56 |
| full | cpu | shared | production | 7 | 1.00× | 否 | 12876.9 | 35130.9 | 0.00 |
| full | cpu | shared | scan | 1 | 1.12× | 否 | 13834.5 | 33683.5 | 390.62 |
| full | cpu | small | bucket | 1 | 0.11× | 否 | 4728.3 | 8244.3 | 16.52 |
| full | cpu | small | current | 1 | 1.04× | 否 | 3383.1 | 3745.5 | 0.00 |
| full | cpu | small | direct | 1 | 0.11× | 否 | 4253.8 | 7750.2 | 17.45 |
| full | cpu | small | production | 1 | 1.00× | 否 | 3266.4 | 3642.9 | 0.00 |
| full | cpu | small | scan | 1 | 0.97× | 否 | 4424.3 | 4812.5 | 0.39 |
| full | gpu | large | direct | 1 | 0.47× | 否 | 15420.3 | 45687.4 | 836.86 |
| full | gpu | large | padded | 3 | 1.04× | 否 | 15434.7 | 29098.2 | 1351.56 |
| full | gpu | large | production | 5 | 1.00× | 否 | 14756.2 | 28898.9 | 0.00 |
| full | gpu | large | scan | 1 | 1.01× | 否 | 15137.2 | 29208.4 | 390.62 |
| full | gpu | long | bucket | 1 | 0.59× | 否 | 4784.4 | 24065.7 | 20.04 |
| full | gpu | long | current | 1 | 1.01× | 否 | 3563.7 | 14864.3 | 0.00 |
| full | gpu | long | direct | 1 | 0.55× | 否 | 4799.0 | 25698.3 | 24.83 |
| full | gpu | long | production | 1 | 1.00× | 否 | 3794.2 | 15209.5 | 0.00 |
| full | gpu | long | scan | 1 | 0.98× | 否 | 4849.5 | 16501.6 | 3906.25 |
| full | gpu | shared | direct | 1 | 0.47× | 否 | 14284.4 | 41379.2 | 836.86 |
| full | gpu | shared | padded | 3 | 1.23× | 是 | 13891.9 | 24253.5 | 1351.56 |
| full | gpu | shared | production | 5 | 1.00× | 否 | 13295.2 | 26056.3 | 0.00 |
| full | gpu | shared | scan | 1 | 0.99× | 否 | 13690.1 | 26541.5 | 390.62 |
| full | gpu | small | bucket | 1 | 0.56× | 否 | 4343.2 | 20641.1 | 16.52 |
| full | gpu | small | current | 1 | 0.92× | 否 | 3279.0 | 13155.4 | 0.00 |
| full | gpu | small | direct | 1 | 0.49× | 否 | 4300.4 | 22927.7 | 17.45 |
| full | gpu | small | production | 1 | 1.00× | 否 | 3451.2 | 12549.2 | 0.00 |
| full | gpu | small | scan | 1 | 0.95× | 否 | 4016.4 | 13624.3 | 0.39 |
| synapse | cpu | burst | bucket | 1 | 78.93× | 否 | 2616.3 | 3575.8 | 1266.13 |
| synapse | cpu | burst | current | 3 | 1.00× | 否 | 1968.3 | 75627.6 | 0.00 |
| synapse | cpu | burst | direct | 3 | 138.67× | 是 | 2586.4 | 3117.5 | 98.66 |
| synapse | cpu | burst | scan | 3 | 1.03× | 否 | 3160.0 | 73949.7 | 1250.00 |
| synapse | cpu | dense | bucket | 1 | 69.42× | 否 | 2572.4 | 3572.7 | 1266.13 |
| synapse | cpu | dense | current | 3 | 1.00× | 否 | 1883.0 | 71305.4 | 0.00 |
| synapse | cpu | dense | direct | 3 | 101.50× | 是 | 2595.2 | 3260.7 | 2520.54 |
| synapse | cpu | dense | padded | 3 | 124.55× | 是 | 2560.2 | 3114.5 | 3222.66 |
| synapse | cpu | dense | scan | 3 | 1.00× | 否 | 3119.9 | 68396.5 | 1250.00 |
| synapse | cpu | large | bucket | 1 | 2.89× | 否 | 3185.6 | 9534.8 | 406.71 |
| synapse | cpu | large | current | 3 | 1.00× | 否 | 1982.4 | 19464.0 | 0.00 |
| synapse | cpu | large | direct | 3 | 6.31× | 是 | 3236.8 | 6002.2 | 836.86 |
| synapse | cpu | large | padded | 3 | 6.96× | 是 | 3207.5 | 5712.2 | 1351.56 |
| synapse | cpu | large | scan | 3 | 1.16× | 是 | 3271.4 | 18362.3 | 390.62 |
| synapse | cpu | long | bucket | 1 | 137.23× | 否 | 2560.9 | 3405.3 | 20.04 |
| synapse | cpu | long | current | 3 | 1.00× | 否 | 2446.6 | 115196.5 | 0.00 |
| synapse | cpu | long | direct | 3 | 190.22× | 是 | 2442.5 | 3104.2 | 24.83 |
| synapse | cpu | long | scan | 3 | 1.11× | 否 | 3432.2 | 104777.2 | 3906.25 |
| synapse | cpu | small | bucket | 1 | 0.69× | 否 | 2525.5 | 2981.1 | 16.52 |
| synapse | cpu | small | current | 3 | 1.00× | 否 | 1258.7 | 1569.0 | 0.00 |
| synapse | cpu | small | direct | 3 | 0.60× | 否 | 2567.5 | 3079.9 | 17.45 |
| synapse | cpu | small | scan | 3 | 1.09× | 否 | 2464.5 | 2753.8 | 0.39 |
| synapse | cpu | sparse | bucket | 1 | 2.13× | 否 | 3387.8 | 4351.2 | 20.04 |
| synapse | cpu | sparse | current | 3 | 1.00× | 否 | 2058.5 | 4117.0 | 0.00 |
| synapse | cpu | sparse | direct | 3 | 3.55× | 是 | 3341.7 | 3955.8 | 28.35 |
| synapse | cpu | sparse | scan | 3 | 1.16× | 是 | 3155.2 | 5062.2 | 39.06 |
| synapse | gpu | burst | bucket | 1 | 0.22× | 否 | 2593.3 | 21199.9 | 1266.13 |
| synapse | gpu | burst | current | 3 | 1.00× | 否 | 1282.1 | 5278.3 | 0.00 |
| synapse | gpu | burst | direct | 3 | 0.35× | 否 | 2631.0 | 14275.9 | 98.66 |
| synapse | gpu | burst | scan | 3 | 1.16× | 是 | 2355.8 | 5867.3 | 1250.00 |
| synapse | gpu | dense | bucket | 1 | 0.19× | 否 | 2546.1 | 23967.3 | 1266.13 |
| synapse | gpu | dense | current | 3 | 1.00× | 否 | 1262.6 | 5281.5 | 0.00 |
| synapse | gpu | dense | direct | 3 | 0.20× | 否 | 2692.3 | 22860.6 | 2520.54 |
| synapse | gpu | dense | padded | 3 | 0.75× | 否 | 2416.1 | 7803.7 | 3222.66 |
| synapse | gpu | dense | scan | 3 | 1.15× | 是 | 2327.5 | 5776.9 | 1250.00 |
| synapse | gpu | large | bucket | 1 | 0.16× | 否 | 3153.3 | 26135.6 | 406.71 |
| synapse | gpu | large | current | 3 | 1.00× | 否 | 1831.6 | 5603.3 | 0.00 |
| synapse | gpu | large | direct | 3 | 0.17× | 否 | 3163.5 | 24754.9 | 836.86 |
| synapse | gpu | large | padded | 3 | 0.65× | 否 | 3008.4 | 8799.6 | 1351.56 |
| synapse | gpu | large | scan | 3 | 0.67× | 否 | 2899.1 | 8598.3 | 390.62 |
| synapse | gpu | long | bucket | 1 | 0.30× | 否 | 2516.5 | 17117.8 | 20.04 |
| synapse | gpu | long | current | 3 | 1.00× | 否 | 1348.6 | 5638.5 | 0.00 |
| synapse | gpu | long | direct | 3 | 0.35× | 否 | 2578.2 | 14784.9 | 24.83 |
| synapse | gpu | long | scan | 3 | 1.11× | 是 | 2367.1 | 6255.0 | 3906.25 |
| synapse | gpu | small | bucket | 1 | 0.34× | 否 | 2464.0 | 14956.6 | 16.52 |
| synapse | gpu | small | current | 3 | 1.00× | 否 | 1326.7 | 5616.5 | 0.00 |
| synapse | gpu | small | direct | 3 | 0.40× | 否 | 2464.9 | 13524.3 | 17.45 |
| synapse | gpu | small | scan | 3 | 1.00× | 否 | 2335.6 | 6611.5 | 0.39 |
| synapse | gpu | sparse | bucket | 1 | 0.22× | 否 | 3321.5 | 18973.5 | 20.04 |
| synapse | gpu | sparse | current | 3 | 1.00× | 否 | 2076.9 | 5462.4 | 0.00 |
| synapse | gpu | sparse | direct | 3 | 0.26× | 否 | 3432.0 | 16658.5 | 28.35 |
| synapse | gpu | sparse | scan | 3 | 0.64× | 否 | 3141.2 | 8453.6 | 39.06 |

首次使用包含源/模型构建、计划准备、首次执行及 full 的初始 reset；首次执行含编译，不能称为纯执行。full 稳态和 100 次重用估算不含后续运行间 reset，不能视为独立重复实验的完整总时间。micro 和 full 的构建范围不同，不跨模式比较冷启动。数组字节只统计显式计划存储，current/production 为零不代表实际内存为零；准备临时完整事件表和 executable 常量没有计入，allocator 统计保留在 worker JSON，不能当作内核峰值。

## 校验与排除

导出 336 条计时记录；校验失败 4 项，排除记录 32 项。
- full_initial/cpu_large_r0.json: timeout
- full_initial/gpu_large_r0.json: timeout
- full_initial/cpu_shared_r0.json: timeout
- full_initial/gpu_shared_r0.json: timeout
- full/cpu/large/worker: timeout — worker deadline exceeded
- full/cpu/long/padded: not_applicable — padded requires 96400 bytes, ratio 12
- full/cpu/shared/worker: timeout — worker deadline exceeded
- full/cpu/small/padded: not_applicable — padded requires 32040 bytes, ratio 40
- full/gpu/large/worker: timeout — worker deadline exceeded
- full/gpu/long/padded: not_applicable — padded requires 96400 bytes, ratio 12
- full/gpu/shared/worker: timeout — worker deadline exceeded
- full/gpu/small/padded: not_applicable — padded requires 32040 bytes, ratio 40
- micro/cpu/burst/padded: not_applicable — padded requires 32004000 bytes, ratio 400
- micro/cpu/long/padded: not_applicable — padded requires 96400 bytes, ratio 12
- micro/cpu/small/padded: not_applicable — padded requires 32040 bytes, ratio 40
- micro/cpu/sparse/padded: not_applicable — padded requires 132000 bytes, ratio 16
- micro/gpu/burst/padded: not_applicable — padded requires 32004000 bytes, ratio 400
- micro/gpu/long/padded: not_applicable — padded requires 96400 bytes, ratio 12
- micro/gpu/small/padded: not_applicable — padded requires 32040 bytes, ratio 40
- micro/gpu/sparse/padded: not_applicable — padded requires 132000 bytes, ratio 16

## 复现与证据

工作流命令见 [benchmark README](../README.md)。原始 CSV、独立 worker、源码快照及 profiler 输出位于该目录 artifacts/delivery 下；delivery_validation.json 保留所有未通过、超时和不适用项。

[结果解释与选型](scheduled-event-delivery.md) · [方案](../../../../docs/design/synapse/proposals/event-delivery-optimization.md) · [第一轮结果](scheduled-event-queries.md)
