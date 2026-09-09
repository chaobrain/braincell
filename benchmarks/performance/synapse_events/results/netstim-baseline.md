# NetStim 突触事件性能实测

原始运行材料仅保存在运行者本地的 artifacts，未随 Git 提供；本页的正文与表格随仓库维护。原始目录、日志或图像链接仅为可选核查入口，缺失时不影响读取已保存的数据。

运行记录：2026-09-08T06:58:02.678767+00:00。成功 130 个设备/配置，失败或超时 0 个。
基于提交 `bd8e2f9dbdb26630906b12b7242d7f2382643f36` 的工作区；完整未提交文件清单保存在 manifest。

## 测量口径

每个配置使用独立进程。CPU/GPU 使用同一 Python 环境及 CPU 生成的固定种子事件表。每条轨迹采用 jit + for_loop；reset 在计时区间外执行并同步，计时包含设备完成等待。
首次调用单列（含编译及执行），另预热 2 次、测量 5 次。查询与聚合为独立编译微基准，不能相加或除以完整运行时间解释成阶段占比。

NetStim 构造生成日程，Cell 每步查询日程并加权 scatter-add；本实验不测 live Cell spike 延迟队列，也不比较 scatter/brainevent 后端。完整运行使用 prepare_run + update，不包括 Network.run 的结果表物化。

## 主要观察

- 规模 sweep 中 GPU 快于 CPU 的配置为 2/18；CPU/GPU 完整运行时间比范围 0.01–3.36。小于 1 表示 CPU 更快。
- CPU / independent：时间表从 10 增至 10000 条/源，完整运行 12.146 → 1286.255 ms （105.90×），查询微基准 5.802 → 1048.087 ms。窗口内到达计数及最终状态的匹配结果见数值检查。
- CPU / shared：时间表从 10 增至 10000 条/源，完整运行 10.022 → 1073.142 ms （107.08×），查询微基准 6.133 → 1061.350 ms。窗口内到达计数及最终状态的匹配结果见数值检查。
- CPU，N=100、M=100：独立突触 485.137 ms，共享突触 235.184 ms；独立/共享时间比 2.06。
- CPU，N=100、M=10：批量连接 53.900 ms，每细胞一次连接 375.018 ms；完整模拟首次调用分别 1.407 / 5.224 s（含编译和执行）。
- GPU / independent：时间表从 10 增至 10000 条/源，完整运行 105.351 → 114.304 ms （1.08×），查询微基准 44.654 → 38.608 ms。窗口内到达计数及最终状态的匹配结果见数值检查。
- GPU / shared：时间表从 10 增至 10000 条/源，完整运行 103.461 → 113.567 ms （1.10×），查询微基准 44.235 → 37.951 ms。窗口内到达计数及最终状态的匹配结果见数值检查。
- GPU，N=100、M=100：独立突触 144.329 ms，共享突触 130.883 ms；独立/共享时间比 1.10。
- GPU，N=100、M=10：批量连接 111.435 ms，每细胞一次连接 746.634 ms；完整模拟首次调用分别 1.871 / 6.285 s（含编译和执行）。

## 数值检查

CPU/GPU 配对 65；独立/共享突触配对 62；时间表长度配对 12；连接声明方式配对 8。
每个 worker 还检查查询计数、聚合权重和最终状态有限性。
配对最大电压差 0.169822693 mV；最大电导差 4.76837158e-07 µS。事件计数要求精确一致；状态 rtol=3e-4，电压 atol=2e-5 mV，电导 atol=2e-7 µS。未因观测结果放宽阈值。
结果：存在不匹配，不能声称所有配置在 CPU/GPU 上严格数值等价；下列条目保留具体误差。

- n1_m1_independent_active_ceec0210e3b4 CPU/GPU: final_voltage_mv differs (max absolute error 0.0260124207)
- n10_m1_independent_active_4b02032e68bf CPU/GPU: final_voltage_mv differs (max absolute error 0.0262908936)
- n10_m10_independent_active_4a9ba5ec4cfd CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n100_m1_independent_active_0d0e4daf7b8b CPU/GPU: final_voltage_mv differs (max absolute error 0.030380249)
- n100_m10_independent_active_43f29176fcb5 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922393799)
- n1_m1_shared_active_603728711089 CPU/GPU: final_voltage_mv differs (max absolute error 0.0260124207)
- n10_m1_shared_active_77f515b58319 CPU/GPU: final_voltage_mv differs (max absolute error 0.0262908936)
- n10_m10_shared_active_5389f4169fa3 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n100_m1_shared_active_7f2e797bbd93 CPU/GPU: final_voltage_mv differs (max absolute error 0.030380249)
- n100_m10_shared_active_7c49478121b0 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922393799)
- n10_m10_independent_active_ddd0c0270c04 CPU/GPU: final_voltage_mv differs (max absolute error 0.153831482)
- n10_m10_independent_active_191f8771a73f CPU/GPU: final_voltage_mv differs (max absolute error 0.00935935974)
- n10_m100_independent_active_1068705dc1d3 CPU/GPU: final_voltage_mv differs (max absolute error 0.0098361969)
- n10_m10_shared_active_2cecb34d030e CPU/GPU: final_voltage_mv differs (max absolute error 0.153831482)
- n10_m10_shared_active_561fbc370f3f CPU/GPU: final_voltage_mv differs (max absolute error 0.00935935974)
- n10_m100_shared_active_1cffb4a4808f CPU/GPU: final_voltage_mv differs (max absolute error 0.0098361969)
- n10_m10_independent_active_5033c37e39ce CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_independent_active_aed68a4eb6ee CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_independent_active_2e6e963e0e30 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_independent_active_e7547670d885 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_shared_active_8dd6c8f3e5c7 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_shared_active_dd0ea1c2dd25 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_shared_active_b706dd97b817 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_shared_active_41967d286da2 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n10_m10_independent_active_0f95979cf2c9 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922584534)
- n10_m10_shared_active_8b7e1613d871 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922584534)
- n10_m10_independent_active_67ecc0feb996 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n100_m10_independent_active_58c993c0dc98 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922393799)
- n10_m10_shared_active_28a2e75c6ea0 CPU/GPU: final_voltage_mv differs (max absolute error 0.00800323486)
- n100_m10_shared_active_3ef3b3316655 CPU/GPU: final_voltage_mv differs (max absolute error 0.00922393799)
- n1_m10_independent_unconnected_f63e4137d009 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n1_m10_independent_silent_6e185c1c9dae CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n10_m10_independent_unconnected_3b36a83f3958 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n10_m10_independent_silent_b65325fbb530 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n100_m10_independent_unconnected_9178e4691766 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n100_m10_independent_silent_7ce4420d202e CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n1_m10_shared_unconnected_379cc86bb164 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n1_m10_shared_silent_3d50ed71346f CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n10_m10_shared_unconnected_58253197623d CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n10_m10_shared_silent_25f6132a4d80 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n100_m10_shared_unconnected_a3c6c03aaa89 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n100_m10_shared_silent_efc94a318b6b CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n1_m10_independent_cell_only_a7dd838afed2 CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n10_m10_independent_cell_only_2def50e7fd5c CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
- n100_m10_independent_cell_only_8d392128a5be CPU/GPU: final_voltage_mv differs (max absolute error 0.169822693)
## 最小案例

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | independent | 默认 | 100 | 4.123 | 92.992 | 0.773/2.091 | 33.610/46.646 |
| 1 | 10 | shared | 默认 | 100 | 3.513 | 75.471 | 1.061/1.896 | 33.861/49.860 |
## 规模 sweep

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | independent | 默认 | 100 | 4.123 | 92.992 | 0.773/2.091 | 33.610/46.646 |
| 1 | 10 | shared | 默认 | 100 | 3.513 | 75.471 | 1.061/1.896 | 33.861/49.860 |
| 1 | 1 | independent | 默认 | 10 | 0.581 | 93.489 | 0.414/0.589 | 27.990/28.303 |
| 1 | 100 | independent | 默认 | 1000 | 56.530 | 103.821 | 3.558/5.686 | 44.064/33.495 |
| 10 | 1 | independent | 默认 | 100 | 7.010 | 95.915 | 0.786/1.975 | 34.182/44.213 |
| 10 | 10 | independent | 默认 | 1000 | 12.317 | 105.094 | 5.725/5.013 | 44.230/34.003 |
| 10 | 100 | independent | 默认 | 9999 | 59.260 | 113.709 | 26.770/19.195 | 44.246/37.339 |
| 100 | 1 | independent | 默认 | 1000 | 18.961 | 106.222 | 5.486/4.106 | 44.007/35.859 |
| 100 | 10 | independent | 默认 | 9999 | 53.900 | 111.435 | 26.574/19.264 | 44.908/37.469 |
| 100 | 100 | independent | 默认 | 99990 | 485.137 | 144.329 | 761.159/185.765 | 34.178/40.660 |
| 1 | 1 | shared | 默认 | 10 | 1.456 | 93.569 | 0.420/0.575 | 23.515/23.775 |
| 1 | 100 | shared | 默认 | 1000 | 9.012 | 84.359 | 5.496/5.806 | 44.913/31.436 |
| 10 | 1 | shared | 默认 | 100 | 6.856 | 96.516 | 0.779/2.102 | 33.403/46.407 |
| 10 | 10 | shared | 默认 | 1000 | 10.061 | 103.944 | 6.054/5.253 | 45.959/29.899 |
| 10 | 100 | shared | 默认 | 9999 | 31.035 | 108.669 | 26.696/22.919 | 44.979/35.483 |
| 100 | 1 | shared | 默认 | 1000 | 17.627 | 106.008 | 5.968/5.628 | 46.651/35.827 |
| 100 | 10 | shared | 默认 | 9999 | 33.922 | 108.717 | 26.725/18.404 | 50.607/37.259 |
| 100 | 100 | shared | 默认 | 99990 | 235.184 | 130.883 | 716.821/218.997 | 33.947/59.945 |
## 事件活动

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 10 | independent | 100 Hz, phase | 1000 | 12.317 | 105.094 | 5.725/5.013 | 44.230/34.003 |
| 10 | 100 | independent | 100 Hz, phase | 9999 | 59.260 | 113.709 | 26.770/19.195 | 44.246/37.339 |
| 10 | 10 | shared | 100 Hz, phase | 1000 | 10.061 | 103.944 | 6.054/5.253 | 45.959/29.899 |
| 10 | 100 | shared | 100 Hz, phase | 9999 | 31.035 | 108.669 | 26.696/22.919 | 44.979/35.483 |
| 10 | 10 | independent | 10 Hz, phase | 100 | 10.824 | 97.408 | 1.884/2.185 | 34.009/28.728 |
| 10 | 10 | independent | 1000 Hz, phase | 9998 | 25.831 | 105.287 | 17.565/27.312 | 42.972/38.778 |
| 10 | 10 | independent | 100 Hz, sync | 1000 | 12.444 | 104.910 | 5.395/5.614 | 41.820/33.994 |
| 10 | 10 | independent | 100 Hz, poisson | 1036 | 19.586 | 105.116 | 12.135/11.852 | 50.680/34.809 |
| 10 | 100 | independent | 10 Hz, phase | 1000 | 45.919 | 106.174 | 5.153/6.212 | 44.199/32.254 |
| 10 | 100 | independent | 1000 Hz, phase | 99990 | 728.689 | 114.010 | 598.306/626.100 | 31.992/43.231 |
| 10 | 100 | independent | 100 Hz, sync | 10000 | 62.613 | 113.437 | 26.759/19.268 | 50.259/37.621 |
| 10 | 100 | independent | 100 Hz, poisson | 9965 | 258.066 | 113.710 | 215.763/179.902 | 31.854/43.178 |
| 10 | 10 | shared | 10 Hz, phase | 100 | 8.999 | 96.514 | 2.033/2.526 | 38.642/50.089 |
| 10 | 10 | shared | 1000 Hz, phase | 9998 | 25.117 | 104.538 | 17.510/15.811 | 45.700/35.278 |
| 10 | 10 | shared | 100 Hz, sync | 1000 | 10.005 | 103.489 | 5.546/4.965 | 45.061/29.888 |
| 10 | 10 | shared | 100 Hz, poisson | 1036 | 18.511 | 104.921 | 12.083/9.617 | 44.431/31.253 |
| 10 | 100 | shared | 10 Hz, phase | 1000 | 18.012 | 102.511 | 5.329/10.101 | 44.326/44.562 |
| 10 | 100 | shared | 1000 Hz, phase | 99990 | 677.215 | 110.113 | 736.775/650.837 | 33.721/41.168 |
| 10 | 100 | shared | 100 Hz, sync | 10000 | 30.811 | 108.582 | 26.793/23.146 | 45.288/35.310 |
| 10 | 100 | shared | 100 Hz, poisson | 9965 | 179.152 | 108.931 | 202.691/193.938 | 33.602/41.836 |
## 时间表长度

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 10 | independent | K=10 | 1000 | 12.146 | 105.351 | 5.802/5.132 | 44.654/33.854 |
| 10 | 10 | independent | K=100 | 1000 | 24.479 | 105.217 | 16.660/16.017 | 44.137/38.711 |
| 10 | 10 | independent | K=1000 | 1000 | 474.229 | 115.192 | 654.609/439.869 | 33.752/44.520 |
| 10 | 10 | independent | K=10000 | 1000 | 1286.255 | 114.304 | 1048.087/1165.505 | 38.608/43.411 |
| 10 | 10 | shared | K=10 | 1000 | 10.022 | 103.461 | 6.133/5.321 | 44.235/29.788 |
| 10 | 10 | shared | K=100 | 1000 | 23.058 | 104.362 | 17.401/14.793 | 44.934/35.670 |
| 10 | 10 | shared | K=1000 | 1000 | 333.829 | 112.970 | 625.303/462.115 | 33.964/40.322 |
| 10 | 10 | shared | K=10000 | 1000 | 1073.142 | 113.567 | 1061.350/1121.618 | 37.951/40.141 |
## 延迟

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 10 | independent | zero | 1000 | 12.317 | 105.094 | 5.725/5.013 | 44.230/34.003 |
| 10 | 10 | shared | zero | 1000 | 10.061 | 103.944 | 6.054/5.253 | 45.959/29.899 |
| 10 | 10 | independent | fixed | 991 | 12.193 | 104.873 | 4.567/4.985 | 44.201/33.996 |
| 10 | 10 | independent | heterogeneous | 976 | 12.561 | 105.160 | 5.556/5.256 | 44.460/33.989 |
| 10 | 10 | shared | fixed | 991 | 10.213 | 103.970 | 5.512/5.691 | 45.120/30.269 |
| 10 | 10 | shared | heterogeneous | 976 | 10.021 | 103.589 | 5.713/5.334 | 44.101/29.918 |
## 连接声明

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 10 | independent | batched | 1000 | 12.317 | 105.094 | 5.725/5.013 | 44.230/34.003 |
| 100 | 10 | independent | batched | 9999 | 53.900 | 111.435 | 26.574/19.264 | 44.908/37.469 |
| 10 | 10 | shared | batched | 1000 | 10.061 | 103.944 | 6.054/5.253 | 45.959/29.899 |
| 100 | 10 | shared | batched | 9999 | 33.922 | 108.717 | 26.725/18.404 | 50.607/37.259 |
| 10 | 10 | independent | per_cell | 1000 | 55.686 | 163.638 | 5.684/40.644 | 41.214/85.891 |
| 100 | 10 | independent | per_cell | 9999 | 375.018 | 746.634 | 26.529/225.178 | 45.120/668.409 |
| 10 | 10 | shared | per_cell | 1000 | 11.848 | 153.913 | 5.128/58.142 | 44.967/73.251 |
| 100 | 10 | shared | per_cell | 9999 | 385.652 | 662.103 | 26.811/279.304 | 44.812/587.902 |
## 对照组

完整运行和查询/聚合均为中位数，单位 ms。`—` 表示无对应测量。

| N | M | 布局 | 配置 | 事件数 | CPU 完整 | GPU 完整 | CPU 查询/聚合 | GPU 查询/聚合 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | independent | active | 100 | 4.123 | 92.992 | 0.773/2.091 | 33.610/46.646 |
| 1 | 10 | shared | active | 100 | 3.513 | 75.471 | 1.061/1.896 | 33.861/49.860 |
| 10 | 10 | independent | active | 1000 | 12.317 | 105.094 | 5.725/5.013 | 44.230/34.003 |
| 100 | 10 | independent | active | 9999 | 53.900 | 111.435 | 26.574/19.264 | 44.908/37.469 |
| 10 | 10 | shared | active | 1000 | 10.061 | 103.944 | 6.054/5.253 | 45.959/29.899 |
| 100 | 10 | shared | active | 9999 | 33.922 | 108.717 | 26.725/18.404 | 50.607/37.259 |
| 1 | 10 | independent | unconnected | 0 | 3.483 | 90.667 | — | — |
| 1 | 10 | independent | silent | 0 | 4.322 | 92.793 | 0.776/1.873 | 31.925/41.020 |
| 10 | 10 | independent | unconnected | 0 | 9.440 | 93.535 | — | — |
| 10 | 10 | independent | silent | 0 | 12.465 | 105.168 | 5.929/5.497 | 45.083/33.957 |
| 100 | 10 | independent | unconnected | 0 | 33.095 | 98.868 | — | — |
| 100 | 10 | independent | silent | 0 | 52.908 | 110.782 | 26.809/19.256 | 43.881/37.805 |
| 1 | 10 | shared | unconnected | 0 | 1.296 | 75.229 | — | — |
| 1 | 10 | shared | silent | 0 | 3.383 | 75.782 | 0.916/1.997 | 34.092/49.454 |
| 10 | 10 | shared | unconnected | 0 | 6.775 | 91.598 | — | — |
| 10 | 10 | shared | silent | 0 | 10.079 | 103.374 | 4.908/5.959 | 44.940/29.946 |
| 100 | 10 | shared | unconnected | 0 | 15.157 | 97.253 | — | — |
| 100 | 10 | shared | silent | 0 | 33.845 | 108.798 | 26.771/18.427 | 44.924/37.361 |
| 1 | 10 | independent | cell_only | 0 | 1.133 | 54.741 | — | — |
| 10 | 10 | independent | cell_only | 0 | 3.377 | 54.719 | — | — |
| 100 | 10 | independent | cell_only | 0 | 7.920 | 58.525 | — | — |

## 构造、准备与首次调用

选取各布局最大的规模配置；全部配置及每次重复值见 summary.csv / 原始 JSON。单位 s。

| 设备 | N×M | 布局 | 构造（含源） | 其中源构造 | 初始化准备 | 完整模拟首次调用 |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | 100×100 | independent | 2.111 | 1.467 | 2.347 | 8.446 |
| cpu | 100×100 | shared | 1.360 | 1.296 | 1.302 | 8.239 |
| gpu | 100×100 | independent | 2.184 | 1.485 | 2.008 | 9.201 |
| gpu | 100×100 | shared | 1.628 | 1.491 | 1.263 | 9.210 |

## 精度补充诊断

CPU/GPU 分进程运行，所有精度使用同样的 CPU float32 随机源日程；该对照用于调查单精度电压差，不与主性能基线混合，也不改变主检查的阈值。

| 精度 | N×M | 布局 | 刺激 | CPU/GPU 最大电压差 (mV) | 事件计数一致 |
| --- | --- | --- | --- | --- | --- |
| 32 | 10×10 | independent | 10 Hz phase | 0.153831482 | True |
| 32 | 10×10 | shared | 10 Hz phase | 0.153831482 | True |
| 64 | 10×10 | independent | 10 Hz phase | 2.28794761e-12 | True |
| 64 | 10×10 | shared | 10 Hz phase | 2.28794761e-12 | True |

原始配置和结果保存在同一运行目录的 `diagnostics/precision32` 与 `diagnostics/precision64`。
主表的电压不匹配仍保留；精度诊断仅覆盖上表配置，不能外推为所有形态和求解器均已验证。

代表配置的跨设备电压差在双精度下显著减小，支持误差来自单精度细胞数值计算的判断。事件计数检查独立通过；尚未把电压差定位到某个求解器运算，不能将其归因于 NetStim 路由错误。

## GPU trace：执行粒度与归因边界

trace 在预热和正式计时后单独采集。下面是 GPU 事件时长之和，不是完整运行墙钟时间；关闭 command buffer 的运行仅供归因诊断，不纳入基准速度比较。

| command buffer | 布局 | GPU 事件数 | 每步约事件数 | GPU 事件时长之和 (ms) | 匹配 scope 事件数 |
| --- | --- | --- | --- | --- | --- |
| default | independent | 64012 | 16.00 | 101.430 | 0 |
| default | shared | 52012 | 13.00 | 82.495 | 0 |
| disabled | independent | 64012 | 16.00 | 129.636 | 0 |
| disabled | shared | 52012 | 13.00 | 105.693 | 0 |

independent 的前五个 GPU kernel 分组（按设备事件时长之和）：

| kernel 名称 | 次数 | 累计 ms | 平均 µs |
| --- | --- | --- | --- |
| `loop_gather_fusion` | 4000 | 11.951 | 2.988 |
| `memcpy32_post` | 8000 | 10.487 | 1.311 |
| `loop_select_slice_fusion` | 4000 | 8.873 | 2.218 |
| `loop_divide_fusion` | 4000 | 8.451 | 2.113 |
| `wrapped_add` | 4000 | 6.787 | 1.697 |

[原始 trace 解析结果](../artifacts/baseline/diagnostics/trace/independent_summary.json)

shared 的前五个 GPU kernel 分组（按设备事件时长之和）：

| kernel 名称 | 次数 | 累计 ms | 平均 µs |
| --- | --- | --- | --- |
| `loop_gather_fusion` | 4000 | 11.254 | 2.813 |
| `memcpy32_post` | 8000 | 10.443 | 1.305 |
| `loop_select_slice_fusion` | 4000 | 8.621 | 2.155 |
| `loop_divide_fusion` | 4000 | 8.094 | 2.023 |
| `wrapped_add` | 4000 | 6.822 | 1.705 |

[原始 trace 解析结果](../artifacts/baseline/diagnostics/trace/shared_summary.json)

本环境的 XPlane 解析未匹配到 BrainCell named scope，因此不提供 event/积分的精确占比。融合 kernel 名称本身不足以证明它只执行某个阶段；可确认的是大量短 GPU 事件反复执行。

## 图与原始记录

[可选本地图表：scaling](../artifacts/baseline/scaling.png)

[可选本地图表：schedule](../artifacts/baseline/schedule.png)

[原始 manifest](../artifacts/baseline/manifest.json) · [完整 CSV](../artifacts/baseline/summary.csv)

## 环境与限制

- CPU：cpu，x64=False，Python 3.11.15，版本 `{"jax": "0.10.1", "jaxlib": "0.10.1", "brainstate": "0.5.4", "brainunit": "0.3.0"}`。
- GPU：NVIDIA A100-SXM4-80GB，x64=False，Python 3.11.15，版本 `{"jax": "0.10.1", "jaxlib": "0.10.1", "brainstate": "0.5.4", "brainunit": "0.3.0"}`。
- 宿主 CPU：Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz；进程可用逻辑 CPU 数：128。

GPU 初始占用快照：

```text
index, name, utilization.gpu [%], memory.used [MiB]
0, NVIDIA A100-SXM4-80GB, 66 %, 932 MiB
1, NVIDIA A100-SXM4-80GB, 100 %, 1758 MiB
2, NVIDIA A100-SXM4-80GB, 100 %, 73539 MiB
3, NVIDIA A100-SXM4-80GB, 79 %, 73539 MiB
4, NVIDIA A100-SXM4-80GB, 77 %, 73539 MiB
5, NVIDIA A100-SXM4-80GB, 100 %, 73539 MiB
6, NVIDIA A100-SXM4-80GB, 100 %, 73539 MiB
7, NVIDIA A100-SXM4-80GB, 0 %, 14 MiB
```

构造阶段包含首次随机生成及初始化操作的编译，不能等同于纯 Python 拓扑构造。进程内分配器的内存统计是设备观测值，不代表单个 event kernel 的独占峰值。
静默对照保留时间表形状，但编译器仍可能优化常量；差值只表示该对照实验的整体变化。其他 GPU/CPU 上同时运行的工作会影响宿主机负载；小差异需结合重复值判断。


## 原工作流命令记录

以下保留原 README 的运行命令与示例，包含运行者的解释器路径和设备选择。它们不代表各命令均已执行，实际执行范围以本页实测记录为准，也不构成重新运行授权。其他用户使用公共 README 的参数化入口。

```bash
# 最小案例：两种布局 × CPU/GPU
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.benchmark \
  --suite smoke --devices both --gpu 7

# 全部实验：65 个配置，130 个设备/配置，去除实验间重复项
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.benchmark \
  --suite all --devices both --gpu 7 --out benchmarks/performance/synapse_events/artifacts/baseline

# 输出表格、图、配对检查和 Markdown
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.report

# 自定义规模；大规模通过显式参数启用
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.benchmark \
  --suite scaling --cells 100,1000 --inputs 10,100 --devices gpu --gpu 7 \
  --out benchmarks/performance/synapse_events/artifacts/large

# 独立 profiler 运行；trace 在正式计时结束后采集
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.benchmark \
  --suite smoke --devices gpu --gpu 7 --trace \
  --out benchmarks/performance/synapse_events/artifacts/trace

# 从 XPlane 检查现有 BrainCell scope
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.profiling.parse_xplane_trace \
  --trace-dir benchmarks/performance/synapse_events/artifacts/trace --mode leaf

# CPU/GPU 双精度诊断；选用主实验中电压差较大的低频配置
/home/swl/anaconda3/envs/braincell_311/bin/python -m benchmarks.performance.synapse_events.benchmark \
  --suite scaling --cells 10 --inputs 10 --rate-hz 10 --precision 64 \
  --out benchmarks/performance/synapse_events/artifacts/baseline/diagnostics/precision64
# 对应单精度诊断将 --precision 改为 32，--out 的 precision64 改为 precision32
```
