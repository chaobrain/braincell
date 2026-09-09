# 预定事件查询第一版实测

原始运行材料仅保存在运行者本地的 artifacts，未随 Git 提供；本页的正文与表格随仓库维护。原始目录、日志或图像链接仅为可选核查入口，缺失时不影响读取已保存的数据。

四种查询方式使用相同 spike 和连接，对照现有 NetStim.event_count。实验只包含事件查询、乘权归约与 ExpSyn 电导推进，未包含完整 Cell 电压求解或 Network 接入。

记录创建于 2026-09-08T08:42:15.023563+00:00。成功 72 个设备/场景，CPU/GPU 输入与计数配对 36 组；记录或配对失败 0 项。

窗口 100.0 ms，dt=0.025 ms，float32，预热 2 次，测量 5 次。每方法清理编译缓存，GPU 同步等待；稳态表不包含准备和首次编译执行。

## 主要比较

- cpu：36 个场景中，synapse 最小中位数的计数为 current 3、scan 8、cursor 1、bucket 24。这是本矩阵的描述，未作噪声显著性检验。
- cpu，K=10000、窗口内 1000 个到达事件：current 1192.222 ms、scan 1062.350 ms、cursor 40.382 ms、bucket 5.631 ms。
- gpu：36 个场景中，synapse 最小中位数的计数为 current 25、scan 11、cursor 0、bucket 0。这是本矩阵的描述，未作噪声显著性检验。
- gpu，K=10000、窗口内 1000 个到达事件：current 44.765 ms、scan 39.198 ms、cursor 217.722 ms、bucket 157.538 ms。

## 环境与验证

CPU 型号：Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz。GPU 为 worker 实际选择设备，物理设备编号见 manifest 协议。

- cpu: cpu；JAX 0.10.1，BrainState 0.5.4，BrainUnit 0.3.0。
- gpu: NVIDIA A100-SXM4-80GB；JAX 0.10.1，BrainState 0.5.4，BrainUnit 0.3.0。

每个成功 worker 已逐步逐连接比较全部计数，逐步比较全部目标的 ExpSyn 电导；各候选在同一设备上对照 current，计数必须完全相同，电导 rtol=3e-5、atol=1e-7 uS。跨设备检查相同 source 和计数的 SHA256，不把这项检查宣称为跨设备电压一致性。

## 稳态运行时间

单位 ms，均为完整窗口的中位数。query 仅查询；synapse 为查询、乘权归约和 ExpSyn 的统一事件后精确衰减。

| 设备 | N/M/K | 布局/活动/扇出/delay/block | 模式 | current | scan | cursor | bucket |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| cpu | 1/10/10 | independent/phase/1/zero/128 | query | 1.212 | 0.983 | 2.583 | 2.459 |
| cpu | 1/10/10 | independent/phase/1/zero/128 | synapse | 3.191 | 2.347 | 4.272 | 4.356 |
| gpu | 1/10/10 | independent/phase/1/zero/128 | query | 37.030 | 37.650 | 156.171 | 122.483 |
| gpu | 1/10/10 | independent/phase/1/zero/128 | synapse | 45.309 | 45.338 | 188.167 | 151.139 |
| cpu | 1/1/10 | independent/phase/1/zero/128 | query | 0.576 | 0.480 | 0.490 | 2.097 |
| cpu | 1/1/10 | independent/phase/1/zero/128 | synapse | 0.677 | 0.577 | 0.592 | 4.479 |
| gpu | 1/1/10 | independent/phase/1/zero/128 | query | 23.675 | 25.197 | 129.937 | 115.005 |
| gpu | 1/1/10 | independent/phase/1/zero/128 | synapse | 26.163 | 25.875 | 135.803 | 121.154 |
| cpu | 1/1/10 | shared/phase/1/zero/128 | query | 0.564 | 0.478 | 0.471 | 1.912 |
| cpu | 1/1/10 | shared/phase/1/zero/128 | synapse | 0.653 | 0.577 | 0.571 | 4.334 |
| gpu | 1/1/10 | shared/phase/1/zero/128 | query | 27.257 | 27.369 | 123.760 | 115.346 |
| gpu | 1/1/10 | shared/phase/1/zero/128 | synapse | 29.541 | 29.451 | 133.645 | 117.035 |
| cpu | 1/10/10 | shared/phase/1/zero/128 | query | 0.448 | 1.023 | 2.754 | 2.567 |
| cpu | 1/10/10 | shared/phase/1/zero/128 | synapse | 2.242 | 2.822 | 4.298 | 4.994 |
| gpu | 1/10/10 | shared/phase/1/zero/128 | query | 37.806 | 37.587 | 157.850 | 123.930 |
| gpu | 1/10/10 | shared/phase/1/zero/128 | synapse | 34.768 | 35.170 | 175.625 | 138.932 |
| cpu | 1/100/10 | independent/phase/1/zero/128 | query | 4.736 | 4.902 | 5.081 | 4.452 |
| cpu | 1/100/10 | independent/phase/1/zero/128 | synapse | 4.850 | 4.467 | 5.487 | 5.132 |
| gpu | 1/100/10 | independent/phase/1/zero/128 | query | 54.379 | 41.475 | 192.300 | 151.980 |
| gpu | 1/100/10 | independent/phase/1/zero/128 | synapse | 33.123 | 56.914 | 197.711 | 167.402 |
| cpu | 1/100/10 | shared/phase/1/zero/128 | query | 4.516 | 4.905 | 5.312 | 5.056 |
| cpu | 1/100/10 | shared/phase/1/zero/128 | synapse | 4.838 | 5.193 | 5.926 | 5.286 |
| gpu | 1/100/10 | shared/phase/1/zero/128 | query | 47.120 | 34.208 | 184.873 | 140.319 |
| gpu | 1/100/10 | shared/phase/1/zero/128 | synapse | 43.782 | 33.693 | 187.920 | 139.994 |
| cpu | 10/1/10 | independent/phase/1/zero/128 | query | 1.200 | 1.014 | 2.779 | 2.500 |
| cpu | 10/1/10 | independent/phase/1/zero/128 | synapse | 2.468 | 2.421 | 5.251 | 4.301 |
| gpu | 10/1/10 | independent/phase/1/zero/128 | query | 38.024 | 34.093 | 149.786 | 111.194 |
| gpu | 10/1/10 | independent/phase/1/zero/128 | synapse | 45.556 | 45.399 | 175.641 | 133.792 |
| cpu | 10/1/10 | shared/phase/1/zero/128 | query | 1.197 | 0.986 | 1.482 | 0.643 |
| cpu | 10/1/10 | shared/phase/1/zero/128 | synapse | 2.879 | 2.196 | 4.707 | 3.284 |
| gpu | 10/1/10 | shared/phase/1/zero/128 | query | 41.284 | 41.216 | 158.441 | 107.354 |
| gpu | 10/1/10 | shared/phase/1/zero/128 | synapse | 55.117 | 55.139 | 182.230 | 130.316 |
| cpu | 10/10/10 | independent/phase/1/zero/128 | query | 4.566 | 3.520 | 4.980 | 3.395 |
| cpu | 10/10/10 | independent/phase/1/zero/128 | synapse | 4.682 | 3.185 | 5.290 | 5.761 |
| gpu | 10/10/10 | independent/phase/1/zero/128 | query | 51.264 | 38.744 | 194.932 | 135.417 |
| gpu | 10/10/10 | independent/phase/1/zero/128 | synapse | 33.159 | 58.795 | 209.944 | 149.888 |
| cpu | 10/10/10 | shared/phase/1/zero/128 | query | 4.976 | 4.550 | 5.262 | 4.888 |
| cpu | 10/10/10 | shared/phase/1/zero/128 | synapse | 4.604 | 4.451 | 5.343 | 5.206 |
| gpu | 10/10/10 | shared/phase/1/zero/128 | query | 45.484 | 39.544 | 203.100 | 144.755 |
| gpu | 10/10/10 | shared/phase/1/zero/128 | synapse | 32.412 | 54.458 | 218.894 | 162.888 |
| cpu | 10/100/10 | independent/phase/1/zero/128 | query | 24.295 | 25.492 | 18.194 | 5.019 |
| cpu | 10/100/10 | independent/phase/1/zero/128 | synapse | 20.818 | 17.809 | 23.050 | 10.414 |
| gpu | 10/100/10 | independent/phase/1/zero/128 | query | 47.152 | 48.882 | 277.682 | 183.540 |
| gpu | 10/100/10 | independent/phase/1/zero/128 | synapse | 34.669 | 46.157 | 297.273 | 205.499 |
| cpu | 10/100/10 | shared/phase/1/zero/128 | query | 24.572 | 25.447 | 18.439 | 6.072 |
| cpu | 10/100/10 | shared/phase/1/zero/128 | synapse | 22.069 | 19.371 | 26.594 | 13.734 |
| gpu | 10/100/10 | shared/phase/1/zero/128 | query | 44.197 | 44.594 | 284.072 | 191.507 |
| gpu | 10/100/10 | shared/phase/1/zero/128 | synapse | 39.843 | 47.687 | 262.191 | 196.623 |
| cpu | 100/1/10 | independent/phase/1/zero/128 | query | 3.503 | 5.406 | 5.082 | 4.711 |
| cpu | 100/1/10 | independent/phase/1/zero/128 | synapse | 2.995 | 5.241 | 5.511 | 5.809 |
| gpu | 100/1/10 | independent/phase/1/zero/128 | query | 45.103 | 34.534 | 193.989 | 131.013 |
| gpu | 100/1/10 | independent/phase/1/zero/128 | synapse | 33.062 | 47.367 | 209.759 | 145.707 |
| cpu | 100/1/10 | shared/phase/1/zero/128 | query | 5.103 | 5.230 | 5.379 | 4.434 |
| cpu | 100/1/10 | shared/phase/1/zero/128 | synapse | 4.518 | 4.829 | 5.300 | 4.042 |
| gpu | 100/1/10 | shared/phase/1/zero/128 | query | 50.816 | 38.639 | 200.854 | 143.005 |
| gpu | 100/1/10 | shared/phase/1/zero/128 | synapse | 33.018 | 53.317 | 218.136 | 159.071 |
| cpu | 100/10/10 | independent/phase/1/zero/128 | query | 24.349 | 25.568 | 18.150 | 5.112 |
| cpu | 100/10/10 | independent/phase/1/zero/128 | synapse | 20.851 | 17.846 | 23.052 | 9.999 |
| gpu | 100/10/10 | independent/phase/1/zero/128 | query | 52.823 | 44.274 | 258.735 | 184.095 |
| gpu | 100/10/10 | independent/phase/1/zero/128 | synapse | 34.719 | 47.484 | 296.392 | 202.038 |
| cpu | 100/10/10 | shared/phase/1/zero/128 | query | 24.316 | 25.434 | 18.499 | 5.685 |
| cpu | 100/10/10 | shared/phase/1/zero/128 | synapse | 17.413 | 14.785 | 22.165 | 9.670 |
| gpu | 100/10/10 | shared/phase/1/zero/128 | query | 45.941 | 44.051 | 284.063 | 194.906 |
| gpu | 100/10/10 | shared/phase/1/zero/128 | synapse | 35.015 | 47.525 | 296.436 | 199.577 |
| cpu | 100/100/10 | independent/phase/1/zero/128 | query | 606.774 | 637.675 | 197.682 | 15.366 |
| cpu | 100/100/10 | independent/phase/1/zero/128 | synapse | 176.602 | 152.257 | 254.734 | 64.535 |
| gpu | 100/100/10 | independent/phase/1/zero/128 | query | 44.198 | 50.426 | 296.399 | 192.637 |
| gpu | 100/100/10 | independent/phase/1/zero/128 | synapse | 37.716 | 47.559 | 309.598 | 202.123 |
| cpu | 100/100/10 | shared/phase/1/zero/128 | query | 675.320 | 629.441 | 194.076 | 15.332 |
| cpu | 100/100/10 | shared/phase/1/zero/128 | synapse | 209.746 | 184.778 | 324.527 | 99.948 |
| gpu | 100/100/10 | shared/phase/1/zero/128 | query | 36.481 | 52.711 | 304.708 | 207.449 |
| gpu | 100/100/10 | shared/phase/1/zero/128 | synapse | 54.157 | 63.106 | 324.388 | 225.316 |
| cpu | 10/10/100 | independent/phase/1/zero/128 | query | 17.215 | 17.090 | 4.878 | 4.389 |
| cpu | 10/10/100 | independent/phase/1/zero/128 | synapse | 16.604 | 15.909 | 5.526 | 5.526 |
| gpu | 10/10/100 | independent/phase/1/zero/128 | query | 56.932 | 48.118 | 189.506 | 131.345 |
| gpu | 10/10/100 | independent/phase/1/zero/128 | synapse | 38.837 | 33.334 | 205.243 | 146.966 |
| cpu | 10/10/1000 | independent/phase/1/zero/128 | query | 669.148 | 668.759 | 5.288 | 3.257 |
| cpu | 10/10/1000 | independent/phase/1/zero/128 | synapse | 275.313 | 422.336 | 5.955 | 4.297 |
| gpu | 10/10/1000 | independent/phase/1/zero/128 | query | 43.584 | 43.975 | 195.001 | 137.049 |
| gpu | 10/10/1000 | independent/phase/1/zero/128 | synapse | 43.049 | 34.498 | 210.629 | 151.910 |
| cpu | 10/10/10000 | independent/phase/1/zero/128 | query | 1113.537 | 888.546 | 38.542 | 4.663 |
| cpu | 10/10/10000 | independent/phase/1/zero/128 | synapse | 1192.222 | 1062.350 | 40.382 | 5.631 |
| gpu | 10/10/10000 | independent/phase/1/zero/128 | query | 37.900 | 47.611 | 195.916 | 142.363 |
| gpu | 10/10/10000 | independent/phase/1/zero/128 | synapse | 44.765 | 39.198 | 217.722 | 157.538 |
| cpu | 10/100/10 | independent/sparse/1/zero/128 | query | 24.335 | 25.425 | 12.613 | 5.501 |
| cpu | 10/100/10 | independent/sparse/1/zero/128 | synapse | 20.810 | 17.852 | 17.514 | 9.539 |
| gpu | 10/100/10 | independent/sparse/1/zero/128 | query | 59.460 | 44.896 | 185.365 | 142.729 |
| gpu | 10/100/10 | independent/sparse/1/zero/128 | synapse | 34.679 | 47.622 | 200.966 | 158.996 |
| cpu | 10/100/10 | independent/random/1/zero/128 | query | 24.087 | 25.224 | 17.801 | 5.538 |
| cpu | 10/100/10 | independent/random/1/zero/128 | synapse | 20.528 | 17.589 | 22.717 | 9.825 |
| gpu | 10/100/10 | independent/random/1/zero/128 | query | 50.866 | 50.751 | 263.687 | 184.035 |
| gpu | 10/100/10 | independent/random/1/zero/128 | synapse | 34.642 | 53.636 | 286.495 | 206.960 |
| cpu | 10/100/10 | independent/sync/1/zero/128 | query | 24.304 | 25.469 | 11.239 | 5.474 |
| cpu | 10/100/10 | independent/sync/1/zero/128 | synapse | 20.733 | 17.921 | 16.020 | 9.384 |
| gpu | 10/100/10 | independent/sync/1/zero/128 | query | 43.286 | 42.694 | 168.309 | 133.365 |
| gpu | 10/100/10 | independent/sync/1/zero/128 | synapse | 34.648 | 44.224 | 187.510 | 148.742 |
| cpu | 10/100/320 | independent/burst/1/zero/128 | query | 774.133 | 779.430 | 51.452 | 5.011 |
| cpu | 10/100/320 | independent/burst/1/zero/128 | synapse | 651.688 | 658.868 | 69.615 | 9.796 |
| gpu | 10/100/320 | independent/burst/1/zero/128 | query | 32.661 | 52.885 | 176.426 | 182.152 |
| gpu | 10/100/320 | independent/burst/1/zero/128 | synapse | 41.011 | 35.279 | 194.599 | 199.253 |
| cpu | 10/100/320 | independent/dense/1/zero/128 | query | 837.369 | 782.783 | 98.962 | 4.924 |
| cpu | 10/100/320 | independent/dense/1/zero/128 | synapse | 692.644 | 667.693 | 105.645 | 10.314 |
| gpu | 10/100/320 | independent/dense/1/zero/128 | query | 40.148 | 45.171 | 284.707 | 209.258 |
| gpu | 10/100/320 | independent/dense/1/zero/128 | synapse | 40.928 | 35.379 | 298.845 | 204.634 |
| cpu | 10/100/10 | independent/silent/1/zero/128 | query | 24.204 | 25.507 | 11.141 | 5.081 |
| cpu | 10/100/10 | independent/silent/1/zero/128 | synapse | 20.873 | 17.963 | 15.575 | 9.117 |
| gpu | 10/100/10 | independent/silent/1/zero/128 | query | 54.124 | 54.619 | 168.564 | 118.369 |
| gpu | 10/100/10 | independent/silent/1/zero/128 | synapse | 34.674 | 57.973 | 182.249 | 133.591 |
| cpu | 10/10/10 | independent/phase/1/heterogeneous/128 | query | 5.139 | 4.269 | 5.450 | 4.688 |
| cpu | 10/10/10 | independent/phase/1/heterogeneous/128 | synapse | 5.095 | 3.976 | 6.801 | 5.203 |
| gpu | 10/10/10 | independent/phase/1/heterogeneous/128 | query | 53.941 | 40.312 | 194.971 | 149.271 |
| gpu | 10/10/10 | independent/phase/1/heterogeneous/128 | synapse | 33.152 | 56.219 | 214.700 | 165.116 |
| cpu | 10/10/10 | independent/phase/10/zero/128 | query | 24.511 | 25.432 | 12.739 | 5.764 |
| cpu | 10/10/10 | independent/phase/10/zero/128 | synapse | 20.796 | 18.033 | 17.305 | 9.463 |
| gpu | 10/10/10 | independent/phase/10/zero/128 | query | 43.243 | 42.680 | 193.183 | 150.451 |
| gpu | 10/10/10 | independent/phase/10/zero/128 | synapse | 34.635 | 44.111 | 208.495 | 166.807 |
| cpu | 10/10/10 | independent/phase/10/heterogeneous/128 | query | 24.296 | 25.434 | 18.068 | 5.487 |
| cpu | 10/10/10 | independent/phase/10/heterogeneous/128 | synapse | 20.808 | 17.852 | 22.720 | 9.711 |
| gpu | 10/10/10 | independent/phase/10/heterogeneous/128 | query | 52.746 | 42.904 | 272.843 | 175.980 |
| gpu | 10/10/10 | independent/phase/10/heterogeneous/128 | synapse | 34.657 | 43.617 | 284.063 | 218.375 |
| cpu | 10/10/10 | independent/phase/100/zero/128 | query | 564.550 | 466.066 | 362.046 | 14.687 |
| cpu | 10/10/10 | independent/phase/100/zero/128 | synapse | 178.111 | 151.019 | 300.274 | 63.833 |
| gpu | 10/10/10 | independent/phase/100/zero/128 | query | 36.255 | 54.187 | 194.931 | 147.628 |
| gpu | 10/10/10 | independent/phase/100/zero/128 | synapse | 37.797 | 48.124 | 209.607 | 169.807 |
| cpu | 10/10/10 | independent/phase/100/heterogeneous/128 | query | 640.841 | 510.112 | 194.280 | 15.352 |
| cpu | 10/10/10 | independent/phase/100/heterogeneous/128 | synapse | 175.929 | 151.339 | 223.939 | 64.670 |
| gpu | 10/10/10 | independent/phase/100/heterogeneous/128 | query | 44.189 | 44.531 | 283.399 | 213.573 |
| gpu | 10/10/10 | independent/phase/100/heterogeneous/128 | synapse | 37.832 | 48.092 | 298.873 | 233.880 |
| cpu | 10/100/10 | independent/sparse/1/zero/32 | query | 24.354 | 25.447 | 18.075 | 5.032 |
| cpu | 10/100/10 | independent/sparse/1/zero/32 | synapse | 20.736 | 17.851 | 17.293 | 9.278 |
| gpu | 10/100/10 | independent/sparse/1/zero/32 | query | 44.052 | 45.529 | 193.848 | 146.691 |
| gpu | 10/100/10 | independent/sparse/1/zero/32 | synapse | 34.686 | 45.045 | 206.848 | 150.495 |
| cpu | 10/100/10 | independent/sparse/1/zero/512 | query | 24.276 | 25.472 | 12.685 | 5.457 |
| cpu | 10/100/10 | independent/sparse/1/zero/512 | synapse | 20.684 | 17.934 | 17.283 | 9.740 |
| gpu | 10/100/10 | independent/sparse/1/zero/512 | query | 44.078 | 44.579 | 195.701 | 141.311 |
| gpu | 10/100/10 | independent/sparse/1/zero/512 | synapse | 34.669 | 44.256 | 209.666 | 161.545 |
| cpu | 10/100/320 | independent/burst/1/zero/32 | query | 776.568 | 781.178 | 51.407 | 5.506 |
| cpu | 10/100/320 | independent/burst/1/zero/32 | synapse | 645.022 | 664.111 | 69.552 | 10.622 |
| gpu | 10/100/320 | independent/burst/1/zero/32 | query | 38.949 | 45.205 | 177.876 | 340.681 |
| gpu | 10/100/320 | independent/burst/1/zero/32 | synapse | 42.440 | 35.343 | 193.965 | 317.796 |
| cpu | 10/100/320 | independent/burst/1/zero/512 | query | 786.604 | 774.773 | 51.673 | 5.554 |
| cpu | 10/100/320 | independent/burst/1/zero/512 | synapse | 698.203 | 669.901 | 69.466 | 9.912 |
| gpu | 10/100/320 | independent/burst/1/zero/512 | query | 35.166 | 67.479 | 178.444 | 145.724 |
| gpu | 10/100/320 | independent/burst/1/zero/512 | synapse | 40.940 | 35.340 | 193.327 | 149.116 |

## 准备、内存与重用

CSV 的 one_use_ms 包含源构建、冷准备、初始游标和首次执行；reuse_100_ms 再加 99 次稳态运行。break-even 在候选稳态更快时估算需要多少次重用才能抵消冷启动差额，未计入运行间修改参数导致的重建。

prepare_cold_ms 拆为主机校验/收集、设备量化冷执行、设备到主机、桶排序、上传与打包。量化冷执行包含算子编译和输入传输，不是纯算术时间。current 的常量时间表转入编译程序的成本留在首次执行中。

query_array_bytes 与 cursor_bytes 是持久设备调度数组的精确字节数；source_host_bytes 是源时间表和 mask。arrival_table_bytes 和 gathered_source_bytes 记录准备中间数组的逻辑大小，不能相加称为峰值。bucket 持久存储为 O(E+T+block)，但准备仍有 O(CK) 临时工作；allocator 统计保留在 JSON，不能解释为每内核峰值。
current 的 query_array_bytes=0 表示没有新增显式调度数组，不代表没有设备内存：实际生产路径的时间表常量包含在编译程序中，此列不统计 executable 常量。

代表场景的 synapse 成本（ms）；设备调度列只列显式数组和游标，单位 KiB：

| 设备 | N/M/K/活动 | 方法 | 冷准备 | 首次执行 | 一次使用 | 100次重用 | 设备调度 KiB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 1/10/10/phase | current | 0.48 | 194.98 | 1316.09 | 1631.96 | 0.00 |
| cpu | 1/10/10/phase | scan | 1233.13 | 148.43 | 2520.30 | 2752.63 | 0.39 |
| cpu | 1/10/10/phase | cursor | 1329.43 | 212.69 | 2840.18 | 3263.08 | 0.51 |
| cpu | 1/10/10/phase | bucket | 1280.64 | 192.02 | 2607.10 | 3038.36 | 16.52 |
| gpu | 1/10/10/phase | current | 0.36 | 354.06 | 1446.76 | 5932.35 | 0.00 |
| gpu | 1/10/10/phase | scan | 1054.57 | 203.57 | 2354.40 | 6842.88 | 0.39 |
| gpu | 1/10/10/phase | cursor | 1147.54 | 395.92 | 2897.71 | 21526.25 | 0.51 |
| gpu | 1/10/10/phase | bucket | 1104.27 | 355.82 | 2549.80 | 17512.60 | 16.52 |
| cpu | 100/100/10/phase | current | 2.43 | 407.45 | 1997.47 | 19481.11 | 0.00 |
| cpu | 100/100/10/phase | scan | 1405.01 | 341.15 | 3351.01 | 18424.50 | 390.62 |
| cpu | 100/100/10/phase | cursor | 1490.62 | 550.39 | 3834.30 | 29052.92 | 507.81 |
| cpu | 100/100/10/phase | bucket | 1465.97 | 277.44 | 3345.13 | 9734.05 | 406.71 |
| gpu | 100/100/10/phase | current | 1.49 | 370.59 | 1973.50 | 5707.38 | 0.00 |
| gpu | 100/100/10/phase | scan | 1024.88 | 233.23 | 2880.50 | 7588.80 | 390.62 |
| gpu | 100/100/10/phase | cursor | 1220.08 | 623.98 | 3789.26 | 34439.51 | 507.81 |
| gpu | 100/100/10/phase | bucket | 1178.32 | 410.10 | 3185.85 | 23196.00 | 406.71 |
| cpu | 10/10/10000/phase | current | 10.45 | 1547.61 | 2588.65 | 120618.59 | 0.00 |
| cpu | 10/10/10000/phase | scan | 1185.50 | 1386.55 | 3621.09 | 108793.73 | 3906.25 |
| cpu | 10/10/10000/phase | cursor | 1468.59 | 249.77 | 3005.65 | 7003.48 | 3907.42 |
| cpu | 10/10/10000/phase | bucket | 1238.97 | 189.70 | 2451.11 | 3008.59 | 20.04 |
| gpu | 10/10/10000/phase | current | 9.12 | 474.86 | 1513.34 | 5945.07 | 0.00 |
| gpu | 10/10/10000/phase | scan | 1150.85 | 195.51 | 2391.93 | 6272.51 | 3906.25 |
| gpu | 10/10/10000/phase | cursor | 1324.10 | 425.10 | 3120.82 | 24675.27 | 3907.42 |
| gpu | 10/10/10000/phase | bucket | 1234.49 | 364.32 | 2622.36 | 18218.66 | 20.04 |
| cpu | 10/100/320/burst | current | 8.81 | 912.63 | 1944.15 | 66461.30 | 0.00 |
| cpu | 10/100/320/burst | scan | 1308.79 | 846.20 | 3193.64 | 68421.53 | 1250.00 |
| cpu | 10/100/320/burst | cursor | 1462.12 | 263.76 | 3002.69 | 9894.54 | 1261.72 |
| cpu | 10/100/320/burst | bucket | 1407.19 | 224.07 | 2660.62 | 3630.39 | 1266.13 |
| gpu | 10/100/320/burst | current | 7.12 | 401.81 | 1384.06 | 5444.17 | 0.00 |
| gpu | 10/100/320/burst | scan | 1163.45 | 190.22 | 2344.48 | 5837.07 | 1250.00 |
| gpu | 10/100/320/burst | cursor | 1286.22 | 418.36 | 3009.10 | 22274.40 | 1261.72 |
| gpu | 10/100/320/burst | bucket | 1231.18 | 372.02 | 2572.54 | 22298.57 | 1266.13 |

## 解释边界

- scan 对照用于分离预计算收益。cursor 与 bucket 的数组为动态 JIT 参数，current 保留实际生产源码的常量处理；这是接口候选实验，不是完全相同 HLO 的算法复杂度证明。
- cursor 按连接推进，仍每步检查连接；同一步多重事件会增加批量循环。bucket 按块读紧凑区间，再形成每连接计数；后续稀疏直接投递可能有不同结果。
- burst 是显式受控时间表（每源每 10 ms 同时到达 32 条），由原有 NetStim.event_count 消费；dense 在同一默认 100 ms 窗口具有相同事件数但均匀分布。它们不是 NetStim 标准参数能直接表达的同一种随机过程。
- query 与 synapse 独立编译，不能相加为完整模型的时间占比。这里的 ExpSyn 精确衰减消费者不复刻 Cell solver，生产加速需要下一阶段验证。

## 复现

以下为历史复现示例，具体执行次数与设备以本页记录为准；新测量须先确认执行清单。报告命令现默认写入输入 artifacts，审阅后再更新本页。

从仓库根目录运行：

```bash
python -m benchmarks.performance.synapse_events.query_benchmark --suite all --devices both --gpu 7
python -m benchmarks.performance.synapse_events.query_report
```

[方案](../../../../docs/design/synapse/proposals/event-delivery-optimization.md) · [固定版本参考](../../../../docs/design/synapse/references/scheduled-event-delivery.md) · [实验接口](../../../../docs/design/synapse/current/experimental-scheduled-events.md) · [驱动与命令](../README.md)

## 测量版本与边界修正

原始 72 个 worker 完成后，dense 的间隔由 100/K 改为 100/(K+1)，避免末尾事件取整到窗口外；仅重测受影响的 CPU/GPU 两个 worker，原记录保存在 superseded/。修正后 dense 与 burst 的窗口到达数均为 320000。

最终实验接口追加了时钟数值范围检查（包括 float32 大绝对步号），原矩阵全部满足该条件。query/reset 的 AST 与原测量版本完全相同；其他场景的准备时间未为这项额外主机校验重测。manifest 的每条记录标明 source_snapshot，baseline 与 corrected 两份完整测量源码及哈希随 artifacts 保留，protocol.sources 对应原始扫描版本。

## 测量源码标识

批次创建时间：`2026-09-08T08:42:15.023563+00:00`。以下为 `queries` manifest 中保存的测量源码 SHA-256；代表该批次，后续修正/补测的适用范围以正文说明为准。当前仓库中的报告代码和文档可能已更新，不能将当前工作区视为原始测量快照。

| 测量源码 | SHA-256 |
| --- | --- |
| `benchmarks/performance/synapse_events/query_benchmark.py` | `6bafe37eeb3aad870ed325b17e6f7c8aa84e3b8b33d294b782e33a5bac67cbc0` |
| `braincell/experimental/scheduled_events.py` | `f95874bb7de3891b14a5380411da7957fa8c12c44f4e5a8b5238fa1e9fad9a80` |
| `braincell/network/event.py` | `40c1288eebdb05f27f533bd455f4c5902353b0649fb5d1b5ec8ce97de8bfbccb` |
| `braincell/synapse/exponential.py` | `a9048621cbbfd80499c36677e2d59caa4dc8305c01a922bf8fd9d78e8951ed56` |
