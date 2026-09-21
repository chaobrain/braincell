# HH 三条切片：结果图

基于 2026-09-10 的 12 组完整配对数据绘图，新增模型执行次数为 **0**。固定 H200 NVL、B=S=16、40 ms、dt=0.025 ms、T=1600、float64、staggered solver、recursive backsub；完整环境、计数、误差和局限见[补测报告](hh-crossover-h200-20260910-supplement.md)。C=81、Ntheta=162/243 的 BPTT 来自补测，RTRL 复用初测，未新增独立复测。

## Nx–Ntheta 三条切片

横坐标是每条轨迹的 active Nx，纵坐标是每个 seed 的可训练 Ntheta。蓝/橙/绿分别表示只解冻 Leak、Leak+K、Leak+K+Na；所有 CV 始终 paint 完整 HH。三条线分别满足 Ntheta=Nx/4、Nx/2、3Nx/4。

每个标记对应实测配置，数字 r 为 RTRL/BPTT 稳态时间比。C=1 的三个点在左上局部放大。连线只显示切片方向，不代表已测出连续性能边界。



## 稳态时间与速度比

横坐标仍为 Nx，三种颜色沿用上述切片；左图实线 BPTT、虚线 RTRL，纵坐标为对数时间。误差棒为每个 worker 的 5 个正式样本的 Q25–Q75，较小的波动被点标记遮住，不代表波动为零。右图灰带为 r=0.9–1.1 的接近候选区。



全部已测点 r=0.2054–0.4762，RTRL 更快；速度比不随 Nx 单调变化，目前没有可二分的胜负变化区间。

## 内存与编译

左图为 XLA temporary 分析值，非实测进程峰值显存；右图为梯度 kernel tracing/lowering/compile 总时间，不含 target 准备。两者纵坐标均采用对数刻度。内存的 argument/output/alias 和 RTRL logical carry 完整数值保留在补测报告与 CSV 中。



图表由本地 artifact 的 `analysis/figures/` 保存，结果页不提交生成图片。

## 生成方式

使用 [plot_hh_crossover.py](../../../analysis/hh_crossover/plot.py) 读取本地 ignored `artifacts/hh_crossover_20260910_completed/raw/paired_results.csv` 及 `raw/trials/*.json`，没有导入或执行模型。生成后检查图形布局；PNG/SVG/PDF 与 `plot_sources.json` 保存在 artifact 的 `analysis/figures/`，不随 Git 提供。

从仓库根目录执行：

```bash
python -m benchmarks.performance.optim_gradient_scaling.analysis.hh_crossover.plot \
  benchmarks/performance/optim_gradient_scaling/artifacts/hh_crossover_20260910_completed \
  --context 'H200 NVL | B=16, S=16 | T=1600, dt=0.025 ms | float64 | completed coarse scan'
```

脚本拒绝未通过校验的配对、缺失切片、重复点和不符合切片维度的点；同时核对配对 median 与保存样本一致。检查采用现有数据渲染、来源哈希与文档链接核对；未运行模型测试。
