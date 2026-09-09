# Synapse Network CPU Gradient Timings

本页保留独立双向 population 梯度计时；它与多 CV scaling 是不同实验，不能合并比较。
模型及正确性验证见 [突触与网络验证](../../../../docs/design/optim/current/results/synapse-network-learning.md)。
源码精确提交、完整依赖版本及未列出的硬件信息未记录。原始材料未保存，以下记录独立可读。

## 独立 CPU 计时

来源为本次整理之前、2026-09-07 会话中的只读独立进程测量；未保存独立原始 artifact，
也没有可引用的 benchmark commit。下表是会话记录，不伪装成现有 scaling CLI 输出。

- Intel Xeon Platinum 8358P，affinity 为逻辑 CPU 0/1/2/3；没有声明独占机器。
- Python 3.11、JAX 0.8.0、CPU、float64、scatter、固定异质 delay，dt=0.025 ms。
- 模型来自同一 bidirectional.build，grouped=True/False 分别为 15/45 个标量坐标。
- 每种方法和配置独立进程、串行测量；OMP/OPENBLAS/MKL_NUM_THREADS=1。
- engine.prepare 后，把当前 roots 与形状 (T,5)、值为 -60 mV 数值的 target 作为动态参数，
  分别编译引擎 _bptt/_rtrl；它们是本次测量使用的实验私有方法，不是公共 API。
- 首次执行同步完成后，再同步计时 7 次，取中位数；每次含 reset 与整段 loss/gradient，
  不含构建/trace、编译、Adam、目标生成和进程启动。
- 800/8000 步对应 20/200 ms，长轨迹不追加 clamp，后段没有新增刺激；不是持续放电负载。
- BPTT 未使用 checkpoint；RTRL 返回逐步 losses，但不输出 sensitivity history。

| 参数数 | 步数 | BPTT median | RTRL median | BPTT 工作内存 | RTRL 工作内存 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 800 | 58.000 ms | 52.999 ms | 4.49 MiB | 0.078 MiB |
| 15 | 8000 | 516.739 ms | 357.917 ms | 44.70 MiB | 0.408 MiB |
| 45 | 800 | 54.897 ms | 93.528 ms | 4.49 MiB | 0.146 MiB |
| 45 | 8000 | 544.125 ms | 636.154 ms | 44.70 MiB | 0.475 MiB |

工作内存为 XLA memory_analysis 的 argument + output + temporary - alias；本次 alias 均为 0。
它不是 RSS、GPU 峰值、纯 sensitivity carry 或分配器保留总量。以下保存字节口径与编译结果，
避免后续将 MiB 舍入值当成新的原始数据：

| P/T | 方法 | Compile (s) | Temporary bytes | Argument bytes | Output bytes | Host peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 15/800 | BPTT | 8.071531 | 4667656 | 32120 | 6664 | 1211.44 |
| 15/800 | RTRL | 6.164166 | 43160 | 32120 | 6664 | 1196.21 |
| 15/8000 | BPTT | 8.053768 | 46485256 | 320120 | 64264 | 1254.66 |
| 15/8000 | RTRL | 5.946149 | 43160 | 320120 | 64264 | 1184.76 |
| 45/800 | BPTT | 8.686402 | 4669176 | 32360 | 6904 | 1246.57 |
| 45/800 | RTRL | 6.244573 | 113384 | 32360 | 6904 | 1219.29 |
| 45/8000 | BPTT | 8.425463 | 46486776 | 320360 | 64504 | 1291.93 |
| 45/8000 | RTRL | 6.078932 | 113384 | 320360 | 64504 | 1219.36 |

Host peak RSS 用 Linux ru_maxrss，包含导入和编译。导入后 baseline RSS 约 472 MiB；
构建/prepare 另外约 9.55-12.12 s。两种方法整个进程都约 1.2 GiB，不能说进程 RAM 小了百倍。
四组配对最大绝对梯度差依次为 9.313e-10、2.736e-9、2.983e-10、6.112e-10。

在这些配置内，15 根时 RTRL 速度接近或更快，45 根时慢约 17%-70%；工作内存约小
31-110 倍。15 根/800 步的计时范围有重叠，不能据此宣传稳定加速。
延长时间时 RTRL temporary 不变，而总工作内存仍因输入和逐步 loss 增长。
优势取决于全网状态 H 与独立参数 P，不是每 Cell 的局部 hidden 数；不能外推到任意
多 CV、多 population、其他后端或 checkpoint BPTT。

复测需按上述协议建立新的独立测量，保存新环境与原始结果；本页不提供不存在的 CLI。
