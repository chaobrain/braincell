# Synapse comparisons

这里保存 postsynaptic mechanism、event routing 与外部 event source 的 BrainCell / NEURON 对比。

- `exp_syn_compare.ipynb`：单指数突触的基础数值对比。
- `hh_2x2_neuron_compare.ipynb`：小型 HH 网络中的突触传递对比。
- `netstim_heterogeneous_compare.ipynb`：异质 NetStim、Connection weight/delay 与 ExpSyn 的端到端对比。

NetStim 的随机数实现不要求与 NEURON 共用同一随机流。第三个 notebook 先由 BrainCell 生成确定的事件时刻，再在 NEURON 中逐事件重放，因此比较的是 event routing、delay、突触动力学与膜积分，而不是两个随机数生成器是否逐位相同。
