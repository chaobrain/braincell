from neuron import h
from neuron import load_mechanisms
import numpy as np
import matplotlib.pyplot as plt
h.load_file('stdrun.hoc')
# --------------------------------------------------
# 1. 载入你编译好的 mod 机制
#    这里填的是“运行 nrnivmodl 的目录”，不是 x86_64 目录本身
# --------------------------------------------------
# --------------------------------------------------
# 2. 建一个单室神经元
# --------------------------------------------------
soma = h.Section(name="soma")
soma.L = 10      # um
soma.diam = 100/np.pi   # um
soma.nseg = 1
soma.cm = 1      # uF/cm2
soma.Ra = 100    # ohm*cm

# 插入被动漏电流，避免只有 Kv 通道时完全不稳定/不合理
soma.insert("pas")
for seg in soma:
    seg.pas.g = 1e-4
    seg.pas.e = -65

# 插入你的 Kv 通道
soma.insert("Kv")
for seg in soma:
    seg.Kv.gbar = 0.00   # S/cm2，可自己调
    seg.Kv.v12 = 25      # mV
    seg.Kv.q = 9         # mV

soma.ek = -80

# --------------------------------------------------
# 3. 加一个电流刺激
# --------------------------------------------------
stim = h.IClamp(soma(0.5))
stim.delay = 0   # ms
stim.dur = 100    # ms
stim.amp = 0.01     # nA，可调大一点看响应

# --------------------------------------------------
# 4. 记录变量
# --------------------------------------------------
t_vec = h.Vector().record(h._ref_t)
v_vec = h.Vector().record(soma(0.5)._ref_v)
ik_vec = h.Vector().record(soma(0.5)._ref_ik)
n_vec = h.Vector().record(soma(0.5).Kv._ref_n)

# --------------------------------------------------
# 5. 运行仿真
# --------------------------------------------------
h.finitialize(-65)
h.tstop = 100
h.run()

# --------------------------------------------------
# 6. 画图
# --------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(t_vec, v_vec)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane potential (mV)")
plt.title("Single-compartment neuron with Kv channel")
plt.tight_layout()
plt.show()
plt.savefig('v_fig')

plt.figure(figsize=(8, 4))
plt.plot(t_vec, ik_vec)
plt.xlabel("Time (ms)")
plt.ylabel("ik (mA/cm2)")
plt.title("Potassium current")
plt.tight_layout()
plt.show()
plt.savefig('ik_fig')

plt.figure(figsize=(8, 4))
plt.plot(t_vec, n_vec)
plt.xlabel("Time (ms)")
plt.ylabel("n")
plt.title("Kv activation variable")
plt.tight_layout()
plt.show()
plt.savefig('n_fig')


# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import brainstate
import braintools
import brainunit as u
import braincell

class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='ind_exp_euler'):
        super().__init__(size, solver=solver, V_initializer = braintools.init.Uniform(-65 * u.mV, -65. * u.mV))

        # self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        # self.na.add(INa=braincell.channel.INa_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-80. * u.mV)
        self.k.add(IK=braincell.channel.IK_Kv_test(size, g_max = 0* (u.mS / u.cm ** 2)))

        self.IL = braincell.channel.IL(size, E=-65 * u.mV, g_max= 0.1  * (u.mS / u.cm ** 2))


hh = HH(1, solver='rk4')
hh.init_state()


def step_fun(t):
    with brainstate.environ.context(t=t):
        spike = hh.update(1000 * u.nA / u.cm ** 2)
    return hh.V.value


with brainstate.environ.context(dt=0.025 * u.ms):
    times = u.math.arange(0. * u.ms, 100 * u.ms, brainstate.environ.get_dt())
    vs = brainstate.transform.for_loop(step_fun, times)

plt.figure(figsize=(8, 4))
plt.plot(times, u.math.squeeze(vs))
plt.show()
plt.savefig('braincell')


# 你的数据
v0 = np.asarray(v_vec)[:-1]  # 参考数据
vs = np.asarray(u.math.squeeze(vs) / u.mV)  # 与 v0 对齐成 (T,)

# 计算各种误差指标
print(f"v0 shape: {v0.shape}")
print(f"vs shape: {vs.shape}")

error = v0 - vs

# 1. 平均绝对误差 (MAE)
mae = np.mean(np.abs(error))
print(f"MAE: {mae:.6f} mV")

# 2. 均方根误差 (RMSE)
rmse = np.sqrt(np.mean(error**2))
print(f"RMSE: {rmse:.6f} mV")

# 3. 最大绝对误差
max_error = np.max(np.abs(error))
print(f"Max Error: {max_error:.6f} mV")

# 4. 相对误差（百分比）
relative_error = np.mean(np.abs(error / (np.abs(v0) + 1e-10))) * 100
print(f"Relative Error: {relative_error:.2f}%")

# 5. 误差分布统计
print(f"\nError Statistics:")
print(f"  Mean: {np.mean(error):.6f} mV")
print(f"  Std: {np.std(error):.6f} mV")
print(f"  Min: {np.min(error):.6f} mV")
print(f"  Max: {np.max(error):.6f} mV")
