# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


import brainunit as bu
import matplotlib.pyplot as plt
import braintools as bts

import dendritex as dx

cat = dx.channels.ICaT_HP1992(1)
caht = dx.channels.ICaHT_HM1992(1)

vs = bu.math.arange(-100 * bu.mV, 0 * bu.mV, 0.1 * bu.mV)

fig, gs = bts.visualize.get_figure(1, 2, 3., 4.5)

q_inf = cat.f_q_inf(vs)
p_inf = cat.f_p_inf(vs)

fig.add_subplot(gs[0, 0])
plt.plot(vs / bu.mV, q_inf, label='q_inf')
plt.plot(vs / bu.mV, p_inf, label='p_inf')
plt.legend()
plt.fill_between([-80, -60], 1., alpha=0.2)
plt.title('Low-threshold Calcium Channel')
plt.xlabel('mV')
# plt.show()

q_inf = caht.f_q_inf(vs)
p_inf = caht.f_p_inf(vs)
fig.add_subplot(gs[0, 1])
plt.plot(vs / bu.mV, q_inf, label='q_inf')
plt.plot(vs / bu.mV, p_inf, label='p_inf')
plt.fill_between([-60, -40], 1., alpha=0.2)
plt.legend()
plt.xlabel('mV')
plt.title('High-threshold Calcium Channel')
plt.show()
