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

import brainstate
import brainunit as u
import matplotlib.pyplot as plt

import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.IK_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))


hh = HH(1, solver='exp_euler')
hh = HH(1, solver='ind_exp_euler')
hh.init_state()


def step_fun(t):
    with brainstate.environ.context(t=t):
        spike = hh.update(10 * u.nA / u.cm ** 2)
    return hh.V.value


with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 100 * u.ms, brainstate.environ.get_dt())
    vs = brainstate.compile.for_loop(step_fun, times)

plt.plot(times, u.math.squeeze(vs))
plt.show()
