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

#
# Implementation of the paper:
#
# - Brette, R., Rudolph, M., Carnevale, T., Hines, M., Beeman, D., Bower, J. M., et al. (2007),
#   Simulation of networks of spiking neurons: a review of tools and strategies., J. Comput. Neurosci., 23, 3, 349–98
#

import brainpy
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt

import braincell
from braincell import mech

V_th = -20. * u.mV
area = 20000 * u.um ** 2  # membrane area of one neuron
diam = u.math.sqrt(area / u.math.pi)  # a cylinder as long as it is wide


def HH(in_size):
    # Conductances are densities and the membrane capacitance is 1 uF/cm^2;
    # the synaptic weights below are totals, which the cell divides by its area.
    cell = braincell.Cell.point(
        in_size,
        length=diam,
        radius=diam / 2,
        V_init=braintools.init.Uniform(-70. * u.mV, -60. * u.mV),
        solver='ind_exp_euler',
    )
    cell.paint(
        mech.Ion('SodiumFixed', E=50. * u.mV),
        mech.Channel('Na_TM1991', g_max=100. * u.mS / u.cm ** 2, V_sh=-63. * u.mV),
        mech.Ion('PotassiumFixed', E=-90 * u.mV),
        mech.Channel('K_TM1991', g_max=30. * u.mS / u.cm ** 2, V_sh=-63. * u.mV),
        mech.Channel('IL', E=-60. * u.mV, g_max=5. * u.nS / u.cm ** 2),
    )
    return cell


class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc = 3200
        self.n_inh = 800
        self.num = self.n_exc + self.n_inh
        self.N = HH(self.num)

        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=0.02, conn_weight=6. * u.nS),
            syn=brainpy.state.Expon(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA(E=0. * u.mV),
            post=self.N
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=0.02, conn_weight=67. * u.nS),
            syn=brainpy.state.Expon(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA(E=-80. * u.mV),
            post=self.N
        )

    def update(self, t):
        with brainstate.environ.context(t=t):
            spk = self.N.spike.value
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            spk = self.N(0. * u.nA)
            return spk


# network
net = EINet()
brainstate.nn.init_all_states(net)

# simulation
with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 100. * u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(net.update, times, pbar=brainstate.transform.ProgressBar(10))

# visualization
t_indices, n_indices = u.math.where(spikes)
plt.scatter(times[t_indices], n_indices, s=1)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.show()
