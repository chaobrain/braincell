# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- McCarthy, M. M., et al. "Striatal origin of the pathologic beta oscillations in Parkinson's disease."
  Proceedings of the national academy of sciences 108.28 (2011): 11620-11625.
"""

import brainpy
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import braincell

brainstate.environ.set(dt=0.1 * u.ms)


class NaChannel(braincell.channel.INa_p3q_markov):
    def f_p_alpha(self, V):
        return 0.32 * 4. / u.math.exprel(-(V / u.mV + 54.) / 4.)

    def f_p_beta(self, V):
        return 0.28 * 5. / u.math.exprel((V / u.mV + 27.) / 5.)

    def f_q_alpha(self, V):
        return 0.128 * u.math.exp(-(V / u.mV + 50.) / 18.)

    def f_q_beta(self, V):
        return 4. / (1 + u.math.exp(-(V / u.mV + 27.) / 5.))


class KChannel(braincell.channel.IK_p4_markov):
    def f_p_alpha(self, V):
        return 0.032 * 5. / u.math.exprel(-(V / u.mV + 52.) / 5.)

    def f_p_beta(self, V):
        return 0.5 * u.math.exp(-(V / u.mV + 57.) / 40.)


class MChannel(braincell.channel.PotassiumChannel):
    def __init__(self, size, g_max=1.3 * (u.mS / u.cm ** 2), E=-95. * u.mV, T=u.celsius2kelvin(37)):
        super().__init__(size)
        self.g_max = g_max
        self.E = E
        self.T = T
        self.phi = 2.3 ** ((u.kelvin2celsius(T) - 23.) / 10)  # temperature scaling factor

    def f_p_alpha(self, V):
        return self.phi * 1e-4 * 9 / u.math.exprel(-(V / u.mV + 30.) / 9.)

    def f_p_beta(self, V):
        return self.phi * 1e-4 * 9 / u.math.exp((V / u.mV + 30.) / 9.)

    def current(self, V, K: braincell.IonInfo):
        return self.g_max * self.p.value * (K.E - V)

    def compute_derivative(self, V, K: braincell.IonInfo):
        # Update the channel state based on the membrane potential V and time step dt
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        p_inf = alpha / (alpha + beta)
        p_tau = 1. / (alpha + beta) * u.ms
        self.p.derivative = (p_inf - self.p.value) / p_tau

    def init_state(self, V, K: braincell.IonInfo, *args, **kwargs):
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        p_inf = alpha / (alpha + beta)
        self.p = braincell.DiffEqState(p_inf)


class GABAa(brainpy.state.Synapse):
    def __init__(self, in_size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms):
        super().__init__(in_size)
        self.g_max = g_max
        self.tau = tau

    def init_state(self, **kwargs):
        self.g = brainstate.HiddenState(u.math.zeros(self.out_size))

    def g_gaba(self, V):
        return 2. * (1. + u.math.tanh(V / (4.0 * u.mV))) / u.ms

    def update(self, pre_V):
        dg = lambda g: self.g_gaba(pre_V) * (1. - g) - g / self.tau
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        return self.g.value


class MSNCell(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4', g_M=1.3 * (u.mS / u.cm ** 2)):
        super().__init__(size, solver=solver, C=1.0 * u.uF / u.cm ** 2)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=NaChannel(size, g_max=100. * (u.mS / u.cm ** 2)))

        self.k = braincell.ion.PotassiumFixed(size, E=-100. * u.mV)
        self.k.add(IK=KChannel(size, g_max=80. * (u.mS / u.cm ** 2)))
        self.k.add(IM=MChannel(size, g_max=g_M))

        self.IL = braincell.channel.IL(size, E=-67. * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))


class StraitalNetwork(brainstate.nn.Module):
    def __init__(self, size, g_M=1.3 * (u.mS / u.cm ** 2)):
        super().__init__()

        self.pop = MSNCell(size, solver='ind_exp_euler', g_M=g_M)
        self.syn = GABAa(size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms)
        self.conn = brainpy.state.CurrentProj(
            # comm=brainstate.nn.FixedNumConn(size, size, 0.3, 0.1 / (size * 0.3) * (u.mS / u.cm ** 2)),
            comm=brainstate.nn.AllToAll(size, size, w_init=0.1 / size * (u.mS / u.cm ** 2)),
            out=brainpy.state.COBA(E=-80. * u.mV),
            post=self.pop,
        )

    def update(self, x=0. * u.nA / u.cm ** 2):
        self.conn(self.syn.g.value)
        spk = self.pop(x)
        self.syn(self.pop.V.value)
        return spk


def try_single_neuron():
    neuron = MSNCell(1, solver='ind_exp_euler', g_M=1.3 * (u.mS / u.cm ** 2))
    brainstate.nn.init_all_states(neuron)

    def step_run(i):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i):
            spk = neuron((0.12 + brainstate.random.randn() * 1.26) * (u.uA / u.cm ** 2))
            return neuron.V.value

    indices = np.arange(10000 * u.ms / brainstate.environ.get_dt(), dtype=np.int32)
    V = brainstate.transform.for_loop(step_run, indices)

    plt.plot(indices * brainstate.environ.get_dt(), V)
    plt.show()


def try_network_model():
    net = StraitalNetwork(100)
    brainstate.nn.init_all_states(net)
    duration = 10000 * u.ms
    indices = np.arange(duration / brainstate.environ.get_dt(), dtype=np.int32)

    def step_run(i):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i):
            inp = (0.12 + brainstate.random.randn() * 1.4) * (u.uA / u.cm ** 2)
            spk = net.update(inp)
            current = net.conn.out(net.pop.V.value)
            return spk, u.math.sum(current)

    spk, current = brainstate.transform.for_loop(step_run, indices)

    fig, gs = braintools.visualize.get_figure(2, 1, 4, 12)
    fig.add_subplot(gs[0, 0])
    plt.plot(current)
    fig.add_subplot(gs[1, 0])
    times, indices = u.math.where(spk)
    plt.scatter(times, indices, s=1)
    plt.xlim(-1, duration / u.ms + 1)
    plt.show()


if __name__ == '__main__':
    try_single_neuron()
    try_network_model()
