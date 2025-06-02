# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import brainunit as u

import braincell
import brainstate


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


class MChannel(braincell.Channel):
    root_type = braincell.SingleCompartment

    def __init__(self, size, g_max=1.3 * (u.mS / u.cm ** 2), E=-54.387 * u.mV, T=u.celsius2kelvin(37)):
        super().__init__(size)
        self.g_max = g_max
        self.E = E
        self.T = T
        self.phi = 2.3 ** ((u.kelvin2celsius(T) - 23.) / 10)  # temperature scaling factor

    def f_p_alpha(self, V):
        return self.phi * 1e-4 * 9 / u.math.exprel(-(V / u.mV + 30.) / 9.)

    def f_p_beta(self, V):
        return self.phi * 1e-4 * 9 / u.math.exp((V / u.mV + 30.) / 9.)

    def current(self, V):
        return self.g_max * self.p.value * (self.E - V)

    def compute_derivative(self, V):
        # Update the channel state based on the membrane potential V and time step dt
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        p_inf = alpha / (alpha + beta)
        p_tau = 1. / (alpha + beta) * u.ms
        self.p.derivative = (p_inf - self.p.value) / p_tau

    def init_state(self, V):
        alpha = self.f_p_alpha(V)
        beta = self.f_p_beta(V)
        p_inf = alpha / (alpha + beta)
        self.p = braincell.DiffEqState(p_inf)


class GABAa(brainstate.nn.Synapse):
    def __init__(self, in_size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms):
        super().__init__(in_size)
        self.g_max = g_max
        self.tau = tau

    def init_state(self):
        self.g = brainstate.HiddenState(u.math.zeros(self.out_size))

    def g_gaba(self, V):
        return 2. * (1. + u.math.tanh(V / (4.0 * u.mV))) / u.ms

    def update(self, pre_V):
        dg = lambda g: self.g_gaba(pre_V) * (1. - g) - g / self.tau
        self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
        return self.g.value


class MSNCell(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=NaChannel(size, g_max=100. * (u.mS / u.cm ** 2)))

        self.k = braincell.ion.PotassiumFixed(size, E=-100. * u.mV)
        self.k.add(IK=KChannel(size, g_max=80. * (u.mS / u.cm ** 2)))

        self.IL = braincell.channel.IL(size, E=-67. * u.mV, g_max=0.1 * (u.mS / u.cm ** 2))

        self.IM = MChannel(size, g_max=1.3 * (u.mS / u.cm ** 2), E=-67. * u.mV)


class StraitalNetwork(brainstate.nn.Module):
    def __init__(self, size):
        super().__init__()

        self.pop = MSNCell(size, solver='ind_exp_euler')
        self.syn = GABAa(size, g_max=0.1 * (u.mS / u.cm ** 2), tau=13.0 * u.ms)
        self.conn = brainstate.nn.CurrentProj(
            comm=brainstate.nn.AllToAll(size, size, w_init=0.1 * (u.mS / u.cm ** 2)),
            out=brainstate.nn.COBA(E=-80. * u.mV),
            post=self.pop
        )

    def update(self, *args, **kwargs):
        self.conn(self.syn.g.value)
        self.pop()
        self.syn(self.pop.V.value)


if __name__ == '__main__':
    net = StraitalNetwork(100)



