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

import braincell

s = u.siemens / u.cm ** 2


class INa(braincell.channel.SodiumChannel):
    def __init__(self, size, g_max):
        super().__init__(size)

        self.g_max = braintools.init.param(g_max, self.varshape)

    def init_state(self, V, Na: braincell.IonInfo, batch_size: int = None):
        self.m = braincell.DiffEqState(self.m_inf(V))
        self.h = braincell.DiffEqState(self.h_inf(V))

    def compute_derivative(self, V, Na: braincell.IonInfo):
        self.m.derivative = (self.m_alpha(V) * (1 - self.m.value) - self.m_beta(V) * self.m.value) / u.ms
        self.h.derivative = (self.h_alpha(V) * (1 - self.h.value) - self.h_beta(V) * self.h.value) / u.ms

    def current(self, V, Na: braincell.IonInfo):
        return self.g_max * self.m.value ** 3 * self.h.value * (Na.E - V)

    # m channel
    m_alpha = lambda self, V: 1. / u.math.exprel(-(V / u.mV + 40.) / 10.)  # nan
    m_beta = lambda self, V: 4. * u.math.exp(-(V / u.mV + 65.) / 18.)
    m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

    # h channel
    h_alpha = lambda self, V: 0.07 * u.math.exp(-(V / u.mV + 65.) / 20.)
    h_beta = lambda self, V: 1. / (1. + u.math.exp(-(V / u.mV + 35.) / 10.))
    h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))


class IK(braincell.channel.PotassiumChannel):
    def __init__(self, size, g_max):
        super().__init__(size)
        self.g_max = braintools.init.param(g_max, self.varshape)

    def init_state(self, V, K: braincell.IonInfo, batch_size: int = None):
        self.n = braincell.DiffEqState(self.n_inf(V))

    def compute_derivative(self, V, K: braincell.IonInfo):
        self.n.derivative = (self.n_alpha(V) * (1 - self.n.value) - self.n_beta(V) * self.n.value) / u.ms

    def current(self, V, K: braincell.IonInfo):
        return self.g_max * self.n.value ** 4 * (K.E - V)

    n_alpha = lambda self, V: 0.1 / u.math.exprel(-(V / u.mV + 55.) / 10.)
    n_beta = lambda self, V: 0.125 * u.math.exp(-(V / u.mV + 65.) / 80.)
    n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))


class ThreeCompartmentHH(braincell.MultiCompartment):
    def __init__(self, n_neuron: int, g_na, g_k):
        super().__init__(
            size=(n_neuron, 3),
            connection=((0, 1), (1, 2)),
            Ra=100. * u.ohm * u.cm,
            cm=1.0 * u.uF / u.cm ** 2,
            diam=(12.6157, 1., 1.) * u.um,
            L=(12.6157, 200., 400.) * u.um,
            V_th=20. * u.mV,
            V_initializer=braintools.init.Constant(-65 * u.mV),
            spk_fun=braintools.surrogate.ReluGrad(),
        )

        self.IL = braincell.channel.IL(self.size, E=(-54.3, -65., -65.) * u.mV, g_max=[0.0003, 0.001, 0.001] * s)

        self.na = braincell.ion.SodiumFixed(self.size, E=50. * u.mV)
        self.na.add(INa=INa(self.size, g_max=(g_na, 0., 0.) * s))

        self.k = braincell.ion.PotassiumFixed(self.size, E=-77. * u.mV)
        self.k.add(IK=IK(self.size, g_max=(g_k, 0., 0.) * s))

    def step_run(self, t, inp):
        braincell.rk4_step(self, t, inp)
        return self.V.value, self.spike.value
