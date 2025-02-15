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

"""
Implementation of the following models in the paper:

- Li, Guoshi, Craig S. Henriquez, and Flavio Fröhlich. “Unified thalamic model generates
  multiple distinct oscillations with state-dependent entrainment by stimulation.”
  PLoS computational biology 13.10 (2017): e1005797.
"""

import time

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt

import braincell


class ThalamusNeuron(braincell.neurons.SingleCompartment):
    def update(self, I_ext=0. * u.nA / u.cm ** 2):
        brainstate.augment.vmap(
            lambda: braincell.exp_euler_step(self, brainstate.environ.get('t'), I_ext),
            in_states=self.states()
        )()
        return self.post_integral(I_ext)

    def compute_derivative(self, x=0. * u.nA):
        return super().compute_derivative(x * self.area)

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            self.update(inp)
            return self.V.value


class HTC(ThalamusNeuron):
    def __init__(self, size, gKL=0.01 * (u.mS / u.cm ** 2), V_initializer=brainstate.init.Constant(-65. * u.mV)):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV)

        self.area = 1e-3 / (2.9e-4 * u.cm ** 2)

        self.na = braincell.ions.SodiumFixed(size, E=50. * u.mV)
        self.na.add_elem(INa=braincell.channels.INa_Ba2002(size, V_sh=-30 * u.mV))

        self.k = braincell.ions.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add_elem(IKL=braincell.channels.IK_Leak(size, g_max=gKL))
        self.k.add_elem(IDR=braincell.channels.IKDR_Ba2002(size, V_sh=-30. * u.mV, phi=0.25))

        self.ca = braincell.ions.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add_elem(ICaL=braincell.channels.ICaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaN=braincell.channels.ICaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaT=braincell.channels.ICaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaHT=braincell.channels.ICaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add_elem(IAHP=braincell.channels.IAHP_De1994(size, g_max=0.3 * (u.mS / u.cm ** 2)))

        self.Ih = braincell.channels.Ih_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
        self.IL = braincell.channels.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)


class RTC(ThalamusNeuron):
    def __init__(self, size, gKL=0.01 * (u.mS / u.cm ** 2), V_initializer=brainstate.init.Constant(-65. * u.mV)):
        super().__init__(size, V_initializer=V_initializer, V_th=20 * u.mV)

        self.area = 1e-3 / (2.9e-4 * u.cm ** 2)

        self.na = braincell.ions.SodiumFixed(size)
        self.na.add_elem(INa=braincell.channels.INa_Ba2002(size, V_sh=-40 * u.mV))

        self.k = braincell.ions.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add_elem(IDR=braincell.channels.IKDR_Ba2002(size, V_sh=-40 * u.mV, phi=0.25))
        self.k.add_elem(IKL=braincell.channels.IK_Leak(size, g_max=gKL))

        self.ca = braincell.ions.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add_elem(ICaL=braincell.channels.ICaL_IS2008(size, g_max=0.3 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaN=braincell.channels.ICaN_IS2008(size, g_max=0.6 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaT=braincell.channels.ICaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaHT=braincell.channels.ICaHT_HM1992(size, g_max=0.6 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add_elem(IAHP=braincell.channels.IAHP_De1994(size, g_max=0.1 * (u.mS / u.cm ** 2)))

        self.Ih = braincell.channels.Ih_HM1992(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV)
        self.IL = braincell.channels.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV)


class IN(ThalamusNeuron):
    def __init__(self, size, V_initializer=brainstate.init.Constant(-70. * u.mV)):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV)

        self.area = 1e-3 / (1.7e-4 * u.cm ** 2)

        self.na = braincell.ions.SodiumFixed(size)
        self.na.add_elem(INa=braincell.channels.INa_Ba2002(size, V_sh=-30 * u.mV))

        self.k = braincell.ions.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add_elem(IDR=braincell.channels.IKDR_Ba2002(size, V_sh=-30 * u.mV, phi=0.25))
        self.k.add_elem(IKL=braincell.channels.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))

        self.ca = braincell.ions.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add_elem(ICaN=braincell.channels.ICaN_IS2008(size, g_max=0.1 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaHT=braincell.channels.ICaHT_HM1992(size, g_max=2.5 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add_elem(IAHP=braincell.channels.IAHP_De1994(size, g_max=0.2 * (u.mS / u.cm ** 2)))

        self.IL = braincell.channels.IL(size, g_max=0.0075 * (u.mS / u.cm ** 2), E=-60 * u.mV)
        self.Ih = braincell.channels.Ih_HM1992(size, g_max=0.05 * (u.mS / u.cm ** 2), E=-43 * u.mV)


class TRN(ThalamusNeuron):
    def __init__(self, size, V_initializer=brainstate.init.Constant(-70. * u.mV), gl=0.0075):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV)

        self.area = 1e-3 / (1.43e-4 * u.cm ** 2)

        self.na = braincell.ions.SodiumFixed(size)
        self.na.add_elem(INa=braincell.channels.INa_Ba2002(size, V_sh=-40 * u.mV))

        self.k = braincell.ions.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add_elem(IDR=braincell.channels.IKDR_Ba2002(size, V_sh=-40 * u.mV))
        self.k.add_elem(IKL=braincell.channels.IK_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))

        self.ca = braincell.ions.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=100. * u.ms, d=0.5 * u.um)
        self.ca.add_elem(ICaN=braincell.channels.ICaN_IS2008(size, g_max=0.2 * (u.mS / u.cm ** 2)))
        self.ca.add_elem(ICaT=braincell.channels.ICaT_HP1992(size, g_max=1.3 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add_elem(IAHP=braincell.channels.IAHP_De1994(size, g_max=0.2 * (u.mS / u.cm ** 2)))

        # self.IL = dx.channel.IL(size, g_max=0.01 * (u.mS / u.cm ** 2), E=-60 * u.mV)
        self.IL = braincell.channels.IL(size, g_max=gl * (u.mS / u.cm ** 2), E=-60 * u.mV)


def try_trn_neuron():
    brainstate.environ.set(dt=0.02 * u.ms)

    I = braintools.input.section_input(values=[0, -0.05, 0], durations=[500 * u.ms, 200 * u.ms, 1000 * u.ms]) * u.uA
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()

    # neu = HTC([1, 1])  # [n_neuron, n_compartment]
    # neu = IN([1, 1])  # [n_neuron, n_compartment]
    # neu = RTC(1)  # [n_neuron, n_compartment]
    neu = TRN([1, 1], gl=0.0075)  # [n_neuron, n_compartment]
    neu.init_state()

    t0 = time.time()
    vs = brainstate.compile.for_loop(neu.step_run, times, I)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.4f} s")

    plt.plot(times.to_decimal(u.ms), u.math.squeeze(vs.to_decimal(u.mV)))
    plt.show()


if __name__ == '__main__':
    try_trn_neuron()
