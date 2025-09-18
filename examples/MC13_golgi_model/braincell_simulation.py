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

import jax
import time
import brainstate
import brainunit as u
import matplotlib.pyplot as plt

# brainstate.environ.set(precision=64, platform='gpu')
brainstate.environ.set(precision=64, )

import braincell
import numpy as np
import braintools


def is_basal(idx):
    return (
        0 <= idx <= 3
        or 16 <= idx <= 17
        or 33 <= idx <= 41
        or idx == 84
        or 105 <= idx <= 150
    )


def is_apical(idx):
    return (
        4 <= idx <= 15
        or 18 <= idx <= 32
        or 42 <= idx <= 83
        or 85 <= idx <= 104
    )


def step_input(num, dur, amp):
    value = u.math.zeros((len(dur), num))
    for i in range(len(value)):
        value = value.at[i, 0].set(amp[i])
    return braintools.input.section_input(values=value, durations=dur * u.ms) * u.nA


def seg_ion_params(morphology):
    # segment index for each type
    index_soma = []
    index_axon = []
    index_dend_basal = []
    index_dend_apical = []

    for i, seg in enumerate(morphology.segments):
        name = str(seg.section_name)
        if name.startswith("soma"):
            index_soma.append(i)
        elif name.startswith("axon"):
            index_axon.append(i)
        elif name.startswith("dend_"):
            idx = int(name.split("_")[-1])
            if is_basal(idx):
                index_dend_basal.append(i)
            if is_apical(idx):
                index_dend_apical.append(i)

    n_compartments = len(morphology.segments)

    # conductance values
    conduct_values = 1e3 * np.array(
        [
            0.00499506303209, 0.01016375552607, 0.00247172479141, 0.00128859564935,
            3.690771983E-05, 0.0080938853146, 0.01226052748146, 0.01650689958385,
            0.00139885617712, 0.14927733727426, 0.00549507510519, 0.14910988921938,
            0.00406420380423, 0.01764345789036, 0.10177335775222, 0.0087689418803,
            3.407734319E-05, 0.0003371456442, 0.00030643090764, 0.17233663543619,
            0.00024381226198, 0.10008178886943, 0.00595046001148, 0.0115, 0.0091
        ]
    )

    # IL
    gl = np.ones(n_compartments)
    gl[index_soma] = 0.03
    gl[index_axon] = 0.001
    gl[index_axon[0:5]] = 0.03
    gl[index_dend_basal] = 0.03
    gl[index_dend_apical] = 0.03

    # IKv11_Ak2007
    gkv11 = np.zeros(n_compartments)
    gkv11[index_soma] = conduct_values[10]

    # IKv34_Ma2020
    gkv34 = np.zeros(n_compartments)
    gkv34[index_soma] = conduct_values[11]
    gkv34[index_axon[5:]] = 9.1

    # IKv43_Ma2020
    gkv43 = np.zeros(n_compartments)
    gkv43[index_soma] = conduct_values[12]

    # ICaGrc_Ma2020
    gcagrc = np.zeros(n_compartments)
    gcagrc[index_soma] = conduct_values[15]
    gcagrc[index_dend_basal] = conduct_values[8]
    gcagrc[index_axon[0:5]] = conduct_values[22]

    # ICav23_Ma2020
    gcav23 = np.zeros(n_compartments)
    gcav23[index_dend_apical] = conduct_values[3]

    # ICav31_Ma2020
    gcav31 = np.zeros(n_compartments)
    gcav31[index_soma] = conduct_values[16]
    gcav31[index_dend_apical] = conduct_values[4]

    # INa_Rsg
    gnarsg = np.zeros(n_compartments)
    gnarsg[index_soma] = conduct_values[9]
    gnarsg[index_dend_apical] = conduct_values[0]
    gnarsg[index_dend_basal] = conduct_values[5]
    gnarsg[index_axon[0:5]] = conduct_values[19]
    gnarsg[index_axon[5:]] = 11.5

    # Ih1_Ma2020
    gh1 = np.zeros(n_compartments)
    gh1[index_axon[0:5]] = conduct_values[17]

    # Ih2_Ma2020
    gh2 = np.zeros(n_compartments)
    gh2[index_axon[0:5]] = conduct_values[18]

    # IKca3_1_Ma2020
    gkca31 = np.zeros(n_compartments)
    gkca31[index_soma] = conduct_values[14]

    return gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31


class Golgi(braincell.MultiCompartment):
    def __init__(
        self,
        popsize,
        morphology,
        E_L,
        gl,
        gh1,
        gh2,
        E_K,
        gkv11,
        gkv34,
        gkv43,
        E_Na,
        gnarsg,
        V_init=-65 * u.mV,
    ):
        super().__init__(
            popsize=popsize,
            morphology=morphology,
            V_th=20. * u.mV,
            V_initializer=brainstate.init.Constant(V_init),
            spk_fun=brainstate.surrogate.ReluGrad(),
            solver='staggered'
        )

        self.IL = braincell.channel.IL(self.varshape, E=E_L, g_max=gl * u.mS / (u.cm ** 2))
        self.Ih1 = braincell.channel.Ih1_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh1 * u.mS / (u.cm ** 2))
        self.Ih2 = braincell.channel.Ih2_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh2 * u.mS / (u.cm ** 2))

        self.k = braincell.ion.PotassiumFixed(self.varshape, E=E_K)
        self.k.add(IKv11=braincell.channel.IKv11_Ak2007(self.varshape, g_max=gkv11 * u.mS / (u.cm ** 2)))
        self.k.add(IKv34=braincell.channel.IKv34_Ma2020(self.varshape, g_max=gkv34 * u.mS / (u.cm ** 2)))
        self.k.add(IKv43=braincell.channel.IKv43_Ma2020(self.varshape, g_max=gkv43 * u.mS / (u.cm ** 2)))

        self.na = braincell.ion.SodiumFixed(self.varshape, E=E_Na)
        self.na.add(INa_Rsg=braincell.channel.INa_Rsg(self.varshape, g_max=gnarsg * u.mS / (u.cm ** 2), solver='rk4'))
        # self.na.add(INa_Rsg=braincell.channel.INa_Rsg(self.varshape, g_max=gnarsg * u.mS / (u.cm ** 2),
        #                                               solver='backward_euler'))

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            self.update(inp)
            return self.V.value


morphology = braincell.Morphology.from_asc('golgi.asc')
morphology.set_passive_params()

gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31 = seg_ion_params(morphology)
cell_braincell = Golgi(
    popsize=1,  # number of cells in the population
    morphology=morphology,
    E_L=-55. * u.mV,
    gl=gl,
    gh1=gh1,
    gh2=gh2,
    E_K=-80. * u.mV,
    gkv11=gkv11,
    gkv34=gkv34,
    gkv43=gkv43,
    E_Na=60. * u.mV,
    gnarsg=gnarsg,
    V_init=-65 * u.mV,
)


@brainstate.transform.jit
def simulate(I):
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()
    cell_braincell.init_state()
    vs = brainstate.compile.for_loop(cell_braincell.step_run, times, I)  # vs =
    return times.to_decimal(u.ms), vs.to_decimal(u.mV)


brainstate.environ.set(dt=0.01 * u.ms)
I = step_input(num=len(morphology.segments), dur=[100, 0, 0], amp=[0, 0, 0])

t0 = time.time()
t_braincell, v_braincell = jax.block_until_ready(simulate(I))
t1 = time.time()

t2 = time.time()
t_braincell, v_braincell = jax.block_until_ready(simulate(I))
t3 = time.time()
print(f'First run time = {t1 - t0} s, second run time = {t3 - t2} s')

plt.plot(t_braincell, v_braincell[:, 0, 0], label='soma')
plt.plot(t_braincell, v_braincell[:, 0, -1], label='distal dendrite')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.legend()
plt.show()
