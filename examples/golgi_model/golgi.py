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

import time

import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np

import braincell

brainstate.environ.set(precision=64)

## morphology params
loaded_params = np.load('golgi_morphology.npz')

connection = loaded_params['connection']
L = loaded_params['L']  # um
diam = loaded_params['diam']  # um
Ra = loaded_params['Ra']  # ohm * cm
cm = loaded_params['cm']  # uF / cm ** 2

n_neuron = 1
n_compartments = len(L)
size = (n_neuron, n_compartments)

index_soma = loaded_params['index_soma']
index_axon = loaded_params['index_axon']
index_dend_basal = loaded_params['index_dend_basal']
index_dend_apical = loaded_params['index_dend_apical']

## conductvalues 
conductvalues = 1e3 * np.array([

    0.00499506303209, 0.01016375552607, 0.00247172479141, 0.00128859564935,
    3.690771983E-05, 0.0080938853146, 0.01226052748146, 0.01650689958385,
    0.00139885617712, 0.14927733727426, 0.00549507510519, 0.14910988921938,
    0.00406420380423, 0.01764345789036, 0.10177335775222, 0.0087689418803,
    3.407734319E-05, 0.0003371456442, 0.00030643090764, 0.17233663543619,
    0.00024381226198, 0.10008178886943, 0.00595046001148, 0.0115, 0.0091
])

## IL 
gl = np.ones(n_compartments)
gl[index_soma] = 0.03
gl[index_axon] = 0.001
gl[index_axon[0]] = 0.03
gl[index_dend_basal] = 0.03
gl[index_dend_apical] = 0.03

## IKv11_Ak2007
gkv11 = np.zeros(n_compartments)
gkv11[index_soma] = conductvalues[10]

## IKv34_Ma2020  
gkv34 = np.zeros(n_compartments)
gkv34[index_soma] = conductvalues[11]
gkv34[index_axon[1:]] = 9.1

## IKv43_Ma2020
gkv43 = np.zeros(n_compartments)
gkv43[index_soma] = conductvalues[12]

## ICaGrc_Ma2020
gcagrc = np.zeros(n_compartments)
gcagrc[index_soma] = conductvalues[15]
gcagrc[index_dend_basal] = conductvalues[8]
gcagrc[index_axon[0]] = conductvalues[22]

## ICav23_Ma2020
gcav23 = np.zeros(n_compartments)
gcav23[index_dend_apical] = conductvalues[3]

## ICav31_Ma2020 
gcav31 = np.zeros(n_compartments)
gcav31[index_soma] = conductvalues[16]
gcav31[index_dend_apical] = conductvalues[4]

## INa_Rsg
gnarsg = np.zeros(n_compartments)
gnarsg[index_soma] = conductvalues[9]
gnarsg[index_dend_apical] = conductvalues[0]
gnarsg[index_dend_basal] = conductvalues[5]
gnarsg[index_axon[0]] = conductvalues[19]
gnarsg[index_axon[1:]] = 11.5

## Ih1_Ma2020 
gh1 = np.zeros(n_compartments)
gh1[index_axon[0]] = conductvalues[17]

## Ih1_Ma2020 
gh2 = np.zeros(n_compartments)
gh2[index_axon[0]] = conductvalues[18]

## IKca3_1_Ma2020 
gkca31 = np.zeros(n_compartments)
gkca31[index_soma] = conductvalues[14]

## IKca2_2_Ma2020
gkca22 = np.zeros(n_compartments)
gkca31[index_dend_apical] = conductvalues[2]
gkca31[index_dend_basal] = conductvalues[7]

## IKca1_1_Ma2020
gkca11 = np.zeros(n_compartments)
gkca11[index_soma] = conductvalues[13]
gkca11[index_dend_apical] = conductvalues[1]
gkca11[index_dend_basal] = conductvalues[6]
gkca11[index_axon[0]] = conductvalues[21]
gkca11[index_axon[1:]] = conductvalues[13]

## IKM_Grc_Ma2020
gkmgrc = np.zeros(n_compartments)
gkmgrc[index_axon[0]] = conductvalues[20]


class Golgi(braincell.neuron.MultiCompartment):
    def __init__(self, size, connection, Ra, cm, diam, L, gl, gkv11):
        super().__init__(
            size=size,
            connection=connection,
            Ra=Ra * u.ohm * u.cm,
            cm=cm * u.uF / u.cm ** 2,
            diam=diam * u.um,
            L=L * u.um,
            V_th=20. * u.mV,
            V_initializer=brainstate.init.Constant(-55 * u.mV),
            spk_fun=brainstate.surrogate.ReluGrad(),
        )

        self.IL = braincell.channel.IL(self.size, E=-55. * u.mV, g_max=gl * u.mS / (u.cm ** 2))
        self.k = braincell.ion.PotassiumFixed(self.size, E=-80. * u.mV)
        self.k.add(k=braincell.channel.IKv11_Ak2007(self.size, g_max=gkv11 * u.mS / (u.cm ** 2)))


@brainstate.compile.jit(static_argnums=6)
def simulate(Ra, cm, diam, L, gl, gkv11, method='ieuler'):
    cell = Golgi(size, connection, Ra, cm, diam, L, gl, gkv11)
    cell.init_state()
    cell.reset_state()

    def step(t, *args):
        inp_a = np.full((n_neuron, n_compartments), 0.) * u.nA
        inp_a[..., 30] = 0.02 * u.nA
        inp_b = np.full((n_neuron, n_compartments), 0.) * u.nA
        cell.compute_derivative(u.math.where(t < 100 * u.ms, inp_a, inp_b))

    def save(t, *args):
        return cell.V.value

    with jax.ensure_compile_time_eval():
        dt = 0.01 * u.ms
        ts = u.math.arange(0. * u.ms, 200. * u.ms, dt)
    ts, ys, steps = braincell.diffrax_solve(
        step, method, 0. * u.ms, 200. * u.ms, dt, ts,
        savefn=save, atol=1e-5,  # max_steps=200000,
    )

    return ts, ys, steps


@brainstate.compile.jit
def simulate2(Ra, cm, diam, L, gl, gkv11):
    cell = Golgi(size, connection, Ra, cm, diam, L, gl, gkv11)
    cell.init_state()
    cell.reset_state()

    def step_run(t):
        inp_a = np.full((n_neuron, n_compartments), 0.) * u.nA
        inp_a[..., 30] = 0.02 * u.nA
        inp_b = np.full((n_neuron, n_compartments), 0.) * u.nA
        inp = u.math.where(t < 100 * u.ms, inp_a, inp_b)
        braincell.rk4_step(cell, t, inp)
        return cell.V.value

    with jax.ensure_compile_time_eval():
        dt = 0.001 * u.ms
        ts = u.math.arange(0. * u.ms, 200. * u.ms, dt)
    with brainstate.environ.context(dt=dt):
        ys = brainstate.compile.for_loop(step_run, ts)
    return ts, ys[::10], ts.size


def visualize_a_simulate(Ra, cm, diam, L, gl, gkv11):
    t0 = time.time()
    ts, ys_kvaerno5, steps = simulate(Ra, cm, diam, L, gl, gkv11, 'kvaerno5')
    print('kvaerno5', steps)
    print('time', time.time() - t0)

    t0 = time.time()
    ts, ys_ieuler, steps = simulate(Ra, cm, diam, L, gl, gkv11, 'ieuler')
    print('ieuler', steps)
    print('time', time.time() - t0)

    t0 = time.time()
    ts, ys_tsit5, steps = simulate(Ra, cm, diam, L, gl, gkv11, 'tsit5')
    print('tsit5', steps)
    print('time', time.time() - t0)

    t0 = time.time()
    ts, ys_dopri5, steps = simulate(Ra, cm, diam, L, gl, gkv11, 'dopri5')
    print('dopri5', steps)
    print('time', time.time() - t0)

    t0 = time.time()
    ts2, ys2_rk4, steps = simulate2(Ra, cm, diam, L, gl, gkv11)
    print('rk4', steps)
    print('time', time.time() - t0)

    def plot(data, ax, title):
        ax.plot(ts.to_decimal(u.ms), u.math.squeeze(data.to_decimal(u.mV)))
        plt.xlabel('Time [ms]')
        plt.ylabel('Potential [mV]')
        plt.title(title)

    fig, gs = braintools.visualize.get_figure(1, 5, 3.0, 4.0)
    plot(ys_kvaerno5 - ys_ieuler, fig.add_subplot(gs[0, 0]), 'kvaerno5 - ieuler')
    plot(ys_ieuler - ys_tsit5, fig.add_subplot(gs[0, 1]), 'ieuler - tsit5')
    plot(ys_tsit5 - ys_dopri5, fig.add_subplot(gs[0, 2]), 'tsit5 - dopri5')
    plot(ys_dopri5 - ys2_rk4, fig.add_subplot(gs[0, 3]), 'dopri5 - rk4')
    plot(ys_ieuler, fig.add_subplot(gs[0, 4]), 'ieuler')
    plt.show()


visualize_a_simulate(Ra, cm, diam, L, gl, gkv11)
