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


from typing import Union, Callable

import brainstate
import braintools
import brainunit as u
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import braincell

brainstate.environ.set(dt=0.01 * u.ms)

# Load Input and Output Data
df_inp_traces = pd.read_csv('neuron_data/input_traces_hh.csv')
df_out_traces = pd.read_csv('neuron_data/output_traces_hh.csv')

area = 20000 * u.um ** 2  # neuronal area
inp_traces = df_inp_traces.to_numpy()[:, 1:] * 1e9 * u.nA  # input currents
mem_traces = df_out_traces.to_numpy()[:, 1:] * u.mV  # membrane potentials to record


class INa(braincell.Channel):
    root_type = braincell.HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        ENa: Union[brainstate.typing.ArrayLike, Callable] = 50. * u.mV,
        gNa: Union[brainstate.typing.ArrayLike, Callable] = 120. * u.mS,
        vth: Union[brainstate.typing.ArrayLike, Callable] = -63 * u.mV,
    ):
        super().__init__(size)
        self.ENa = braintools.init.param(ENa, self.varshape)
        self.gNa = braintools.init.param(gNa, self.varshape)
        self.V_th = braintools.init.param(vth, self.varshape)

    def init_state(self, V, batch_size=None):
        self.m = braincell.DiffEqState(braintools.init.param(u.math.zeros, self.varshape))
        self.h = braincell.DiffEqState(braintools.init.param(u.math.zeros, self.varshape))

    #  m channel
    m_alpha = lambda self, V: 0.32 * 4 / u.math.exprel((13. * u.mV - V + self.V_th).to_decimal(u.mV) / 4.)
    m_beta = lambda self, V: 0.28 * 5 / u.math.exprel((V - self.V_th - 40. * u.mV).to_decimal(u.mV) / 5.)
    m_inf = lambda self, V: self.m_alpha(V) / (self.m_alpha(V) + self.m_beta(V))

    # h channel
    h_alpha = lambda self, V: 0.128 * u.math.exprel((17. * u.mV - V + self.V_th).to_decimal(u.mV) / 18.)
    h_beta = lambda self, V: 4. / (1 + u.math.exp((40. * u.mV - V + self.V_th).to_decimal(u.mV) / 5.))
    h_inf = lambda self, V: self.h_alpha(V) / (self.h_alpha(V) + self.h_beta(V))

    def compute_derivative(self, V, *args, **kwargs):
        m = self.m.value
        h = self.h.value
        self.m.derivative = (self.m_alpha(V) * (1 - m) - self.m_beta(V) * m) / u.ms
        self.h.derivative = (self.h_alpha(V) * (1 - h) - self.h_beta(V) * h) / u.ms

    def current(self, V, *args, **kwargs):
        m = self.m.value
        h = self.h.value
        return (self.gNa * m * m * m * h) * (self.ENa - V)


class IK(braincell.Channel):
    root_type = braincell.HHTypedNeuron

    def __init__(
        self,
        size: brainstate.typing.Size,
        EK: Union[brainstate.typing.ArrayLike, Callable] = -90. * u.mV,
        gK: Union[brainstate.typing.ArrayLike, Callable] = 36. * u.mS,
        vth: Union[brainstate.typing.ArrayLike, Callable] = -63 * u.mV,
    ):
        super().__init__(size)
        self.EK = braintools.init.param(EK, self.varshape)
        self.gK = braintools.init.param(gK, self.varshape)
        self.V_th = braintools.init.param(vth, self.varshape)

    def init_state(self, V, batch_size=None):
        self.n = braincell.DiffEqState(braintools.init.param(u.math.zeros, self.varshape))

    # n channel
    n_alpha = lambda self, V: 0.032 * 5 / u.math.exprel((15. * u.mV - V + self.V_th).to_decimal(u.mV) / 5.)
    n_beta = lambda self, V: .5 * u.math.exp((10. * u.mV - V + self.V_th).to_decimal(u.mV) / 40.)
    n_inf = lambda self, V: self.n_alpha(V) / (self.n_alpha(V) + self.n_beta(V))

    def compute_derivative(self, V, *args, **kwargs):
        n = self.n.value
        self.n.derivative = (self.n_alpha(V) * (1 - n) - self.n_beta(V) * n) / u.ms

    def current(self, V, *args, **kwargs):
        n2 = self.n.value ** 2
        return (self.gK * n2 * n2) * (self.EK - V)


class HH(braincell.SingleCompartment):
    def __init__(
        self,
        size,
        v_initializer: Callable = braintools.init.Uniform(-70 * u.mV, -60. * u.mV),
        gL: Union[brainstate.typing.ArrayLike, Callable] = 0.003 * u.mS,
        gNa: Union[brainstate.typing.ArrayLike, Callable] = 120. * u.mS,
        gK: Union[brainstate.typing.ArrayLike, Callable] = 36. * u.mS,
        C: Union[brainstate.typing.ArrayLike, Callable] = 1. * (u.uF / u.cm ** 2),
        solver: str = 'rk4'
    ):
        super().__init__(size, V_initializer=v_initializer, C=C, solver=solver)
        self.ina = INa(size, gNa=gNa)
        self.ik = IK(size, gK=gK)
        self.il = braincell.channel.IL(size, g_max=gL, E=-65. * u.mV)


def visualize_target(voltages):
    fig, gs = braintools.visualize.get_figure(2, voltages.shape[1], 3, 4.5)
    times = np.arange(voltages.shape[0]) * 0.01
    for i in range(voltages.shape[1]):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(times, voltages[:, i], label='target')
        plt.xlabel('Time [ms]')
        plt.legend()
        ax = plt.subplot(gs[1, i])
        ax.plot(times, inp_traces[i])
        plt.xlabel('Time [ms]')
    plt.show()


def visualize(voltages, gl, g_na, g_kd, C):
    # currents: [T, B]
    # voltages: [T, B]
    simulated_vs = simulate_model(gl, g_na, g_kd, C)
    voltages = voltages.mantissa

    fig, gs = braintools.visualize.get_figure(2, simulated_vs.shape[1], 3, 4.5)
    for i in range(simulated_vs.shape[1]):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(voltages[:, i], label='target')
        ax.plot(simulated_vs[:, i], label='simulated')
        plt.legend()
        ax = plt.subplot(gs[1, i])
        ax.plot(inp_traces[i])
    plt.show()


def simulate_model(gl, g_na, g_kd, C):
    current = inp_traces.T
    assert current.ndim == 2  # [T, B]
    n_input = current.shape[1]
    hh = HH((n_input, 1), gL=gl, gNa=g_na, gK=g_kd, C=C, v_initializer=braintools.init.Constant(-65. * u.mV), )
    hh.init_state()

    def step_fun(i, inp):
        with brainstate.environ.context(i=i, t=brainstate.environ.get_dt() * i):
            spike = hh(inp)
        return hh.V.value

    indices = np.arange(current.shape[0])
    current = u.math.expand_dims(current, axis=-1)  # [T, B, 1]
    return brainstate.transform.for_loop(step_fun, indices, current)  # (T, B)


def compare_potentials(param):
    vs = simulate_model(param['gl'], param['g_na'], param['g_kd'], param['C'])  # (T, B)
    losses = braintools.metric.squared_error(vs.mantissa, target_vs.mantissa)
    return losses.mean()


# inp_traces: [B, T]
# target_vs: [B, T, 1]
# target_vs = simulate_model(50.37195496 * u.nsiemens,
#                            28.01317016 * u.usiemens,
#                            8.15968937 * u.usiemens,
#                            195.2173985 * u.pfarad)
target_vs = u.math.expand_dims(mem_traces.T, axis=-1)  # [T, B, 1]


# visualize_target(target_vs)


def visualize_hh_input_and_output():
    # Load Input and Output Data
    inp_traces = df_inp_traces.to_numpy()
    inp_traces = inp_traces[:, 1:] * 1e9

    out_traces = df_out_traces.to_numpy()
    out_traces = out_traces[:, 1:]

    indices = np.arange(inp_traces.shape[1]) * 0.01

    fig, gs = braintools.visualize.get_figure(3, 1, 1.2, 6.0)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(indices, inp_traces.T)
    plt.xticks([])
    plt.ylabel('Current [nA]', fontsize=13)

    ax2 = fig.add_subplot(gs[1:, 0])
    ax2.plot(indices, out_traces.T)
    plt.ylabel('Potential [mV]', fontsize=13)
    plt.xlabel('Time [ms]')

    fig.align_ylabels([ax, ax2])
    plt.show()


bounds = {
    'gl': [1e0, 1e2] * u.nS,
    'g_na': [1e0, 2e2] * u.uS,
    'g_kd': [1e0, 1e2] * u.uS,
    'C': [0.1, 2] * u.uF * u.cm ** -2 * area,
}


def fitting_by_nevergrad(method='DE', n_sample=200, n_iter=20):
    print(f"Method: {method}, n_sample: {n_sample}")

    @brainstate.transform.jit
    def loss_with_multiple_run(**params):
        return brainstate.transform.vmap(compare_potentials)(params)

    opt = braintools.optim.NevergradOptimizer(
        loss_with_multiple_run,
        n_sample=n_sample,
        bounds=bounds,
        method=method,
    )
    opt.initialize()
    param = opt.minimize(n_iter)
    loss = compare_potentials(param)
    print(param)
    print(loss)
    visualize(target_vs, **param)
    return param, loss


if __name__ == '__main__':
    fitting_by_nevergrad(n_sample=100)
