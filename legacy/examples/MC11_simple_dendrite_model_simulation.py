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


from typing import Callable

import brainunit as u
import diffrax as dfx
import matplotlib.pyplot as plt
import numpy as np

import braincell
from MC11_simple_dendrite_model import ThreeCompartmentHH


def solve_explicit_solver(
    params,
    f_current: Callable[[u.Quantity, ...], u.Quantity],
    saveat: u.Quantity,
    t1: u.Quantity = 100. * u.ms,
    args=(),
    dt: u.Quantity = 0.01 * u.ms,
    method: str = 'tsit5'
):
    hh = ThreeCompartmentHH(n_neuron=1, g_na=params[0], g_k=params[1])
    hh.init_state()

    def step(t, *args):
        currents = f_current(t, *args)
        hh.compute_derivative(currents)

    def save(t, *args):
        return hh.V.value

    ts, ys, steps = braincell.diffrax_solve(
        step, method, 0. * u.ms, t1, dt,
        savefn=save, saveat=saveat, args=args,
        adjoint=dfx.RecursiveCheckpointAdjoint(1000)
    )
    return ts, ys, steps


def adjoint_solve_explicit_solver(
    params,
    f_current: Callable[[u.Quantity, ...], u.Quantity],
    saveat: u.Quantity,
    t1: u.Quantity = 100. * u.ms,
    args=(),
    dt: u.Quantity = 0.01 * u.ms,
    method: str = 'tsit5',
    max_steps: int = 100000
):
    hh = ThreeCompartmentHH(n_neuron=1, g_na=params[0], g_k=params[1])
    hh.init_state()

    def step(t, *args):
        currents = f_current(t, *args)
        hh.compute_derivative(currents)

    ts, ys, steps = braincell.diffrax_solve_adjoint(
        step, method, 0. * u.ms, t1, dt, saveat=saveat, args=args, max_steps=max_steps
    )
    return ts, ys[0], steps


def simulate():
    g = np.asarray([0.12, 0.036, 0.0003, 0.001, 0.001])
    f_current = lambda t: np.asarray([0.2, 0., 0.]) * u.nA
    saveat = u.math.arange(0., 100., 0.1) * u.ms
    ts, ys, steps = solve_explicit_solver(g, f_current, saveat)
    print(steps)
    plt.plot(ts.to_decimal(u.ms), u.math.squeeze(ys).to_decimal(u.mV))
    plt.xlabel('Time [ms]')
    plt.ylabel('Potential [mV]')
    plt.show()


if __name__ == '__main__':
    simulate()
