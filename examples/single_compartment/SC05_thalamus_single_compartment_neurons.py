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

"""
Implementation of the following models in the paper:

- Li, Guoshi, Craig S. Henriquez, and Flavio Fröhlich. “Unified thalamic model generates
  multiple distinct oscillations with state-dependent entrainment by stimulation.”
  PLoS computational biology 13.10 (2017): e1005797.
"""

import time

import braincell
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
from braincell import mech


def thalamic_cell(size, area, V_init, solver):
    """A point neuron with membrane ``area``: a cylinder as long as it is wide.

    Currents given to ``update`` or by projections are totals; the cell
    divides them by ``area``.
    """
    diam = u.math.sqrt(area / u.math.pi)
    return braincell.Cell.point(
        size, length=diam, radius=diam / 2, V_init=V_init, V_th=20. * u.mV, solver=solver,
    )


def step_run(cell, t, inp):
    with brainstate.environ.context(t=t):
        cell.update(inp)
        return cell.V.value


def HTC(
    size,
    gKL=0.01 * (u.mS / u.cm ** 2),
    V_init=-65. * u.mV,
    solver: str = 'ind_exp_euler'
):
    cell = thalamic_cell(size, 2.9e-4 * u.cm ** 2, V_init, solver)
    cell.paint(
        mech.Ion('SodiumFixed', E=50. * u.mV),
        mech.Channel('Na_Ba2002', V_sh=-30 * u.mV),

        mech.Ion('PotassiumFixed', E=-90. * u.mV),
        mech.Channel('K_Leak', g_max=gKL),
        mech.Channel(
            'KDR_Ba2002',
            V_sh=-30. * u.mV,
            temp=u.celsius2kelvin(36.0),
            q10=0.25,
            temp_ref=u.celsius2kelvin(26.0),
        ),

        mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um),
        mech.Channel('CaL_IS2008', g_max=0.5 * (u.mS / u.cm ** 2)),
        mech.Channel('CaN_IS2008', g_max=0.5 * (u.mS / u.cm ** 2)),
        mech.Channel('CaT_HM1992', g_max=2.1 * (u.mS / u.cm ** 2)),
        mech.Channel('CaHT_HM1992', g_max=3.0 * (u.mS / u.cm ** 2)),

        # Reads the potassium and the calcium ion.
        mech.Channel('AHP_De1994', g_max=0.3 * (u.mS / u.cm ** 2)),

        mech.Channel('HCN_HM1992', g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV),
        mech.Channel('IL', g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV),
    )
    return cell


def RTC(
    size,
    gKL=0.01 * (u.mS / u.cm ** 2),
    V_init=-65. * u.mV,
    solver: str = 'ind_exp_euler'
):
    cell = thalamic_cell(size, 2.9e-4 * u.cm ** 2, V_init, solver)
    cell.paint(
        mech.Ion('SodiumFixed'),
        mech.Channel('Na_Ba2002', V_sh=-40 * u.mV),

        mech.Ion('PotassiumFixed', E=-90. * u.mV),
        mech.Channel(
            'KDR_Ba2002',
            V_sh=-40 * u.mV,
            temp=u.celsius2kelvin(36.0),
            q10=0.25,
            temp_ref=u.celsius2kelvin(26.0),
        ),
        mech.Channel('K_Leak', g_max=gKL),

        mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um),
        mech.Channel('CaL_IS2008', g_max=0.3 * (u.mS / u.cm ** 2)),
        mech.Channel('CaN_IS2008', g_max=0.6 * (u.mS / u.cm ** 2)),
        mech.Channel('CaT_HM1992', g_max=2.1 * (u.mS / u.cm ** 2)),
        mech.Channel('CaHT_HM1992', g_max=0.6 * (u.mS / u.cm ** 2)),

        mech.Channel('AHP_De1994', g_max=0.1 * (u.mS / u.cm ** 2)),

        mech.Channel('HCN_HM1992', g_max=0.01 * (u.mS / u.cm ** 2), E=-43 * u.mV),
        mech.Channel('IL', g_max=0.0075 * (u.mS / u.cm ** 2), E=-70 * u.mV),
    )
    return cell


def IN(
    size,
    gKL=0.01 * (u.mS / u.cm ** 2),
    V_init=-70. * u.mV,
    solver: str = 'ind_exp_euler'
):
    cell = thalamic_cell(size, 1.7e-4 * u.cm ** 2, V_init, solver)
    cell.paint(
        mech.Ion('SodiumFixed'),
        mech.Channel('Na_Ba2002', V_sh=-30 * u.mV),

        mech.Ion('PotassiumFixed', E=-90. * u.mV),
        mech.Channel(
            'KDR_Ba2002',
            V_sh=-30 * u.mV,
            temp=u.celsius2kelvin(36.0),
            q10=0.25,
            temp_ref=u.celsius2kelvin(26.0),
        ),
        mech.Channel('K_Leak', g_max=gKL),

        mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um),
        mech.Channel('CaN_IS2008', g_max=0.1 * (u.mS / u.cm ** 2)),
        mech.Channel('CaHT_HM1992', g_max=2.5 * (u.mS / u.cm ** 2)),

        mech.Channel('AHP_De1994', g_max=0.2 * (u.mS / u.cm ** 2)),

        mech.Channel('IL', g_max=0.0075 * (u.mS / u.cm ** 2), E=-60 * u.mV),
        mech.Channel('HCN_HM1992', g_max=0.05 * (u.mS / u.cm ** 2), E=-43 * u.mV),
    )
    return cell


def TRN(
    size,
    gKL=0.01 * (u.mS / u.cm ** 2),
    V_init=-70. * u.mV,
    gl=0.0075,
    solver: str = 'ind_exp_euler'
):
    cell = thalamic_cell(size, 1.43e-4 * u.cm ** 2, V_init, solver)
    cell.paint(
        mech.Ion('SodiumFixed'),
        mech.Channel('Na_Ba2002', V_sh=-40 * u.mV),

        mech.Ion('PotassiumFixed', E=-90. * u.mV),
        mech.Channel('KDR_Ba2002', V_sh=-40 * u.mV),
        mech.Channel('K_Leak', g_max=gKL),

        mech.Ion('CalciumDetailed', C_rest=5e-5 * u.mM, tau=100. * u.ms, d=0.5 * u.um),
        mech.Channel('CaN_IS2008', g_max=0.2 * (u.mS / u.cm ** 2)),
        mech.Channel('CaT_HP1992', g_max=1.3 * (u.mS / u.cm ** 2)),

        mech.Channel('AHP_De1994', g_max=0.2 * (u.mS / u.cm ** 2)),

        # mech.Channel('IL', g_max=0.01 * (u.mS / u.cm ** 2), E=-60 * u.mV),
        mech.Channel('IL', g_max=gl * (u.mS / u.cm ** 2), E=-60 * u.mV),
    )
    return cell


def try_neuron_simulation():
    brainstate.environ.set(dt=0.1 * u.ms)

    # Total current, divided by the membrane area.
    I = braintools.input.section(values=[0, 0.05, 0], durations=[50 * u.ms, 200 * u.ms, 100 * u.ms]) * u.nA
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()

    neu = TRN(1)  # [n_neuron, ]
    # neu = IN(1)  # [n_neuron, ]
    # neu = RTC(1)  # [n_neuron, ]
    # neu = HTC(1, solver='ind_exp_euler')  # [n_neuron,]
    neu.init_state()

    t0 = time.time()
    vs = brainstate.transform.for_loop(lambda t, inp: step_run(neu, t, inp), times, I)
    t1 = time.time()
    print(f"Elapsed time: {t1 - t0:.4f} s")

    plt.plot(times.to_decimal(u.ms), u.math.squeeze(vs.to_decimal(u.mV)))
    plt.show()


if __name__ == '__main__':
    try_neuron_simulation()
