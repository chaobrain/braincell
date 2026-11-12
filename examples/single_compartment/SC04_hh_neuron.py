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

import brainstate
import brainunit as u
import matplotlib.pyplot as plt

import braincell
from braincell import mech


def HH(size, solver='rk4'):
    # V starts at rest (-65 mV) and every gate at its steady state there.
    cell = braincell.Cell.point(size, solver=solver)
    cell.paint(
        mech.Ion('SodiumFixed', E=50. * u.mV),
        mech.Channel('Na_HH1952'),

        mech.Ion('PotassiumFixed', E=-77. * u.mV),
        mech.Channel('K_HH1952'),

        mech.Channel('IL', E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2)),
    )
    return cell


hh = HH(1, solver='ind_exp_euler')
hh.init_state()


def step_fun(t):
    with brainstate.environ.context(t=t):
        spike = hh.update(10 * u.nA / u.cm ** 2)
    return hh.V.value


with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 100 * u.ms, brainstate.environ.get_dt())
    vs = brainstate.transform.for_loop(step_fun, times)

plt.plot(times, u.math.squeeze(vs))
plt.show()
