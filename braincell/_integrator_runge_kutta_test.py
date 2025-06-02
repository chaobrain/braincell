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


import brainstate
import brainunit as u
import matplotlib.pyplot as plt

import braincell


class HH(braincell.SingleCompartment):
    def __init__(self, size, solver='rk4'):
        super().__init__(size, solver=solver)

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.INa_HH1952(size))

        self.k = braincell.ion.PotassiumFixed(size, E=-77. * u.mV)
        self.k.add(IK=braincell.channel.IK_HH1952(size))

        self.IL = braincell.channel.IL(size, E=-54.387 * u.mV, g_max=0.03 * (u.mS / u.cm ** 2))


def integrate(method: str, dt=0.01 * u.ms):
    brainstate.random.seed(1)
    hh = HH(1, solver=method)
    hh.init_state()

    def step_fun(t):
        with brainstate.environ.context(t=t):
            spike = hh.update(10 * u.nA / u.cm ** 2)
        return hh.V.value

    with brainstate.environ.context(dt=dt):
        times = u.math.arange(0. * u.ms, 10 * u.ms, brainstate.environ.get_dt())
        vs = brainstate.compile.for_loop(step_fun, times)
    return vs


def compare(method):
    norm = []
    dts = [1e-3 * u.ms, 2e-3 * u.ms, 4e-3 * u.ms, 8e-3 * u.ms, 1e-2 * u.ms, 2e-2 * u.ms]
    for dt in dts:
        gold_vs = integrate('exp_euler', dt=dt)
        vs = integrate(method, dt=dt)
        norm.append(u.linalg.norm(gold_vs - vs))
    return u.math.asarray(dts), u.math.asarray(norm)


class TestRungeKutta:
    def test_euler_step(self):
        dts, norms = compare('euler')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_midpoint_step(self):
        dts, norms = compare('midpoint')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_rk2_step(self):
        dts, norms = compare('rk2')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_heun2_step(self):
        dts, norms = compare('heun2')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_ralston2_step(self):
        dts, norms = compare('ralston2')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_rk3_step(self):
        dts, norms = compare('rk3')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_heun3_step(self):
        dts, norms = compare('heun3')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_ssprk3_step(self):
        dts, norms = compare('ssprk3')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_ralston3_step(self):
        dts, norms = compare('ralston3')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_rk4_step(self):
        dts, norms = compare('rk4')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()

    def test_ralston4_step(self):
        dts, norms = compare('ralston4')
        plt.loglog(dts, norms)
        plt.show()
        plt.close()
