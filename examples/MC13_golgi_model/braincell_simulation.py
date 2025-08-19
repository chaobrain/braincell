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


import brainstate
import brainunit as u
import numpy as np

import braincell
from utils import seg_ion_params, BraincellRun, step_input

brainstate.environ.set(precision=64)


class Golgi(braincell.MultiCompartment):
    def __init__(self, popsize, morphology, el, gl, gh1, gh2, ek, gkv11, gkv34, gkv43, ena, gnarsg, V_init=-65):
        super().__init__(
            popsize=popsize,
            morphology=morphology,
            V_th=20. * u.mV,
            V_initializer=brainstate.init.Constant(V_init * u.mV),
            spk_fun=brainstate.surrogate.ReluGrad(),
            solver='staggered'
        )

        self.IL = braincell.channel.IL(self.varshape, E=el * u.mV, g_max=gl * u.mS / (u.cm ** 2))
        self.Ih1 = braincell.channel.Ih1_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh1 * u.mS / (u.cm ** 2))
        self.Ih2 = braincell.channel.Ih2_Ma2020(self.varshape, E=-20. * u.mV, g_max=gh2 * u.mS / (u.cm ** 2))

        self.k = braincell.ion.PotassiumFixed(self.varshape, E=ek * u.mV)
        self.k.add(IKv11=braincell.channel.IKv11_Ak2007(self.varshape, g_max=gkv11 * u.mS / (u.cm ** 2)))
        self.k.add(IKv34=braincell.channel.IKv34_Ma2020(self.varshape, g_max=gkv34 * u.mS / (u.cm ** 2)))
        self.k.add(IKv43=braincell.channel.IKv43_Ma2020(self.varshape, g_max=gkv43 * u.mS / (u.cm ** 2)))

        self.na = braincell.ion.SodiumFixed(self.varshape, E=ena * u.mV)
        self.na.add(INa_Rsg=braincell.channel.INa_Rsg(self.varshape, g_max=gnarsg * u.mS / (u.cm ** 2)))

    def step_run(self, t, inp):
        with brainstate.environ.context(t=t):
            self.update(inp)
            return self.V.value


Golgi_mor = braincell.Morphology.from_asc('golgi.asc')
Golgi_mor.set_passive_params()

gl, gh1, gh2, gkv11, gkv34, gkv43, gnarsg, gcagrc, gcav23, gcav31, gkca31 = seg_ion_params(Golgi_mor)
nseg = len(Golgi_mor.segments)
El = -55
Ek = -80
Ena = 60
V_init = -65 * np.ones(nseg)
popsize = 10  # number of cells in the population
cell_braincell = Golgi(
    popsize=popsize, morphology=Golgi_mor, el=El, gl=gl,
    gh1=gh1, gh2=gh2, ek=Ek, gkv11=gkv11, gkv34=gkv34,
    gkv43=gkv43, ena=Ena, gnarsg=gnarsg, V_init=V_init
)

DT = 0.01
I = step_input(num=nseg, dur=[100, 0, 0], amp=[0, 0, 0], dt=DT)
t_braincell, v_braincell = BraincellRun(cell=cell_braincell, I=I, dt=DT)
