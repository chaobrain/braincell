# -*- coding: utf-8 -*-
# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Calcium-dependent potassium channels built directly on templates."""

from typing import Callable, Optional, Union

import brainstate
import braintools
import brainunit as u
import jax

from braincell._base import IonInfo
from braincell.channel._base import Gate, HH, Markov, Transition
from braincell.ion import Calcium, Potassium
from braincell.mech import register_channel

__all__ = [
    "AHP_De1994",
    "Kca3p1_MA2020_GoC",
    "Kca2p2_MA2020_GoC",
    "Kca1p1_MA2020_GoC",
]


_KCA_ROOT_TYPE = brainstate.mixin.JointTypes[Potassium, Calcium]


def _q10_factor(temp, q10, *, ref_celsius: float):
    return q10 ** (((temp - u.celsius2kelvin(ref_celsius)) / u.kelvin) / 10.0)


@register_channel("AHP_De1994")
class AHP_De1994(HH):
    r"""Destexhe 1994 calcium-dependent after-hyperpolarization current."""

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    gates = (
        Gate("p", power=2, phi=lambda self: self.phi),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        n: Union[brainstate.typing.ArrayLike, Callable] = 2,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 48.0,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.09,
        phi: Union[brainstate.typing.ArrayLike, Callable] = 1.0,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.n = braintools.init.param(n, self.varshape, allow_none=False)
        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)
        self.phi = braintools.init.param(phi, self.varshape, allow_none=False)

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, K, Ca) * (K.E - V)

    def f_p_alpha(self, V, K: IonInfo, Ca: IonInfo):
        return self.alpha * u.math.power(Ca.Ci / u.mM, self.n)

    def f_p_beta(self, V, K: IonInfo, Ca: IonInfo):
        return self.beta


@register_channel("Kca3p1_MA2020_GoC")
class Kca3p1_MA2020_GoC(HH):
    r"""Template-based import of ``Kca3p1_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    gates = (
        Gate("p"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 120.0 * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.0,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name)
        self.temp = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.p_beta = 0.05

    def current(self, V, K: IonInfo, Ca: IonInfo):
        return self.g_max * self.conductance_factor(V, K, Ca) * (K.E - V)

    def p_tau(self, V, Ca):
        return 1 / (self.p_alpha(V, Ca) + self.p_beta)

    def p_inf(self, V, Ca):
        return self.p_alpha(V, Ca) / (self.p_alpha(V, Ca) + self.p_beta)

    def p_alpha(self, V, Ca):
        V = V / u.mV
        return self.p_vdep(V) * self.p_concdep(Ca)

    def p_vdep(self, V):
        return u.math.exp((V + 70.0) / 27.0)

    def p_concdep(self, Ca):
        concdep_1 = 500 * 0.0013 / u.math.exprel((0.015 - Ca.Ci / u.mM) / 0.0013)
        with jax.ensure_compile_time_eval():
            concdep_2 = 500 * 0.005 / (u.math.exp(0.005 / 0.0013) - 1)
        return u.math.where(Ca.Ci / u.mM < 0.01, concdep_1, concdep_2)

    def f_p_alpha(self, V, K: IonInfo, Ca: IonInfo):
        return self.p_alpha(V, Ca)

    def f_p_beta(self, V, K: IonInfo, Ca: IonInfo):
        return self.p_beta


@register_channel("Kca2p2_MA2020_GoC")
class Kca2p2_MA2020_GoC(Markov):
    r"""Template-based import of ``Kca2p2_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    pairs = (
        Transition("C1", "C2", "dirc2_t_ca", "invc1_t"),
        Transition("C2", "C3", "dirc3_t_ca", "invc2_t"),
        Transition("C3", "C4", "dirc4_t_ca", "invc3_t"),
        Transition("C3", "O1", "diro1_t", "invo1_t"),
        Transition("C4", "O2", "diro2_t", "invo2_t"),
    )
    dependent_state = "C1"

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 38.0 * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.0,
        diff: brainstate.typing.ArrayLike = 3.0,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)
        self.temp = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.diff = braintools.init.param(diff, self.varshape, allow_none=False)

        self.invc1 = 80e-3
        self.invc2 = 80e-3
        self.invc3 = 200e-3

        self.invo1 = 1.0
        self.invo2 = 100e-3
        self.diro1 = 160e-3
        self.diro2 = 1.2

        self.dirc2 = 200.0
        self.dirc3 = 160.0
        self.dirc4 = 80.0

    def _phi(self):
        return _q10_factor(self.temp, self.T_base, ref_celsius=23.0)

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, K, Ca, batch_size=batch_size)

    def current(self, V, K: IonInfo, Ca: IonInfo):
        states = self.state_values()
        return self.g_max * (states["O1"] + states["O2"]) * (K.E - V)

    def dirc2_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc2 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def dirc3_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc3 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def dirc4_t_ca(self, V, K: IonInfo, Ca: IonInfo):
        return self.dirc4 * self._phi() * (Ca.Ci / u.mM) / self.diff

    def invc1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc1 * self._phi()

    def invc2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc2 * self._phi()

    def invc3_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invc3 * self._phi()

    def invo1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invo1 * self._phi()

    def invo2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.invo2 * self._phi()

    def diro1_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.diro1 * self._phi()

    def diro2_t(self, V, K: IonInfo, Ca: IonInfo):
        return self.diro2 * self._phi()


@register_channel("Kca1p1_MA2020_GoC")
class Kca1p1_MA2020_GoC(Markov):
    r"""Template-based import of ``Kca1p1_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = _KCA_ROOT_TYPE
    current_owner_type = Potassium
    pairs = (
        Transition("C0", "C1", "c01", "c10"),
        Transition("C1", "C2", "c12", "c21"),
        Transition("C2", "C3", "c23", "c32"),
        Transition("C3", "C4", "c34", "c43"),
        Transition("O0", "O1", "o01", "o10"),
        Transition("O1", "O2", "o12", "o21"),
        Transition("O2", "O3", "o23", "o32"),
        Transition("O3", "O4", "o34", "o43"),
        Transition("C0", "O0", "f0", "b0"),
        Transition("C1", "O1", "f1", "b1"),
        Transition("C2", "O2", "f2", "b2"),
        Transition("C3", "O3", "f3", "b3"),
        Transition("C4", "O4", "f4", "b4"),
    )
    dependent_state = "C0"

    def __init__(
        self,
        size: brainstate.typing.Size,
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 10.0 * (u.mS / u.cm ** 2),
        T_base: brainstate.typing.ArrayLike = 3.0,
        T: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        name: Optional[str] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
    ):
        super().__init__(size=size, name=name, solver=solver, substeps=substeps)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.temp = braintools.init.param(T, self.varshape, allow_none=False)
        self.T_base = braintools.init.param(T_base, self.varshape, allow_none=False)

        self.Qo = 0.73
        self.Qc = -0.67
        self.k1 = 1.0e3
        self.onoffrate = 1.0
        self.L0 = 1806
        self.Kc = 11.0e-3
        self.Ko = 1.1e-3

        self.pf0 = 2.39e-3
        self.pf1 = 7.0e-3
        self.pf2 = 40e-3
        self.pf3 = 295e-3
        self.pf4 = 557e-3

        self.pb0 = 3936e-3
        self.pb1 = 1152e-3
        self.pb2 = 659e-3
        self.pb3 = 486e-3
        self.pb4 = 92e-3

    def _phi(self):
        return _q10_factor(self.temp, self.T_base, ref_celsius=23.0)

    def _alpha_factor(self, V):
        return u.math.exp((self.Qo * u.faraday_constant * V) / (u.gas_constant * self.temp))

    def _beta_factor(self, V):
        return u.math.exp((self.Qc * u.faraday_constant * V) / (u.gas_constant * self.temp))

    def reset_state(self, V, K: IonInfo, Ca: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, K, Ca, batch_size=batch_size)

    def current(self, V, K: IonInfo, Ca: IonInfo):
        states = self.state_values()
        return self.g_max * (
            states["O0"] + states["O1"] + states["O2"] + states["O3"] + states["O4"]
        ) * (K.E - V)

    def c01(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c12(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c23(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c34(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o01(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o12(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o23(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def o34(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * (Ca.Ci / u.mM) * self.k1 * self.onoffrate * self._phi()

    def c10(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c21(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c32(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def c43(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * self.Kc * self.k1 * self.onoffrate * self._phi()

    def o10(self, V, K: IonInfo, Ca: IonInfo):
        return 1 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o21(self, V, K: IonInfo, Ca: IonInfo):
        return 2 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o32(self, V, K: IonInfo, Ca: IonInfo):
        return 3 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def o43(self, V, K: IonInfo, Ca: IonInfo):
        return 4 * self.Ko * self.k1 * self.onoffrate * self._phi()

    def f0(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf0 * self._alpha_factor(V) * self._phi()

    def f1(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf1 * self._alpha_factor(V) * self._phi()

    def f2(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf2 * self._alpha_factor(V) * self._phi()

    def f3(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf3 * self._alpha_factor(V) * self._phi()

    def f4(self, V, K: IonInfo, Ca: IonInfo):
        return self.pf4 * self._alpha_factor(V) * self._phi()

    def b0(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb0 * self._beta_factor(V) * self._phi()

    def b1(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb1 * self._beta_factor(V) * self._phi()

    def b2(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb2 * self._beta_factor(V) * self._phi()

    def b3(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb3 * self._beta_factor(V) * self._phi()

    def b4(self, V, K: IonInfo, Ca: IonInfo):
        return self.pb4 * self._beta_factor(V) * self._phi()
