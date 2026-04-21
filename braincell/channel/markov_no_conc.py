# -*- coding: utf-8 -*-


from typing import Callable
from typing import Optional
from typing import Union

import brainstate
import braintools
import brainunit as u

from braincell._base import IonInfo
from braincell.channel._template import Markov
from braincell.ion import Sodium
from braincell.mech import register_channel
from braincell.quad._protocol import IndependentIntegration

__all__ = [
    "Nav1p6_MA20_GoC",
    "Nav1p6_MA24_PC",
    "Nav1p6_MA25_BC",
    "Nav1p6_RI21_SC",
    "Nav1p1_MA25_BC",
    "Nav1p1_RI21_SC",
    "Nav_MA20_GrC",
    "NaFHF_MA20_GrC",
]


@register_channel("Nav1p6_MA20_GoC")
class Nav1p6_MA20_GoC(Markov, IndependentIntegration):
    """Template-based import of ``Nav1p6_MA20_GoC.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C1", "I1", "fi1", "bi1"),
        ("I1", "I2", "f11", "b11"),
        ("C2", "I2", "fi2", "bi2"),
        ("I2", "I3", "f12", "b12"),
        ("C3", "I3", "fi3", "bi3"),
        ("I3", "I4", "f13", "b13"),
        ("C4", "I4", "fi4", "bi4"),
        ("I4", "I5", "f14", "b14"),
        ("C5", "I5", "fi5", "bi5"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "B", "fip", "bip"),
        ("I5", "I6", "f1n", "b1n"),
        ("O", "I6", "fin", "bin"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 16.0 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
        solver: str = "rk4",
        substeps: int = 5,
    ):
        super().__init__(size=size, name=name)
        IndependentIntegration.__init__(self, solver=solver)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1.")

        self.Con = 0.005
        self.Coff = 0.5
        self.Oon = 0.75
        self.Ooff = 0.005
        self.alpha = 150.0
        self.beta = 3.0
        self.gamma = 150.0
        self.delta = 40.0
        self.epsilon = 1.75
        self.zeta = 0.03

        self.x1 = 20.0
        self.x2 = -20.0
        self.x3 = 1e12
        self.x4 = -1e12
        self.x5 = 1e12
        self.x6 = -25.0
        self.vshifta = 0.0
        self.vshifti = 0.0
        self.vshiftk = 0.0

        self.alfac = (self.Oon / self.Con) ** (1 / 4)
        self.btfac = (self.Ooff / self.Coff) ** (1 / 4)

    def make_integration(self, *args, **kwargs):
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self.solver(self, *args, **kwargs),
                u.math.arange(self.substeps),
            )

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    f01 = lambda self, V, *unused: 4 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f02 = lambda self, V, *unused: 3 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f03 = lambda self, V, *unused: 2 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f04 = lambda self, V, *unused: 1 * self.alpha * u.math.exp((V / u.mV) / self.x1) * self.phi
    f0O = lambda self, V, *unused: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fip = lambda self, V, *unused: self.epsilon * u.math.exp((V / u.mV) / self.x5) * self.phi
    f11 = lambda self, V, *unused: 4 * self.alpha * self.alfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x1) * self.phi
    f12 = lambda self, V, *unused: 3 * self.alpha * self.alfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x1) * self.phi
    f13 = lambda self, V, *unused: 2 * self.alpha * self.alfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x1) * self.phi
    f14 = lambda self, V, *unused: 1 * self.alpha * self.alfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x1) * self.phi
    f1n = lambda self, V, *unused: self.gamma * u.math.exp((V / u.mV) / self.x3) * self.phi
    fi1 = lambda self, V, *unused: self.Con * self.phi
    fi2 = lambda self, V, *unused: self.Con * self.alfac * self.phi
    fi3 = lambda self, V, *unused: self.Con * self.alfac ** 2 * self.phi
    fi4 = lambda self, V, *unused: self.Con * self.alfac ** 3 * self.phi
    fi5 = lambda self, V, *unused: self.Con * self.alfac ** 4 * self.phi
    fin = lambda self, V, *unused: self.Oon * self.phi

    b01 = lambda self, V, *unused: 1 * self.beta * u.math.exp(
        (V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b02 = lambda self, V, *unused: 2 * self.beta * u.math.exp(
        (V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b03 = lambda self, V, *unused: 3 * self.beta * u.math.exp(
        (V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b04 = lambda self, V, *unused: 4 * self.beta * u.math.exp(
        (V / u.mV + self.vshifta) / (self.x2 + self.vshiftk)) * self.phi
    b0O = lambda self, V, *unused: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bip = lambda self, V, *unused: self.zeta * u.math.exp(V / u.mV / self.x6) * self.phi
    b11 = lambda self, V, *unused: 1 * self.beta * self.btfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x2) * self.phi
    b12 = lambda self, V, *unused: 2 * self.beta * self.btfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x2) * self.phi
    b13 = lambda self, V, *unused: 3 * self.beta * self.btfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x2) * self.phi
    b14 = lambda self, V, *unused: 4 * self.beta * self.btfac * u.math.exp(
        (V / u.mV + self.vshifti) / self.x2) * self.phi
    b1n = lambda self, V, *unused: self.delta * u.math.exp(V / u.mV / self.x4) * self.phi
    bi1 = lambda self, V, *unused: self.Coff * self.phi
    bi2 = lambda self, V, *unused: self.Coff * self.btfac * self.phi
    bi3 = lambda self, V, *unused: self.Coff * self.btfac ** 2 * self.phi
    bi4 = lambda self, V, *unused: self.Coff * self.btfac ** 3 * self.phi
    bi5 = lambda self, V, *unused: self.Coff * self.btfac ** 4 * self.phi
    bin = lambda self, V, *unused: self.Ooff * self.phi


@register_channel("Nav1p6_MA24_PC")
class Nav1p6_MA24_PC(Nav1p6_MA20_GoC):
    """Template-based import of ``Nav1p6_MA24_PC.mod``."""

    __module__ = "braincell.channel"


@register_channel("Nav1p6_MA25_BC")
class Nav1p6_MA25_BC(Nav1p6_MA20_GoC):
    """Template-based import of ``Nav1p6_MA25_BC.mod``."""

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p6_RI21_SC")
class Nav1p6_RI21_SC(Nav1p6_MA20_GoC):
    """Template-based import of ``Nav1p6_RI21_SC.mod``."""

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav1p1_MA25_BC")
class Nav1p1_MA25_BC(Nav1p6_MA20_GoC):
    """Template-based import of ``Nav1p1_MA25_BC.mod``."""

    __module__ = "braincell.channel"

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(22.0),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 8.0 * (u.mS / u.cm ** 2),
        gateCurrent: Union[brainstate.typing.ArrayLike, Callable] = 0.0,
        name: Optional[str] = None,
        solver: str = "rk4",
        substeps: int = 5,
    ):
        super().__init__(
            size=size,
            temp=temp,
            g_max=g_max,
            name=name,
            solver=solver,
            substeps=substeps,
        )
        self.phi = 2.7 ** (((self.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.gateCurrent = braintools.init.param(
            gateCurrent, self.varshape, allow_none=False
        )
        self.Oon = 2.3
        self.epsilon = 1e-12
        self.zgate = 2.5435
        self.gunit = 15.0e-9 * u.mS
        self.e0 = 1.60217646e-19 * u.coulomb
        self.alfac = (self.Oon / self.Con) ** (1 / 4)

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)

    def current(self, V, Na: IonInfo):
        conductive = self.g_max * self.O.value * (Na.E - V)
        gate_flip = (
                        self.f01(V) * self.C1.value
                        + (self.f02(V) - self.b01(V)) * self.C2.value
                        + (self.f03(V) - self.b02(V)) * self.C3.value
                        + (self.f04(V) - self.b03(V)) * self.C4.value
                        - self.b04(V) * self.C5.value
                        + self.f11(V) * self.I1.value
                        + (self.f12(V) - self.b11(V)) * self.I2.value
                        + (self.f13(V) - self.b12(V)) * self.I3.value
                        + (self.f14(V) - self.b13(V)) * self.I4.value
                        - self.b14(V) * self.I5.value
                    ) / u.ms
        nc = 1e12 * self.g_max / self.gunit
        igate = nc * 1e6 * self.e0 * self.zgate * gate_flip
        return conductive - u.math.where(self.gateCurrent != 0, igate, 0.0 * igate)


@register_channel("Nav1p1_RI21_SC")
class Nav1p1_RI21_SC(Nav1p1_MA25_BC):
    """Template-based import of ``Nav1p1_RI21_SC.mod``."""

    __module__ = "braincell.channel"

    def reset_state(self, V, Na: IonInfo, batch_size: int = None):
        self.reset_steady_state(V, Na, batch_size=batch_size)


@register_channel("Nav_MA20_GrC")
class Nav_MA20_GrC(Markov, IndependentIntegration):
    """Template-based import of ``Nav_MA20_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "OB", "fip", "bip"),
        ("I1", "I2", "f11", "b11"),
        ("I2", "I3", "f12", "b12"),
        ("I3", "I4", "f13", "b13"),
        ("I4", "I5", "f14", "b14"),
        ("C1", "I1", "fi1", "bi1"),
        ("C2", "I2", "fi2", "bi2"),
        ("C3", "I3", "fi3", "bi3"),
        ("C4", "I4", "fi4", "bi4"),
        ("C5", "I5", "fi5", "bi5"),
        ("O", "I6", "fin", "bin"),
        ("I5", "I6", "f1n", "b1n"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(32.0),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 13.0 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
        solver: str = "rk4",
        substeps: int = 5,
    ):
        super().__init__(size=size, name=name)
        IndependentIntegration.__init__(self, solver=solver)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1.")

        self.Aalfa = 353.91
        self.Valfa = 13.99
        self.Abeta = 1.272
        self.Vbeta = 13.99
        self.Agamma = 150.0
        self.Adelta = 40.0
        self.Aepsilon = 1.75
        self.Ateta = 0.0201
        self.Vteta = 25.0
        self.ACon = 0.005
        self.ACoff = 0.5
        self.AOon = 0.75
        self.AOoff = 0.005
        self.n1 = 5.422
        self.n2 = 3.279
        self.n3 = 1.83
        self.n4 = 0.738

    def make_integration(self, *args, **kwargs):
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self.solver(self, *args, **kwargs),
                u.math.arange(self.substeps),
            )

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    alfa = lambda self, V, *unused: self.phi * self.Aalfa * u.math.exp((V / u.mV) / self.Valfa)
    beta = lambda self, V, *unused: self.phi * self.Abeta * u.math.exp(-(V / u.mV) / self.Vbeta)
    teta = lambda self, V, *unused: self.phi * self.Ateta * u.math.exp(-(V / u.mV) / self.Vteta)
    gamma = lambda self, V, *unused: self.phi * self.Agamma
    delta = lambda self, V, *unused: self.phi * self.Adelta
    epsilon = lambda self, V, *unused: self.phi * self.Aepsilon
    Con = lambda self, V, *unused: self.phi * self.ACon
    Coff = lambda self, V, *unused: self.phi * self.ACoff
    Oon = lambda self, V, *unused: self.phi * self.AOon
    Ooff = lambda self, V, *unused: self.phi * self.AOoff
    a_factor = lambda self, V, *unused: (self.Oon(V) / self.Con(V)) ** 0.25
    b_factor = lambda self, V, *unused: (self.Ooff(V) / self.Coff(V)) ** 0.25

    f01 = lambda self, V, *unused: self.n1 * self.alfa(V)
    f02 = lambda self, V, *unused: self.n2 * self.alfa(V)
    f03 = lambda self, V, *unused: self.n3 * self.alfa(V)
    f04 = lambda self, V, *unused: self.n4 * self.alfa(V)
    f0O = lambda self, V, *unused: self.gamma(V)
    fip = lambda self, V, *unused: self.epsilon(V)
    f11 = lambda self, V, *unused: self.n1 * self.alfa(V) * self.a_factor(V)
    f12 = lambda self, V, *unused: self.n2 * self.alfa(V) * self.a_factor(V)
    f13 = lambda self, V, *unused: self.n3 * self.alfa(V) * self.a_factor(V)
    f14 = lambda self, V, *unused: self.n4 * self.alfa(V) * self.a_factor(V)
    f1n = lambda self, V, *unused: self.gamma(V)
    fi1 = lambda self, V, *unused: self.Con(V)
    fi2 = lambda self, V, *unused: self.Con(V) * self.a_factor(V)
    fi3 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 2
    fi4 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 3
    fi5 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 4
    fin = lambda self, V, *unused: self.Oon(V)

    b01 = lambda self, V, *unused: self.n4 * self.beta(V)
    b02 = lambda self, V, *unused: self.n3 * self.beta(V)
    b03 = lambda self, V, *unused: self.n2 * self.beta(V)
    b04 = lambda self, V, *unused: self.n1 * self.beta(V)
    b0O = lambda self, V, *unused: self.delta(V)
    bip = lambda self, V, *unused: self.teta(V)
    b11 = lambda self, V, *unused: self.n4 * self.beta(V) * self.b_factor(V)
    b12 = lambda self, V, *unused: self.n3 * self.beta(V) * self.b_factor(V)
    b13 = lambda self, V, *unused: self.n2 * self.beta(V) * self.b_factor(V)
    b14 = lambda self, V, *unused: self.n1 * self.beta(V) * self.b_factor(V)
    b1n = lambda self, V, *unused: self.delta(V)
    bi1 = lambda self, V, *unused: self.Coff(V)
    bi2 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V)
    bi3 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 2
    bi4 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 3
    bi5 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 4
    bin = lambda self, V, *unused: self.Ooff(V)


@register_channel("NaFHF_MA20_GrC")
class NaFHF_MA20_GrC(Markov, IndependentIntegration):
    """Template-based import of ``NaFHF_MA20_GrC.mod``."""

    __module__ = "braincell.channel"
    root_type = Sodium

    pairs = (
        ("C1", "C2", "f01", "b01"),
        ("C2", "C3", "f02", "b02"),
        ("C3", "C4", "f03", "b03"),
        ("C4", "C5", "f04", "b04"),
        ("C5", "O", "f0O", "b0O"),
        ("O", "OB", "fip", "bip"),
        ("I1", "I2", "f11", "b11"),
        ("I2", "I3", "f12", "b12"),
        ("I3", "I4", "f13", "b13"),
        ("I4", "I5", "f14", "b14"),
        ("L3", "L4", "f33", "b33"),
        ("L4", "L5", "f34", "b34"),
        ("L5", "L6", "f3n", "b3n"),
        ("C1", "I1", "fi1", "bi1"),
        ("C2", "I2", "fi2", "bi2"),
        ("C3", "I3", "fi3", "bi3"),
        ("C4", "I4", "fi4", "bi4"),
        ("C5", "I5", "fi5", "bi5"),
        ("C3", "L3", "fl3", "bl3"),
        ("C4", "L4", "fl4", "bl4"),
        ("C5", "L5", "fl5", "bl5"),
        ("O", "L6", "fl6", "bl6"),
        ("O", "I6", "fin", "bin"),
        ("I5", "I6", "f1n", "b1n"),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: brainstate.typing.ArrayLike = u.celsius2kelvin(32.0),
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 13.0 * (u.mS / u.cm ** 2),
        name: Optional[str] = None,
        solver: str = "rk4",
        substeps: int = 5,
    ):
        super().__init__(size=size, name=name)
        IndependentIntegration.__init__(self, solver=solver)

        self.temp = braintools.init.param(temp, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)
        self.phi = 3 ** (((self.temp - u.celsius2kelvin(20.0)) / u.kelvin) / 10.0)
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be at least 1.")

        self.Aalfa = 353.91
        self.Valfa = 13.99
        self.Abeta = 1.272
        self.Vbeta = 13.99
        self.Agamma = 150.0
        self.Adelta = 40.0
        self.Aepsilon = 1.75
        self.Ateta = 0.0201
        self.Vteta = 25.0
        self.ACon = 0.025
        self.ACoff = 0.5
        self.AOon = 0.75
        self.AOoff = 0.002
        self.n1 = 5.422
        self.n2 = 3.279
        self.n3 = 1.83
        self.n4 = 0.738
        self.ALon = 0.001
        self.ALoff = 0.5
        self.c = 20.0
        self.d = 0.075

    def make_integration(self, *args, **kwargs):
        with brainstate.environ.context(dt=brainstate.environ.get_dt() / self.substeps):
            brainstate.transform.for_loop(
                lambda i: self.solver(self, *args, **kwargs),
                u.math.arange(self.substeps),
            )

    def current(self, V, Na: IonInfo):
        return self.g_max * self.O.value * (Na.E - V)

    alfa = lambda self, V, *unused: self.phi * self.Aalfa * u.math.exp((V / u.mV) / self.Valfa)
    beta = lambda self, V, *unused: self.phi * self.Abeta * u.math.exp(-(V / u.mV) / self.Vbeta)
    teta = lambda self, V, *unused: self.phi * self.Ateta * u.math.exp(-(V / u.mV) / self.Vteta)
    gamma = lambda self, V, *unused: self.phi * self.Agamma
    delta = lambda self, V, *unused: self.phi * self.Adelta
    epsilon = lambda self, V, *unused: self.phi * self.Aepsilon
    Con = lambda self, V, *unused: self.phi * self.ACon
    Coff = lambda self, V, *unused: self.phi * self.ACoff
    Oon = lambda self, V, *unused: self.phi * self.AOon
    Ooff = lambda self, V, *unused: self.phi * self.AOoff
    a_factor = lambda self, V, *unused: (self.Oon(V) / self.Con(V)) ** 0.25
    b_factor = lambda self, V, *unused: (self.Ooff(V) / self.Coff(V)) ** 0.25
    Lon = lambda self, V, *unused: self.phi * self.ALon
    Loff = lambda self, V, *unused: self.phi * self.ALoff

    f01 = lambda self, V, *unused: self.n1 * self.alfa(V)
    f02 = lambda self, V, *unused: self.n2 * self.alfa(V)
    f03 = lambda self, V, *unused: self.n3 * self.alfa(V)
    f04 = lambda self, V, *unused: self.n4 * self.alfa(V)
    f0O = lambda self, V, *unused: self.gamma(V)
    fip = lambda self, V, *unused: self.epsilon(V)
    f11 = lambda self, V, *unused: self.n1 * self.alfa(V) * self.a_factor(V)
    f12 = lambda self, V, *unused: self.n2 * self.alfa(V) * self.a_factor(V)
    f13 = lambda self, V, *unused: self.n3 * self.alfa(V) * self.a_factor(V)
    f14 = lambda self, V, *unused: self.n4 * self.alfa(V) * self.a_factor(V)
    f1n = lambda self, V, *unused: self.gamma(V)
    f33 = lambda self, V, *unused: self.n3 * self.alfa(V) * self.c
    f34 = lambda self, V, *unused: self.n4 * self.alfa(V) * self.c
    f3n = lambda self, V, *unused: self.gamma(V)
    fi1 = lambda self, V, *unused: self.Con(V)
    fi2 = lambda self, V, *unused: self.Con(V) * self.a_factor(V)
    fi3 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 2
    fi4 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 3
    fi5 = lambda self, V, *unused: self.Con(V) * self.a_factor(V) ** 4
    fin = lambda self, V, *unused: self.Oon(V)
    fl3 = lambda self, V, *unused: self.Lon(V)
    fl4 = lambda self, V, *unused: self.Lon(V) * self.c
    fl5 = lambda self, V, *unused: self.Lon(V) * self.c ** 2
    fl6 = lambda self, V, *unused: self.Lon(V) * self.c ** 2

    b01 = lambda self, V, *unused: self.n4 * self.beta(V)
    b02 = lambda self, V, *unused: self.n3 * self.beta(V)
    b03 = lambda self, V, *unused: self.n2 * self.beta(V)
    b04 = lambda self, V, *unused: self.n1 * self.beta(V)
    b0O = lambda self, V, *unused: self.delta(V)
    bip = lambda self, V, *unused: self.teta(V)
    b11 = lambda self, V, *unused: self.n4 * self.beta(V) * self.b_factor(V)
    b12 = lambda self, V, *unused: self.n3 * self.beta(V) * self.b_factor(V)
    b13 = lambda self, V, *unused: self.n2 * self.beta(V) * self.b_factor(V)
    b14 = lambda self, V, *unused: self.n1 * self.beta(V) * self.b_factor(V)
    b1n = lambda self, V, *unused: self.delta(V)
    b33 = lambda self, V, *unused: self.n2 * self.alfa(V) * self.d
    b34 = lambda self, V, *unused: self.n1 * self.alfa(V) * self.d
    b3n = lambda self, V, *unused: self.delta(V)
    bi1 = lambda self, V, *unused: self.Coff(V)
    bi2 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V)
    bi3 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 2
    bi4 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 3
    bi5 = lambda self, V, *unused: self.Coff(V) * self.b_factor(V) ** 4
    bin = lambda self, V, *unused: self.Ooff(V)
    bl3 = lambda self, V, *unused: self.Loff(V)
    bl4 = lambda self, V, *unused: self.Loff(V) * self.d
    bl5 = lambda self, V, *unused: self.Loff(V) * self.d ** 2
    bl6 = lambda self, V, *unused: self.Loff(V) * self.d ** 2
