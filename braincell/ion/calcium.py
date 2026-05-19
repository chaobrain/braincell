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

# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainstate
import braintools
import brainunit as u

from braincell._base import Ion, HHTypedNeuron
from braincell.mech import register_ion
from braincell.ion._base import Conserve, DynamicNernstIon, Factor, FixedIon, InitNernstIon, KineticIon, Reaction, Source, Species

__all__ = [
    'Calcium',
    'CalciumFixed',
    'CalciumInitNernst',
    'CalciumDetailed',
    'CalciumFirstOrder',
    'ToyCaBindingKinetic_SU2015_DCN',
    'ToyCaBindingSourceKinetic_SU2015_DCN',
    'ToyCaBindingIcaSourceKinetic_SU2015_DCN',
    'ToyDiamFactorKinetic_SU2015_DCN',
    'ToyCaPumpFactorKinetic_SU2015_DCN',
    'CdpStC_CAMOnly_MA2020_GoC',
    'CdpStC_NoCAM_MA2020_GoC',
    'CdpStC_MA2020_GoC',
    'CdpHVA_SU2015_DCN',
    'CdpLVA_SU2015_DCN',
]


class Calcium(Ion):
    """
    Base class for modeling Calcium ion.

    This class serves as the foundation for all calcium ion models in the braincell library.
    It inherits from the Ion class and sets the root_type to HHTypedNeuron.

    Note:
        This is an abstract base class and should be subclassed to implement specific
        calcium ion models with their own dynamics and properties.
    """
    __module__ = 'braincell.ion'

    root_type = HHTypedNeuron
    ion_symbol = 'Ca'
    default_Ci = 5e-05 * u.mM
    default_Co = 2.0 * u.mM
    default_valence = 2


@register_ion("CalciumFixed")
class CalciumFixed(Calcium, FixedIon):
    """Fixed Calcium dynamics.

    This calcium model has no dynamics. It holds fixed reversal
    potential :math:`E` and concentration :math:`C`.
    """
    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable, None] = 120. * u.mV,
        Ci: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        valence: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_fixed_ion(Ci=Ci, Co=Co, E=E, valence=valence)


@register_ion("CalciumInitNernst")
class CalciumInitNernst(Calcium, InitNernstIon):
    """Fixed ``Ci/Co`` calcium model with ``E`` initialized from Nernst."""

    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        Ci: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        valence: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_nernst_ion(Ci=Ci, Co=Co, temp=temp, valence=valence)


@register_ion("CalciumDetailed")
class CalciumDetailed(Calcium, DynamicNernstIon):
    r"""Dynamical Calcium model proposed.

    **1. The dynamics of intracellular** :math:`Ca^{2+}`

    The dynamics of intracellular :math:`Ca^{2+}` were determined by two contributions [1]_ :

    *(i) Influx of* :math:`Ca^{2+}` *due to Calcium currents*

    :math:`Ca^{2+}` ion enter through :math:`Ca^{2+}` channel and diffuse into the
    interior of the cell. Only the :math:`Ca^{2+}` concentration in a thin shell beneath
    the membrane was modeled. The influx of :math:`Ca^{2+}` into such a thin shell followed:

    .. math::

        [Ca]_{i}=-\frac{I_{Ca}}{2 F d}

    where :math:`F=96489\, \mathrm{C\, mol^{-1}}` is the Faraday constant,
    :math:`d=1\, \mathrm{\mu m}` is the depth of the shell beneath the membrane,
    :math:`I_T` in :math:`\mathrm{\mu A/cm^{2}}` and :math:`[Ca]_{i}` in millimolar,
    and :math:`I_{Ca}` is the summation of all :math:`Ca^{2+}` currents.

    *(ii) Efflux of* :math:`Ca^{2+}` *due to an active pump*

    In a thin shell beneath the membrane, :math:`Ca^{2+}` retrieval usually consists of a
    combination of several processes, such as binding to :math:`Ca^{2+}` buffers, calcium
    efflux due to :math:`Ca^{2+}` ATPase pump activity and diffusion to neighboring shells.
    Only the :math:`Ca^{2+}` pump was modeled here. We adopted the following kinetic scheme:

    .. math::

        Ca _{i}^{2+}+ P \overset{c_1}{\underset{c_2}{\rightleftharpoons}} CaP \xrightarrow{c_3} P+ Ca _{0}^{2+}

    where P represents the :math:`Ca^{2+}` pump, CaP is an intermediate state,
    :math:`Ca _{ o }^{2+}` is the extracellular :math:`Ca^{2+}` concentration,
    and :math:`c_{1}, c_{2}` and :math:`c_{3}` are rate constants. :math:`Ca^{2+}`
    ion have a high affinity for the pump :math:`P`, whereas extrusion of
    :math:`Ca^{2+}` follows a slower process (Blaustein, 1988 ). Therefore,
    :math:`c_{3}` is low compared to :math:`c_{1}` and :math:`c_{2}` and the
    Michaelis-Menten approximation can be used for describing the kinetics of the pump.
    According to such a scheme, the kinetic equation for the :math:`Ca^{2+}` pump is:

    .. math::

        \frac{[Ca^{2+}]_{i}}{dt}=-\frac{K_{T}[Ca]_{i}}{[Ca]_{i}+K_{d}}

    where :math:`K_{T}=10^{-4}\, \mathrm{mM\, ms^{-1}}` is the product of :math:`c_{3}`
    with the total concentration of :math:`P` and :math:`K_{d}=c_{2} / c_{1}=10^{-4}\, \mathrm{mM}`
    is the dissociation constant, which can be interpreted here as the value of
    :math:`[Ca]_{i}` at which the pump is half activated (if :math:`[Ca]_{i} \ll K_{d}`
    then the efflux is negligible).

    **2.A simple first-order model**

    While, in (Bazhenov, et al., 1998) [2]_, the :math:`Ca^{2+}` dynamics is
    described by a simple first-order model,

    .. math::

        \frac{d\left[Ca^{2+}\right]_{i}}{d t}=-\frac{I_{Ca}}{z F d}+\frac{\left[Ca^{2+}\right]_{rest}-\left[C a^{2+}\right]_{i}}{\tau_{Ca}}

    where :math:`I_{Ca}` is the summation of all :math:`Ca ^{2+}` currents, :math:`d`
    is the thickness of the perimembrane "shell" in which calcium is able to affect
    membrane properties :math:`(1.\, \mathrm{\mu M})`, :math:`z=2` is the valence of the
    :math:`Ca ^{2+}` ion, :math:`F` is the Faraday constant, and :math:`\tau_{C a}` is
    the :math:`Ca ^{2+}` removal rate. The resting :math:`Ca ^{2+}` concentration was
    set to be :math:`\left[ Ca ^{2+}\right]_{\text {rest}}=.05\, \mathrm{\mu M}` .

    **3. The reversal potential**

    The reversal potential of calcium :math:`Ca ^{2+}` is calculated according to the
    Nernst equation:

    .. math::

        E = k'{RT \over 2F} log{[Ca^{2+}]_0 \over [Ca^{2+}]_i}

    where :math:`R=8.31441 \, \mathrm{J} /(\mathrm{mol}^{\circ} \mathrm{K})`,
    :math:`T=309.15^{\circ} \mathrm{K}`,
    :math:`F=96,489 \mathrm{C} / \mathrm{mol}`,
    and :math:`\left[\mathrm{Ca}^{2+}\right]_{0}=2 \mathrm{mM}`.

    Parameters
    ----------
    d : float
      The thickness of the peri-membrane "shell".
    F : float
      The Faraday constant. (:math:`C*mmol^{-1}`)
    tau : float
      The time constant of the :math:`Ca ^{2+}` removal rate. (ms)
    C_rest : float
      The resting :math:`Ca ^{2+}` concentration.
    Co : float
      The :math:`Ca ^{2+}` concentration outside of the membrane.
    R : float
      The gas constant. (:math:` J*mol^{-1}*K^{-1}`)

    References
    ----------

    .. [1] Destexhe, Alain, Agnessa Babloyantz, and Terrence J. Sejnowski.
           "Ionic mechanisms for intrinsic slow oscillations in thalamic
           relay neuron." Biophysical journal 65, no. 4 (1993): 1538-1552.
    .. [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and Terrence J.
           Sejnowski. "Cellular and network models for intrathalamic augmenting
           responses during 10-Hz stimulation." Journal of neurophysiology 79,
           no. 5 (1998): 2730-2748.

    """
    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        d: Union[brainstate.typing.ArrayLike, Callable] = 1. * u.um,
        tau: Union[brainstate.typing.ArrayLike, Callable] = 5. * u.ms,
        C_rest: Union[brainstate.typing.ArrayLike, Callable] = 2.4e-4 * u.mM,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        # parameters
        self.d = braintools.init.param(d, self.varshape, allow_none=False)
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)
        self.C_rest = braintools.init.param(C_rest, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        drive = total_current / (2 * u.faraday_constant * self.d)
        drive = u.math.maximum(drive, u.math.zeros_like(drive))
        return drive + (self.C_rest - Ci) / self.tau


@register_ion("CalciumFirstOrder")
class CalciumFirstOrder(Calcium, DynamicNernstIon):
    r"""
    The first-order calcium concentration model.

    .. math::

       Ca' = -\alpha I_{Ca} + -\beta Ca

    """
    __module__ = 'braincell.ion'
    uses_total_current = True

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.13,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.075,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = braintools.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        # parameters
        self.alpha = braintools.init.param(alpha, self.varshape, allow_none=False)
        self.beta = braintools.init.param(beta, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        drive = u.math.maximum(self.alpha * total_current, 0. * u.mM)
        return drive - self.beta * Ci


@register_ion("ToyCaBindingKinetic_SU2015_DCN")
class ToyCaBindingKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy for ``KineticIon`` validation.

    The mechanism models one reversible buffering step:

    .. math::

       Ca_i + B \rightleftharpoons BC

    with the conserved pool:

    .. math::

       B + BC = B_{tot}

    ``B`` is solved algebraically from the conservation rule while ``Ci`` and
    ``BC`` are integrated as differential species.
    """

    __module__ = "braincell.ion"

    species = (
        Species("Ci", init=0.10 * u.mM),
        Species("B", init=1.00 * u.mM),
        Species("BC", init=0.00 * u.mM),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "B": 1},
            rhs={"BC": 1},
            forward=lambda self, V, x: self.kf,
            backward=lambda self, V, x: self.kb,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("B", "BC"),
            algebraic="B",
            total=lambda self, V, x: self.Btot,
        ),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kf: Union[brainstate.typing.ArrayLike, Callable] = 2.0 / (u.mM * u.ms),
        kb: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        Btot: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.mM,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.10 * u.mM,
        BC_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.Ci_initializer = Ci_initializer
        self.BC_initializer = BC_initializer
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)


@register_ion("ToyCaBindingSourceKinetic_SU2015_DCN")
class ToyCaBindingSourceKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy with a constant ``Ci`` source.

    The mechanism keeps the same reversible binding network as
    :class:`ToyCaBindingKinetic_SU2015_DCN`:

    .. math::

       Ca_i + B \rightleftharpoons BC

    and adds one constant source term on ``Ci``:

    .. math::

       \frac{d Ca_i}{dt}\Big|_{\text{source}} = s_{Ca}
    """

    __module__ = "braincell.ion"

    species = ToyCaBindingKinetic_SU2015_DCN.species
    reactions = ToyCaBindingKinetic_SU2015_DCN.reactions
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x: self.ci_source,
        ),
    )
    conserves = ToyCaBindingKinetic_SU2015_DCN.conserves

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kf: Union[brainstate.typing.ArrayLike, Callable] = 2.0 / (u.mM * u.ms),
        kb: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        Btot: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.mM,
        ci_source: Union[brainstate.typing.ArrayLike, Callable] = 0.002 * u.mM / u.ms,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.10 * u.mM,
        BC_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.Ci_initializer = Ci_initializer
        self.BC_initializer = BC_initializer
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)
        self.ci_source = braintools.init.param(ci_source, self.varshape, allow_none=False)


@register_ion("ToyCaBindingIcaSourceKinetic_SU2015_DCN")
class ToyCaBindingIcaSourceKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal reversible calcium-binding toy with current-driven ``Ci`` source.

    The mechanism keeps the same reversible binding network as the earlier
    toy kinetic ions and drives ``Ci`` with inward-positive calcium current.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    species = ToyCaBindingKinetic_SU2015_DCN.species
    reactions = ToyCaBindingKinetic_SU2015_DCN.reactions
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: (
                braintools.init.param(0.0 * (u.mM / u.ms), self.varshape)
                if total_current is None else
                (
                    self.kCa.to_decimal(self.kCa.unit)
                    / self.depth.to_decimal(u.um)
                    * total_current.to_decimal(u.mA / u.cm ** 2)
                    * 1e4
                ) * (u.mM / u.ms)
            ),
        ),
    )
    conserves = ToyCaBindingKinetic_SU2015_DCN.conserves

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kf: Union[brainstate.typing.ArrayLike, Callable] = 2.0 / (u.mM * u.ms),
        kb: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        Btot: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.mM,
        kCa: Union[brainstate.typing.ArrayLike, Callable] = 3.45e-7 / u.coulomb,
        depth: Union[brainstate.typing.ArrayLike, Callable] = 0.2 * u.um,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.10 * u.mM,
        BC_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.00 * u.mM,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "BC": BC_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.Ci_initializer = Ci_initializer
        self.BC_initializer = BC_initializer
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.Btot = braintools.init.param(Btot, self.varshape, allow_none=False)
        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)


@register_ion("ToyCaPumpFactorKinetic_SU2015_DCN")
class ToyCaPumpFactorKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal factor-crossing toy with cytosolic and pump-area compartments.

    ``Ci`` lives in a cytosolic volume factor while pump states live in an
    area-like factor. The toy keeps the state count minimal while exercising
    mixed-factor reaction, conservation, and current-driven source paths.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.cyt_volume),
        Factor("pump_area", lambda self: self.pump_area),
    )
    species = (
        Species("Ci", init=0.10 * u.mM, factor="cyto"),
        Species("PumpFree", init=1.00 * u.mM * u.um, factor="pump_area"),
        Species("PumpBound", init=0.00 * u.mM * u.um, factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "PumpFree": 1},
            rhs={"PumpBound": 1},
            forward=lambda self, V, x: self.kf * self.pump_area,
            backward=lambda self, V, x: self.kb * self.pump_area,
        ),
        Reaction(
            lhs={"PumpBound": 1},
            rhs={"PumpFree": 1},
            forward=lambda self, V, x: self.k_rel * self.pump_area,
            backward=None,
        ),
    )
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: (
                braintools.init.param(0.0 * (u.mM * u.um ** 3 / u.ms), self.varshape)
                if total_current is None else
                self.cyt_volume * (
                    (
                        self.kCa.to_decimal(self.kCa.unit)
                        / self.depth.to_decimal(u.um)
                        * total_current.to_decimal(u.mA / u.cm ** 2)
                        * 1e4
                    ) * (u.mM / u.ms)
                )
            ),
        ),
    )
    conserves = (
        Conserve(
            species=("PumpFree", "PumpBound"),
            algebraic="PumpFree",
            total=lambda self, V, x: self.PumpTot * self.pump_area,
        ),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kf: Union[brainstate.typing.ArrayLike, Callable] = 2.0 / (u.mM * u.ms),
        kb: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        k_rel: Union[brainstate.typing.ArrayLike, Callable] = 0.05 / u.ms,
        PumpTot: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.mM * u.um,
        kCa: Union[brainstate.typing.ArrayLike, Callable] = 3.45e-7 / u.coulomb,
        depth: Union[brainstate.typing.ArrayLike, Callable] = 0.2 * u.um,
        cyt_volume: Union[brainstate.typing.ArrayLike, Callable] = 3.0 * u.um ** 3,
        pump_area: Union[brainstate.typing.ArrayLike, Callable] = 3.0 * u.um ** 2,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.10 * u.mM,
        PumpBound_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.00 * u.mM * u.um,
        solver: str = "rk4",
        substeps: int = 5,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "PumpBound": PumpBound_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.Ci_initializer = Ci_initializer
        self.PumpBound_initializer = PumpBound_initializer
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.k_rel = braintools.init.param(k_rel, self.varshape, allow_none=False)
        self.PumpTot = braintools.init.param(PumpTot, self.varshape, allow_none=False)
        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)
        self.cyt_volume = braintools.init.param(cyt_volume, self.varshape, allow_none=False)
        self.pump_area = braintools.init.param(pump_area, self.varshape, allow_none=False)


@register_ion("ToyDiamFactorKinetic_SU2015_DCN")
class ToyDiamFactorKinetic_SU2015_DCN(Calcium, KineticIon):
    r"""Minimal geometry-factor toy with runtime-derived cytosolic strip factors.

    ``Ci`` lives in a thin strip volume derived from the runtime midpoint
    diameter, while ``PumpFree`` and ``PumpBound`` live on a line-like
    membrane factor:

    .. math::

       cyto = \pi \cdot diam_{mid} \cdot depth

       pump\_area = \pi \cdot diam_{mid}

    The mechanism then exercises a reversible reaction

    .. math::

       Ca_i + PumpFree \rightleftharpoons PumpBound

    together with the conserved pool

    .. math::

       PumpFree + PumpBound = PumpTot \cdot pump\_area
    """

    __module__ = "braincell.ion"

    factors = (
        Factor("cyto", lambda self: u.math.pi * self.diam_mid * self.depth),
        Factor("pump_area", lambda self: u.math.pi * self.diam_mid),
    )
    species = (
        Species("Ci", init=0.10 * u.mM, factor="cyto"),
        Species("PumpFree", init=1.00 * u.mM * u.um, factor="pump_area"),
        Species("PumpBound", init=0.00 * u.mM * u.um, factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "PumpFree": 1},
            rhs={"PumpBound": 1},
            forward=lambda self, V, x: self.kf * u.math.pi * self.diam_mid,
            backward=lambda self, V, x: self.kb * u.math.pi * self.diam_mid,
        ),
    )
    sources = ()
    conserves = (
        Conserve(
            species=("PumpFree", "PumpBound"),
            algebraic="PumpFree",
            total=lambda self, V, x: self.PumpTot * u.math.pi * self.diam_mid,
        ),
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kf: Union[brainstate.typing.ArrayLike, Callable] = 2.0 / (u.mM * u.ms),
        kb: Union[brainstate.typing.ArrayLike, Callable] = 0.5 / u.ms,
        PumpTot: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.mM * u.um,
        depth: Union[brainstate.typing.ArrayLike, Callable] = 1.0 * u.um,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.10 * u.mM,
        PumpBound_initializer: Union[brainstate.typing.ArrayLike, Callable] = 0.00 * u.mM * u.um,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers={
                "Ci": Ci_initializer,
                "PumpBound": PumpBound_initializer,
            },
            solver=solver,
            substeps=substeps,
        )
        self.Ci_initializer = Ci_initializer
        self.PumpBound_initializer = PumpBound_initializer
        self.kf = braintools.init.param(kf, self.varshape, allow_none=False)
        self.kb = braintools.init.param(kb, self.varshape, allow_none=False)
        self.PumpTot = braintools.init.param(PumpTot, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)


@register_ion("CdpStC_CAMOnly_MA2020_GoC")
class CdpStC_CAMOnly_MA2020_GoC(Calcium, KineticIon):
    r"""Template-based import of ``CdpStC_CAMOnly_MA20_GoC.mod``.

    This variant keeps only the calmodulin subnetwork from the imported GoC
    calcium pool so the CAM-specific semantics can be validated separately
    from pump and non-CaM buffers.
    """

    __module__ = "braincell.ion"
    uses_total_current = False

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        # NEURON sparse treats the CAM rows differently from ``ca``: their
        # reaction rates carry the same ``dsqvol`` unit bridge, but the CAM
        # state rows are not multiplied by the cytosolic compartment factor
        # again. We match that by giving CAM states a unit-compatible factor
        # whose magnitude is 1 instead of ``dsqvol``.
        Factor(
            "cam_unit",
            lambda self: u.math.ones_like(self.dsqvol.to_decimal(u.um ** 2)) * (u.um ** 2),
        ),
    )
    species = (
        Species("Ci", init=0.0 * u.mM, factor="cyto"),
        Species("CAM0", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1N2C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2N1C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1C1N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM4", init=0.0 * u.mM, factor="cam_unit"),
    )
    reactions = (
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2C": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N2C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM2N": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N1C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
    )
    sources = ()
    conserves = ()

    _diffeq_species = (
        "Ci",
        "CAM0",
        "CAM1C",
        "CAM2C",
        "CAM1N2C",
        "CAM1N",
        "CAM2N",
        "CAM2N1C",
        "CAM1C1N",
        "CAM4",
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(25.0),
        Nannuli: Union[brainstate.typing.ArrayLike, Callable] = 10.9495,
        cainull: Union[brainstate.typing.ArrayLike, Callable] = 45e-6 * u.mM,
        CAM_start: Union[brainstate.typing.ArrayLike, Callable] = 0.03 * u.mM,
        K1Coff: Union[brainstate.typing.ArrayLike, Callable] = 0.04 / u.ms,
        K1Con: Union[brainstate.typing.ArrayLike, Callable] = 5.4 / (u.mM * u.ms),
        K2Coff: Union[brainstate.typing.ArrayLike, Callable] = 0.00925 / u.ms,
        K2Con: Union[brainstate.typing.ArrayLike, Callable] = 15.0 / (u.mM * u.ms),
        K1Noff: Union[brainstate.typing.ArrayLike, Callable] = 2.5 / u.ms,
        K1Non: Union[brainstate.typing.ArrayLike, Callable] = 142.5 / (u.mM * u.ms),
        K2Noff: Union[brainstate.typing.ArrayLike, Callable] = 0.75 / u.ms,
        K2Non: Union[brainstate.typing.ArrayLike, Callable] = 175.0 / (u.mM * u.ms),
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.CAM_start = braintools.init.param(CAM_start, self.varshape, allow_none=False)
        self.K1Coff = braintools.init.param(K1Coff, self.varshape, allow_none=False)
        self.K1Con = braintools.init.param(K1Con, self.varshape, allow_none=False)
        self.K2Coff = braintools.init.param(K2Coff, self.varshape, allow_none=False)
        self.K2Con = braintools.init.param(K2Con, self.varshape, allow_none=False)
        self.K1Noff = braintools.init.param(K1Noff, self.varshape, allow_none=False)
        self.K1Non = braintools.init.param(K1Non, self.varshape, allow_none=False)
        self.K2Noff = braintools.init.param(K2Noff, self.varshape, allow_none=False)
        self.K2Non = braintools.init.param(K2Non, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self.Ci_initializer = initializers["Ci"]
        self.species_initializers = dict(initializers)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _resolve_species_initializers(
        self,
        *,
        Ci_initializer,
        species_initializers,
    ) -> dict[str, object]:
        species_initializers = dict(species_initializers or {})
        invalid = set(species_initializers).difference(self._diffeq_species)
        if invalid:
            invalid_names = ", ".join(sorted(invalid))
            raise ValueError(
                f"{type(self).__name__} only accepts differential-species overrides; got {invalid_names}."
            )

        defaults = {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "CAM0": self.CAM_start,
            "CAM1C": 0.0 * u.mM,
            "CAM2C": 0.0 * u.mM,
            "CAM1N2C": 0.0 * u.mM,
            "CAM1N": 0.0 * u.mM,
            "CAM2N": 0.0 * u.mM,
            "CAM2N1C": 0.0 * u.mM,
            "CAM1C1N": 0.0 * u.mM,
            "CAM4": 0.0 * u.mM,
        }
        defaults.update(species_initializers)
        return {name: self._as_initializer(value) for name, value in defaults.items()}

    def _as_initializer(self, value):
        if callable(value):
            return value
        if isinstance(value, tuple):
            resolved = []
            for item in value:
                if hasattr(item, "value"):
                    resolved.append(item.value)
                else:
                    resolved.append(item)
            first = resolved[0]
            if hasattr(first, "unit"):
                unit = first.unit
                decimals = [u.Quantity(item).to_decimal(unit) for item in resolved]
                return u.Quantity(u.math.asarray(decimals), unit)
            return u.math.asarray(resolved)
        return braintools.init.Constant(value)

    def _require_diam_mid(self):
        if not hasattr(self, "diam_mid"):
            raise AttributeError(
                f"{type(self).__name__} requires 'diam_mid' before kinetic state initialization."
            )
        return self.diam_mid

    @property
    def vrat(self):
        dr2 = 0.25 / (self.Nannuli - 1.0)
        return u.math.pi * (0.5 - (dr2 / 2.0)) * 2.0 * dr2

    @property
    def dsq(self):
        diam_mid = self._require_diam_mid()
        return diam_mid * diam_mid

    @property
    def dsqvol(self):
        return self.dsq * self.vrat

    def _ion_init_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_init_state_hook(self, V, batch_size=batch_size)

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_reset_state_hook(self, V, batch_size=batch_size)


@register_ion("CdpStC_NoCAM_MA2020_GoC")
class CdpStC_NoCAM_MA2020_GoC(Calcium, KineticIon):
    r"""Template-based import of ``CdpStC_NoCAM_MA20_GoC.mod``.

    This variant keeps the pump and non-calmodulin buffer subnetworks from the
    imported GoC calcium pool while removing the CAM reactions entirely.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
    )
    species = (
        Species("Ci", init=0.0 * u.mM, factor="cyto"),
        Species("mg", init=0.0 * u.mM, factor="cyto"),
        Species("Buff1", init=0.0 * u.mM, factor="cyto"),
        Species("Buff1_ca", init=0.0 * u.mM, factor="cyto"),
        Species("Buff2", init=0.0 * u.mM, factor="cyto"),
        Species("Buff2_ca", init=0.0 * u.mM, factor="cyto"),
        Species("BTC", init=0.0 * u.mM, factor="cyto"),
        Species("BTC_ca", init=0.0 * u.mM, factor="cyto"),
        Species("DMNPE", init=0.0 * u.mM, factor="cyto"),
        Species("DMNPE_ca", init=0.0 * u.mM, factor="cyto"),
        Species("PV", init=0.0 * u.mM, factor="cyto"),
        Species("PV_ca", init=0.0 * u.mM, factor="cyto"),
        Species("PV_mg", init=0.0 * u.mM, factor="cyto"),
        Species("pump", init=0.0 * (u.mol / u.cm ** 2), factor="pump_area"),
        Species("pumpca", init=0.0 * (u.mol / u.cm ** 2), factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"pump": 1, "Ci": 1},
            rhs={"pumpca": 1},
            forward=lambda self, V, x: self.kpmp1 * self.parea,
            backward=lambda self, V, x: self.kpmp2 * self.parea,
        ),
        Reaction(
            lhs={"pumpca": 1},
            rhs={"pump": 1},
            forward=lambda self, V, x: self.kpmp3 * self.parea,
            backward=None,
        ),
        Reaction(
            lhs={"Ci": 1, "Buff1": 1},
            rhs={"Buff1_ca": 1},
            forward=lambda self, V, x: self.rf1 * self.dsqvol,
            backward=lambda self, V, x: self.rf2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "Buff2": 1},
            rhs={"Buff2_ca": 1},
            forward=lambda self, V, x: self.rf3 * self.dsqvol,
            backward=lambda self, V, x: self.rf4 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "BTC": 1},
            rhs={"BTC_ca": 1},
            forward=lambda self, V, x: self.b1 * self.dsqvol,
            backward=lambda self, V, x: self.b2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "DMNPE": 1},
            rhs={"DMNPE_ca": 1},
            forward=lambda self, V, x: self.c1 * self.dsqvol,
            backward=lambda self, V, x: self.c2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "PV": 1},
            rhs={"PV_ca": 1},
            forward=lambda self, V, x: self.m1 * self.dsqvol,
            backward=lambda self, V, x: self.m2 * self.dsqvol,
        ),
        Reaction(
            lhs={"mg": 1, "PV": 1},
            rhs={"PV_mg": 1},
            forward=lambda self, V, x: self.p1 * self.dsqvol,
            backward=lambda self, V, x: self.p2 * self.dsqvol,
        ),
    )
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: self._ci_source_flux(total_current),
        ),
    )
    conserves = (
        Conserve(
            species=("pump", "pumpca"),
            algebraic="pumpca",
            total=lambda self, V, x: self.TotalPump * self.parea,
        ),
    )

    _diffeq_species = (
        "Ci",
        "mg",
        "Buff1",
        "Buff1_ca",
        "Buff2",
        "Buff2_ca",
        "BTC",
        "BTC_ca",
        "DMNPE",
        "DMNPE_ca",
        "PV",
        "PV_ca",
        "PV_mg",
        "pump",
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(25.0),
        Nannuli: Union[brainstate.typing.ArrayLike, Callable] = 10.9495,
        cainull: Union[brainstate.typing.ArrayLike, Callable] = 45e-6 * u.mM,
        mginull: Union[brainstate.typing.ArrayLike, Callable] = 0.59 * u.mM,
        Buffnull1: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        rf1: Union[brainstate.typing.ArrayLike, Callable] = 0.0134329 / (u.mM * u.ms),
        rf2: Union[brainstate.typing.ArrayLike, Callable] = 0.0397469 / u.ms,
        Buffnull2: Union[brainstate.typing.ArrayLike, Callable] = 60.9091 * u.mM,
        rf3: Union[brainstate.typing.ArrayLike, Callable] = 0.1435 / (u.mM * u.ms),
        rf4: Union[brainstate.typing.ArrayLike, Callable] = 0.0014 / u.ms,
        BTCnull: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        b1: Union[brainstate.typing.ArrayLike, Callable] = 5.33 / (u.mM * u.ms),
        b2: Union[brainstate.typing.ArrayLike, Callable] = 0.08 / u.ms,
        DMNPEnull: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        c1: Union[brainstate.typing.ArrayLike, Callable] = 5.63 / (u.mM * u.ms),
        c2: Union[brainstate.typing.ArrayLike, Callable] = 0.107e-3 / u.ms,
        PVnull: Union[brainstate.typing.ArrayLike, Callable] = 0.08 * u.mM,
        m1: Union[brainstate.typing.ArrayLike, Callable] = 1.07e2 / (u.mM * u.ms),
        m2: Union[brainstate.typing.ArrayLike, Callable] = 9.5e-4 / u.ms,
        p1: Union[brainstate.typing.ArrayLike, Callable] = 0.8 / (u.mM * u.ms),
        p2: Union[brainstate.typing.ArrayLike, Callable] = 2.5e-2 / u.ms,
        kpmp1: Union[brainstate.typing.ArrayLike, Callable] = 3e-3 / (u.mM * u.ms),
        kpmp2: Union[brainstate.typing.ArrayLike, Callable] = 1.75e-5 / u.ms,
        kpmp3: Union[brainstate.typing.ArrayLike, Callable] = 7.255e-5 / u.ms,
        TotalPump: Union[brainstate.typing.ArrayLike, Callable] = 1e-9 * (u.mol / u.cm ** 2),
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.PVnull = braintools.init.param(PVnull, self.varshape, allow_none=False)
        self.m1 = braintools.init.param(m1, self.varshape, allow_none=False)
        self.m2 = braintools.init.param(m2, self.varshape, allow_none=False)
        self.p1 = braintools.init.param(p1, self.varshape, allow_none=False)
        self.p2 = braintools.init.param(p2, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self.Ci_initializer = initializers["Ci"]
        self.species_initializers = dict(initializers)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _resolve_species_initializers(
        self,
        *,
        Ci_initializer,
        species_initializers,
    ) -> dict[str, object]:
        species_initializers = dict(species_initializers or {})
        invalid = set(species_initializers).difference(self._diffeq_species)
        if invalid:
            invalid_names = ", ".join(sorted(invalid))
            raise ValueError(
                f"{type(self).__name__} only accepts differential-species overrides; got {invalid_names}."
            )

        defaults = {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "PV": self._ss_pv_free(),
            "PV_ca": self._ss_pv_ca(),
            "PV_mg": self._ss_pv_mg(),
            "pump": self.TotalPump,
        }
        defaults.update(species_initializers)
        return {name: self._as_initializer(value) for name, value in defaults.items()}

    def _as_initializer(self, value):
        if callable(value):
            return value
        if isinstance(value, tuple):
            resolved = []
            for item in value:
                if hasattr(item, "value"):
                    resolved.append(item.value)
                else:
                    resolved.append(item)
            first = resolved[0]
            if hasattr(first, "unit"):
                unit = first.unit
                decimals = [u.Quantity(item).to_decimal(unit) for item in resolved]
                return u.Quantity(u.math.asarray(decimals), unit)
            return u.math.asarray(resolved)
        return braintools.init.Constant(value)

    def _ss_buffer_free(self, total, kon, koff, cai):
        return total / (1.0 + (kon / koff) * cai)

    def _ss_buffer_bound(self, total, kon, koff, cai):
        return total / (1.0 + koff / (kon * cai))

    def _kdc(self):
        return (self.cainull * self.m1) / self.m2

    def _kdm(self):
        return (self.mginull * self.p1) / self.p2

    def _ss_pv_free(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return self.PVnull / (1.0 + kdc + kdm)

    def _ss_pv_ca(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return (self.PVnull * kdc) / (1.0 + kdc + kdm)

    def _ss_pv_mg(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return (self.PVnull * kdm) / (1.0 + kdc + kdm)

    def _require_diam_mid(self):
        if not hasattr(self, "diam_mid"):
            raise AttributeError(
                f"{type(self).__name__} requires 'diam_mid' before kinetic state initialization."
            )
        return self.diam_mid

    @property
    def vrat(self):
        dr2 = 0.25 / (self.Nannuli - 1.0)
        return u.math.pi * (0.5 - (dr2 / 2.0)) * 2.0 * dr2

    @property
    def parea(self):
        return u.math.pi * self._require_diam_mid()

    @property
    def dsq(self):
        diam_mid = self._require_diam_mid()
        return diam_mid * diam_mid

    @property
    def dsqvol(self):
        return self.dsq * self.vrat

    def _ci_source_flux(self, total_current):
        if total_current is None:
            return self.dsqvol * (0.0 * u.mM / u.ms)
        return -(total_current * u.math.pi * self._require_diam_mid()) / (2.0 * u.faraday_constant)

    def _ion_init_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_init_state_hook(self, V, batch_size=batch_size)

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_reset_state_hook(self, V, batch_size=batch_size)


@register_ion("CdpStC_MA2020_GoC")
class CdpStC_MA2020_GoC(Calcium, KineticIon):
    r"""Template-based import of ``CdpStC_MA20_GoC.mod``.

    ``Ci`` corresponds to the NMODL calcium pool ``ca`` / ``cai``. The
    reversible kinetic scheme is preserved as 20 explicit reactions, plus the
    original current-driven source and the single pump conservation relation.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    # The imported NMODL uses ``COMPARTMENT (1e10)*parea`` because ``pump``
    # and ``pumpca`` are stored visibly in ``mol/cm2`` and NEURON needs an
    # explicit area conversion to reach amount space. BrainCell factors already
    # provide that visible-to-amount mapping, so keeping the extra ``1e10``
    # here would double-apply the pump compartment scaling.
    factors = (
        Factor("cyto", lambda self: self.dsqvol),
        Factor("pump_area", lambda self: self.parea),
        Factor(
            "cam_unit",
            lambda self: u.math.ones_like(self.dsqvol.to_decimal(u.um ** 2)) * (u.um ** 2),
        ),
    )
    species = (
        Species("Ci", init=0.0 * u.mM, factor="cyto"),
        Species("mg", init=0.0 * u.mM, factor="cyto"),
        Species("Buff1", init=0.0 * u.mM, factor="cyto"),
        Species("Buff1_ca", init=0.0 * u.mM, factor="cyto"),
        Species("Buff2", init=0.0 * u.mM, factor="cyto"),
        Species("Buff2_ca", init=0.0 * u.mM, factor="cyto"),
        Species("BTC", init=0.0 * u.mM, factor="cyto"),
        Species("BTC_ca", init=0.0 * u.mM, factor="cyto"),
        Species("DMNPE", init=0.0 * u.mM, factor="cyto"),
        Species("DMNPE_ca", init=0.0 * u.mM, factor="cyto"),
        Species("PV", init=0.0 * u.mM, factor="cyto"),
        Species("PV_ca", init=0.0 * u.mM, factor="cyto"),
        Species("PV_mg", init=0.0 * u.mM, factor="cyto"),
        Species("CAM0", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1N2C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM2N1C", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM1C1N", init=0.0 * u.mM, factor="cam_unit"),
        Species("CAM4", init=0.0 * u.mM, factor="cam_unit"),
        Species("pump", init=0.0 * (u.mol / u.cm ** 2), factor="pump_area"),
        Species("pumpca", init=0.0 * (u.mol / u.cm ** 2), factor="pump_area"),
    )
    reactions = (
        Reaction(
            lhs={"pump": 1, "Ci": 1},
            rhs={"pumpca": 1},
            forward=lambda self, V, x: self.kpmp1 * self.parea,
            backward=lambda self, V, x: self.kpmp2 * self.parea,
        ),
        Reaction(
            lhs={"pumpca": 1},
            rhs={"pump": 1},
            forward=lambda self, V, x: self.kpmp3 * self.parea,
            backward=None,
        ),
        Reaction(
            lhs={"Ci": 1, "Buff1": 1},
            rhs={"Buff1_ca": 1},
            forward=lambda self, V, x: self.rf1 * self.dsqvol,
            backward=lambda self, V, x: self.rf2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "Buff2": 1},
            rhs={"Buff2_ca": 1},
            forward=lambda self, V, x: self.rf3 * self.dsqvol,
            backward=lambda self, V, x: self.rf4 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "BTC": 1},
            rhs={"BTC_ca": 1},
            forward=lambda self, V, x: self.b1 * self.dsqvol,
            backward=lambda self, V, x: self.b2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "DMNPE": 1},
            rhs={"DMNPE_ca": 1},
            forward=lambda self, V, x: self.c1 * self.dsqvol,
            backward=lambda self, V, x: self.c2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "PV": 1},
            rhs={"PV_ca": 1},
            forward=lambda self, V, x: self.m1 * self.dsqvol,
            backward=lambda self, V, x: self.m2 * self.dsqvol,
        ),
        Reaction(
            lhs={"mg": 1, "PV": 1},
            rhs={"PV_mg": 1},
            forward=lambda self, V, x: self.p1 * self.dsqvol,
            backward=lambda self, V, x: self.p2 * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2C": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N2C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM0": 1},
            rhs={"CAM1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM2N": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM2N1C": 1},
            rhs={"CAM4": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Non * self.dsqvol,
            backward=lambda self, V, x: self.K1Noff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1N": 1},
            rhs={"CAM1C1N": 1},
            forward=lambda self, V, x: self.K1Con * self.dsqvol,
            backward=lambda self, V, x: self.K1Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM1N2C": 1},
            forward=lambda self, V, x: self.K2Con * self.dsqvol,
            backward=lambda self, V, x: self.K2Coff * self.dsqvol,
        ),
        Reaction(
            lhs={"Ci": 1, "CAM1C1N": 1},
            rhs={"CAM2N1C": 1},
            forward=lambda self, V, x: self.K2Non * self.dsqvol,
            backward=lambda self, V, x: self.K2Noff * self.dsqvol,
        ),
    )
    sources = (
        Source(
            target="Ci",
            flux=lambda self, V, x, total_current=None: self._ci_source_flux(total_current),
        ),
    )
    conserves = (
        Conserve(
            species=("pump", "pumpca"),
            algebraic="pumpca",
            total=lambda self, V, x: self.TotalPump * self.parea,
        ),
    )

    _diffeq_species = (
        "Ci",
        "mg",
        "Buff1",
        "Buff1_ca",
        "Buff2",
        "Buff2_ca",
        "BTC",
        "BTC_ca",
        "DMNPE",
        "DMNPE_ca",
        "PV",
        "PV_ca",
        "PV_mg",
        "CAM0",
        "CAM1C",
        "CAM2C",
        "CAM1N2C",
        "CAM1N",
        "CAM2N",
        "CAM2N1C",
        "CAM1C1N",
        "CAM4",
        "pump",
    )

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(25.0),
        Nannuli: Union[brainstate.typing.ArrayLike, Callable] = 10.9495,
        cainull: Union[brainstate.typing.ArrayLike, Callable] = 45e-6 * u.mM,
        mginull: Union[brainstate.typing.ArrayLike, Callable] = 0.59 * u.mM,
        Buffnull1: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        rf1: Union[brainstate.typing.ArrayLike, Callable] = 0.0134329 / (u.mM * u.ms),
        rf2: Union[brainstate.typing.ArrayLike, Callable] = 0.0397469 / u.ms,
        Buffnull2: Union[brainstate.typing.ArrayLike, Callable] = 60.9091 * u.mM,
        rf3: Union[brainstate.typing.ArrayLike, Callable] = 0.1435 / (u.mM * u.ms),
        rf4: Union[brainstate.typing.ArrayLike, Callable] = 0.0014 / u.ms,
        BTCnull: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        b1: Union[brainstate.typing.ArrayLike, Callable] = 5.33 / (u.mM * u.ms),
        b2: Union[brainstate.typing.ArrayLike, Callable] = 0.08 / u.ms,
        DMNPEnull: Union[brainstate.typing.ArrayLike, Callable] = 0.0 * u.mM,
        c1: Union[brainstate.typing.ArrayLike, Callable] = 5.63 / (u.mM * u.ms),
        c2: Union[brainstate.typing.ArrayLike, Callable] = 0.107e-3 / u.ms,
        PVnull: Union[brainstate.typing.ArrayLike, Callable] = 0.08 * u.mM,
        m1: Union[brainstate.typing.ArrayLike, Callable] = 1.07e2 / (u.mM * u.ms),
        m2: Union[brainstate.typing.ArrayLike, Callable] = 9.5e-4 / u.ms,
        p1: Union[brainstate.typing.ArrayLike, Callable] = 0.8 / (u.mM * u.ms),
        p2: Union[brainstate.typing.ArrayLike, Callable] = 2.5e-2 / u.ms,
        CAM_start: Union[brainstate.typing.ArrayLike, Callable] = 0.03 * u.mM,
        K1Coff: Union[brainstate.typing.ArrayLike, Callable] = 0.04 / u.ms,
        K1Con: Union[brainstate.typing.ArrayLike, Callable] = 5.4 / (u.mM * u.ms),
        K2Coff: Union[brainstate.typing.ArrayLike, Callable] = 0.00925 / u.ms,
        K2Con: Union[brainstate.typing.ArrayLike, Callable] = 15.0 / (u.mM * u.ms),
        K1Noff: Union[brainstate.typing.ArrayLike, Callable] = 2.5 / u.ms,
        K1Non: Union[brainstate.typing.ArrayLike, Callable] = 142.5 / (u.mM * u.ms),
        K2Noff: Union[brainstate.typing.ArrayLike, Callable] = 0.75 / u.ms,
        K2Non: Union[brainstate.typing.ArrayLike, Callable] = 175.0 / (u.mM * u.ms),
        kpmp1: Union[brainstate.typing.ArrayLike, Callable] = 3e-3 / (u.mM * u.ms),
        kpmp2: Union[brainstate.typing.ArrayLike, Callable] = 1.75e-5 / u.ms,
        kpmp3: Union[brainstate.typing.ArrayLike, Callable] = 7.255e-5 / u.ms,
        TotalPump: Union[brainstate.typing.ArrayLike, Callable] = 1e-9 * (u.mol / u.cm ** 2),
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        species_initializers: Optional[dict[str, object]] = None,
        solver: str = "backward_euler",
        substeps: int = 1,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size=size, name=name, **channels)
        self.Nannuli = braintools.init.param(Nannuli, self.varshape, allow_none=False)
        self.cainull = braintools.init.param(cainull, self.varshape, allow_none=False)
        self.mginull = braintools.init.param(mginull, self.varshape, allow_none=False)
        self.Buffnull1 = braintools.init.param(Buffnull1, self.varshape, allow_none=False)
        self.rf1 = braintools.init.param(rf1, self.varshape, allow_none=False)
        self.rf2 = braintools.init.param(rf2, self.varshape, allow_none=False)
        self.Buffnull2 = braintools.init.param(Buffnull2, self.varshape, allow_none=False)
        self.rf3 = braintools.init.param(rf3, self.varshape, allow_none=False)
        self.rf4 = braintools.init.param(rf4, self.varshape, allow_none=False)
        self.BTCnull = braintools.init.param(BTCnull, self.varshape, allow_none=False)
        self.b1 = braintools.init.param(b1, self.varshape, allow_none=False)
        self.b2 = braintools.init.param(b2, self.varshape, allow_none=False)
        self.DMNPEnull = braintools.init.param(DMNPEnull, self.varshape, allow_none=False)
        self.c1 = braintools.init.param(c1, self.varshape, allow_none=False)
        self.c2 = braintools.init.param(c2, self.varshape, allow_none=False)
        self.PVnull = braintools.init.param(PVnull, self.varshape, allow_none=False)
        self.m1 = braintools.init.param(m1, self.varshape, allow_none=False)
        self.m2 = braintools.init.param(m2, self.varshape, allow_none=False)
        self.p1 = braintools.init.param(p1, self.varshape, allow_none=False)
        self.p2 = braintools.init.param(p2, self.varshape, allow_none=False)
        self.CAM_start = braintools.init.param(CAM_start, self.varshape, allow_none=False)
        self.K1Coff = braintools.init.param(K1Coff, self.varshape, allow_none=False)
        self.K1Con = braintools.init.param(K1Con, self.varshape, allow_none=False)
        self.K2Coff = braintools.init.param(K2Coff, self.varshape, allow_none=False)
        self.K2Con = braintools.init.param(K2Con, self.varshape, allow_none=False)
        self.K1Noff = braintools.init.param(K1Noff, self.varshape, allow_none=False)
        self.K1Non = braintools.init.param(K1Non, self.varshape, allow_none=False)
        self.K2Noff = braintools.init.param(K2Noff, self.varshape, allow_none=False)
        self.K2Non = braintools.init.param(K2Non, self.varshape, allow_none=False)
        self.kpmp1 = braintools.init.param(kpmp1, self.varshape, allow_none=False)
        self.kpmp2 = braintools.init.param(kpmp2, self.varshape, allow_none=False)
        self.kpmp3 = braintools.init.param(kpmp3, self.varshape, allow_none=False)
        self.TotalPump = braintools.init.param(TotalPump, self.varshape, allow_none=False)

        initializers = self._resolve_species_initializers(
            Ci_initializer=Ci_initializer,
            species_initializers=species_initializers,
        )
        self.Ci_initializer = initializers["Ci"]
        self.species_initializers = dict(initializers)
        self._init_kinetic_ion(
            Co=Co,
            temp=temp,
            valence=None,
            species_initializers=initializers,
            solver=solver,
            substeps=substeps,
        )

    def _resolve_species_initializers(
        self,
        *,
        Ci_initializer,
        species_initializers,
    ) -> dict[str, object]:
        species_initializers = dict(species_initializers or {})
        invalid = set(species_initializers).difference(self._diffeq_species)
        if invalid:
            invalid_names = ", ".join(sorted(invalid))
            raise ValueError(
                f"{type(self).__name__} only accepts differential-species overrides; got {invalid_names}."
            )

        defaults = {
            "Ci": self.cainull if Ci_initializer is None else Ci_initializer,
            "mg": self.mginull,
            "Buff1": self._ss_buffer_free(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff1_ca": self._ss_buffer_bound(self.Buffnull1, self.rf1, self.rf2, self.cainull),
            "Buff2": self._ss_buffer_free(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "Buff2_ca": self._ss_buffer_bound(self.Buffnull2, self.rf3, self.rf4, self.cainull),
            "BTC": self._ss_buffer_free(self.BTCnull, self.b1, self.b2, self.cainull),
            "BTC_ca": self._ss_buffer_bound(self.BTCnull, self.b1, self.b2, self.cainull),
            "DMNPE": self._ss_buffer_free(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "DMNPE_ca": self._ss_buffer_bound(self.DMNPEnull, self.c1, self.c2, self.cainull),
            "PV": self._ss_pv_free(),
            "PV_ca": self._ss_pv_ca(),
            "PV_mg": self._ss_pv_mg(),
            "CAM0": self.CAM_start,
            "CAM1C": 0.0 * u.mM,
            "CAM2C": 0.0 * u.mM,
            "CAM1N2C": 0.0 * u.mM,
            "CAM1N": 0.0 * u.mM,
            "CAM2N": 0.0 * u.mM,
            "CAM2N1C": 0.0 * u.mM,
            "CAM1C1N": 0.0 * u.mM,
            "CAM4": 0.0 * u.mM,
            "pump": self.TotalPump,
        }
        defaults.update(species_initializers)
        return {name: self._as_initializer(value) for name, value in defaults.items()}

    def _as_initializer(self, value):
        if callable(value):
            return value
        if isinstance(value, tuple):
            resolved = []
            for item in value:
                if hasattr(item, "value"):
                    resolved.append(item.value)
                else:
                    resolved.append(item)
            first = resolved[0]
            if hasattr(first, "unit"):
                unit = first.unit
                decimals = [u.Quantity(item).to_decimal(unit) for item in resolved]
                return u.Quantity(u.math.asarray(decimals), unit)
            return u.math.asarray(resolved)
        return braintools.init.Constant(value)

    def _ss_buffer_free(self, total, kon, koff, cai):
        return total / (1.0 + (kon / koff) * cai)

    def _ss_buffer_bound(self, total, kon, koff, cai):
        return total / (1.0 + koff / (kon * cai))

    def _kdc(self):
        return (self.cainull * self.m1) / self.m2

    def _kdm(self):
        return (self.mginull * self.p1) / self.p2

    def _ss_pv_free(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return self.PVnull / (1.0 + kdc + kdm)

    def _ss_pv_ca(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return (self.PVnull * kdc) / (1.0 + kdc + kdm)

    def _ss_pv_mg(self):
        kdc = self._kdc()
        kdm = self._kdm()
        return (self.PVnull * kdm) / (1.0 + kdc + kdm)

    def _require_diam_mid(self):
        if not hasattr(self, "diam_mid"):
            raise AttributeError(
                f"{type(self).__name__} requires 'diam_mid' before kinetic state initialization."
            )
        return self.diam_mid

    @property
    def vrat(self):
        dr2 = 0.25 / (self.Nannuli - 1.0)
        return u.math.pi * (0.5 - (dr2 / 2.0)) * 2.0 * dr2

    @property
    def parea(self):
        return u.math.pi * self._require_diam_mid()

    @property
    def dsq(self):
        diam_mid = self._require_diam_mid()
        return diam_mid * diam_mid

    @property
    def dsqvol(self):
        return self.dsq * self.vrat

    def _ci_source_flux(self, total_current):
        if total_current is None:
            return self.dsqvol * (0.0 * u.mM / u.ms)
        return -(total_current * u.math.pi * self._require_diam_mid()) / (2.0 * u.faraday_constant)

    def _ion_init_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_init_state_hook(self, V, batch_size=batch_size)

    def _ion_reset_state_hook(self, V, batch_size: int = None):
        self._require_diam_mid()
        KineticIon._ion_reset_state_hook(self, V, batch_size=batch_size)


@register_ion("CdpHVA_SU2015_DCN")
class CdpHVA_SU2015_DCN(Calcium, DynamicNernstIon):
    r"""Template-based import of ``CdpHVA_SU15_DCN.mod``.

    The imported NEURON mechanism evolves intracellular calcium via:

    .. math::

       cai' = -\frac{kCa}{depth} \cdot ica \cdot 10^4 - \frac{cai - caiBase}{tauCa}

    In the first comparison notebook we only exercise the zero-``ica`` path,
    which reduces to pure relaxation toward ``caiBase``.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kCa: Union[brainstate.typing.ArrayLike, Callable] = 3.45e-7 / u.coulomb,
        tauCa: Union[brainstate.typing.ArrayLike, Callable] = 70.0 * u.ms,
        caiBase: Union[brainstate.typing.ArrayLike, Callable] = 50e-6 * u.mM,
        depth: Union[brainstate.typing.ArrayLike, Callable] = 0.2 * u.um,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        if Ci_initializer is None:
            Ci_initializer = braintools.init.Constant(caiBase)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        self.kCa = braintools.init.param(kCa, self.varshape, allow_none=False)
        self.tauCa = braintools.init.param(tauCa, self.varshape, allow_none=False)
        self.caiBase = braintools.init.param(caiBase, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        if total_current is None:
            total_current = braintools.init.param(0.0 * (u.mA / u.cm ** 2), self.varshape)
        # The imported NMODL uses:
        #   cai' = -(kCa / depth) * ica * 1e4 - (cai - caiBase) / tauCa
        # where NEURON raw ``ica`` is negative for inward current. BrainCell
        # channels follow the repo-wide inward-positive current convention, so
        # the equivalent imported-ion drive here is positive in ``total_current``.
        drive_value = (
            self.kCa.to_decimal(self.kCa.unit)
            / self.depth.to_decimal(u.um)
            * total_current.to_decimal(u.mA / u.cm ** 2)
            * 1e4
        )
        drive = drive_value * (u.mM / u.ms)
        return drive - (Ci - self.caiBase) / self.tauCa


@register_ion("CdpLVA_SU2015_DCN")
class CdpLVA_SU2015_DCN(Calcium, DynamicNernstIon):
    r"""Template-based import of ``CdpLVA_SU15_DCN.mod``.

    The imported NEURON mechanism evolves intracellular calcium via:

    .. math::

       cali' = -\frac{kCal}{depth} \cdot ical \cdot 10^4 - \frac{cali - caliBase}{tauCal}

    In BrainCell this pool is still exposed through the standard calcium
    ``Ci/Co/E`` interface, with ``Ci`` corresponding to the NMODL ``cali``.
    """

    __module__ = "braincell.ion"
    uses_total_current = True

    def __init__(
        self,
        size: brainstate.typing.Size,
        temp: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.0),
        kCal: Union[brainstate.typing.ArrayLike, Callable] = 3.45e-7 / u.coulomb,
        tauCal: Union[brainstate.typing.ArrayLike, Callable] = 70.0 * u.ms,
        caliBase: Union[brainstate.typing.ArrayLike, Callable] = 50e-6 * u.mM,
        depth: Union[brainstate.typing.ArrayLike, Callable] = 0.2 * u.um,
        Co: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        Ci_initializer: Union[brainstate.typing.ArrayLike, Callable, None] = None,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        if Ci_initializer is None:
            Ci_initializer = braintools.init.Constant(caliBase)
        self._init_dynamic_nernst_ion(
            Co=Co,
            temp=temp,
            valence=None,
            Ci_initializer=Ci_initializer,
        )

        self.kCal = braintools.init.param(kCal, self.varshape, allow_none=False)
        self.tauCal = braintools.init.param(tauCal, self.varshape, allow_none=False)
        self.caliBase = braintools.init.param(caliBase, self.varshape, allow_none=False)
        self.depth = braintools.init.param(depth, self.varshape, allow_none=False)

    def derivative(self, Ci, V, total_current=None):
        _ = V
        if total_current is None:
            total_current = braintools.init.param(0.0 * (u.mA / u.cm ** 2), self.varshape)
        # The imported NMODL uses:
        #   cali' = -(kCal / depth) * ical * 1e4 - (cali - caliBase) / tauCal
        # where NEURON raw ``ical`` is negative for inward current. BrainCell
        # channel currents use the repo-wide inward-positive convention, so
        # the equivalent imported-ion drive here is positive in ``total_current``.
        drive_value = (
            self.kCal.to_decimal(self.kCal.unit)
            / self.depth.to_decimal(u.um)
            * total_current.to_decimal(u.mA / u.cm ** 2)
            * 1e4
        )
        drive = drive_value * (u.mM / u.ms)
        return drive - (Ci - self.caliBase) / self.tauCal
