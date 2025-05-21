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

# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainstate
import brainunit as u

from braincell._base import Ion, Channel, HHTypedNeuron
from braincell._protocol import DiffEqState

__all__ = [
    'Calcium',
    'CalciumFixed',
    'CalciumDetailed',
    'CalciumFirstOrder',
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


class CalciumFixed(Calcium):
    """Fixed Calcium dynamics.

    This calcium model has no dynamics. It holds fixed reversal
    potential :math:`E` and concentration :math:`C`.
    """
    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        E: Union[brainstate.typing.ArrayLike, Callable] = 120. * u.mV,
        C: Union[brainstate.typing.ArrayLike, Callable] = 2.4e-4 * u.mM,
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)
        self.E = brainstate.init.param(E, self.varshape, allow_none=False)
        self.C = brainstate.init.param(C, self.varshape, allow_none=False)

    def reset_state(self, V, batch_size=None):
        ca_info = self.pack_info()
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(type(self), *tuple(nodes))
        for node in nodes:
            node.reset_state(V, ca_info, batch_size=batch_size)


class _CalciumDynamics(Calcium):
    """Calcium ion flow with dynamics.

    Parameters
    ----------
    size: int, tuple of int
      The ion size.
    C0: brainstate.typing.ArrayLike, Callable
      The Calcium concentration outside of membrane.
    T: brainstate.typing.ArrayLike, Callable
      The temperature.
    C_initializer: brainstate.typing.ArrayLike, Callable
      The initializer for Calcium concentration.
    name: str
      The ion name.
    """

    def __init__(
        self,
        size: brainstate.typing.Size,
        C0: Union[brainstate.typing.ArrayLike, Callable] = 2. * u.mM,
        T: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        C_initializer: Union[brainstate.typing.ArrayLike, Callable] = brainstate.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, **channels)

        # parameters
        self.C0 = brainstate.init.param(C0, self.varshape, allow_none=False)
        self.T = brainstate.init.param(T, self.varshape, allow_none=False)  # temperature
        self._constant = u.gas_constant * self.T / (2 * u.faraday_constant)
        self._C_initializer = C_initializer

    def init_state(self, V, batch_size=None):
        # Calcium concentration
        self.C = DiffEqState(brainstate.init.param(self._C_initializer, self.varshape, batch_size))
        super().init_state(V, batch_size)

    def reset_state(self, V, batch_size=None):
        self.C.value = brainstate.init.param(self._C_initializer, self.varshape, batch_size)
        super().reset_state(V, batch_size)

    def derivative(self, C, V):
        raise NotImplementedError

    def compute_derivative(self, V):
        ca_info = self.pack_info()
        nodes = brainstate.graph.nodes(self, Channel, allowed_hierarchy=(1, 1)).values()
        self.check_hierarchies(type(self), *tuple(nodes))
        for node in nodes:
            node.compute_derivative(V, ca_info)
        self.C.derivative = self.derivative(self.C.value, V)

    @property
    def E(self):
        return self._reversal_potential(self.C.value)

    def _reversal_potential(self, C):
        # The Nernst relation:
        #
        # E_{\mathrm{Ca}}=\frac{RT}{2F}\log\frac{[\mathrm{Ca}]_{\mathrm{o}}}{[\mathrm{Ca}]_{\mathrm{i}}}
        return self._constant * u.math.log(self.C0 / C)


class CalciumDetailed(_CalciumDynamics):
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
    C0 : float
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

    def __init__(
        self,
        size: brainstate.typing.Size,
        T: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        d: Union[brainstate.typing.ArrayLike, Callable] = 1. * u.um,
        tau: Union[brainstate.typing.ArrayLike, Callable] = 5. * u.ms,
        C_rest: Union[brainstate.typing.ArrayLike, Callable] = 2.4e-4 * u.mM,
        C0: Union[brainstate.typing.ArrayLike, Callable] = 2. * u.mM,
        C_initializer: Union[brainstate.typing.ArrayLike, Callable] = brainstate.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(size, name=name, T=T, C0=C0, C_initializer=C_initializer, **channels)

        # parameters
        self.d = brainstate.init.param(d, self.varshape, allow_none=False)
        self.tau = brainstate.init.param(tau, self.varshape, allow_none=False)
        self.C_rest = brainstate.init.param(C_rest, self.varshape, allow_none=False)

    def derivative(self, C, V):
        ICa = self.current(V, include_external=True)
        drive = ICa / (2 * u.faraday_constant * self.d)
        drive = u.math.maximum(drive, u.math.zeros_like(drive))
        return drive + (self.C_rest - C) / self.tau


class CalciumFirstOrder(_CalciumDynamics):
    r"""
    The first-order calcium concentration model.

    .. math::

       Ca' = -\alpha I_{Ca} + -\beta Ca

    """
    __module__ = 'braincell.ion'

    def __init__(
        self,
        size: brainstate.typing.Size,
        T: Union[brainstate.typing.ArrayLike, Callable] = u.celsius2kelvin(36.),
        alpha: Union[brainstate.typing.ArrayLike, Callable] = 0.13,
        beta: Union[brainstate.typing.ArrayLike, Callable] = 0.075,
        C0: Union[brainstate.typing.ArrayLike, Callable] = 2. * u.mM,
        C_initializer: Union[brainstate.typing.ArrayLike, Callable] = brainstate.init.Constant(2.4e-4 * u.mM),
        name: Optional[str] = None,
        **channels
    ):
        super().__init__(
            size,
            name=name,
            T=T,
            C0=C0,
            C_initializer=C_initializer,
            **channels
        )

        # parameters
        self.alpha = brainstate.init.param(alpha, self.varshape, allow_none=False)
        self.beta = brainstate.init.param(beta, self.varshape, allow_none=False)

    def derivative(self, C, V):
        ICa = self.current(V, include_external=True)
        drive = u.math.maximum(self.alpha * ICa, 0. * u.mM)
        return drive - self.beta * C
