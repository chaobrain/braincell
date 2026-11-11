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

"""GABA_A synapse whose reversal potential follows intracellular chloride."""

import brainunit as u
import numpy as np

from braincell._base_channel import Synapse
from braincell._base_neuron import HHTypedNeuron
from braincell._typing import Initializer, Size
from braincell.ion._base import _nernst
from braincell.ion.chloride import Chloride
from braincell.mech import ScalarEventInput, StateSpec, positive, register_synapse

__all__ = ['GabaA_JE2011']


@register_synapse("GabaA_JE2011")
class GabaA_JE2011(Synapse):
    r"""GABA_A synapse of Jedlicka et al. (2011), ``gabaA_Cl.mod``.

    A difference-of-exponentials conductance, as in
    :class:`~braincell.synapse.Exp2Syn`, shared by chloride and
    bicarbonate [1]_:

    - states ``A`` and ``B`` in ``uS``, ``A' = -A / tau1``, ``B' = -B / tau2``;
      an event of weight ``w`` adds ``factor * w`` to both, ``g = B - A``
    - :math:`I_{Cl} = (1 - P)\,g\,(E_{Cl} - V)` and
      :math:`I_{HCO_3} = P\,g\,(E_{HCO_3} - V)`, inward positive
    - :math:`E_{HCO_3} = \frac{RT}{F}\ln(\mathrm{HCO3i}/\mathrm{HCO3e})`
      (valence -1), -12.892 mV with the defaults

    ``ion_type`` is :class:`~braincell.ion.Chloride`: in a
    :class:`~braincell.Cell`, ``current(V_post, ion_info)`` receives the
    chloride :class:`~braincell.IonInfo` of the CV that owns each
    placement and returns :math:`I_{Cl} + I_{HCO_3}` (``nA``);
    :meth:`ion_current` returns :math:`I_{Cl}`, which the Cell adds to the
    chloride ion's current, divided by that CV's area. A dynamic pool
    such as :class:`~braincell.ion.Cldif_JE2011` then accumulates the
    chloride influx, and :math:`E_{GABA} = P E_{HCO_3} + (1 - P) E_{Cl}`
    rises with it. The Cell needs exactly one chloride ion and, when that
    ion's concentration is dynamic, the staggered solver with
    ``cache_ion_total_current=True``.

    Parameters
    ----------
    size : brainstate.typing.Size
        Packed synapse shape.
    name : str or None, optional
        Runtime node name.
    tau1 : brainstate.typing.ArrayLike or callable, optional
        Positive rise time constant, default 0.5 ms (the published hoc; the
        ``.mod`` default is 0.1 ms).
    tau2 : brainstate.typing.ArrayLike or callable, optional
        Decay time constant greater than ``tau1``, default 6 ms (the hoc;
        the ``.mod`` default is 10 ms).
    P : brainstate.typing.ArrayLike or callable, optional
        Bicarbonate share of the conductance, in ``[0, 1]``, default 0.18.
        ``P = 0`` gives an ``Exp2Syn`` whose reversal potential is
        :math:`E_{Cl}`.
    HCO3i, HCO3e : brainstate.typing.ArrayLike or callable, optional
        Intra- and extracellular bicarbonate, default 16 and 26 mM.
        Bicarbonate is not a BrainCell ion; as in the ``.mod`` it is a
        synapse parameter.
    temp : brainstate.typing.ArrayLike or callable, optional
        Absolute temperature of :math:`E_{HCO_3}`, default 35 degrees Celsius
        (the published hoc).

    Raises
    ------
    ValueError
        If ``tau1 >= tau2``, a time constant or concentration is not
        positive, or ``P`` lies outside ``[0, 1]``.

    Notes
    -----
    Chloride does not diffuse between CVs in BrainCell (see
    :class:`~braincell.ion.Cldif_JE2011`), so the accumulation near a
    synapse on a multi-CV dendrite is larger than in the published model.

    References
    ----------
    .. [1] Jedlicka, P., Deller, T., Gutkin, B. S., & Backus, K. H. (2011).
           Activity-dependent intracellular chloride accumulation and
           diffusion controls GABA(A) receptor-mediated synaptic
           transmission. Hippocampus, 21(8), 885-898.
           doi:10.1002/hipo.20804

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell
        >>> from braincell.synapse import GabaA_JE2011
        >>> syn = GabaA_JE2011(size=1)
        >>> syn.init_state()
        >>> syn.B.value = 0.001 * u.uS
        >>> cl = braincell.IonInfo(Ci=5.0 * u.mM, Co=133.5 * u.mM, E=-87.0 * u.mV, valence=-1)
        >>> syn.ion_current(-70.0 * u.mV, cl).to_decimal(u.nA)
        Array([-0.01394], dtype=float32, weak_type=True)
    """

    root_type = HHTypedNeuron
    ion_type = Chloride
    states = {"A": StateSpec(0.0 * u.uS), "B": StateSpec(0.0 * u.uS)}
    event_input = ScalarEventInput(u.uS, aggregation="sum")

    def __init__(
        self,
        size: Size,
        name: str | None = None,
        tau1: Initializer = 0.5 * u.ms,
        tau2: Initializer = 6.0 * u.ms,
        P: Initializer = 0.18,
        HCO3i: Initializer = 16.0 * u.mM,
        HCO3e: Initializer = 26.0 * u.mM,
        temp: Initializer = u.celsius2kelvin(35.0),
    ):
        super().__init__(size=size, name=name)
        self._init_parameters(tau1=tau1, tau2=tau2, P=P, HCO3i=HCO3i, HCO3e=HCO3e, temp=temp)

    @classmethod
    def validate_parameter_values(cls, parameters) -> None:
        """Require ``tau1 < tau2``, positive concentrations and ``0 <= P <= 1``, in numpy (see ``Exp2Syn``)."""
        for field in ("tau1", "tau2", "HCO3i", "HCO3e", "temp"):
            positive(parameters[field], field)
        tau1 = np.asarray(u.Quantity(parameters["tau1"]).to_decimal(u.ms))
        tau2 = np.asarray(u.Quantity(parameters["tau2"]).to_decimal(u.ms))
        if np.any(tau1 >= tau2):
            raise ValueError("GabaA_JE2011 requires tau1 < tau2.")
        P = np.asarray(u.get_mantissa(parameters["P"]))
        if np.any((P < 0.0) | (P > 1.0)):
            raise ValueError("GabaA_JE2011 requires 0 <= P <= 1.")

    @property
    def g(self):
        return self.B.value - self.A.value

    @property
    def E_HCO3(self):
        """Bicarbonate reversal potential."""
        return _nernst(Ci=self.HCO3i, Co=self.HCO3e, temp=self.temp, valence=-1)

    def apply_events(self, payload, V_post=None):
        _ = V_post
        self.event_input.validate_payload(payload)
        tp = (self.tau1 * self.tau2) / (self.tau2 - self.tau1) * u.math.log(self.tau2 / self.tau1)
        factor = -u.math.exp(-(tp / self.tau1)) + u.math.exp(-(tp / self.tau2))
        delta = payload * (1.0 / factor)
        self.A.value = self.A.value + delta
        self.B.value = self.B.value + delta

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.A.derivative = -self.A.value / self.tau1
        self.B.derivative = -self.B.value / self.tau2

    def ion_current(self, V_post, ion_info):
        return (1.0 - self.P) * self.g * (ion_info.E - V_post)

    def current(self, V_post, ion_info):
        return self.ion_current(V_post, ion_info) + self.P * self.g * (self.E_HCO3 - V_post)
