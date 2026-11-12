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

"""Synapses with short-term plasticity: a release model inside the synapse (CELL-12b).

A release model needs the number of events, not only their summed weight,
so these synapses declare ``ScalarEventInput(unit, aggregation="sum_count")``
and receive both on every boundary.
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base_channel import Synapse
from braincell._base_neuron import HHTypedNeuron
from braincell._typing import Initializer, Size
from braincell.mech import ScalarEventInput, StateSpec, positive, register_synapse

__all__ = ['ExpSynSTD']


@register_synapse("ExpSynSTD")
class ExpSynSTD(Synapse):
    r"""Exponential conductance synapse with short-term depression.

    An :class:`~braincell.synapse.ExpSyn` whose events draw on a depleting
    resource, the Tsodyks–Markram model [1]_ in its "release, then
    consume" order:

    - states ``g`` in ``uS`` and the dimensionless resource ``x``, which
      starts at 1
    - ``g' = -g / tau``, ``x' = (1 - x) / tau_rec``
    - one event of weight ``w``: ``g <- g + w U x``, then ``x <- x (1 - U)``
    - inward-positive point current ``I = g (e - V_post)``

    The event input is ``ScalarEventInput(u.uS, aggregation="sum_count")``:
    on each boundary a row receives the summed weight ``W`` of its events
    and their number ``K``. ``K`` events release the resource as ``K``
    single events in a row would, without recovery between them:

    .. math::

        \Delta g = \frac{W}{K}\, x\, \bigl(1 - (1 - U)^K\bigr), \qquad
        x \leftarrow x\, (1 - U)^K .

    So two events on one boundary give what two events in quick succession
    give: with ``w = 1 nS`` and ``U = 0.5`` from ``x = 1``, 0.5 nS and
    then 0.25 nS, 0.75 nS in all, and ``x = 0.25``. The events of one
    boundary have no order, so events of different weights share the
    release in proportion to their mean ``W / K``. With a plasticity rule
    on the connection, ``W`` carries each row's current weight, so an
    event adds ``w`` times the released fraction.

    Each placed synapse has its own resource; the rows connected to one
    synapse share it.

    Parameters
    ----------
    size : brainstate.typing.Size
        Packed synapse shape.
    name : str or None, optional
        Runtime node name.
    tau : brainstate.typing.ArrayLike or callable, optional
        Positive decay time constant of ``g``, default 0.1 ms (as
        ``ExpSyn``).
    e : brainstate.typing.ArrayLike or callable, optional
        Reversal voltage, default 0 mV.
    U : brainstate.typing.ArrayLike or callable, optional
        Fraction of the resource one event releases, in ``(0, 1]``,
        default 0.5. Dimensionless.
    tau_rec : brainstate.typing.ArrayLike or callable, optional
        Positive recovery time constant of ``x``, default 800 ms.

    Raises
    ------
    ValueError
        If ``tau`` or ``tau_rec`` is not positive, or ``U`` lies outside
        ``(0, 1]``.

    References
    ----------
    .. [1] Tsodyks, M. V., & Markram, H. (1997). The neural code between
           neocortical pyramidal neurons depends on neurotransmitter
           release probability. Proceedings of the National Academy of
           Sciences, 94(2), 719-723. doi:10.1073/pnas.94.2.719

    Examples
    --------
    Two 1 nS events on one boundary:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell.synapse import ExpSynSTD
        >>> syn = ExpSynSTD(size=1, U=0.5)
        >>> syn.init_state()
        >>> syn.apply_events(np.array([0.002]) * u.uS, count=np.array([2.0]))
        >>> round(float(syn.g.value.to_decimal(u.uS)[0]), 8), round(float(syn.x.value[0]), 8)
        (0.00075, 0.25)

    In a Cell, place it like any synapse and connect sources to it. Two
    events 1 ms apart leave about a quarter of the resource:

    .. code-block:: python

        >>> import braincell
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> _ = cell.place(braincell.mech.Synapse("ExpSynSTD", name="std", tau=2.0 * u.ms, U=0.5))
        >>> _ = braincell.connect(
        ...     "drive",
        ...     source=braincell.EventSequence.from_times(([1.0, 2.0] * u.ms,)),
        ...     synapse=cell.synapses["std"],
        ...     weight=0.001 * u.uS,
        ... )
        >>> _ = cell.record("x", braincell.observe.synapse(name="std").state("x"))
        >>> result = cell.run(dt=0.1 * u.ms, duration=3.0 * u.ms)
        >>> round(float(result.samples["x"].values[-1, 0]), 3)
        0.251
    """

    root_type = HHTypedNeuron
    states = {"g": StateSpec(0.0 * u.uS), "x": StateSpec(1.0)}
    event_input = ScalarEventInput(u.uS, aggregation="sum_count")

    def __init__(
        self,
        size: Size,
        name: str | None = None,
        tau: Initializer = 0.1 * u.ms,
        e: Initializer = 0.0 * u.mV,
        U: Initializer = 0.5,
        tau_rec: Initializer = 800.0 * u.ms,
    ):
        super().__init__(size=size, name=name)
        self._init_parameters(tau=tau, e=e, U=U, tau_rec=tau_rec)

    @classmethod
    def validate_parameter_values(cls, parameters) -> None:
        """Require positive ``tau`` and ``tau_rec`` and ``0 < U <= 1``, in numpy (see ``Exp2Syn``)."""
        positive(parameters["tau"], "tau")
        positive(parameters["tau_rec"], "tau_rec")
        U = np.asarray(u.get_mantissa(parameters["U"]))
        if np.any((U <= 0.0) | (U > 1.0)):
            raise ValueError("ExpSynSTD requires 0 < U <= 1.")

    def apply_events(self, payload, V_post=None, *, count):
        """Release the resource for the ``count`` events of summed weight ``payload`` on each row."""
        _ = V_post
        self.event_input.validate_payload(payload)
        x = self.x.value
        count = jnp.asarray(count, dtype=jnp.result_type(x))
        has_events = count > 0
        # (1 - U)^K; the exponent stays positive on rows without events, so
        # the derivative in U is finite there even at U = 1 (0^0).
        kept = jnp.where(has_events, (1.0 - self.U) ** jnp.where(has_events, count, 1.0), 1.0)
        self.g.value = self.g.value + payload / jnp.maximum(count, 1.0) * (x * (1.0 - kept))
        self.x.value = x * kept

    def compute_derivative(self, V_post=None):
        _ = V_post
        self.g.derivative = -self.g.value / self.tau
        self.x.derivative = (1.0 - self.x.value) / self.tau_rec

    def current(self, V_post):
        return self.g.value * (self.e - V_post)
