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

"""Synapses with short-term plasticity: a release model inside the synapse.

``ExpSynSTD`` depresses (CELL-12b); ``ExpSynSTP`` also facilitates
(CELL-12e). A release model needs the number of events, not only their
summed weight, so these synapses declare
``ScalarEventInput(unit, aggregation="sum_count")`` and receive both on
every boundary.
"""

import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._base_channel import Synapse
from braincell._base_neuron import HHTypedNeuron
from braincell._typing import Initializer, Size
from braincell.mech import ScalarEventInput, StateSpec, positive, register_synapse

__all__ = ['ExpSynSTD', 'ExpSynSTP']


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


@register_synapse("ExpSynSTP")
class ExpSynSTP(Synapse):
    r"""Exponential conductance synapse with short-term facilitation and depression.

    An :class:`ExpSynSTD` whose release fraction ``u`` grows with each event
    and decays between events, the Tsodyks–Markram model in the
    continuous form of [1]_. Facilitation needs ``tau_fac > 0``: with the
    default ``tau_fac = 0`` every event releases the fraction ``U`` and the
    model is :class:`ExpSynSTD` with the same ``U`` and ``tau_rec``.

    - states ``g`` in ``uS``, the dimensionless release fraction ``u``,
      which starts at its rest value 0, and the dimensionless resource
      ``x``, which starts at 1
    - ``g' = -g / tau``, ``u' = -u / tau_fac``, ``x' = (1 - x) / tau_rec``
    - one event of weight ``w``: first ``u <- u + U (1 - u)``, then
      ``g <- g + w u x`` and ``x <- x (1 - u)``
    - inward-positive point current ``I = g (e - V_post)``

    An event after a long rest releases ``U x``, as in ``ExpSynSTD``; events
    in quick succession raise ``u`` towards 1, so with a small ``U`` and a
    long ``tau_fac`` the increments grow before depletion of ``x`` takes
    over.

    The event input is ``ScalarEventInput(u.uS, aggregation="sum_count")``:
    on each boundary a row receives the summed weight ``W`` of its events
    and their number ``K``. Starting from ``u_0`` and ``x_0``, the ``j``-th of
    ``K`` events in a row uses ``u_j = 1 - (1 - u_0)(1 - U)^j``, so

    .. math::

        x_K = x_0 (1 - u_0)^K (1 - U)^{K(K+1)/2}, \qquad
        \Delta g = \frac{W}{K} (x_0 - x_K), \qquad
        u \leftarrow 1 - (1 - u_0)(1 - U)^K .

    With ``U = 0.25`` from rest, two events release 0.25 and then 0.328125
    of the resource, 0.578125 in all, whether they arrive on one boundary
    or in quick succession. The events of one boundary have no order, so
    events of different weights share the release in proportion to their
    mean ``W / K``. With a plasticity rule on the connection, ``W`` carries
    each row's current weight.

    With ``tau_fac = 0`` there is no facilitation: every event uses ``U``,
    ``K`` events keep ``x (1 - U)^K``, and ``u`` is not used and is held at
    0. This is not the limit ``tau_fac -> 0`` for events on one boundary:
    with any ``tau_fac > 0`` they have no time to decay and facilitate each
    other. ``U = 1`` also gives the ``ExpSynSTD`` increments, for any
    ``tau_fac``, since each boundary's first event releases all of ``x``.

    The increments are those of NEST's ``tsodyks2_synapse``, which follows
    Eqs. 4-5 of [2]_ event by event with the same parameters ``U``,
    ``tau_fac`` and ``tau_rec``: NEST's ``u`` at an event is the ``u`` just
    after the jump here, ``U + (1 - U) u``. The tests compare each event's
    increment with efficacies recorded from NEST 3.10.0
    (``braincell/synapse/short_term_test.py``). BrainPy's ``STP`` has the
    same equations but starts at ``u = U``, so its first event releases
    more than ``U x``.

    Each placed synapse has its own ``u`` and ``x``; the rows connected to
    one synapse share them. ``u`` and ``x`` relax linearly and the Cell's
    solver integrates them: after 50 ms of recovery with ``dt`` 0.1 ms and
    ``tau_fac`` 100 ms the relative error of ``u`` is within rounding for
    ``staggered``, ``exp_euler`` and ``ind_exp_euler``, 5.7e-15 for
    ``rk4``, 4.2e-8 for ``crank_nicolson``, 8.3e-8 for ``ralston2`` (the
    default of :meth:`~braincell.Cell.point`) and 2.5e-4 for ``euler`` and
    ``backward_euler``.

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
        Increment of ``u`` per event, and the fraction an event releases
        after a long rest, in ``(0, 1]``, default 0.5. Dimensionless.
    tau_fac : brainstate.typing.ArrayLike or callable, optional
        Non-negative decay time constant of ``u``, default 0 ms, which turns
        facilitation off. Set it above 0 for facilitation.
    tau_rec : brainstate.typing.ArrayLike or callable, optional
        Positive recovery time constant of ``x``, default 800 ms.

    Raises
    ------
    ValueError
        If ``tau`` or ``tau_rec`` is not positive, ``tau_fac`` is negative,
        or ``U`` lies outside ``(0, 1]``.

    See Also
    --------
    ExpSynSTD : Depression only.

    References
    ----------
    .. [1] Tsodyks, M., Pawelzik, K., & Markram, H. (1998). Neural networks
           with dynamic synapses. Neural Computation, 10(4), 821-835.
           doi:10.1162/089976698300017502
    .. [2] Maass, W., & Markram, H. (2002). Synapses as dynamic memory
           buffers. Neural Networks, 15(2), 155-161.
           doi:10.1016/S0893-6080(01)00144-7

    Examples
    --------
    Two 1 nS events on one boundary, from rest:

    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell.synapse import ExpSynSTP
        >>> syn = ExpSynSTP(size=1, U=0.25, tau_fac=100.0 * u.ms)
        >>> syn.init_state()
        >>> syn.apply_events(np.array([0.002]) * u.uS, count=np.array([2.0]))
        >>> round(float(syn.g.value.to_decimal(u.uS)[0]), 9), float(syn.u.value[0]), float(syn.x.value[0])
        (0.000578125, 0.4375, 0.421875)

    In a Cell, with the facilitating parameters of NEST's
    ``tsodyks2_synapse`` example, an event 20 ms after the first adds about
    1.73 times as much:

    .. code-block:: python

        >>> import braincell
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> _ = cell.place(braincell.mech.Synapse(
        ...     "ExpSynSTP", name="stp", tau=2.0 * u.ms, U=0.1, tau_fac=1000.0 * u.ms, tau_rec=100.0 * u.ms
        ... ))
        >>> _ = braincell.connect(
        ...     "drive",
        ...     source=braincell.EventSequence.from_times(([5.0, 25.0] * u.ms,)),
        ...     synapse=cell.synapses["stp"],
        ...     weight=0.001 * u.uS,
        ... )
        >>> _ = cell.record("g", braincell.observe.synapse(name="stp").state("g"))
        >>> g = cell.run(dt=0.1 * u.ms, duration=30.0 * u.ms).samples["g"].values[:, 0]
        >>> round(float(g[251] / g[51]), 2)
        1.73
    """

    root_type = HHTypedNeuron
    states = {"g": StateSpec(0.0 * u.uS), "u": StateSpec(0.0), "x": StateSpec(1.0)}
    event_input = ScalarEventInput(u.uS, aggregation="sum_count")

    def __init__(
        self,
        size: Size,
        name: str | None = None,
        tau: Initializer = 0.1 * u.ms,
        e: Initializer = 0.0 * u.mV,
        U: Initializer = 0.5,
        tau_fac: Initializer = 0.0 * u.ms,
        tau_rec: Initializer = 800.0 * u.ms,
    ):
        super().__init__(size=size, name=name)
        self._init_parameters(tau=tau, e=e, U=U, tau_fac=tau_fac, tau_rec=tau_rec)

    @classmethod
    def validate_parameter_values(cls, parameters) -> None:
        """Require positive ``tau`` and ``tau_rec``, ``tau_fac >= 0`` and ``0 < U <= 1``, in numpy."""
        positive(parameters["tau"], "tau")
        positive(parameters["tau_rec"], "tau_rec")
        if np.any(np.asarray(u.get_mantissa(parameters["tau_fac"])) < 0.0):
            raise ValueError("Mechanism parameter 'tau_fac' must be >= 0.")
        U = np.asarray(u.get_mantissa(parameters["U"]))
        if np.any((U <= 0.0) | (U > 1.0)):
            raise ValueError("ExpSynSTP requires 0 < U <= 1.")

    def _facilitates(self):
        """Rows with ``tau_fac > 0``."""
        return jnp.asarray(u.get_mantissa(self.tau_fac)) > 0.0

    def apply_events(self, payload, V_post=None, *, count):
        """Release the resource for the ``count`` events of summed weight ``payload`` on each row."""
        _ = V_post
        self.event_input.validate_payload(payload)
        u0 = self.u.value
        x0 = self.x.value
        count = jnp.asarray(count, dtype=jnp.result_type(x0))
        has_events = count > 0
        facilitates = self._facilitates()
        # The exponents stay positive on rows without events, so the
        # derivatives in U and u are finite there even at U = 1 (0^0).
        k = jnp.where(has_events, count, 1.0)
        exponent = jnp.where(facilitates, k * (k + 1.0) / 2.0, k)
        kept = jnp.where(facilitates, (1.0 - u0) ** k, 1.0) * (1.0 - self.U) ** exponent
        kept = jnp.where(has_events, kept, 1.0)
        self.g.value = self.g.value + payload / jnp.maximum(count, 1.0) * (x0 * (1.0 - kept))
        self.x.value = x0 * kept
        u_after = jnp.where(has_events, 1.0 - (1.0 - u0) * (1.0 - self.U) ** k, u0)
        # Without facilitation u is not used and is held at its rest value 0.
        self.u.value = jnp.where(facilitates, u_after, 0.0)

    def compute_derivative(self, V_post=None):
        _ = V_post
        facilitates = self._facilitates()
        unit = u.get_unit(self.tau_fac)
        # A placeholder time constant on rows without facilitation keeps the
        # unselected branch (and its derivative in tau_fac) finite.
        tau_fac = u.math.where(facilitates, self.tau_fac, 1.0 * unit)
        self.g.derivative = -self.g.value / self.tau
        self.u.derivative = u.math.where(facilitates, -self.u.value / tau_fac, 0.0 / unit)
        self.x.derivative = (1.0 - self.x.value) / self.tau_rec

    def current(self, V_post):
        return self.g.value * (self.e - V_post)
