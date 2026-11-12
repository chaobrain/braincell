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

"""Runtime base class of a Connection plasticity rule (CELL-12a).

A rule is declared with :class:`braincell.mech.ConnectionRule` on one
``connect`` call and changes the weight of that call's rows while the model
runs. The runtime class registered under the declaration's name subclasses
:class:`ConnectionRule` here; ``braincell.mech`` may not import this module
(it is a leaf), so the module reports the base to the registry, which
checks a class passed to a declaration against it.
"""

from typing import ClassVar

import brainunit as u
import numpy as np

from braincell.mech._registry import _CATEGORY_CONNECTION_RULE, _register_runtime_base

__all__ = ["ConnectionRule"]


class ConnectionRule:
    """Runtime base class of a Connection plasticity rule.

    A rule owns the dynamic weight of every row of one ``connect`` call and
    any per-row state it needs (traces, for example). It reads signals and
    returns its next state; it changes nothing else. The Cell that owns the
    rows keeps the state, advances it once per event boundary ``t_n = n dt``
    before the synapses consume that boundary's events, and delivers each
    arrival with the weight before that boundary's update.

    Register a subclass with :func:`braincell.mech.register_connection_rule`
    and declare it with :class:`braincell.mech.ConnectionRule`. ``connect``
    builds one instance per call, as ``cls(size, weight_unit, **params)``,
    so a constructor checks its parameters there.

    Parameters
    ----------
    size : int
        Number of rows of the ``connect`` call.
    weight_unit : brainunit.Unit
        Unit of the rows' weight: the event-input unit of the target
        synapse.

    Attributes
    ----------
    signals : tuple of str
        Signals the rule reads besides ``pre``, the arrivals of each row's
        own source (after its delay), which every rule receives and which
        cannot be bound to anything else. ``"post"`` is the events of the
        target Cell member that owns each row's synapse: by default the
        Cell's ``event_outputs["spike"]``, or an event source bound with
        ``ConnectionRule(..., signals={"post": source})`` whose endpoints
        are the target's population members.

    Notes
    -----
    A subclass implements :meth:`update`, and :meth:`init_state` if it keeps
    state besides the weight ``"w"``. Every state value has one entry per
    row; ``"w"`` is a quantity in ``weight_unit``. Parameters are constant
    once the rule is built.

    Examples
    --------
    A rule that adds a fixed amount to a row's weight at each of its arrivals:

    .. code-block:: python

        >>> import brainunit as u
        >>> import braincell
        >>> @braincell.mech.register_connection_rule("PerEventStep")
        ... class PerEventStep(braincell.ConnectionRule):
        ...     def __init__(self, size, weight_unit, *, step):
        ...         super().__init__(size, weight_unit)
        ...         self.step = u.Quantity(step).in_unit(weight_unit)
        ...     def update(self, state, *, pre, dt):
        ...         return {"w": state["w"] + self.step * pre}
        >>> cell = braincell.Cell.point(1, V_init=-65.0 * u.mV)
        >>> cell = cell.place(braincell.mech.Synapse("ExpSyn", name="s"))
        >>> source = braincell.EventSequence.from_times(([0.2, 0.5] * u.ms,))
        >>> view = braincell.connect(
        ...     "c",
        ...     source=source,
        ...     synapse=cell.synapses["s"],
        ...     weight=0.01 * u.uS,
        ...     plasticity=braincell.mech.ConnectionRule("PerEventStep", step=0.001 * u.uS),
        ... )
        >>> _ = cell.run(dt=0.1 * u.ms, duration=1.0 * u.ms)
        >>> print(cell.connections["c"].rule_state("w"))
        [0.012] uS
        >>> braincell.mech.get_registry().unregister("connection_rule", "PerEventStep")
    """

    __module__ = "braincell"

    signals: ClassVar[tuple[str, ...]] = ()

    def __init__(self, size: int, weight_unit) -> None:
        self.size = int(size)
        self.weight_unit = weight_unit

    def init_state(self, weight) -> dict:
        """Return the state at ``init_state`` and ``reset_state``.

        Parameters
        ----------
        weight : brainunit.Quantity
            The rows' initial weight, shape ``(size,)``, in ``weight_unit``.

        Returns
        -------
        dict
            ``{name: value}``, one entry per row; ``"w"`` is the dynamic
            weight. The base class returns ``{"w": weight}``.
        """
        return {"w": weight}

    def update(self, state: dict, *, pre, dt, **signals) -> dict:
        """Advance the state over one event boundary.

        Parameters
        ----------
        state : dict
            The state just before the boundary, as :meth:`init_state`
            returned it.
        pre : jax.Array
            Arrivals of each row at the boundary, shape ``(size,)``; the
            same events the target synapse receives there.
        dt : brainunit.Quantity
            The simulation step, the time since the previous boundary.
        **signals
            One ``(size,)`` array per name in :attr:`signals`, read at the
            same boundary: ``post`` counts the events of each row's member.

        Returns
        -------
        dict
            The state after the boundary, with the same names. The rule
            changes nothing else.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement update(state, *, pre, dt, ...).")

    def check_weight(self, weight) -> None:
        """Reject an initial weight this rule cannot start from.

        ``connect`` and ``ConnectionView.set(weight=...)`` call it before
        storing ``weight``. The base class accepts every weight.

        Parameters
        ----------
        weight : brainunit.Quantity
            The rows' weight, shape ``(size,)``.

        Raises
        ------
        ValueError
            If the rule cannot start from ``weight``.
        """
        _ = weight


_register_runtime_base(_CATEGORY_CONNECTION_RULE, ConnectionRule)


def row_parameter(value, *, unit, size: int, name: str, rule: str) -> u.Quantity:
    """Return a rule parameter as a quantity in ``unit``, scalar or one value per row.

    Raises
    ------
    TypeError
        If ``value`` is a bare number while ``unit`` has a dimension.
    ValueError
        If its unit is incompatible with ``unit``, its shape is neither
        scalar nor ``(size,)``, or a value is not finite.
    """
    if not isinstance(value, u.Quantity):
        if not u.get_unit(unit).is_unitless:
            raise TypeError(
                f"{rule} parameter {name!r} requires a quantity compatible with {unit}; got {value!r} (no unit)."
            )
        value = u.Quantity(value)
    try:
        decimal = np.asarray(value.to_decimal(unit), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - brainunit raises its own mismatch type
        raise ValueError(
            f"{rule} parameter {name!r} requires a quantity compatible with {unit}; got {value!r}."
        ) from exc
    if decimal.shape not in ((), (size,)):
        raise ValueError(
            f"{rule} parameter {name!r} must be a scalar or have one value per row, shape ({size},); "
            f"got shape {decimal.shape}."
        )
    if not np.all(np.isfinite(decimal)):
        raise ValueError(f"{rule} parameter {name!r} must be finite; got {value!r}.")
    return u.Quantity(decimal, unit)
