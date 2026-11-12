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

"""Connection plasticity rule declaration (CELL-12a).

:class:`ConnectionRule` names a rule and its parameters, like
:class:`~braincell.mech.Synapse` names a synapse model. It holds no state:
``connect(..., plasticity=rule)`` resolves the name, checks the parameters
against the rows' weight, and builds the runtime rule for that call, so
one declaration can serve several calls, each with its own state.
"""

from types import MappingProxyType
from typing import Any, Mapping

from braincell.mech._density import _resolve_class_name
from braincell.mech._params import Params
from braincell.mech._registry import _CATEGORY_CONNECTION_RULE

__all__ = ["ConnectionRule"]


class ConnectionRule:
    """Connection plasticity rule declaration, by registry name or class.

    Pass it to ``braincell.connect(..., plasticity=...)`` or
    ``Network.connect(..., plasticity=...)``. The rule then changes the
    weight of every row of that call while the model runs; read the weight
    with ``ConnectionView.rule_state("w")``.

    Parameters
    ----------
    rule_type : str or type
        Registry name of the runtime rule class (such as ``"PairSTDP"``,
        :class:`braincell.plasticity.PairSTDP`), or the class itself, a
        subclass of :class:`braincell.ConnectionRule`. A name is resolved at
        ``connect``; an unregistered class is kept by identity.
    signals : Mapping[str, EventSource or EventSourceView], optional
        Bindings for the signals the rule declares in
        :attr:`braincell.ConnectionRule.signals`. ``"post"`` takes an event
        source with one endpoint per member of the target Cell: a scheduled
        source (``EventSequence``, ``NetStim``) as a teaching signal, or a
        live source of the target Cell itself. Omitted signals take their
        defaults (``"post"``: the target Cell's ``event_outputs["spike"]``).
        ``connect`` raises ``TypeError`` for a name the rule does not declare.
    **params
        Rule parameters, scalar or one value per row; ``connect`` checks
        them.

    Raises
    ------
    TypeError
        If ``rule_type`` is neither a string nor a ``ConnectionRule``
        subclass, or ``signals`` is not a mapping with string keys.
    ValueError
        If ``rule_type`` is an empty string.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import ConnectionRule
        >>> rule = ConnectionRule(
        ...     "PairSTDP",
        ...     A_plus=0.01 * u.uS,
        ...     A_minus=0.0105 * u.uS,
        ...     tau_plus=20.0 * u.ms,
        ...     tau_minus=20.0 * u.ms,
        ... )
        >>> rule.rule_type
        'PairSTDP'
        >>> sorted(rule.params)
        ['A_minus', 'A_plus', 'tau_minus', 'tau_plus']
    """

    __slots__ = ("rule_type", "params", "signals")

    def __init__(
        self,
        rule_type: str | type,
        /,
        *,
        signals: Mapping[str, Any] | None = None,
        **params: Any,
    ) -> None:
        rule_type = _resolve_class_name(_CATEGORY_CONNECTION_RULE, rule_type)
        if signals is None:
            signals = {}
        if not isinstance(signals, Mapping):
            raise TypeError(f"ConnectionRule signals must be a mapping of names to event sources, got {signals!r}.")
        if not all(isinstance(name, str) and name for name in signals):
            raise TypeError(f"ConnectionRule signal names must be non-empty strings, got {tuple(signals)!r}.")
        object.__setattr__(self, "rule_type", rule_type)
        object.__setattr__(self, "params", Params(params) if params else Params())
        object.__setattr__(self, "signals", MappingProxyType(dict(signals)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot set attribute {name!r}.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; cannot delete attribute {name!r}.")

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        # Signal sources compare by identity: a source is a model object.
        return (
            self.rule_type == other.rule_type
            and self.params == other.params
            and set(self.signals) == set(other.signals)
            and all(self.signals[name] is other.signals[name] for name in self.signals)
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.rule_type, self.params, tuple(sorted(self.signals))))

    def __repr__(self) -> str:
        signals = f", signals={dict(self.signals)!r}" if self.signals else ""
        return f"ConnectionRule(rule_type={self.rule_type!r}, params={self.params!r}{signals})"
