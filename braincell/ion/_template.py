# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import ClassVar

import brainstate
import braintools
import brainunit as u

from braincell._base import Channel
from braincell.quad import DiffEqState
from braincell.quad import IndependentIntegration

__all__ = [
    "IonData",
    "Constant",
    "InitNernst",
    "DynamicNernst",
]


@dataclass(frozen=True)
class IonData:
    """Template-only ion payload used by channel prototypes."""

    Ci: Any
    Co: Any
    E: Any
    valence: Any

    @property
    def C(self):
        return self.Ci

    @property
    def C0(self):
        return self.Co

    @property
    def z(self):
        return self.valence


def _resolve_value(owner, value):
    return value(owner) if callable(value) else value


def _materialize(value):
    return value.value if isinstance(value, brainstate.State) else value


def _channel_nodes(owner):
    return tuple(brainstate.graph.nodes(owner, Channel, allowed_hierarchy=(1, 1)).values())


def _check_hierarchies(owner, nodes):
    if hasattr(owner, "check_hierarchies"):
        owner.check_hierarchies(type(owner), *tuple(nodes))


def _init_child_channels(owner, V, batch_size=None):
    nodes = _channel_nodes(owner)
    _check_hierarchies(owner, nodes)
    ion_data = owner.pack_info()
    for node in nodes:
        node.init_state(V, ion_data, batch_size)


def _reset_child_channels(owner, V, batch_size=None):
    nodes = _channel_nodes(owner)
    ion_data = owner.pack_info()
    for node in nodes:
        node.reset_state(V, ion_data, batch_size)


def _compute_child_channel_derivatives(owner, V):
    ion_data = owner.pack_info()
    for node in _channel_nodes(owner):
        if not isinstance(node, IndependentIntegration):
            node.compute_derivative(V, ion_data)


def _nernst(Ci, Co, valence, T):
    return (u.gas_constant * T / (valence * u.faraday_constant)) * u.math.log(Co / Ci)


class _IonTemplateMixin:
    ci_attr: ClassVar[str] = "Ci"
    co_attr: ClassVar[str] = "Co"
    e_attr: ClassVar[str] = "E"
    valence_attr: ClassVar[str] = "valence"
    temp_attr: ClassVar[str] = "T"

    def _ci_value(self):
        return _materialize(getattr(self, type(self).ci_attr))

    def _co_value(self):
        return _materialize(getattr(self, type(self).co_attr))

    def _e_value(self):
        return _materialize(getattr(self, type(self).e_attr))

    def _valence_value(self):
        return _materialize(getattr(self, type(self).valence_attr))

    def _temp_value(self):
        return _materialize(getattr(self, type(self).temp_attr))


class Constant(_IonTemplateMixin):
    """Fixed ``Ci/Co/E`` ion template with no Nernst coupling."""

    def init_state(self, V, batch_size: int = None):
        _init_child_channels(self, V, batch_size=batch_size)

    def reset_state(self, V, batch_size: int = None):
        _reset_child_channels(self, V, batch_size=batch_size)

    def compute_derivative(self, V):
        _compute_child_channel_derivatives(self, V)

    def pack_info(self) -> IonData:
        return IonData(
            Ci=self._ci_value(),
            Co=self._co_value(),
            E=self._e_value(),
            valence=self._valence_value(),
        )


class InitNernst(_IonTemplateMixin):
    """Fixed ``Ci/Co`` with ``E`` computed once during init/reset."""

    def _update_reversal(self):
        E = _nernst(
            Ci=self._ci_value(),
            Co=self._co_value(),
            valence=self._valence_value(),
            T=self._temp_value(),
        )
        setattr(self, type(self).e_attr, E)

    def init_state(self, V, batch_size: int = None):
        self._update_reversal()
        _init_child_channels(self, V, batch_size=batch_size)

    def reset_state(self, V, batch_size: int = None):
        self._update_reversal()
        _reset_child_channels(self, V, batch_size=batch_size)

    def compute_derivative(self, V):
        _compute_child_channel_derivatives(self, V)

    def pack_info(self) -> IonData:
        return IonData(
            Ci=self._ci_value(),
            Co=self._co_value(),
            E=self._e_value(),
            valence=self._valence_value(),
        )


class DynamicNernst(_IonTemplateMixin):
    """Dynamic ``Ci`` with per-step Nernst reversal updates."""

    ci_state: ClassVar[str] = "Ci"
    ci_initializer: ClassVar[Any] = 0.0 * u.mM

    def _ci_state(self):
        return getattr(self, type(self).ci_state)

    def _ci_value(self):
        return self._ci_state().value

    @property
    def E(self):
        return _nernst(
            Ci=self._ci_value(),
            Co=self._co_value(),
            valence=self._valence_value(),
            T=self._temp_value(),
        )

    def init_state(self, V, batch_size: int = None):
        initial = _resolve_value(self, type(self).ci_initializer)
        setattr(
            self,
            type(self).ci_state,
            DiffEqState(braintools.init.param(initial, self.varshape, batch_size)),
        )
        _init_child_channels(self, V, batch_size=batch_size)

    def reset_state(self, V, batch_size: int = None):
        value = braintools.init.param(
            _resolve_value(self, type(self).ci_initializer),
            self.varshape,
            batch_size,
        )
        self._ci_state().value = value
        if isinstance(batch_size, int):
            assert value.shape[0] == batch_size
        _reset_child_channels(self, V, batch_size=batch_size)

    def compute_derivative(self, V):
        _compute_child_channel_derivatives(self, V)
        total_current = self.current(V, include_external=True)
        self._ci_state().derivative = self.ci_derivative(self._ci_value(), V, total_current)

    def ci_derivative(self, Ci, V, total_current):
        raise NotImplementedError

    def pack_info(self) -> IonData:
        return IonData(
            Ci=self._ci_value(),
            Co=self._co_value(),
            E=self.E,
            valence=self._valence_value(),
        )
