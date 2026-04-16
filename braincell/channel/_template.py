# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import ClassVar

import braintools
import brainunit as u

from braincell._base import Channel
from braincell.quad import DiffEqState

__all__ = [
    "Gate",
    "Transition",
    "OpenState",
    "Passive",
    "HH",
    "Markov",
    "ghk_flux",
]


def ghk_flux(V, ci, co, z, T):
    """Unit-aware GHK flux helper with a small-zeta stable branch."""

    zeta = (z * u.faraday_constant * V) / (u.gas_constant * T)
    exp_term = u.math.exp(-zeta)
    numerator = ci - co * exp_term
    small_branch = (z * u.faraday_constant) * numerator * (1 + zeta / 2)
    regular_branch = (z * zeta * u.faraday_constant) * numerator / (1 - exp_term)
    return u.math.where(u.math.abs(1 - exp_term) <= 1e-6, small_branch, regular_branch)


def _resolve_value(owner, value):
    return value(owner) if callable(value) else value


@dataclass(frozen=True)
class Gate:
    """Metadata for one HH gate."""

    name: str
    power: int = 1
    phi: Any | None = None
    q10: Any | None = None
    temp_ref: Any | None = None

    def __post_init__(self):
        has_phi = self.phi is not None
        has_q10 = self.q10 is not None
        has_temp_ref = self.temp_ref is not None

        if has_phi and (has_q10 or has_temp_ref):
            raise ValueError(
                f"Gate {self.name!r}: phi cannot be provided together with q10/temp_ref."
            )
        if has_q10 != has_temp_ref:
            raise ValueError(
                f"Gate {self.name!r}: q10 and temp_ref must be provided together."
            )


@dataclass(frozen=True)
class Transition:
    """One directed/reversible transition used by Markov channels."""

    src: str
    dst: str
    forward: str
    backward: str | None = None


@dataclass(frozen=True)
class OpenState:
    """One conducting state and its contribution weight."""

    name: str
    weight: float = 1.0


class Passive(Channel):
    """No-state channel dynamics."""

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (self, V, ions, batch_size)

    def reset_state(self, V, *ions, batch_size: int = None):
        _ = (self, V, ions, batch_size)

    def compute_derivative(self, V, *ions):
        _ = (self, V, ions)

    def conductance_factor(self, V, *ions):
        _ = (self, V, ions)
        return 1.0


class HH(Channel):
    """HH gate dynamics with per-gate auto-detected form.

    For each gate ``g`` exactly one of the following method pairs must exist:

    - ``f_<g>_inf`` and ``f_<g>_tau``
    - ``f_<g>_alpha`` and ``f_<g>_beta``
    """

    gates: ClassVar[tuple[Gate | tuple[Any, ...], ...]] = ()

    def _iter_gates(self) -> tuple[Gate, ...]:
        items = []
        for gate in type(self).gates:
            if isinstance(gate, Gate):
                items.append(gate)
            else:
                items.append(Gate(*gate))
        return tuple(items)

    def _gate_state(self, gate: Gate) -> DiffEqState:
        return getattr(self, gate.name)

    def _gate_value(self, gate: Gate):
        return self._gate_state(gate).value

    def gate_phi(self, gate: Gate):
        """Resolve one gate's temperature factor.

        Resolution order is intentionally simple:

        1. explicit ``phi``
        2. ``q10`` + ``temp_ref`` using ``self.temp``
        3. default ``1.0``
        """
        if gate.phi is not None:
            return _resolve_value(self, gate.phi)
        if gate.q10 is not None:
            q10 = _resolve_value(self, gate.q10)
            temp_ref = _resolve_value(self, gate.temp_ref)
            return q10 ** (((self.temp - temp_ref) / u.kelvin) / 10.0)
        return 1.0

    def _has_inf_tau(self, gate: Gate) -> bool:
        return hasattr(self, f"f_{gate.name}_inf") and hasattr(self, f"f_{gate.name}_tau")

    def _has_alpha_beta(self, gate: Gate) -> bool:
        return hasattr(self, f"f_{gate.name}_alpha") and hasattr(self, f"f_{gate.name}_beta")

    def _gate_form(self, gate: Gate) -> str:
        has_inf_tau = self._has_inf_tau(gate)
        has_alpha_beta = self._has_alpha_beta(gate)
        if has_inf_tau and has_alpha_beta:
            raise ValueError(
                f"Gate {gate.name!r} defines both inf/tau and alpha/beta forms; choose one."
            )
        if has_inf_tau:
            return "inf_tau"
        if has_alpha_beta:
            return "alpha_beta"
        raise ValueError(
            f"Gate {gate.name!r} must define either inf/tau or alpha/beta methods."
        )

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for gate in self._iter_gates():
            setattr(
                self,
                gate.name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
            )

    def conductance_factor(self, V, *ions):
        _ = (V, ions)
        product = 1.0
        for gate in self._iter_gates():
            value = self._gate_value(gate)
            product = product * (value if gate.power == 1 else value ** gate.power)
        return product


    def reset_state(self, V, *ions, batch_size: int = None):
        for gate in self._iter_gates():
            form = self._gate_form(gate)
            if form == "inf_tau":
                value = getattr(self, f"f_{gate.name}_inf")(V, *ions)
            else:
                alpha = getattr(self, f"f_{gate.name}_alpha")(V, *ions)
                beta = getattr(self, f"f_{gate.name}_beta")(V, *ions)
                value = alpha / (alpha + beta)
            self._gate_state(gate).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def compute_derivative(self, V, *ions):
        for gate in self._iter_gates():
            value = self._gate_value(gate)
            phi = self.gate_phi(gate)
            form = self._gate_form(gate)
            if form == "inf_tau":
                inf = getattr(self, f"f_{gate.name}_inf")(V, *ions)
                tau = getattr(self, f"f_{gate.name}_tau")(V, *ions)
                derivative = phi * (inf - value) / tau / u.ms
            else:
                alpha = getattr(self, f"f_{gate.name}_alpha")(V, *ions)
                beta = getattr(self, f"f_{gate.name}_beta")(V, *ions)
                derivative = phi * (alpha * (1.0 - value) - beta * value) / u.ms
            self._gate_state(gate).derivative = derivative


class Markov(Channel):
    """Probability-state channel kinetics described by transition pairs."""

    pairs: ClassVar[tuple[Transition | tuple[Any, ...], ...]] = ()
    conserve: ClassVar[Any] = 1.0
    dependent_state: ClassVar[str | None] = None
    conducting: ClassVar[tuple[OpenState | tuple[Any, ...], ...]] = ()

    def _iter_pairs(self) -> tuple[Transition, ...]:
        items = []
        for pair in type(self).pairs:
            if isinstance(pair, Transition):
                items.append(pair)
            else:
                items.append(Transition(*pair))
        return tuple(items)

    def _iter_conducting(self) -> tuple[OpenState, ...]:
        items = []
        for state in type(self).conducting:
            if isinstance(state, OpenState):
                items.append(state)
            else:
                items.append(OpenState(*state))
        return tuple(items)

    def _state_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for pair in self._iter_pairs():
            for name in (pair.src, pair.dst):
                if name not in seen:
                    names.append(name)
                    seen.add(name)
        for state in self._iter_conducting():
            if state.name not in seen:
                names.append(state.name)
                seen.add(state.name)
        return tuple(names)

    def _dependent_state_name(self) -> str:
        state_names = self._state_names()
        if len(state_names) < 2:
            raise ValueError("Markov requires at least two states.")
        if type(self).dependent_state is not None:
            if type(self).dependent_state not in state_names:
                raise ValueError(
                    f"dependent_state {type(self).dependent_state!r} is not present in Markov states."
                )
            return type(self).dependent_state
        return state_names[-1]

    def _independent_state_names(self) -> tuple[str, ...]:
        dependent = self._dependent_state_name()
        return tuple(name for name in self._state_names() if name != dependent)

    def _state_zero(self):
        independent = self._independent_state_names()
        if not independent:
            raise ValueError("Markov requires at least one independent state.")
        return u.math.zeros_like(getattr(self, independent[0]).value)

    def _conserve_value(self):
        return _resolve_value(self, type(self).conserve)

    def init_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            setattr(
                self,
                name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
            )

    def reset_state(self, V, *ions, batch_size: int = None):
        _ = (V, ions)
        for name in self._independent_state_names():
            value = braintools.init.param(u.math.zeros, self.varshape, batch_size)
            getattr(self, name).value = value
            if isinstance(batch_size, int):
                assert value.shape[0] == batch_size

    def state_values(self):
        states = {
            name: getattr(self, name).value
            for name in self._independent_state_names()
        }
        dependent = self._dependent_state_name()
        total = None
        for value in states.values():
            total = value if total is None else (total + value)
        if total is None:
            total = 0.0
        states[dependent] = self._conserve_value() - total
        return states

    def compute_derivative(self, V, *ions):
        states = self.state_values()
        derivatives = {name: self._state_zero() for name in states}

        for pair in self._iter_pairs():
            forward = getattr(self, pair.forward)(V, *ions)
            derivatives[pair.src] = derivatives[pair.src] - states[pair.src] * forward
            derivatives[pair.dst] = derivatives[pair.dst] + states[pair.src] * forward

            if pair.backward is not None:
                backward = getattr(self, pair.backward)(V, *ions)
                derivatives[pair.src] = derivatives[pair.src] + states[pair.dst] * backward
                derivatives[pair.dst] = derivatives[pair.dst] - states[pair.dst] * backward

        for name in self._independent_state_names():
            getattr(self, name).derivative = derivatives[name] / u.ms

    def conductance_factor(self, V, *ions):
        _ = (V, ions)
        states = self.state_values()
        factor = 0.0
        for state in self._iter_conducting():
            factor = factor + states[state.name] * state.weight
        return factor
