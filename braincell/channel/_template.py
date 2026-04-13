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
    "GateChannelTemplate",
    "q10_scale",
    "shifted_voltage",
]


def shifted_voltage(V, V_sh):
    return V - V_sh


def q10_scale(temp, temp_ref, q10):
    return q10 ** (((temp - temp_ref) / u.kelvin) / 10.0)


@dataclass(frozen=True)
class Gate:
    """Minimal metadata for one gate in a gate-based channel."""

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

    def f_phi(self, channel):
        if self.phi is not None:
            return self.phi
        if self.q10 is not None:
            return q10_scale(channel.temp, self.temp_ref, self.q10)
        return 1.0


class GateChannelTemplate(Channel):
    """Minimal gate-based channel skeleton using only ``inf/tau`` kinetics."""

    gate_defs: ClassVar[tuple[Gate, ...]] = ()
    g_max_attr: ClassVar[str] = "g_max"

    def _iter_gate_defs(self) -> tuple[Gate, ...]:
        return type(self).gate_defs

    def _expected_ion_arg_count(self) -> int:
        root_type = getattr(self, "root_type", None)
        if hasattr(root_type, "__args__"):
            return len(root_type.__args__)
        return 1

    def _split_ion_infos_and_batch_size(self, args, batch_size):
        if batch_size is not None:
            return tuple(args), batch_size
        expected = self._expected_ion_arg_count()
        if len(args) == expected + 1 and (args[-1] is None or isinstance(args[-1], int)):
            return tuple(args[:-1]), args[-1]
        return tuple(args), None

    def _gate_state(self, gate: Gate) -> DiffEqState:
        return getattr(self, gate.name)

    def _gate_value(self, gate: Gate):
        return self._gate_state(gate).value

    def gate_phi(self, gate: Gate):
        return gate.f_phi(self)

    def _gate_inf(self, gate: Gate, V, *ion_infos):
        return getattr(self, f"f_{gate.name}_inf")(V, *ion_infos)

    def _gate_tau(self, gate: Gate, V, *ion_infos):
        return getattr(self, f"f_{gate.name}_tau")(V, *ion_infos)

    def _gate_derivative(self, gate: Gate, V, *ion_infos):
        value = self._gate_value(gate)
        phi = self.gate_phi(gate)
        return phi * (self._gate_inf(gate, V, *ion_infos) - value) / self._gate_tau(gate, V, *ion_infos) / u.ms

    def gating_product(self):
        product = 1.0
        for gate in self._iter_gate_defs():
            value = self._gate_value(gate)
            product = product * (value if gate.power == 1 else value ** gate.power)
        return product

    def conductance(self, V, *ion_infos):
        return getattr(self, self.g_max_attr) * self.gating_product()

    def drive(self, V, *ion_infos):
        raise NotImplementedError

    def current(self, V, *ion_infos):
        return self.conductance(V, *ion_infos) * self.drive(V, *ion_infos)

    def init_state(self, V, *args, batch_size: int = None):
        _, batch_size = self._split_ion_infos_and_batch_size(args, batch_size)
        for gate in self._iter_gate_defs():
            setattr(
                self,
                gate.name,
                DiffEqState(braintools.init.param(u.math.zeros, self.varshape, batch_size)),
            )

    def reset_state(self, V, *args, batch_size: int = None):
        ion_infos, batch_size = self._split_ion_infos_and_batch_size(args, batch_size)
        for gate in self._iter_gate_defs():
            self._gate_state(gate).value = self._gate_inf(gate, V, *ion_infos)
            if isinstance(batch_size, int):
                assert self._gate_state(gate).value.shape[0] == batch_size

    def compute_derivative(self, V, *ion_infos):
        for gate in self._iter_gate_defs():
            self._gate_state(gate).derivative = self._gate_derivative(gate, V, *ion_infos)
