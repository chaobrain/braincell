from __future__ import annotations

from dataclasses import dataclass
from .._units import u


@dataclass(frozen=True)
class CableProperties:
    resting_potential: u.Quantity[u.mV]
    membrane_capacitance: u.Quantity[u.uF / u.cm**2]
    axial_resistivity: u.Quantity[u.ohm * u.cm]
    temperature: u.Quantity[u.kelvin] = u.celsius2kelvin(36.0)
