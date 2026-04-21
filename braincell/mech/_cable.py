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

"""Passive cable property declaration.

:class:`CableProperty` is the frozen dataclass that records the
passive electrical properties of a region — resting potential,
specific capacitance, axial resistivity, and temperature. It is used
by ``Cell.paint(region, CableProperty(...))`` and lowered into per-CV
cable defaults during :class:`braincell.Cell` rebuild.
"""

from dataclasses import dataclass, field, replace
from typing import Any

import brainunit as u
import numpy as np

__all__ = ["CableProperty"]


def _default_temperature() -> Any:
    return u.celsius2kelvin(36.0)


@dataclass(frozen=True)
class CableProperty:
    """Passive cable properties of a region.

    Parameters
    ----------
    resting_potential : Quantity[mV]
        Reference voltage used as the default voltage initializer.
    membrane_capacitance : Quantity[uF / cm**2]
        Specific membrane capacitance.
    axial_resistivity : Quantity[ohm * cm]
        Cytoplasmic axial resistivity.
    temperature : Quantity[kelvin]
        Absolute temperature used for Q10 scaling in channel kinetics.
        Defaults to 36 °C (309.15 K).

    Raises
    ------
    TypeError
        If ``temperature`` is not a scalar ``brainunit`` quantity.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import CableProperty
        >>> cp = CableProperty(
        ...     resting_potential=-65.0 * u.mV,
        ...     membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
        ...     axial_resistivity=100.0 * (u.ohm * u.cm),
        ... )
        >>> cp.temperature.to_decimal(u.kelvin) > 309.0
        True
    """

    resting_potential: Any
    membrane_capacitance: Any
    axial_resistivity: Any
    temperature: Any = field(default_factory=_default_temperature)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature",
            _coerce_temperature(self.temperature, name="temperature"),
        )

    def with_updates(self, **kwargs: Any) -> "CableProperty":
        """Return a copy with ``kwargs`` applied.

        Parameters
        ----------
        **kwargs
            Fields to override.

        Returns
        -------
        CableProperty
            A new instance. ``self`` is unchanged.
        """
        return replace(self, **kwargs)


def _coerce_temperature(value: Any, *, name: str) -> Any:
    if not hasattr(value, "to_decimal") or not callable(
        getattr(value, "to_decimal")
    ):
        raise TypeError(
            f"CableProperty.{name} must be a temperature Quantity, "
            f"got {value!r}."
        )
    decimal = np.asarray(value.to_decimal(u.kelvin), dtype=float)
    if decimal.ndim != 0:
        raise TypeError(
            f"CableProperty.{name} must be a scalar temperature "
            f"Quantity, got shape {decimal.shape!r}."
        )
    return u.Quantity(float(decimal), u.kelvin)
