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

"""Shared fixture builders for ``braincell.mech`` tests.

The leading underscore in the filename keeps pytest from discovering this
module as a test file. Nothing here is part of the public API.

``make_cable`` lives here rather than in
``braincell/_discretization/_testing.py`` because :class:`CableProperty`
is a ``braincell.mech`` type; ``_discretization`` re-exports it so its
own tests keep a single import site, the same way
``braincell/filter/_testing.py`` re-exports the ``braincell.morph``
builders.

This module imports only from ``braincell.mech`` itself, preserving the
leaf-package invariant documented in :mod:`braincell.mech`.
"""

import brainunit as u

from braincell.mech._cable import CableProperty

__all__ = ["make_cable"]


def make_cable(cm: float = 1.0, ra: float = 100.0, v: float = -65.0) -> CableProperty:
    """Build a :class:`CableProperty` from bare floats in the usual units.

    The defaults are the values that were retyped as a literal at
    seventeen call sites across the repository.

    Parameters
    ----------
    cm : float
        Specific membrane capacitance in ``uF / cm ** 2``.
    ra : float
        Axial resistivity in ``ohm * cm``.
    v : float
        Resting potential in ``mV``.

    Returns
    -------
    CableProperty
        The assembled declaration.
    """
    return CableProperty(
        resting_potential=v * u.mV,
        membrane_capacitance=cm * (u.uF / u.cm**2),
        axial_resistivity=ra * (u.ohm * u.cm),
    )
