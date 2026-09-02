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

"""Shared fixtures for the ``braincell.channel`` test modules.

The catalogue's tests all need the same three things: a voltage array, an
:class:`~braincell.IonInfo` to drive a channel with, and a unit to compare
current densities in. Each ``*_test.py`` used to carry its own copy, which
is how the potassium tests ended up asserting against ``K.Ci = 0.04 mM``
while the potassium-calcium tests used the physiological ``140 mM`` -- a
divergence that was invisible while the two definitions sat in different
files. The builders below take the concentrations as keyword arguments so a
module that needs a different value states it at the call site.

This module is deliberately named with a leading underscore so pytest does
not collect it.
"""

from typing import Sequence

import brainunit as u
import jax.numpy as jnp

from braincell._base_channel import IonInfo

#: Re-exported from ``braincell.ion._testing`` rather than redefined, following
#: the ``vis/_testing.py`` -> ``io/_testing.py`` precedent. Spelled out because
#: every test body binds a local ``V`` for the potential it is testing at.
from braincell.ion._testing import V as voltage

__all__ = [
    "DENSITY_UNIT",
    "assert_channels_agree",
    "ca_info",
    "k_info",
    "na_info",
    "nonspecific_info",
    "voltage",
]

#: Current densities are reported in several equivalent unit spellings
#: depending on how many factors a channel's ``current`` multiplies, so
#: every comparison goes through ``to_decimal`` with this one unit.
DENSITY_UNIT = u.mS / u.cm**2 * u.mV


def k_info(size: int = 1, *, Ci: float = 0.04, Co: float = 2.5, E: float = -90.0) -> IonInfo:
    """Return a potassium :class:`~braincell.IonInfo` of shape ``(size,)``."""
    return IonInfo(
        Ci=jnp.full((size,), Ci) * u.mM,
        Co=jnp.full((size,), Co) * u.mM,
        E=jnp.full((size,), E) * u.mV,
        valence=1,
    )


def na_info(size: int = 1, *, Ci: float = 0.04, Co: float = 140.0, E: float = 50.0) -> IonInfo:
    """Return a sodium :class:`~braincell.IonInfo` of shape ``(size,)``."""
    return IonInfo(
        Ci=jnp.full((size,), Ci) * u.mM,
        Co=jnp.full((size,), Co) * u.mM,
        E=jnp.full((size,), E) * u.mV,
        valence=1,
    )


def ca_info(size: int = 1, *, Ci: float = 1e-4, Co: float = 2.0, E: float = 120.0) -> IonInfo:
    """Return a calcium :class:`~braincell.IonInfo` of shape ``(size,)``."""
    return IonInfo(
        Ci=jnp.full((size,), Ci) * u.mM,
        Co=jnp.full((size,), Co) * u.mM,
        E=jnp.full((size,), E) * u.mV,
        valence=2,
    )


def nonspecific_info(size: int = 1, *, Ci: float = 1.0, Co: float = 1.0, E: float = 0.0) -> IonInfo:
    """Return a non-specific-cation :class:`~braincell.IonInfo` of shape ``(size,)``."""
    return IonInfo(
        Ci=jnp.full((size,), Ci) * u.mM,
        Co=jnp.full((size,), Co) * u.mM,
        E=jnp.full((size,), E) * u.mV,
        valence=1,
    )


def assert_channels_agree(
    case,
    expected,
    actual,
    V,
    *ions: IonInfo,
    states: Sequence[str] = (),
    atol: float = 1e-6,
) -> None:
    """Assert two channels evolve identically from the same reset state.

    Most of the catalogue is cell-type variants: the same mechanism imported
    for a different model, differing only in citation, registration key and
    default parameters. Constructed at matched parameters they must agree
    exactly, and this is the assertion that says so -- reset both, compare
    every state in ``states``, compare its derivative, compare the current.

    Parameters
    ----------
    case : unittest.TestCase
        The test case, used for its assertion methods and failure messages.
    expected, actual : Channel
        The two channels, already constructed at matched parameters.
    V : ArrayLike
        Membrane potential to evaluate both at.
    *ions : IonInfo
        Ion states, in the declaration order of ``root_type``.
    states : sequence of str, optional
        Names of the states to compare. Empty compares the current alone,
        which is what a channel with an all-or-nothing correction term (a
        gating current, say) needs.
    atol : float, optional
        Absolute tolerance, default ``1e-6``. Derivatives are compared with
        the same number in ``u.Hz`` and currents in :data:`DENSITY_UNIT`.
    """
    for channel in (expected, actual):
        channel.init_state(V, *ions)
        channel.reset_state(V, *ions)
    for name in states:
        case.assertTrue(
            u.math.allclose(getattr(actual, name).value, getattr(expected, name).value, atol=atol),
            f"state {name!r} diverged",
        )

    for channel in (expected, actual):
        channel.compute_derivative(V, *ions)
    for name in states:
        case.assertTrue(
            u.math.allclose(
                getattr(actual, name).derivative,
                getattr(expected, name).derivative,
                atol=atol * u.Hz,
            ),
            f"derivative of state {name!r} diverged",
        )

    case.assertTrue(
        u.math.allclose(
            actual.current(V, *ions).to_decimal(DENSITY_UNIT),
            expected.current(V, *ions).to_decimal(DENSITY_UNIT),
            atol=atol,
        ),
        "current diverged",
    )
