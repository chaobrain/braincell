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

"""Distributed (density) mechanism declarations.

This module defines :class:`DensityMechanism`, the single frozen
dataclass that records a paint request for a distributed channel or
ion species. It also exposes the ergonomic factory functions
:func:`Channel` and :func:`Ion`, which are what user-facing paint
calls typically use.

``DensityMechanism`` is deliberately decoupled from any runtime state
— it only carries the information needed to later look up a concrete
class in the :mod:`braincell.mech._registry` registry and construct
an instance. The declaration layer has zero dependency on JAX or
``brainstate``.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from ._params import Params

__all__ = [
    "Channel",
    "DensityMechanism",
    "Ion",
]


_CHANNEL = "channel"
_ION = "ion"
_VALID_CATEGORIES = frozenset({_CHANNEL, _ION})


@dataclass(frozen=True)
class DensityMechanism:
    """Declarative spec for a distributed channel or ion mechanism.

    This is the declaration-layer type that :meth:`Cell.paint
    <braincell.Cell.paint>` records when the user paints a channel or
    ion onto a region of a cell. It is pure data — no runtime state,
    no JAX, no class resolution. Runtime lookup happens later during
    ``Cell`` compile, via
    :func:`braincell.mech.get_registry().get(category, class_name)
    <braincell.mech.MechanismRegistry.get>`.

    Parameters
    ----------
    category : str
        One of ``"channel"`` or ``"ion"``.
    class_name : str
        Registry key for the concrete class. Must be resolvable via
        :func:`braincell.mech.get_registry` once the corresponding
        module (``braincell.channel`` / ``braincell.ion``) has been
        imported.
    params : Params or Mapping
        Parameter name → value mapping. Coerced to :class:`Params` in
        :meth:`__post_init__`.
    name : str or None
        Optional instance label. When ``None``, ``class_name`` is used
        as the display label (see :attr:`instance_name`).
    coverage_area_fraction : float
        Fraction in ``[0, 1]`` of the target control volume's lateral
        area that this declaration covers. Set by
        :func:`braincell.cv._mech._scale_density_for_coverage` during
        lazy rebuild when a paint region only partially overlaps a
        CV. Defaults to ``1.0`` (full coverage). This is a geometry
        annotation, not a mechanism parameter, which is why it lives
        as a dedicated field rather than inside ``params``.

    Raises
    ------
    ValueError
        If ``category`` is not ``"channel"`` or ``"ion"``, if
        ``class_name`` is empty, or if ``coverage_area_fraction`` is
        outside ``[0, 1]``.

    See Also
    --------
    Channel : Ergonomic factory for channel-category mechanisms.
    Ion : Ergonomic factory for ion-category mechanisms.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import DensityMechanism, Channel, Params
        >>> leak = Channel("IL", g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV)
        >>> leak.class_name
        'IL'
        >>> leak.category
        'channel'
        >>> leak.params["g_max"]
        0.1 * mS / cm ** 2

        >>> # Direct construction — rarely needed
        >>> leak2 = DensityMechanism(
        ...     category="channel",
        ...     class_name="IL",
        ...     params=Params(g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV),
        ... )
        >>> leak == leak2
        True
    """

    category: str
    class_name: str
    params: Params = field(default_factory=Params)
    name: str | None = None
    coverage_area_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.category not in _VALID_CATEGORIES:
            raise ValueError(
                f"DensityMechanism.category must be one of "
                f"{sorted(_VALID_CATEGORIES)!r}, got {self.category!r}."
            )
        if not isinstance(self.class_name, str) or not self.class_name:
            raise ValueError(
                f"DensityMechanism.class_name must be a non-empty "
                f"string, got {self.class_name!r}."
            )
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError(
                f"DensityMechanism.name must be a string or None, "
                f"got {type(self.name).__name__!r}."
            )
        coverage = float(self.coverage_area_fraction)
        if not (0.0 <= coverage <= 1.0):
            raise ValueError(
                f"DensityMechanism.coverage_area_fraction must lie in "
                f"[0, 1], got {coverage!r}."
            )
        object.__setattr__(self, "coverage_area_fraction", coverage)
        object.__setattr__(self, "params", Params.coerce(self.params))

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def instance_name(self) -> str:
        """Display label for this declaration.

        Returns ``self.name`` when set, otherwise ``self.class_name``.
        """
        return self.name if self.name is not None else self.class_name

    @property
    def identity(self) -> tuple[str, str]:
        """Return ``(instance_name, class_name)`` for table views."""
        return (self.instance_name, self.class_name)

    # ------------------------------------------------------------------
    # non-mutating updates
    # ------------------------------------------------------------------

    def with_params(self, **updates: Any) -> "DensityMechanism":
        """Return a copy with ``updates`` merged into ``params``.

        Parameters
        ----------
        **updates
            Parameters to add or replace.

        Returns
        -------
        DensityMechanism
            A new instance. ``self`` is unchanged.
        """
        return replace(self, params=self.params.with_updates(**updates))

    def with_coverage(self, fraction: float) -> "DensityMechanism":
        """Return a copy with a new ``coverage_area_fraction``.

        Parameters
        ----------
        fraction : float
            New fraction, must lie in ``[0, 1]``.

        Returns
        -------
        DensityMechanism
            A new instance. ``self`` is unchanged.
        """
        return replace(self, coverage_area_fraction=float(fraction))

    def with_name(self, name: str | None) -> "DensityMechanism":
        """Return a copy with a new instance label."""
        return replace(self, name=name)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def Channel(
    class_name: str,
    /,
    *,
    name: str | None = None,
    coverage_area_fraction: float = 1.0,
    **params: Any,
) -> DensityMechanism:
    """Build a :class:`DensityMechanism` for an ion channel.

    Parameters
    ----------
    class_name : str
        Registry key for the target channel class. Must be resolvable
        after ``braincell.channel`` is imported.
    name : str or None
        Optional instance label. See
        :attr:`DensityMechanism.instance_name`.
    coverage_area_fraction : float
        Fraction of the target CV's lateral area this declaration
        covers. Callers rarely set this directly — it is typically
        computed by the paint lowering pass.
    **params
        Channel parameters, passed as keyword arguments with
        ``brainunit`` quantity values (e.g. ``g_max=0.1 * u.mS /
        u.cm ** 2``, ``E=-70 * u.mV``).

    Returns
    -------
    DensityMechanism
        A new declaration with ``category="channel"``.

    Examples
    --------

    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell.mech import Channel
        >>> spec = Channel("IL", g_max=0.1 * u.mS / u.cm ** 2, E=-70 * u.mV)
        >>> spec.category
        'channel'
    """
    return DensityMechanism(
        category=_CHANNEL,
        class_name=class_name,
        params=Params(params),
        name=name,
        coverage_area_fraction=coverage_area_fraction,
    )


def Ion(
    class_name: str,
    /,
    *,
    name: str | None = None,
    coverage_area_fraction: float = 1.0,
    **params: Any,
) -> DensityMechanism:
    """Build a :class:`DensityMechanism` for an ion species.

    See :func:`Channel` for the parameter semantics. The only
    difference is that the returned declaration carries
    ``category="ion"``.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.mech import Ion
        >>> spec = Ion("CalciumFixed", C=5e-5, E=120.0)
        >>> spec.category
        'ion'
    """
    return DensityMechanism(
        category=_ION,
        class_name=class_name,
        params=Params(params),
        name=name,
        coverage_area_fraction=coverage_area_fraction,
    )
