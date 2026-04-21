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

from typing import Callable, Mapping

# Importing the step modules below has the side effect of populating the
# global registry via @register_integrator decorators on each *_step function.
from ._backward_euler import backward_euler_step
from ._diffrax import (
    diffrax_bosh3_step,
    diffrax_bwd_euler_step,
    diffrax_dopri5_step,
    diffrax_dopri8_step,
    diffrax_euler_step,
    diffrax_heun_step,
    diffrax_kvaerno3_step,
    diffrax_kvaerno4_step,
    diffrax_kvaerno5_step,
    diffrax_midpoint_step,
    diffrax_ralston_step,
    diffrax_tsit5_step,
)
from ._exp_euler import (
    exp_euler_step,
    ind_exp_euler_step,
)
from ._implicit import (
    cn_exp_euler_step,
    cn_rk4_step,
    exp_exp_euler_step,
    implicit_euler_step,
    implicit_exp_euler_step,
    implicit_rk4_step,
    splitting_step,
)
from .protocol import (
    DiffEqModule,
    DiffEqState,
    IndependentIntegration,
)
from ._registry import (
    IntegratorEntry,
    IntegratorRegistry,
    get_registry,
    register_integrator,
)
from ._runge_kutta import (
    euler_step,
    heun2_step,
    heun3_step,
    midpoint_step,
    ralston2_step,
    ralston3_step,
    ralston4_step,
    rk2_step,
    rk3_step,
    rk4_step,
    ssprk3_step,
)
from ._staggered import staggered_step

__all__ = [
    # registry
    'get_integrator',
    'register_integrator',
    'get_registry',
    'IntegratorEntry',
    'IntegratorRegistry',
    'all_integrators',

    # implicit backward Euler
    'backward_euler_step',

    # exponential Euler
    'exp_euler_step',
    'ind_exp_euler_step',

    # runge-kutta methods
    'euler_step',
    'midpoint_step',
    'rk2_step',
    'heun2_step',
    'ralston2_step',
    'rk3_step',
    'heun3_step',
    'ssprk3_step',
    'ralston3_step',
    'rk4_step',
    'ralston4_step',

    # diffrax explicit methods
    'diffrax_euler_step',
    'diffrax_heun_step',
    'diffrax_midpoint_step',
    'diffrax_ralston_step',
    'diffrax_bosh3_step',
    'diffrax_tsit5_step',
    'diffrax_dopri5_step',
    'diffrax_dopri8_step',

    # diffrax implicit methods
    'diffrax_bwd_euler_step',
    'diffrax_kvaerno3_step',
    'diffrax_kvaerno4_step',
    'diffrax_kvaerno5_step',

    # staggered
    'staggered_step',

    # implicit methods
    'implicit_euler_step',
    'splitting_step',
    'implicit_rk4_step',
    'implicit_exp_euler_step',
    'cn_rk4_step',
    'cn_exp_euler_step',
    'exp_exp_euler_step',

    # protocol
    'DiffEqState',
    'DiffEqModule',
    'IndependentIntegration',

]


class _RegistryDictView(Mapping[str, Callable]):
    """Read-only mapping view backed by the integrator registry.

    Provides backwards compatibility for the legacy ``all_integrators`` dict
    while keeping the registry as the single source of truth. Alias names
    are included so existing lookups such as ``all_integrators['explicit']``
    continue to resolve.
    """

    def __init__(self, registry: IntegratorRegistry) -> None:
        self._registry = registry

    def __getitem__(self, name: str) -> Callable:
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(name)

    def __iter__(self):
        return iter(self._registry.as_dict(include_aliases=True))

    def __len__(self) -> int:
        return len(self._registry.as_dict(include_aliases=True))

    def __contains__(self, name: object) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"_RegistryDictView({sorted(self._registry.as_dict(include_aliases=True))!r})"


#: Backwards-compatible mapping view of the integrator registry.
#:
#: This object behaves like the legacy ``{name: func}`` dict that lived in
#: this module, but it is now backed by :class:`IntegratorRegistry`. Treat it
#: as read-only; new integrators should be registered with
#: :func:`register_integrator`.
all_integrators: Mapping[str, Callable] = _RegistryDictView(get_registry())


def get_integrator(method: str | Callable) -> Callable:
    """Resolve a numerical integrator from a string name or a callable.

    Parameters
    ----------
    method : str or Callable
        Either the registered name (canonical or alias) of an integrator or
        a step function to use directly.

    Returns
    -------
    Callable
        The integrator step function corresponding to ``method``.

    Raises
    ------
    ValueError
        If ``method`` is a string that does not match any registered
        integrator. The error message includes a "did you mean ...?"
        suggestion when a close match exists.
    TypeError
        If ``method`` is neither a string nor a callable.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.quad import get_integrator
        >>> get_integrator('euler')              # doctest: +ELLIPSIS
        <function euler_step at ...>
        >>> get_integrator('explicit') is get_integrator('euler')
        True
        >>> get_integrator('stagger') is get_integrator('staggered')
        True
    """
    if callable(method):
        return method
    if isinstance(method, str):
        registry = get_registry()
        try:
            return registry[method]
        except KeyError:
            suggestions = registry.suggest(method, n=1)
            hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
            available = ", ".join(registry.names())
            raise ValueError(
                f"Unknown integrator {method!r}.{hint} "
                f"Available: {available}."
            )
    raise TypeError(
        f"Integrator method must be a string or callable, got {type(method).__name__}."
    )
