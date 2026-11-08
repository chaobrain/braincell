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

"""Integrator name resolution for ``braincell.quad``.

braincell has no registry of its own: integrator names live in the
:mod:`braintools.quad` registry. braincell's own stateful steps (``exp_euler``,
``ind_exp_euler``, ``backward_euler``, ``implicit_euler``, ``staggered``,
``dhs_voltage``) are registered there under the ``'module'`` kind, with the
signature ``step(target, *args) -> None``. The pure-function ODE steppers of
braintools (``'ode'`` kind, ``step(f, y, t)``) are wrapped into module steps
by :func:`module_step_from_ode`.

:func:`get_integrator` resolves a name in this order:

1. a table of removed names, which raises ``ValueError`` naming the
   replacements (e.g. ``'rk2'`` was Ralston's method in braincell but is
   Heun's method in braintools, so the name must not silently change meaning);
2. the ``'module'`` kind;
3. the ``'ode'`` kind, wrapped by :func:`module_step_from_ode`.
"""

import difflib
from typing import Callable

import braintools.quad as btq

from braincell._misc import set_module_as

__all__ = [
    'get_integrator',
    'integrator_info',
    'list_integrators',
]

#: Names that do not resolve, with the message explaining what to use instead.
#: ``rk2`` stays here for at least one minor version: after it is removed,
#: ``get_integrator('rk2')`` resolves to braintools' ``rk2`` (Heun's method),
#: which differs from the ``rk2`` (Ralston's method) of braincell <= 0.2.
_REMOVED: dict[str, str] = {
    'rk2': (
        "braincell's 'rk2' was Ralston's second-order method; it is now called 'ralston2' "
        "(same numbers). braintools' 'rk2' is Heun's method, available here as 'heun2'."
    ),
    'explicit': "'explicit' was an alias of 'euler'; use 'euler'.",
    'expeuler': (
        "braintools' 'expeuler' advances a single array and cannot integrate several "
        "DiffEqStates; use 'ind_exp_euler' (the same formula applied per DiffEqState) "
        "or 'exp_euler' (coupled, full Jacobian)."
    ),
}


def _same_function(a: Callable, b: Callable) -> bool:
    return (
        getattr(a, '__module__', None) == getattr(b, '__module__', None)
        and getattr(a, '__qualname__', None) == getattr(b, '__qualname__', None)
        and getattr(a, '__qualname__', None) is not None
    )


def register_module_step(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    category: str | None = None,
    order: int | None = None,
    description: str = '',
) -> Callable[[Callable], Callable]:
    """Register a braincell module step in the braintools ``'module'`` kind.

    Registration is idempotent: when the name is already taken by a function
    with the same ``__module__`` and ``__qualname__`` (braincell's own step,
    e.g. after ``importlib.reload``), the entry is replaced. A name taken by
    any other function raises ``ValueError`` (from
    :func:`braintools.quad.register_method`).
    """

    def decorator(fn: Callable) -> Callable:
        try:
            existing = btq.method_info(name, 'module').fn
        except KeyError:
            existing = None
        overwrite = existing is not None and _same_function(existing, fn)
        btq.register_method(
            name,
            fn,
            kind='module',
            overwrite=overwrite,
            order=order,
            category=category,
            aliases=aliases,
            description=description,
        )
        return fn

    return decorator


def _module_names() -> list[str]:
    return btq.list_methods('module')


def _ode_names() -> list[str]:
    # A removed canonical name is listed under its first alias, if any
    # (braintools' 'rk2' as 'heun2').
    names = []
    for name in btq.list_methods('ode'):
        if name not in _REMOVED:
            names.append(name)
        else:
            names.extend(btq.method_info(name, 'ode').aliases[:1])
    return names


def _all_names_and_aliases() -> list[str]:
    names = []
    for kind, canonical in (('module', _module_names()), ('ode', btq.list_methods('ode'))):
        for name in canonical:
            names.append(name)
            names.extend(btq.method_info(name, kind).aliases)
    return [n for n in names if n not in _REMOVED]


def _lookup(name: str) -> tuple[str, 'btq.MethodInfo']:
    """Return ``(kind, info)`` for ``name``; raise ``ValueError`` if it does not resolve."""
    if name in _REMOVED:
        raise ValueError(f'Integrator {name!r} is not available. {_REMOVED[name]}')
    for kind in ('module', 'ode'):
        try:
            info = btq.method_info(name, kind)
        except KeyError:
            continue
        # A removed canonical name may still be reachable through an alias
        # ('heun2' for braintools' 'rk2'), but not through its full function
        # name (e.g. 'ode_expeuler_step').
        if info.name in _REMOVED and name not in info.aliases:
            raise ValueError(f'Integrator {name!r} is not available. {_REMOVED[info.name]}')
        return kind, info
    close = difflib.get_close_matches(name, _all_names_and_aliases(), n=1, cutoff=0.6)
    hint = f' Did you mean {close[0]!r}?' if close else ''
    raise ValueError(f'Unknown integrator {name!r}.{hint} Available: {", ".join(list_integrators())}.')


@set_module_as('braincell.quad')
def get_integrator(method: str | Callable) -> Callable:
    """Resolve a numerical integrator from a string name or a callable.

    Parameters
    ----------
    method : str or Callable
        A name or alias registered in :mod:`braintools.quad` (``'module'``
        kind for braincell's stateful steps, ``'ode'`` kind for the
        pure-function steppers), or a step function ``step(target, *args)``
        to use directly.

    Returns
    -------
    Callable
        The module step ``step(target, *args) -> None``. An ``'ode'`` method
        is wrapped by :func:`module_step_from_ode`; the same wrapper object
        is returned on every call.

    Raises
    ------
    ValueError
        If ``method`` is a removed name (``'rk2'``, ``'explicit'``,
        ``'expeuler'``; the message names the replacement), or matches no
        registered integrator (the message includes a "did you mean ...?"
        suggestion when a close match exists).
    TypeError
        If ``method`` is neither a string nor a callable.

    See Also
    --------
    integrator_info : Metadata of an integrator.
    list_integrators : All integrator names.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.quad import get_integrator, rk4_step
        >>> get_integrator('rk4') is rk4_step
        True
        >>> get_integrator('stagger') is get_integrator('staggered')
        True
        >>> get_integrator('heun3') is get_integrator('heun')
        True
    """
    if callable(method):
        return method
    if isinstance(method, str):
        kind, info = _lookup(method)
        if kind == 'module':
            return info.fn
        from braincell.quad._explicit import module_step_from_ode

        return module_step_from_ode(info.fn)
    raise TypeError(f"Integrator method must be a string or callable, got {type(method).__name__}.")


@set_module_as('braincell.quad')
def integrator_info(name: str) -> 'btq.MethodInfo':
    """Return the metadata of an integrator name.

    Parameters
    ----------
    name : str
        A name or alias accepted by :func:`get_integrator`.

    Returns
    -------
    braintools.quad.MethodInfo
        The canonical name, kind (``'module'`` or ``'ode'``), order, category,
        aliases and description. For an ``'ode'`` method, ``fn`` is the
        pure-function stepper of braintools; use :func:`get_integrator` for
        the module step.

    Raises
    ------
    ValueError
        As :func:`get_integrator`, for removed and unknown names.
    TypeError
        If ``name`` is not a string.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.quad import integrator_info
        >>> info = integrator_info('stagger')
        >>> info.name, info.kind
        ('staggered', 'module')
        >>> integrator_info('rk4').order
        4
    """
    if not isinstance(name, str):
        raise TypeError(f'Integrator name must be a string, got {type(name).__name__}.')
    return _lookup(name)[1]


@set_module_as('braincell.quad')
def list_integrators() -> list[str]:
    """Return the sorted canonical names accepted by :func:`get_integrator`.

    The list holds braincell's stateful steps (``'module'`` kind of
    :mod:`braintools.quad`) and the braintools ODE steppers, without aliases
    and without the removed names; braintools' ``'rk2'`` (Heun's method) is
    listed as ``'heun2'``.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.quad import list_integrators
        >>> 'staggered' in list_integrators() and 'rk4' in list_integrators()
        True
        >>> 'rk2' in list_integrators()
        False
    """
    return sorted(set(_module_names()) | set(_ode_names()))
