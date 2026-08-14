# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""The integration protocol shared by every BrainCell neuron model.

Defines the state classes solvers consume — the :class:`DiffEqState`
marker mixin and its two concrete carriers :class:`DiffEqSingleState`
and :class:`DiffEqGroupState` — the :class:`DiffEqModule` mixin that
declares a module integrable, and the host-scoped factory that chooses
between the grouped and ungrouped classes.

Which class a hidden state gets is a per-host decision:
:class:`braincell.SingleCompartment` has no spatial axis and uses plain
hidden states, while :class:`braincell.Cell` groups its trailing
compartment axis. See ``docs/specs/2026-08-13-cell-hidden-group-state.md``
and ``docs/design/cell.md``.
"""

import contextlib
import contextvars
from typing import Callable, Iterator

import brainstate
from brainstate._state import record_state_value_write

from braincell._misc import set_module_as

__all__ = [
    'DiffEqState',
    'DiffEqSingleState',
    'DiffEqGroupState',
    'DiffEqModule',
    'IndependentIntegration',
    'state_grouping',
    'state',
    'hidden_state',
]


class DiffEqState(brainstate.mixin.Mixin):
    """Marker mixin for a state that participates in numerical integration.

    A :class:`DiffEqState` is the unit of work consumed by every solver in
    :mod:`braincell.quad`. It contributes two slots — ``derivative`` and
    ``diffusion`` — that the surrounding solver writes during one ODE/SDE
    step:

    - ``derivative`` is the right-hand side :math:`f(t, y)` for an ODE
      :math:`\\dot y = f(t, y)`, or the *drift* term for an SDE
      :math:`dy = f(t, y)\\,dt + g(t, y)\\,dW`.
    - ``diffusion`` is the SDE noise coefficient :math:`g(t, y)`. It
      stays ``None`` for plain ODE systems.

    This class is a :class:`brainstate.mixin.Mixin`, **not** a
    :class:`brainstate.State`. It carries no storage and cannot be
    instantiated; ``DiffEqState(value)`` raises :exc:`TypeError`. Mixing
    it into something that is not a :class:`brainstate.State` yields an
    object no solver will accept. Use :func:`state` to allocate,
    or name :class:`DiffEqSingleState` / :class:`DiffEqGroupState`
    explicitly.

    Separating the marker from the storage layout is what lets the two
    concrete classes be siblings. Solver state selection goes through
    ``isinstance(value, DiffEqState)``
    (see :func:`braincell.quad._util.split_diffeq_states`), which stays
    ``True`` for both.

    Attributes
    ----------
    derivative : brainstate.typing.PyTree
        Time derivative (or SDE drift) of the state. Set inside
        :meth:`DiffEqModule.compute_derivative`. Must carry units that
        satisfy ``unit(derivative) * unit(dt) == unit(value)``.
    diffusion : brainstate.typing.PyTree
        Optional SDE diffusion coefficient. ``None`` denotes a
        deterministic ODE system.

    See Also
    --------
    DiffEqSingleState : Concrete ungrouped state, used by
        ``SingleCompartment``.
    DiffEqGroupState : Concrete grouped state, used by ``Cell``.
    DiffEqModule : Container that owns and updates the states.
    """

    __module__ = 'braincell'

    #: Class-level defaults in place of ``__init__``: a brainstate
    #: ``Mixin`` must not define one. The property setters below shadow
    #: these per instance on first write.
    _derivative = None
    _diffusion = None

    @property
    def derivative(self):
        """
        Get the derivative of the state.

        Returns
        -------
        brainstate.typing.PyTree
            The derivative of the state, used to compute the derivative of the ODE system
            or the drift of the SDE system.
        """
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        """
        Set the derivative of the state.

        Parameters
        ----------
        value : brainstate.typing.PyTree
            The new value for the derivative of the state.
        """
        record_state_value_write(self)
        self._derivative = value

    @property
    def diffusion(self):
        """
        Get the diffusion of the state.

        Returns
        -------
        brainstate.typing.PyTree
            The diffusion of the state, used to compute the diffusion of the SDE system.
            If it is None, the system is considered as an ODE system.
        """
        return self._diffusion

    @diffusion.setter
    def diffusion(self, value):
        """
        Set the diffusion of the state.

        Parameters
        ----------
        value : brainstate.typing.PyTree
            The new value for the diffusion of the state.
        """
        record_state_value_write(self)
        self._diffusion = value

    def __pretty_repr_item__(self, k, v):
        if k == '_derivative':
            if self._derivative is not None:
                return 'derivative', self._derivative
            else:
                return None
        if k == '_diffusion':
            if self._diffusion is not None:
                return 'diffusion', self._diffusion
            else:
                return None
        return super().__pretty_repr_item__(k, v)


class DiffEqSingleState(DiffEqState, brainstate.HiddenState):
    """An integrable hidden state with no trailing state axis.

    This is the state class used by every hidden variable owned by a
    :class:`braincell.SingleCompartment`, which has no spatial axis: one
    value per element of ``varshape``, and nothing to group.

    See Also
    --------
    DiffEqGroupState : The grouped counterpart used by ``Cell``.
    state : Host-scoped factory that picks between the two.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> import braincell
        >>> state = braincell.DiffEqSingleState(np.zeros(4) * u.mV)
        >>> state.varshape
        (4,)
        >>> isinstance(state, braincell.DiffEqState)
        True
    """

    __module__ = 'braincell'


class DiffEqGroupState(DiffEqState, brainstate.HiddenGroupState):
    """An integrable hidden state whose trailing axis indexes independent states.

    This is the state class used by every hidden variable owned by a
    :class:`braincell.Cell`. A ``Cell`` is a *spatial* model: its runtime
    arrays are shaped ``pop_size + (n_cv,)`` for voltage and
    ``pop_size + (n_point,)`` for mechanism variables, so the trailing
    axis enumerates compartments (or points) that evolve independently.
    That is exactly the contract of
    :class:`brainstate.HiddenGroupState` — ``varshape`` is everything but
    the last axis and ``num_state`` is the last axis — which lets an
    eligibility-trace learner treat one array as ``num_state`` separately
    traced hidden units.

    Notes
    -----
    :class:`brainstate.HiddenGroupState` requires ``value.ndim >= 2``.
    This is why :class:`braincell.Cell` makes its population axis
    mandatory (``pop_size`` defaults to ``1`` and may not be empty) — the
    validation is inherited unmodified rather than relaxed.

    See Also
    --------
    DiffEqSingleState : The ungrouped counterpart used by
        ``SingleCompartment``.
    state : Host-scoped factory that picks between the two.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> import braincell
        >>> state = braincell.DiffEqGroupState(np.zeros((1, 4)) * u.mV)
        >>> state.varshape
        (1,)
        >>> state.num_state
        4
        >>> isinstance(state, braincell.DiffEqState)
        True
    """

    __module__ = 'braincell'


_STATE_GROUPING = contextvars.ContextVar('braincell_state_grouping', default=False)


@set_module_as('braincell')
@contextlib.contextmanager
def state_grouping(enabled: bool = True) -> Iterator[bool]:
    """Scope whether :func:`state` / :func:`hidden_state` group.

    Channel, ion, and synapse code is shared by
    :class:`braincell.SingleCompartment` and :class:`braincell.Cell`, so
    the correct hidden-state class cannot be chosen statically at the
    creation site. Instead the *host* publishes its identity for the
    duration of ``init_state`` / ``reset_state``, and the factories read
    it from this context.

    A :mod:`contextvars` variable — not a module-level global — is used so
    that nesting, exceptions, and threads all restore the previous value
    correctly.

    Parameters
    ----------
    enabled : bool, default True
        ``True`` inside a :class:`braincell.Cell`, ``False`` inside a
        :class:`braincell.SingleCompartment`. Both hosts set it
        explicitly, so a :class:`braincell.Network` holding a mix of the
        two is correct regardless of construction order.

    Yields
    ------
    bool
        The value now in effect, for convenience.

    See Also
    --------
    state : Allocate an integrable hidden state under this scope.
    hidden_state : Allocate a non-integrable hidden state under this scope.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> import braincell
        >>> with braincell.state_grouping(True):
        ...     st = braincell.state(np.zeros((1, 4)) * u.mV)
        >>> type(st).__name__
        'DiffEqGroupState'
    """
    token = _STATE_GROUPING.set(bool(enabled))
    try:
        yield bool(enabled)
    finally:
        _STATE_GROUPING.reset(token)


@set_module_as('braincell')
def state(value, **kwargs) -> DiffEqState:
    """Allocate the integrable hidden state class the current host wants.

    Parameters
    ----------
    value : ArrayLike
        Initial value, normally a :class:`brainunit.Quantity`.
    **kwargs
        Forwarded to the state constructor.

    Returns
    -------
    DiffEqState
        A :class:`DiffEqGroupState` inside :func:`state_grouping`
        (i.e. within a :class:`braincell.Cell`), otherwise a
        :class:`DiffEqSingleState`.

    See Also
    --------
    state_grouping : Scope that selects the class.
    hidden_state : The non-integrable counterpart.
    """
    cls = DiffEqGroupState if _STATE_GROUPING.get() else DiffEqSingleState
    return cls(value, **kwargs)


@set_module_as('braincell')
def hidden_state(value, **kwargs) -> brainstate.HiddenState:
    """Allocate the non-integrable hidden state class the host wants.

    Used for hidden variables that are written algebraically rather than
    integrated — for example an ion concentration held fixed, or a
    species value recomputed from a conservation law.

    Parameters
    ----------
    value : ArrayLike
        Initial value, normally a :class:`brainunit.Quantity`.
    **kwargs
        Forwarded to the state constructor.

    Returns
    -------
    brainstate.HiddenState
        A :class:`brainstate.HiddenGroupState` inside
        :func:`state_grouping` (i.e. within a :class:`braincell.Cell`),
        otherwise a plain :class:`brainstate.HiddenState`.

    See Also
    --------
    state_grouping : Scope that selects the class.
    state : The integrable counterpart.
    """
    cls = brainstate.HiddenGroupState if _STATE_GROUPING.get() else brainstate.HiddenState
    return cls(value, **kwargs)


class DiffEqModule(brainstate.mixin.Mixin):
    """Mixin marking a module as integrable by :mod:`braincell.quad`.

    Any class that mixes in :class:`DiffEqModule` exposes the small
    interface that every numerical integrator in :mod:`braincell.quad`
    relies on:

    - :meth:`pre_integral` — invoked once at the start of each step,
      before any derivative is computed. Use it to refresh
      voltage-dependent rate constants, recompute synaptic input, or
      perform other one-time-per-step bookkeeping.
    - :meth:`compute_derivative` — required override that writes
      ``state.derivative`` (and optionally ``state.diffusion``) for every
      :class:`DiffEqState` owned by the module.
    - :meth:`post_integral` — invoked once at the end of each step, after
      the integrated values have been written back. Use it to clamp
      states, project onto manifolds, or fire post-step events.

    Concrete subclasses include :class:`braincell.SingleCompartment` and
    :class:`braincell.Cell`. Solvers receive a :class:`DiffEqModule`
    as their ``target`` argument and read ``t``/``dt`` from the active
    :mod:`brainstate.environ` context.

    See Also
    --------
    DiffEqState : Per-variable state container the solvers update.
    IndependentIntegration : Excludes a submodule from the main solver.
    """

    __module__ = 'braincell'

    def pre_integral(self, *args, **kwargs):
        """
        Perform any necessary operations before the integration step.

        This method can be overridden to implement custom pre-integration logic.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass

    def compute_derivative(self, *args, **kwargs):
        """
        Compute the derivative of the differential equation.

        This method must be implemented by subclasses to define the specific
        differential equation for the system.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        NotImplemented
            This method should be overridden in subclasses.

        Raises
        ------
        NotImplementedError
            If this method is not overridden in a subclass.
        """
        raise NotImplementedError

    def post_integral(self, *args, **kwargs):
        """
        Perform any necessary operations after the integration step.

        This method can be overridden to implement custom post-integration logic.

        Parameters
        ----------
        *args : tuple
            Variable length argument list.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        pass


class IndependentIntegration(brainstate.mixin.Mixin):
    """Mixin that opts a submodule out of its parent's integration loop.

    States owned by an :class:`IndependentIntegration` submodule are
    filtered out by :func:`braincell.quad._util.split_diffeq_states`, so
    they are *not* touched by whichever solver is driving the parent
    :class:`DiffEqModule`. The submodule then advances its own states by
    calling :meth:`make_integration`, which dispatches through whatever
    solver was named at construction time.

    This is the right tool when a sub-system needs a different time step
    or a fundamentally different solver from the rest of the cell — for
    example, fast voltage gating that should run with exponential Euler
    while the surrounding model uses RK4, or a calcium pool that prefers
    backward Euler.

    Parameters
    ----------
    solver : str or Callable
        Name of a registered integrator (canonical or alias) or a step
        function. Resolved through :func:`braincell.quad.get_integrator`,
        so unknown strings raise :class:`ValueError`.
    **kwargs
        Forwarded to other ``Mixin`` bases in the MRO.

    See Also
    --------
    DiffEqModule : Parent integration interface.
    braincell.quad.get_integrator : Solver lookup.

    Examples
    --------

    .. code-block:: python

        >>> from braincell import DiffEqModule, IndependentIntegration
        >>> class FastGate(IndependentIntegration, DiffEqModule):
        ...     def __init__(self):
        ...         super().__init__(solver='exp_euler')
        ...     def compute_derivative(self, *args):
        ...         ...                                    # doctest: +SKIP
    """

    def __init__(self, solver: str | Callable, **kwargs):
        from . import get_integrator

        self.solver = get_integrator(solver)
        super().__init__(**kwargs)

    def make_integration(self, *args, **kwargs):
        """Run one step of the configured solver on this submodule."""
        self.solver(self, *args, **kwargs)
