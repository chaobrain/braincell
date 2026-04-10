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

from typing import Callable

import brainstate
from brainstate._state import record_state_value_write

__all__ = [
    'DiffEqState',
    'DiffEqModule',
    'IndependentIntegration',
]


class DiffEqState(brainstate.HiddenState):
    """A :mod:`brainstate` state that participates in numerical integration.

    A :class:`DiffEqState` is the unit of work consumed by every solver in
    :mod:`braincell.quad`. It extends :class:`brainstate.HiddenState` with
    two extra slots — ``derivative`` and ``diffusion`` — that the
    surrounding solver writes during one ODE/SDE step:

    - ``derivative`` is the right-hand side :math:`f(t, y)` for an ODE
      :math:`\\dot y = f(t, y)`, or the *drift* term for an SDE
      :math:`dy = f(t, y)\\,dt + g(t, y)\\,dW`.
    - ``diffusion`` is the SDE noise coefficient :math:`g(t, y)`. It
      stays ``None`` for plain ODE systems.

    Solver step functions (``*_step``) read the current ``value`` of every
    :class:`DiffEqState` in a :class:`DiffEqModule`, call
    :meth:`DiffEqModule.compute_derivative` to populate ``derivative``
    (and optionally ``diffusion``), and then write the integrated result
    back into ``value``.

    Both setters call :func:`brainstate._state.record_state_value_write`
    so that any active state-trace stack picks up the assignment — the
    Runge-Kutta and exponential-Euler drivers rely on this to discover
    which states actually participate in the integration.

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
    DiffEqModule : Container that owns and updates :class:`DiffEqState`
        instances.
    IndependentIntegration : Mixin that excludes a submodule's states
        from the main integrator.
    """

    __module__ = 'braincell'

    derivative: brainstate.typing.PyTree
    diffusion: brainstate.typing.PyTree

    def __init__(self, *args, **kwargs):
        """
        Initialize the DiffEqState.

        Parameters
        ----------
        *args : Any
            Variable length argument list to be passed to the parent class constructor.
        **kwargs : Any
            Arbitrary keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(*args, **kwargs)
        self._derivative = None
        self._diffusion = None

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
    :class:`braincell.cell.Cell`. Solvers receive a :class:`DiffEqModule`
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

        >>> from braincell.quad import DiffEqModule, IndependentIntegration
        >>> class FastGate(IndependentIntegration, DiffEqModule):
        ...     def __init__(self):
        ...         super().__init__(solver='exp_euler')
        ...     def compute_derivative(self, *args):
        ...         ...                                    # doctest: +SKIP
    """

    def __init__(self, solver: str | Callable, **kwargs):
        from . import get_integrator
        self.solver = get_integrator(solver)

    def make_integration(self, *args, **kwargs):
        """Run one step of the configured solver on this submodule."""
        self.solver(self, *args, **kwargs)
