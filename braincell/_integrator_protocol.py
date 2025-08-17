# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


class DiffEqState(brainstate.ShortTermState):
    """
    A state that integrates the state of the system to the integral of the state.

    This class represents a differential equation state, which can be used for both
    Ordinary Differential Equations (ODE) and Stochastic Differential Equations (SDE).
    It provides properties for the derivative and diffusion of the state.

    Attributes
    ----------
    derivative : brainstate.typing.PyTree
        The derivative of the differential equation state.
    diffusion : brainstate.typing.PyTree
        The diffusion of the differential equation state.

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


class DiffEqModule(brainstate.mixin.Mixin):
    """
    A mixin class that provides differential equation functionality.

    This class serves as a mixin to add differential equation capabilities to other classes.
    It defines the core interface for implementing ordinary differential equations (ODEs)
    and stochastic differential equations (SDEs).

    The class includes methods for pre-integration preparation, derivative computation,
    and post-integration processing. Subclasses must implement the compute_derivative
    method to define the specific differential equation for the system.
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
    """
    Mixin class to indicate independent integration of module states.

    This class serves as a marker for modules whose states should be excluded from
    the main integration process and instead be integrated independently. When a
    module inherits from `IndependentIntegration`, its states are not included in
    the set of states to be integrated by the primary numerical integrator.

    This is useful in scenarios where certain subsystems require specialized or
    decoupled integration strategies, such as different time steps or custom solvers.

    Usage of this mixin allows the integration framework to identify and handle
    such modules separately, ensuring modularity and flexibility in the integration
    of complex systems.

    Examples
    --------
    >>> class MySubsystem(IndependentIntegration, DiffEqModule):
    ...     pass
    >>> # States in MySubsystem will be integrated independently.

    Notes
    -----
    - This class does not implement any additional methods or properties.
    - It is intended to be used as a mixin alongside other module base classes.
    """

    def __init__(self, solver: str | Callable, **kwargs):
        from ._integrator import get_integrator
        self.solver = get_integrator(solver)

    def make_integration(self, *args, **kwargs):
        print('Updating IndependentIntegration with solver:', self.solver)
        self.solver(self, *args, **kwargs)
