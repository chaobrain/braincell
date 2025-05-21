# -*- coding: utf-8 -*-

"""
This module implements leakage channel.

"""

from typing import Union, Callable, Sequence, Optional

import brainstate
import brainunit as u

from braincell._base import HHTypedNeuron, Channel

__all__ = [
    'LeakageChannel',
    'IL',
]


class LeakageChannel(Channel):
    """
    Base class for leakage channel dynamics.
    """
    __module__ = 'braincell.channel'

    root_type = HHTypedNeuron

    def pre_integral(self, V):
        """
        Perform any necessary operations before the integration step.

        Parameters
        -----------
        V : array-like
            The membrane potential.
        """
        pass

    def post_integral(self, V):
        """
        Perform any necessary operations after the integration step.

        Parameters
        -----------
        V : array-like
            The membrane potential.
        """
        pass

    def compute_derivative(self, V):
        """
        Compute the derivative of the channel state variables.

        Parameters
        -----------
        V : array-like
            The membrane potential.
        """
        pass

    def current(self, V):
        """
        Calculate the current through the leakage channel.

        Parameters
        -----------
        V : array-like
            The membrane potential.

        Raises:
        -------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def init_state(self, V, batch_size: int = None):
        """
        Initialize the state of the leakage channel.

        Parameters
        -----------
        V : array-like
            The membrane potential.
        batch_size : int, optional
            The batch size for initialization.
        """
        pass

    def reset_state(self, V, batch_size: int = None):
        """
        Reset the state of the leakage channel.

        Parameters
        -----------
        V : array-like
            The membrane potential.
        batch_size : int, optional
            The batch size for resetting.
        """
        pass


class IL(LeakageChannel):
    """The leakage channel current.

    Parameters
    ----------
    g_max : float
      The leakage conductance.
    E : float
      The reversal potential.
    """
    __module__ = 'braincell.channel'
    root_type = HHTypedNeuron

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Union[brainstate.typing.ArrayLike, Callable] = 0.1 * (u.mS / u.cm ** 2),
        E: Union[brainstate.typing.ArrayLike, Callable] = -70. * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(size=size, name=name, )

        self.E = brainstate.init.param(E, self.varshape, allow_none=False)
        self.g_max = brainstate.init.param(g_max, self.varshape, allow_none=False)

    def current(self, V):
        return self.g_max * (self.E - V)
