# -*- coding: utf-8 -*-
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


"""
This module implements leakage channel.

"""

from typing import Union, Sequence, Optional

import braintools
import brainunit as u

from braincell._base import HHTypedNeuron, Channel
from braincell._typing import Initializer
from braincell.mech import register_channel

__all__ = [
    'LeakageChannel',
    'IL',
]


class LeakageChannel(Channel):
    """Base class for leakage channel dynamics.

    A leak conductance is voltage-independent: it has no gating variables
    and no state to integrate, so every lifecycle hook below is a no-op
    (or, for :meth:`current`, an obligation for the subclass to fill in).
    Subclass this directly to model a passive, always-open conductance
    such as the resting leak of a compartment. Use an ion-specific
    channel template instead -- :class:`~braincell.channel._base.HH` or
    :class:`~braincell.channel._base.Markov` -- when the conductance
    depends on voltage, time, or an ion concentration.

    See Also
    --------
    IL : Concrete leak channel with a fixed conductance and reversal
        potential.

    Notes
    -----
    ``root_type`` is :class:`~braincell._base.HHTypedNeuron`: instances
    are attached to a Hodgkin-Huxley-typed neuron, not to an ion. There
    is no equation here -- :meth:`current` raises ``NotImplementedError``
    and must be implemented by a subclass such as :class:`IL`.
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


@register_channel("IL", aliases=("leaky",))
class IL(LeakageChannel):
    r"""The leakage channel current.

    A generic, fixed-conductance leak: :attr:`g_max` and :attr:`E` are
    constants set at construction and never change with voltage, time,
    or concentration. Use it for the passive resting leak of a
    compartment, or as a placeholder before an ion-specific channel is
    modeled.

    Parameters
    ----------
    size : int or sequence of int
        Shape of the simulation target this channel is attached to.
    g_max : ArrayLike or Callable, optional
        Leakage conductance density. Default is ``0.1 mS/cm2``.
    E : ArrayLike or Callable, optional
        Leak reversal potential. Default is ``-70.0 mV``.
    name : str, optional
        Instance name. Default is ``None``, in which case a name is
        auto-generated.

    See Also
    --------
    LeakageChannel : Abstract base this class implements.

    Notes
    -----
    The current follows Ohm's law with a constant driving force:

    .. math::

        I = g_{max} \cdot (E - V)

    Registered with :func:`~braincell.mech.register_channel` under the
    name ``"IL"``, with ``"leaky"`` as an alias.
    """

    __module__ = 'braincell.channel'
    root_type = HHTypedNeuron

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        g_max: Initializer = 0.1 * (u.mS / u.cm**2),
        E: Initializer = -70.0 * u.mV,
        name: Optional[str] = None,
    ):
        super().__init__(
            size=size,
            name=name,
        )

        self.E = braintools.init.param(E, self.varshape, allow_none=False)
        self.g_max = braintools.init.param(g_max, self.varshape, allow_none=False)

    def current(self, V):
        return self.g_max * (self.E - V)
