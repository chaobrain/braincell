# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

"""The Hodgkin-Huxley typed neuron base class.

``HHTypedNeuron`` lives in its own module so that
:mod:`braincell._base_ion` can name it as ``root_type`` with a plain
top-of-file import. The class and the ion classes once shared a module
and reached each other through bottom-of-file imports; that arrangement
worked only because of the exact order of statements across two files.
Keeping the class here makes the cycle structurally impossible rather
than merely avoided, which is what
``_base_neuron_test.BaseNeuronExportTest`` pins.
"""

import contextlib
from typing import Optional, Tuple

import brainpy
import brainunit as u
import jax
import jax.numpy as jnp

from braincell._typing import Size
from braincell._base_channel import IonChannel
from braincell._misc import Container, TreeNode, cast_like as _cast_like
from braincell.quad.protocol import DiffEqModule

__all__ = ["HHTypedNeuron"]

# The value inputs of the step each model has in progress, keyed by
# ``id(model)``: ``(table, key, value)`` for every one the step removes when
# it ends. Kept here rather than on the model so that graph traversal of the
# model never meets it.
_STEP_VALUE_INPUTS: dict[int, tuple] = {}


def run_state_lifecycle(method: str, entries, batch_size=None) -> None:
    """Run ``init_state`` or ``reset_state`` over a model's top-level nodes, ions first.

    Parameters
    ----------
    method : {"init_state", "reset_state"}
        The lifecycle hook to run.
    entries : iterable of (object, tuple)
        ``(node, args)`` per top-level node, in iteration order; ``args``
        are the node's positional arguments (the voltage it reads).
    batch_size : int, optional
        Forwarded to every hook.

    Notes
    -----
    Every ion's own state (its concentrations, for a dynamic ion) is
    allocated or reset before any channel's, and the channels then follow
    in the nodes' order. A channel that reads more than one ion (a K(Ca)
    channel under the potassium ion reads the calcium concentration) is a
    child of only one of them. When each ion ran its own state and then its
    children's in turn, that channel's ``init_state`` read the calcium ion
    before it had a ``Ci`` whenever the calcium ion's key sorted after the
    potassium ion's, so naming the calcium ion changed whether a cell
    initialized (MECH-51). An ion opts in through
    ``_splits_state_lifecycle``; any other node, or an ion whose class
    overrides the hook, runs its whole hook in the second pass.
    """
    entries = tuple(entries)
    split = tuple(bool(getattr(node, "_splits_state_lifecycle", lambda _: False)(method)) for node, _ in entries)
    for (node, args), is_split in zip(entries, split):
        if is_split:
            node._run_own_state_lifecycle(method, *args, batch_size=batch_size)
    for (node, args), is_split in zip(entries, split):
        if is_split:
            node._run_child_state_lifecycle(method, *args, batch_size=batch_size)
        else:
            getattr(node, method)(*args, batch_size=batch_size)


def _first_step_only(value, first):
    """``value`` where the traced flag ``first`` is true, zero elsewhere (units kept)."""
    return jax.tree.map(lambda leaf: jnp.where(first, leaf, jnp.zeros_like(leaf)), value)


def _zero_spike_like(V):
    """Return a zero spike buffer matching a membrane-voltage value.

    Parameters
    ----------
    V : array-like or Quantity
        Membrane voltage value whose shape should be mirrored.

    Returns
    -------
    array-like
        Unitless zero-valued spike buffer with the same shape as ``V``.

    Notes
    -----
    ``init_state`` and ``reset_state`` seed ``last_V`` and ``next_V`` with
    the same value, so no threshold crossing can occur.  Building the zero
    buffer directly avoids tracing the surrogate spike function during these
    lifecycle phases.
    """
    return u.math.zeros_like(u.get_magnitude(V))


def _threshold_crossing(last_V, next_V, threshold, spk_fun, *, direction="rising"):
    """Detect arrival at a threshold with matching forward/JVP/VJP rules."""
    sign = 1.0 if direction == "rising" else -1.0
    denom = _cast_like(20.0 * u.mV, next_V)
    threshold = _cast_like(threshold, next_V)
    old = sign * (last_V - threshold) / denom
    new = sign * (next_V - threshold) / denom
    return spk_fun(new) * (1.0 - spk_fun(old))


class HHTypedNeuron(brainpy.state.Dynamics, Container, DiffEqModule):
    """Base class for Hodgkin-Huxley typed neuronal membrane dynamics.

    Subclasses own the membrane voltage ``V`` and the integration scheme;
    this class owns the ion-channel container and the lifecycle calls that
    fan out to it.

    Parameters
    ----------
    size : brainstate.typing.Size
        The size of the simulation target: an int, or a tuple/list of ints
        read as ``(..., n_neuron, n_compartment)``.
    name : str, optional
        The name of the instance. A default is generated when omitted.
    **ion_channels
        Ion channel instances to add to the neuron, keyed by attribute name.

    Raises
    ------
    ValueError
        If ``size`` is neither an int nor a non-empty tuple/list of ints.
    """

    __module__ = 'braincell'
    _container_name = 'ion_channels'

    def __init__(self, size: Size, name: Optional[str] = None, **ion_channels):
        super().__init__(size, name=name)

        # attribute for ``Container``
        self.ion_channels = self._format_elements(IonChannel, **ion_channels)

    @property
    def pop_size(self) -> Tuple[int, ...]:
        """Neuron-population shape, that is ``size`` without the compartment axis.

        Returns
        -------
        tuple of int
            The population size in each dimension: ``(100, 50)`` is a 2-D
            population of 100 by 50 neurons.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    @property
    def n_compartment(self) -> int:
        """Number of compartments per neuron.

        Returns
        -------
        int
            How many compartments -- soma, dendrite, axon sections -- each
            neuron in the group is divided into.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    def current(self, *args, **kwargs):
        """Sum the membrane current contributed by every ion channel.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Returns
        -------
        brainunit.Quantity
            Total membrane current, in the subclass's current units.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def pre_integral(self, *args, **kwargs):
        """Run whatever must happen before each integration step.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError

    def compute_derivative(self, *args, **kwargs):
        """Fill in the derivative of every state this neuron integrates.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.

        Raises
        ------
        NotImplementedError
            Always, in the base class. Subclasses must override.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def post_integral(self, *args, **kwargs):
        """Run whatever must happen after each integration step.

        Does nothing by default. Subclasses override it to publish spikes,
        clamp concentrations, or otherwise fix up state the solver wrote.

        Parameters
        ----------
        *args
            Positional arguments defined by the subclass.
        **kwargs
            Keyword arguments defined by the subclass.
        """

    def init_state(self, batch_size=None):
        """Allocate the state of every direct ion-channel child.

        Parameters
        ----------
        batch_size : int, optional
            Leading batch dimension to allocate channel states with.

        Raises
        ------
        TypeError
            If a child channel's ``root_type`` does not accept this neuron.

        Notes
        -----
        Channels are seeded from the current membrane potential
        ``self.V.value``, so ``V`` must already be initialized.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        TreeNode.check_hierarchies(self.__class__, *nodes)
        run_state_lifecycle("init_state", ((channel, (self.V.value,)) for channel in nodes), batch_size)

    def reset_state(self, batch_size=None):
        """Return every direct ion-channel child to its initial state.

        Parameters
        ----------
        batch_size : int, optional
            Leading batch dimension to reset channel states with.

        Notes
        -----
        Unlike :meth:`init_state` this does not re-check hierarchies: the
        state already exists, so the check has already run.
        """
        nodes = self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values()
        run_state_lifecycle("reset_state", ((channel, (self.V.value,)) for channel in nodes), batch_size)

    def add(self, **elements):
        """Add ion channels to the neuron after construction.

        Parameters
        ----------
        **elements
            Ion channel instances to add, keyed by attribute name.

        Raises
        ------
        TypeError
            If an element's ``root_type`` does not accept this neuron, or if
            it is not an :class:`~braincell.IonChannel`.
        """
        TreeNode.check_hierarchies(type(self), **elements)
        self.ion_channels.update(self._format_elements(IonChannel, **elements))

    _INPUT_TABLES = ("_current_inputs", "_delta_inputs")

    def _add_to_input_table(self, index: int, key, item) -> None:
        """Put ``item`` under ``key`` in the current (0) or delta (1) input table."""
        name = self._INPUT_TABLES[index]
        if getattr(self, name) is None:
            setattr(self, name, {})
        getattr(self, name)[key] = item

    def _take_value_inputs(self) -> tuple:
        """Remove the value (non-callable) inputs from the input tables and return them.

        A compiled loop takes them before it runs and passes them in, so
        that they are an input of the loop rather than constants of its
        trace, and :meth:`_value_input_step` applies them to its first step.

        Returns
        -------
        tuple of dict
            ``(current, delta)``, each mapping an input key to its value.
        """
        taken = []
        for name in self._INPUT_TABLES:
            table = getattr(self, name) or {}
            values = {key: item for key, item in table.items() if not callable(item)}
            for key in values:
                del table[key]
            taken.append(values)
        return tuple(taken)

    def _restore_value_inputs(self, values) -> None:
        """Put back the inputs :meth:`_take_value_inputs` took, for a loop that failed."""
        for index, items in enumerate(values):
            for key, item in items.items():
                if key not in (getattr(self, self._INPUT_TABLES[index]) or {}):
                    self._add_to_input_table(index, key, item)

    def _in_value_input_step(self) -> bool:
        """Whether :meth:`_value_input_step` is in progress on this model."""
        return id(self) in _STEP_VALUE_INPUTS

    @contextlib.contextmanager
    def _value_input_step(self, values=None, *, first=None):
        """Run one step that consumes the value inputs registered for it.

        brainpy removes a value (non-callable) input the first time it is
        summed, so a solver with several stages used to count it in its
        first stage only. Inside this block the sums keep the value inputs,
        so every stage of the step -- and every recording sampled in it --
        reads the same snapshot, and the block's end removes the values that
        were registered when it began. Callable inputs are unaffected. A
        block opened inside another one on the same model does nothing: the
        outer block is the step.

        Parameters
        ----------
        values : tuple of dict, optional
            Inputs taken by :meth:`_take_value_inputs` before a compiled
            loop, ``(current, delta)``; they are registered for this step.
        first : bool array, optional
            Traced flag of the loop's first step. ``values`` act where it is
            true and are zero elsewhere, so a loop applies them once.
        """
        if self._in_value_input_step():
            yield
            return
        for index, items in enumerate(values or ()):
            for key, item in items.items():
                self._add_to_input_table(index, key, item if first is None else _first_step_only(item, first))
        tables = tuple(getattr(self, name) or {} for name in self._INPUT_TABLES)
        _STEP_VALUE_INPUTS[id(self)] = tuple(
            (table, key, item) for table in tables for key, item in table.items() if not callable(item)
        )
        try:
            yield
        finally:
            for table, key, item in _STEP_VALUE_INPUTS.pop(id(self)):
                if table.get(key) is item:
                    del table[key]

    def sum_current_inputs(self, init, *args, label=None, pop=True, **kwargs):
        """Add the current inputs to ``init`` (brainpy's ``sum_current_inputs``).

        A callable input is called with ``*args`` and ``**kwargs``; a value
        input is added as it is. Outside a step a value is removed once
        summed, as in brainpy; within a step (``update``) every stage
        adds it and the step removes it when it ends.

        Parameters
        ----------
        init : object
            The value the inputs are added to.
        *args, **kwargs
            Passed to every callable input.
        label : str, optional
            Sum only the inputs registered with this label.
        pop : bool, default True
            Remove each summed value input; ignored within a step.

        Returns
        -------
        object
            ``init`` plus the inputs.
        """
        pop = pop and not self._in_value_input_step()
        return super().sum_current_inputs(init, *args, label=label, pop=pop, **kwargs)

    def sum_delta_inputs(self, init, *args, label=None, pop=True, **kwargs):
        """Add the delta inputs to ``init`` (brainpy's ``sum_delta_inputs``).

        The value rule of :meth:`sum_current_inputs` applies: within a step
        a value input is kept until the step ends.

        Parameters
        ----------
        init : object
            The value the inputs are added to.
        *args, **kwargs
            Passed to every callable input.
        label : str, optional
            Sum only the inputs registered with this label.
        pop : bool, default True
            Remove each summed value input; ignored within a step.

        Returns
        -------
        object
            ``init`` plus the inputs.
        """
        pop = pop and not self._in_value_input_step()
        return super().sum_delta_inputs(init, *args, label=label, pop=pop, **kwargs)

    def get_spike(self, last_V, next_V):
        """Surrogate-gradient spike indicator at the ``V_th`` crossing.

        Uses ``self.V_th`` (threshold voltage) and ``self.spk_fun``
        (surrogate-gradient callable) supplied by subclasses. The
        product of rising- and falling-crossing terms produces a
        non-zero value only when ``last_V < V_th <= next_V``.
        """
        return _threshold_crossing(last_V, next_V, self.V_th, self.spk_fun)
