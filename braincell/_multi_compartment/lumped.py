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

"""Lumped runtime of a :class:`~braincell.Cell` with ``cv_policy=Lumped()``.

A lumped Cell keeps the cable Cell's declaration phase (paint / place rules,
the one-CV discretization, ``cvs``) and replaces only the runtime. At
``init_state`` this module lowers the single CV into:

- :class:`LumpedRuntime`, a point-neuron engine with the voltage, derivative
  and update of :class:`~braincell.SingleCompartment`, whose mechanisms are
  instantiated with ``size=pop_size`` -- there is no CV axis;
- :class:`LumpedClamps`, which evaluates the placed current clamps with the
  same helpers as the cable runtime and returns their total current.

Parameters are resolved on the one CV by the cable rules (callables of
:class:`~braincell.mech.CVContext` evaluated at the CV, ion initializers
sampled once) and checked against the same unit prototypes, with the same
error messages, as on a cable Cell. Channels are bound to ions by the same
code as the cable runtime (``root_type``, ``ion_name`` / ``ion_names``,
placeholder ions).

External input (CELL-23) reaches the runtime as one current density per
member: :func:`as_current_density` turns a current into a density over the CV
area, and :meth:`LumpedRuntime.sum_current_inputs` applies the same rule to
the current inputs registered on the owning Cell (brainpy projections), at
every solver stage.

At ``init_state`` an explicit Cell solver is checked against the HH gates it
integrates (:func:`warn_stiff_gates`): a gate that relaxes faster than the
solver's stability bound turns every practical ``dt`` into NaN, so the Cell
warns, naming the channel and the solvers that are stable for it.
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, NamedTuple

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._compute.bindings import (
    _instantiate_runtime_node,
    apply_subsolver_schedule,
    install_channel_on_ions,
)
from braincell._compute.bridge import _ION_GEOMETRY_ATTRS, broadcast_to_shape
from braincell._compute.ions import (
    _build_ion_alias_map,
    _ion_runtime_attr_name,
    _placeholder_family_keys,
    _runtime_ion_species_key,
    _supported_ion_runtime_params,
)
from braincell._compute.layouts import (
    _allocate_current_clamp_buffer,
    _allocate_current_clamp_delay_buffer,
    _allocate_state_buffer,
    _eval_function_clamp,
    _eval_sine_clamp,
    _mechanism_var_names,
    _spatial_density_error_prefix,
    _time_ge,
    _time_lt,
)
from braincell._compute.parameters import (
    RuntimeParameterState,
    set_runtime_parameter_value,
    density_error_context,
    density_parameter_names,
    density_parameter_schema,
    density_parameter_value,
    spatial_callable_source,
)
from braincell._compute.state import CellRuntimeState
from braincell._misc import is_traced_value
from braincell._single_compartment import SingleCompartment
from braincell.channel._base import HH
from braincell.ion import build_placeholder_ions
from braincell.mech import (
    CurrentClamp,
    Density,
    FunctionClamp,
    SineClamp,
    Synapse as SynapsePlacement,
    get_registry,
)
from braincell.quad import get_integrator, integrator_info, list_integrators
from braincell.quad.protocol import IndependentIntegration, state_grouping

if TYPE_CHECKING:
    from braincell._multi_compartment.cell import Cell

__all__ = [
    "CLAMP_TYPES",
    "CURRENT_DENSITY_UNIT",
    "LumpedChannel",
    "LumpedClamps",
    "LumpedRuntime",
    "StiffGate",
    "as_current_density",
    "build_lumped_runtime",
    "stiff_gates",
    "warn_stiff_gates",
    "with_delta_inputs",
]

#: Point mechanisms a lumped Cell accepts in ``place``.
CLAMP_TYPES = (CurrentClamp, SineClamp, FunctionClamp)

#: Solvers that need the cable point tree and so cannot step a lumped Cell.
CABLE_ONLY_SOLVERS = frozenset({"staggered", "dhs_voltage"})

#: Unit of the current density a lumped Cell hands its runtime (clamps and
#: external input).
CURRENT_DENSITY_UNIT = u.uA / u.cm**2
_CURRENT_DIM = u.get_dim(u.amp)
_CURRENT_DENSITY_DIM = u.get_dim(u.amp / u.meter**2)

#: Voltages (mV) over which :func:`stiff_gates` scans a gate's relaxation rate:
#: -100 to 60 mV in 0.25 mV steps, the grid of the channel catalogue's stiffness table.
STIFFNESS_GRID_MV = np.linspace(-100.0, 60.0, 641)
#: The ``dt`` (ms) the check assumes: NEURON's default and the one the docs use.
STIFFNESS_DT_MS = 0.025
#: Largest ``dt * rate`` an explicit Runge-Kutta step stays stable at on a
#: decaying mode: 2 for Euler and the two-stage methods, 2.51 for three stages,
#: 2.79 for four; the check uses the smallest.
EXPLICIT_STABILITY_LIMIT = 2.0
#: Members times grid points up to which every member meets every grid voltage
#: (817 members); see :func:`stiff_gates`. The voltage grid then broadcasts
#: against the members, so most of the arithmetic runs on the 641 voltages only.
_FULL_GRID_ELEMENTS = 1 << 19
#: Voltages evaluated per gate for a larger population, where each one costs
#: full-size arithmetic; see :func:`stiff_gates`.
_ROUND_ELEMENTS = 1 << 17


def as_current_density(value, *, area, shape, what: str):
    """Return an external input of a lumped Cell as a current density.

    Parameters
    ----------
    value : Quantity
        A current (e.g. nA), divided by ``area``, or a current density
        (e.g. uA/cm^2), taken as is.
    area : Quantity
        Membrane area of the Cell's one CV.
    shape : tuple of int
        Voltage shape, ``(batch,) + pop_size``; ``value`` is broadcast to it.
    what : str
        Name of the input in error messages.

    Returns
    -------
    Quantity
        The density in :data:`CURRENT_DENSITY_UNIT`, shaped ``shape``.

    Raises
    ------
    UnitMismatchError
        If ``value`` is neither a current nor a current density (a bare
        number included).
    ValueError
        If ``value`` does not broadcast to ``shape``.
    """
    dim = u.get_dim(value)
    if dim not in (_CURRENT_DIM, _CURRENT_DENSITY_DIM):
        unit = u.get_unit(value)
        raise u.UnitMismatchError(
            f"{what} on a lumped Cell must be a current (e.g. nA) or a current density "
            f"(e.g. uA/cm^2); got a value {'without units' if unit.is_unitless else f'in {unit}'}."
        )
    # Checked on ``value``, before dividing by ``area`` (shaped ``pop_size``).
    value_shape = tuple(u.math.shape(value))
    try:
        broadcasts = np.broadcast_shapes(value_shape, tuple(shape)) == tuple(shape)
    except ValueError:
        broadcasts = False
    if not broadcasts:
        raise ValueError(
            f"{what} has shape {value_shape}, which does not broadcast to the "
            f"voltage shape {tuple(shape)} ((batch,) + pop_size) of the lumped Cell."
        )
    density = value / area if dim == _CURRENT_DIM else value
    density = u.Quantity(density).in_unit(CURRENT_DENSITY_UNIT)
    return u.math.broadcast_to(density, shape)


def with_delta_inputs(owner, V, *args, **kwargs):
    """Return ``V`` plus the delta inputs registered on ``owner``.

    ``owner.add_delta_input`` (the entry of brainpy's ``DeltaProj``)
    registers a voltage increment; brainpy's rule applies: a callable is
    called with no arguments at every step, a value is added once and
    removed.

    Parameters
    ----------
    owner : Cell
        The Cell holding the inputs.
    V : Quantity
        The voltage the inputs are added to, ``owner.V.value``.
    *args, **kwargs
        Passed on to ``owner.sum_delta_inputs`` (brainpy's ``label``, and
        arguments for the callables).

    Returns
    -------
    Quantity
        ``V`` itself when no input is registered, otherwise ``V`` plus every
        input.

    Raises
    ------
    ValueError
        If the inputs do not broadcast to the shape of ``V``, which the sum
        would change; the message names the inputs and the shape.
    """
    keys = tuple(owner.delta_inputs or ())
    if not keys:
        return V
    total = owner.sum_delta_inputs(V, *args, **kwargs)
    shape = tuple(u.math.shape(V))
    if tuple(u.math.shape(total)) != shape:
        raise ValueError(
            f"Delta inputs {keys!r} do not broadcast to the voltage shape {shape}: their sum has shape "
            f"{tuple(u.math.shape(total))}. V is shaped (batch,) + pop_size on a lumped Cell and "
            "(batch,) + pop_size + (n_cv,) on a cable Cell, where one value per member has shape pop_size + (1,)."
        )
    return total


class LumpedRuntime(SingleCompartment):
    """Point-neuron engine of a lumped Cell.

    It is :class:`~braincell.SingleCompartment` with the membrane area of the
    Cell's one CV in place of ``2 pi radius length``. The Cell calls
    :meth:`update` with the clamp current density, plus any ``I_ext`` given
    to ``Cell.update``, as ``I_ext``; the current inputs registered on the
    Cell are summed by :meth:`sum_current_inputs`.

    Parameters
    ----------
    size : tuple of int
        The Cell's ``pop_size``.
    area : Quantity
        Membrane area of the CV.
    C : Quantity
        Specific membrane capacitance of the CV.
    V_th : Quantity
        Spike threshold, scalar or shaped ``size``.
    V_initializer : Quantity
        Initial voltage shaped ``size``.
    spk_fun : callable
        Surrogate spike function.
    solver : callable
        Step function from :func:`braincell.quad.get_integrator`.
    ion_channels : dict
        Ions and root-level channels, keyed by attribute name.
    owner : Cell, optional
        The Cell whose current inputs (``add_current_input``) drive this
        runtime. Held by weak reference: the Cell owns the runtime.
    """

    __module__ = "braincell._multi_compartment.lumped"

    def __init__(
        self, size, *, area, C, V_th, V_initializer, spk_fun, solver, ion_channels, owner=None, synapses=(), n_point=1
    ):
        super().__init__(
            size,
            C=C,
            V_th=V_th,
            V_initializer=V_initializer,
            spk_fun=spk_fun,
            solver=solver,
            **ion_channels,
        )
        self._cv_area = area
        self._owner_ref = None if owner is None else weakref.ref(owner)
        # Point synapses live one level down, so the SingleCompartment hooks
        # (which call every direct channel with ``V``) do not reach them; this
        # runtime calls them with the voltage of each synapse row instead.
        self._synapse_layouts = tuple(layout for layout, _ in synapses)
        self._n_point = int(n_point)
        self.synapse_nodes = _LumpedSynapses(synapses) if synapses else None

    def _layout_runtime(self):
        owner = None if self._owner_ref is None else self._owner_ref()
        return None if owner is None else owner._runtime

    @property
    def area(self):
        """Membrane area of the Cell's one CV, from the geometry View's current values."""
        runtime = self._layout_runtime()
        if runtime is None or runtime.geometry is None:
            return self._cv_area
        return runtime.current_cable().area[..., 0]

    @property
    def C(self):
        """Specific membrane capacitance, from the geometry View's ``cm``."""
        runtime = self._layout_runtime()
        if runtime is None or runtime.geometry is None:
            return self._C
        return runtime.geometry["cm"].value[..., 0]

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def V_th(self):
        """Spike threshold: the owning Cell's ``V_th`` (``CellView.set``, trainable)."""
        owner = None if self._owner_ref is None else self._owner_ref()
        parameter = None if owner is None else getattr(owner, "_V_th_parameter", None)
        return self._V_th if parameter is None else parameter.value

    @V_th.setter
    def V_th(self, value):
        self._V_th = value

    @property
    def V_initializer(self):
        """Initial voltage: the owning Cell's ``V_init`` (``CellView.set``)."""
        owner = None if self._owner_ref is None else self._owner_ref()
        parameter = None if owner is None else getattr(owner, "_V_init_materialized", None)
        return self._V_initializer if parameter is None else parameter.value

    @V_initializer.setter
    def V_initializer(self, value):
        self._V_initializer = value

    def _synapses(self):
        if self.synapse_nodes is None:
            return ()
        return tuple((layout, getattr(self.synapse_nodes, f"layout_{layout.id}")) for layout in self._synapse_layouts)

    def synapse_voltage(self, layout, V=None):
        """Voltage of each row of one synapse layout: its member's ``V``."""
        V = self.V.value if V is None else V
        return layout.gather_points(u.math.broadcast_to(V[..., None], u.math.shape(V) + (self._n_point,)))

    def _synapse_density(self, V):
        from braincell._multi_compartment.currents import synapse_current

        total = None
        for layout, node in self._synapses():
            rows = synapse_current(self._layout_runtime(), layout, node, self.synapse_voltage(layout, V))
            if rows is None:
                continue
            members = u.Quantity(
                jnp.zeros(u.math.shape(V) + (self._n_point,), dtype=u.get_mantissa(rows).dtype), u.get_unit(rows)
            )
            members = u.math.sum(layout.scatter_add_points(members, rows), axis=-1)
            total = members if total is None else total + members
        return None if total is None else (total / self.area).in_unit(CURRENT_DENSITY_UNIT)

    def _run_synapse_hook(self, hook: str):
        for layout, node in self._synapses():
            getattr(node, hook)(self.synapse_voltage(layout))

    def pre_integral(self, I_ext=0.0 * u.nA / u.cm**2):
        super().pre_integral(I_ext)
        self._run_synapse_hook("pre_integral")

    def compute_derivative(self, I_ext=0.0 * u.nA / u.cm**2):
        super().compute_derivative(I_ext)
        self._run_synapse_hook("compute_derivative")

    def post_integral(self, I_ext=0.0 * u.nA / u.cm**2):
        super().post_integral(I_ext)
        self._run_synapse_hook("post_integral")

    def init_state(self, batch_size=None):
        super().init_state(batch_size)
        with state_grouping(False):
            for layout, node in self._synapses():
                node.init_state(self.synapse_voltage(layout), batch_size=batch_size)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        with state_grouping(False):
            for layout, node in self._synapses():
                node.reset_state(self.synapse_voltage(layout), batch_size=batch_size)

    def sum_delta_inputs(self, init, *args, **kwargs):
        """Add the owning Cell's delta inputs (``add_delta_input``, brainpy ``DeltaProj``).

        :class:`~braincell.SingleCompartment`'s ``post_integral`` calls this
        at the end of every solver step, after the voltage and the mechanism
        states have advanced, so an input moves ``V`` for the spike test of
        that step and for the steps after it (:func:`with_delta_inputs`).
        """
        owner = None if self._owner_ref is None else self._owner_ref()
        if owner is None:
            return super().sum_delta_inputs(init, *args, **kwargs)
        return with_delta_inputs(owner, init, *args, **kwargs)

    def membrane_current(self, I_ext):
        """Total membrane current density per member, the terms of :meth:`compute_derivative`.

        Parameters
        ----------
        I_ext : Quantity
            The clamp (and external) current density of this step.

        Returns
        -------
        Quantity
            Channel, synapse, clamp and current-input density in nA/cm^2,
            shaped like ``V``; a cable Cell's ``observe.membrane_current()``.
        """
        V = self.V.value
        # A recording reads the value inputs without consuming them.
        total = self.sum_current_inputs(I_ext, V, pop=False)
        for channel in self._ion_channels().values():
            contribution = channel.current(V)
            if contribution is not None:
                total = total + contribution
        return u.Quantity(total).in_unit(u.nA / u.cm**2)

    def update(self, I_ext=0.0 * u.nA / u.cm**2):
        """One step of :class:`~braincell.SingleCompartment`.

        An ion whose concentration integrates its total current, and which
        carries an ion-reading synapse's current, reads a snapshot taken at
        the step-start voltage (the cable Cell's ``cache_ion_total_current``):
        its integrator must not differentiate through the synapse's states.
        """
        snapshot = []
        for node in self._ion_channels().values():
            if getattr(type(node), "uses_total_current", False) and any(
                getattr(fun, "requires_snapshot", None) is not None for fun in node.external_currents.values()
            ):
                node._cached_total_current = node.current(self.V.value, include_external=True)
                snapshot.append(node)
        try:
            return super().update(I_ext)
        finally:
            for node in snapshot:
                del node._cached_total_current

    def sum_current_inputs(self, init, *args, pop=True, **kwargs):
        """Add the owning Cell's current inputs, each as a current density.

        ``compute_derivative`` calls this at every solver stage with the
        stage voltage, as :class:`~braincell.SingleCompartment` does. Each
        input -- a brainpy projection's output such as ``COBA``, or any
        callable registered with ``Cell.add_current_input`` -- may return a
        current or a current density (:func:`as_current_density`). A
        non-callable input acts on one step: within the Cell's step every
        stage adds it and the step removes it; outside a step it is removed
        once added, as in brainpy.

        Parameters
        ----------
        init : Quantity
            The density the inputs are added to.
        *args, **kwargs
            Passed to each callable input; ``compute_derivative`` passes the
            voltage.
        pop : bool, default True
            Remove each value input once added; ignored within the Cell's
            step.

        Returns
        -------
        Quantity
            ``init`` plus every input, shaped like the voltage.

        Raises
        ------
        UnitMismatchError
            If an input is neither a current nor a current density; the
            message names its key.
        """
        if self.synapse_nodes is not None and args:
            synaptic = self._synapse_density(args[0])
            if synaptic is not None:
                init = init + synaptic
        owner = None if self._owner_ref is None else self._owner_ref()
        inputs = None if owner is None else owner.current_inputs
        if not inputs:
            return init
        pop = pop and not owner._in_value_input_step()
        shape = self.V.value.shape
        for key in tuple(inputs):
            item = inputs[key]
            value = item(*args, **kwargs) if callable(item) else item
            init = init + as_current_density(value, area=self.area, shape=shape, what=f"Current input {key!r}")
            if pop and not callable(item):
                del inputs[key]
        return init


class _LumpedSynapses(brainstate.nn.Module):
    """Container of a lumped runtime's synapse nodes, one per synapse type."""

    def __init__(self, synapses) -> None:
        super().__init__()
        for layout, node in synapses:
            setattr(self, f"layout_{layout.id}", node)


class LumpedChannel(NamedTuple):
    """A channel of a lumped runtime with the ions it reads.

    Kept outside the runtime module so that the node is reachable through one
    graph path only.
    """

    #: The declaration's name, quoted with its class when they differ.
    label: str
    #: The runtime channel node.
    node: object
    #: The runtime ions the channel reads, in its ``root_type`` order.
    ions: tuple


class StiffGate(NamedTuple):
    """An HH gate whose relaxation rate exceeds an explicit solver's stability bound."""

    #: :attr:`LumpedChannel.label` of the channel.
    channel: str
    #: Gate name.
    gate: str
    #: Largest relaxation rate over :data:`STIFFNESS_GRID_MV` and the population, in 1/ms.
    rate: float
    #: Voltage of that rate, in mV.
    V: float


def build_lumped_runtime(
    cell: "Cell", *, V_init, V_th
) -> tuple[LumpedRuntime, tuple[LumpedChannel, ...], CellRuntimeState]:
    """Lower the one CV of a lumped ``cell`` into a :class:`LumpedRuntime`.

    Parameters
    ----------
    cell : Cell
        A Cell whose ``cv_policy`` is :class:`~braincell.Lumped`.
    V_init, V_th : Quantity
        Materialized initial voltage and threshold, shaped ``cell.pop_size``.

    Returns
    -------
    runtime : LumpedRuntime
        Engine with mechanisms of size ``cell.pop_size``.
    channels : tuple of LumpedChannel
        Its channels in declaration order, with the ions each reads.
    state : CellRuntimeState
        Layouts and parameter states of its mechanisms.

    Raises
    ------
    ValueError
        On the same declaration errors as the cable runtime: unsupported ion
        parameters, ion instance names that clash, an ion binding that cannot
        be resolved, a callable parameter that does not return a scalar, a
        parameter whose units do not match its prototype.
    TypeError
        A parameter without units where the prototype has them, as on the
        cable runtime.
    """
    (cv,) = cell.cvs
    (context,) = cell.cv_contexts
    pop_size = tuple(cell.pop_size)
    registry = get_registry()

    ions: dict[str, object] = {}
    family_candidates: dict[str, list[str]] = {}
    class_candidates: dict[str, list[str]] = {}
    schedule_pairs: list[tuple[Density, object]] = []
    channel_declarations: list[Density] = []
    nodes: dict[int, object] = {}
    bound_ion_keys: dict[int, tuple[str, ...]] = {}

    # Discretization has already rejected an ion or channel painted twice on
    # the CV, so each instance name appears once here.
    for declaration in cv.density_mech:
        if declaration.category == "channel":
            channel_declarations.append(declaration)
            continue
        runtime_cls = registry.get("ion", declaration.class_name)
        species_key = _runtime_ion_species_key(runtime_cls)
        name = declaration.instance_name
        if name in _placeholder_family_keys() and name != species_key:
            raise ValueError(
                f"Ion instance name {name!r} conflicts with canonical family key for a different ion family."
            )
        invalid = set(declaration.params) - set(_supported_ion_runtime_params(runtime_cls))
        if invalid:
            raise ValueError(
                f"Ion instance {name!r} uses unsupported runtime ion params {sorted(invalid)!r} "
                f"on {runtime_cls.__name__!r}."
            )
        node = runtime_cls(size=pop_size, name=name, **_resolve_params(declaration, context, cell.paint_rules))
        nodes[id(declaration)] = node
        ions[name] = node
        family_candidates.setdefault(species_key, []).append(name)
        class_candidates.setdefault(runtime_cls.__name__, []).append(name)
        schedule_pairs.append((declaration, node))

    for family_key, default_ion in build_placeholder_ions(size=pop_size).items():
        if family_key in family_candidates:
            continue
        ions[family_key] = default_ion
        family_candidates[family_key] = [family_key]
        class_candidates.setdefault(type(default_ion).__name__, []).append(family_key)

    ion_aliases = _build_ion_alias_map(
        ions=ions,
        ion_family_candidates=family_candidates,
        ion_class_candidates=class_candidates,
    )
    frozen_candidates = {key: tuple(value) for key, value in family_candidates.items()}
    _attach_ion_geometry(ions, cv=cv, pop_size=pop_size)

    root_channels: dict[str, object] = {}
    channels: list[LumpedChannel] = []
    for index, declaration in enumerate(channel_declarations):
        runtime_cls = registry.get("channel", declaration.class_name)
        node = runtime_cls(size=pop_size, **_resolve_params(declaration, context, cell.paint_rules))
        nodes[id(declaration)] = node
        bound_keys, owner = install_channel_on_ions(
            node,
            runtime_cls=runtime_cls,
            mechanism=declaration,
            ions=ions,
            ion_aliases=ion_aliases,
            ion_family_candidates=frozen_candidates,
            layout_id=index,
        )
        bound_ion_keys[id(declaration)] = tuple(bound_keys)
        if owner is None:
            root_channels[_unique_key(root_channels, ions, declaration.instance_name)] = node
        schedule_pairs.append((declaration, node))
        label = declaration.class_name
        if declaration.instance_name != declaration.class_name:
            label = f"{declaration.instance_name!r} ({declaration.class_name})"
        channels.append(LumpedChannel(label, node, tuple(ions[key] for key in bound_keys)))

    apply_subsolver_schedule(schedule_pairs, solver=cell.subsolver, substeps=cell.substeps)

    state, synapses = _adapt_cable_layouts(
        cell,
        nodes=nodes,
        ions=ions,
        ion_aliases=ion_aliases,
        family_candidates=frozen_candidates,
        class_candidates=class_candidates,
        bound_ion_keys=bound_ion_keys,
    )

    runtime = LumpedRuntime(
        pop_size,
        area=cv.area,
        C=cv.cm,
        V_th=V_th,
        V_initializer=V_init,
        spk_fun=cell.spk_fun,
        solver=cell.solver,
        ion_channels={**ions, **root_channels},
        owner=cell,
        synapses=synapses,
        n_point=state.n_point,
    )
    return runtime, tuple(channels), state


def _adapt_cable_layouts(cell, *, nodes, ions, ion_aliases, family_candidates, class_candidates, bound_ion_keys):
    """Shape adapter: the single-CV cable layouts and buffers drive the lumped nodes.

    Buffers keep the cable shape ``pop_size + (1,)``; each lumped node reads a
    mirror state shaped ``pop_size`` that ``_sync_runtime_node_param`` refreshes
    from its buffer (``buffer[..., 0]``) after every write.
    """

    runtime = CellRuntimeState.from_cell(cell, build_nodes=False)
    pop_size = tuple(cell.pop_size)
    mirrors = {}
    synapses = []
    for layout in runtime.layouts:
        mechanism = runtime.layout_mechanisms[layout.id]
        if isinstance(mechanism, Density):
            node = nodes[id(mechanism)]
            runtime.runtime_nodes[layout.id] = node
            is_ion = mechanism.category == "ion"
            for (layout_id, name), buffer in runtime.state_buffers.items():
                if layout_id != layout.id or not isinstance(buffer, RuntimeParameterState):
                    continue
                attr = _ion_runtime_attr_name(type(node), name) if is_ion else name
                if is_ion and name.endswith("_initializer"):
                    mirrors[(layout.id, name)] = (buffer, _mirror_ion_initializer(node, attr, name, buffer, pop_size))
                    continue
                current = vars(node).get(attr)
                if (
                    current is None
                    or callable(current)
                    or isinstance(current, brainstate.State)
                    or type(current) in (bool, int)
                ):
                    continue
                if isinstance(density_parameter_value(mechanism, name), braintools.init.Initialization):
                    # The channel sampled this initializer (a cable Cell rejects it); the buffer takes its values.
                    set_runtime_parameter_value(buffer, u.math.broadcast_to(current, pop_size)[..., None])
                mirror = RuntimeParameterState(buffer.dense_value(masked=True)[..., 0], axis="row", full_shape=pop_size)
                setattr(node, attr, mirror)
                mirrors[(layout.id, name)] = (buffer, mirror)
        elif isinstance(mechanism, SynapsePlacement):
            node, keys, _ = _instantiate_runtime_node(
                layout=layout,
                mechanism=mechanism,
                state_buffers=runtime.state_buffers,
                ions=ions,
                ion_aliases=ion_aliases,
                ion_family_candidates=family_candidates,
                pop_size=pop_size,
            )
            runtime.runtime_nodes[layout.id] = node
            runtime.bound_ion_keys[layout.id] = tuple(keys)
            synapses.append((layout, node))
    for layout in runtime.layouts:
        mechanism = runtime.layout_mechanisms[layout.id]
        if isinstance(mechanism, Density) and mechanism.category == "channel":
            runtime.bound_ion_keys[layout.id] = bound_ion_keys[id(mechanism)]
    runtime.ions.update(ions)
    runtime.ion_aliases.update(ion_aliases)
    runtime.ion_family_candidates.update(family_candidates)
    runtime.ion_class_candidates.update({key: tuple(value) for key, value in class_candidates.items()})
    runtime.lumped_mirrors = mirrors
    return runtime, tuple(synapses)


def _mirror_ion_initializer(node, attr, name, buffer, pop_size):
    """Route an ion's ``*_initializer`` through a ``pop_size`` mirror of its buffer, as the cable ion does.

    The ion keeps its own initializer for every member until a view or a
    trainable writes the field: a write sets the member's entry of the
    ``pop_size + (1,)`` override mask (shared with the cable layout), and the
    member then starts from the mirror.
    """
    mirror = RuntimeParameterState(buffer.dense_value()[..., 0], axis="row", full_shape=pop_size)
    mask = brainstate.LongTermState(jnp.zeros(buffer.full_shape, dtype=bool))
    masks = vars(node).setdefault("_runtime_ion_initial_masks", {})
    masks[name] = mask
    default = getattr(node, attr)

    def initialize(shape):
        return u.math.where(mask.value[..., 0], mirror.value, braintools.init.param(default, shape))

    setattr(node, attr, initialize)
    return mirror


def _is_explicit_step(fn) -> bool:
    """Whether ``fn`` is the module step of a braintools explicit Runge-Kutta method (``'ode'`` kind)."""
    return any(fn is get_integrator(name) for name in list_integrators() if integrator_info(name).kind == "ode")


def stiff_gates(channels, *, shape: tuple[int, ...]) -> list[StiffGate]:
    """Return the HH gates that an explicit solver cannot step at :data:`STIFFNESS_DT_MS`.

    For each dependent HH channel (the main solver integrates its gates; a
    self-integrating channel advances in ``ind_update`` instead), the rate at
    which each gate relaxes is evaluated on :data:`STIFFNESS_GRID_MV` with the
    channel's own parameters and current ion state. A gate is stiff when
    ``STIFFNESS_DT_MS * rate > EXPLICIT_STABILITY_LIMIT`` somewhere on the grid.
    Channels without HH gates are not checked, nor is a channel whose rates
    cannot be evaluated on the grid (for example, rates that do not
    broadcast): the check is advisory and must not fail ``init_state``.

    Every member meets every grid voltage up to 817 members
    (:data:`_FULL_GRID_ELEMENTS`). A larger population is evaluated in
    ``R = max(1, _ROUND_ELEMENTS // members)`` rounds stacked on a leading
    axis, each placing the members on successive grid voltages: each round
    covers the whole grid, so every voltage is evaluated on about
    ``R * members / 641`` members, and each member at ``R`` voltages spread
    over the range. A population whose members share their parameters gets
    the exact maximum either way.

    Parameters
    ----------
    channels : sequence of LumpedChannel
        Channels of an initialized runtime.
    shape : tuple of int
        Shape of the voltage state (batch and population axes).

    Returns
    -------
    list of StiffGate
        In channel and gate declaration order.
    """
    grid = _voltage_grid(tuple(shape))
    found = []
    for channel in channels:
        node = channel.node
        if not isinstance(node, HH) or isinstance(node, IndependentIntegration):
            continue
        try:
            rates = _max_relaxation_rates(node, channel.ions, grid)
        except Exception:  # noqa: BLE001 - advisory check; see the docstring
            continue
        for gate, (rate, V) in rates.items():
            if STIFFNESS_DT_MS * rate > EXPLICIT_STABILITY_LIMIT:
                found.append(StiffGate(channel.label, gate, rate, V))
    return found


def _voltage_grid(shape: tuple[int, ...]) -> np.ndarray:
    """Voltages (mV) that broadcast with the members of ``shape``; see :func:`stiff_gates`."""
    n_grid = STIFFNESS_GRID_MV.size
    n_members = int(np.prod(shape, dtype=int))
    if n_members * n_grid <= _FULL_GRID_ELEMENTS:
        return STIFFNESS_GRID_MV.reshape((-1,) + (1,) * len(shape))
    # Here n_members > _FULL_GRID_ELEMENTS / n_grid > n_grid, so each round covers the whole grid.
    rounds = max(1, _ROUND_ELEMENTS // n_members)
    shifts = (np.arange(rounds) * n_grid // rounds).reshape((-1,) + (1,) * len(shape))
    return STIFFNESS_GRID_MV[(np.arange(n_members).reshape(shape) + shifts) % n_grid]


def _max_relaxation_rates(node, ions, grid) -> dict[str, tuple[float, float]]:
    """Largest relaxation rate (1/ms) of each gate of ``node`` over ``grid``, and its voltage (mV)."""
    infos = tuple(ion.pack_info() for ion in ions)
    best = {}
    for gate, rate in node._gate_relaxation_rates(u.Quantity(grid, u.mV), *infos).items():
        if is_traced_value(rate):
            raise TypeError("the rates are traced")
        values = np.asarray(u.Quantity(rate).to_decimal(u.ms**-1), dtype=float)
        values = np.broadcast_to(values, np.broadcast_shapes(values.shape, grid.shape))
        values = np.where(np.isfinite(values), values, -np.inf)
        index = int(np.argmax(values))
        best[gate] = (float(values.flat[index]), float(np.broadcast_to(grid, values.shape).flat[index]))
    return best


def warn_stiff_gates(runtime: LumpedRuntime, channels, *, solver, solver_name: str, stacklevel: int = 2) -> None:
    """Warn when ``solver`` is explicit and some HH gate of ``runtime`` is stiff for it.

    Parameters
    ----------
    runtime : LumpedRuntime
        An initialized runtime: the gates' rates read the ions' state.
    channels : sequence of LumpedChannel
        The runtime's channels, from :func:`build_lumped_runtime`.
    solver : callable
        The Cell's step function. Only the module steps of the explicit
        Runge-Kutta methods are checked; ``ind_exp_euler``, ``exp_euler``,
        ``backward_euler``, ``crank_nicolson`` and user callables are not.
    solver_name : str
        The Cell's ``solver_name``, quoted in the message.
    stacklevel : int
        As for :func:`warnings.warn` called by the caller: 2 attributes the
        warning to the caller's caller.

    Warns
    -----
    UserWarning
        Naming each stiff gate, its largest rate and the ``dt`` below which an
        explicit step is stable on it, and the stable solvers.
    """
    if not _is_explicit_step(solver):
        return
    found = stiff_gates(channels, shape=tuple(np.shape(u.get_mantissa(runtime.V.value))))
    if not found:
        return
    gates = "; ".join(
        f"{gate.channel} gate {gate.gate!r} relaxes at up to {gate.rate:.3g} /ms at {gate.V:g} mV "
        f"(stable for dt < {EXPLICIT_STABILITY_LIMIT / gate.rate:.2g} ms)"
        for gate in found
    )
    warnings.warn(
        f"Cell solver {solver_name!r} is an explicit Runge-Kutta method, and these HH gates are stiff for an "
        f"explicit solver at dt = {STIFFNESS_DT_MS} ms (rates scanned between {STIFFNESS_GRID_MV[0]:g} and "
        f"{STIFFNESS_GRID_MV[-1]:g} mV): {gates}. Larger steps grow without bound and the run returns NaN. "
        "Use solver='exp_euler' (second order on HH cells, about 20 times the cost per step) or "
        "solver='ind_exp_euler' (first order, about the cost of 'ralston2'); both are stable on these gates.",
        UserWarning,
        stacklevel=stacklevel + 1,
    )


def _resolve_params(declaration: Density, context, paint_rules) -> dict[str, object]:
    """Resolve and check declared parameters on the one CV, as the cable runtime does."""
    schema = density_parameter_schema(declaration)
    # The cable runtime checks the parameters it stores as numeric buffers;
    # configuration such as ``species_initializers`` is passed through.
    checked = set(density_parameter_names(declaration)) & set(schema)
    owner, source = density_error_context(declaration, paint_rules)
    params = {}
    for name in declaration.params:
        value = density_parameter_value(declaration, name)
        value_source = source
        if callable(value) and not isinstance(value, braintools.init.Initialization):
            value_source = spatial_callable_source(value) + (f"; {source}" if source else ".")
            value = _evaluate_spatial_param(declaration, name, value, context)
        # A Channel initializer is sampled by the runtime class itself.
        if name in checked and not isinstance(value, braintools.init.Initialization):
            schema[name].validate(value, name, owner=owner, source=value_source)
        params[name] = value
    return params


def _evaluate_spatial_param(declaration: Density, name: str, fn, context):
    """Evaluate a ``fn(CVContext)`` parameter at the CV and require a scalar."""
    prefix = _spatial_density_error_prefix(declaration, name, context)
    try:
        result = fn(context)
    except Exception as exc:
        raise ValueError(f"{prefix} callable raised {type(exc).__name__}: {exc}") from exc
    shape = tuple(getattr(result, "shape", ()))
    if shape not in ((), (1,)):
        raise TypeError(f"{prefix} must return a scalar, got shape {shape!r}.")
    if isinstance(result, u.Quantity):
        return u.math.reshape(result, ())
    try:
        return float(np.asarray(result, dtype=np.float64).reshape(()))
    except Exception as exc:
        raise TypeError(f"{prefix} must return a numeric scalar or a Quantity, got {result!r}.") from exc


def _attach_ion_geometry(ions: dict[str, object], *, cv, pop_size: tuple[int, ...]) -> None:
    """Give each ion the CV geometry (``area``, ``diam_mid``, ...) shaped ``pop_size``."""
    for ion in ions.values():
        for attr_name in _ION_GEOMETRY_ATTRS:
            setattr(ion, attr_name, broadcast_to_shape(getattr(cv, attr_name), pop_size, name=f"ion.{attr_name}"))


def _unique_key(taken: dict[str, object], ions: dict[str, object], name: str) -> str:
    if name not in taken and name not in ions:
        return name
    suffix = 2
    while f"{name}_{suffix}" in taken or f"{name}_{suffix}" in ions:
        suffix += 1
    return f"{name}_{suffix}"


class LumpedClamps:
    """Current clamps placed on a lumped Cell.

    Buffers are allocated by the cable runtime's clamp helpers on a one-row
    layout, so step, sine and function clamps keep their cable semantics
    (timing tolerance, delays, per-member values). Step-clamp start and end
    times are static, so they are tabulated once here rather than summed on
    every step, and the per-step work is elementwise.

    Parameters
    ----------
    cell : Cell
        A declared lumped Cell. Its clamp store fixes the order of the
        components returned by :meth:`evaluate`.
    """

    def __init__(self, cell: "Cell", *, density_unit, with_components: bool = True) -> None:
        self.pop_size = tuple(cell.pop_size)
        self.with_components = bool(with_components)
        # nA / cm^2 -> density_unit; the area is the geometry View's current one.
        self.density_unit = density_unit
        self._per_area = float((1.0 * u.nA / u.cm**2).to_decimal(density_unit))
        self.n_population = int(np.prod(self.pop_size, dtype=int))
        store = cell._get_clamp_store()
        self.n_component = len(store.id)
        self.state_buffers: dict[tuple[int, str], object] = {}
        self._entries: list[tuple[int, object]] = []
        # A population view places synapses only, so every clamp placement
        # covers every member. The clamp store numbers rows in placement
        # order, so concatenating per-placement currents gives store order.
        for placement in cell.point_placements:
            if not isinstance(placement.mechanism, CLAMP_TYPES):
                continue
            key = int(placement.id)
            self._allocate(key, placement.mechanism)
            self._entries.append((key, self._tabulate(key, placement.mechanism)))

    def __len__(self) -> int:
        return len(self._entries)

    def _allocate(self, key: int, mechanism) -> None:
        for var_name in _mechanism_var_names(mechanism):
            if isinstance(mechanism, CurrentClamp) and var_name == "delay":
                self.state_buffers[(key, var_name)] = _allocate_current_clamp_delay_buffer(
                    mechanism=mechanism, pop_size=self.pop_size, n_active=1
                )
            elif isinstance(mechanism, CurrentClamp) and var_name in ("durations", "amplitudes"):
                quantity, mask = _allocate_current_clamp_buffer(
                    mechanism=mechanism, var_name=var_name, pop_size=self.pop_size, n_active=1
                )
                self.state_buffers[(key, var_name)] = quantity
                self.state_buffers[(key, f"_mask_{var_name}")] = mask
            else:
                self.state_buffers[(key, var_name)] = _allocate_state_buffer(
                    mechanism, var_name=var_name, shape=self.pop_size + (1,)
                )

    def _tabulate(self, key: int, mechanism):
        """Static step table of a CurrentClamp (``_eval_current_clamp`` math)."""
        if not isinstance(mechanism, CurrentClamp):
            return type(mechanism).__name__
        durations = np.asarray(self.state_buffers[(key, "durations")].to_decimal(u.ms))[..., 0, :]
        ends = np.cumsum(durations, axis=-1)
        starts = np.concatenate([np.zeros_like(ends[..., :1]), ends[..., :-1]], axis=-1)
        return {
            "delay": np.asarray(self.state_buffers[(key, "delay")].to_decimal(u.ms))[..., 0],
            "starts": starts,
            "ends": ends,
            "amplitudes": np.asarray(self.state_buffers[(key, "amplitudes")].to_decimal(u.nA))[..., 0, :],
            "mask": np.asarray(self.state_buffers[(key, "_mask_durations")])[..., 0, :],
        }

    def _placement_current(self, key: int, table, t):
        """Current of one placement at time ``t`` in nA (decimal), shaped ``pop_size``."""
        if isinstance(table, dict):
            local_t = jnp.expand_dims(jnp.asarray(t.to_decimal(u.ms)) - table["delay"], axis=-1)
            active = (
                _time_ge(local_t, 0.0)
                & _time_ge(local_t, table["starts"])
                & _time_lt(local_t, table["ends"])
                & table["mask"]
            )
            return jnp.sum(jnp.where(active, table["amplitudes"], 0.0), axis=-1)
        rows = np.zeros((1,), dtype=np.int64)
        if table == "FunctionClamp":
            value = _eval_function_clamp(self, layout_id=key, local_index=0, t=t)
            return jnp.broadcast_to(jnp.asarray(value.to_decimal(u.nA)), self.pop_size)
        delay = self.state_buffers[(key, "delay")]
        local_t = (t - u.Quantity(delay.mantissa[..., rows], delay.unit)).in_unit(u.ms)
        value = _eval_sine_clamp(self, layout_id=key, rows=rows, local_t=local_t)
        return jnp.broadcast_to(value.to_decimal(u.nA)[..., 0], self.pop_size)

    def evaluate(self, t, *, area):
        """Return ``(density, components)`` at time ``t``.

        Parameters
        ----------
        t : Quantity
            Sample time.
        area : Quantity
            Membrane area of the CV, scalar or shaped ``pop_size``.

        Returns
        -------
        density : Quantity
            Summed clamp current per population member divided by the CV
            area, in ``density_unit``, shaped ``pop_size``.
        components : Quantity or None
            Current of each logical clamp in clamp-store row order, nA;
            ``None`` unless ``with_components``.
        """
        total = 0.0
        parts = []
        for key, table in self._entries:
            value = self._placement_current(key, table, t)
            total = total + value
            if self.with_components:
                parts.append(value.reshape(-1))
        # nA / cm^2 first, as the cable runtime divides clamp currents by the point area.
        density = jnp.asarray(total) / u.get_mantissa(area.to_decimal(u.cm**2)) * self._per_area
        density = u.Quantity(jnp.broadcast_to(density, self.pop_size), self.density_unit)
        if not self.with_components:
            return density, None
        components = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype=float)
        return density, u.Quantity(components, u.nA)
