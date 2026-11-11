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

"""``Cell`` — single-class multi-compartment neuron.

``cell[index]`` returns a ``CellView``: a population selection that keeps only
the root reference and stable indices, and gathers population-shaped values
from the root on demand. The root ``Cell`` stays the only owner of morphology,
discretization caches and runtime state.

A ``Cell`` carries both the declaration (morphology, CV policy, paint /
place rules, solver, spike config) and the runtime (``V`` / ``spike`` /
``current_time`` brainstate states, node tree, axial operator,
installed channel / ion nodes).

The lifecycle has two phases:

1. **DECLARING** (default). ``paint`` / ``place`` / ``cv_policy`` /
   ``V_th`` / ``V_init`` / ``solver`` / ``spk_fun`` setters are all
   mutable. Runtime methods raise.
2. **INITIALIZED**. After :meth:`init_state`, structure is frozen, existing
   numerical parameters can be updated through Views, and
   the runtime surface (:meth:`run`, :meth:`update`,
   :meth:`sample_probe`, inspection, ...) becomes available. Call
   :meth:`reset` to drop the runtime and re-enter DECLARING.

``run(dt=, duration=)`` auto-calls :meth:`init_state` on first use for
convenience. Subsequent ``run`` calls never re-initialize.
"""

import operator
import warnings
import weakref
from types import MappingProxyType
from typing import Callable, ClassVar, Mapping, Optional
from dataclasses import dataclass, replace

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base_channel import Channel, IonChannel, Synapse as RuntimeSynapse
from braincell._base_ion import Ion, MixIons
from braincell._base_neuron import HHTypedNeuron, _zero_spike_like
from braincell._deprecation import Deprecation
from braincell._misc import (
    is_traced_value,
    scalar_decimal,
    profiler_call_name as _call_name,
    profiler_scope_name as _scope_name,
)
from braincell._typing import Initializer, Size
from braincell._compute.table import MechanismObjectTable, build_mechanism_object_table
from braincell._compute.scheduling import build_node_scheduling
from braincell._compute.state import CellRuntimeState
from braincell._compute.bindings import _is_root_level_runtime_node
from braincell._compute.layouts import layout_id_from_key
from braincell._discretization.mechanism import (
    PaintRule,
    PlaceRule,
    default_paint_rules,
    merge_paint_rules,
    merge_place_rules,
    missing_branch_name,
    moved_index_message,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell._discretization.policy import CompositeByTypePolicy, CVPerBranch, CVPerBranchList, CVPolicy, Lumped
from braincell._discretization.base import (
    CV,
    CVTree,
    Discretization,
    Node,
    NodeTree,
    build_discretization,
)
from braincell.filter import AllRegion, LocsetBatch, LocsetExpr, LocsetMask, RegionExpr, RegionMask, at
from braincell.filter._index_refs import held_branch_indices
from braincell.network.event import EventOutputCollection, _CellSpikeSource
from braincell.network.recording import (
    RecordingSpec,
    _ClampCurrentObservable,
    compile_recording,
    recording_is_active,
)
from braincell.reduction import ReductionOutput, ReductionViewCollection
from braincell.reduction.runtime import ReductionInputRuntime, build_reduction_input_runtime
from braincell.morph.morphology import Morphology, frozen_snapshot
from braincell.quad import get_integrator, ind_exp_euler_step, integrator_info, list_integrators
from braincell.quad._exp_euler import _ind_exp_euler_step_selected
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqGroupState, IndependentIntegration, state_grouping
from braincell.mech import CableProperty, CVContext, Density, Point, Synapse as SynapsePlacement
from braincell._multi_compartment import boundary, currents, field_resolution, probes, run as run_module
from braincell._compute import bridge
from braincell._multi_compartment.synapses import SynapseView, _SynapseStore, raise_on_name_type_conflict
from braincell._multi_compartment.clamps import ClampView, _ClampStore
from braincell._multi_compartment.selection import BranchSelector, CVSelector, _CellScope, _Undeclarable
from braincell._multi_compartment.lifecycle import DiscreteView
from braincell._multi_compartment.density_views import ChannelView, IonView
from braincell._multi_compartment.geometry import GeometryView
from braincell._multi_compartment.lumped import (
    CABLE_ONLY_SOLVERS,
    CURRENT_DENSITY_UNIT as _LUMPED_CURRENT_UNIT,
    LumpedClamps,
    as_current_density,
    build_lumped_runtime,
    warn_stiff_gates,
    with_delta_inputs,
)
from braincell.morph.morphology import Morphology as _Morphology

__all__ = ["Cell", "CellView", "MultiCompartment"]

#: ``Cell.bind_synapse_input`` is deprecated; ``removed_in`` is provisional.
_BIND_SYNAPSE_INPUT = Deprecation(
    "braincell.Cell.bind_synapse_input()",
    since="0.3",
    removed_in="0.4",
    replacement="an EventSource with braincell.connect()",
)


@dataclass(frozen=True)
class AxialOperatorCache:
    float_dtype: jnp.dtype
    operator: object


@dataclass(frozen=True)
class RuntimeIonBinding(DiscreteView):
    """One runtime ion seen through a CV or node inspection view."""

    name: str
    runtime: object
    cell: "Cell"
    cv_ids: tuple[int, ...] = ()
    point_ids: tuple[int, ...] = ()

    def __post_init__(self):
        self._bind_view(self.cell)

    def get(self, field: str):
        """Return one field projected into the local CV or node view."""
        if not hasattr(self.runtime, field):
            raise AttributeError(f"Runtime ion {self.name!r} has no field {field!r}.")
        raw = getattr(self.runtime, field)
        if self.cv_ids:
            values = field_resolution.coerce_named_cv_values(self.cell, raw, caller="Runtime CV inspection")
            return _select_local_values(values, ids=self.cv_ids)
        if self.point_ids:
            values = field_resolution.coerce_runtime_point_values(self.cell, raw)
            return _select_local_values(values, ids=self.point_ids)
        return raw

    def __getattr__(self, field: str):
        if field.startswith("_"):
            raise AttributeError(field)
        return self.get(field)


@dataclass(frozen=True)
class RuntimeCVView(DiscreteView):
    """Readonly runtime inspection view anchored at one static CV."""

    id: int
    declaration: CV
    layout_ids: tuple[int, ...]
    mid_node_id: int
    ions: Mapping[str, RuntimeIonBinding]
    _cell: "Cell"

    def __post_init__(self):
        self._bind_view(self._cell)


@dataclass(frozen=True)
class RuntimeNodeView(DiscreteView):
    """Readonly runtime inspection view anchored at one static node."""

    id: int
    declaration: Node
    layout_ids: tuple[int, ...]
    source_cv_ids: tuple[int, ...]
    ions: Mapping[str, RuntimeIonBinding]
    _cell: "Cell"

    def __post_init__(self):
        self._bind_view(self._cell)


class _CellFacade:
    """Share population-selection behavior without owning model data."""

    @property
    def _view_root(self) -> "Cell":
        raise NotImplementedError

    @property
    def root(self) -> "Cell":
        """Return the root Cell that owns this view."""
        return self._view_root

    @property
    def _scope(self) -> _CellScope:
        return self._view_root._root_scope()

    def _with_scope(self, scope: _CellScope) -> "CellView":
        return CellView(self._view_root, scope.population_indices, scope=scope)

    def __getitem__(self, selection) -> "CellView":
        """Return a view selected relative to this cell-like object."""
        return self._with_scope(self._scope.select_population(selection))

    def on(self, region: RegionExpr | RegionMask) -> "CellView":
        """Select CVs with positive area coverage by a continuous region."""
        return self._with_scope(self._scope.select_region(self._view_root, region))

    def loc(self, locset: LocsetExpr | LocsetMask | LocsetBatch) -> "CellView":
        """Select the owning CVs of ordered continuous locations."""
        return self._with_scope(self._scope.select_locations(self._view_root, locset))

    @property
    def branch(self) -> BranchSelector:
        """Return a morphology-branch selector for this scope."""
        return BranchSelector(self)

    @property
    def cv(self) -> CVSelector:
        """Return a control-volume selector for this scope."""
        return CVSelector(self)

    @property
    def soma(self) -> "CellView":
        """Select branches whose morphology type is ``soma``."""
        return self.branch.by_type("soma")

    @property
    def axon(self) -> "CellView":
        """Select branches whose morphology type is ``axon``."""
        return self.branch.by_type("axon")

    @property
    def dendrite(self) -> "CellView":
        """Select branches whose morphology type is ``dendrite``."""
        return self.branch.by_type("dendrite")

    @property
    def basal_dendrite(self) -> "CellView":
        """Select branches whose morphology type is ``basal_dendrite``."""
        return self.branch.by_type("basal_dendrite")

    @property
    def apical_dendrite(self) -> "CellView":
        """Select branches whose morphology type is ``apical_dendrite``."""
        return self.branch.by_type("apical_dendrite")

    @property
    def channels(self) -> ChannelView:
        """Return Channel logical owners intersecting this scope."""
        return ChannelView(self._view_root, self._scope)

    @property
    def ions(self) -> IonView:
        """Return Ion logical owners intersecting this scope."""
        return IonView(self._view_root, self._scope)

    @property
    def geometry(self) -> GeometryView:
        """Return fixed-grid runtime geometry parameters for this scope."""
        self._raise_if_not_initialized("geometry; call init_state() first")
        return GeometryView(self._view_root, self._scope)

    # CV selections expose geometry fields directly.  ``geometry`` remains a
    # compatibility namespace, while these aliases make geometry a content
    # view at the same level as ``channels`` and ``synapses``.
    @property
    def length(self):
        self._raise_if_not_initialized("length; call init_state() first")
        return self.geometry.length

    @property
    def radius_scale(self):
        self._raise_if_not_initialized("radius_scale; call init_state() first")
        return self.geometry.radius_scale

    @property
    def Ra(self):
        self._raise_if_not_initialized("Ra; call init_state() first")
        return self.geometry.Ra

    @property
    def cm(self):
        self._raise_if_not_initialized("cm; call init_state() first")
        return self.geometry.cm

    @property
    def radius_mid(self):
        self._raise_if_not_initialized("radius_mid; call init_state() first")
        return self.geometry.radius_mid

    @property
    def diam_arc_mean(self):
        self._raise_if_not_initialized("diam_arc_mean; call init_state() first")
        return self.geometry.diam_arc_mean

    def record(
        self,
        name: str,
        observable,
        *,
        period=None,
        frequency=None,
        start=0.0 * u.ms,
    ) -> RecordingSpec:
        """Register an observer over this population/spatial scope.

        Parameters
        ----------
        name : str
            Cell-local recording name.
        observable : object
            Descriptor created by :mod:`braincell.observe`.
        period, frequency : Quantity, optional
            Mutually exclusive regular sampling interval declarations.
        start : Quantity, optional
            Global schedule start. Defaults to ``0 ms``.

        Returns
        -------
        RecordingSpec
            Frozen declaration owned by the root Cell.

        Notes
        -----
        The recording is a declaration, like ``paint`` and ``place``. If the
        grid or the morphology changes before :meth:`Cell.init_state` (a new
        ``cv_policy``, or an edit of the tree), ``init_state`` selects it
        again from the same population members and region or locations, so
        it records the branches it named: a region recording declared before
        a branch was appended includes that branch. A selection of CVs
        (``cv[...]``) cannot be made again on other CVs, and
        ``init_state`` raises ``RuntimeError`` naming the recording. A
        region or location that names a branch by index (``cell.branch[1]``,
        ``cell.on(BranchSlice(1, 0.0, 1.0))``) is not followed through a
        removal or a move of a branch: ``init_state`` raises ``ValueError``.
        """
        return self._view_root._add_recording(
            RecordingSpec(
                name=name,
                scope=self._scope,
                observable=observable,
                period=period,
                frequency=frequency,
                start=start,
            )
        )


class PopulationRuntimeView(DiscreteView):
    """Provide read-only population selection over one runtime object."""

    __slots__ = ("_runtime", "_population_indices", "_population_size", "_packed_population_index")

    def __init__(
        self,
        runtime,
        population_indices: tuple[int, ...],
        population_size: int,
        *,
        cell,
        packed_population_index: np.ndarray | None = None,
    ) -> None:
        self._runtime = runtime
        self._population_indices = population_indices
        self._population_size = population_size
        self._packed_population_index = packed_population_index
        self._bind_view(cell)

    @property
    def root(self):
        """Return the unselected runtime object."""
        return self._runtime

    @property
    def population_indices(self) -> tuple[int, ...]:
        """Return selected population indices."""
        return self._population_indices

    def get(self, field: str):
        """Return one runtime field gathered over the selected population."""
        if not hasattr(self._runtime, field):
            raise AttributeError(f"Runtime object {type(self._runtime).__name__!r} has no field {field!r}.")
        value = getattr(self._runtime, field)
        if callable(value) or isinstance(value, (dict, list, set)):
            raise AttributeError(f"Runtime field {field!r} is not available through the read-only population view.")
        if isinstance(value, brainstate.State):
            value = value.value
        if self._packed_population_index is not None:
            return _select_packed_population_value(
                value,
                owners=self._packed_population_index,
                population_indices=self._population_indices,
            )
        return _select_population_value(
            value,
            population_indices=self._population_indices,
            population_size=self._population_size,
        )

    def __getattr__(self, field: str):
        if field.startswith("_"):
            raise AttributeError(field)
        return self.get(field)


class CellView(DiscreteView, _CellFacade):
    """View selected members of a homogeneous :class:`Cell` population.

    A view owns no morphology, discretization, or runtime state. It stores a
    root cell reference and stable population indices, then gathers selected
    values on demand.

    Parameters
    ----------
    cell : Cell
        Root population cell.
    population_indices : tuple of int
        Stable root-population indices.
    """

    __slots__ = ("_cell", "_population_indices", "_selection_scope")

    def __init__(
        self,
        cell: "Cell",
        population_indices: tuple[int, ...],
        *,
        scope: _CellScope | None = None,
    ) -> None:
        self._cell = cell
        self._population_indices = tuple(population_indices)
        self._selection_scope = (
            cell._root_scope().select_population(tuple(population_indices)) if scope is None else scope
        )
        self._bind_view(cell)

    @property
    def root(self) -> "Cell":
        """Return the root :class:`Cell` that owns all data."""
        return self._cell

    @property
    def cell(self) -> "Cell":
        """Return the root cell (compatibility spelling)."""
        return self._cell

    @property
    def population_indices(self) -> tuple[int, ...]:
        """Return selected root-population indices."""
        return self._population_indices

    @property
    def _view_root(self) -> "Cell":
        return self._cell

    @property
    def _scope(self) -> _CellScope:
        self._check_view()
        return self._selection_scope

    def _with_scope(self, scope: _CellScope) -> "CellView":
        return CellView(self._cell, scope.population_indices, scope=scope)

    @property
    def indices(self) -> np.ndarray:
        """Return selected root-population indices as an integer array."""
        return np.asarray(self._population_indices, dtype=np.int64)

    @property
    def shape(self) -> tuple[int]:
        """Return the one-dimensional selection shape."""
        return (len(self),)

    @property
    def size(self) -> int:
        """Return the number of selected population members."""
        return len(self)

    @property
    def pop_size(self) -> tuple[int]:
        """Return the selected population shape."""
        return self.shape

    @property
    def varshape(self) -> tuple[int, int]:
        """Return selected population-by-CV shape."""
        if not self._scope.spatially_restricted:
            return self.shape + (self._cell.n_cv,)
        return (len(self._scope.pairs),)

    def __len__(self) -> int:
        self._check_view()
        return len(self._population_indices)

    @property
    def morpho(self) -> Morphology:
        """Return the root Cell's :attr:`Cell.morpho`, without copying it."""
        return self._cell.morpho

    @property
    def cv_policy(self) -> CVPolicy:
        """Return the shared control-volume policy."""
        return self._cell.cv_policy

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        """Return population-wide paint declarations."""
        return self._cell.paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        """Return place declarations that affect at least one selected cell."""
        selected = set(self._population_indices)
        if not selected:
            return ()
        return tuple(
            rule
            for rule in self._cell.place_rules
            if rule.population_indices is None or selected.intersection(rule.population_indices)
        )

    @property
    def solver(self):
        """Return the shared voltage solver."""
        return self._cell.solver

    @property
    def solver_name(self) -> str:
        """Return the shared voltage solver name."""
        return self._cell.solver_name

    @property
    def subsolver(self):
        """Return the shared ion/channel subsolver."""
        return self._cell.subsolver

    @property
    def subsolver_name(self) -> str:
        """Return the shared ion/channel subsolver name."""
        return self._cell.subsolver_name

    @property
    def substeps(self) -> int:
        """Return the shared substep count."""
        return self._cell.substeps

    @property
    def membrane_linearizer(self) -> str:
        """Return the shared membrane linearizer name."""
        return self._cell.membrane_linearizer

    @property
    def spk_fun(self):
        """Return the shared spike function."""
        return self._cell.spk_fun

    @property
    def name(self) -> str | None:
        """Return the root cell name."""
        return self._cell.name

    @property
    def n_cv(self) -> int:
        """Return the shared number of control volumes."""
        return self._cell.n_cv

    @property
    def cvs(self) -> tuple[CV, ...]:
        """Return control-volume declarations selected by this scope."""
        if not self._scope.spatially_restricted:
            return self._cell.cvs
        return CVCollectionView(
            self._cell,
            tuple(self._cell.cvs[cv_id] for cv_id in self._scope.cv_ids),
            self._scope,
        )

    @property
    def cv_midpoints(self) -> LocsetMask:
        """Return continuous midpoint locations for selected CVs."""
        if not self._scope.spatially_restricted:
            return self._cell.cv_midpoints
        root = self._cell.cv_midpoints
        ids = np.asarray(self._scope.cv_ids, dtype=np.int64)
        return LocsetMask.from_columns(root.branch_id[ids], root.branch_x[ids])

    @property
    def spatial_pairs(self) -> np.ndarray:
        """Return selected ``(population_index, cv_id)`` rows."""
        return np.asarray(self._scope.pairs, dtype=np.int64).reshape(-1, 2)

    @property
    def locations(self):
        """Return ordered, duplicate-preserving source location rows."""
        return self._scope.locations

    def at_x(self, branch_x: float) -> "CellView":
        """Select one coordinate on an exactly selected morphology branch."""
        branch_ids = self._scope.exact_branch_ids
        if branch_ids is None or len(branch_ids) != 1:
            raise ValueError("CellView.at_x(...) requires exactly one branch selected through cell.branch[...].")
        return self.loc(at(int(branch_ids[0]), branch_x))

    @property
    def cv_tree(self) -> CVTree:
        """Return the shared control-volume tree."""
        return self._cell.cv_tree

    @property
    def node_tree(self) -> NodeTree:
        """Return the shared electrical node tree."""
        return self._cell.node_tree

    @property
    def cv_contexts(self) -> tuple[CVContext, ...]:
        """Return shared control-volume contexts."""
        return self._cell.cv_contexts

    @property
    def n_point(self) -> int:
        """Return the runtime electrical point count."""
        return self._cell.n_point

    @property
    def n_compartment(self) -> int:
        """Return the shared compartment count."""
        return self._cell.n_compartment

    @property
    def current_time(self):
        """Return the root cell's shared simulation time."""
        return self._cell.current_time

    @property
    def layouts(self):
        """Return shared runtime layout metadata."""
        return self._cell.layouts

    @property
    def voltage_shape(self):
        """Return the selected population voltage shape."""
        self._cell._raise_if_not_initialized("CellView.voltage_shape")
        return self.varshape

    @property
    def point_placements(self):
        """Return placement declarations effective for selected cells."""
        selected = set(self._population_indices)
        if not selected:
            return ()
        return tuple(
            placement
            for placement in self._cell.point_placements
            if placement.population_index is None or int(placement.population_index) in selected
        )

    @property
    def synapses(self):
        """Return logical synapse instances owned by selected cells."""
        selected = self._cell.synapses.for_population(self._population_indices)
        if self._scope.spatially_restricted:
            selected = selected.for_scope_pairs(self._scope.pairs)
        return selected

    @property
    def clamps(self):
        """Return logical current clamps owned by this population/spatial view."""
        selected = self._cell.clamps.for_population(self._population_indices)
        if self._scope.spatially_restricted:
            selected = selected.for_scope_pairs(self._scope.pairs)
        return selected

    @property
    def connections(self):
        """Return routing rows whose destination synapses belong to selected cells."""
        selected = self._cell.connections.for_population(self._population_indices)
        if self._scope.spatially_restricted:
            selected = selected.by_synapse_ids(self.synapses.id)
        return selected

    @property
    def event_outputs(self) -> EventOutputCollection:
        """Return named live event outputs for selected population members."""
        return self._cell.event_outputs.for_population(self._population_indices)

    @property
    def reductions(self) -> ReductionViewCollection:
        """Return registered reduction models over selected population members."""
        if self._scope.spatially_restricted:
            raise RuntimeError("Reduction parameters can only be selected by population, not by morphology location.")
        return ReductionViewCollection(self._cell, self._population_indices)

    @property
    def outputs(self) -> Mapping[str, object]:
        """Return initialized model outputs gathered over selected members."""
        if self._scope.spatially_restricted:
            raise RuntimeError("Model outputs can only be selected by population, not by morphology location.")
        return self._cell._selected_outputs(self._population_indices)

    @property
    def V_init(self):
        """Return effective initial voltages for selected cells."""
        return self._selected_voltage_parameter("V_init")

    @V_init.setter
    def V_init(self, value) -> None:
        self.set(V_init=value)

    @property
    def V_th(self):
        """Return effective spike thresholds for selected cells."""
        return self._selected_voltage_parameter("V_th")

    def _selected_voltage_parameter(self, name):
        values = self._cell._selected_population_parameter(name, self._population_indices)
        if not self._scope.spatially_restricted:
            return values
        if not self._cell.pop_size:
            return values[self._scope.pair_cv_id]
        local = {population: i for i, population in enumerate(self._population_indices)}
        rows = np.asarray([local[p] for p, _ in self._scope.pairs], dtype=np.int32)
        if self._cell._is_lumped:
            return values[rows]
        return values[rows, self._scope.pair_cv_id]

    @V_th.setter
    def V_th(self, value) -> None:
        self.set(V_th=value)

    @property
    def V(self):
        """Return initialized membrane voltage gathered over selected cells."""
        self._cell._raise_if_not_initialized("CellView.V")
        if self._cell._is_lumped:
            # No CV axis: a spatial selection reads the member of each pair.
            populations = (
                self._scope.pair_population_index if self._scope.spatially_restricted else self._population_indices
            )
            return self._cell.V.value[..., np.asarray(populations, dtype=np.int32)]
        if self._scope.spatially_restricted:
            if len(self._cell.pop_size) == 0:
                return self._cell.V.value[..., self._scope.pair_cv_id]
            return self._cell.V.value[..., self._scope.pair_population_index, self._scope.pair_cv_id]
        return _select_population_value(
            self._cell.V.value,
            population_indices=self._population_indices,
            population_size=self._cell._population_size,
        )

    @property
    def spike(self):
        """Return initialized spike values gathered over selected cells."""
        self._cell._raise_if_not_initialized("CellView.spike")
        if self._cell._uses_reduction:
            return _select_model_output(
                self._cell.spike.value,
                population_indices=self._population_indices,
                population_size=self._cell._population_size,
                batch_size=self._cell._runtime_batch_size,
                detailed=False,
            )
        if self._cell._is_lumped:
            return self._cell.spike.value[..., np.asarray(self._population_indices, dtype=np.int32)]
        return _select_population_value(
            self._cell.spike.value,
            population_indices=self._population_indices,
            population_size=self._cell._population_size,
        )

    def set(self, **parameters) -> "CellView":
        """Set runtime initial-voltage/threshold parameters on selected CVs.

        Parameters
        ----------
        **parameters
            Supported keys are ``V_init`` and ``V_th``. Values must be
            concrete voltage quantities that broadcast within the selected
            population-by-CV shape. Spatial selections accept a scalar or
            one value per selected logical row. Requires initialization.

        Returns
        -------
        CellView
            This view.
        """
        self._cell._set_runtime_voltage_parameters(self._scope, parameters)
        return self

    def place(self, *args):
        """Place point mechanisms through this view.

        On a location view (``cell.loc(locset)``) the view supplies the
        locset: ``cell.loc(locset).place(*mechanisms)`` is
        ``cell.place(locset, *mechanisms)`` and accepts any point mechanism.
        On a population view (``cell[indices]``) pass the locset first; the
        mechanisms are placed on the selected members only, which currently
        supports synapses.

        Parameters
        ----------
        *args
            ``*mechanisms`` on a location view; ``locset, *mechanisms`` on a
            population view, where ``locset`` is a shared locset, an aligned
            :class:`LocsetBatch`, or one possibly ragged locset per selected
            member.

        Returns
        -------
        Cell or CellView
            The root Cell for a location view; this view for a population
            view.

        Raises
        ------
        TypeError
            If a location view is also given a locset.
        ValueError
            If the view selects a region or control volumes rather than
            locations, or nests a location selection inside another spatial
            selection.
        IndexError
            If a resolved mask names a branch the morphology does not have.

        A call that raises leaves the Cell's declarations as they were.
        """
        scope = self._scope
        if not scope.spatially_restricted:
            if self._cell._is_lumped and args and isinstance(args[0], Point):
                # A lumped Cell has one site, as in Cell.place without a locset.
                args = (self._cell.cv_midpoints, *args)
            if not args:
                raise TypeError("CellView.place(locset, *mechanisms) needs a locset on a population view.")
            locset, *mechanisms = args
            self._cell._place_selected(self._population_indices, locset, tuple(mechanisms))
            return self
        target = scope.target
        if not isinstance(target, (LocsetExpr, LocsetMask, LocsetBatch)):
            raise ValueError(
                f"CellView.place() needs a location view such as cell.loc(locset); this view is "
                f"{_describe_target(target)}."
            )
        if args and isinstance(args[0], (LocsetExpr, LocsetMask, LocsetBatch, tuple, list)):
            raise TypeError(
                "A location view places at its own locations; call view.place(*mechanisms) without a locset."
            )
        if self._covers_population:
            return self._cell.place(target, *args)
        self._cell._place_selected(self._population_indices, target, tuple(args))
        return self._cell

    def paint(self, *mechanisms) -> "Cell":
        """Paint density mechanisms onto this view's region.

        ``cell.soma.paint(m)`` is ``cell.paint(branch_in("type", ["soma"]), m)``
        and ``cell.on(region).paint(m)`` is ``cell.paint(region, m)``; nested
        region selections paint their intersection.

        Parameters
        ----------
        *mechanisms : Density or CableProperty
            Declarations to paint.

        Returns
        -------
        Cell
            The root Cell, for chaining.

        Raises
        ------
        TypeError
            If a region is passed as well; the view supplies it.
        ValueError
            If the view selects population members (declarations are the
            same for every member) or anything other than a region.
        """
        if not self._covers_population:
            raise ValueError(
                "CellView.paint() paints every population member alike; call paint() on the root Cell "
                "or on a spatial view of it (cell.soma, cell.on(region)), not on cell[...]."
            )
        if mechanisms and isinstance(mechanisms[0], (RegionExpr, RegionMask)):
            raise TypeError("A spatial view paints its own region; call view.paint(*mechanisms) without a region.")
        target = self._scope.target
        if target is None:
            return self._cell.paint(*mechanisms)
        if not isinstance(target, RegionExpr):
            raise ValueError(
                f"CellView.paint() needs a region view such as cell.soma or cell.on(region); this view is "
                f"{_describe_target(target)}."
            )
        return self._cell.paint(target, *mechanisms)

    @property
    def _covers_population(self) -> bool:
        return set(self._population_indices) == set(range(self._cell._population_size))

    def init_state(self, *args, **kwargs):
        """Reject lifecycle transitions on a population view."""
        raise RuntimeError("CellView cannot own runtime state; call init_state() on its root Cell.")

    def reset(self, *args, **kwargs):
        """Reject lifecycle transitions on a population view."""
        raise RuntimeError("CellView cannot reset runtime state; call reset() on its root Cell.")

    def reset_state(self, *args, **kwargs):
        """Reject runtime mutation on a population view."""
        raise RuntimeError("CellView runtime state is read-only; call reset_state() on its root Cell.")

    def run(self, *args, **kwargs):
        """Reject simulation execution on a population view."""
        raise RuntimeError("CellView cannot run independently; call run() on its root Cell.")

    def get_ion(self, name: str) -> PopulationRuntimeView:
        """Return read-only selected-population inspection for one runtime ion."""
        runtime = self._cell.get_ion(name)
        return PopulationRuntimeView(runtime, self._population_indices, self._cell._population_size, cell=self._cell)

    def get_runtime_node(self, layout_id: int) -> PopulationRuntimeView:
        """Return read-only selected-population inspection for a runtime node."""
        runtime = self._cell.get_runtime_node(layout_id)
        layout = self._cell.layouts[int(layout_id)]
        return PopulationRuntimeView(
            runtime,
            self._population_indices,
            self._cell._population_size,
            cell=self._cell,
            packed_population_index=layout.population_index,
        )

    def __repr__(self) -> str:
        return (
            f"CellView(name={self.name!r}, population_indices={self._population_indices!r}, "
            f"cv_ids={self._scope.cv_ids!r})"
        )


def _describe_target(target) -> str:
    """Name a view's spatial selection in an error message."""
    if isinstance(target, _Undeclarable):
        return target.reason
    if isinstance(target, (LocsetExpr, LocsetMask, LocsetBatch)):
        return f"a location selection ({target!r})"
    return f"a region selection ({target!r})"


class CVCollectionView(tuple):
    """Tuple-compatible CV position view with unified geometry fields."""

    def __new__(cls, cell, records, scope=None):
        obj = super().__new__(cls, records)
        obj._cell = cell
        obj._scope = scope
        return obj

    @property
    def cell(self):
        return self._cell

    def __getitem__(self, selector):
        result = super().__getitem__(selector)
        if isinstance(selector, slice):
            scope = self._scope
            ids = tuple(cv.id for cv in result)
            scope = (self._cell._root_scope() if scope is None else scope).select_cv_ids(ids)
            return CVCollectionView(self._cell, result, scope)
        return result

    def _geometry(self):
        self._cell._raise_if_not_initialized("CV geometry; call init_state() before reading runtime geometry")
        scope = self._scope
        if scope is None:
            scope = self._cell._root_scope().select_cv_ids(tuple(cv.id for cv in self))
        return GeometryView(self._cell, scope)

    def _content_view(self):
        scope = self._scope
        if scope is None:
            scope = self._cell._root_scope()
        else:
            scope = scope.select_cv_ids(tuple(cv.id for cv in self))
        return CellView(self._cell, scope.population_indices, scope=scope)

    @property
    def channels(self):
        return self._content_view().channels

    @property
    def ions(self):
        return self._content_view().ions

    @property
    def synapses(self):
        return self._content_view().synapses

    @property
    def connections(self):
        return self._content_view().connections

    @property
    def length(self):
        return self._geometry().length

    @property
    def radius_scale(self):
        return self._geometry().radius_scale

    @property
    def Ra(self):
        return self._geometry().Ra

    @property
    def cm(self):
        return self._geometry().cm

    @property
    def radius_mid(self):
        return self._geometry().radius_mid

    @property
    def diam_arc_mean(self):
        return self._geometry().diam_arc_mean

    @property
    def area(self):
        return self._geometry().area

    @property
    def resistance_prox(self):
        return self._geometry().resistance_prox

    @property
    def resistance_dist(self):
        return self._geometry().resistance_dist


class Cell(_CellFacade, HHTypedNeuron):
    """Multi-compartment cell with explicit declaration / initialization phases.

    Parameters
    ----------
    morpho : Morphology
        Morphology tree.
    cv_policy : CVPolicy, optional
        Control-volume splitting policy; defaults to :class:`CVPerBranch`.
        :class:`Lumped` makes a lumped (point) cell: ``V``, ``spike`` and
        mechanism states have shape ``pop_size`` with no CV axis, and the
        cell has a single site at the CV midpoint. Its views, trainable
        parameters, synapses, recordings and ``Network`` populations give
        what the same declarations give with ``CVPerBranch()`` on the one-CV
        soma; views and recordings drop that one-CV axis. :attr:`runtime`
        is the one-CV layout state the lumped mechanisms run on. See
        :meth:`point`.
    V_th : Quantity
        Spike-detection threshold (default ``0. mV``). An integer-valued
        quantity such as ``0 * u.mV`` is converted to the default float dtype.
    V_init : Quantity or Callable or None
        Initial voltage. ``None`` means "use per-CV resting potential".
        An integer-valued quantity such as ``-65 * u.mV``, or an integer
        result of a callable, is converted to the default float dtype.
    spk_fun : Callable
        Surrogate-gradient spike function.
    solver : str or Callable or None
        Integrator name (registry lookup) or callable step function. ``None``
        (the default) selects ``"staggered"`` on a cable cell and
        ``"ralston2"`` on a lumped cell; a lumped cell rejects the cable-only
        ``"staggered"`` and ``"dhs_voltage"``. :attr:`solver_name` reports the
        resolved name. Every name of :func:`braincell.quad.list_integrators`
        advances a cable cell: a generic one (the explicit Runge-Kutta
        methods, ``"exp_euler"``, ``"ind_exp_euler"``, ``"backward_euler"``,
        ``"crank_nicolson"``) integrates ``V`` with the dependent mechanism
        states, after which Markov channels and kinetic ions advance at the
        new voltage through their own subsolver, in the staggered order.
        ``"exp_euler"``, ``"backward_euler"`` and ``"crank_nicolson"`` solve
        all states as one block per cell and reject point synapses at
        :meth:`init_state` with ``ValueError``. ``"dhs_voltage"`` advances
        ``V`` only. A callable is called as ``solver(cell)`` and owns the
        whole step. On a lumped cell an explicit Runge-Kutta solver (the
        default ``"ralston2"`` included) integrates the HH gates with ``V``;
        a gate faster than about 80 /ms somewhere between -100 and 60 mV
        makes it diverge at ``dt = 0.025 ms`` (the catalogue's
        ``Cav1p2_*``, ``Cav1p3_*``, ``HCN1_*`` and ``Na_ZH2019_IO``), and
        :meth:`init_state` warns naming the gate. ``"exp_euler"`` (second
        order, about 20 times the cost per step) and ``"ind_exp_euler"``
        (first order, about the cost of ``"ralston2"``) are stable on them.
    subsolver : str or Callable or None
        Integrator for Markov channels and kinetic ions. Together with
        ``substeps=None``, ``None`` selects ``"backward_euler"``. A
        declaration that sets its own ``solver``/``substeps`` overrides it;
        a channel class's ``default_solver`` does not. The catalogue's
        sodium Markov channels are stiff, so use only ``"backward_euler"``
        or ``"exp_euler"`` for them: explicit methods, ``"crank_nicolson"``
        and ``"ind_exp_euler"`` diverge. ``"backward_euler"`` is first
        order (at ``dt = 0.025 ms`` their voltage-clamp currents are off by
        7-20 %, as in NEURON); ``"exp_euler"`` is exact while the voltage
        is constant over a step and costs about 4-6 times as much per step.
        To use it for one channel, declare
        ``bc.mech.Channel("Nav1p6_MA2020_GoC", solver="exp_euler", substeps=1)``.
        See :class:`~braincell.channel.Markov`.
    substeps : int or None
        Number of subsolver steps per main cell step. Together with
        ``subsolver=None``, ``None`` selects one step.
    ion_channel_update_order : {"family", "integration"}
        Post-voltage ion/channel scheduling. ``"family"`` updates all ions
        before all channels; ``"integration"`` preserves the previous
        IndependentIntegration-grouped scheduling: the dependent nodes first
        (with the dependent channels of a self-integrating ion such as a
        ``KineticIon``, which read its state before it advances), then each
        node's own ``ind_update``.
    membrane_linearizer : {"point", "generic"}
        Compatibility selector for the membrane-current linearization
        strategy. Both values currently differentiate the assembled CV-space
        membrane derivative; painted density mechanisms already live on the
        CV axis, while sparse point contributions are gathered first.
    name : str, optional
        Cell name.
    """

    __module__ = "braincell"

    # A Cell lays its DiffEqState leaves out per compartment, so a solver that
    # flattens them must concatenate rather than stack. See DiffEqModule.
    diffeq_state_merging: ClassVar[str] = "concat"

    # ------------------------------------------------------------------
    # Construction

    def __init__(
        self,
        morpho: Morphology,
        *,
        pop_size: Size = 1,
        cv_policy: CVPolicy | None = None,
        V_th: u.Quantity = 0 * u.mV,
        V_init: Optional[Initializer] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable | None = None,
        subsolver: str | Callable | None = None,
        substeps: int | None = None,
        cache_ion_total_current: bool = True,
        ion_channel_update_order: str = "family",
        membrane_linearizer: str = "point",
        name: str | None = None,
    ) -> None:
        normalized_pop_size = _normalize_pop_size(pop_size)
        HHTypedNeuron.__init__(self, size=normalized_pop_size + (1,), name=name)

        if not isinstance(morpho, Morphology):
            raise TypeError(f"Cell expects Morphology, got {type(morpho).__name__!s}.")

        self._declaration_morpho = morpho
        self._morpho = morpho
        self._pop_size = normalized_pop_size

        self._discretization_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._discretization_policy, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(self._discretization_policy).__name__!s}.")
        self._cv_policy_index_basis = _cv_policy_index_basis(self._discretization_policy, morpho)

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_th_declaration = V_th
        self._V_th_parameter = None
        self._next_detector_id = 0
        self._V_init = V_init
        self._V_init_materialized = None
        self._spk_fun = spk_fun
        self._name = name
        self._solver_declaration = solver
        self._solver_name, self._solver_fn = _resolve_cell_solver(
            solver, lumped=isinstance(self._discretization_policy, Lumped)
        )
        (
            self._subsolver_name,
            self._subsolver_fn,
            self._substeps,
        ) = _resolve_subsolver_schedule(subsolver, substeps)
        self.cache_ion_total_current = bool(cache_ion_total_current)
        self.ion_channel_update_order = _validate_ion_channel_update_order(ion_channel_update_order)
        self._membrane_linearizer = _validate_membrane_linearizer(membrane_linearizer)

        self._discretization_cache: Discretization | None = None
        self._cv_collection_cache = None
        self._discretization_cache_key: object = None
        # Set by init_state when the Cell starts running on a frozen copy of
        # its declaration tree (MORPH-14): the tree's revision and index
        # watermark at the copy, and the last revision run() warned about.
        self._snapshot_source_revision: int | None = None
        self._snapshot_reindex_revision: int | None = None
        self._snapshot_warned_revision: int | None = None
        self._view_generation = 0
        self._runtime_generation = 0
        self._runtime_V_init = None
        self._runtime_V_init_mask = None
        self._root_scope_cache: _CellScope | None = None

        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._node_scheduling_cache: dict[tuple[str, int], object] = {}
        self._run_loop_cache: dict[tuple[object, ...], object] = {}

        self._runtime: CellRuntimeState | None = None
        self._lumped_runtime = None
        self._lumped_clamps: LumpedClamps | None = None
        self._reduction_models: dict[str, object] = {}
        self._selected_model_name = "detailed"
        self._reduction_input_runtime: ReductionInputRuntime | None = None
        self._reduction_output_states: dict[str, brainstate.State] = {}
        self._pending_reduction_inputs = None
        self._runtime_batch_size: int | None = None
        # Set by ``pre_integral`` when a solver step is called directly, so
        # its stages read the clamp current sampled for this step.
        self._direct_step_clamps_prepared = False
        # Algebraic boundary rows of the derivative solvers, built on first use.
        self._boundary_rows_cache: boundary.BoundaryRows | None = None
        self._runtime_cvs_cache: tuple[RuntimeCVView, ...] | None = None
        self._runtime_nodes_cache: tuple[RuntimeNodeView, ...] | None = None
        self._synapse_store_cache: _SynapseStore | None = None
        self._clamp_store_cache: _ClampStore | None = None
        self._spike_event_source_cache: _CellSpikeSource | None = None
        self._synapse_input_bindings: dict[str, list[tuple[object, object, object]]] = {}
        self._synapse_origins: dict[int, SynapsePlacement] = {}
        self._connection_store_cache = None
        self._connection_weight_declarations = ()
        self._network_owner_ref: weakref.ReferenceType | None = None
        self._recording_specs: dict[str, RecordingSpec] = {}
        # Per recording: (tree, revision, grid fingerprint, index basis) it
        # was resolved on; init_state resolves it again when these change.
        self._recording_bases: dict[str, tuple] = {}
        self._grid_fingerprint_cache: tuple | None = None
        self._compiled_recording_cache: dict[tuple, tuple] = {}

        self._initialized = False
        # Set by init_state: the solver is a generic registered step, so the
        # Markov channels and kinetic ions advance after it (``ind_update``).
        self._ind_update_after_solver = False

        from braincell.trainable import TrainableManager

        self.trainables = TrainableManager(self)

        self.discretize()

    # ------------------------------------------------------------------
    # Phase guards

    def _raise_if_initialized(self, action: str) -> None:
        owner = self.network_owner
        if owner is not None and owner._initialized and not getattr(owner, "_cell_lifecycle_active", False):
            owner_name = owner.name if owner.name is not None else "<unnamed>"
            raise RuntimeError(f"Cannot {action} after owning Network {owner_name!r} has been initialized.")
        if self._initialized:
            raise RuntimeError(f"Cannot {action} after init_state(); call reset() first.")

    def _raise_if_not_initialized(self, action: str) -> None:
        if not self._initialized:
            raise RuntimeError(f"{action} requires init_state() first.")

    @property
    def network_owner(self):
        """Return the Network that owns execution of this cell, if any."""
        return None if self._network_owner_ref is None else self._network_owner_ref()

    def _bind_network_owner(self, network) -> None:
        """Bind this cell to one Network execution scope."""
        owner = self.network_owner
        if owner is not None and owner is not network:
            owner_name = owner.name if owner.name is not None else "<unnamed>"
            raise RuntimeError(f"Cell already belongs to Network {owner_name!r}.")
        self._network_owner_ref = weakref.ref(network)

    def _raise_if_network_owned(self, action: str) -> None:
        """Reject public lifecycle mutation outside the owning Network."""
        owner = self.network_owner
        if owner is None or getattr(owner, "_cell_lifecycle_active", False):
            return
        owner_name = owner.name if owner.name is not None else "<unnamed>"
        raise RuntimeError(f"Cell belongs to Network {owner_name!r}; use Network.{action} instead.")

    # ------------------------------------------------------------------
    # Read-only accessors / guarded config setters

    @property
    def morpho(self) -> Morphology:
        """The morphology this Cell discretizes and runs on.

        Before :meth:`init_state` it is the tree the Cell was declared with,
        shared and not copied: an edit of that tree is seen at the next grid
        read. :meth:`init_state` replaces it with a read-only snapshot of
        that tree, which the Cell runs on until :meth:`reset`. Editing the
        snapshot raises :class:`RuntimeError` before writing anything;
        editing the declaration tree leaves the run unchanged (:meth:`run`
        warns once per edit), and the edit applies at the next
        :meth:`reset` and :meth:`init_state`.

        Examples
        --------
        >>> import brainunit as u
        >>> from braincell import Branch, Cell, Morphology
        >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> cell = Cell(tree)
        >>> cell.morpho is tree
        True
        >>> cell.init_state()
        >>> cell.morpho is tree, cell.morpho == tree
        (False, True)
        >>> cell.reset()
        >>> cell.morpho is tree
        True
        """
        return self._morpho

    @morpho.setter
    def morpho(self, value: Morphology) -> None:
        self.morphology = value

    @property
    def morphology(self) -> Morphology:
        """Alias of :attr:`morpho`: the declaration tree, or its snapshot once initialized."""
        return self._morpho

    @morphology.setter
    def morphology(self, value: Morphology) -> None:
        """Replace the declaration morphology and refresh the fixed grid."""
        self._raise_if_initialized("assign morphology")
        if not isinstance(value, Morphology):
            raise TypeError(f"morphology must be Morphology, got {type(value).__name__!s}.")
        previous = self._morpho
        self._morpho = value
        try:
            self.discretize()
        except Exception:
            self._morpho = previous
            raise
        self._declaration_morpho = value

    @property
    def cv_policy(self) -> CVPolicy:
        return self._discretization_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        self._raise_if_initialized("assign cv_policy")
        if not isinstance(value, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(value).__name__!s}.")
        lumped = isinstance(value, Lumped)
        solver = _resolve_cell_solver(self._solver_declaration, lumped=lumped)
        previous = self._discretization_policy, self._cv_policy_index_basis
        self._discretization_policy = value
        self._cv_policy_index_basis = _cv_policy_index_basis(value, self._morpho)
        try:
            self.discretize()
        except Exception:
            self._discretization_policy, self._cv_policy_index_basis = previous
            raise
        self._solver_name, self._solver_fn = solver
        self._run_loop_cache.clear()

    @property
    def _is_lumped(self) -> bool:
        """Whether this cell runs the lumped (point) runtime."""
        return isinstance(self._discretization_policy, Lumped)

    @classmethod
    def point(
        cls,
        pop_size: Size = 1,
        *,
        length: u.Quantity = 10.0 * u.um,
        radius: u.Quantity = 5.0 * u.um,
        V_th: u.Quantity = 0 * u.mV,
        V_init: Optional[Initializer] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable | None = None,
        subsolver: str | Callable | None = None,
        substeps: int | None = None,
        name: str | None = None,
    ) -> "Cell":
        """Create a lumped (point) neuron population.

        Shorthand for ``Cell(Morphology.from_soma(length=length,
        radius=radius), pop_size=pop_size, cv_policy=Lumped(), ...)``. The
        cell is an ordinary :class:`Cell`: paint mechanisms on it, place
        clamps and synapses on its one site, read and write it through its
        views, train its parameters, record and ``run`` it, and add it to a
        :class:`~braincell.Network`. ``V`` and ``spike`` have shape
        ``pop_size`` (plus a leading batch axis).

        Parameters
        ----------
        pop_size : int or tuple of int, optional
            Population shape. Default is 1.
        length, radius : Quantity, optional
            Soma cylinder; its lateral area ``2 pi radius length`` is the
            membrane area. Defaults (10 um, 5 um) match
            :class:`~braincell.SingleCompartment`.
        V_th, V_init, spk_fun, subsolver, substeps, name
            As for :class:`Cell`. ``V_init=None`` starts at the resting
            potential of the cable property (default -65 mV).
        solver : str or Callable or None, optional
            Any single-step integrator of :func:`braincell.quad.get_integrator`
            except ``"staggered"`` and ``"dhs_voltage"``. Default ``None``
            selects ``"ralston2"``, which :meth:`init_state` warns about when
            an HH gate is too fast for it (see :class:`Cell`'s ``solver``).

        Returns
        -------
        Cell
            The declared, uninitialized lumped cell.

        Raises
        ------
        ValueError
            If ``solver`` is ``"staggered"`` or ``"dhs_voltage"``, or
            ``length`` / ``radius`` is not a positive scalar length.
        TypeError
            If ``length`` or ``radius`` is not a length quantity.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import braincell as bc
            >>> cell = bc.Cell.point(100, length=20 * u.um, radius=10 * u.um, V_init=-65 * u.mV)
            >>> cell.paint(
            ...     bc.filter.AllRegion(),
            ...     bc.mech.Ion("SodiumFixed", E=50 * u.mV),
            ...     bc.mech.Channel("Na_HH1952", g_max=120 * u.mS / u.cm**2),
            ... )  # doctest: +SKIP
            >>> cell.place(bc.mech.CurrentClamp(durations=50 * u.ms, amplitudes=0.1 * u.nA))  # doctest: +SKIP
            >>> cell.solver_name
            'ralston2'
        """
        return cls(
            _Morphology.from_soma(length=length, radius=radius),
            pop_size=pop_size,
            cv_policy=Lumped(),
            V_th=V_th,
            V_init=V_init,
            spk_fun=spk_fun,
            solver=solver,
            subsolver=subsolver,
            substeps=substeps,
            name=name,
        )

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    @property
    def V_th(self):
        return self._V_th if self._V_th_parameter is None else self._V_th_parameter.value

    @V_th.setter
    def V_th(self, value) -> None:
        # ``install_cell_runtime`` overwrites V_th with a vectorised
        # version during ``init_state``; that call is permitted because
        # ``_initialized`` is still False at that point. After
        # ``init_state`` completes, the guard rejects further assignment.
        self._raise_if_initialized("assign V_th")
        if self._V_th_parameter is not None:
            from braincell.trainable._targets import require_unbound

            require_unbound(self, "threshold", "V_th", range(self._population_size * self.n_compartment), "threshold")
            self._V_th_parameter.value = bridge.fill_like(self.varshape, value)
        self._V_th = value

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._raise_if_initialized("assign V_init")
        self._V_init = value
        self._V_init_materialized = None

    @property
    def solver(self):
        return self._solver_fn

    @solver.setter
    def solver(self, value) -> None:
        self._raise_if_initialized("assign solver")
        self._solver_name, self._solver_fn = _resolve_cell_solver(value, lumped=self._is_lumped)
        self._solver_declaration = value

    @property
    def solver_name(self) -> str:
        return self._solver_name

    @property
    def subsolver(self):
        """Return the effective Markov/kinetic-ion integrator callable."""
        return self._subsolver_fn

    @property
    def subsolver_name(self) -> str:
        """Return the effective Markov/kinetic-ion integrator name."""
        return self._subsolver_name

    @property
    def substeps(self) -> int:
        """Return the effective Markov/kinetic-ion substep count."""
        return self._substeps

    @property
    def membrane_linearizer(self) -> str:
        return self._membrane_linearizer

    @membrane_linearizer.setter
    def membrane_linearizer(self, value: str) -> None:
        self._raise_if_initialized("assign membrane_linearizer")
        self._membrane_linearizer = _validate_membrane_linearizer(value)

    @property
    def spk_fun(self):
        return self._spk_fun

    @spk_fun.setter
    def spk_fun(self, value) -> None:
        self._raise_if_initialized("assign spk_fun")
        self._spk_fun = value

    @property
    def name(self) -> str | None:
        return self._name

    # ------------------------------------------------------------------
    # Declaration mutators

    def _place_selected(self, population_indices, locset, mechanisms) -> None:
        self._raise_if_initialized("place()")
        if any(not isinstance(mechanism, SynapsePlacement) for mechanism in mechanisms):
            raise TypeError("Cell population-specific place() currently supports Synapse point mechanisms only.")
        self._validate_synapse_names(mechanisms)
        normalized_indices = tuple(population_indices)
        if _is_per_cell_locset_sequence(locset):
            self._place_per_cell(normalized_indices, tuple(locset), mechanisms)
            return
        aligned = isinstance(locset, LocsetBatch)
        if aligned and len(locset) != len(normalized_indices):
            raise ValueError(
                "LocsetBatch batch rows must match the selected population size, "
                f"got {len(locset)!r} rows for {len(normalized_indices)!r} cells."
            )
        previous_rules = self._place_rules
        self._place_rules = merge_place_rules(
            previous_rules,
            (
                normalize_place_rule(
                    locset,
                    mechanisms,
                    population_indices=normalized_indices,
                    aligned=aligned,
                    morpho=self._morpho,
                ),
            ),
        )
        try:
            self.discretize()
            if self._is_lumped:
                self._check_lumped_sites()
        except Exception:
            self._place_rules = previous_rules
            raise

    def _place_per_cell(self, population_indices, locsets, mechanisms) -> None:
        """Append one independently sized locset row per selected cell."""
        if len(locsets) != len(population_indices):
            raise ValueError(
                "Per-cell locset rows must match the selected population size, "
                f"got {len(locsets)!r} rows for {len(population_indices)!r} cells."
            )
        if any(not isinstance(locset, (LocsetExpr, LocsetMask)) for locset in locsets):
            raise TypeError("Each per-cell location row must be a LocsetExpr or LocsetMask.")
        if any(not isinstance(mechanism, SynapsePlacement) for mechanism in mechanisms):
            raise TypeError("Per-cell locset sequences currently support Synapse point mechanisms only.")

        lengths = tuple(_resolved_locset_length(locset, self._morpho) for locset in locsets)
        per_mechanism_rows = tuple(_split_synapse_rows(mechanism, lengths=lengths) for mechanism in mechanisms)
        incoming = []
        for row, (population_index, locset) in enumerate(zip(population_indices, locsets)):
            row_mechanisms = tuple(rows[row] for rows in per_mechanism_rows)
            for source, materialized in zip(mechanisms, row_mechanisms):
                self._synapse_origins[id(materialized)] = source
            incoming.append(
                normalize_place_rule(
                    locset,
                    row_mechanisms,
                    population_indices=(int(population_index),),
                    aligned=False,
                    morpho=self._morpho,
                )
            )
        previous_rules = self._place_rules
        self._place_rules = merge_place_rules(previous_rules, tuple(incoming))
        try:
            self.discretize()
            if self._is_lumped:
                self._check_lumped_sites()
        except Exception:
            self._place_rules = previous_rules
            raise

    def _set_runtime_voltage_parameters(self, scope, parameters) -> None:
        self._raise_if_not_initialized("CellView.set(); use Cell constructor parameters before init_state()")
        unknown = set(parameters) - {"V_init", "V_th"}
        if unknown:
            raise KeyError(f"CellView.set() does not support parameters {sorted(unknown)!r}.")
        pairs = tuple(scope.pairs)
        if not pairs:
            return
        pop = np.asarray([p for p, _ in pairs], dtype=np.int32)
        cv = np.asarray([c for _, c in pairs], dtype=np.int32)
        indices = (pop, cv) if self.pop_size else (cv,)
        pending = {}
        for name, value in parameters.items():
            if value is None:
                raise TypeError(f"CellView {name} must be a voltage quantity, not None.")
            if scope.spatially_restricted:
                rows = _normalize_selected_voltage_parameter(value, count=1, n_cv=len(pairs), name=name)
            else:
                rows = _normalize_selected_voltage_parameter(
                    value, count=len(scope.population_indices), n_cv=self.n_cv, name=name
                )
            values = jnp.asarray(u.math.stack(rows).to_decimal(u.mV)).reshape(-1)
            if name == "V_th":
                from braincell.trainable._targets import require_unbound

                require_unbound(self, "threshold", "V_th", pop * self.n_cv + cv, "threshold")
                # A lumped Cell's threshold has no CV axis (shape pop_size).
                current = self.V_th if self._is_lumped else bridge.fill_like(self.varshape, self.V_th)
            else:
                current = self._V_init_materialized.value
            # Flatten the population axes for the logical selector's indices.
            shape = current.shape
            flat = current.reshape((self._population_size, self.n_cv)) if self.pop_size else current
            updated = flat.at[indices].set(u.Quantity(values, u.mV))
            pending[name] = updated.reshape(shape)
        for name, value in pending.items():
            if name == "V_init":
                if not self._is_lumped:
                    self._runtime_V_init.value = value
                    mask = self._runtime_V_init_mask.value
                    flat_mask = mask.reshape((self._population_size, self.n_cv)) if self.pop_size else mask
                    self._runtime_V_init_mask.value = flat_mask.at[indices].set(True).reshape(mask.shape)
                self._V_init_materialized.value = value
            elif self._V_th_parameter is not None:
                self._V_th_parameter.value = value
            else:
                self._V_th = value
        self._run_loop_cache.clear()

    def _materialize_population_parameter(self, name: str):
        if name == "V_th":
            value = bridge.fill_like(self.varshape, self.V_th)
        elif name == "V_init":
            initializer = self._V_init
            if initializer is None:
                initializer = bridge.cv_value_vector(self, attr_name="v")
            elif not callable(initializer):
                initializer = bridge.fill_like(self.varshape, initializer)
            value = braintools.init.param(initializer, self.varshape)
        else:  # pragma: no cover - internal invariant
            raise KeyError(name)
        # Integer-valued declarations (``-65 * u.mV``) and callable results
        # would otherwise make the step carry int-typed while the update is
        # float-typed, which only fails deep inside ``run``.
        value = _as_float_voltage(value)
        if name == "V_init" and self._runtime_V_init is not None:
            value = u.math.where(self._runtime_V_init_mask.value, self._runtime_V_init.value, value)
        return value

    def _selected_population_parameter(self, name: str, population_indices):
        if name == "V_th" and self._initialized:
            values = self.V_th
        elif name == "V_init" and self._V_init_materialized is not None:
            values = self._V_init_materialized.value
        else:
            if name == "V_init" and callable(self._V_init):
                raise RuntimeError(
                    "CellView.V_init cannot inspect a callable initializer before init_state(); "
                    "initialize the root Cell and inspect CellView.V_init or CellView.V."
                )
            values = self._materialize_population_parameter(name)
        return _select_population_value(
            values,
            population_indices=tuple(population_indices),
            population_size=self._population_size,
        )

    def paint(self, *args) -> "Cell":
        """Paint density mechanisms onto a region of every population member.

        Parameters
        ----------
        *args
            ``region, *mechanisms``, or just ``*mechanisms`` to paint the
            whole cell (``AllRegion()``). ``region`` is a
            :class:`~braincell.filter.RegionExpr`; each mechanism is a
            :class:`~braincell.mech.Density` (``Channel``, ``Ion``) or a
            :class:`~braincell.mech.CableProperty`.

        Returns
        -------
        Cell
            This Cell, for chaining.

        Raises
        ------
        TypeError
            If no mechanism is given, or an argument has the wrong type.
        ValueError
            If a region is given without mechanisms, or if an earlier
            declaration names branches by an index that an edit has since
            shifted (see Notes).
        RuntimeError
            After :meth:`init_state`.

        Notes
        -----
        A region that names branches by default index -- ``BranchSlice(1,
        ...)``, ``SubtreeRegion(1)``, ``branch_in("index", ...)``, or a view
        such as ``cell.branch[1]`` -- is evaluated again at every rebuild,
        but the index does not follow an edit: once a branch is removed or
        moved (``Morphology.remove``, ``Morphology.reattach``), the next
        rebuild (the first grid read, or ``init_state``) raises
        ``ValueError`` rather than paint other branches. Appending a branch,
        or replacing or transforming one in place, keeps every index. Name
        the branches (``branch_in("name", "dend")``, ``SubtreeRegion("dend")``)
        to follow edits.
        """
        self._raise_if_initialized("paint()")
        if not args:
            raise TypeError("Cell.paint(...) needs at least one mechanism, optionally after a region.")
        if isinstance(args[0], (Density, CableProperty)):
            region, mechanisms = AllRegion(), args
        else:
            region, mechanisms = args[0], args[1:]
        previous_rules = self._paint_rules
        self._paint_rules = merge_paint_rules(
            previous_rules, normalize_paint_rules(region, mechanisms, morpho=self._morpho)
        )
        try:
            # Paint is a declaration. Keep the derived grid current immediately
            # so init_state() can consume it without a hidden policy pass.
            self.discretize()
        except Exception:
            self._paint_rules = previous_rules
            raise
        return self

    def place(
        self,
        locset: LocsetExpr | LocsetMask | LocsetBatch | None = None,
        *mechanisms,
    ) -> "Cell":
        """Place point mechanisms at shared or per-cell locations.

        Parameters
        ----------
        locset : LocsetExpr, LocsetMask, LocsetBatch, or sequence, optional
            One shared locset, an aligned rectangular batch, or one possibly
            ragged locset per population member. A lumped cell
            (``cv_policy=Lumped()``) has one site, its CV midpoint: the locset
            may be omitted (``cell.place(clamp)``), and an explicit one must
            resolve to that midpoint (``RootLocation(0.5)``,
            ``cell.cv_midpoints``).
        *mechanisms : Point
            Point-mechanism declarations placed independently at each row.

        Returns
        -------
        Cell
            This Cell for chaining.

        Raises
        ------
        TypeError
            If the locset is omitted on a cable cell.
        ValueError
            On a lumped cell, if a location is not the CV midpoint. Also if
            an earlier declaration's expression names branches by an index
            that an edit has since shifted (see Notes).
        IndexError
            If a :class:`~braincell.filter.LocsetMask` or
            :class:`~braincell.filter.LocsetBatch` names a branch the
            morphology does not have.
        RuntimeError
            If an earlier declaration holds a resolved mask whose branch
            indices an edit has since shifted (see Notes).

        A call that raises leaves the Cell's declarations as they were.

        Notes
        -----
        A resolved :class:`~braincell.filter.LocsetMask` or
        :class:`~braincell.filter.LocsetBatch` holds the default branch
        indices of the morphology as it is at this call. They stay valid
        when branches are appended or replaced in place; once a branch is
        removed or moved, the next rebuild (the first grid read, or
        ``init_state``) raises ``RuntimeError`` rather than place the
        mechanisms on other branches. A location expression such as
        ``at("axon", 0.5)`` is evaluated again at every rebuild and follows
        the edit. An expression that names its branch by index, such as
        ``at(1, 0.5)``, is evaluated again too, but the index does not
        follow the edit: after a removal or a move the next rebuild raises
        ``ValueError``.
        """
        self._raise_if_initialized("place()")
        # A user-supplied mask holds branch indices the Cell must watch; the
        # lumped default below is the Cell's own single CV midpoint.
        index_morpho = self._morpho
        if locset is None or isinstance(locset, Point):
            index_morpho = None
            mechanisms = (() if locset is None else (locset,)) + tuple(mechanisms)
            if not self._is_lumped:
                raise TypeError(
                    "Cell.place(...) needs a locset on a cable Cell; only a lumped Cell "
                    "(cv_policy=Lumped()) has a single site to place on by default."
                )
            locset = self.cv_midpoints
        self._validate_synapse_names(mechanisms)
        if _is_per_cell_locset_sequence(locset):
            if len(self.pop_size) != 1:
                raise ValueError(f"Per-cell locset placement requires one-dimensional pop_size; got {self.pop_size!r}.")
            self._place_per_cell(tuple(range(int(self.pop_size[0]))), tuple(locset), mechanisms)
            return self
        population_indices = None
        aligned = isinstance(locset, LocsetBatch)
        if aligned:
            if len(self.pop_size) != 1:
                raise ValueError(f"LocsetBatch placement requires one-dimensional pop_size; got {self.pop_size!r}.")
            population_indices = tuple(range(int(self.pop_size[0])))
            if len(locset) != len(population_indices):
                raise ValueError(
                    "LocsetBatch batch rows must match the population size, "
                    f"got {len(locset)!r} rows for {len(population_indices)!r} cells."
                )
        previous_rules = self._place_rules
        self._place_rules = merge_place_rules(
            previous_rules,
            (
                normalize_place_rule(
                    locset,
                    mechanisms,
                    population_indices=population_indices,
                    aligned=aligned,
                    morpho=index_morpho,
                ),
            ),
        )
        try:
            self.discretize()
            if self._is_lumped:
                self._check_lumped_sites()
        except Exception:
            self._place_rules = previous_rules
            raise
        return self

    def _check_lumped_sites(self) -> None:
        """Require every placement of a lumped cell to sit at the CV midpoint."""
        (cv,) = self.cvs
        for placement in self.point_placements:
            if placement.branch_id != cv.branch_id or abs(float(placement.branch_x) - float(cv.midpoint)) > 1e-9:
                raise ValueError(
                    "A lumped Cell (cv_policy=Lumped()) has a single site, the CV midpoint "
                    f"(branch {cv.branch_id}, x={float(cv.midpoint):g}); got a location at "
                    f"branch {placement.branch_id}, x={float(placement.branch_x):g}. "
                    "Omit the locset or use RootLocation(0.5) / cell.cv_midpoints."
                )

    def bind_synapse_input(self, synapse: str, source, *, weight=1.0, transform=None) -> "Cell":
        """Deprecated compatibility adapter for per-boundary event payloads.

        .. deprecated:: 0.3
           Create an ``EventSource`` and connect it with
           ``braincell.connect()``; this method is removed in braincell 0.4.

        Parameters
        ----------
        synapse : str
            Synapse instance name, matching ``braincell.mech.Synapse(name=...)``
            or its default instance name.
        source : array-like or callable
            Presynaptic drive source. Callables are evaluated every step; this
            supports bindings such as ``lambda: pre_cell.spike.value``.
        weight : array-like, optional
            Multiplicative weight applied to ``source``.
        transform : callable, optional
            Optional mapping called as ``transform(source_value)`` before
            weighting, useful when the source shape does not directly broadcast
            to the target synapse shape.
        """
        _BIND_SYNAPSE_INPUT.warn()
        key = str(synapse)
        self._synapse_input_bindings.setdefault(key, []).append((source, weight, transform))
        return self

    # ------------------------------------------------------------------
    # Static discretization (valid in both phases)

    def _invalidate_discretization_cache(self) -> None:
        self._discretization_cache = None
        self._discretization_cache_key = None
        self._grid_fingerprint_cache = None
        self._cv_collection_cache = None
        self._synapse_store_cache = None
        self._clamp_store_cache = None
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        self._root_scope_cache = None
        self._run_loop_cache.clear()

    def _discretization_key(self) -> tuple[object, ...]:
        # ``Morphology`` is mutable and shared -- before initialization
        # ``cell.morpho`` hands the caller the very tree this cell
        # discretizes -- so identity alone cannot say whether it still looks
        # the way it did when the cache was filled. ``revision`` advances by
        # one on every successful edit; pairing it with the identity is the
        # same test ``braincell.filter.SelectionCache`` applies.
        #
        # The ``id()`` is sound here, unlike for a temporary whose address
        # can be recycled: this cell holds ``self._morpho`` alive for as
        # long as the key it appears in. The morphology setter and reset()
        # invalidate the cache when they reassign the attribute; init_state
        # swaps in a frozen copy of the same structure and re-keys the cache
        # to it, and a failed init_state restores the previous key.
        return (
            id(self._morpho),
            self._morpho.revision,
            self._discretization_policy,
            self._paint_rules,
            self._place_rules,
        )

    @property
    def _discretization(self) -> Discretization:
        if self._initialized:
            if self._discretization_cache_key != self._discretization_key():
                raise RuntimeError("Initialized morphology has changed; reset() before changing declarations.")
            return self._discretization_cache
        key = self._discretization_key()
        if self._discretization_cache is not None and self._discretization_cache_key == key:
            return self._discretization_cache
        self.discretize()
        return self._discretization_cache

    def discretize(self) -> "Cell":
        """Atomically rebuild the declaration preview before initialization.

        Creation, policy assignment, initialization and dirty preview reads
        call this method automatically. Existing selections expire on success.
        Runtime parameters never trigger rediscretization.

        Returns
        -------
        Cell
            This Cell with a refreshed discrete preview.

        Raises
        ------
        RuntimeError
            If already initialized. Call ``reset()`` before editing declarations.
            Also if a declaration holding branch indices -- a resolved
            ``LocsetMask``/``LocsetBatch`` given to :meth:`place`, or a
            :class:`~braincell.CVPerBranchList` policy -- was made before an
            edit of this morphology that shifted those indices (a branch
            removal or a move).
        ValueError
            If a :meth:`paint` or :meth:`place` expression that names
            branches by default index (``BranchSlice(1, ...)``,
            ``at(1, 0.5)``, ``cell.branch[1]``) was declared before such an
            edit.
        """
        self._raise_if_initialized("discretize()")
        self._raise_if_branch_indices_moved()
        discretization = build_discretization(
            self._morpho,
            policy=self._discretization_policy,
            paint_rules=self._paint_rules,
            place_rules=self._place_rules,
        )
        self._invalidate_discretization_cache()
        self._discretization_cache = discretization
        self._discretization_cache_key = self._discretization_key()
        self._view_generation += 1
        return self

    def _raise_if_branch_indices_moved(self) -> None:
        """Refuse to rebuild from branch indices that an edit has since shifted.

        A ``LocsetMask``/``LocsetBatch`` place rule and a policy listing CV
        counts per branch (:class:`CVPerBranchList`) hold default branch
        indices of ``self._morpho`` at the revision they were declared.
        Appending a branch, or replacing or transforming one in place, keeps
        every existing index; removing or moving a branch
        (``Morphology.remove``, ``Morphology.reattach``) moves the
        morphology's index watermark past that revision, and replaying the
        integers would put the declaration on other branches without a word. Declarations made on another morphology object are not
        checked.

        A paint or place expression that names branches by default index
        (``BranchSlice(1, ...)``, ``at(1, 0.5)``, ``branch_in("index", ...)``)
        records the same basis and is checked the same way (MORPH-17).

        Raises
        ------
        ValueError
            Naming the first such expression, its call site and its indices.
        RuntimeError
            Naming the first such mask or policy.
        """
        new_cell = "Declare it on a new Cell"
        for rule in self._paint_rules:
            if self._held_indices_moved(rule._index_basis):
                where = "" if rule.origin is None else f" at {rule.origin}"
                raise ValueError(
                    moved_index_message(f"Cell.paint(...){where}", rule.region, rule._index_basis[1], new_cell)
                )
        for rule in self._place_rules:
            if self._held_indices_moved(rule._index_basis):
                where = "" if rule.origin is None else f" at {rule.origin}"
                if isinstance(rule.locset, LocsetExpr):
                    raise ValueError(
                        moved_index_message(f"Cell.place(...){where}", rule.locset, rule._index_basis[1], new_cell)
                    )
                raise RuntimeError(
                    f"Cell.place(...){where} was given a resolved {type(rule.locset).__name__}, whose branch "
                    f"indices are those of morphology revision {rule._index_basis[1]}; a branch has since been "
                    "removed or moved, so they no longer name the same branches. Place a location expression "
                    'such as at("axon", 0.5) instead of an evaluated mask, on a newly declared Cell.'
                )
        if self._held_indices_moved(self._cv_policy_index_basis):
            raise RuntimeError(
                f"cv_policy {self._discretization_policy!r} lists CV counts by default branch index, as of "
                f"morphology revision {self._cv_policy_index_basis[1]}; a branch has since been removed or "
                "moved, so the counts no longer match their branches. Assign cell.cv_policy again for the "
                "current branches."
            )

    def _check_view_generation(self, generation: int) -> None:
        if not self._initialized:
            _ = self._discretization
        if generation != self._view_generation:
            raise RuntimeError(
                "This discrete View is stale; select it again after discretize(), init_state() or reset()."
            )

    def _root_scope(self) -> _CellScope:
        """Return the cached unrestricted population/CV scope for this Cell.

        The root scope materializes one entry per ``(population, CV)`` pair, so
        rebuilding it per attribute access costs ``pop_size * n_cv`` tuples on
        every ``cell.soma``, ``cell.channels``, or ``cell.record`` lookup. It
        depends only on the discretization and is invalidated alongside it.
        """
        _ = self._discretization
        if self._root_scope_cache is None:
            self._root_scope_cache = _CellScope.root(self)
        return self._root_scope_cache

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self) -> tuple[CV, ...]:
        records = self._discretization.cvs
        if self._cv_collection_cache is None:
            self._cv_collection_cache = CVCollectionView(self, records)
        return self._cv_collection_cache

    @property
    def cv_midpoints(self) -> LocsetMask:
        """Return one resolved continuous midpoint for every control volume."""
        cvs = self.cvs
        return LocsetMask.from_columns(
            [cv.branch_id for cv in cvs],
            [cv.midpoint for cv in cvs],
        )

    @property
    def cv_tree(self) -> CVTree:
        return self._discretization.cv_tree

    @property
    def node_tree(self) -> NodeTree:
        return self._discretization.node_tree

    def topology_data(self, level: str = "node", *, value=None, region=None, locset=None):
        """Return this cell's topology as a graph for :func:`braintools.visualize.morphology.plot_topology_graph`.

        The cell can be drawn at three granularities, from finest to
        coarsest. ``"node"`` has one graph node per runtime point, the level
        the solver works at, so only it can show per-point runtime state.
        ``"cv"`` has one node per control volume, and ``"branch"`` one per
        morphology branch (topology only: a branch has no single runtime
        value).

        A graph either highlights a selection (``region`` / ``locset``) or
        carries values (``value``), never both.

        Parameters
        ----------
        level : {"node", "cv", "branch"}, optional
            Topology abstraction level. ``"node"`` needs an initialized cell;
            ``"cv"`` and ``"branch"`` also work on a declaration-only cell.
        value : object, optional
            Values to colour by. Not supported at ``level="branch"``. One of:

            - an array in the level's own space (``n_point`` or ``n_cv``), or
              in the other space, which is mapped across;
            - a ``(T, n)`` array, one row per frame (pick one with
              ``plot_topology_graph(..., frame=i)``);
            - ``"V"`` or ``"voltage"``;
            - ``("ion", ion_name, field)``, ``("channel", class_name, field)``
              or ``("layout_id", layout_id, field)``.
        region : braincell.filter.RegionExpr or braincell.filter.RegionMask, optional
            Continuous selection to highlight. At ``"node"`` and ``"cv"`` it
            is reduced to CV granularity (membrane-area fractions; at
            ``"node"`` each CV is attributed to its midpoint point); at
            ``"branch"`` the fractions are lengths along each branch.
        locset : braincell.filter.LocsetExpr or braincell.filter.LocsetMask, optional
            Discrete locations to highlight at full intensity, each mapped to
            the CV that owns it. Not supported at ``level="branch"``.

        Returns
        -------
        braintools.visualize.morphology.TopologyGraph
            Graph with ``level`` set, and ``highlight_fractions`` or
            ``values`` and ``value_label`` filled in when requested.

        Raises
        ------
        ValueError
            If ``level`` is unknown; if ``value`` is combined with ``region``
            or ``locset``; if ``level="branch"`` is given ``locset`` or
            ``value``; if ``level="cv"`` finds more than one root CV; if a
            value cannot be mapped into the level's space; for a
            multi-member population; or if a channel selection is ambiguous.
        RuntimeError
            If ``level="node"`` is used before :meth:`init_state`.
        AttributeError
            If a named mechanism field does not exist.
        KeyError
            If a ``layout_id`` does not exist.

        Examples
        --------

        .. code-block:: python

            >>> import braincell as bc
            >>> from braincell.morph._testing import make_soma_dend_tree
            >>> cell = bc.Cell(make_soma_dend_tree(), cv_policy=bc.CVPerBranch())
            >>> graph = cell.topology_data("cv", region=bc.filter.BranchSlice(1, 0.0, 0.5))
            >>> graph.level, graph.node_ids
            ('cv', (0, 1))
            >>> round(graph.highlight_fractions[1], 4)
            0.5833

        Draw it with ``braintools.visualize.morphology.plot_topology_graph``:

        .. code-block:: python

            >>> import braintools.visualize.morphology as bvm
            >>> ax = bvm.plot_topology_graph(graph)
        """
        from braintools.visualize.morphology import TopologyGraph

        levels = ("node", "cv", "branch")
        if level not in levels:
            options = ", ".join(repr(name) for name in levels)
            raise ValueError(f"Cell.topology_data(...) level must be one of {{{options}}}, got {level!r}.")
        caller = f"Cell.topology_data(level={level!r})"
        if value is not None and (region is not None or locset is not None):
            raise ValueError(f"{caller} does not support value together with region/locset highlighting.")

        if level == "branch":
            unsupported = [name for name, given in (("locset", locset), ("value", value)) if given is not None]
            if unsupported:
                raise ValueError(
                    f"{caller} does not support: {', '.join(unsupported)}. Branch-level graphs are topology-only."
                )
            morpho = self.morpho
            fractions = (
                None if region is None else field_resolution.branch_coverage_fractions(self, region, caller=caller)
            )
            return TopologyGraph(
                node_ids=tuple(branch.index for branch in morpho.branches),
                edges=tuple((edge.parent.index, edge.child.index) for edge in morpho.edges),
                root_id=int(morpho.root.index),
                level="branch",
                highlight_fractions=fractions,
            )

        if level == "node":
            field_resolution.require_initialized(self, caller)
            tree = self.node_tree
            node_ids = tuple(int(node.id) for node in tree.nodes)
            edges = tuple((int(edge.parent_node_id), int(edge.child_node_id)) for edge in tree.edges)
            root_id = int(tree.root_node_id)
            highlight = field_resolution.node_highlight_fractions
            resolve = field_resolution.resolve_node_field_values
        else:
            cvs = self.cvs
            root_ids = [cv.id for cv in cvs if cv.parent_cv is None]
            if len(root_ids) != 1:
                raise ValueError(f"{caller} expects exactly one root CV, got {root_ids!r}.")
            node_ids = tuple(int(cv.id) for cv in cvs)
            edges = tuple((int(cv.parent_cv), int(cv.id)) for cv in cvs if cv.parent_cv is not None)
            root_id = int(root_ids[0])
            highlight = field_resolution.cv_highlight_fractions
            resolve = field_resolution.resolve_cv_field_values

        fractions = None
        values = None
        value_label = None
        if region is not None or locset is not None:
            fractions = highlight(self, region=region, locset=locset, caller=caller)
        elif value is not None:
            if not isinstance(value, (str, tuple)) and np.ndim(value) == 2:
                frames = [resolve(self, row, caller=caller) for row in value]
                value_label = frames[0][1]
                rows = [frame[0] for frame in frames]
                values = u.math.stack(rows) if isinstance(rows[0], u.Quantity) else np.stack(rows)
            else:
                values, value_label = resolve(self, value, caller=caller)
        return TopologyGraph(
            node_ids=node_ids,
            edges=edges,
            root_id=root_id,
            level=level,
            values=values,
            value_label=value_label,
            highlight_fractions=fractions,
        )

    @property
    def cv_contexts(self) -> tuple[CVContext, ...]:
        """Return read-only spatial contexts in stable CV order.

        Returns
        -------
        tuple of CVContext
            Geometry and path-distance metadata used to resolve callable
            cable and density parameters.
        """
        return self._discretization.cv_contexts

    @property
    def point_placements(self):
        """Return point-mechanism placements in stable declaration order.

        Returns
        -------
        tuple of PointPlacement
            Static placement records available before and after initialization.
        """
        return self._discretization.point_placements

    @property
    def synapses(self) -> SynapseView:
        """Return a view over all logical point-synapse instances."""
        return SynapseView(self)

    @property
    def clamps(self) -> ClampView:
        """Return a view over all logical current-clamp instances."""
        return ClampView(self)

    @property
    def connections(self):
        """Return a unified view over all direct event-routing rows."""
        from braincell.network.connection import ConnectionView

        return ConnectionView(self._get_connection_store())

    @property
    def event_outputs(self) -> EventOutputCollection:
        """Return named live event-output ports for this cell population."""
        return EventOutputCollection(self)

    @property
    def reductions(self) -> ReductionViewCollection:
        """Return all Cell-local registered reduction models."""
        return ReductionViewCollection(self, tuple(range(self._population_size)))

    @property
    def outputs(self) -> Mapping[str, object]:
        """Return the selected model's initialized raw outputs."""
        return self._selected_outputs(tuple(range(self._population_size)))

    def add_reduction(self, name: str, model):
        """Register one interchangeable reduced model under a Cell-local name."""
        self._raise_if_initialized("add a reduction model")
        if not isinstance(name, str) or not name:
            raise ValueError("Reduction model name must be a non-empty string.")
        if name == "detailed":
            raise ValueError("Reduction model name 'detailed' is reserved for the full Cell model.")
        if name in self._reduction_models:
            raise ValueError(f"Cell already has a reduction model named {name!r}.")
        required = ("init_state", "update", "reset_state", "reset")
        missing = tuple(method for method in required if not callable(getattr(model, method, None)))
        if missing:
            raise TypeError(f"Reduction model {type(model).__name__!r} is missing callable methods {missing!r}.")
        self._reduction_models[name] = model
        self._run_loop_cache.clear()
        return self.reductions[name]

    def use_model(self, name: str = "detailed") -> "Cell":
        """Select the detailed model or one registered reduction for the next initialization."""
        self._raise_if_initialized("select a Cell execution model")
        if name != "detailed" and name not in self._reduction_models:
            raise KeyError(f"Unknown Cell model {name!r}; available reductions: {tuple(self._reduction_models)!r}.")
        self._selected_model_name = name
        self._run_loop_cache.clear()
        self._compiled_recording_cache.clear()
        return self

    def _get_spike_event_source(self) -> _CellSpikeSource:
        if self._spike_event_source_cache is None:
            self._spike_event_source_cache = _CellSpikeSource(self)
        return self._spike_event_source_cache

    def _get_synapse_store(self) -> _SynapseStore:
        """Return the private logical synapse store for the current declaration."""
        if self._synapse_store_cache is None:
            self._synapse_store_cache = _SynapseStore(self)
        return self._synapse_store_cache

    def _get_clamp_store(self) -> _ClampStore:
        """Return the logical current-clamp store for this declaration."""
        if self._clamp_store_cache is None:
            self._clamp_store_cache = _ClampStore(self)
        return self._clamp_store_cache

    def _get_connection_store(self):
        """Return the private Cell-owned routing-row store."""
        if self._connection_store_cache is None:
            from braincell.network.connection import _ConnectionStore

            self._connection_store_cache = _ConnectionStore(self)
        return self._connection_store_cache

    def _validate_synapse_names(self, mechanisms) -> None:
        """Reject one logical group name being reused across model types."""
        declared = raise_on_name_type_conflict(
            (mechanism.instance_name, mechanism.synapse_type)
            for rule in self._place_rules
            for mechanism in rule.mechanisms
            if isinstance(mechanism, SynapsePlacement)
        )
        raise_on_name_type_conflict(
            (
                (mechanism.instance_name, mechanism.synapse_type)
                for mechanism in mechanisms
                if isinstance(mechanism, SynapsePlacement)
            ),
            known=declared,
        )

    @property
    def recordings(self) -> Mapping[str, RecordingSpec]:
        """Return a read-only snapshot of Cell recording declarations."""
        return MappingProxyType(dict(self._recording_specs))

    def _add_recording(self, spec: RecordingSpec) -> RecordingSpec:
        self._raise_if_initialized("add a recording")
        if spec.name in self._recording_specs:
            raise ValueError(f"Cell already has a recording named {spec.name!r}.")
        target = spec.scope.target
        index_basis = None
        if isinstance(target, (RegionMask, LocsetMask, LocsetBatch)) or held_branch_indices(target):
            # Resolved masks and expressions naming branches by index (MORPH-16, MORPH-17).
            index_basis = self._index_basis()
        self._recording_bases[spec.name] = (self._morpho, self._morpho.revision, self._grid_fingerprint(), index_basis)
        self._recording_specs[spec.name] = spec
        self._compiled_recording_cache.clear()
        self._run_loop_cache.clear()
        return spec

    def _grid_fingerprint(self) -> tuple:
        """``(branch_id, prox, dist)`` of every CV: equal on two grids that have the same CVs."""
        if self._grid_fingerprint_cache is None:
            self._grid_fingerprint_cache = tuple((cv.branch_id, cv.prox, cv.dist) for cv in self._discretization.cvs)
        return self._grid_fingerprint_cache

    def _resolve_recordings(self) -> None:
        """Select each recording again from its declaration if its grid or tree changed (MORPH-15).

        A recording keeps the CVs it selected when it was declared. If the
        grid has other CVs now (a new ``cv_policy`` or an edit), or the tree
        was edited, the selection is made again from what it was declared
        with -- the population members and the region or locset -- so it
        records the same branches and locations. Recordings on an unchanged
        tree and grid are kept as they are.

        Raises
        ------
        RuntimeError
            Before changing any recording, naming the first one that cannot
            be selected again: it was declared on CVs (``cv[...]``) or on
            nested selections and the grid changed, or on a resolved mask
            whose branch indices a removal or move has shifted.
        ValueError
            Likewise, naming the recording and its indices, if its expression
            names branches by default index (``cell.on(BranchSlice(1, ...))``,
            ``cell.branch[1]``) and a removal or move has shifted them.
        KeyError
            Naming the recording, if its expression names a branch the tree
            no longer has.
        """
        if not self._recording_specs:
            return
        morpho, grid = self._morpho, self._grid_fingerprint()
        same_grid: dict[int, bool] = {}
        updated = {}
        for name, spec in self._recording_specs.items():
            declared_morpho, revision, declared_grid, index_basis = self._recording_bases[name]
            if id(declared_grid) not in same_grid:
                same_grid[id(declared_grid)] = declared_grid is grid or declared_grid == grid
            grid_kept = same_grid[id(declared_grid)]
            if grid_kept and declared_morpho is morpho and revision == morpho.revision:
                continue
            target = spec.scope.target
            if isinstance(target, _Undeclarable):
                if grid_kept:
                    # Same CVs: the selection still names what it named.
                    updated[name] = spec
                    continue
                raise RuntimeError(
                    f"Recording {name!r} was declared on {target.reason}, and the grid has changed since "
                    "(a new cv_policy or an edit of the morphology), so which CVs it means is unknown. Restore "
                    "the grid it was declared on, or record on a new Cell with a region or location "
                    "(cell.on(...), cell.loc(...))."
                )
            if self._held_indices_moved(index_basis):
                if not isinstance(target, (RegionMask, LocsetMask, LocsetBatch)):
                    raise ValueError(
                        moved_index_message(f"Recording {name!r}", target, index_basis[1], "Record on a new Cell")
                    )
                raise RuntimeError(
                    f"Recording {name!r} selects a resolved {type(target).__name__}, whose branch indices are "
                    f"those of morphology revision {index_basis[1]}; a branch has since been removed or moved, "
                    "so they no longer name the same branches. Record on a new Cell with a location expression "
                    'such as at("axon", 0.5) instead of an evaluated mask.'
                )
            try:
                updated[name] = replace(spec, scope=spec.scope.replayed(self))
            except KeyError as error:
                missing = missing_branch_name(morpho, error)
                if missing is None:
                    raise
                raise KeyError(
                    f"Recording {name!r} selects branch {missing!r}, which the morphology (revision "
                    f"{morpho.revision}) does not have: an edit removed it. Attach a branch named {missing!r} "
                    "again, or record on a new Cell."
                ) from error
        for name, spec in updated.items():
            self._recording_specs[name] = spec
            self._recording_bases[name] = (morpho, morpho.revision, grid, self._recording_bases[name][3])
        if updated:
            self._compiled_recording_cache.clear()
            self._run_loop_cache.clear()

    def _resolve_declarations(self) -> None:
        """Bring recordings and connection rows declared on an older grid or tree up to date (MORPH-15).

        ``init_state`` calls this first, and ``Network.init_state`` calls it
        on every Cell before initializing any, so that an error leaves the
        whole Network declaring. It raises before changing anything.

        A detector connected to this Cell whose location holds branch
        indices that an edit has shifted raises here too (MORPH-17), rather
        than at its first ``cv_id`` read after initialization; a connection
        whose rows were all removed is skipped.
        """
        _ = self._discretization
        self._resolve_recordings()
        store = self._connection_store_cache
        if store is not None:
            for call in store._calls:
                moved = getattr(call.source, "_moved_location_error", None)
                error = None if moved is None else moved()
                if error is not None and store.active[store._row_of_id[call.row_ids]].any():
                    raise error
            store.follow_synapses(self._get_synapse_store())

    def _compiled_recordings(self, dt) -> tuple:
        self._raise_if_not_initialized("compile recordings")
        dt_ms = scalar_decimal(dt, u.ms)
        key = (dt_ms, tuple(self._recording_specs))
        cached = self._compiled_recording_cache.get(key)
        if cached is None:
            cached = tuple(
                compile_recording(self, spec, dt=dt)
                for spec in self._recording_specs.values()
                if recording_is_active(self, spec)
            )
            self._compiled_recording_cache[key] = cached
        return cached

    @property
    def _uses_reduction(self) -> bool:
        return self._selected_model_name != "detailed"

    def _selected_outputs(self, population_indices: tuple[int, ...]) -> Mapping[str, object]:
        if not self._initialized:
            return MappingProxyType({})
        if not self._uses_reduction:
            values = {"voltage": self.V.value}
        else:
            values = {name: state.value for name, state in self._reduction_output_states.items()}
        selected = {
            name: _select_model_output(
                value,
                population_indices=population_indices,
                population_size=self._population_size,
                batch_size=self._runtime_batch_size,
                detailed=not (self._uses_reduction or self._is_lumped),
            )
            for name, value in values.items()
        }
        return MappingProxyType(selected)

    def _validate_reduction_output(self, output) -> ReductionOutput:
        if not isinstance(output, ReductionOutput):
            raise TypeError(
                f"Reduction model {self._selected_model_name!r} must return ReductionOutput, "
                f"got {type(output).__name__!s}."
            )
        prefix = ((self._runtime_batch_size,) if self._runtime_batch_size is not None else ()) + self.pop_size
        event_shape = tuple(getattr(output.event, "shape", ()))
        if event_shape != prefix:
            raise ValueError(f"Reduction event shape must be {prefix!r} for this Cell, got {event_shape!r}.")
        for name, value in output.values.items():
            shape = tuple(getattr(value, "shape", ()))
            if shape[: len(prefix)] != prefix:
                raise ValueError(
                    f"Reduction output {name!r} must start with Cell runtime shape {prefix!r}, got {shape!r}."
                )
        return output

    def _raise_if_reduction_has_delta_inputs(self) -> None:
        """A reduction model has no membrane voltage for a delta input to move."""
        if self.delta_inputs:
            raise ValueError(
                f"Cell model {self._selected_model_name!r} is a reduction without a membrane voltage, so the "
                f"delta inputs {tuple(self.delta_inputs)!r} (add_delta_input, brainpy DeltaProj) cannot apply; "
                "remove them, or select the detailed model with use_model('detailed') before init_state()."
            )

    def _publish_reduction_output(self, output, *, initialize: bool = False) -> None:
        output = self._validate_reduction_output(output)
        if initialize:
            self._reduction_output_states = {
                name: brainstate.ShortTermState(value) for name, value in output.values.items()
            }
            self.spike = brainstate.ShortTermState(output.event)
            return
        if tuple(output.values) != tuple(self._reduction_output_states):
            raise ValueError(
                "Reduction output names must remain unchanged after init_state(); "
                f"expected {tuple(self._reduction_output_states)!r}, got {tuple(output.values)!r}."
            )
        for name, value in output.values.items():
            current = self._reduction_output_states[name].value
            expected = tuple(current.shape)
            actual = tuple(getattr(value, "shape", ()))
            if actual != expected:
                raise ValueError(f"Reduction output {name!r} changed shape from {expected!r} to {actual!r}.")
            _require_same_value_type(current, value, name=f"Reduction output {name!r}")
            self._reduction_output_states[name].value = value
        _require_same_value_type(self.spike.value, output.event, name="Reduction event")
        self.spike.value = output.event

    def get_point_placement(self, placement_id: int):
        """Return one static point placement by its stable id."""
        if isinstance(placement_id, bool) or not isinstance(placement_id, (int, np.integer)):
            raise TypeError("placement_id must be an integer.")
        index = int(placement_id)
        placements = self.point_placements
        if index < 0 or index >= len(placements):
            raise IndexError(f"placement_id out of range: {index!r}.")
        return placements[index]

    # ------------------------------------------------------------------
    # Phase transitions

    def init_state(self, batch_size=None) -> None:
        """Lower the declaration into runtime state and allocate V / spike.

        The Cell then runs on a read-only snapshot of the morphology it was
        declared with (:attr:`morpho`), on the grid already built from it.
        Later edits of the declaration tree apply at the next :meth:`reset`
        and ``init_state()``. If initialization fails, :attr:`morpho` is
        the declaration tree again.

        Raises
        ------
        RuntimeError
            If the cell is already initialized. Call :meth:`reset` first.

        Warns
        -----
        UserWarning
            On a lumped cell with an explicit Runge-Kutta solver, when an HH
            gate relaxes faster than ``2 / 0.025 ms`` somewhere between -100
            and 60 mV, so that the run would return NaN. The message names
            each such gate and the ``dt`` below which the solver is stable on
            it, and suggests ``"exp_euler"`` or ``"ind_exp_euler"``.
        """
        self._raise_if_network_owned("init_state()")
        self._raise_if_initialized("init_state()")

        if batch_size is not None:
            if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
                raise TypeError("Cell init_state() batch_size must be an integer or None.")
            if int(batch_size) < 1:
                raise ValueError("Cell init_state() batch_size must be >= 1.")
        # Declaration mutations (morphology, policy and paint/place rules) keep
        # the discretization cache current while the cell is uninitialized.
        # Initialization consumes that snapshot; it must never silently rerun a
        # policy or create a second grid generation.
        self._resolve_declarations()
        declaration_key = self._discretization_cache_key
        self._take_morphology_snapshot()
        try:
            self._init_runtime(batch_size)
        except BaseException:
            self._morpho = self._declaration_morpho
            self._discretization_cache_key = declaration_key
            self._snapshot_source_revision = self._snapshot_reindex_revision = None
            self._snapshot_warned_revision = None
            raise

    def _take_morphology_snapshot(self) -> None:
        """Run on a frozen copy of the declaration tree from here to :meth:`reset` (MORPH-14).

        The copy has the structure the grid was just built from, so the
        cached discretization stays valid and is re-keyed to the copy rather
        than rebuilt: no second grid generation.
        """
        declaration = self._declaration_morpho
        self._morpho = frozen_snapshot(declaration)
        self._snapshot_source_revision = declaration.revision
        self._snapshot_reindex_revision = declaration._reindex_revision
        self._snapshot_warned_revision = declaration.revision
        self._discretization_cache_key = self._discretization_key()

    def _unannounced_declaration_edit(self) -> bool:
        """Whether the declaration tree was edited after init_state, unannounced so far.

        True once per new revision of the declaration tree, which the
        running snapshot does not have; :meth:`run` and ``Network.run`` warn
        when it is.
        """
        revision = self._declaration_morpho.revision
        if self._snapshot_warned_revision in (None, revision):  # not initialized, or already said
            return False
        self._snapshot_warned_revision = revision
        return True

    def _index_basis(self) -> tuple[Morphology, int]:
        """``(tree, revision)`` whose default branch indices ``self.morpho`` has now.

        Once initialized that is the declaration tree at the snapshot, so
        indices resolved on the snapshot are still checked after
        :meth:`reset` (see :meth:`_held_indices_moved`).
        """
        if self._snapshot_source_revision is not None:
            return self._declaration_morpho, self._snapshot_source_revision
        return self._morpho, self._morpho.revision

    def _held_indices_moved(self, basis: tuple[Morphology, int] | None) -> bool:
        """Whether default branch indices held on ``basis = (tree, revision)`` moved.

        The indices are compared with the tree this Cell runs: the
        declaration tree, or once initialized the snapshot of it, whose
        indices are those of the declaration tree at the snapshot. A basis
        on any other tree is not checked.
        """
        if basis is None:
            return False
        tree, revision = basis
        if tree is self._morpho:
            return tree._reindexed_after(revision)
        if tree is self._declaration_morpho and self._snapshot_reindex_revision is not None:
            return self._snapshot_reindex_revision > revision
        return False

    def _init_runtime(self, batch_size) -> None:
        """Lower the declaration into runtime state on the cable, lumped or reduction path."""
        self._boundary_rows_cache = None
        self._runtime_batch_size = None if batch_size is None else int(batch_size)
        store = self._get_connection_store()
        self._connection_weight_declarations = tuple((call, call.weight) for call in store._calls)
        self._V_th_declaration = self._V_th
        if self._uses_reduction:
            self._raise_if_reduction_has_delta_inputs()
            self._init_reduction_state(batch_size=batch_size)
            self._runtime_generation += 1
            return
        if self._is_lumped:
            self._init_lumped_state(batch_size=batch_size)
            self._runtime_generation += 1
            return
        self._runtime = CellRuntimeState.from_cell(self)
        self._ind_update_after_solver = self._check_solver_against_mechanisms()
        self._runtime.refresh_geometry()
        self._register_synapse_ion_currents()

        self._in_size = self.varshape
        self._out_size = self.varshape

        root_nodes = dict(self._runtime.ions)
        for layout in self._runtime.layouts:
            node = self._runtime.runtime_nodes.get(layout.id)
            if node is None:
                continue
            if _is_root_level_runtime_node(layout.kind):
                root_nodes[f"layout_{layout.id}"] = node

        self.ion_channels = self._format_elements(IonChannel, **root_nodes)
        self.C = bridge.broadcast_to_shape(self._runtime.cable.cm, self.pop_size + (self.n_cv,), name="Cell.C")
        self._V_th = self._materialize_population_parameter("V_th")

        v_value = self._materialize_population_parameter("V_init")
        from braincell._parameter_schema import RuntimeParameterState

        self._V_init_materialized = RuntimeParameterState(v_value)
        self._V_th_parameter = RuntimeParameterState(self._V_th)
        self._runtime_V_init = RuntimeParameterState(v_value)
        self._runtime_V_init_mask = brainstate.LongTermState(jnp.zeros(self.pop_size + (self.n_cv,), dtype=bool))
        v_value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        # A Cell is spatial: every hidden state's trailing axis enumerates
        # CVs (V and painted density state) or sparse point-layout rows, so all
        # are group states. Channel / ion / synapse code is shared with
        # SingleCompartment, hence the scoped factory rather than a
        # per-call-site class choice.
        self.V = DiffEqGroupState(v_value)
        self.spike = brainstate.ShortTermState(_zero_spike_like(self.V.value))
        self._event_previous_V = brainstate.ShortTermState(self.V.value)
        clamp_store = self._get_clamp_store()
        self._step_clamp_components = brainstate.ShortTermState(
            u.Quantity(jnp.zeros((len(clamp_store.id),), dtype=float), u.nA)
        )
        self._step_clamp_point_current = brainstate.ShortTermState(
            u.Quantity(jnp.zeros(self._runtime.pop_size + (self._runtime.n_point,), dtype=float), u.nA)
        )
        self._point_V = brainstate.ShortTermState(self._initial_point_voltage(self.V.value))
        self._next_synapse_inputs_prepared = brainstate.ShortTermState(jnp.asarray(False))
        self._current_time_state.value = 0.0 * u.ms

        cv_V = self.V.value
        with state_grouping(True):
            for path, channel in self._runtime_objects_unchecked(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, cv_V)
                channel.init_state(*args, batch_size=batch_size)
            # Mechanism init hooks allocate state; reset hooks materialize the
            # model-defined initial values from V_init and current parameters.
            for path, channel in self._runtime_objects_unchecked(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, cv_V)
                channel.reset_state(*args, batch_size=batch_size)

        # Dense CV axial operators are only needed by derivative-based voltage
        # solvers. The default DHS/staggered path builds its own static source,
        # so defer this matrix until ``_get_axial_operator()`` is actually used.
        self._runtime.axial_operator_source = None
        self._runtime.axial_operator_cache = None
        self._initialized = True
        # Selections created during declaration are not runtime Views.  A
        # successful initialization creates the runtime generation and makes
        # those pre-init selections stale; callers must select again.
        self._view_generation += 1
        self._runtime_generation += 1
        self._runtime_cvs_cache = self._build_runtime_cv_views()
        self._runtime_nodes_cache = self._build_runtime_node_views()

    def _check_solver_against_mechanisms(self) -> bool:
        """Reject a solver that cannot advance this Cell's mechanisms.

        Returns
        -------
        bool
            Whether the solver is a generic registered step, after which the
            IndependentIntegration mechanisms advance in ``ind_update``. It is
            ``False`` for ``"staggered"`` (which schedules them itself),
            ``"dhs_voltage"`` (voltage only) and a user callable (which owns
            the whole step).

        Raises
        ------
        ValueError
            If ``exp_euler``, ``backward_euler`` or ``crank_nicolson`` meets a
            point synapse.
        """
        if not _is_generic_step(self._solver_fn):
            return False
        if any(self._solver_fn is get_integrator(name) for name in _BLOCK_SOLVERS):
            synapses = sorted({type(synapse).__name__ for _, synapse in self._runtime.iter_synapse_layouts()})
            if synapses:
                staggered = "" if self._is_lumped else "'staggered', "
                raise ValueError(
                    f"solver {self._solver_name!r} cannot advance a Cell with point synapses "
                    f"({', '.join(synapses)}): it integrates all states as one block per cell, and "
                    f"synapse states have no CV axis. Use {staggered}'ind_exp_euler' or an explicit "
                    "Runge-Kutta solver such as 'ralston2'."
                )
        return True

    def _register_synapse_ion_currents(self) -> None:
        """Add each ion-reading synapse's ion current to its ion (``_SynapseIonCurrent``)."""
        for callback in currents.register_synapse_ion_currents(self):
            ion = self._runtime.get_ion(callback.ion_key)
            if not self.cache_ion_total_current and getattr(type(ion), "uses_total_current", False):
                # Without the step-start snapshot the ion's own integrator
                # would differentiate through the synapse's states.
                raise ValueError(
                    f"{callback.requires_snapshot}, whose concentration integrates its total current; "
                    "this needs cache_ion_total_current=True."
                )

    def _init_lumped_state(self, *, batch_size=None) -> None:
        """Build the lumped runtime: SingleCompartment math with no CV axis."""
        self._in_size = self.pop_size
        self._out_size = self.pop_size
        V_th = self._materialize_population_parameter("V_th")[..., 0]
        V_init = self._materialize_population_parameter("V_init")[..., 0]
        from braincell._parameter_schema import RuntimeParameterState

        self._V_init_materialized = RuntimeParameterState(V_init)
        self._V_th = V_th
        # The runtime reads both through its owner, so CellView.set() and a
        # trainable spike threshold reach it.
        self._V_th_parameter = RuntimeParameterState(V_th)
        runtime, channels, runtime_state = build_lumped_runtime(self, V_init=V_init, V_th=V_th)
        self._lumped_runtime = runtime
        self._runtime = runtime_state
        self._check_solver_against_mechanisms()
        self._register_synapse_ion_currents()
        self._lumped_clamps = LumpedClamps(
            self,
            density_unit=_LUMPED_CURRENT_UNIT,
            with_components=any(
                isinstance(spec.observable, _ClampCurrentObservable) for spec in self._recording_specs.values()
            ),
        )
        runtime.init_state(batch_size)
        # Channel init hooks allocate gates; reset hooks put them at steady
        # state for V_init, as on a cable Cell.
        runtime.reset_state(batch_size)
        # CELL-29: an explicit solver on a stiff HH gate returns NaN at every
        # practical dt; say so here rather than leave a silent NaN to the run.
        warn_stiff_gates(runtime, channels, solver=self._solver_fn, solver_name=self._solver_name, stacklevel=4)
        self.V = runtime.V
        self.spike = runtime.spike
        self._event_previous_V = brainstate.ShortTermState(self.V.value)
        self._next_synapse_inputs_prepared = brainstate.ShortTermState(jnp.asarray(False))
        self._lumped_I_ext = brainstate.ShortTermState(self._zero_lumped_current())
        self._step_clamp_components = brainstate.ShortTermState(
            u.Quantity(jnp.zeros((self._lumped_clamps.n_component,), dtype=float), u.nA)
        )
        self._current_time_state.value = 0.0 * u.ms
        self._initialized = True
        self._view_generation += 1

    def _lumped_membrane_current(self):
        return self._lumped_runtime.membrane_current(self._lumped_I_ext.value)

    def _zero_lumped_current(self):
        return u.Quantity(jnp.zeros(self.pop_size, dtype=float), _LUMPED_CURRENT_UNIT)

    def _init_reduction_state(self, *, batch_size=None) -> None:
        """Allocate only packed synapse inputs plus the selected reduced model."""
        self._in_size = self.pop_size
        self._out_size = self.pop_size
        self._reduction_input_runtime = build_reduction_input_runtime(self)
        model = self._reduction_models[self._selected_model_name]
        output = model.init_state(self._reduction_input_runtime.context, batch_size=batch_size)
        self._publish_reduction_output(output, initialize=True)
        self._pending_reduction_inputs = self._reduction_input_runtime.take_inputs()
        self._next_synapse_inputs_prepared = brainstate.ShortTermState(jnp.asarray(False))
        self._current_time_state.value = 0.0 * u.ms
        self._initialized = True

    def reset(self) -> None:
        """Drop runtime and per-step state; return to DECLARING.

        Raises
        ------
        RuntimeError
            If the cell is not initialized.

        Notes
        -----
        ``reset()`` is distinct from :meth:`reset_state`. ``reset_state``
        reseeds ``V`` / ``spike`` / ``current_time`` in place and stays
        in the INITIALIZED phase. ``reset()`` fully tears down the
        runtime and returns to DECLARING so ``paint`` / ``place`` can
        run again. :attr:`morpho` is the declaration tree again, with any
        edit made to it since :meth:`init_state`; the next grid read
        rebuilds the grid from it.
        """
        self._raise_if_network_owned("reset()")
        self._raise_if_not_initialized("reset()")

        self.connections.clear_runtime()
        for call, weight in self._connection_weight_declarations:
            call.weight = weight
        self._connection_weight_declarations = ()

        if self._uses_reduction:
            self._reduction_models[self._selected_model_name].reset()

        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            if hasattr(self, name):
                delattr(self, name)

        # Restore scalar V_th (init_state overwrote it with a vector).
        self._V_th = self._V_th_declaration

        if hasattr(self, "V"):
            delattr(self, "V")
        if hasattr(self, "spike"):
            delattr(self, "spike")
        if hasattr(self, "_event_previous_V"):
            delattr(self, "_event_previous_V")
        if self._lumped_runtime is not None:
            del self._lumped_runtime
            self._lumped_runtime = None
            self._lumped_clamps = None
        for name in (
            "_step_clamp_components",
            "_step_clamp_point_current",
            "_point_V",
            "_lumped_I_ext",
            "_next_synapse_inputs_prepared",
        ):
            if hasattr(self, name):
                delattr(self, name)
        self._current_time_state.value = 0.0 * u.ms

        self._runtime = None
        self._reduction_input_runtime = None
        self._reduction_output_states = {}
        self._pending_reduction_inputs = None
        self._runtime_batch_size = None
        self.trainables.clear()
        self._runtime_V_init = None
        self._runtime_V_init_mask = None
        self._V_th_parameter = None
        self._view_generation += 1
        self._runtime_generation += 1
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        self._node_scheduling_cache.clear()
        self._boundary_rows_cache = None
        self._run_loop_cache.clear()
        self._compiled_recording_cache.clear()

        self._morpho = self._declaration_morpho
        self._snapshot_source_revision = self._snapshot_reindex_revision = None
        self._snapshot_warned_revision = None
        self._invalidate_discretization_cache()
        self._V_init_materialized = None

        self._initialized = False

    # ------------------------------------------------------------------
    # Static topology + runtime inspection views

    @property
    def runtime(self) -> CellRuntimeState:
        self._raise_if_not_initialized("runtime")
        if self._uses_reduction:
            raise RuntimeError("Detailed Cell runtime is unavailable while a reduction model is selected.")
        return self._runtime

    @property
    def n_point(self) -> int:
        self._raise_if_not_initialized("n_point")
        if self._is_lumped:
            raise RuntimeError("A lumped Cell (cv_policy=Lumped()) has no electrical point tree.")
        if self._uses_reduction:
            raise RuntimeError("Detailed point runtime is unavailable while a reduction model is selected.")
        return self._runtime.n_point

    @property
    def pop_size(self) -> tuple[int, ...]:
        return self._pop_size

    @property
    def _population_size(self) -> int:
        if len(self.pop_size) == 0:
            return 1
        if len(self.pop_size) != 1:
            raise ValueError(f"Cell population views require one-dimensional pop_size; got {self.pop_size!r}.")
        return int(self.pop_size[0])

    @property
    def _view_root(self) -> "Cell":
        return self

    @property
    def varshape(self) -> tuple[int, ...]:
        return self.pop_size + (self.n_cv,)

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    def runtime_objects(self, *args, **kwargs):
        """Return runtime graph objects from the inherited container API."""
        self._raise_if_not_initialized("runtime_objects()")
        return self._runtime_objects_unchecked(*args, **kwargs)

    def _runtime_objects_unchecked(self, *args, **kwargs):
        """Return runtime graph objects without an initialization guard."""
        return super().nodes(*args, **kwargs)

    @property
    def runtime_cvs(self) -> tuple[RuntimeCVView, ...]:
        self._raise_if_not_initialized("runtime_cvs")
        if self._runtime_cvs_cache is None:
            self._runtime_cvs_cache = self._build_runtime_cv_views()
        return self._runtime_cvs_cache

    @property
    def runtime_nodes(self) -> tuple[RuntimeNodeView, ...]:
        self._raise_if_not_initialized("runtime_nodes")
        if self._runtime_nodes_cache is None:
            self._runtime_nodes_cache = self._build_runtime_node_views()
        return self._runtime_nodes_cache

    def _build_runtime_cv_views(self) -> tuple[RuntimeCVView, ...]:
        runtime = self.runtime
        node_tree = self.node_tree
        return tuple(
            RuntimeCVView(
                _cell=self,
                id=int(cv.id),
                declaration=cv,
                layout_ids=tuple(int(layout.id) for layout in runtime.get_cv_layouts(int(cv.id))),
                mid_node_id=int(node_tree.cv_to_mid_node_id[int(cv.id)]),
                ions=self._build_local_ion_bindings(cv_ids=(int(cv.id),)),
            )
            for cv in self.cvs
        )

    def _build_runtime_node_views(self) -> tuple[RuntimeNodeView, ...]:
        runtime = self.runtime
        return tuple(
            RuntimeNodeView(
                _cell=self,
                id=int(node.id),
                declaration=node,
                layout_ids=tuple(int(layout.id) for layout in runtime.get_point_layouts(int(node.id))),
                source_cv_ids=node.source_cv_ids,
                ions=self._build_local_ion_bindings(point_ids=(int(node.id),)),
            )
            for node in self.node_tree.nodes
        )

    def _build_local_ion_bindings(
        self,
        *,
        cv_ids: tuple[int, ...] = (),
        point_ids: tuple[int, ...] = (),
    ) -> Mapping[str, RuntimeIonBinding]:
        runtime = self.runtime
        return {
            name: RuntimeIonBinding(
                name=name,
                runtime=ion,
                cell=self,
                cv_ids=cv_ids,
                point_ids=point_ids,
            )
            for name, ion in runtime.ions.items()
        }

    def node_scheduling(self, *, max_group_size: int = 256, algorithm: str = "dhs"):
        self._raise_if_not_initialized("node_scheduling()")
        return self._node_scheduling_unchecked(max_group_size=max_group_size, algorithm=algorithm)

    def _node_scheduling_unchecked(self, *, max_group_size: int = 256, algorithm: str = "dhs"):
        key = (algorithm, int(max_group_size))
        cached = self._node_scheduling_cache.get(key)
        if cached is not None:
            return cached
        scheduling = build_node_scheduling(
            self.node_tree,
            max_group_size=max_group_size,
            algorithm=algorithm,
        )
        self._node_scheduling_cache[key] = scheduling
        return scheduling

    # ------------------------------------------------------------------
    # Time

    @property
    def current_time(self):
        self._raise_if_not_initialized("current_time")
        return self._current_time_state.value

    def _set_current_time(self, value) -> None:
        self._current_time_state.value = value

    # ------------------------------------------------------------------
    # Repr

    def __repr__(self) -> str:
        if self._initialized and self._is_lumped:
            return f"Cell(root={self._morpho.root.name!r}, lumped=True, pop_size={self.pop_size!r}, initialized=True)"
        if self._initialized:
            return (
                f"Cell(root={self._morpho.root.name!r}, n_cv={self.n_cv!r}, n_point={self.n_point!r}, initialized=True)"
            )
        return (
            f"Cell(root={self._morpho.root.name!r}, "
            f"n_branches={len(self._morpho.branches)}, "
            f"n_paint_rules={len(self._paint_rules)}, "
            f"n_place_rules={len(self._place_rules)}, "
            f"initialized=False)"
        )

    # ------------------------------------------------------------------
    # Bridging (runtime-only)

    def _cv_to_point(self, cv_values):
        self._raise_if_not_initialized("_cv_to_point()")
        if self._is_lumped:
            return u.math.broadcast_to(cv_values[..., None], u.math.shape(cv_values) + (self._runtime.n_point,))
        return bridge.cv_to_point(cv_values, self._runtime)

    def _initial_point_voltage(self, cv_values):
        """Expand CV voltage to every point for the initial DHS linearization."""

        return cv_values[..., self._runtime.point_to_representative_cv_np]

    def _dhs_point_voltage(self, cv_values=None):
        """Return cached DHS point voltage with current midpoint values."""

        values = self.V.value if cv_values is None else cv_values
        point_values = self._point_V.value
        midpoint_ids = self._runtime.node_tree.cv_to_mid_node_id
        return point_values.at[..., midpoint_ids].set(values)

    def _set_dhs_point_voltage(self, point_values):
        """Publish one complete point-tree voltage solution."""

        self._point_V.value = point_values

    def _point_voltage_for_mechanisms(self, cv_values=None):
        """Return the electrical point voltage point mechanisms, events and records read.

        Midpoints carry ``cv_values``. A boundary (algebraic) point carries the
        voltage last solved for it: by DHS at every point for the
        ``"staggered"`` and ``"dhs_voltage"`` solvers, and by the boundary-row
        solve of the other solvers (in ``pre_integral`` and ``post_integral``)
        at every boundary with a clamp or synapse. A boundary without one
        carries its representative CV's voltage under the other solvers.
        """

        values = self.V.value if cv_values is None else cv_values
        if self._is_lumped:
            return u.math.broadcast_to(values[..., None], u.math.shape(values) + (self._runtime.n_point,))
        if self._solver_name in {"staggered", "dhs_voltage"}:
            return self._dhs_point_voltage(values)
        return boundary.point_voltage(self._boundary_rows(), values, self._point_V.value)

    def _boundary_rows(self) -> boundary.BoundaryRows:
        """Return the algebraic points that carry a clamp or synapse (cached)."""
        if self._boundary_rows_cache is None:
            self._boundary_rows_cache = boundary.build_boundary_rows(self._runtime, self.cvs)
        return self._boundary_rows_cache

    def _solve_boundary_rows(self, V):
        """Solve the boundary rows at ``V``; return the derivative term and the voltages.

        Returns ``None`` when no boundary carries an input. The cached point
        voltage is the Newton start value.
        """
        rows = self._boundary_rows()
        if not rows.has_inputs:
            return None
        guess = self._point_V.value.to_decimal(u.mV)[..., rows.input_ids]
        return boundary.solve_boundary(self, rows, V, t=self._resolve_t(), guess=guess)

    def _update_boundary_voltage(self) -> None:
        """Store the boundary voltages solved at the current ``V`` in ``_point_V``."""
        solved = self._solve_boundary_rows(self.V.value)
        if solved is None:
            return
        cached = self._point_V.value.to_decimal(u.mV)
        self._point_V.value = u.Quantity(cached.at[..., self._boundary_rows().input_ids].set(solved[1]), u.mV)

    def _cv_to_point_unchecked(self, cv_values):
        return bridge.cv_to_point(cv_values, self._runtime)

    def _point_to_cv(self, point_values):
        self._raise_if_not_initialized("_point_to_cv()")
        return bridge.point_to_cv(point_values, self._runtime)

    # ------------------------------------------------------------------
    # Solver path (runtime-only)

    def _resolve_t(self):
        try:
            return brainstate.environ.get("t")
        except KeyError:
            return self.current_time

    def _run_integrated_hook(self, hook: str, cv_V, stage_point_V=None) -> None:
        """Call one integration hook on every mechanism a generic solver advances.

        A node marked :class:`~braincell.quad.IndependentIntegration` advances
        itself in ``ind_update``. The dependent channels under such an ion (a
        kinetic ion's HH calcium channel) are still the solver's, as in the
        staggered schedule. ``stage_point_V`` is the point voltage of the stage
        being evaluated (see :meth:`_runtime_node_phase_args`).
        """
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                args = self._runtime_node_phase_args(path, node, cv_V, stage_point_V=stage_point_V)
                getattr(node, hook)(*args)
            elif isinstance(node, Ion):
                node._run_on_dependent_children(node._channels(), hook, cv_V)

    def pre_integral(self):
        """Start one solver step: sample the clamps, then run the mechanism hooks.

        ``update()``, ``run()`` and ``Network.run`` sample every current
        clamp once per main step, at ``t + dt / 2``, before calling the
        solver. A solver step called directly (``braincell.quad.rk4_step(cell)``)
        reaches this hook first, at the step's start time, so the clamps
        are sampled here instead, and every stage of the step reads that
        value.
        """
        self._raise_if_not_initialized("pre_integral()")
        if not brainstate.environ.get("_braincell_step_clamps_prepared", False):
            self._prepare_step_clamps()
            self._direct_step_clamps_prepared = True
        if self._is_lumped:
            return self._lumped_runtime.pre_integral(self._lumped_I_ext.value)
        # Events applied at step start, or a write to V between steps, move
        # the boundary voltages; re-solve them for the hooks and as the
        # Newton start of every stage.
        self._update_boundary_voltage()
        self._run_integrated_hook("pre_integral", self.V.value)

    def compute_derivative(self):
        self._raise_if_not_initialized("compute_derivative()")
        if self._is_lumped:
            return self._lumped_runtime.compute_derivative(self._lumped_I_ext.value)
        # One boundary solve per stage: its voltages enter both the voltage
        # derivative and the point mechanisms' arguments (CELL-2b).
        derivative, stage_point_V = self._voltage_derivative_and_point_voltage(self.V.value)
        self.V.derivative = derivative
        self._run_integrated_hook("compute_derivative", self.V.value, stage_point_V=stage_point_V)

    def compute_membrane_derivative(self, V):
        self._raise_if_not_initialized("compute_membrane_derivative()")
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, t=t)
        return I_total / self._current_capacitance()

    def _current_capacitance(self):
        """Return capacitance from current geometry states."""
        cable = self._runtime.current_cable()
        return bridge.broadcast_to_shape(cable.cm, self.pop_size + (self.n_cv,), name="Cell.C")

    def _voltage_linearizer(self):
        """Return the configured voltage-only membrane linearizer."""
        membrane_derivative = jax.named_call(
            self.compute_membrane_derivative,
            name="braincell_dhs_compute_membrane_derivative",
        )
        return brainstate.transform.vector_grad(
            membrane_derivative,
            argnums=0,
            return_value=True,
            unit_aware=False,
        )

    def _get_axial_operator(self):
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("_get_axial_operator() requires init_state() first.")
        if runtime.geometry is not None and any(is_traced_value(state.value) for state in runtime.geometry.values()):
            return build_cv_axial_operator(
                self,
                node_tree=self.node_tree,
                scheduling=self._node_scheduling_unchecked(algorithm="dhs"),
            ) * (u.ms**-1)
        float_dtype = jnp.asarray(0.0).dtype
        cache = runtime.axial_operator_cache
        if cache is not None and cache.float_dtype == float_dtype:
            return cache.operator

        source = runtime.axial_operator_source
        if source is None:
            source = build_cv_axial_operator(
                self,
                node_tree=self.node_tree,
                scheduling=self._node_scheduling_unchecked(algorithm="dhs"),
            )
            if not is_traced_value(source):
                runtime.axial_operator_source = source

        operator = jnp.asarray(source, dtype=brainstate.environ.dftype()) * (u.ms**-1)
        cache = AxialOperatorCache(float_dtype=float_dtype, operator=operator)
        if not is_traced_value(operator):
            runtime.axial_operator_cache = cache
        return operator

    def _refresh_geometry_runtime(self):
        """Refresh cable-derived runtime values without rebuilding the grid."""
        self._raise_if_not_initialized("refresh geometry")
        self._runtime.refresh_geometry()
        if self._is_lumped:
            # The lumped runtime reads area and cm from the geometry states.
            self._run_loop_cache.clear()
            return
        self.C = bridge.broadcast_to_shape(self._runtime.cable.cm, self.pop_size + (self.n_cv,), name="Cell.C")
        self._run_loop_cache.clear()

    def compute_axial_derivative(self, V):
        self._raise_if_not_initialized("compute_axial_derivative()")
        V_mv = u.Quantity(u.math.asarray(V.to_decimal(u.mV)), u.mV)
        axial_operator = self._get_axial_operator()
        return -u.math.matmul(axial_operator, V_mv[..., None])[..., 0]

    def compute_voltage_derivative(self, V):
        return self._voltage_derivative_and_point_voltage(V)[0]

    def _voltage_derivative_and_point_voltage(self, V):
        """Return the voltage derivative at ``V`` and the point voltage it solved.

        The point voltage is ``None`` when no boundary carries an input;
        otherwise midpoints carry ``V`` and every boundary with an input the
        voltage its row solves to at ``V``.
        """
        derivative = self.compute_membrane_derivative(V) + self.compute_axial_derivative(V)
        # The axial operator eliminates the boundary rows as if they had no
        # input; add what their clamps and synapses change.
        solved = self._solve_boundary_rows(V)
        if solved is None:
            return derivative, None
        correction, V_b = solved
        return derivative + correction, boundary.stage_point_voltage(self._boundary_rows(), V, V_b)

    def _top_level_ion_channel_nodes(self):
        return tuple(self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items())

    def _family_ion_nodes(self):
        return tuple((path, node) for path, node in self._top_level_ion_channel_nodes() if isinstance(node, Ion))

    def _family_channel_nodes(self):
        nodes = []
        for path, node in self._top_level_ion_channel_nodes():
            # ``Ion`` and ``MixIons`` are siblings, not a hierarchy
            # (both derive from ``IonChannel, Container``), and a channel
            # under either is reached the same way.
            if isinstance(node, (Ion, MixIons)):
                nodes.extend(self._child_channel_nodes(path, node))
            elif isinstance(node, Channel):
                nodes.append((path, node))
        return tuple(nodes)

    @staticmethod
    def _child_channel_nodes(path, node):
        """``(full path, channel)`` for the channels under an Ion or MixIons that own state."""
        return tuple(
            (path + child_path, child)
            for child_path, child in brainstate.graph.nodes(node, Channel, allowed_hierarchy=(1, 1)).items()
            if not getattr(child, "_skip_family_update", False)
        )

    def _integrate_dependent_child_channels(self, path, node, cv_V):
        """Advance the non-independent channels under ``node`` with ``ind_exp_euler``, reading its ion info."""
        for child_path, child in self._child_channel_nodes(path, node):
            if self._is_independent_channel(child):
                continue
            target, args = self._channel_integration_target_and_args(child_path, child, cv_V)
            with jax.named_scope(_scope_name("braincell:ion_update:channel", child_path, child)):
                jax.named_call(
                    ind_exp_euler_step,
                    name=_call_name("braincell:ion_update:channel_step", child_path, child),
                )(target, *args)

    def _integrate_selected_ion_self_states(
        self,
        ion_nodes,
        selected_paths,
        cv_V,
        excluded_paths,
    ):
        selected_paths = tuple(tuple(path) for path in selected_paths)
        if not selected_paths:
            return

        selected_path_set = set(selected_paths)

        def _run_phase(hook_name):
            for path, ion in ion_nodes:
                if path in selected_path_set:
                    getattr(ion, hook_name)(cv_V, recursive_child=False)

        _ind_exp_euler_step_selected(
            self,
            include_paths=selected_paths,
            excluded_paths=excluded_paths,
            pre_integral=lambda: _run_phase("pre_integral"),
            compute_derivative=lambda: _run_phase("compute_derivative"),
            post_integral=lambda: _run_phase("post_integral"),
            allow_empty=True,
        )

    def _update_ion_channels_by_integration(self, cv_V):
        with jax.named_scope("braincell:ion_update:integration:dependent"):
            for path, node in self._top_level_ion_channel_nodes():
                if isinstance(node, IndependentIntegration):
                    # A self-integrating ion (KineticIon) advances its own
                    # species only; its dependent channels advance here, as
                    # in the "family" order's channel phase (CELL-32).
                    if isinstance(node, (Ion, MixIons)):
                        self._integrate_dependent_child_channels(path, node, cv_V)
                    continue
                args = self._runtime_node_phase_args(path, node, cv_V)
                with jax.named_scope(_scope_name("braincell:ion_update:node", path, node)):
                    jax.named_call(
                        ind_exp_euler_step,
                        name=_call_name("braincell:ion_update:node_step", path, node),
                    )(node, *args)

        self._ind_update_top_level_nodes(cv_V)

    def _ind_update_top_level_nodes(self, cv_V):
        """Call ``ind_update`` on every top-level node (the ``"integration"`` order)."""
        with jax.named_scope("braincell:ion_update:integration:independent"):
            for path, node in self._top_level_ion_channel_nodes():
                args = self._runtime_node_phase_args(path, node, cv_V)
                with jax.named_scope(_scope_name("braincell:ion_update:node", path, node)):
                    jax.named_call(
                        node.ind_update,
                        name=_call_name("braincell:ion_update:node_ind_update", path, node),
                    )(*args)

    def _update_ion_channel_families(self, cv_V):
        ion_nodes = self._family_ion_nodes()
        channel_nodes = self._family_channel_nodes()

        dependent_ion_paths = [path for path, node in ion_nodes if not isinstance(node, IndependentIntegration)]
        channel_paths = [path for path, _ in channel_nodes]

        # Family mode splits ion self states from channel states. This phase
        # advances dependent Ion states only; V and all channel states are
        # excluded explicitly so no child channel is integrated through Ion
        # recursion.
        with jax.named_scope("braincell:ion_update:family:dependent_ion_self"):
            self._integrate_selected_ion_self_states(
                ion_nodes,
                dependent_ion_paths,
                cv_V,
                excluded_paths=[("V",), *channel_paths],
            )

        # Independent Ion states use their own updater, still without
        # recursing into child channels.
        self._update_independent_ions(ion_nodes, cv_V)

        # Channel nodes include Ion child channels, MixIons child channels,
        # and top-level channels. The owner path rebuilds the right ion args.
        with jax.named_scope("braincell:ion_update:family:dependent_channel"):
            for path, node in channel_nodes:
                if not self._is_independent_channel(node):
                    target, args = self._channel_integration_target_and_args(
                        path,
                        node,
                        cv_V,
                    )
                    with jax.named_scope(_scope_name("braincell:ion_update:channel", path, node)):
                        jax.named_call(
                            ind_exp_euler_step,
                            name=_call_name("braincell:ion_update:channel_step", path, node),
                        )(target, *args)

        # Independent channels finish through their own update rule.
        self._update_independent_channels(channel_nodes, cv_V)

    def _update_independent_ions(self, ion_nodes, cv_V):
        """Advance each IndependentIntegration ion, without its child channels."""
        with jax.named_scope("braincell:ion_update:family:independent_ion"):
            for path, node in ion_nodes:
                if isinstance(node, IndependentIntegration):
                    with jax.named_scope(_scope_name("braincell:ion_update:ion", path, node)):
                        jax.named_call(
                            node.ind_update,
                            name=_call_name("braincell:ion_update:ion_ind_update", path, node),
                        )(cv_V, recursive_child=False)

    def _update_independent_channels(self, channel_nodes, cv_V):
        """Advance each IndependentIntegration channel through its own solver."""
        with jax.named_scope("braincell:ion_update:family:independent_channel"):
            for path, node in channel_nodes:
                if not self._is_independent_channel(node):
                    continue
                target, args = self._channel_integration_target_and_args(
                    path,
                    node,
                    cv_V,
                )
                with jax.named_scope(_scope_name("braincell:ion_update:channel", path, node)):
                    jax.named_call(
                        target.ind_update,
                        name=_call_name("braincell:ion_update:channel_ind_update", path, node),
                    )(*args)

    def _update_independent_mechanisms(self, cv_V):
        """Advance the IndependentIntegration mechanisms after a generic solver step.

        A generic solver (an explicit Runge-Kutta method, ``exp_euler``,
        ``backward_euler`` ...) integrates ``V`` and the dependent mechanism
        states together, and only reads the states of Markov channels and
        kinetic ions. These then advance at the new voltage, in the order
        the staggered schedule uses for them under
        ``ion_channel_update_order``. An ion driven by its total current
        reads it from a snapshot taken at the new voltage, as staggered
        takes one before its mechanism phase; a synapse current carried by
        the ion needs that snapshot.
        """
        self.cache_ion_total_currents(cv_V)
        if self.ion_channel_update_order == "family":
            self._update_independent_ions(self._family_ion_nodes(), cv_V)
            self._update_independent_channels(self._family_channel_nodes(), cv_V)
        else:
            self._ind_update_top_level_nodes(cv_V)

    @staticmethod
    def _is_independent_channel(node):
        channel = getattr(node, "_channel", node)
        return isinstance(channel, IndependentIntegration)

    def _channel_integration_target_and_args(self, path, node, cv_V):
        if hasattr(node, "_channel") and hasattr(node, "_infos"):
            return node._channel, (cv_V, *node._infos())
        return node, self._channel_update_args(path, node, cv_V)

    def _channel_update_args(self, path, node, cv_V):
        if len(path) >= 4 and path[-2] == "channels":
            owner = self._node_at_path(path[:-2])
            if isinstance(owner, Ion):
                return cv_V, owner.pack_info()
            if isinstance(owner, MixIons):
                infos = tuple([owner._get_ion(root).pack_info() for root in node.root_type.__args__])
                return (cv_V, *infos)
        return (cv_V,)

    def _runtime_node_phase_args(self, path, node, cv_V, stage_point_V=None):
        """Return the arguments of a runtime node's integration hooks.

        A synapse reads the voltage of its points: ``stage_point_V`` when the
        caller evaluates a derivative stage that solved the boundary rows
        (each stage passes its own; nothing is stored between calls),
        otherwise the voltage :meth:`_point_voltage_for_mechanisms` gives for
        ``cv_V``.
        """
        if isinstance(node, RuntimeSynapse):
            layout_id = layout_id_from_key(path)
            layout = self._runtime.layouts[layout_id]
            if layout.point_index is None:
                raise ValueError(f"Synapse layout {layout.id!r} is missing point_index.")
            point_V = self._point_voltage_for_mechanisms(cv_V) if stage_point_V is None else stage_point_V
            return (layout.gather_points(point_V),)
        return self._channel_update_args(path, node, cv_V)

    @staticmethod
    def _node_at_path_from(root, path):
        node = root
        for part in path:
            if isinstance(node, dict):
                node = node[part]
            else:
                node = getattr(node, part)
        return node

    def _node_at_path(self, path):
        return self._node_at_path_from(self, path)

    def cache_ion_total_currents(self, V=None) -> None:
        """Cache ion source currents before voltage advances in staggered mode."""
        self._raise_if_not_initialized("cache_ion_total_currents()")
        if not self.cache_ion_total_current:
            return
        cv_V = self.V.value if V is None else V
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not getattr(type(node), "uses_total_current", False):
                continue
            # The same call the ion's derivative hook makes when no snapshot
            # is set; never retried without ``include_external``, which would
            # silently drop the external currents.
            with jax.named_scope(_scope_name("braincell:ion_current_cache:node", path, node)):
                node._cached_total_current = jax.named_call(
                    node.current,
                    name=_call_name("braincell:ion_current_cache:node_current", path, node),
                )(cv_V, include_external=True)

    def clear_ion_total_current_cache(self) -> None:
        """Remove per-step ion source-current caches."""
        self._raise_if_not_initialized("clear_ion_total_current_cache()")
        for _, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if hasattr(node, "_cached_total_current"):
                delattr(node, "_cached_total_current")

    def _apply_delta_inputs(self, *, point_tree: bool = False) -> None:
        """Add the delta inputs to ``V``, the last part of every solver step.

        ``add_delta_input`` (brainpy ``DeltaProj``) registers a voltage
        increment. As in ``SingleCompartment``, a step adds it after the
        voltage and the mechanism states have advanced: it takes no part in
        the step it ends, and the spike test of that step sees it. The
        generic solvers call this from :meth:`post_integral`; the
        ``"staggered"`` and ``"dhs_voltage"`` steps call it last with
        ``point_tree=True``, which shifts their stored point-tree voltage by
        the same increment, each point by its representative CV's.
        """
        if not self.delta_inputs:
            return
        before = self.V.value
        self.V.value = with_delta_inputs(self, before)
        if point_tree:
            increment = self.V.value - before
            self._point_V.value = self._point_V.value + increment[..., self._runtime.point_to_representative_cv_np]

    def post_integral(self):
        """Finish one solver step: add delta inputs, then run the mechanism hooks."""
        self._raise_if_not_initialized("post_integral()")
        if self._is_lumped:
            self._direct_step_clamps_prepared = False
            return self._lumped_runtime.post_integral(self._lumped_I_ext.value)
        self._apply_delta_inputs()
        self._update_boundary_voltage()
        self._run_integrated_hook("post_integral", self.V.value)
        self._direct_step_clamps_prepared = False

    def update(self, I_ext=None):
        """Advance the cell by one simulation step.

        This method is the standalone cell-level step wrapper. It applies
        already-prepared synaptic events, advances continuous membrane and
        mechanism dynamics by one ``dt``, computes the spike output, and
        prepares synaptic input for the next standalone step. It reads
        ``dt`` from ``brainstate.environ`` and ``t`` from the environment
        or, if unset, ``current_time``; it does not advance
        ``current_time``.

        Parameters
        ----------
        I_ext : Quantity, optional
            External current for this step; lumped cells
            (``cv_policy=Lumped()``, :meth:`point`) only. Either a current
            (e.g. ``0.1 * u.nA``), divided by the membrane area, or a
            current density (e.g. ``10 * u.uA / u.cm**2``), broadcast to
            ``V.value.shape``, i.e. ``(batch,) + pop_size``: a scalar drives
            every member, a ``pop_size`` array one value per member. It is
            added to the clamp current of the step, and every stage of the
            solver step reads the same value, so evaluate a time-varying
            input at the step midpoint ``t + dt / 2`` to match the clamps.
            ``None`` (default) adds nothing. Current inputs registered with
            :meth:`add_current_input` (brainpy projections) follow the same
            unit rule but are evaluated at every stage with the stage
            voltage; a registered value acts on this step only, in every
            stage.

        Returns
        -------
        object
            Spike value produced by the transition from the previous membrane
            voltage to the updated membrane voltage, shaped like ``V.value``.

        Raises
        ------
        RuntimeError
            Before ``init_state()``.
        ValueError
            If ``brainstate.environ['dt']`` is not set, or ``I_ext`` does not
            broadcast to ``V.value.shape``.
        TypeError
            If ``I_ext`` is given to a cable Cell: inject current there
            with a placed :class:`~braincell.mech.CurrentClamp` or
            :class:`~braincell.mech.FunctionClamp`.
        UnitMismatchError
            If ``I_ext`` is neither a current nor a current density; also if
            a registered current input returns another unit.

        Notes
        -----
        The standalone update order is:

        1. Apply prepared runtime synapse events. The first step after
           ``init_state()`` or ``reset_state()`` prepares them first, as
           :meth:`run` does, so an event at the start time is delivered.
        2. Advance voltage and mechanism dynamics through ``self.solver``.
        3. Detect and store the current spike.
        4. Prepare discrete event payloads for the next step.

        In network execution, delayed event delivery is scheduled outside the
        cell. ``Network.run(...)`` writes arrivals into runtime state buffers
        before calling the corresponding internal cell phases.

        Calling the cell, ``cell(I_ext)``, runs its ``before_updates``
        (e.g. the synapse of a ``brainpy.state.AlignPostProj``) and then
        ``update(I_ext)``.

        Examples
        --------
        Drive a lumped cell from a user loop with a total current:

        .. code-block:: python

            >>> import brainstate
            >>> import brainunit as u
            >>> import numpy as np
            >>> import braincell as bc
            >>> cell = bc.Cell.point(3, length=20 * u.um, radius=10 * u.um, V_init=-65 * u.mV)
            >>> cell = cell.paint(bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV))
            >>> cell.init_state()
            >>> amplitudes = np.array([0.0, 0.05, 0.1]) * u.nA
            >>> def step(t):
            ...     with brainstate.environ.context(t=t):
            ...         cell.update(amplitudes)
            ...         return cell.V.value
            >>> with brainstate.environ.context(dt=0.025 * u.ms):
            ...     times = u.math.arange(0.0 * u.ms, 5.0 * u.ms, 0.025 * u.ms)
            ...     trace = brainstate.transform.for_loop(step, times)
            >>> trace.shape
            (200, 3)
            >>> bool(trace[-1, 2] > trace[-1, 1] > trace[-1, 0])
            True

        Drive it from a brainpy projection; the conductance may be total
        (nS) or per area (mS/cm^2):

        .. code-block:: python

            >>> import brainpy
            >>> cell = bc.Cell.point(3, length=20 * u.um, radius=10 * u.um, V_init=-65 * u.mV)
            >>> cell = cell.paint(bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV))
            >>> proj = brainpy.state.AlignPostProj(
            ...     comm=lambda spikes: spikes * (5.0 * u.nS),
            ...     syn=brainpy.state.Expon.desc(3, tau=5.0 * u.ms),
            ...     out=brainpy.state.COBA.desc(E=0.0 * u.mV),
            ...     post=cell,
            ... )
            >>> _ = brainstate.nn.init_all_states(proj)  # also initializes the cell
            >>> with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            ...     proj(np.array([1.0, 0.0, 0.0]))
            ...     spike = cell()
            >>> spike.shape
            (3,)
            >>> bool(cell.V.value[0] > cell.V.value[1])
            True
        """
        if I_ext is not None and not self._is_lumped:
            raise TypeError(
                "Cell.update(I_ext) drives a lumped Cell (cv_policy=Lumped(), Cell.point) only. "
                "On a cable Cell, inject current with a placed clamp at a site: "
                "cell.place(locset, braincell.mech.CurrentClamp(...)) or "
                "braincell.mech.FunctionClamp(fn=...); after init_state(), call reset() before placing it."
            )
        self._raise_if_not_initialized("update()")
        # Checked before any state changes, so a bad input leaves the cell as it was.
        external = None
        if I_ext is not None:
            external = as_current_density(
                I_ext, area=self._lumped_runtime.area, shape=tuple(self.V.value.shape), what="I_ext"
            )
        self._prepare_first_synapse_inputs(t=self._resolve_t())
        self._prepare_step_clamps()
        self._begin_step()
        with brainstate.environ.context(_braincell_step_clamps_prepared=True):
            spk = self._update_dynamics(external)
        self._prepare_next_synapse_inputs(t=self._resolve_t() + brainstate.environ.get_dt())
        return spk

    def add_current_input(self, key, inp, label=None):
        """Register a current input, the entry brainpy projections use.

        ``brainpy.state.AlignPostProj`` and ``CurrentProj`` call this on
        their ``post`` model with their synaptic output (e.g. ``COBA``), a
        callable of the membrane voltage. Registering a callable also drops
        the compiled loops of :meth:`run` (and of the owning
        :class:`~braincell.Network`), so the next run includes it.

        Parameters
        ----------
        key : str
            Unique name of the input.
        inp : callable or Quantity
            A callable of the voltage, evaluated at every solver stage of
            every step, or a value, which drives the next step only: every
            stage of that step adds it, and the step removes it.
        label : str, optional
            brainpy's grouping label.

        Raises
        ------
        ValueError
            If ``key`` is already registered with a different input.

        Notes
        -----
        A value is consumed by the next :meth:`update`, or by the first step
        of the next :meth:`run` (or ``Network.run``, ``Network.update``),
        which passes it into its compiled loop. A loop of your own that
        traces ``update()`` once (``brainstate.transform.for_loop``) reads a
        value registered before it as a constant and adds it on every
        iteration; register the value inside the loop body, as brainpy
        projections do, or use :meth:`run`.

        On a lumped cell the input may return a current (e.g. nA), divided
        by the membrane area, or a current density (e.g. uA/cm^2), as
        :meth:`update`'s ``I_ext``; it is called with the voltage shaped
        ``(batch,) + pop_size``. On a cable cell it is called with the point
        voltage, shaped ``(batch,) + pop_size + (n_point,)``, and must return
        a current density.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import brainunit as u
            >>> import braincell as bc
            >>> cell = bc.Cell.point(2, V_init=-65 * u.mV)
            >>> cell = cell.paint(bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65 * u.mV))
            >>> cell.add_current_input("drive", lambda V: 0.02 * u.nA)
            >>> cell.init_state()
            >>> with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            ...     _ = cell.update()
            >>> bool((cell.V.value > -65 * u.mV).all())
            True
        """
        super().add_current_input(key, inp, label=label)
        if callable(inp):
            self._drop_compiled_loops()

    def add_delta_input(self, key, inp, label=None):
        """Register a voltage increment, the entry brainpy's ``DeltaProj`` uses.

        Every solver adds the registered inputs to ``V`` at the end of each
        step, after the voltage and the mechanism states have advanced, as
        ``SingleCompartment`` does; the spike test of the step sees them.
        Registering a callable also drops the compiled loops of :meth:`run`
        (and of the owning :class:`~braincell.Network`), so the next run
        includes it; a value is passed into the loop, as for
        :meth:`add_current_input`.

        Parameters
        ----------
        key : str
            Unique name of the input.
        inp : callable or Quantity
            A voltage (e.g. ``1.0 * u.mV``), added at the end of the next
            step only (the first step of the next :meth:`run`), or a
            callable without arguments returning one, called at every step.
        label : str, optional
            brainpy's grouping label.

        Raises
        ------
        ValueError
            If ``key`` is already registered with a different input.

        Notes
        -----
        The step that adds the inputs raises ``ValueError``:

        - if they do not broadcast to ``V.value.shape``, which is
          ``(batch,) + pop_size`` on a lumped cell and
          ``(batch,) + pop_size + (n_cv,)`` on a cable cell, where one value
          per member has shape ``pop_size + (1,)``;
        - on a Cell running a reduction model, which has no membrane
          voltage (``init_state`` raises when an input is already there).

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import brainunit as u
            >>> import braincell as bc
            >>> cell = bc.Cell.point(2, V_init=-65 * u.mV)
            >>> cell = cell.paint(bc.mech.Channel("IL", g_max=0.3 * u.mS / u.cm**2, E=-65 * u.mV))
            >>> cell.init_state()
            >>> cell.add_delta_input("kick", 5.0 * u.mV)
            >>> with brainstate.environ.context(t=0.0 * u.ms, dt=0.025 * u.ms):
            ...     _ = cell.update()
            >>> cell.V.value
            Quantity([-60. -60.], "mV")
        """
        super().add_delta_input(key, inp, label=label)
        if callable(inp):
            self._drop_compiled_loops()

    def _drop_compiled_loops(self) -> None:
        """Drop the compiled loops of :meth:`run` and of the owning Network.

        They trace the callable inputs (``add_current_input``,
        ``add_delta_input``) once, so a later registration would be missed.
        Value inputs are arguments of the loops (:meth:`_take_value_inputs`).
        """
        self._run_loop_cache.clear()
        owner = self.network_owner
        if owner is not None:
            owner._drop_compiled_loops()

    def _take_value_inputs(self) -> tuple:
        """Take the value inputs for a compiled loop; none on a reduction model, which has no membrane."""
        if self._uses_reduction:
            return {}, {}
        return super()._take_value_inputs()

    def _begin_step(self):
        """Apply prepared discrete synaptic events at step start.

        Notes
        -----
        The public membrane voltage is stored in CV space, while placed
        runtime synapses live on morphology point layouts. This method first
        projects ``V`` from CVs to points, then calls each runtime synapse's
        ``apply_events`` hook.

        This phase consumes synaptic input that has already been prepared or
        delivered; it does not integrate continuous synapse dynamics.
        """
        self._raise_if_not_initialized("_begin_step()")
        self._next_synapse_inputs_prepared.value = jnp.asarray(False)
        if self._uses_reduction:
            self._pending_reduction_inputs = self._reduction_input_runtime.take_inputs()
            return
        if self._is_lumped:
            if self._runtime.event_buffers:
                self._apply_runtime_synapse_events(self._cv_to_point(self.V.value))
            return
        point_V = self._point_voltage_for_mechanisms(self.V.value)
        self._apply_runtime_synapse_events(point_V)

    def _prepare_step_clamps(self, *, t=None, dt=None) -> None:
        """Sample all current clamps at the main-step midpoint and cache them."""
        self._raise_if_not_initialized("_prepare_step_clamps()")
        if self._uses_reduction:
            return
        if self._is_lumped:
            if len(self._lumped_clamps) == 0:
                return
            step_t = self._resolve_t() if t is None else t
            step_dt = brainstate.environ.get_dt() if dt is None else dt
            density, components = self._lumped_clamps.evaluate(step_t + 0.5 * step_dt, area=self._lumped_runtime.area)
            self._lumped_I_ext.value = density
            if components is not None:
                self._step_clamp_components.value = components
            return
        if len(self._get_clamp_store().id) == 0:
            return
        step_t = self._resolve_t() if t is None else t
        step_dt = brainstate.environ.get_dt() if dt is None else dt
        components = self._get_clamp_store().evaluate(self._runtime, t=step_t + 0.5 * step_dt)
        self._step_clamp_components.value = components
        self._step_clamp_point_current.value = self._get_clamp_store().scatter_to_points(
            components,
            pop_size=self._runtime.pop_size,
            n_point=self._runtime.n_point,
        )

    def _solver_clamp_point_current(self, *, t):
        """Return the point current prepared for this step.

        Outside a prepared step (a derivative evaluated on its own, or a
        voltage solver such as ``staggered_step`` called directly, which has
        no ``pre_integral``), the clamps are sampled at ``t + dt / 2``.
        """
        if len(self._get_clamp_store().id) == 0:
            return self._step_clamp_point_current.value
        if self._direct_step_clamps_prepared or brainstate.environ.get("_braincell_step_clamps_prepared", False):
            return self._step_clamp_point_current.value
        try:
            dt = brainstate.environ.get_dt()
        except KeyError:
            return self._runtime.evaluate_point_clamps(t=t)
        if u.get_unit(dt).is_unitless:
            return self._runtime.evaluate_point_clamps(t=t)
        components = self._get_clamp_store().evaluate(self._runtime, t=t + 0.5 * dt)
        return self._get_clamp_store().scatter_to_points(
            components,
            pop_size=self._runtime.pop_size,
            n_point=self._runtime.n_point,
        )

    def _update_dynamics(self, external=None):
        """Advance continuous cell dynamics and update spike state.

        Parameters
        ----------
        external : Quantity, optional
            Lumped cells only: external current density for this step,
            already converted and broadcast by :meth:`update`; it is added
            to the prepared clamp density.

        Returns
        -------
        object
            Spike value computed from the transition between the old and new
            membrane voltage.

        Notes
        -----
        This method requires ``brainstate.environ['dt']`` to be set. The
        selected solver is responsible for advancing membrane voltage and
        mechanism states.

        For the default ``"staggered"`` solver, the typical order is:

        1. Cache ion total currents when enabled.
        2. Advance membrane voltage with the DHS voltage solver.
        3. Integrate runtime synapse continuous dynamics.
        4. Update ion and channel states.
        5. Add the delta inputs (:meth:`add_delta_input`) to the voltage.
        6. Clear temporary current caches.
        7. Detect threshold crossing and write ``self.spike``.

        The value inputs registered for the step are read by every stage
        and removed when the step ends (:meth:`_value_input_step`).
        """
        self._raise_if_not_initialized("_update_dynamics()")

        if brainstate.environ.get("dt", None) is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")
        if self._uses_reduction:
            self._raise_if_reduction_has_delta_inputs()
            output = self._reduction_models[self._selected_model_name].update(self._pending_reduction_inputs)
            self._publish_reduction_output(output)
            return self.spike.value
        # A value input (add_current_input, add_delta_input) acts on this
        # step only: every solver stage reads it, and the step removes it.
        with self._value_input_step():
            return self._advance_membrane(external)

    def _advance_membrane(self, external):
        """The solver step and spike test of :meth:`_update_dynamics` (not reduction)."""
        if self._is_lumped:
            I_ext = self._lumped_I_ext.value
            if external is not None:
                I_ext = I_ext + external
            # VoltageCrossingSource compares the voltage before and after the step.
            self._event_previous_V.value = self.V.value
            with jax.named_scope("braincell:cell_update:lumped"):
                return self._lumped_runtime.update(I_ext)

        last_V = self.V.value
        self._event_previous_V.value = last_V

        with jax.named_scope("braincell:cell_update:solver"):
            self.solver(self)

        if self._ind_update_after_solver:
            with jax.named_scope("braincell:cell_update:independent_update"):
                self._update_independent_mechanisms(self.V.value)

        with jax.named_scope("braincell:cell_update:clear_ion_total_current_cache"):
            self.clear_ion_total_current_cache()

        with jax.named_scope("braincell:cell_update:spike_update"):
            spk = self.get_spike(last_V, self.V.value)
            self.spike.value = spk
        return spk

    def _prepare_next_synapse_inputs(self, *, t=None):
        """Prepare runtime synapse inputs for a later step.

        Notes
        -----
        This method projects the updated CV voltage to point space and
        rebuilds the runtime synapse event payload. In standalone
        ``Cell.update()``, the prepared input is consumed by the next call to
        ``update``.

        In network execution, delayed arrivals are written by the network
        delivery layer before this preparation phase.

        The prepared payload adds to the buffer, so a boundary must be
        prepared once: ``_begin_step`` consumes it and clears the
        ``_next_synapse_inputs_prepared`` flag this method sets.
        """
        self._raise_if_not_initialized("_prepare_next_synapse_inputs()")
        # Neither the packed reduction inputs nor the lumped synapses (whose
        # voltage is their member's V) bind point voltages here.
        point_V = None if self._uses_reduction or self._is_lumped else self._point_voltage_for_mechanisms(self.V.value)
        if t is None:
            self._prepare_runtime_synapse_inputs(point_V)
        else:
            with brainstate.environ.context(t=t):
                self._prepare_runtime_synapse_inputs(point_V)
        self._next_synapse_inputs_prepared.value = jnp.asarray(True)

    def _prepare_first_synapse_inputs(self, *, t) -> None:
        """Prepare the inputs of the boundary ``t`` a run or ``update()`` starts at, unless a step did.

        Every ``Cell.run`` step and every ``update()`` prepares the boundary
        that follows it, so a run or ``update()`` continuing either finds its
        boundary prepared. Preparing it again would deliver each event on it
        twice (CELL-34); only the first step after ``init_state`` or
        ``reset_state`` prepares here, so an event at the start time is
        delivered once whichever drives that step (CELL-35). The test is on a
        state, not a Python flag, so it holds inside jitted loops.
        """
        prepared = self._next_synapse_inputs_prepared.value
        if not is_traced_value(prepared):
            # An eager call: decide here, since ``cond`` would trace the
            # preparation on every call, taken or not.
            if not bool(prepared):
                self._prepare_next_synapse_inputs(t=t)
            return
        brainstate.transform.cond(prepared, lambda: None, lambda: self._prepare_next_synapse_inputs(t=t))

    def _apply_runtime_synapse_events(self, point_V):
        """Apply bound discrete events to all runtime synapses.

        Parameters
        ----------
        point_V : Quantity
            Membrane voltage projected onto morphology point layouts.

        Notes
        -----
        Runtime synapses receive point-local voltage because placed synapse
        mechanisms are indexed by point layout. The bound discrete event drive
        should already be present on each synapse before this method is called.
        """
        self._raise_if_not_initialized("_apply_runtime_synapse_events()")
        for layout, synapse in self._runtime.iter_synapse_layouts():
            if layout.id not in self._runtime.event_buffers:
                continue
            payload = self._runtime.get_event_buffer(layout.id)
            args = (layout.gather_points(point_V),)
            synapse.apply_events(payload, *args)
            self._runtime.clear_event_buffer(layout.id)

    def _prepare_runtime_synapse_inputs(self, point_V):
        """Bind this step's presynaptic drive to runtime synapses.

        Parameters
        ----------
        point_V : Quantity
            Membrane voltage projected onto morphology point layouts.

        Notes
        -----
        ``SynapsePlacement`` is the ``cell.place(...)`` declaration, while
        ``RuntimeSynapse`` is the executable point mechanism. This method only
        prepares discrete input; synapse dynamics are integrated later by the
        active solver schedule.

        For each runtime synapse layout, the total discrete drive is assembled
        from three sources:

        1. The private per-layout event buffer, where network delivery writes
           delayed events.
        2. Direct Connection arrivals evaluated at the current simulation time.
        3. User-bound inputs registered with ``bind_synapse_input``.

        The accumulated payload is consumed at the next event boundary.
        """
        _ = point_V
        self._raise_if_not_initialized("_prepare_runtime_synapse_inputs()")
        t = self._resolve_t()
        for layout in self._event_layouts():
            if layout.id not in self._event_runtime().event_buffers:
                continue
            total_drive = self._event_runtime().get_event_buffer(layout.id)
            contact_drive = self._evaluate_contact_inputs(
                layout,
                t=t,
                template=total_drive,
                scheduled_only=True,
            )
            total_drive = total_drive + _coerce_drive_like(contact_drive, total_drive)
            total_drive = total_drive + self._evaluate_bound_synapse_inputs(
                layout,
                total_drive,
            )
            self._set_event_buffer(layout.id, total_drive)

    def _evaluate_contact_inputs(self, layout, *, t, template, scheduled_only=True):
        """Return weighted Connection arrivals addressed to one synapse layout."""
        if layout.placement_index is None:
            return u.math.zeros_like(template)
        dt = brainstate.environ.get("dt", None)
        if dt is None:
            raise ValueError("Connection event delivery requires brainstate.environ['dt'].")

        output = _zeros_like_event_template(template)
        synapse_store = self._get_synapse_store()
        for connection in self.connections._call_views(scheduled=scheduled_only):
            synapse_type = str(connection.synapse_type[0])
            if synapse_store.layout_id(synapse_type) != int(layout.id):
                continue
            row_index = np.arange(len(connection), dtype=np.int32)
            local_index = synapse_store.runtime_rows(connection.synapse_id).astype(np.int32)
            event_count = connection.source.event_count(
                connection.source_index[row_index],
                t=t,
                delay=connection.delay[row_index],
                dt=dt,
            )
            connection_weight = connection.weight
            if connection_weight is not None:
                connection_weight = connection_weight[row_index]
            contribution = event_count * u.math.asarray(_connection_event_weight(template, connection_weight))
            output = output.at[local_index].add(contribution)
        return _rewrap_event_template(template, output)

    def _apply_direct_live_connection_events(self) -> None:
        """Route live direct sources and run target handlers at this boundary."""
        live_connections = self.connections._call_views(scheduled=False)
        if not live_connections:
            return
        dt = brainstate.environ.get("dt", None)
        if dt is None:
            raise ValueError("Live Connection delivery requires brainstate.environ['dt'].")
        t = self._resolve_t()
        layouts = self._event_layouts()
        drives = {}
        event_runtime = self._event_runtime()
        for layout in layouts:
            if layout.id not in event_runtime.event_buffers:
                continue
            drives[layout.id] = _zeros_like_event_template(event_runtime.get_event_buffer(layout.id))

        synapse_store = self._get_synapse_store()
        for connection in live_connections:
            counts = connection.event_count(t=t, dt=dt)
            synapse_type = str(connection.synapse_type[0])
            layout_id = synapse_store.layout_id(synapse_type)
            if layout_id not in drives:
                continue
            template = event_runtime.get_event_buffer(layout_id)
            contribution = counts * u.math.asarray(_connection_event_weight(template, connection.weight))
            local_indices = synapse_store.runtime_rows(connection.synapse_id).astype(np.int32)
            drives[layout_id] = drives[layout_id].at[local_indices].add(contribution)

        point_v = None if self._uses_reduction else self._point_voltage_for_mechanisms(self.V.value)
        for layout in layouts:
            if layout.id not in drives:
                continue
            template = event_runtime.get_event_buffer(layout.id)
            drive = _rewrap_event_template(template, drives[layout.id])
            self._apply_synapse_layout_event_drive(layout.id, drive, point_v=point_v)

    def _apply_synapse_layout_event_drive(self, layout_id: int, drive, *, point_v=None) -> None:
        """Apply one already-aggregated boundary payload to a runtime layout."""
        if self._uses_reduction:
            current = self._reduction_input_runtime.get_event_buffer(layout_id)
            self._set_event_buffer(layout_id, current + _coerce_drive_like(drive, current))
            return
        layout = self._runtime.layouts[int(layout_id)]
        synapse = self._runtime.get_runtime_node(layout.id)
        if point_v is None:
            point_v = self._point_voltage_for_mechanisms(self.V.value)
        args = (layout.gather_points(point_v),)
        synapse.apply_events(drive, *args)

    def _event_runtime(self):
        """Return the detailed or reduced owner of packed event buffers."""
        return self._reduction_input_runtime if self._uses_reduction else self._runtime

    def _event_layouts(self):
        """Return executable synapse input layouts without exposing runtime nodes."""
        if self._uses_reduction:
            return self._reduction_input_runtime.layouts
        return tuple(layout for layout, _ in self._runtime.iter_synapse_layouts())

    def _event_layout(self, layout_id: int):
        """Return one input layout by its stable runtime id."""
        for layout in self._event_layouts():
            if int(layout.id) == int(layout_id):
                return layout
        raise KeyError(f"Unknown synapse event layout id {layout_id!r}.")

    def _set_event_buffer(self, layout_id: int, value) -> None:
        self._event_runtime().event_buffers[int(layout_id)].value = value

    def _write_event_arrival(self, layout_id: int, arrival) -> None:
        """Merge one Network arrival with any event staged for this boundary."""
        if self._uses_reduction:
            current = self._reduction_input_runtime.get_event_buffer(layout_id)
            arrival = current + _coerce_drive_like(arrival, current)
        self._set_event_buffer(layout_id, arrival)

    def _evaluate_bound_synapse_inputs(self, layout, template):
        drive = u.math.zeros_like(template)
        if layout.synapse_index is None:
            return drive
        layout_ids = np.asarray(layout.synapse_index, dtype=np.int64)
        for instance_name, bindings in self._synapse_input_bindings.items():
            target = self.synapses[instance_name]
            if len(target) == 0:
                continue
            selected_ids = target.id[np.isin(target.id, layout_ids)]
            if selected_ids.size == 0:
                continue
            rows = self._get_synapse_store().runtime_rows(selected_ids)
            selected_template = template[..., rows]
            for source, weight, transform in bindings:
                value = source() if callable(source) else source
                if transform is not None:
                    value = transform(value)
                try:
                    contribution = _coerce_drive_like(value * weight, selected_template)
                    drive = _scatter_drive_rows(drive, rows, contribution)
                except ValueError as exc:
                    raise ValueError(
                        f"Bound synapse input for {instance_name!r} cannot broadcast "
                        f"from shape {getattr(value, 'shape', None)!r} to "
                        f"{getattr(selected_template, 'shape', None)!r}."
                    ) from exc
        return drive

    def _update_runtime_synapses(self, point_V):
        """Advance runtime synapse dynamics.

        Parameters
        ----------
        point_V : Quantity
            Membrane voltage projected onto morphology point layouts.

        Notes
        -----
        This helper refreshes discrete synaptic input and then integrates
        runtime synapse continuous states. It is used by solver schedules that
        update synapses as part of the post-voltage mechanism phase.
        """
        self._prepare_runtime_synapse_inputs(point_V)
        self._integrate_runtime_synapse_dynamics(point_V)

    def _integrate_runtime_synapse_dynamics(self, point_V):
        """Integrate continuous runtime synapse states.

        Parameters
        ----------
        point_V : Quantity
            Membrane voltage projected onto morphology point layouts.

        Notes
        -----
        Only runtime synapse nodes are advanced here. Discrete events should
        already have been applied before this method is called.
        """
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, RuntimeSynapse):
                continue
            layout_id = layout_id_from_key(path)
            layout = self._runtime.layouts[layout_id]
            args = (layout.gather_points(point_V),)
            with jax.named_scope(_scope_name("braincell:synapse_update:runtime", path, node)):
                jax.named_call(
                    ind_exp_euler_step,
                    name=_call_name("braincell:synapse_update:runtime_step", path, node),
                )(node, *args)

    def reset_state(self, batch_size=None) -> None:
        """Reseed ``V`` / ``spike`` / ``current_time`` without leaving INITIALIZED.

        Distinct from :meth:`reset`: ``reset_state`` is the in-phase
        brainstate lifecycle hook; ``reset`` tears down the runtime
        entirely and returns the cell to DECLARING.
        """
        self._raise_if_network_owned("reset_state()")
        self._raise_if_not_initialized("reset_state()")
        if batch_size is not None and (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))):
            raise TypeError("Cell reset_state() batch_size must be an integer or None.")
        requested_batch = None if batch_size is None else int(batch_size)
        if requested_batch != self._runtime_batch_size:
            raise ValueError(
                "Cell reset_state() must preserve the init_state() batch size; "
                f"expected {self._runtime_batch_size!r}, got {requested_batch!r}."
            )
        self._direct_step_clamps_prepared = False
        if self._uses_reduction:
            self.connections.reset_runtime()
            self._reduction_input_runtime.clear_event_buffers()
            output = self._reduction_models[self._selected_model_name].reset_state(batch_size=batch_size)
            self._publish_reduction_output(output)
            self._pending_reduction_inputs = self._reduction_input_runtime.take_inputs()
            self._next_synapse_inputs_prepared.value = jnp.asarray(False)
            self._current_time_state.value = 0.0 * u.ms
            return
        if self._is_lumped:
            if self.trainables.bindings():
                self.trainables.materialize()
            self.connections.reset_runtime()
            for layout_id in self._runtime.event_buffers:
                self._runtime.clear_event_buffer(layout_id)
            self._next_synapse_inputs_prepared.value = jnp.asarray(False)
            self._lumped_runtime.reset_state(batch_size)
            self._event_previous_V.value = self.V.value
            self._lumped_I_ext.value = self._zero_lumped_current()
            self._step_clamp_components.value = u.Quantity(
                jnp.zeros_like(self._step_clamp_components.value.mantissa), u.nA
            )
            self._current_time_state.value = 0.0 * u.ms
            return
        if self.trainables.bindings():
            self.trainables.materialize()
        self.connections.reset_runtime()
        v_value = self._materialize_population_parameter("V_init")
        self._V_init_materialized.value = v_value
        self.V.value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        self._point_V.value = self._initial_point_voltage(self.V.value)
        self.spike.value = _zero_spike_like(self.V.value)
        self._event_previous_V.value = self.V.value
        self._current_time_state.value = 0.0 * u.ms
        self._step_clamp_components.value = u.Quantity(jnp.zeros_like(self._step_clamp_components.value.mantissa), u.nA)
        self._step_clamp_point_current.value = u.Quantity(
            jnp.zeros_like(self._step_clamp_point_current.value.mantissa), u.nA
        )
        for layout_id in self._runtime.event_buffers:
            self._runtime.clear_event_buffer(layout_id)
        self._next_synapse_inputs_prepared.value = jnp.asarray(False)
        with state_grouping(True):
            for path, channel in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, self.V.value)
                channel.reset_state(*args, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Inspection forwards (runtime-only)

    @property
    def layouts(self):
        self._raise_if_not_initialized("layouts")
        if self._is_lumped:
            raise RuntimeError("A lumped Cell (cv_policy=Lumped()) has no cable mechanism layouts.")
        if self._uses_reduction:
            raise RuntimeError("Detailed mechanism layouts are unavailable while a reduction model is selected.")
        return self._runtime.layouts

    @property
    def voltage_shape(self):
        self._raise_if_not_initialized("voltage_shape")
        if self._is_lumped:
            return tuple(self.pop_size)
        return self._runtime.voltage_shape

    def get_point_layouts(self, point_id):
        self._raise_if_not_initialized("get_point_layouts()")
        return self._runtime.get_point_layouts(point_id)

    def get_cv_layouts(self, cv_id):
        self._raise_if_not_initialized("get_cv_layouts()")
        return self._runtime.get_cv_layouts(cv_id)

    def expected_state_shape(self, layout_id, var_name):
        self._raise_if_not_initialized("expected_state_shape()")
        return self._runtime.expected_state_shape(layout_id, var_name)

    def get_state(self, layout_id, var_name):
        self._raise_if_not_initialized("get_state()")
        return self._runtime.get_state(layout_id, var_name)

    def set_state(self, layout_id, var_name, value) -> None:
        self._raise_if_not_initialized("set_state()")
        self._runtime.set_state(layout_id, var_name, value)

    def get_point_state(self, point_id):
        self._raise_if_not_initialized("get_point_state()")
        return self._runtime.get_point_state(point_id)

    def get_placement_state(self, placement_id):
        """Return runtime state for one independent point placement."""
        self._raise_if_not_initialized("get_placement_state()")
        self.get_point_placement(placement_id)
        return self._runtime.get_placement_state(placement_id)

    def get_cv_state(self, cv_id):
        self._raise_if_not_initialized("get_cv_state()")
        return self._runtime.get_cv_state(cv_id)

    def get_runtime_node(self, layout_id):
        self._raise_if_not_initialized("get_runtime_node()")
        return self._runtime.get_runtime_node(layout_id)

    def get_ion(self, name):
        self._raise_if_not_initialized("get_ion()")
        return self._runtime.get_ion(name)

    # ------------------------------------------------------------------
    # Probes + mech_table (runtime-only)

    def sample_probe(self, name: str):
        self._raise_if_not_initialized("sample_probe()")
        if self._uses_reduction:
            raise KeyError(f"Detailed probe {name!r} is inactive while a reduction model is selected.")
        return probes.sample_probe(self, name)

    def sample_probes(self) -> dict[str, object]:
        self._raise_if_not_initialized("sample_probes()")
        if self._uses_reduction:
            return {}
        return probes.sample_probes(self)

    def mech_table(self) -> MechanismObjectTable:
        """Return the point-domain mechanism table for this cell.

        Returns
        -------
        braincell._compute.table.MechanismObjectTable
            Rows are mechanism identities, columns are point ids.

        Raises
        ------
        RuntimeError
            If :meth:`init_state` has not been called.
        """
        self._raise_if_not_initialized("mech_table()")
        if self._uses_reduction:
            raise RuntimeError("Detailed mechanism inspection is unavailable while a reduction model is selected.")
        return build_mechanism_object_table(self._runtime, self.cvs)

    # ------------------------------------------------------------------
    # Run (auto-inits from DECLARING)

    def run(self, *, dt, duration):
        """Run the cell for ``duration`` at ``dt`` and return probe traces.

        If :meth:`init_state` has not been called yet, ``run`` calls it
        automatically. Once initialized the cell will *not* be
        re-initialized on subsequent ``run`` invocations: each ``run`` (or
        :meth:`update`) continues from where the last one stopped, so
        splitting a run into segments, at event times included, gives the
        same trajectory as one run.

        Warns
        -----
        RuntimeWarning
            Once per edit of the declaration morphology made after
            :meth:`init_state`: the run uses the snapshot taken at
            initialization, and the edit applies after :meth:`reset` and
            :meth:`init_state`.
        """
        owner = self.network_owner
        if owner is not None:
            owner_name = owner.name if owner.name is not None else "<unnamed>"
            raise RuntimeError(f"Cell belongs to Network {owner_name!r}; run it through Network {owner_name!r}.")
        if not self._initialized:
            self.init_state()
        elif not self._uses_reduction and self.trainables.bindings():
            self.trainables.materialize()
        if self._unannounced_declaration_edit():
            warnings.warn(
                "This Cell runs on the snapshot of its morphology taken at init_state(); the morphology it "
                "was declared with has been edited since, and the edit does not apply to this run. Call "
                "reset() and then init_state() to run the edited morphology.",
                RuntimeWarning,
                stacklevel=2,
            )
        return run_module.run(self, dt=dt, duration=duration)


#: Alias of :class:`Cell`, named for what the model is rather than for the
#: shorthand. ``MultiCompartment is Cell`` — the two names are the same
#: object, so ``isinstance``, subclassing, and pickling behave identically
#: through either. Prefer it when the surrounding code also mentions
#: :class:`~braincell.SingleCompartment` and the contrast matters.
MultiCompartment = Cell


# ----------------------------------------------------------------------
# Helpers


def _lists_branch_indices(policy: CVPolicy) -> bool:
    """Whether ``policy`` gives CV counts by default branch index, directly or in a composite."""
    if isinstance(policy, CVPerBranchList):
        return True
    if isinstance(policy, CompositeByTypePolicy):
        return _lists_branch_indices(policy.default_policy) or any(
            _lists_branch_indices(rule.policy) for rule in policy.rules
        )
    return False


def _cv_policy_index_basis(policy: CVPolicy, morpho: Morphology) -> tuple[Morphology, int] | None:
    """``(morpho, revision)`` when ``policy`` holds branch indices, else ``None``."""
    return (morpho, morpho.revision) if _lists_branch_indices(policy) else None


def _is_per_cell_locset_sequence(value) -> bool:
    """Return whether ``value`` is the public sequence-of-locsets form."""
    if isinstance(value, (LocsetExpr, LocsetMask, LocsetBatch, str, bytes)):
        return False
    return isinstance(value, (tuple, list))


def _resolved_locset_length(locset, morpho: Morphology) -> int:
    """Resolve only the row count needed for parameter broadcasting."""
    if isinstance(locset, LocsetExpr):
        return len(locset.evaluate(morpho))
    return len(locset)


def _split_synapse_rows(
    mechanism: SynapsePlacement,
    *,
    lengths: tuple[int, ...],
) -> tuple[SynapsePlacement, ...]:
    """Materialize per-cell declaration params for rectangular or ragged rows."""
    parameter_rows = {
        name: _split_synapse_parameter_rows(value, lengths=lengths, name=name)
        for name, value in mechanism.params.items()
    }
    return tuple(
        SynapsePlacement(
            mechanism.synapse_type,
            name=mechanism.name,
            **{name: rows[row] for name, rows in parameter_rows.items()},
        )
        for row in range(len(lengths))
    )


def _split_synapse_parameter_rows(value, *, lengths: tuple[int, ...], name: str) -> tuple[object, ...]:
    """Split one synapse parameter according to per-cell location lengths."""
    n_row = len(lengths)
    n_total = int(sum(lengths))
    common_length = lengths[0] if lengths and all(length == lengths[0] for length in lengths) else None

    if _is_ragged_parameter_sequence(value, n_row=n_row):
        rows = []
        for row, (item, length) in enumerate(zip(value, lengths)):
            rows.append(_normalize_synapse_parameter_row(item, length=length, name=name, row=row))
        return tuple(rows)

    unit = value.unit if isinstance(value, u.Quantity) else None
    array = np.asarray(value.to_decimal(unit) if unit is not None else value)
    if array.shape == ():
        return tuple(value for _ in lengths)
    if common_length is not None and array.shape == (common_length,):
        return tuple(_with_optional_unit(np.array(array, copy=True), unit) for _ in lengths)
    if array.shape == (n_row, 1):
        return tuple(_with_optional_unit(array[row, 0], unit) for row in range(n_row))
    if common_length is not None and array.shape == (n_row, common_length):
        return tuple(_with_optional_unit(np.array(array[row], copy=True), unit) for row in range(n_row))
    if array.shape == (n_total,):
        rows = []
        offset = 0
        for length in lengths:
            rows.append(_with_optional_unit(np.array(array[offset : offset + length], copy=True), unit))
            offset += length
        return tuple(rows)
    raise ValueError(
        f"Synapse parameter {name!r} with shape {array.shape!r} cannot broadcast to "
        f"per-cell location lengths {lengths!r}."
    )


def _is_ragged_parameter_sequence(value, *, n_row: int) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != n_row:
        return False
    return any(isinstance(item, (tuple, list)) or getattr(item, "shape", ()) not in ((), None) for item in value)


def _normalize_synapse_parameter_row(value, *, length: int, name: str, row: int):
    unit = value.unit if isinstance(value, u.Quantity) else None
    array = np.asarray(value.to_decimal(unit) if unit is not None else value)
    if array.shape == ():
        return value
    if array.shape != (length,):
        raise ValueError(
            f"Synapse parameter {name!r} row {row!r} must be scalar or shape {(length,)!r}, got {array.shape!r}."
        )
    return _with_optional_unit(np.array(array, copy=True), unit)


def _with_optional_unit(value, unit):
    return u.Quantity(value, unit) if unit is not None else value


def _select_local_values(values, *, ids: tuple[int, ...]):
    """Return one localized item or a small indexed slice from an array-like."""
    if len(ids) == 1:
        return values[int(ids[0])]
    return values[list(int(idx) for idx in ids)]


def _select_population_value(value, *, population_indices: tuple[int, ...], population_size: int):
    """Gather the population axis of an array-like value when it has one."""
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) == 0:
        return value
    if len(shape) >= 2 and shape[-2] == population_size:
        axis = len(shape) - 2
    elif shape[0] == population_size:
        axis = 0
    else:
        return value
    index = [slice(None)] * len(shape)
    index[axis] = np.asarray(population_indices, dtype=np.int32)
    return value[tuple(index)]


def _select_model_output(
    value,
    *,
    population_indices: tuple[int, ...],
    population_size: int,
    batch_size: int | None,
    detailed: bool,
):
    """Gather the explicit population axis from a detailed or reduced output."""
    shape = tuple(getattr(value, "shape", ()))
    if not shape:
        return value
    axis = (1 if batch_size is not None else 0) if not detailed else len(shape) - 2
    if shape[axis] != population_size:
        raise RuntimeError(f"Cell output population axis has size {shape[axis]!r}; expected {population_size!r}.")
    index = [slice(None)] * len(shape)
    index[axis] = np.asarray(population_indices, dtype=np.int32)
    return value[tuple(index)]


def _require_same_value_type(previous, current, *, name: str) -> None:
    """Require one reduced output to preserve dtype and exact unit."""
    previous_unit = u.get_unit(previous)
    current_unit = u.get_unit(current)
    if previous_unit != current_unit:
        raise TypeError(f"{name} changed unit from {previous_unit!r} to {current_unit!r}.")
    previous_dtype = jnp.asarray(u.get_magnitude(previous)).dtype
    current_dtype = jnp.asarray(u.get_magnitude(current)).dtype
    if previous_dtype != current_dtype:
        raise TypeError(f"{name} changed dtype from {previous_dtype!r} to {current_dtype!r}.")


def _select_packed_population_value(value, *, owners: np.ndarray, population_indices: tuple[int, ...]):
    """Gather packed point-instance rows owned by selected population members."""
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) == 0:
        return value
    selected = np.flatnonzero(np.isin(np.asarray(owners), np.asarray(population_indices)))
    if shape[0] == len(owners):
        return value[selected]
    if shape[-1] == len(owners):
        return value[..., selected]
    return value


def _as_float_voltage(value):
    """Cast an integer-valued voltage to the default float dtype.

    Floating values keep their dtype; a ``Quantity`` keeps its unit.
    """
    mantissa = value.mantissa if isinstance(value, u.Quantity) else value
    mantissa = jnp.asarray(mantissa)
    if jnp.issubdtype(mantissa.dtype, jnp.floating):
        return value
    mantissa = mantissa.astype(brainstate.environ.dftype())
    return u.Quantity(mantissa, value.unit) if isinstance(value, u.Quantity) else mantissa


def _normalize_selected_voltage_parameter(value, *, count: int, n_cv: int, name: str) -> tuple[object, ...]:
    """Normalize selected cell-level voltage declarations to one CV row each."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"CellView {name} must be a voltage quantity.")
    try:
        decimal = np.asarray(value.to_decimal(u.mV), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"CellView {name} must have voltage units.") from exc

    target_shape = (count, n_cv)
    if decimal.ndim == 0:
        normalized = np.broadcast_to(decimal, target_shape)
    elif decimal.shape == (count,):
        normalized = np.broadcast_to(decimal[:, None], target_shape)
    elif count == 1 and decimal.shape == (n_cv,):
        normalized = decimal[None, :]
    else:
        try:
            normalized = np.broadcast_to(decimal, target_shape)
        except ValueError as exc:
            raise ValueError(
                f"CellView {name} with shape {decimal.shape!r} cannot broadcast to {target_shape!r}."
            ) from exc
    return tuple(u.Quantity(np.array(row, copy=True), u.mV) for row in normalized)


def _coerce_drive_like(value, template):
    """Coerce dimensionless zero drives to a quantity template unit."""
    if isinstance(template, u.Quantity) and not isinstance(value, u.Quantity):
        return value * template.unit
    return value


def _scatter_drive_rows(target, rows, contribution):
    """Add selected event-drive rows without changing the target unit."""
    rows = np.asarray(rows, dtype=np.int32)
    if isinstance(target, u.Quantity):
        if not isinstance(contribution, u.Quantity):
            raise TypeError("Synapse drive contribution requires a quantity.")
        mantissa = jnp.asarray(target.to_decimal(target.unit))
        values = jnp.asarray(contribution.to_decimal(target.unit))
        return u.Quantity(mantissa.at[..., rows].add(values), target.unit)
    if isinstance(contribution, u.Quantity):
        raise TypeError("Dimensionless synapse drive cannot consume a quantity contribution.")
    return jnp.asarray(target).at[..., rows].add(jnp.asarray(contribution))


def _resolve_solver(solver):
    if isinstance(solver, str):
        # Record the canonical name of braincell's own steps, so an alias such
        # as "stagger" selects the same solver-specific paths as "staggered".
        # ODE methods keep the given name: braintools' canonical name of
        # "heun2" is "rk2", which braincell no longer accepts.
        info = integrator_info(solver)
        return (info.name if info.kind == "module" else solver), get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(f"solver must be str or callable, got {type(solver).__name__!s}.")


#: Generic steps that merge every DiffEqState of a Cell into one array per
#: cell (``diffeq_state_merging``) and solve it as a block.
_BLOCK_SOLVERS = ("exp_euler", "backward_euler", "crank_nicolson")


def _is_generic_step(fn) -> bool:
    """Whether ``fn`` is a registered integrator other than a cable-only step."""
    return any(get_integrator(name) is fn for name in list_integrators() if name not in CABLE_ONLY_SOLVERS)


def _resolve_cell_solver(solver, *, lumped: bool):
    """Resolve a Cell solver declaration for the cable or the lumped runtime."""
    if solver is None:
        solver = "ralston2" if lumped else "staggered"
    name, fn = _resolve_solver(solver)
    if lumped and name in CABLE_ONLY_SOLVERS:
        raise ValueError(
            f"solver {name!r} steps the cable point tree and cannot advance a lumped Cell "
            "(cv_policy=Lumped()); use a single-step ODE solver such as 'ralston2', 'rk4' or 'exp_euler'."
        )
    return name, fn


def _resolve_subsolver_schedule(subsolver, substeps):
    if subsolver is None and substeps is None:
        subsolver = "backward_euler"
        substeps = 1
    elif subsolver is None or substeps is None:
        raise ValueError("subsolver and substeps must be provided together or both be None.")
    if isinstance(substeps, bool):
        raise TypeError("substeps must be an integer, got bool.")
    try:
        normalized_substeps = operator.index(substeps)
    except TypeError as exc:
        raise TypeError(f"substeps must be an integer, got {type(substeps).__name__!s}.") from exc
    if normalized_substeps < 1:
        raise ValueError(f"substeps must be at least 1, got {normalized_substeps!r}.")
    solver_name, solver_fn = _resolve_solver(subsolver)
    return solver_name, solver_fn, normalized_substeps


def _zeros_like_event_template(template):
    """Build a zeroed mantissa array shaped like one event buffer template."""
    raw = template.to_decimal(template.unit) if isinstance(template, u.Quantity) else template
    return jnp.zeros_like(jnp.asarray(raw))


def _rewrap_event_template(template, mantissa):
    """Restore the event buffer template's unit onto an accumulated mantissa."""
    return u.Quantity(mantissa, template.unit) if isinstance(template, u.Quantity) else mantissa


def _connection_event_weight(template, connection_weight):
    """Coerce one Connection weight into its target event buffer's unit system.

    Parameters
    ----------
    template : array_like or brainunit.Quantity
        Event buffer the weighted arrivals are accumulated into.
    connection_weight : array_like, brainunit.Quantity, or None
        Declared Connection weight, already narrowed to the contributing rows.
        ``None`` marks a trigger-only Connection.

    Returns
    -------
    array_like
        Weight expressed in ``template``'s unit, ready to multiply event counts.

    Raises
    ------
    TypeError
        If the weight and the target event buffer disagree on dimensionality.
    """
    if connection_weight is None:
        if isinstance(template, u.Quantity):
            raise TypeError("Trigger-only Connection cannot target a physical event buffer.")
        return 1.0
    if isinstance(template, u.Quantity):
        if not isinstance(connection_weight, u.Quantity):
            raise TypeError("Connection weight is dimensionless but its target event buffer is not.")
        return connection_weight.to_decimal(template.unit)
    if isinstance(connection_weight, u.Quantity):
        raise TypeError("Connection weight has units but its target event buffer is dimensionless.")
    return connection_weight


_RANK0_POP_SIZE_MESSAGE = (
    "Cell requires a population axis, so pop_size must not be empty. "
    "Use pop_size=1 for a single cell; runtime state is then shaped "
    "pop_size + (n_cv,). The trailing compartment axis is what makes every "
    "Cell hidden state a brainstate.HiddenGroupState, which requires rank >= 2."
)


def _normalize_pop_size(pop_size) -> tuple[int, ...]:
    """Normalize the public ``Cell(pop_size=...)`` argument.

    A ``Cell`` always carries a population axis: the canonical shape has at
    least one entry, so runtime state is at least two-dimensional
    (``pop_size + (n_cv,)``). Rank-0 populations are rejected so that every
    hidden state groups its trailing compartment axis and rank-0 and rank-1
    populations never take different code paths.

    Parameters
    ----------
    pop_size : int, sequence of int, or None
        User-facing homogeneous population shape. ``None`` means
        "unspecified" and normalizes to ``(1,)``.

    Returns
    -------
    tuple of int
        Canonical population-shape tuple, never empty.

    Raises
    ------
    TypeError
        If ``pop_size`` is not an integer or sequence of integers.
    ValueError
        If any requested dimension is non-positive, or if an explicitly
        empty ``pop_size`` is given.

    Examples
    --------
    .. code-block:: python

        >>> from braincell._multi_compartment.cell import _normalize_pop_size
        >>> _normalize_pop_size(None)
        (1,)
        >>> _normalize_pop_size(4)
        (4,)
        >>> _normalize_pop_size((2, 3))
        (2, 3)
    """
    if pop_size is None:
        return (1,)
    if isinstance(pop_size, (int, np.integer)):
        if int(pop_size) <= 0:
            raise ValueError(f"pop_size must be > 0, got {pop_size!r}.")
        return (int(pop_size),)
    if isinstance(pop_size, (tuple, list)):
        if len(pop_size) == 0:
            raise ValueError(_RANK0_POP_SIZE_MESSAGE)
        normalized = []
        for dim in pop_size:
            if not isinstance(dim, (int, np.integer)):
                raise TypeError(f"pop_size entries must be integers, got {type(dim).__name__!s}.")
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"pop_size entries must be > 0, got {pop_size!r}.")
            normalized.append(dim)
        return tuple(normalized)
    raise TypeError(f"pop_size must be int or tuple/list of int, got {type(pop_size).__name__!s}.")


def _validate_ion_channel_update_order(value: str) -> str:
    # "family" is the ion-before-channel schedule; "integration" is the
    # previous schedule grouped by IndependentIntegration at the top level.
    if value not in {"family", "integration"}:
        raise ValueError(f"ion_channel_update_order must be 'family' or 'integration', got {value!r}.")
    return value


def _validate_membrane_linearizer(value: str) -> str:
    if value not in {"point", "generic"}:
        raise ValueError(f"membrane_linearizer must be 'point' or 'generic', got {value!r}.")
    return value
