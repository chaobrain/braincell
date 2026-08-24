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

The population-view contract is specified in
``docs/specs/2026-08-20-cell-population-view.md``.

A ``Cell`` carries both the declaration (morphology, CV policy, paint /
place rules, solver, spike config) and the runtime (``V`` / ``spike`` /
``current_time`` brainstate states, node tree, axial operator,
installed channel / ion nodes).

The lifecycle has two phases:

1. **DECLARING** (default). ``paint`` / ``place`` / ``cv_policy`` /
   ``V_th`` / ``V_init`` / ``solver`` / ``spk_fun`` setters are all
   mutable. Runtime methods raise.
2. **INITIALIZED**. After :meth:`init_state`, mutation is frozen and
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
from typing import Callable, Mapping, Optional
from dataclasses import dataclass

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base import (
    Channel,
    HHTypedNeuron,
    Ion,
    IonChannel,
    MixIons,
    Synapse as RuntimeSynapse,
    _cast_like,
    _zero_spike_like,
)
from braincell._misc import is_traced_value
from braincell._typing import Initializer, Size
from braincell._compute.table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell._compute.scheduling import build_node_scheduling
from braincell._compute.state import CellRuntimeState
from braincell._compute.bindings import _is_root_level_runtime_node
from braincell._compute.layouts import mechanism_signature
from braincell._discretization.mechanism import (
    PaintRule,
    PlaceRule,
    default_paint_rules,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell._discretization.policy import CVPerBranch, CVPolicy
from braincell._discretization.base import (
    CV,
    CVTree,
    Discretization,
    Node,
    NodeTree,
    build_discretization,
)
from braincell._discretization.node_build import (
    _EPS_PARAM,
    _locate_branch_cv_by_x,
    locate_node_on_branch,
)
from braincell.filter import LocsetBatch, LocsetExpr, LocsetMask, RegionExpr, RegionMask, at
from braincell.filter.helper import normalize_region_intervals
from braincell.event import EventOutputCollection, _CellSpikeSource
from braincell.recording import RecordingSpec, compile_recording
from braincell.morph.morphology import Morphology, clone_morpho
from braincell.quad import get_integrator, ind_exp_euler_step
from braincell.quad._exp_euler import _ind_exp_euler_step_selected
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqGroupState, IndependentIntegration, state_grouping
from braincell.mech import CVContext, SynapseSpec as SynapsePlacement
from . import currents, probes, run as run_module
from braincell._compute import bridge
from .synapses import SynapseView, _SynapseStore
from .selection import BranchSelector, CVSelector, _CellScope
from .density_views import ChannelView, IonView

__all__ = ["Cell", "CellView", "CellSelection", "MultiCompartment"]


@dataclass(frozen=True)
class AxialOperatorCache:
    float_dtype: jnp.dtype
    operator: object


@dataclass(frozen=True)
class RuntimeIonBinding:
    """One runtime ion seen through a CV or node inspection view."""

    name: str
    runtime: object
    cell: "Cell"
    cv_ids: tuple[int, ...] = ()
    point_ids: tuple[int, ...] = ()

    def get(self, field: str):
        """Return one field projected into the local CV or node view."""
        if not hasattr(self.runtime, field):
            raise AttributeError(f"Runtime ion {self.name!r} has no field {field!r}.")
        raw = getattr(self.runtime, field)
        if self.cv_ids:
            values = self.cell._coerce_named_vis_cv_values_object(raw)
            return _select_local_values(values, ids=self.cv_ids)
        if self.point_ids:
            values = self.cell._coerce_runtime_point_values_object(raw)
            return _select_local_values(values, ids=self.point_ids)
        return raw

    def __getattr__(self, field: str):
        if field.startswith("_"):
            raise AttributeError(field)
        return self.get(field)


@dataclass(frozen=True)
class RuntimeCVView:
    """Readonly runtime inspection view anchored at one static CV."""

    id: int
    declaration: CV
    layout_ids: tuple[int, ...]
    mid_node_id: int
    ions: Mapping[str, RuntimeIonBinding]


@dataclass(frozen=True)
class RuntimeNodeView:
    """Readonly runtime inspection view anchored at one static node."""

    id: int
    declaration: Node
    layout_ids: tuple[int, ...]
    source_cv_ids: tuple[int, ...]
    ions: Mapping[str, RuntimeIonBinding]


class _CellFacade:
    """Share population-selection behavior without owning model data."""

    @property
    def _view_root(self) -> "Cell":
        raise NotImplementedError

    @property
    def _view_population_indices(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def root(self) -> "Cell":
        """Return the root Cell that owns this view."""
        return self._view_root

    @property
    def _scope(self) -> _CellScope:
        return _CellScope.root(self._view_root)

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


class PopulationRuntimeView:
    """Provide read-only population selection over one runtime object."""

    __slots__ = ("_runtime", "_population_indices", "_population_size", "_packed_population_index")

    def __init__(
        self,
        runtime,
        population_indices: tuple[int, ...],
        population_size: int,
        *,
        packed_population_index: np.ndarray | None = None,
    ) -> None:
        self._runtime = runtime
        self._population_indices = population_indices
        self._population_size = population_size
        self._packed_population_index = packed_population_index

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


class CellView(_CellFacade):
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
            _CellScope.root(cell).select_population(tuple(population_indices)) if scope is None else scope
        )

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
    def _view_population_indices(self) -> tuple[int, ...]:
        return self._population_indices

    @property
    def _scope(self) -> _CellScope:
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
        return len(self._population_indices)

    @property
    def morpho(self) -> Morphology:
        """Return the shared root morphology without copying it."""
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
        return tuple(self._cell.cvs[cv_id] for cv_id in self._scope.cv_ids)

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
    def V_init(self):
        """Return effective initial voltages for selected cells."""
        return self._cell._selected_population_parameter("V_init", self._population_indices)

    @V_init.setter
    def V_init(self, value) -> None:
        self.set(V_init=value)

    @property
    def V_th(self):
        """Return effective spike thresholds for selected cells."""
        return self._cell._selected_population_parameter("V_th", self._population_indices)

    @V_th.setter
    def V_th(self, value) -> None:
        self.set(V_th=value)

    @property
    def V(self):
        """Return initialized membrane voltage gathered over selected cells."""
        self._cell._raise_if_not_initialized("CellView.V")
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
        return _select_population_value(
            self._cell.spike.value,
            population_indices=self._population_indices,
            population_size=self._cell._population_size,
        )

    def set(self, **parameters) -> "CellView":
        """Set shape-preserving declaration parameters for selected cells.

        Parameters
        ----------
        **parameters
            Supported keys are ``V_init`` and ``V_th``. Values must be
            concrete voltage quantities that broadcast within the selected
            population-by-CV shape. Passing ``None`` for ``V_init`` removes
            selected overrides and restores the root declaration.

        Returns
        -------
        CellView
            This view.
        """
        self._cell._set_selected_population_parameters(self._population_indices, parameters)
        return self

    def place(
        self,
        locset: LocsetExpr | LocsetMask | LocsetBatch,
        *mechanisms,
    ) -> "CellView":
        """Place independent point instances on selected population members.

        Parameters
        ----------
        locset : LocsetExpr, LocsetMask, LocsetBatch, or sequence
            One shared locset, an aligned rectangular batch, or one possibly
            ragged locset per selected population member.
        *mechanisms : Point
            Point declarations. Per-cell locset sequences currently accept
            synapse declarations.

        Returns
        -------
        CellView
            This population view.
        """
        if self._scope.spatially_restricted:
            raise RuntimeError(
                "Spatial CellView.place() is not supported; pass the desired Locset to Cell.place() "
                "or select population members only before place()."
            )
        self._cell._place_selected(self._population_indices, locset, mechanisms)
        return self

    def paint(self, region: RegionExpr, *mechanisms):
        """Reject population-specific density painting in v1."""
        raise NotImplementedError(
            "CellView.paint() does not support population-specific density mechanisms; "
            "use root Cell.paint() to paint the whole population."
        )

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
        return PopulationRuntimeView(runtime, self._population_indices, self._cell._population_size)

    def get_runtime_node(self, layout_id: int) -> PopulationRuntimeView:
        """Return read-only selected-population inspection for a runtime node."""
        runtime = self._cell.get_runtime_node(layout_id)
        layout = self._cell.layouts[int(layout_id)]
        return PopulationRuntimeView(
            runtime,
            self._population_indices,
            self._cell._population_size,
            packed_population_index=layout.population_index,
        )

    def __repr__(self) -> str:
        return (
            f"CellView(name={self.name!r}, population_indices={self._population_indices!r}, "
            f"cv_ids={self._scope.cv_ids!r})"
        )


# Compatibility name retained for code written against the first selection API.
CellSelection = CellView


class Cell(_CellFacade, HHTypedNeuron):
    """Multi-compartment cell with explicit declaration / initialization phases.

    Parameters
    ----------
    morpho : Morphology
        Morphology tree.
    cv_policy : CVPolicy, optional
        Control-volume splitting policy; defaults to :class:`CVPerBranch`.
    V_th : Quantity
        Spike-detection threshold (default ``0. mV``).
    V_init : Quantity or Callable or None
        Initial voltage. ``None`` means "use per-CV resting potential".
    spk_fun : Callable
        Surrogate-gradient spike function.
    solver : str or Callable
        Integrator name (registry lookup) or callable step function.
    subsolver : str or Callable or None
        Integrator for Markov channels and kinetic ions. Together with
        ``substeps=None``, ``None`` selects ``"backward_euler"``.
    substeps : int or None
        Number of subsolver steps per main cell step. Together with
        ``subsolver=None``, ``None`` selects one step.
    ion_channel_update_order : {"family", "integration"}
        Post-voltage ion/channel scheduling. ``"family"`` updates all ions
        before all channels; ``"integration"`` preserves the previous
        IndependentIntegration-grouped scheduling.
    membrane_linearizer : {"point", "generic"}
        Membrane-current linearization strategy. ``"point"`` differentiates
        the point-local current kernel before gathering CV midpoints;
        ``"generic"`` retains whole-CV automatic differentiation.
    name : str, optional
        Cell name.
    """

    __module__ = "braincell"

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
        solver: str | Callable = "staggered",
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

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_th_declaration = V_th
        self._V_init = V_init
        self._V_init_materialized = None
        self._population_parameter_overrides: dict[str, dict[int, object]] = {
            "V_init": {},
            "V_th": {},
        }
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)
        (
            self._subsolver_name,
            self._subsolver_fn,
            self._substeps,
        ) = _resolve_subsolver_schedule(subsolver, substeps)
        self.cache_ion_total_current = bool(cache_ion_total_current)
        self.ion_channel_update_order = _validate_ion_channel_update_order(ion_channel_update_order)
        self._membrane_linearizer = _validate_membrane_linearizer(membrane_linearizer)

        self._discretization_cache: Discretization | None = None
        self._discretization_cache_key: object = None

        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._node_scheduling_cache: dict[tuple[str, int], object] = {}
        self._run_loop_cache: dict[tuple[object, ...], object] = {}

        self._runtime: CellRuntimeState | None = None
        self._runtime_cvs_cache: tuple[RuntimeCVView, ...] | None = None
        self._runtime_nodes_cache: tuple[RuntimeNodeView, ...] | None = None
        self._synapse_store_cache: _SynapseStore | None = None
        self._spike_event_source_cache: _CellSpikeSource | None = None
        self._axial_jax = None
        self._synapse_input_bindings: dict[str, list[tuple[object, object, object]]] = {}
        self._synapse_parameter_overrides: dict[tuple[int, int, str], object] = {}
        self._density_parameter_overrides: dict[tuple[str, str, int, int, str], object] = {}
        self._synapse_spec_origins: dict[int, SynapsePlacement] = {}
        self._connection_store_cache = None
        self._network_owner_ref: weakref.ReferenceType | None = None
        self._recording_specs: dict[str, RecordingSpec] = {}
        self._compiled_recording_cache: dict[tuple, tuple] = {}

        self._initialized = False

        # Eager policy validation via the preview.
        _ = self.cvs

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
        return self._morpho

    @property
    def cv_policy(self) -> CVPolicy:
        return self._discretization_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        self._raise_if_initialized("assign cv_policy")
        if not isinstance(value, CVPolicy):
            raise TypeError(f"cv_policy must be CVPolicy, got {type(value).__name__!s}.")
        self._discretization_policy = value
        self._invalidate_discretization_cache()

    @property
    def paint_rules(self) -> tuple[PaintRule, ...]:
        return self._paint_rules

    @property
    def place_rules(self) -> tuple[PlaceRule, ...]:
        return self._place_rules

    @property
    def V_th(self):
        return self._V_th

    @V_th.setter
    def V_th(self, value) -> None:
        # ``install_cell_runtime`` overwrites V_th with a vectorised
        # version during ``init_state``; that call is permitted because
        # ``_initialized`` is still False at that point. After
        # ``init_state`` completes, the guard rejects further assignment.
        self._raise_if_initialized("assign V_th")
        self._V_th = value
        if hasattr(self, "_population_parameter_overrides"):
            self._population_parameter_overrides["V_th"].clear()

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._raise_if_initialized("assign V_init")
        self._V_init = value
        self._V_init_materialized = None
        if hasattr(self, "_population_parameter_overrides"):
            self._population_parameter_overrides["V_init"].clear()

    @property
    def solver(self):
        return self._solver_fn

    @solver.setter
    def solver(self, value) -> None:
        self._raise_if_initialized("assign solver")
        self._solver_name, self._solver_fn = _resolve_solver(value)

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
        self._place_rules = merge_place_rules(
            self._place_rules,
            (
                normalize_place_rule(
                    locset,
                    mechanisms,
                    population_indices=normalized_indices,
                    aligned=aligned,
                ),
            ),
        )
        self._invalidate_discretization_cache()

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
        per_mechanism_rows = tuple(_split_synapse_spec_rows(mechanism, lengths=lengths) for mechanism in mechanisms)
        incoming = []
        for row, (population_index, locset) in enumerate(zip(population_indices, locsets)):
            row_mechanisms = tuple(rows[row] for rows in per_mechanism_rows)
            for source, materialized in zip(mechanisms, row_mechanisms):
                self._synapse_spec_origins[id(materialized)] = source
            incoming.append(
                normalize_place_rule(
                    locset,
                    row_mechanisms,
                    population_indices=(int(population_index),),
                    aligned=False,
                )
            )
        self._place_rules = merge_place_rules(self._place_rules, tuple(incoming))
        self._invalidate_discretization_cache()

    def _set_selected_population_parameters(self, population_indices, parameters) -> None:
        """Store declaration-time overrides without changing population shape."""
        self._raise_if_initialized("set CellView parameters")
        unknown = set(parameters) - {"V_init", "V_th"}
        if unknown:
            raise KeyError(f"CellView.set() does not support parameters {sorted(unknown)!r}.")
        indices = tuple(int(index) for index in population_indices)
        for name, value in parameters.items():
            overrides = self._population_parameter_overrides[name]
            if value is None:
                if name != "V_init":
                    raise TypeError("CellView V_th must be a voltage quantity, not None.")
                for index in indices:
                    overrides.pop(index, None)
                self._V_init_materialized = None
                continue
            normalized = _normalize_selected_voltage_parameter(
                value,
                count=len(indices),
                n_cv=self.n_cv,
                name=name,
            )
            for index, item in zip(indices, normalized):
                overrides[index] = item
            if name == "V_init":
                self._V_init_materialized = None

    def _materialize_population_parameter(self, name: str):
        if name == "V_th":
            value = bridge.fill_like(self.varshape, self._V_th)
        elif name == "V_init":
            initializer = self._V_init
            if initializer is None:
                initializer = bridge.cv_value_vector(self, attr_name="v")
            elif not callable(initializer):
                initializer = bridge.fill_like(self.varshape, initializer)
            value = braintools.init.param(initializer, self.varshape)
        else:  # pragma: no cover - internal invariant
            raise KeyError(name)
        return _apply_population_parameter_overrides(
            value,
            overrides=self._population_parameter_overrides[name],
            name=name,
        )

    def _selected_population_parameter(self, name: str, population_indices):
        if name == "V_th" and self._initialized:
            values = self._V_th
        elif name == "V_init" and self._V_init_materialized is not None:
            values = self._V_init_materialized
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

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        """Paint mechanisms onto ``region``. Returns ``self`` for chaining."""
        self._raise_if_initialized("paint()")
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._invalidate_discretization_cache()
        return self

    def place(
        self,
        locset: LocsetExpr | LocsetMask | LocsetBatch,
        *mechanisms,
    ) -> "Cell":
        """Place point mechanisms at shared or per-cell locations.

        Parameters
        ----------
        locset : LocsetExpr, LocsetMask, LocsetBatch, or sequence
            One shared locset, an aligned rectangular batch, or one possibly
            ragged locset per population member.
        *mechanisms : Point
            Point-mechanism declarations placed independently at each row.

        Returns
        -------
        Cell
            This Cell for chaining.
        """
        self._raise_if_initialized("place()")
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
        self._place_rules = merge_place_rules(
            self._place_rules,
            (
                normalize_place_rule(
                    locset,
                    mechanisms,
                    population_indices=population_indices,
                    aligned=aligned,
                ),
            ),
        )
        self._invalidate_discretization_cache()
        return self

    def bind_synapse_input(self, synapse: str, source, *, weight=1.0, transform=None) -> "Cell":
        """Deprecated compatibility adapter for per-boundary event payloads.

        Parameters
        ----------
        synapse : str
            Synapse instance name, matching ``braincell.mech.SynapseSpec(name=...)``
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
        warnings.warn(
            "Cell.bind_synapse_input() is deprecated; create an EventSource and use braincell.connect().",
            DeprecationWarning,
            stacklevel=2,
        )
        key = str(synapse)
        self._synapse_input_bindings.setdefault(key, []).append((source, weight, transform))
        return self

    # ------------------------------------------------------------------
    # Static discretization (valid in both phases)

    def _invalidate_discretization_cache(self) -> None:
        self._discretization_cache = None
        self._discretization_cache_key = None
        self._synapse_store_cache = None
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        self._run_loop_cache.clear()

    def _discretization_key(self) -> tuple[object, ...]:
        return (
            id(self._morpho),
            self._discretization_policy,
            self._paint_rules,
            self._place_rules,
        )

    @property
    def _discretization(self) -> Discretization:
        key = self._discretization_key()
        if self._discretization_cache is not None and self._discretization_cache_key == key:
            return self._discretization_cache

        discretization = build_discretization(
            self._morpho,
            policy=self._discretization_policy,
            paint_rules=self._paint_rules,
            place_rules=self._place_rules,
        )
        self._discretization_cache = discretization
        self._discretization_cache_key = key
        return discretization

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self) -> tuple[CV, ...]:
        return self._discretization.cvs

    @property
    def cv_midpoints(self) -> LocsetMask:
        """Return one resolved continuous midpoint for every control volume."""
        cvs = self.cvs
        return LocsetMask.from_columns(
            [cv.branch_id for cv in cvs],
            [(cv.prox + cv.dist) * 0.5 for cv in cvs],
        )

    @property
    def cv_tree(self) -> CVTree:
        return self._discretization.cv_tree

    @property
    def node_tree(self) -> NodeTree:
        return self._discretization.node_tree

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
    def connections(self):
        """Return a unified view over all direct event-routing rows."""
        from braincell.connection import ConnectionView

        return ConnectionView(self._get_connection_store())

    @property
    def event_outputs(self) -> EventOutputCollection:
        """Return named live event-output ports for this cell population."""
        return EventOutputCollection(self)

    def _get_spike_event_source(self) -> _CellSpikeSource:
        if self._spike_event_source_cache is None:
            self._spike_event_source_cache = _CellSpikeSource(self)
        return self._spike_event_source_cache

    def _get_synapse_store(self) -> _SynapseStore:
        """Return the private logical synapse store for the current declaration."""
        if self._synapse_store_cache is None:
            self._synapse_store_cache = _SynapseStore(self)
        return self._synapse_store_cache

    def _get_connection_store(self):
        """Return the private Cell-owned routing-row store."""
        if self._connection_store_cache is None:
            from braincell.connection import _ConnectionStore

            self._connection_store_cache = _ConnectionStore(self)
        return self._connection_store_cache

    def _validate_synapse_names(self, mechanisms) -> None:
        """Reject one logical group name being reused across model types."""
        declared = {}
        for rule in self._place_rules:
            for mechanism in rule.mechanisms:
                if isinstance(mechanism, SynapsePlacement):
                    declared.setdefault(mechanism.instance_name, mechanism.synapse_type)
        for mechanism in mechanisms:
            if not isinstance(mechanism, SynapsePlacement):
                continue
            previous = declared.setdefault(mechanism.instance_name, mechanism.synapse_type)
            if previous != mechanism.synapse_type:
                raise ValueError(
                    f"Synapses with the same name {mechanism.instance_name!r} cannot use different synapse types "
                    f"({previous!r} and {mechanism.synapse_type!r})."
                )

    @property
    def recordings(self) -> Mapping[str, RecordingSpec]:
        """Return a read-only snapshot of Cell recording declarations."""
        return MappingProxyType(dict(self._recording_specs))

    def _add_recording(self, spec: RecordingSpec) -> RecordingSpec:
        self._raise_if_initialized("add a recording")
        if spec.name in self._recording_specs:
            raise ValueError(f"Cell already has a recording named {spec.name!r}.")
        self._recording_specs[spec.name] = spec
        self._compiled_recording_cache.clear()
        self._run_loop_cache.clear()
        return spec

    def _compiled_recordings(self, dt) -> tuple:
        self._raise_if_not_initialized("compile recordings")
        dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
        key = (dt_ms, tuple(self._recording_specs))
        cached = self._compiled_recording_cache.get(key)
        if cached is None:
            cached = tuple(compile_recording(self, spec, dt=dt) for spec in self._recording_specs.values())
            self._compiled_recording_cache[key] = cached
        return cached

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

        Raises
        ------
        RuntimeError
            If the cell is already initialized. Call :meth:`reset` first.
        """
        self._raise_if_network_owned("init_state()")
        self._raise_if_initialized("init_state()")

        morpho = clone_morpho(self._morpho)
        self._morpho = morpho
        self._invalidate_discretization_cache()
        _ = self._discretization
        self._runtime = CellRuntimeState.from_cell(self)

        # Save scalar V_th declaration before the vector overwrite below.
        self._V_th_declaration = self._V_th

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
        self.C = bridge.cv_value_vector(self, attr_name="cm")
        self._V_th = self._materialize_population_parameter("V_th")

        v_value = self._materialize_population_parameter("V_init")
        self._V_init_materialized = v_value
        v_value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        # A Cell is spatial: every hidden state's trailing axis enumerates
        # compartments (V) or points (mechanism variables), so all of them
        # are group states. Channel / ion / synapse code is shared with
        # SingleCompartment, hence the scoped factory rather than a
        # per-call-site class choice.
        self.V = DiffEqGroupState(v_value)
        self.spike = brainstate.ShortTermState(_zero_spike_like(self.V.value))
        self._event_previous_V = brainstate.ShortTermState(self.V.value)
        self._current_time_state.value = 0.0 * u.ms

        point_V = self._cv_to_point_unchecked(self.V.value)
        with state_grouping(True):
            for path, channel in self._runtime_objects_unchecked(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, point_V)
                channel.init_state(*args, batch_size=batch_size)
            # Mechanism init hooks allocate state; reset hooks materialize the
            # model-defined initial values from V_init and current parameters.
            for path, channel in self._runtime_objects_unchecked(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, point_V)
                channel.reset_state(*args, batch_size=batch_size)

        # Dense CV axial operators are only needed by derivative-based voltage
        # solvers. The default DHS/staggered path builds its own static source,
        # so defer this matrix until ``_get_axial_operator()`` is actually used.
        self._runtime.axial_operator_np = None
        self._runtime.axial_operator_cache = None
        self._axial_jax = None
        self._initialized = True
        self._runtime_cvs_cache = self._build_runtime_cv_views()
        self._runtime_nodes_cache = self._build_runtime_node_views()

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
        run again.
        """
        self._raise_if_network_owned("reset()")
        self._raise_if_not_initialized("reset()")

        self.connections.clear_runtime()

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
        self._current_time_state.value = 0.0 * u.ms

        self._runtime = None
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        self._axial_jax = None
        self._node_scheduling_cache.clear()
        self._run_loop_cache.clear()
        self._compiled_recording_cache.clear()

        self._morpho = self._declaration_morpho
        self._invalidate_discretization_cache()
        self._V_init_materialized = None

        self._initialized = False

    # ------------------------------------------------------------------
    # Static topology + runtime inspection views

    @property
    def runtime(self) -> CellRuntimeState:
        self._raise_if_not_initialized("runtime")
        return self._runtime

    @property
    def n_point(self) -> int:
        self._raise_if_not_initialized("n_point")
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
    def _view_population_indices(self) -> tuple[int, ...]:
        return tuple(range(self._population_size))

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

    def vis_topology(
        self,
        *,
        level: str = "node",
        preset: str = "dendrotweaks",
        layout: str | None = None,
        layout_scale: float = 1.0,
        region: RegionExpr | RegionMask | None = None,
        locset: LocsetExpr | LocsetMask | None = None,
        coverage_mode: str = "fraction",
        highlight_color: str = "#ef4444",
        value=None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        norm=None,
        value_label: str | None = None,
        show_colorbar: bool = True,
        node_color: str | None = None,
        edge_color: str | None = None,
        root_color: str | None = None,
        ax=None,
        show: bool = True,
    ) -> object:
        """Dispatch to one of the topology visualization levels.

        This is a thin wrapper around :meth:`vis_node`,
        :meth:`vis_cv`, and :meth:`vis_branch`. It lets callers select
        a topology level dynamically while keeping one stable entry
        point.

        Parameters
        ----------
        level : {"node", "cv", "branch"}, optional
            Topology abstraction level to render.
        preset : str, optional
            Name of the built-in topology preset.
        layout : str or None, optional
            Explicit layout algorithm override.
        layout_scale : float, optional
            Global spacing multiplier for the resolved layout.
        region : RegionExpr or RegionMask or None, optional
            Region selection used for highlighting / coverage.
        locset : LocsetExpr or LocsetMask or None, optional
            Discrete location selection. Supported by ``level="node"``
            and ``level="cv"`` only.
        coverage_mode : {"fraction", "any", "all"}, optional
            Coverage display rule for region-based highlighting.
        highlight_color : str, optional
            Highlight colour used for selected nodes.
        value : object, optional
            Runtime value selector. Supported by ``level="node"`` and
            ``level="cv"`` only.
        cmap : str or None, optional
            Matplotlib colormap name used in value mode.
        vmin, vmax : float or None, optional
            Explicit lower and upper bounds for the value colormap.
        norm : matplotlib.colors.Normalize or None, optional
            Explicit normalization object for value mode.
        value_label : str or None, optional
            Colorbar label override for value mode.
        show_colorbar : bool, optional
            If ``True`` (default), draw a colorbar in value mode.
        node_color, edge_color, root_color : str or None, optional
            Base style colour overrides.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes. When ``None``, a fresh figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), call ``matplotlib.pyplot.show()``
            after rendering.

        Returns
        -------
        object
            The rendered Matplotlib axes.

        Raises
        ------
        ValueError
            If ``level`` is invalid, or if branch-level rendering is
            given parameters that are only supported by node/CV value
            modes.

        See Also
        --------
        vis_node
            Render the full runtime node topology.
        vis_cv
            Render one node per control volume.
        vis_branch
            Render one node per morphology branch.

        Examples
        --------
        Render node topology:

        >>> ax = cell.vis_topology(level="node", show=False)  # doctest: +SKIP

        Render CV topology:

        >>> ax = cell.vis_topology(level="cv", show=False)  # doctest: +SKIP

        Render branch topology:

        >>> ax = cell.vis_topology(level="branch", show=False)  # doctest: +SKIP
        """
        if level == "node":
            return self.vis_node(
                preset=preset,
                layout=layout,
                layout_scale=layout_scale,
                region=region,
                locset=locset,
                coverage_mode=coverage_mode,
                highlight_color=highlight_color,
                value=value,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                value_label=value_label,
                show_colorbar=show_colorbar,
                node_color=node_color,
                edge_color=edge_color,
                root_color=root_color,
                ax=ax,
                show=show,
            )
        if level == "cv":
            return self.vis_cv(
                preset=preset,
                layout=layout,
                layout_scale=layout_scale,
                region=region,
                locset=locset,
                coverage_mode=coverage_mode,
                highlight_color=highlight_color,
                value=value,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                value_label=value_label,
                show_colorbar=show_colorbar,
                node_color=node_color,
                edge_color=edge_color,
                root_color=root_color,
                ax=ax,
                show=show,
            )
        if level == "branch":
            if locset is not None:
                raise ValueError("Cell.vis_topology(level='branch', ...) does not support locset.")
            if value is not None:
                raise ValueError("Cell.vis_topology(level='branch', ...) does not support value.")
            if cmap is not None or vmin is not None or vmax is not None or norm is not None or value_label is not None:
                raise ValueError("Cell.vis_topology(level='branch', ...) does not support value-colormap parameters.")
            if show_colorbar is not True:
                raise ValueError("Cell.vis_topology(level='branch', ...) does not support show_colorbar.")
            return self.vis_branch(
                preset=preset,
                layout=layout,
                layout_scale=layout_scale,
                region=region,
                coverage_mode=coverage_mode,
                highlight_color=highlight_color,
                node_color=node_color,
                edge_color=edge_color,
                root_color=root_color,
                ax=ax,
                show=show,
            )
        raise ValueError("Cell.vis_topology(...) level must be one of {'node', 'cv', 'branch'}.")

    def vis_node(
        self,
        *,
        preset: str = "dendrotweaks",
        layout: str | None = None,
        layout_scale: float = 1.0,
        region: RegionExpr | RegionMask | None = None,
        locset: LocsetExpr | LocsetMask | None = None,
        coverage_mode: str = "fraction",
        highlight_color: str = "#ef4444",
        value=None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        norm=None,
        value_label: str | None = None,
        show_colorbar: bool = True,
        node_color: str | None = None,
        edge_color: str | None = None,
        root_color: str | None = None,
        ax=None,
        show: bool = True,
    ) -> object:
        """Visualize the runtime node tree with cell-aware inputs.

        ``Cell.vis_node(...)`` is the high-level node-tree entry point.
        It resolves ``region`` / ``locset`` selections against the cell's
        morphology and CVs, maps those selections to runtime midpoint
        points, and can also colour points by runtime state such as
        voltage or mechanism parameters.

        Parameters
        ----------
        preset : str, optional
            Name of the built-in point-topology preset.
        layout : str or None, optional
            Explicit layout algorithm override.
        layout_scale : float, optional
            Global spacing multiplier for the resolved layout.
        region : RegionExpr or RegionMask or None, optional
            Continuous morphology selection to highlight. In v1,
            selected CVs are mapped to their midpoint point ids only.
        locset : LocsetExpr or LocsetMask or None, optional
            Discrete morphology locations to highlight. In v1, each
            location is mapped to the CV midpoint point that owns the
            location.
        coverage_mode : {"fraction", "any", "all"}, optional
            Coverage display rule for ``region``. Locset-backed
            highlights are always treated as full intensity.
        highlight_color : str, optional
            Colour used for highlighted points.
        value : object, optional
            Point colouring source. Supported forms are:

            - point-space array of length ``n_point``
            - CV-space array of length ``n_cv`` (scattered to point space)
            - ``"V"`` or ``"voltage"``
            - ``("ion", ion_name, field)``
            - ``("channel", class_name, field)``
            - ``("layout_id", layout_id, field)``

            ``value`` is mutually exclusive with ``region`` / ``locset``
            in v1.
        cmap : str or None, optional
            Matplotlib colormap name used in value mode.
        vmin, vmax : float or None, optional
            Explicit lower and upper bounds for the value colormap.
        norm : matplotlib.colors.Normalize or None, optional
            Explicit normalization object for value mode.
        value_label : str or None, optional
            Colorbar label override. When ``None``, certain named value
            selectors infer a label automatically.
        show_colorbar : bool, optional
            If ``True`` (default), draw a colorbar in value mode.
        node_color, edge_color, root_color : str or None, optional
            Base style colour overrides.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes. When ``None``, a fresh figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), call ``matplotlib.pyplot.show()``
            after rendering.

        Returns
        -------
        object
            The rendered Matplotlib axes.

        Raises
        ------
        RuntimeError
            If the cell is not initialized.
        ValueError
            If ``value`` is combined with ``region`` or ``locset``, or
            if a supplied value source cannot be mapped into point
            space.

        Notes
        -----
        Highlight mode and value mode are mutually exclusive in v1.
        Region and locset mappings use CV midpoint semantics to stay
        consistent with the current runtime lowering model.

        Examples
        --------
        Highlight a region:

        >>> ax = cell.vis_node(region=some_region, show=False)  # doctest: +SKIP

        Colour nodes by voltage:

        >>> ax = cell.vis_node(value="V", cmap="viridis", show=False)  # doctest: +SKIP

        Colour nodes by a channel parameter:

        >>> ax = cell.vis_node(value=("channel", "IL", "g_max"), show=False)  # doctest: +SKIP
        """
        self._raise_if_not_initialized("vis_node()")
        if value is not None and (region is not None or locset is not None):
            raise ValueError("Cell.vis_node(...) does not support value together with region/locset highlighting.")

        highlight_point_ids = None
        highlight_fractions = None
        values = None
        resolved_value_label = value_label

        if region is not None or locset is not None:
            highlight_fractions = self._node_highlight_fractions(region=region, locset=locset)
        elif value is not None:
            values, inferred_label = self._resolve_vis_node_values(value)
            if resolved_value_label is None:
                resolved_value_label = inferred_label

        from braincell.vis.point_topology import plot_point_topology

        rendered_ax = plot_point_topology(
            self.node_tree,
            preset=preset,
            layout=layout,
            layout_scale=layout_scale,
            highlight_point_ids=highlight_point_ids,
            highlight_fractions=highlight_fractions,
            coverage_mode=coverage_mode,
            highlight_color=highlight_color,
            values=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            value_label=resolved_value_label,
            show_colorbar=show_colorbar,
            node_color=node_color,
            edge_color=edge_color,
            root_color=root_color,
            ax=ax,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return rendered_ax

    def vis_cv(
        self,
        *,
        preset: str = "dendrotweaks",
        layout: str | None = None,
        layout_scale: float = 1.0,
        region: RegionExpr | RegionMask | None = None,
        locset: LocsetExpr | LocsetMask | None = None,
        coverage_mode: str = "fraction",
        highlight_color: str = "#ef4444",
        value=None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        norm=None,
        value_label: str | None = None,
        show_colorbar: bool = True,
        node_color: str | None = None,
        edge_color: str | None = None,
        root_color: str | None = None,
        ax=None,
        show: bool = True,
    ) -> object:
        """Visualize the cell at the control-volume topology level.

        ``Cell.vis_cv(...)`` renders one node per control volume. It
        shares the same high-level selector model as
        :meth:`vis_node`, but collapses the view to CV granularity.

        Parameters
        ----------
        preset : str, optional
            Name of the built-in topology preset.
        layout : str or None, optional
            Explicit layout algorithm override.
        layout_scale : float, optional
            Global spacing multiplier for the resolved layout.
        region : RegionExpr or RegionMask or None, optional
            Region used to compute per-CV coverage.
        locset : LocsetExpr or LocsetMask or None, optional
            Discrete morphology locations to highlight. Each location
            is mapped to its owning CV.
        coverage_mode : {"fraction", "any", "all"}, optional
            Coverage display rule. ``"fraction"`` blends by overlap
            fraction, ``"any"`` highlights any overlap fully, and
            ``"all"`` only highlights fully covered CVs.
        highlight_color : str, optional
            Highlight colour used in coverage mode.
        value : object, optional
            CV colouring source. Supports the same high-level selector
            forms as :meth:`vis_node`.
        cmap : str or None, optional
            Matplotlib colormap name used in value mode.
        vmin, vmax : float or None, optional
            Explicit lower and upper bounds for the value colormap.
        norm : matplotlib.colors.Normalize or None, optional
            Explicit normalization object for value mode.
        value_label : str or None, optional
            Colorbar label override.
        show_colorbar : bool, optional
            If ``True`` (default), draw a colorbar in value mode.
        node_color, edge_color, root_color : str or None, optional
            Base style colour overrides.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes. When ``None``, a fresh figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), call ``matplotlib.pyplot.show()``
            after rendering.

        Returns
        -------
        object
            The rendered Matplotlib axes.

        Raises
        ------
        ValueError
            If the cell has no unique root CV, or if ``value`` is
            combined with ``region`` / ``locset``.

        Notes
        -----
        Each CV is represented by one node. ``region`` / ``locset``
        and ``value`` remain mutually exclusive in v1.

        See Also
        --------
        vis_node
            Render the lower-level runtime node topology.
        vis_branch
            Render the higher-level morphology branch topology.
        vis_topology
            Thin dispatcher over the available topology levels.

        Examples
        --------
        Highlight a region at CV level:

        >>> ax = cell.vis_cv(region=some_region, show=False)  # doctest: +SKIP

        Colour CVs by voltage:

        >>> ax = cell.vis_cv(value="V", cmap="viridis", show=False)  # doctest: +SKIP
        """
        cvs = self.cvs
        root_ids = [cv.id for cv in cvs if cv.parent_cv is None]
        if len(root_ids) != 1:
            raise ValueError(f"Cell.vis_cv(...) expects exactly one root CV, got {root_ids!r}.")
        if value is not None and (region is not None or locset is not None):
            raise ValueError("Cell.vis_cv(...) does not support value together with region/locset highlighting.")

        coverage_fractions = None
        values = None
        resolved_value_label = value_label
        if region is not None or locset is not None:
            coverage_fractions = self._cv_highlight_fractions(region=region, locset=locset)
        elif value is not None:
            values, inferred_label = self._resolve_vis_cv_values(value)
            if resolved_value_label is None:
                resolved_value_label = inferred_label

        from braincell.vis.point_topology import _plot_discrete_topology_graph

        rendered_ax = _plot_discrete_topology_graph(
            node_ids=tuple(cv.id for cv in cvs),
            edges=tuple((int(cv.parent_cv), int(cv.id)) for cv in cvs if cv.parent_cv is not None),
            root_id=int(root_ids[0]),
            preset=preset,
            layout=layout,
            layout_scale=layout_scale,
            highlight_fractions=coverage_fractions,
            coverage_mode=coverage_mode,
            highlight_color=highlight_color,
            values=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            value_label=resolved_value_label,
            show_colorbar=show_colorbar,
            node_color=node_color,
            edge_color=edge_color,
            root_color=root_color,
            ax=ax,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return rendered_ax

    def vis_branch(
        self,
        *,
        preset: str = "dendrotweaks",
        layout: str | None = None,
        layout_scale: float = 1.0,
        region: RegionExpr | RegionMask | None = None,
        coverage_mode: str = "fraction",
        highlight_color: str = "#ef4444",
        node_color: str | None = None,
        edge_color: str | None = None,
        root_color: str | None = None,
        ax=None,
        show: bool = True,
    ) -> object:
        """Visualize the cell at the branch topology level.

        ``Cell.vis_branch(...)`` renders one node per morphology
        branch. This view is intended for topology and region coverage
        inspection rather than runtime-value inspection.

        Parameters
        ----------
        preset : str, optional
            Name of the built-in topology preset.
        layout : str or None, optional
            Explicit layout algorithm override.
        layout_scale : float, optional
            Global spacing multiplier for the resolved layout.
        region : RegionExpr or RegionMask or None, optional
            Region used to compute per-branch coverage.
        coverage_mode : {"fraction", "any", "all"}, optional
            Coverage display rule. ``"fraction"`` blends by overlap
            fraction, ``"any"`` highlights any overlap fully, and
            ``"all"`` only highlights fully covered branches.
        highlight_color : str, optional
            Highlight colour used in coverage mode.
        node_color, edge_color, root_color : str or None, optional
            Base style colour overrides.
        ax : matplotlib.axes.Axes or None, optional
            Destination axes. When ``None``, a fresh figure and axes
            are created.
        show : bool, optional
            If ``True`` (default), call ``matplotlib.pyplot.show()``
            after rendering.

        Returns
        -------
        object
            The rendered Matplotlib axes.

        Raises
        ------
        ValueError
            If an invalid coverage configuration is supplied by the
            low-level renderer.

        Notes
        -----
        This view is topology-only and does not support value-based
        colormaps. Each branch is represented by one node.

        See Also
        --------
        vis_node
            Render the full runtime node topology.
        vis_cv
            Render one node per control volume.
        vis_topology
            Thin dispatcher over the available topology levels.

        Examples
        --------
        Render partial branch coverage:

        >>> ax = cell.vis_branch(region=some_region, show=False)  # doctest: +SKIP
        """
        morpho = self.morpho
        coverage_fractions = None if region is None else self._branch_coverage_fractions(region)

        from braincell.vis.point_topology import _plot_discrete_topology_graph

        rendered_ax = _plot_discrete_topology_graph(
            node_ids=tuple(branch.index for branch in morpho.branches),
            edges=tuple((edge.parent.index, edge.child.index) for edge in morpho.edges),
            root_id=int(morpho.root.index),
            preset=preset,
            layout=layout,
            layout_scale=layout_scale,
            highlight_fractions=coverage_fractions,
            coverage_mode=coverage_mode,
            highlight_color=highlight_color,
            node_color=node_color,
            edge_color=edge_color,
            root_color=root_color,
            ax=ax,
        )
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return rendered_ax

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
        return bridge.cv_to_point(cv_values, self._runtime)

    def _cv_to_point_unchecked(self, cv_values):
        return bridge.cv_to_point(cv_values, self._runtime)

    # Internal aliases kept while older inspection helpers are still in use.
    def _discretization_to_point(self, cv_values):
        return self._cv_to_point(cv_values)

    def _discretization_to_point_unchecked(self, cv_values):
        return self._cv_to_point_unchecked(cv_values)

    def _point_to_cv(self, point_values):
        self._raise_if_not_initialized("_point_to_cv()")
        return bridge.point_to_cv(point_values, self._runtime)

    def _resolve_vis_node_highlight_ids(
        self,
        *,
        region: RegionExpr | RegionMask | None,
        locset: LocsetExpr | LocsetMask | None,
    ) -> set[int]:
        point_ids: set[int] = set()
        if region is not None:
            point_ids.update(self._region_to_vis_node_ids(region))
        if locset is not None:
            point_ids.update(self._locset_to_vis_node_ids(locset))
        return point_ids

    def _node_highlight_fractions(
        self,
        *,
        region: RegionExpr | RegionMask | None,
        locset: LocsetExpr | LocsetMask | None,
    ) -> dict[int, float]:
        fractions: dict[int, float] = {}
        node_tree = self.node_tree
        if region is not None:
            for cv_id, fraction in self._cv_coverage_fractions(region).items():
                point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
                fractions[point_id] = max(fractions.get(point_id, 0.0), float(fraction))
        if locset is not None:
            for cv_id in self._resolve_vis_locset_cv_ids(locset):
                point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
                fractions[point_id] = max(fractions.get(point_id, 0.0), 1.0)
        return fractions

    def _cv_highlight_fractions(
        self,
        *,
        region: RegionExpr | RegionMask | None,
        locset: LocsetExpr | LocsetMask | None,
    ) -> dict[int, float]:
        fractions: dict[int, float] = {}
        if region is not None:
            fractions.update(self._cv_coverage_fractions(region))
        if locset is not None:
            for cv_id in self._resolve_vis_locset_cv_ids(locset):
                fractions[int(cv_id)] = max(fractions.get(int(cv_id), 0.0), 1.0)
        return fractions

    def _region_to_vis_node_ids(self, region: RegionExpr | RegionMask) -> set[int]:
        branch_intervals = self._resolve_vis_region_intervals(region)

        point_ids: set[int] = set()
        node_tree = self.node_tree
        for cv in self.cvs:
            intervals = branch_intervals.get(int(cv.branch_id))
            if not intervals:
                continue
            midpoint = 0.5 * (float(cv.prox) + float(cv.dist))
            for prox, dist in intervals:
                lo, hi = (prox, dist) if prox <= dist else (dist, prox)
                if lo <= midpoint <= hi:
                    point_ids.add(int(node_tree.cv_to_mid_node_id[cv.id]))
                    break
        return point_ids

    def _resolve_vis_region_intervals(
        self,
        region: RegionExpr | RegionMask,
    ) -> dict[int, tuple[tuple[float, float], ...]]:
        if isinstance(region, RegionExpr):
            mask = region.evaluate(self.morpho)
        elif isinstance(region, RegionMask):
            mask = region
        else:
            raise TypeError(
                f"Cell region visualization expects RegionExpr or RegionMask, got {type(region).__name__!s}."
            )
        normalized = normalize_region_intervals(mask.intervals)
        grouped: dict[int, list[tuple[float, float]]] = {}
        for branch_id, prox, dist in normalized:
            grouped.setdefault(int(branch_id), []).append((float(prox), float(dist)))
        return {branch_id: tuple(intervals) for branch_id, intervals in grouped.items()}

    def _cv_coverage_fractions(self, region: RegionExpr | RegionMask) -> dict[int, float]:
        branch_intervals = self._resolve_vis_region_intervals(region)
        fractions: dict[int, float] = {}
        for cv in self.cvs:
            intervals = branch_intervals.get(int(cv.branch_id), ())
            total = max(float(cv.dist) - float(cv.prox), 1e-12)
            overlap = 0.0
            for left, right in intervals:
                start = max(float(cv.prox), float(left))
                end = min(float(cv.dist), float(right))
                if end - start <= 1e-9:
                    continue
                overlap += end - start
            fractions[int(cv.id)] = float(np.clip(overlap / total, 0.0, 1.0))
        return fractions

    def _branch_coverage_fractions(self, region: RegionExpr | RegionMask) -> dict[int, float]:
        branch_intervals = self._resolve_vis_region_intervals(region)
        fractions: dict[int, float] = {}
        for branch in self.morpho.branches:
            intervals = branch_intervals.get(int(branch.index), ())
            covered = sum(max(0.0, float(right) - float(left)) for left, right in intervals)
            fractions[int(branch.index)] = float(np.clip(covered, 0.0, 1.0))
        return fractions

    def _resolve_vis_locset_cv_ids(self, locset: LocsetExpr | LocsetMask) -> set[int]:
        if isinstance(locset, LocsetExpr):
            mask = locset.evaluate(self.morpho)
        elif isinstance(locset, LocsetMask):
            mask = locset
        else:
            raise TypeError(f"Cell visualization expects LocsetExpr or LocsetMask, got {type(locset).__name__!s}.")

        grouped: dict[int, list[int]] = {}
        for cv in self.cvs:
            grouped.setdefault(int(cv.branch_id), []).append(int(cv.id))
        cv_ids_by_branch = {branch_id: tuple(ids) for branch_id, ids in grouped.items()}

        cv_ids: set[int] = set()
        for branch_id, x in mask.points:
            ids = cv_ids_by_branch.get(int(branch_id))
            if not ids:
                continue
            cv_id = _locate_branch_cv_by_x(ids, self.cvs, x=float(x), epsilon=_EPS_PARAM)
            cv_ids.add(int(cv_id))
        return cv_ids

    def _locset_to_vis_node_ids(self, locset: LocsetExpr | LocsetMask) -> set[int]:
        if isinstance(locset, LocsetExpr):
            mask = locset.evaluate(self.morpho)
        elif isinstance(locset, LocsetMask):
            mask = locset
        else:
            raise TypeError(f"Cell visualization expects LocsetExpr or LocsetMask, got {type(locset).__name__!s}.")
        point_ids: set[int] = set()
        node_tree = self.node_tree
        for branch_id, x in mask.points:
            point_ids.add(
                int(
                    locate_node_on_branch(
                        node_tree,
                        cvs=self.cvs,
                        branch_id=int(branch_id),
                        x=float(x),
                    )
                )
            )
        return point_ids

    def _single_population_view(self, values, *, caller: str, field: str = "value"):
        """Reduce a ``pop_size + (n,)`` field to the single-member ``(n,)`` view.

        Inspection and visualization both answer questions about *one*
        morphology, so the leading population axes have to be singleton
        before the spatial mapping helpers can interpret the trailing axis.
        Collapsing them here keeps a default ``pop_size=1`` cell behaving
        exactly as a rank-0 population used to.

        Parameters
        ----------
        values : array-like or Quantity
            Field values shaped ``pop_size + (n,)``. Scalars and 1-D values
            pass through untouched.
        caller : str
            Entry point description, used only in the error message.
        field : str, default 'value'
            Field name, used only in the error message. The default suits
            the coercers, which are handed whatever the caller passed as
            ``value=``; named-field callers pass the real name.

        Returns
        -------
        array-like or Quantity
            ``values`` with every leading singleton axis removed.

        Raises
        ------
        ValueError
            If any leading population axis holds more than one member, since
            there is then no single morphology to answer for.
        """
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) < 2:
            return values
        leading = shape[:-1]
        if any(int(dim) != 1 for dim in leading):
            raise ValueError(
                f"{caller} addresses a single morphology, but field {field!r} has "
                f"population shape {tuple(int(d) for d in leading)!r} from "
                f"pop_size={tuple(int(d) for d in self.pop_size)!r}. Index the "
                f"population axis first."
            )
        return values[(0,) * len(leading)]

    def _vis_cv_voltage(self):
        """Return ``V`` as a plain ``(n_cv,)`` CV vector for visualization."""
        return self._single_population_view(self.V.value, field="V", caller="Cell.vis_cv(...)")

    def _resolve_vis_node_values(self, value) -> tuple[object, str | None]:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"v", "voltage"}:
                return self._cv_to_node_values(self._vis_cv_voltage()), "V"
            raise ValueError(f"Unsupported Cell.vis_node value string {value!r}.")

        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
            mode = value[0]
            if mode == "ion":
                ion_name, field = str(value[1]), str(value[2])
                ion = self.get_ion(ion_name)
                if not hasattr(ion, field):
                    raise AttributeError(f"Ion {ion_name!r} has no field {field!r}.")
                return self._coerce_named_vis_node_values_object(getattr(ion, field)), f"{ion_name}.{field}"
            if mode == "channel":
                class_name, field = str(value[1]), str(value[2])
                layout = self._resolve_unique_layout_by_kind(f"channel:{class_name}")
                return self._layout_field_to_point_values(layout.id, field), f"{class_name}.{field}"
            if mode == "layout_id":
                layout_id, field = int(value[1]), str(value[2])
                return self._layout_field_to_point_values(layout_id, field), f"layout_{layout_id}.{field}"
            raise ValueError(f"Unsupported Cell.vis_node value tuple selector {mode!r}.")

        return self._coerce_vis_node_values_object(value), None

    def _resolve_vis_cv_values(self, value) -> tuple[object, str | None]:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"v", "voltage"}:
                return self._vis_cv_voltage(), "V"
            raise ValueError(f"Unsupported Cell.vis_cv value string {value!r}.")

        if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
            mode = value[0]
            if mode == "ion":
                ion_name, field = str(value[1]), str(value[2])
                ion = self.get_ion(ion_name)
                if not hasattr(ion, field):
                    raise AttributeError(f"Ion {ion_name!r} has no field {field!r}.")
                return self._coerce_named_vis_cv_values_object(getattr(ion, field)), f"{ion_name}.{field}"
            if mode == "channel":
                class_name, field = str(value[1]), str(value[2])
                layout = self._resolve_unique_layout_by_kind(f"channel:{class_name}")
                return self._layout_field_to_cv_values(layout.id, field), f"{class_name}.{field}"
            if mode == "layout_id":
                layout_id, field = int(value[1]), str(value[2])
                return self._layout_field_to_cv_values(layout_id, field), f"layout_{layout_id}.{field}"
            raise ValueError(f"Unsupported Cell.vis_cv value tuple selector {mode!r}.")

        return self._coerce_vis_cv_values_object(value), None

    def _coerce_vis_node_values_object(self, value):
        """Coerce a caller-supplied field into unmasked point-space values."""
        raw, original, rewrap = _split_unit(self._single_population_view(value, caller="Cell.vis_node(...)"))
        if raw.ndim == 0:
            return rewrap(np.full((self.n_point,), float(raw), dtype=float))
        if raw.ndim != 1:
            raise ValueError("Cell.vis_node(...) only supports scalar or 1-D value arrays.")
        if raw.shape[0] == self.n_point:
            return original
        if raw.shape[0] == self.n_cv:
            return self._cv_to_node_values(original)
        raise ValueError(
            f"Cell.vis_node(value=...) expects a point array of length {self.n_point} "
            f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
        )

    def _coerce_runtime_point_values_object(self, value):
        """Coerce one runtime field into unmasked point-space values."""
        raw, original, rewrap = _split_unit(self._single_population_view(value, caller="Runtime point inspection"))
        if raw.ndim == 0:
            return rewrap(np.full((self.n_point,), float(raw), dtype=float))
        if raw.ndim != 1:
            raise ValueError("Runtime point inspection only supports scalar or 1-D value arrays.")
        if raw.shape[0] == self.n_point:
            return original
        if raw.shape[0] == self.n_cv:
            return self._cv_to_point(original)
        raise ValueError(
            f"Runtime point inspection expects a point array of length {self.n_point} "
            f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
        )

    def _coerce_vis_cv_values_object(self, value):
        """Coerce a caller-supplied field into CV-space values."""
        raw, original, rewrap = _split_unit(self._single_population_view(value, caller="Cell.vis_cv(...)"))
        if raw.ndim == 0:
            return rewrap(np.full((self.n_cv,), float(raw), dtype=float))
        if raw.ndim != 1:
            raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D value arrays.")
        if raw.shape[0] == self.n_cv:
            return original
        if raw.shape[0] == self.n_point:
            return self._point_to_cv(original)
        raise ValueError(
            f"Cell.vis_cv(value=...) expects a CV array of length {self.n_cv} "
            f"or a point array of length {self.n_point}, got length {raw.shape[0]}."
        )

    def _coerce_named_vis_node_values_object(self, value):
        """Coerce a *named* state/parameter field into point-space values.

        A named field already in point space is masked down to midpoints,
        which is the difference from :meth:`_coerce_vis_node_values_object`:
        a named value is defined per CV, so only the midpoint carries it.
        """
        raw, original, rewrap = _split_unit(self._single_population_view(value, caller="Cell.vis_node(...)"))
        if raw.ndim == 0:
            return self._cv_to_node_values(rewrap(np.full((self.n_cv,), float(raw), dtype=float)))
        if raw.ndim != 1:
            raise ValueError("Cell.vis_node(...) only supports scalar or 1-D named value arrays.")
        if raw.shape[0] == self.n_point:
            return self._mask_non_midpoint_points(original)
        if raw.shape[0] == self.n_cv:
            return self._cv_to_node_values(original)
        raise ValueError("Cell.vis_node(...) cannot map the named value into point space.")

    def _coerce_named_vis_cv_values_object(self, value):
        """Coerce a *named* state/parameter field into CV-space values."""
        raw, original, rewrap = _split_unit(self._single_population_view(value, caller="Cell.vis_cv(...)"))
        if raw.ndim == 0:
            return rewrap(np.full((self.n_cv,), float(raw), dtype=float))
        if raw.ndim != 1:
            raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D named value arrays.")
        if raw.shape[0] == self.n_cv:
            return original
        if raw.shape[0] == self.n_point:
            return self._point_to_cv(self._mask_non_midpoint_points(original))
        raise ValueError("Cell.vis_cv(...) cannot map the named value into CV space.")

    def _resolve_unique_layout_by_kind(self, kind: str):
        matches = [layout for layout in self.layouts if layout.kind == kind]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise ValueError(f"Cell.vis_node(...) found no runtime layout with kind {kind!r}.")
        details = ", ".join(f"id={layout.id}:{layout.kind}" for layout in matches)
        raise ValueError(
            f"Cell.vis_node(...) found multiple runtime layouts for {kind!r}: {details}. "
            "Use ('layout_id', id, field) to select one exact layout."
        )

    def _cv_to_node_values(self, cv_values):
        node_tree = self.node_tree
        if hasattr(cv_values, "to_decimal") and hasattr(cv_values, "unit"):
            unit = cv_values.unit
            raw = np.asarray(cv_values.to_decimal(unit), dtype=float).reshape(-1)
            if raw.shape != (self.n_cv,):
                raise ValueError(f"_cv_to_node_values expects shape ({self.n_cv},), got {raw.shape!r}.")
            point_values = np.full((self.n_point,), np.nan, dtype=float)
            point_values[np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)] = raw
            return u.Quantity(point_values, unit)
        raw = np.asarray(cv_values, dtype=float).reshape(-1)
        if raw.shape != (self.n_cv,):
            raise ValueError(f"_cv_to_node_values expects shape ({self.n_cv},), got {raw.shape!r}.")
        point_values = np.full((self.n_point,), np.nan, dtype=float)
        point_values[np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)] = raw
        return point_values

    def _discretization_to_node_values(self, cv_values):
        return self._cv_to_node_values(cv_values)

    def _mask_non_midpoint_points(self, point_values):
        node_tree = self.node_tree
        midpoint_ids = np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)
        midpoint_mask = np.zeros((self.n_point,), dtype=bool)
        midpoint_mask[midpoint_ids] = True
        if hasattr(point_values, "to_decimal") and hasattr(point_values, "unit"):
            unit = point_values.unit
            raw = np.asarray(point_values.to_decimal(unit), dtype=float).reshape(-1)
            if raw.shape != (self.n_point,):
                raise ValueError(f"_mask_non_midpoint_points expects shape ({self.n_point},), got {raw.shape!r}.")
            masked = raw.copy()
            masked[~midpoint_mask] = np.nan
            return u.Quantity(masked, unit)
        raw = np.asarray(point_values, dtype=float).reshape(-1)
        if raw.shape != (self.n_point,):
            raise ValueError(f"_mask_non_midpoint_points expects shape ({self.n_point},), got {raw.shape!r}.")
        masked = raw.copy()
        masked[~midpoint_mask] = np.nan
        return masked

    def _layout_field_to_point_values(self, layout_id: int, field: str):
        layout = next((candidate for candidate in self.layouts if candidate.id == int(layout_id)), None)
        if layout is None:
            raise KeyError(f"Unknown layout id {layout_id!r}.")

        try:
            raw_values = self.get_state(layout.id, field)
        except KeyError:
            node = self.get_runtime_node(layout.id)
            if not hasattr(node, field):
                raise AttributeError(f"Runtime layout {layout.id!r} has no field {field!r}.")
            raw_values = getattr(node, field)
        return self._layout_values_to_point_space(layout, raw_values, field=field)

    def _layout_field_to_cv_values(self, layout_id: int, field: str):
        layout = next((candidate for candidate in self.layouts if candidate.id == int(layout_id)), None)
        if layout is None:
            raise KeyError(f"Unknown layout id {layout_id!r}.")
        try:
            raw_values = self.get_state(layout.id, field)
        except KeyError:
            node = self.get_runtime_node(layout.id)
            if not hasattr(node, field):
                raise AttributeError(f"Runtime layout {layout.id!r} has no field {field!r}.")
            raw_values = getattr(node, field)
        return self._layout_values_to_cv_space(layout, raw_values, field=field)

    def _layout_values_to_point_space(self, layout, raw_values, *, field: str):
        n_point = self.n_point
        raw_values = self._single_population_view(raw_values, field=field, caller="Cell.vis_node(...)")
        if hasattr(raw_values, "to_decimal") and hasattr(raw_values, "unit"):
            unit = raw_values.unit
            raw = np.asarray(raw_values.to_decimal(unit), dtype=float)
            point_values = np.full((n_point,), np.nan, dtype=float)
            if raw.ndim == 0:
                if layout.point_index is None:
                    raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
                point_values[np.asarray(layout.point_index, dtype=np.int32)] = float(raw)
                return u.Quantity(point_values, unit)
            if raw.ndim != 1:
                raise ValueError(f"Cell.vis_node(...) only supports 1-D value fields; {field!r} is not 1-D.")
            array = raw.reshape(-1)
            if array.shape[0] == n_point:
                if layout.point_index is None:
                    raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
                point_values[np.asarray(layout.point_index, dtype=np.int32)] = array[
                    np.asarray(layout.point_index, dtype=np.int32)
                ]
                return u.Quantity(point_values, unit)
            if layout.point_index is None or array.shape[0] != len(layout.point_index):
                raise ValueError(
                    f"Cell.vis_node(...) cannot map field {field!r} from layout {layout.id!r} "
                    f"with shape {array.shape!r} into point space."
                )
            point_values[layout.point_index] = array
            return u.Quantity(point_values, unit)

        raw = np.asarray(raw_values, dtype=float)
        point_values = np.full((n_point,), np.nan, dtype=float)
        if raw.ndim == 0:
            if layout.point_index is None:
                raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
            point_values[np.asarray(layout.point_index, dtype=np.int32)] = float(raw)
            return point_values
        if raw.ndim != 1:
            raise ValueError(f"Cell.vis_node(...) only supports 1-D value fields; {field!r} is not 1-D.")
        array = raw.reshape(-1)
        if array.shape[0] == n_point:
            if layout.point_index is None:
                raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
            point_values[np.asarray(layout.point_index, dtype=np.int32)] = array[
                np.asarray(layout.point_index, dtype=np.int32)
            ]
            return point_values
        if layout.point_index is None or array.shape[0] != len(layout.point_index):
            raise ValueError(
                f"Cell.vis_node(...) cannot map field {field!r} from layout {layout.id!r} "
                f"with shape {array.shape!r} into point space."
            )
        point_values[layout.point_index] = array
        return point_values

    def _layout_values_to_cv_space(self, layout, raw_values, *, field: str):
        n_cv = self.n_cv
        raw_values = self._single_population_view(raw_values, field=field, caller="Cell.vis_cv(...)")
        source_cv_ids = tuple(int(cv_id) for cv_id in layout.source_cv_ids)
        midpoint_by_cv = {cv_id: int(self.node_tree.cv_to_mid_node_id[cv_id]) for cv_id in source_cv_ids}
        if hasattr(raw_values, "to_decimal") and hasattr(raw_values, "unit"):
            unit = raw_values.unit
            raw = np.asarray(raw_values.to_decimal(unit), dtype=float)
            cv_values = np.full((n_cv,), np.nan, dtype=float)
            if raw.ndim == 0:
                for cv_id in source_cv_ids:
                    cv_values[cv_id] = float(raw)
                return u.Quantity(cv_values, unit)
            if raw.ndim != 1:
                raise ValueError(f"Cell.vis_cv(...) only supports 1-D value fields; {field!r} is not 1-D.")
            array = raw.reshape(-1)
            if array.shape[0] == n_cv:
                return raw_values
            if array.shape[0] == self.n_point:
                for cv_id, point_id in midpoint_by_cv.items():
                    cv_values[cv_id] = array[point_id]
                return u.Quantity(cv_values, unit)
            if layout.point_index is None or array.shape[0] != len(layout.point_index):
                raise ValueError(
                    f"Cell.vis_cv(...) cannot map field {field!r} from layout {layout.id!r} "
                    f"with shape {array.shape!r} into CV space."
                )
            value_by_point = {
                int(point_id): float(array[index])
                for index, point_id in enumerate(np.asarray(layout.point_index, dtype=np.int32))
            }
            for cv_id, point_id in midpoint_by_cv.items():
                if point_id in value_by_point:
                    cv_values[cv_id] = value_by_point[point_id]
            return u.Quantity(cv_values, unit)

        raw = np.asarray(raw_values, dtype=float)
        cv_values = np.full((n_cv,), np.nan, dtype=float)
        if raw.ndim == 0:
            for cv_id in source_cv_ids:
                cv_values[cv_id] = float(raw)
            return cv_values
        if raw.ndim != 1:
            raise ValueError(f"Cell.vis_cv(...) only supports 1-D value fields; {field!r} is not 1-D.")
        array = raw.reshape(-1)
        if array.shape[0] == n_cv:
            return array
        if array.shape[0] == self.n_point:
            for cv_id, point_id in midpoint_by_cv.items():
                cv_values[cv_id] = array[point_id]
            return cv_values
        if layout.point_index is None or array.shape[0] != len(layout.point_index):
            raise ValueError(
                f"Cell.vis_cv(...) cannot map field {field!r} from layout {layout.id!r} "
                f"with shape {array.shape!r} into CV space."
            )
        value_by_point = {
            int(point_id): float(array[index])
            for index, point_id in enumerate(np.asarray(layout.point_index, dtype=np.int32))
        }
        for cv_id, point_id in midpoint_by_cv.items():
            if point_id in value_by_point:
                cv_values[cv_id] = value_by_point[point_id]
        return cv_values

    # ------------------------------------------------------------------
    # Solver path (runtime-only)

    def _resolve_t(self):
        try:
            return brainstate.environ.get("t")
        except KeyError:
            return self.current_time

    def pre_integral(self):
        self._raise_if_not_initialized("pre_integral()")
        point_V = self._cv_to_point(self.V.value)
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                args = self._runtime_node_phase_args(path, node, point_V)
                node.pre_integral(*args)

    def compute_derivative(self):
        self._raise_if_not_initialized("compute_derivative()")
        self.V.derivative = self.compute_voltage_derivative(self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                args = self._runtime_node_phase_args(path, node, point_V)
                node.compute_derivative(*args)

    def compute_membrane_derivative(self, V):
        self._raise_if_not_initialized("compute_membrane_derivative()")
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, t=t)
        return I_total / self.C

    def _voltage_linearizer(self):
        """Return the configured voltage-only membrane linearizer."""
        if self._membrane_linearizer == "generic":
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

        runtime = self.runtime
        midpoint_mask = jnp.asarray(runtime.midpoint_mask_np)

        def linearize(V, *args):
            # CV/point mappings stay outside the differentiated function, so
            # reverse-mode AD never needs the large CV-to-point scatter-add.
            point_V = bridge.cv_to_point(V, runtime)
            point_C = bridge.cv_to_point(self.C, runtime)
            capacitance_unit = u.get_unit(point_C)
            safe_point_C = u.Quantity(
                jnp.where(midpoint_mask, u.get_mantissa(point_C), 1.0),
                capacitance_unit,
            )

            def point_membrane_derivative(candidate_point_V, *_args):
                I_point = currents.total_membrane_current_point(
                    self,
                    point_V=candidate_point_V,
                    t=self._resolve_t(),
                )
                derivative = I_point / safe_point_C
                return u.Quantity(
                    jnp.where(
                        midpoint_mask,
                        u.get_mantissa(derivative),
                        0.0,
                    ),
                    u.get_unit(derivative),
                )

            point_linearizer = brainstate.transform.vector_grad(
                point_membrane_derivative,
                argnums=0,
                return_value=True,
                unit_aware=False,
            )
            point_linear, point_derivative = point_linearizer(point_V, *args)
            return (
                bridge.point_to_cv(point_linear, runtime),
                bridge.point_to_cv(point_derivative, runtime),
            )

        return linearize

    def _get_axial_operator(self):
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("_get_axial_operator() requires init_state() first.")
        float_dtype = jnp.asarray(0.0).dtype
        cache = runtime.axial_operator_cache
        if cache is not None and cache.float_dtype == float_dtype:
            self._axial_jax = cache.operator
            return cache.operator

        if runtime.axial_operator_np is None:
            runtime.axial_operator_np = np.asarray(
                build_cv_axial_operator(
                    self,
                    node_tree=self.node_tree,
                    scheduling=self._node_scheduling_unchecked(algorithm="dhs"),
                ),
                dtype=np.float64,
            )

        operator = jnp.asarray(runtime.axial_operator_np, dtype=brainstate.environ.dftype()) * (u.ms**-1)
        cache = AxialOperatorCache(float_dtype=float_dtype, operator=operator)
        if not is_traced_value(operator):
            runtime.axial_operator_cache = cache
        self._axial_jax = operator
        return operator

    def compute_axial_derivative(self, V):
        self._raise_if_not_initialized("compute_axial_derivative()")
        V_mv = u.Quantity(u.math.asarray(V.to_decimal(u.mV)), u.mV)
        axial_operator = self._get_axial_operator()
        return -u.math.matmul(V_mv, axial_operator.T)

    def compute_voltage_derivative(self, V):
        return self.compute_membrane_derivative(V) + self.compute_axial_derivative(V)

    def _top_level_ion_channel_nodes(self):
        return tuple(self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items())

    def _family_ion_nodes(self):
        return tuple((path, node) for path, node in self._top_level_ion_channel_nodes() if isinstance(node, Ion))

    def _family_channel_nodes(self):
        nodes = []
        for path, node in self._top_level_ion_channel_nodes():
            if isinstance(node, Ion):
                for child_path, child in brainstate.graph.nodes(
                    node,
                    Channel,
                    allowed_hierarchy=(1, 1),
                ).items():
                    if getattr(child, "_skip_family_update", False):
                        continue
                    nodes.append((path + child_path, child))
            elif isinstance(node, MixIons):
                for child_path, child in brainstate.graph.nodes(
                    node,
                    Channel,
                    allowed_hierarchy=(1, 1),
                ).items():
                    if getattr(child, "_skip_family_update", False):
                        continue
                    nodes.append((path + child_path, child))
            elif isinstance(node, Channel):
                nodes.append((path, node))
        return tuple(nodes)

    def _integrate_selected_ion_channel_states(self, selected_paths, point_V, excluded_paths=()):
        selected_paths = tuple(tuple(path) for path in selected_paths)
        if not selected_paths:
            return
        excluded_paths = [("V",), *tuple(tuple(path) for path in excluded_paths)]

        def _pre_integral():
            for path, node in self._top_level_ion_channel_nodes():
                if isinstance(node, Ion) and path in selected_paths:
                    node.pre_integral(point_V, recursive_child=False)
                if isinstance(node, Channel) and path in selected_paths:
                    node.pre_integral(point_V)
                if isinstance(node, (Ion, MixIons)):
                    self._run_selected_child_channel_hook(
                        node,
                        path,
                        selected_paths,
                        "pre_integral",
                        point_V,
                    )

        def _compute_derivative():
            for path, node in self._top_level_ion_channel_nodes():
                if isinstance(node, Ion) and path in selected_paths:
                    node.compute_derivative(point_V, recursive_child=False)
                if isinstance(node, Channel) and path in selected_paths:
                    node.compute_derivative(point_V)
                if isinstance(node, (Ion, MixIons)):
                    self._run_selected_child_channel_hook(
                        node,
                        path,
                        selected_paths,
                        "compute_derivative",
                        point_V,
                    )

        def _post_integral():
            for path, node in self._top_level_ion_channel_nodes():
                if isinstance(node, Ion) and path in selected_paths:
                    node.post_integral(point_V, recursive_child=False)
                if isinstance(node, Channel) and path in selected_paths:
                    node.post_integral(point_V)
                if isinstance(node, (Ion, MixIons)):
                    self._run_selected_child_channel_hook(
                        node,
                        path,
                        selected_paths,
                        "post_integral",
                        point_V,
                    )

        _ind_exp_euler_step_selected(
            self,
            include_paths=selected_paths,
            excluded_paths=excluded_paths,
            pre_integral=_pre_integral,
            compute_derivative=_compute_derivative,
            post_integral=_post_integral,
            allow_empty=True,
        )

    def _integrate_selected_ion_self_states(
        self,
        ion_nodes,
        selected_paths,
        point_V,
        excluded_paths,
    ):
        selected_paths = tuple(tuple(path) for path in selected_paths)
        if not selected_paths:
            return

        selected_path_set = set(selected_paths)

        def _run_phase(hook_name):
            for path, ion in ion_nodes:
                if path in selected_path_set:
                    getattr(ion, hook_name)(point_V, recursive_child=False)

        _ind_exp_euler_step_selected(
            self,
            include_paths=selected_paths,
            excluded_paths=excluded_paths,
            pre_integral=lambda: _run_phase("pre_integral"),
            compute_derivative=lambda: _run_phase("compute_derivative"),
            post_integral=lambda: _run_phase("post_integral"),
            allow_empty=True,
        )

    @staticmethod
    def _run_selected_child_channel_hook(parent, parent_path, selected_paths, hook_name, point_V):
        for child_path, child in brainstate.graph.nodes(
            parent,
            Channel,
            allowed_hierarchy=(1, 1),
        ).items():
            full_path = parent_path + child_path
            if full_path not in selected_paths:
                continue
            if isinstance(parent, Ion):
                getattr(child, hook_name)(point_V, parent.pack_info())
            else:
                infos = tuple([parent._get_ion(root).pack_info() for root in child.root_type.__args__])
                getattr(child, hook_name)(point_V, *infos)

    def _update_ion_channels_by_integration(self, point_V):
        with jax.named_scope("braincell:ion_update:integration:dependent"):
            for path, node in self._top_level_ion_channel_nodes():
                if isinstance(node, IndependentIntegration):
                    continue
                args = self._runtime_node_phase_args(path, node, point_V)
                with jax.named_scope(_scope_name("braincell:ion_update:node", path, node)):
                    jax.named_call(
                        ind_exp_euler_step,
                        name=_call_name("braincell:ion_update:node_step", path, node),
                    )(node, *args)

        with jax.named_scope("braincell:ion_update:integration:independent"):
            for path, node in self._top_level_ion_channel_nodes():
                with jax.named_scope(_scope_name("braincell:ion_update:node", path, node)):
                    jax.named_call(
                        node.ind_update,
                        name=_call_name("braincell:ion_update:node_ind_update", path, node),
                    )(point_V)

    def _update_ion_channel_families(self, point_V):
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
                point_V,
                excluded_paths=[("V",), *channel_paths],
            )

        # Independent Ion states use their own updater, still without
        # recursing into child channels.
        with jax.named_scope("braincell:ion_update:family:independent_ion"):
            for path, node in ion_nodes:
                if isinstance(node, IndependentIntegration):
                    with jax.named_scope(_scope_name("braincell:ion_update:ion", path, node)):
                        jax.named_call(
                            node.ind_update,
                            name=_call_name("braincell:ion_update:ion_ind_update", path, node),
                        )(point_V, recursive_child=False)

        # Channel nodes include Ion child channels, MixIons child channels,
        # and top-level channels. The owner path rebuilds the right ion args.
        with jax.named_scope("braincell:ion_update:family:dependent_channel"):
            for path, node in channel_nodes:
                if not self._is_independent_channel(node):
                    target, args = self._channel_integration_target_and_args(
                        path,
                        node,
                        point_V,
                    )
                    with jax.named_scope(_scope_name("braincell:ion_update:channel", path, node)):
                        jax.named_call(
                            ind_exp_euler_step,
                            name=_call_name("braincell:ion_update:channel_step", path, node),
                        )(target, *args)

        # Independent channels finish through their own update rule.
        with jax.named_scope("braincell:ion_update:family:independent_channel"):
            for path, node in channel_nodes:
                if not self._is_independent_channel(node):
                    continue
                target, args = self._channel_integration_target_and_args(
                    path,
                    node,
                    point_V,
                )
                with jax.named_scope(_scope_name("braincell:ion_update:channel", path, node)):
                    jax.named_call(
                        target.ind_update,
                        name=_call_name("braincell:ion_update:channel_ind_update", path, node),
                    )(*args)

    @staticmethod
    def _is_independent_channel(node):
        channel = getattr(node, "_channel", node)
        return isinstance(channel, IndependentIntegration)

    def _channel_integration_target_and_args(self, path, node, point_V):
        if hasattr(node, "_channel") and hasattr(node, "_infos"):
            return node._channel, (point_V, *node._infos())
        return node, self._channel_update_args(path, node, point_V)

    def _channel_update_args(self, path, node, point_V):
        if len(path) >= 4 and path[-2] == "channels":
            owner = self._node_at_path(path[:-2])
            if isinstance(owner, Ion):
                return point_V, owner.pack_info()
            if isinstance(owner, MixIons):
                infos = tuple([owner._get_ion(root).pack_info() for root in node.root_type.__args__])
                return (point_V, *infos)
        return (point_V,)

    def _runtime_node_phase_args(self, path, node, point_V):
        if isinstance(node, RuntimeSynapse):
            layout_id = _layout_id_from_runtime_path(path)
            layout = self._runtime.layouts[layout_id]
            if layout.point_index is None:
                raise ValueError(f"Synapse layout {layout.id!r} is missing point_index.")
            return (_gather_layout_point_values(point_V, layout),)
        return self._channel_update_args(path, node, point_V)

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
        point_V = self._cv_to_point(self.V.value if V is None else V)
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not getattr(type(node), "uses_total_current", False):
                continue
            with jax.named_scope(_scope_name("braincell:ion_current_cache:node", path, node)):
                try:
                    node._cached_total_current = jax.named_call(
                        node.current,
                        name=_call_name("braincell:ion_current_cache:node_current", path, node),
                    )(point_V, include_external=True)
                except TypeError:
                    node._cached_total_current = jax.named_call(
                        node.current,
                        name=_call_name("braincell:ion_current_cache:node_current", path, node),
                    )(point_V)

    def clear_ion_total_current_cache(self) -> None:
        """Remove per-step ion source-current caches."""
        self._raise_if_not_initialized("clear_ion_total_current_cache()")
        for _, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if hasattr(node, "_cached_total_current"):
                delattr(node, "_cached_total_current")

    def post_integral(self):
        self._raise_if_not_initialized("post_integral()")
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                args = self._runtime_node_phase_args(path, node, point_V)
                node.post_integral(*args)

    def update(self):
        """Advance the cell by one simulation step.

        This method is the standalone cell-level step wrapper. It applies
        already-prepared synaptic events, advances continuous membrane and
        mechanism dynamics by one ``dt``, computes the spike output, and
        prepares synaptic input for the next standalone step.

        Returns
        -------
        object
            Spike value produced by the transition from the previous membrane
            voltage to the updated membrane voltage.

        Notes
        -----
        The standalone update order is:

        1. Apply prepared runtime synapse events.
        2. Advance voltage and mechanism dynamics through ``self.solver``.
        3. Detect and store the current spike.
        4. Prepare discrete event payloads for the next step.

        In network execution, delayed event delivery is scheduled outside the
        cell. ``Network.run(...)`` writes arrivals into runtime state buffers
        before calling the corresponding internal cell phases.
        """
        self._raise_if_not_initialized("update()")
        self._begin_step()
        spk = self._update_dynamics()
        self._prepare_next_synapse_inputs(t=self._resolve_t() + brainstate.environ.get_dt())
        return spk

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
        point_V = self._cv_to_point(self.V.value)
        self._apply_runtime_synapse_events(point_V)

    def _update_dynamics(self):
        """Advance continuous cell dynamics and update spike state.

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
        5. Clear temporary current caches.
        6. Detect threshold crossing and write ``self.spike``.
        """
        self._raise_if_not_initialized("_update_dynamics()")

        last_V = self.V.value
        self._event_previous_V.value = last_V
        if brainstate.environ.get("dt", None) is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")

        with jax.named_scope("braincell:cell_update:solver"):
            self.solver(self)

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
        """
        self._raise_if_not_initialized("_prepare_next_synapse_inputs()")
        point_V = self._cv_to_point(self.V.value)
        if t is None:
            self._prepare_runtime_synapse_inputs(point_V)
        else:
            with brainstate.environ.context(t=t):
                self._prepare_runtime_synapse_inputs(point_V)

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
            path = (f"layout_{layout.id}",)
            args = self._runtime_node_phase_args(path, synapse, point_V)
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
        for layout, _ in self._runtime.iter_synapse_layouts():
            if layout.id not in self._runtime.event_buffers:
                continue
            total_drive = self._runtime.get_event_buffer(layout.id)
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
            self._runtime.event_buffers[layout.id].value = total_drive

    def _evaluate_contact_inputs(self, layout, *, t, template, scheduled_only=True):
        """Return weighted Connection arrivals addressed to one synapse layout."""
        if layout.placement_index is None:
            return u.math.zeros_like(template)
        dt = brainstate.environ.get("dt", None)
        if dt is None:
            raise ValueError("Connection event delivery requires brainstate.environ['dt'].")

        output = jnp.zeros_like(
            jnp.asarray(template.to_decimal(template.unit))
            if isinstance(template, u.Quantity)
            else jnp.asarray(template)
        )
        for connection in self.connections._call_views(scheduled=scheduled_only):
            synapse_type = str(connection.synapse_type[0])
            if self._get_synapse_store().layout_id(synapse_type) != int(layout.id):
                continue
            row_index = np.arange(len(connection), dtype=np.int32)
            local_index = self._get_synapse_store().runtime_rows(connection.synapse_id).astype(np.int32)
            event_count = connection.source.event_count(
                connection.source_index[row_index],
                t=t,
                delay=connection.delay[row_index],
                dt=dt,
            )
            connection_weight = connection.weight
            if connection_weight is None:
                if isinstance(template, u.Quantity):
                    raise TypeError("Trigger-only Connection cannot target a physical event buffer.")
                weight = 1.0
            elif isinstance(template, u.Quantity):
                if not isinstance(connection_weight, u.Quantity):
                    raise TypeError("Connection weight is dimensionless but its target event buffer is not.")
                weight = connection_weight[row_index].to_decimal(template.unit)
            else:
                if isinstance(connection_weight, u.Quantity):
                    raise TypeError("Connection weight has units but its target event buffer is dimensionless.")
                weight = connection_weight[row_index]
            contribution = event_count * u.math.asarray(weight)
            output = output.at[local_index].add(contribution)
        return u.Quantity(output, template.unit) if isinstance(template, u.Quantity) else output

    def _apply_direct_live_connection_events(self) -> None:
        """Route live direct sources and run target handlers at this boundary."""
        live_connections = self.connections._call_views(scheduled=False)
        if not live_connections:
            return
        dt = brainstate.environ.get("dt", None)
        if dt is None:
            raise ValueError("Live Connection delivery requires brainstate.environ['dt'].")
        t = self._resolve_t()
        layouts = tuple(self._runtime.iter_synapse_layouts())
        drives = {}
        for layout, _ in layouts:
            if layout.id not in self._runtime.event_buffers:
                continue
            template = self._runtime.get_event_buffer(layout.id)
            raw = template.to_decimal(template.unit) if isinstance(template, u.Quantity) else template
            drives[layout.id] = jnp.zeros_like(jnp.asarray(raw))

        for connection in live_connections:
            counts = connection.event_count(t=t, dt=dt)
            synapse_type = str(connection.synapse_type[0])
            layout_id = self._get_synapse_store().layout_id(synapse_type)
            if layout_id not in drives:
                continue
            template = self._runtime.get_event_buffer(layout_id)
            connection_weight = connection.weight
            if connection_weight is None:
                if isinstance(template, u.Quantity):
                    raise TypeError("Trigger-only Connection cannot target a physical event buffer.")
                weight = 1.0
            elif isinstance(template, u.Quantity):
                if not isinstance(connection_weight, u.Quantity):
                    raise TypeError("Connection weight is dimensionless but its target event buffer is not.")
                weight = connection_weight.to_decimal(template.unit)
            else:
                if isinstance(connection_weight, u.Quantity):
                    raise TypeError("Connection weight has units but its target event buffer is dimensionless.")
                weight = connection_weight
            contribution = counts * u.math.asarray(weight)
            local_indices = self._get_synapse_store().runtime_rows(connection.synapse_id).astype(np.int32)
            drives[layout_id] = drives[layout_id].at[local_indices].add(contribution)

        point_v = self._cv_to_point(self.V.value)
        for layout, synapse in layouts:
            if layout.id not in drives:
                continue
            template = self._runtime.get_event_buffer(layout.id)
            drive = (
                u.Quantity(drives[layout.id], template.unit) if isinstance(template, u.Quantity) else drives[layout.id]
            )
            self._apply_synapse_layout_event_drive(layout.id, drive, point_v=point_v)

    def _apply_synapse_layout_event_drive(self, layout_id: int, drive, *, point_v=None) -> None:
        """Apply one already-aggregated boundary payload to a runtime layout."""
        layout = next(layout for layout in self._runtime.layouts if int(layout.id) == int(layout_id))
        synapse = self._runtime.get_runtime_node(layout.id)
        if point_v is None:
            point_v = self._cv_to_point(self.V.value)
        path = (f"layout_{layout.id}",)
        args = self._runtime_node_phase_args(path, synapse, point_v)
        synapse.apply_events(drive, *args)

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
            args = self._runtime_node_phase_args(path, node, point_V)
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
        self.connections.reset_runtime()
        v_value = self._materialize_population_parameter("V_init")
        self._V_init_materialized = v_value
        self.V.value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        self.spike.value = _zero_spike_like(self.V.value)
        self._event_previous_V.value = self.V.value
        self._current_time_state.value = 0.0 * u.ms
        for layout_id in self._runtime.event_buffers:
            self._runtime.clear_event_buffer(layout_id)
        point_V = self._cv_to_point(self.V.value)
        with state_grouping(True):
            for path, channel in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
                args = self._runtime_node_phase_args(path, channel, point_V)
                channel.reset_state(*args, batch_size=batch_size)

    # ------------------------------------------------------------------
    # Inspection forwards (runtime-only)

    @property
    def layouts(self):
        self._raise_if_not_initialized("layouts")
        return self._runtime.layouts

    @property
    def voltage_shape(self):
        self._raise_if_not_initialized("voltage_shape")
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
        return probes.sample_probe(self, name)

    def sample_probes(self) -> dict[str, object]:
        self._raise_if_not_initialized("sample_probes()")
        return probes.sample_probes(self)

    def mech_table(self) -> MechanismObjectTable:
        self._raise_if_not_initialized("mech_table()")
        runtime = self._runtime
        node_tree = self.node_tree
        column_ids = tuple(range(len(node_tree.nodes)))

        row_keys: list[tuple[str, str]] = []
        row_labels: list[str] = []
        row_index_by_key: dict[tuple[str, str], int] = {}
        pending_cells: list[tuple[int, int, MechanismObjectCell]] = []
        layout_id_by_signature = {
            (layout.target,) + mechanism_signature(runtime.get_layout_mechanism(layout.id)): layout.id
            for layout in runtime.layouts
        }
        layout_id_by_synapse_type = {
            runtime.get_layout_mechanism(layout.id).synapse_type: layout.id
            for layout in runtime.layouts
            if isinstance(runtime.get_layout_mechanism(layout.id), SynapsePlacement)
        }

        def ensure_row(mechanism: object) -> int:
            row_key = mechanism_cell_key(mechanism)
            row_index = row_index_by_key.get(row_key)
            if row_index is not None:
                return row_index
            row_index = len(row_keys)
            row_keys.append(row_key)
            class_name, instance_name = row_key
            row_labels.append(class_name if class_name == instance_name else f"{instance_name}:{class_name}")
            row_index_by_key[row_key] = row_index
            return row_index

        for cv in self.cvs:
            midpoint_point_id = int(node_tree.cv_to_mid_node_id[cv.id])
            for mechanism in cv.density_mech:
                row_key = mechanism_cell_key(mechanism)
                row_index = ensure_row(mechanism)
                layout_id = layout_id_by_signature[("density",) + mechanism_signature(mechanism)]
                pending_cells.append(
                    (
                        row_index,
                        midpoint_point_id,
                        MechanismObjectCell(
                            runtime=runtime,
                            layout_id=int(layout_id),
                            class_name=row_key[0],
                            instance_name=row_key[1],
                            column_id=midpoint_point_id,
                            domain="point",
                            cv_id=None,
                            point_id=midpoint_point_id,
                        ),
                    )
                )

        for point_id, node in enumerate(node_tree.nodes):
            for mechanism in node.point_mech:
                row_key = mechanism_cell_key(mechanism)
                row_index = ensure_row(mechanism)
                layout_id = (
                    layout_id_by_synapse_type[mechanism.synapse_type]
                    if isinstance(mechanism, SynapsePlacement)
                    else layout_id_by_signature[("point",) + mechanism_signature(mechanism)]
                )
                pending_cells.append(
                    (
                        row_index,
                        int(point_id),
                        MechanismObjectCell(
                            runtime=runtime,
                            layout_id=int(layout_id),
                            class_name=row_key[0],
                            instance_name=row_key[1],
                            column_id=int(point_id),
                            domain="point",
                            cv_id=None,
                            point_id=int(point_id),
                        ),
                    )
                )

        values = np.full((len(row_keys), len(column_ids)), None, dtype=object)
        for row_index, column_id, cell in pending_cells:
            values[row_index, int(column_id)] = cell

        return MechanismObjectTable(
            domain="point",
            row_keys=tuple(row_keys),
            row_labels=tuple(row_labels),
            column_ids=column_ids,
            values=values,
        )

    # ------------------------------------------------------------------
    # Run (auto-inits from DECLARING)

    def run(self, *, dt, duration):
        """Run the cell for ``duration`` at ``dt`` and return probe traces.

        If :meth:`init_state` has not been called yet, ``run`` calls it
        automatically. Once initialized the cell will *not* be
        re-initialized on subsequent ``run`` invocations.
        """
        owner = self.network_owner
        if owner is not None:
            owner_name = owner.name if owner.name is not None else "<unnamed>"
            raise RuntimeError(f"Cell belongs to Network {owner_name!r}; run it through Network {owner_name!r}.")
        if not self._initialized:
            self.init_state()
        return run_module.run(self, dt=dt, duration=duration)


#: Alias of :class:`Cell`, named for what the model is rather than for the
#: shorthand. ``MultiCompartment is Cell`` — the two names are the same
#: object, so ``isinstance``, subclassing, and pickling behave identically
#: through either. Prefer it when the surrounding code also mentions
#: :class:`~braincell.SingleCompartment` and the contrast matters.
MultiCompartment = Cell


# ----------------------------------------------------------------------
# Helpers


def _split_unit(value):
    """Separate a possibly-united field into the parts a coercer needs.

    Every ``Cell`` inspection coercer decides what a field means purely
    from its length, then either returns it untouched or maps it between
    point and CV space. Only two steps care about units: reading the
    length needs a bare mantissa, and synthesizing an array from a scalar
    needs the unit put back. Splitting those out here lets each coercer
    state its length rules once instead of once per storage flavour.

    Parameters
    ----------
    value : array-like or Quantity
        A field taken from a runtime buffer or supplied by a caller.

    Returns
    -------
    mantissa : numpy.ndarray
        Unitless values, for shape and length tests.
    original : array-like or Quantity
        ``value`` itself when it carried a unit, else ``mantissa``. This is
        what to return or hand to a spatial mapper, so the unit survives.
    rewrap : Callable[[numpy.ndarray], object]
        Puts the unit back on a freshly built array; identity when there
        was no unit.
    """
    if hasattr(value, "to_decimal") and hasattr(value, "unit"):
        unit = value.unit
        mantissa = np.asarray(value.to_decimal(unit), dtype=float)
        return mantissa, value, lambda array: u.Quantity(array, unit)
    mantissa = np.asarray(value, dtype=float)
    return mantissa, mantissa, lambda array: array


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


def _split_synapse_spec_rows(
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


def _normalize_population_selection(selection, *, size: int) -> tuple[int, ...]:
    """Normalize one integer, slice, or one-dimensional integer selection."""
    if isinstance(selection, slice):
        indices = tuple(range(size))[selection]
    elif isinstance(selection, (int, np.integer)) and not isinstance(selection, bool):
        index = int(selection)
        if index < 0:
            index += size
        indices = (index,)
    else:
        array = np.asarray(selection)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise TypeError("Cell selection must be an integer, slice, or one-dimensional integer sequence.")
        normalized = []
        for raw in array.tolist():
            index = int(raw)
            if index < 0:
                index += size
            normalized.append(index)
        indices = tuple(dict.fromkeys(normalized))
    if any(index < 0 or index >= size for index in indices):
        raise IndexError(f"Cell population selection is outside [0, {size!r}): {indices!r}.")
    return tuple(indices)


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


def _apply_population_parameter_overrides(value, *, overrides: Mapping[int, object], name: str):
    """Scatter per-cell declaration overrides into an existing dense value."""
    if not overrides:
        return value
    if not isinstance(value, u.Quantity):
        raise TypeError(f"Cell {name} must materialize as a voltage quantity before applying CellView overrides.")
    unit = value.unit
    decimal = np.array(value.to_decimal(unit), copy=True)
    if decimal.ndim != 2:
        raise ValueError(f"Cell {name} must have population-by-CV shape, got {decimal.shape!r}.")
    for population_index, row in overrides.items():
        decimal[int(population_index)] = row.to_decimal(unit)
    return u.Quantity(decimal, unit)


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
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(f"solver must be str or callable, got {type(solver).__name__!s}.")


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


def _layout_id_from_runtime_path(path) -> int:
    if len(path) == 0:
        raise ValueError(f"Expected runtime layout path ending with 'layout_<id>', got {path!r}.")
    last = path[-1]
    if not isinstance(last, str) or not last.startswith("layout_"):
        raise ValueError(f"Expected runtime layout path ending with 'layout_<id>', got {path!r}.")
    return int(last.split("_", 1)[1])


def _gather_layout_point_values(values, layout):
    """Gather point values for a broadcast or packed point layout."""
    if layout.point_index is None:
        raise ValueError(f"Point layout {layout.id!r} is missing point_index.")
    if layout.population_index is None:
        return values[..., layout.point_index]
    return values[..., layout.population_index, layout.point_index]


def _scope_name(prefix: str, path, node) -> str:
    """Build a stable, profiler-safe internal JAX scope name."""
    path_name = "_".join(str(part) for part in path) if path else "root"
    class_name = type(getattr(node, "_channel", node)).__name__
    raw = f"{prefix}:{path_name}:{class_name}"
    cleaned = "".join(ch if ch.isalnum() or ch in ":_" else "_" for ch in raw)
    return cleaned[:180]


def _call_name(prefix: str, path, node) -> str:
    """Build a profiler-safe ``jax.named_call`` name."""
    return _scope_name(prefix, path, node).replace(":", "_")


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
    (``pop_size + (n_cv,)``). See ``docs/specs/2026-08-13-cell-hidden-group-state.md``
    for why rank-0 populations are rejected.

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
