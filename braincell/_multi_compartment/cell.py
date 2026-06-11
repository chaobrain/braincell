"""``Cell`` — single-class multi-compartment neuron.

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

from typing import Callable, Mapping, Optional
from dataclasses import dataclass

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base import Channel, HHTypedNeuron, Ion, IonChannel, MixIons, Synapse as RuntimeSynapse, _cast_like
from braincell._misc import is_traced_value
from braincell._typing import Initializer
from braincell._compute.table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell._compute.scheduling import build_node_scheduling
from braincell._compute.runtime import (
    CellRuntimeState,
    _is_root_level_runtime_node,
    build_placeholder_ions,
    clone_morpho,
    cv_value_vector,
    mechanism_signature,
)
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
from braincell.filter import LocsetExpr, LocsetMask, RegionExpr, RegionMask
from braincell.filter.helper import normalize_region_intervals
from braincell.morph.morphology import Morphology
from braincell.quad import get_integrator, ind_exp_euler_step
from braincell.quad._exp_euler import _ind_exp_euler_step_selected
from braincell.quad._staggered import build_cv_axial_operator
from braincell.quad.protocol import DiffEqState, IndependentIntegration
from braincell.mech import Synapse as SynapsePlacement
from . import bridge, currents, probes, run as run_module

__all__ = ["Cell"]


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
            raise AttributeError(
                f"Runtime ion {self.name!r} has no field {field!r}."
            )
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


class Cell(HHTypedNeuron):
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
    ion_channel_update_order : {"family", "integration"}
        Post-voltage ion/channel scheduling. ``"family"`` updates all ions
        before all channels; ``"integration"`` preserves the previous
        IndependentIntegration-grouped scheduling.
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
        pop_size: brainstate.typing.Size = (),
        cv_policy: CVPolicy | None = None,
        V_th: u.Quantity = 0 * u.mV,
        V_init: Optional[Initializer] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        cache_ion_total_current: bool = True,
        ion_channel_update_order: str = "family",
        name: str | None = None,
    ) -> None:
        """Initialize a multi-compartment cell declaration.

        Parameters
        ----------
        morpho : Morphology
            Morphology tree shared by every homogeneous population
            instance.
        pop_size : int or tuple of int, optional
            Optional homogeneous population shape. Runtime state is
            expanded to ``pop_size + (n_cv,)`` and point-space arrays to
            ``pop_size + (n_point,)``.
        cv_policy : CVPolicy, optional
            Control-volume splitting policy.
        V_th : Quantity, optional
            Spike threshold.
        V_init : Initializer, optional
            Initial membrane voltage. ``None`` uses the per-CV resting
            potential.
        spk_fun : Callable, optional
            Surrogate-gradient spike function.
        solver : str or callable, optional
            Integrator name or concrete step function.
        cache_ion_total_current : bool, optional
            Whether to snapshot ion total current at step start for
            NEURON-style schedules.
        ion_channel_update_order : {"family", "integration"}, optional
            Post-voltage ion/channel update order.
        name : str, optional
            Cell name.
        """
        normalized_pop_size = _normalize_pop_size(pop_size)
        HHTypedNeuron.__init__(self, size=normalized_pop_size + (1,), name=name)

        if not isinstance(morpho, Morphology):
            raise TypeError(
                f"Cell expects Morphology, got {type(morpho).__name__!s}."
            )

        self._declaration_morpho = morpho
        self._morpho = morpho
        self._pop_size = normalized_pop_size

        self._discretization_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._discretization_policy, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(self._discretization_policy).__name__!s}."
            )

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_th_declaration = V_th
        self._V_init = V_init
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)
        self.cache_ion_total_current = bool(cache_ion_total_current)
        self.ion_channel_update_order = _validate_ion_channel_update_order(
            ion_channel_update_order
        )

        self._discretization_cache: Discretization | None = None
        self._discretization_cache_key: object = None

        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._node_scheduling_cache: dict[tuple[str, int], object] = {}

        self._runtime: CellRuntimeState | None = None
        self._runtime_cvs_cache: tuple[RuntimeCVView, ...] | None = None
        self._runtime_nodes_cache: tuple[RuntimeNodeView, ...] | None = None
        self._axial_jax = None
        self._synapse_input_bindings: dict[str, list[tuple[object, object, object]]] = {}

        self._initialized = False

        # Eager policy validation via the preview.
        _ = self.cvs

    # ------------------------------------------------------------------
    # Phase guards

    def _raise_if_initialized(self, action: str) -> None:
        if self._initialized:
            raise RuntimeError(
                f"Cannot {action} after init_state(); call reset() first."
            )

    def _raise_if_not_initialized(self, action: str) -> None:
        if not self._initialized:
            raise RuntimeError(f"{action} requires init_state() first.")

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
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(value).__name__!s}."
            )
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

    @property
    def V_init(self):
        return self._V_init

    @V_init.setter
    def V_init(self, value) -> None:
        self._raise_if_initialized("assign V_init")
        self._V_init = value

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

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        """Paint mechanisms onto ``region``. Returns ``self`` for chaining."""
        self._raise_if_initialized("paint()")
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._invalidate_discretization_cache()
        return self

    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":
        """Place point mechanisms at ``locset``. Returns ``self`` for chaining."""
        self._raise_if_initialized("place()")
        self._place_rules = merge_place_rules(
            self._place_rules,
            (normalize_place_rule(locset, mechanisms),),
        )
        self._invalidate_discretization_cache()
        return self

    def bind_synapse_input(self, synapse: str, source, *, weight=1.0, transform=None) -> "Cell":
        """Bind an external presynaptic drive source to a runtime synapse.

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
        key = str(synapse)
        self._synapse_input_bindings.setdefault(key, []).append(
            (source, weight, transform)
        )
        return self

    # ------------------------------------------------------------------
    # Static discretization (valid in both phases)

    def _invalidate_discretization_cache(self) -> None:
        self._discretization_cache = None
        self._discretization_cache_key = None
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None

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
        if (
            self._discretization_cache is not None
            and self._discretization_cache_key == key
        ):
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
    def cv_tree(self) -> CVTree:
        return self._discretization.cv_tree

    @property
    def node_tree(self) -> NodeTree:
        return self._discretization.node_tree

    # ------------------------------------------------------------------
    # Phase transitions

    def init_state(self, batch_size=None) -> None:
        """Lower the declaration into runtime state and allocate V / spike.

        Raises
        ------
        RuntimeError
            If the cell is already initialized. Call :meth:`reset` first.
        """
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
        self.C = cv_value_vector(self, attr_name="cm")
        self.V_th = bridge.fill_like(self.varshape, self.V_th)

        v_initializer = (
            self._V_init if self._V_init is not None
            else cv_value_vector(self, attr_name="v")
        )
        if self._V_init is not None:
            v_initializer = bridge.fill_like(self.varshape, v_initializer)
        v_value = braintools.init.param(v_initializer, self.varshape)
        v_value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        self.V = DiffEqState(v_value)
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        self._current_time_state.value = 0.0 * u.ms

        point_V = self._cv_to_point_unchecked(self.V.value)
        for path, channel in self._runtime_objects_unchecked(
            IonChannel, allowed_hierarchy=(1, 1)
        ).items():
            args = self._runtime_node_phase_args(path, channel, point_V)
            channel.init_state(*args, batch_size=batch_size)

        self._runtime.axial_operator_np = np.asarray(
            build_cv_axial_operator(
                self,
                node_tree=self.node_tree,
                scheduling=self._node_scheduling_unchecked(algorithm="dhs"),
            ),
            dtype=np.float64,
        )
        self._runtime.axial_operator_cache = None
        self._axial_jax = self._get_axial_operator()
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
        self._raise_if_not_initialized("reset()")

        for name in ("_in_size", "_out_size", "ion_channels", "C"):
            if hasattr(self, name):
                delattr(self, name)

        # Restore scalar V_th (init_state overwrote it with a vector).
        self._V_th = self._V_th_declaration

        if hasattr(self, "V"):
            delattr(self, "V")
        if hasattr(self, "spike"):
            delattr(self, "spike")
        self._current_time_state.value = 0.0 * u.ms

        self._runtime = None
        self._runtime_cvs_cache = None
        self._runtime_nodes_cache = None
        self._axial_jax = None
        self._node_scheduling_cache.clear()

        self._morpho = self._declaration_morpho
        self._invalidate_discretization_cache()

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
                layout_ids=tuple(
                    int(layout.id) for layout in runtime.get_cv_layouts(int(cv.id))
                ),
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
                layout_ids=tuple(
                    int(layout.id) for layout in runtime.get_point_layouts(int(node.id))
                ),
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

    def node_scheduling(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        self._raise_if_not_initialized("node_scheduling()")
        return self._node_scheduling_unchecked(
            max_group_size=max_group_size, algorithm=algorithm
        )

    def _node_scheduling_unchecked(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
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
                f"Cell(root={self._morpho.root.name!r}, "
                f"n_cv={self.n_cv!r}, n_point={self.n_point!r}, initialized=True)"
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
            raise TypeError(
                f"Cell visualization expects LocsetExpr or LocsetMask, got {type(locset).__name__!s}."
            )

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
            raise TypeError(
                f"Cell visualization expects LocsetExpr or LocsetMask, got {type(locset).__name__!s}."
            )
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

    def _resolve_vis_node_values(self, value) -> tuple[object, str | None]:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"v", "voltage"}:
                return self._cv_to_node_values(self.V.value), "V"
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
                return self.V.value, "V"
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
        if hasattr(value, "to_decimal") and hasattr(value, "unit"):
            unit = value.unit
            raw = np.asarray(value.to_decimal(unit), dtype=float)
            if raw.ndim == 0:
                return u.Quantity(np.full((self.n_point,), float(raw), dtype=float), unit)
            if raw.ndim != 1:
                raise ValueError("Cell.vis_node(...) only supports scalar or 1-D value arrays.")
            if raw.shape[0] == self.n_point:
                return value
            if raw.shape[0] == self.n_cv:
                return self._cv_to_node_values(value)
            raise ValueError(
                f"Cell.vis_node(value=...) expects a point array of length {self.n_point} "
                f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
            )

        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            return np.full((self.n_point,), float(raw), dtype=float)
        if raw.ndim != 1:
            raise ValueError("Cell.vis_node(...) only supports scalar or 1-D value arrays.")
        if raw.shape[0] == self.n_point:
            return raw
        if raw.shape[0] == self.n_cv:
            return self._cv_to_node_values(raw)
        raise ValueError(
            f"Cell.vis_node(value=...) expects a point array of length {self.n_point} "
            f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
        )

    def _coerce_runtime_point_values_object(self, value):
        """Coerce one runtime field into unmasked point-space values."""
        if hasattr(value, "to_decimal") and hasattr(value, "unit"):
            unit = value.unit
            raw = np.asarray(value.to_decimal(unit), dtype=float)
            if raw.ndim == 0:
                return u.Quantity(
                    np.full((self.n_point,), float(raw), dtype=float),
                    unit,
                )
            if raw.ndim != 1:
                raise ValueError(
                    "Runtime point inspection only supports scalar or 1-D value arrays."
                )
            if raw.shape[0] == self.n_point:
                return value
            if raw.shape[0] == self.n_cv:
                return self._cv_to_point(value)
            raise ValueError(
                f"Runtime point inspection expects a point array of length {self.n_point} "
                f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
            )

        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            return np.full((self.n_point,), float(raw), dtype=float)
        if raw.ndim != 1:
            raise ValueError(
                "Runtime point inspection only supports scalar or 1-D value arrays."
            )
        if raw.shape[0] == self.n_point:
            return raw
        if raw.shape[0] == self.n_cv:
            return self._cv_to_point(raw)
        raise ValueError(
            f"Runtime point inspection expects a point array of length {self.n_point} "
            f"or a CV array of length {self.n_cv}, got length {raw.shape[0]}."
        )

    def _coerce_vis_cv_values_object(self, value):
        if hasattr(value, "to_decimal") and hasattr(value, "unit"):
            unit = value.unit
            raw = np.asarray(value.to_decimal(unit), dtype=float)
            if raw.ndim == 0:
                return u.Quantity(np.full((self.n_cv,), float(raw), dtype=float), unit)
            if raw.ndim != 1:
                raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D value arrays.")
            if raw.shape[0] == self.n_cv:
                return value
            if raw.shape[0] == self.n_point:
                return self._point_to_cv(value)
            raise ValueError(
                f"Cell.vis_cv(value=...) expects a CV array of length {self.n_cv} "
                f"or a point array of length {self.n_point}, got length {raw.shape[0]}."
            )

        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            return np.full((self.n_cv,), float(raw), dtype=float)
        if raw.ndim != 1:
            raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D value arrays.")
        if raw.shape[0] == self.n_cv:
            return raw
        if raw.shape[0] == self.n_point:
            return self._point_to_cv(raw)
        raise ValueError(
            f"Cell.vis_cv(value=...) expects a CV array of length {self.n_cv} "
            f"or a point array of length {self.n_point}, got length {raw.shape[0]}."
        )

    def _coerce_named_vis_node_values_object(self, value):
        if hasattr(value, "to_decimal") and hasattr(value, "unit"):
            unit = value.unit
            raw = np.asarray(value.to_decimal(unit), dtype=float)
            if raw.ndim == 0:
                return self._cv_to_node_values(u.Quantity(np.full((self.n_cv,), float(raw), dtype=float), unit))
            if raw.ndim != 1:
                raise ValueError("Cell.vis_node(...) only supports scalar or 1-D named value arrays.")
            if raw.shape[0] == self.n_point:
                return self._mask_non_midpoint_points(value)
            if raw.shape[0] == self.n_cv:
                return self._cv_to_node_values(value)
            raise ValueError("Cell.vis_node(...) cannot map the named value into point space.")

        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            return self._cv_to_node_values(np.full((self.n_cv,), float(raw), dtype=float))
        if raw.ndim != 1:
            raise ValueError("Cell.vis_node(...) only supports scalar or 1-D named value arrays.")
        if raw.shape[0] == self.n_point:
            return self._mask_non_midpoint_points(raw)
        if raw.shape[0] == self.n_cv:
            return self._cv_to_node_values(raw)
        raise ValueError("Cell.vis_node(...) cannot map the named value into point space.")

    def _coerce_named_vis_cv_values_object(self, value):
        if hasattr(value, "to_decimal") and hasattr(value, "unit"):
            unit = value.unit
            raw = np.asarray(value.to_decimal(unit), dtype=float)
            if raw.ndim == 0:
                return u.Quantity(np.full((self.n_cv,), float(raw), dtype=float), unit)
            if raw.ndim != 1:
                raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D named value arrays.")
            if raw.shape[0] == self.n_cv:
                return value
            if raw.shape[0] == self.n_point:
                return self._point_to_cv(self._mask_non_midpoint_points(value))
            raise ValueError("Cell.vis_cv(...) cannot map the named value into CV space.")

        raw = np.asarray(value, dtype=float)
        if raw.ndim == 0:
            return np.full((self.n_cv,), float(raw), dtype=float)
        if raw.ndim != 1:
            raise ValueError("Cell.vis_cv(...) only supports scalar or 1-D named value arrays.")
        if raw.shape[0] == self.n_cv:
            return raw
        if raw.shape[0] == self.n_point:
            return self._point_to_cv(self._mask_non_midpoint_points(raw))
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
                point_values[np.asarray(layout.point_index, dtype=np.int32)] = array[np.asarray(layout.point_index, dtype=np.int32)]
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
            point_values[np.asarray(layout.point_index, dtype=np.int32)] = array[np.asarray(layout.point_index, dtype=np.int32)]
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
            raise ValueError("Cell runtime is missing axial_operator_np.")

        operator = jnp.asarray(runtime.axial_operator_np, dtype=brainstate.environ.dftype()) * (u.ms ** -1)
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
        return (
            self.compute_membrane_derivative(V)
            + self.compute_axial_derivative(V)
        )

    def _top_level_ion_channel_nodes(self):
        return tuple(self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items())

    def _family_ion_nodes(self):
        return tuple(
            (path, node)
            for path, node in self._top_level_ion_channel_nodes()
            if isinstance(node, Ion)
        )

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
                infos = tuple([
                    parent._get_ion(root).pack_info()
                    for root in child.root_type.__args__
                ])
                getattr(child, hook_name)(point_V, *infos)

    def _update_ion_channels_by_integration(self, point_V):
        for path, node in self._top_level_ion_channel_nodes():
            if isinstance(node, IndependentIntegration):
                continue
            args = self._runtime_node_phase_args(path, node, point_V)
            ind_exp_euler_step(node, *args)

        for _, node in self._top_level_ion_channel_nodes():
            node.ind_update(point_V)

    def _update_ion_channel_families(self, point_V):
        ion_nodes = self._family_ion_nodes()
        channel_nodes = self._family_channel_nodes()

        dependent_ion_paths = [
            path
            for path, node in ion_nodes
            if not isinstance(node, IndependentIntegration)
        ]
        channel_paths = [path for path, _ in channel_nodes]

        # Family mode splits ion self states from channel states. This phase
        # advances dependent Ion states only; V and all channel states are
        # excluded explicitly so no child channel is integrated through Ion
        # recursion.
        self._integrate_selected_ion_self_states(
            ion_nodes,
            dependent_ion_paths,
            point_V,
            excluded_paths=[("V",), *channel_paths],
        )

        # Independent Ion states use their own updater, still without
        # recursing into child channels.
        for _, node in ion_nodes:
            if isinstance(node, IndependentIntegration):
                node.ind_update(point_V, recursive_child=False)

        # Channel nodes include Ion child channels, MixIons child channels,
        # and top-level channels. The owner path rebuilds the right ion args.
        for path, node in channel_nodes:
            if not self._is_independent_channel(node):
                target, args = self._channel_integration_target_and_args(
                    path,
                    node,
                    point_V,
                )
                ind_exp_euler_step(target, *args)

        # Independent channels finish through their own update rule.
        for path, node in channel_nodes:
            if not self._is_independent_channel(node):
                continue
            target, args = self._channel_integration_target_and_args(
                path,
                node,
                point_V,
            )
            target.ind_update(*args)

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
                infos = tuple([
                    owner._get_ion(root).pack_info()
                    for root in node.root_type.__args__
                ])
                return (point_V, *infos)
        return (point_V,)

    def _runtime_node_phase_args(self, path, node, point_V):
        if isinstance(node, RuntimeSynapse):
            layout_id = _layout_id_from_runtime_path(path)
            layout = self._runtime.layouts[layout_id]
            if layout.point_index is None:
                raise ValueError(f"Synapse layout {layout.id!r} is missing point_index.")
            return (point_V[..., layout.point_index],)
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
        for _, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not getattr(type(node), "uses_total_current", False):
                continue
            try:
                node._cached_total_current = node.current(point_V, include_external=True)
            except TypeError:
                node._cached_total_current = node.current(point_V)

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
        self._raise_if_not_initialized("update()")
        self._begin_step()
        spk = self._update_dynamics()
        self._prepare_next_synapse_inputs()
        return spk

    def _begin_step(self):
        self._raise_if_not_initialized("_begin_step()")
        point_V = self._cv_to_point(self.V.value)
        self._apply_runtime_synapse_events(point_V)

    def _update_dynamics(self):
        self._raise_if_not_initialized("_update_dynamics()")

        last_V = self.V.value
        if brainstate.environ.get("dt", None) is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")

        self.solver(self)

        self.clear_ion_total_current_cache()

        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    def _prepare_next_synapse_inputs(self):
        self._raise_if_not_initialized("_prepare_next_synapse_inputs()")
        point_V = self._cv_to_point(self.V.value)
        self._prepare_runtime_synapse_inputs(point_V)

    def _apply_runtime_synapse_events(self, point_V):
        self._raise_if_not_initialized("_apply_runtime_synapse_events()")
        for layout, synapse in self._runtime.iter_synapse_layouts():
            path = (f"layout_{layout.id}",)
            args = self._runtime_node_phase_args(path, synapse, point_V)
            synapse.apply_discrete_events(*args)

    def _prepare_runtime_synapse_inputs(self, point_V):
        """Bind this step's presynaptic drive to runtime synapses.

        ``SynapsePlacement`` is the ``cell.place(...)`` declaration, while
        ``RuntimeSynapse`` is the executable point mechanism. This method only
        prepares discrete input; synapse dynamics are integrated later by the
        active solver schedule.
        """
        _ = point_V
        self._raise_if_not_initialized("_prepare_runtime_synapse_inputs()")
        t = self._resolve_t()
        for layout, synapse in self._runtime.iter_synapse_layouts():
            declaration = self._runtime.get_layout_mechanism(layout.id)
            total_drive = self._runtime.get_state(layout.id, "pre_spike")
            netstim_drive = self._runtime.evaluate_synapse_netstim_drive(
                layout,
                t=t,
            )
            total_drive = total_drive + _coerce_drive_like(netstim_drive, total_drive)
            total_drive = total_drive + self._evaluate_bound_synapse_inputs(
                declaration,
                total_drive,
            )
            synapse.bind_pre_spike(total_drive)

    def _evaluate_bound_synapse_inputs(self, declaration, template):
        instance_name = declaration.instance_name
        bindings = self._synapse_input_bindings.get(instance_name, ())
        drive = u.math.zeros_like(template)
        for source, weight, transform in bindings:
            value = source() if callable(source) else source
            if transform is not None:
                value = transform(value)
            try:
                contribution = value * weight
                drive = drive + _coerce_drive_like(contribution, drive)
            except ValueError as exc:
                raise ValueError(
                    f"Bound synapse input for {instance_name!r} cannot broadcast "
                    f"from shape {getattr(value, 'shape', None)!r} to "
                    f"{getattr(template, 'shape', None)!r}."
                ) from exc
        return drive

    def _update_runtime_synapses(self, point_V):
        """Advance all runtime synapses through their own integrators."""
        self._prepare_runtime_synapse_inputs(point_V)
        self._integrate_runtime_synapse_dynamics(point_V)

    def _integrate_runtime_synapse_dynamics(self, point_V):
        """Advance runtime synapse continuous dynamics without changing input."""
        for path, node in self.runtime_objects(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, RuntimeSynapse):
                continue
            args = self._runtime_node_phase_args(path, node, point_V)
            ind_exp_euler_step(node, *args)

    def reset_state(self, batch_size=None) -> None:
        """Reseed ``V`` / ``spike`` / ``current_time`` without leaving INITIALIZED.

        Distinct from :meth:`reset`: ``reset_state`` is the in-phase
        brainstate lifecycle hook; ``reset`` tears down the runtime
        entirely and returns the cell to DECLARING.
        """
        self._raise_if_not_initialized("reset_state()")
        v_init = self._V_init
        if v_init is None:
            v_init = cv_value_vector(self, attr_name="v")
        else:
            v_init = bridge.fill_like(self.varshape, v_init)
        v_value = braintools.init.param(v_init, self.varshape)
        self.V.value = bridge.expand_with_batch_axis(v_value, batch_size, name="Cell.V")
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time_state.value = 0.0 * u.ms
        point_V = self._cv_to_point(self.V.value)
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

        def ensure_row(mechanism: object) -> int:
            row_key = mechanism_cell_key(mechanism)
            row_index = row_index_by_key.get(row_key)
            if row_index is not None:
                return row_index
            row_index = len(row_keys)
            row_keys.append(row_key)
            class_name, instance_name = row_key
            row_labels.append(
                class_name if class_name == instance_name else f"{instance_name}:{class_name}"
            )
            row_index_by_key[row_key] = row_index
            return row_index

        for cv in self.cvs:
            midpoint_point_id = int(node_tree.cv_to_mid_node_id[cv.id])
            for mechanism in cv.density_mech:
                row_key = mechanism_cell_key(mechanism)
                row_index = ensure_row(mechanism)
                layout_id = layout_id_by_signature[
                    ("density",) + mechanism_signature(mechanism)
                ]
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
                layout_id = layout_id_by_signature[
                    ("point",) + mechanism_signature(mechanism)
                ]
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
        if not self._initialized:
            self.init_state()
        return run_module.run(self, dt=dt, duration=duration)


# ----------------------------------------------------------------------
# Helpers


def _select_local_values(values, *, ids: tuple[int, ...]):
    """Return one localized item or a small indexed slice from an array-like."""
    if len(ids) == 1:
        return values[int(ids[0])]
    return values[list(int(idx) for idx in ids)]


def _coerce_drive_like(value, template):
    """Coerce dimensionless zero drives to a quantity template unit."""
    if isinstance(template, u.Quantity) and not isinstance(value, u.Quantity):
        return value * template.unit
    return value


def _resolve_solver(solver):
    if isinstance(solver, str):
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(
        f"solver must be str or callable, got {type(solver).__name__!s}."
    )


def _layout_id_from_runtime_path(path) -> int:
    if len(path) == 0:
        raise ValueError(f"Expected runtime layout path ending with 'layout_<id>', got {path!r}.")
    last = path[-1]
    if not isinstance(last, str) or not last.startswith("layout_"):
        raise ValueError(f"Expected runtime layout path ending with 'layout_<id>', got {path!r}.")
    return int(last.split("_", 1)[1])


def _normalize_pop_size(pop_size) -> tuple[int, ...]:
    """Normalize the public ``Cell(pop_size=...)`` argument.

    Parameters
    ----------
    pop_size : int, sequence of int, or None
        User-facing homogeneous population shape.

    Returns
    -------
    tuple of int
        Canonical population-shape tuple.

    Raises
    ------
    TypeError
        If ``pop_size`` is not an integer or sequence of integers.
    ValueError
        If any requested dimension is non-positive.
    """
    if pop_size in (None, ()):
        return ()
    if isinstance(pop_size, (int, np.integer)):
        if int(pop_size) <= 0:
            raise ValueError(f"pop_size must be > 0, got {pop_size!r}.")
        return (int(pop_size),)
    if isinstance(pop_size, (tuple, list)):
        if len(pop_size) == 0:
            return ()
        normalized = []
        for dim in pop_size:
            if not isinstance(dim, (int, np.integer)):
                raise TypeError(
                    f"pop_size entries must be integers, got {type(dim).__name__!s}."
                )
            dim = int(dim)
            if dim <= 0:
                raise ValueError(f"pop_size entries must be > 0, got {pop_size!r}.")
            normalized.append(dim)
        return tuple(normalized)
    raise TypeError(
        f"pop_size must be int or tuple/list of int, got {type(pop_size).__name__!s}."
    )


def _validate_ion_channel_update_order(value: str) -> str:
    # "family" is the ion-before-channel schedule; "integration" is the
    # previous schedule grouped by IndependentIntegration at the top level.
    if value not in {"family", "integration"}:
        raise ValueError(
            "ion_channel_update_order must be 'family' or 'integration', "
            f"got {value!r}."
        )
    return value
