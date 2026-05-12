"""``Cell`` — single-class multi-compartment neuron.

A ``Cell`` carries both the declaration (morphology, CV policy, paint /
place rules, solver, spike config) and the runtime (``V`` / ``spike`` /
``current_time`` brainstate states, point tree, axial operator,
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

from typing import Callable, Optional
from dataclasses import dataclass

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base import HHTypedNeuron, IonChannel, _cast_like
from braincell._base_channel import Channel as RuntimeChannel
from braincell._base_ion import Ion as RuntimeIon
from braincell._misc import is_traced_value
from braincell._typing import Initializer
from braincell._compute.table import (
    MechanismObjectCell,
    MechanismObjectTable,
    mechanism_cell_key,
)
from braincell._compute.topology import (
    _EPS_PARAM,
    _locate_branch_cv_by_x,
    build_point_scheduling,
    build_point_tree,
)
from braincell._compute.runtime import (
    CellRuntimeState,
    _is_root_level_runtime_node,
    build_placeholder_ions,
    clone_morpho,
    cv_value_vector,
    mechanism_signature,
)
from braincell._cv.base import build_cvs
from braincell._cv.lower import (
    PaintRule,
    PlaceRule,
    default_paint_rules,
    merge_paint_rules,
    merge_place_rules,
    normalize_paint_rules,
    normalize_place_rule,
)
from braincell._cv.policy import CVPerBranch, CVPolicy
from braincell.filter import LocsetExpr, LocsetMask, RegionExpr, RegionMask
from braincell.filter.helper import normalize_region_intervals
from braincell.morph.morphology import Morphology
from braincell.quad import get_integrator
from braincell.quad._staggered import build_cv_axial_operator, dhs_voltage_step
from braincell.quad._exp_euler import ind_exp_euler_step
from braincell.quad.protocol import DiffEqModule, DiffEqState, IndependentIntegration
from . import bridge, currents, probes, run as run_module

__all__ = ["Cell"]


@dataclass(frozen=True)
class AxialOperatorCache:
    float_dtype: jnp.dtype
    operator: object


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
        cv_policy: CVPolicy | None = None,
        V_th: u.Quantity = 0 * u.mV,
        V_init: Optional[Initializer] = None,
        spk_fun: Callable = braintools.surrogate.ReluGrad(),
        solver: str | Callable = "staggered",
        update_policy: str = "legacy",
        name: str | None = None,
    ) -> None:
        HHTypedNeuron.__init__(self, size=(1,), name=name)

        if not isinstance(morpho, Morphology):
            raise TypeError(
                f"Cell expects Morphology, got {type(morpho).__name__!s}."
            )

        self._declaration_morpho = morpho
        self._morpho = morpho

        self._cv_policy: CVPolicy = CVPerBranch() if cv_policy is None else cv_policy
        if not isinstance(self._cv_policy, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(self._cv_policy).__name__!s}."
            )

        self._paint_rules: tuple[PaintRule, ...] = default_paint_rules()
        self._place_rules: tuple[PlaceRule, ...] = ()

        self._V_th = V_th
        self._V_th_declaration = V_th
        self._V_init = V_init
        self._spk_fun = spk_fun
        self._name = name
        self._solver_name, self._solver_fn = _resolve_solver(solver)
        self._update_policy = _resolve_update_policy(update_policy)

        self._cvs_cache: tuple | None = None
        self._cvs_cache_key: object = None

        self._current_time_state = brainstate.ShortTermState(0.0 * u.ms)
        self._point_scheduling_cache: dict[tuple[str, int], object] = {}

        self._runtime: CellRuntimeState | None = None
        self._point_tree = None
        self._axial_jax = None

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
        return self._cv_policy

    @cv_policy.setter
    def cv_policy(self, value: CVPolicy) -> None:
        self._raise_if_initialized("assign cv_policy")
        if not isinstance(value, CVPolicy):
            raise TypeError(
                f"cv_policy must be CVPolicy, got {type(value).__name__!s}."
            )
        self._cv_policy = value
        self._cvs_cache = None

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

    @property
    def update_policy(self) -> str:
        return self._update_policy

    @update_policy.setter
    def update_policy(self, value: str) -> None:
        self._raise_if_initialized("assign update_policy")
        self._update_policy = _resolve_update_policy(value)

    # ------------------------------------------------------------------
    # Declaration mutators

    def paint(self, region: RegionExpr, *mechanisms) -> "Cell":
        """Paint mechanisms onto ``region``. Returns ``self`` for chaining."""
        self._raise_if_initialized("paint()")
        self._paint_rules = merge_paint_rules(
            self._paint_rules,
            normalize_paint_rules(region, mechanisms),
        )
        self._cvs_cache = None
        return self

    def place(self, locset: LocsetExpr, *mechanisms) -> "Cell":
        """Place point mechanisms at ``locset``. Returns ``self`` for chaining."""
        self._raise_if_initialized("place()")
        self._place_rules = merge_place_rules(
            self._place_rules,
            (normalize_place_rule(locset, mechanisms),),
        )
        self._cvs_cache = None
        return self

    # ------------------------------------------------------------------
    # CV preview (valid in both phases)

    @property
    def n_cv(self) -> int:
        return len(self.cvs)

    @property
    def cvs(self):
        key = (
            id(self._morpho),
            self._cv_policy,
            self._paint_rules,
            self._place_rules,
        )
        if self._cvs_cache is not None and self._cvs_cache_key == key:
            return self._cvs_cache

        cvs = build_cvs(
            self._morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
            place_rules=self._place_rules,
        )
        self._cvs_cache = cvs
        self._cvs_cache_key = key
        return cvs

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
        cvs = build_cvs(
            morpho,
            policy=self._cv_policy,
            paint_rules=self._paint_rules,
            place_rules=self._place_rules,
        )

        self._morpho = morpho
        self._cvs_cache = cvs
        self._cvs_cache_key = (
            id(self._morpho),
            self._cv_policy,
            self._paint_rules,
            self._place_rules,
        )

        self._point_tree = build_point_tree(morpho, cvs=cvs)
        self._runtime = CellRuntimeState.from_cell(self)

        # Save scalar V_th declaration before the vector overwrite below.
        self._V_th_declaration = self._V_th

        self._in_size = (self._runtime.n_cv,)
        self._out_size = (self._runtime.n_cv,)

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
        self.V = DiffEqState(braintools.init.param(v_initializer, self.varshape, batch_size))
        self.spike = brainstate.ShortTermState(self.get_spike(self.V.value, self.V.value))
        self._current_time_state.value = 0.0 * u.ms

        point_V = self._cv_to_point_unchecked(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.init_state(point_V, batch_size=batch_size)

        self._runtime.axial_operator_np = np.asarray(
            build_cv_axial_operator(
                self,
                point_tree=self._point_tree,
                scheduling=self._point_scheduling_unchecked(algorithm="dhs"),
            ),
            dtype=np.float64,
        )
        self._runtime.axial_operator_cache = None
        self._axial_jax = self._get_axial_operator()

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
        self._point_tree = None
        self._axial_jax = None
        self._point_scheduling_cache.clear()

        self._morpho = self._declaration_morpho
        self._cvs_cache = None
        self._cvs_cache_key = None

        self._initialized = False

    # ------------------------------------------------------------------
    # Topology (runtime-only)

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
        return ()

    @property
    def varshape(self) -> tuple[int, ...]:
        return (self.n_cv,)

    @property
    def n_compartment(self) -> int:
        return self.varshape[-1]

    def point_tree(self):
        if self._point_tree is None:
            raise RuntimeError("point_tree() requires init_state() first.")
        return self._point_tree

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
        """Visualize the runtime point tree with cell-aware inputs.

        ``Cell.vis_node(...)`` is the high-level point-tree entry point.
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
            self.point_tree(),
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

    def point_scheduling(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        self._raise_if_not_initialized("point_scheduling()")
        return self._point_scheduling_unchecked(
            max_group_size=max_group_size, algorithm=algorithm
        )

    def _point_scheduling_unchecked(self, *, max_group_size: int = 32, algorithm: str = "dhs"):
        key = (algorithm, int(max_group_size))
        cached = self._point_scheduling_cache.get(key)
        if cached is not None:
            return cached
        scheduling = build_point_scheduling(
            self._point_tree,
            max_group_size=max_group_size,
            algorithm=algorithm,
        )
        self._point_scheduling_cache[key] = scheduling
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
        point_tree = self.point_tree()
        if region is not None:
            for cv_id, fraction in self._cv_coverage_fractions(region).items():
                point_id = int(point_tree.cv_midpoint_point_id[int(cv_id)])
                fractions[point_id] = max(fractions.get(point_id, 0.0), float(fraction))
        if locset is not None:
            for cv_id in self._resolve_vis_locset_cv_ids(locset):
                point_id = int(point_tree.cv_midpoint_point_id[int(cv_id)])
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
        point_tree = self.point_tree()
        for cv in self.cvs:
            intervals = branch_intervals.get(int(cv.branch_id))
            if not intervals:
                continue
            midpoint = 0.5 * (float(cv.prox) + float(cv.dist))
            for prox, dist in intervals:
                lo, hi = (prox, dist) if prox <= dist else (dist, prox)
                if lo <= midpoint <= hi:
                    point_ids.add(int(point_tree.cv_midpoint_point_id[cv.id]))
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
        point_ids: set[int] = set()
        point_tree = self.point_tree()
        for cv_id in self._resolve_vis_locset_cv_ids(locset):
            point_ids.add(int(point_tree.cv_midpoint_point_id[int(cv_id)]))
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
        point_tree = self.point_tree()
        if hasattr(cv_values, "to_decimal") and hasattr(cv_values, "unit"):
            unit = cv_values.unit
            raw = np.asarray(cv_values.to_decimal(unit), dtype=float).reshape(-1)
            if raw.shape != (self.n_cv,):
                raise ValueError(f"_cv_to_node_values expects shape ({self.n_cv},), got {raw.shape!r}.")
            point_values = np.full((self.n_point,), np.nan, dtype=float)
            point_values[np.asarray(point_tree.cv_midpoint_point_id, dtype=np.int32)] = raw
            return u.Quantity(point_values, unit)
        raw = np.asarray(cv_values, dtype=float).reshape(-1)
        if raw.shape != (self.n_cv,):
            raise ValueError(f"_cv_to_node_values expects shape ({self.n_cv},), got {raw.shape!r}.")
        point_values = np.full((self.n_point,), np.nan, dtype=float)
        point_values[np.asarray(point_tree.cv_midpoint_point_id, dtype=np.int32)] = raw
        return point_values

    def _mask_non_midpoint_points(self, point_values):
        point_tree = self.point_tree()
        midpoint_ids = np.asarray(point_tree.cv_midpoint_point_id, dtype=np.int32)
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
        midpoint_by_cv = {cv_id: int(self.point_tree().cv_midpoint_point_id[cv_id]) for cv_id in source_cv_ids}
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

    def pre_integral(self, I_ext=0.0):
        self._raise_if_not_initialized("pre_integral()")
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.pre_integral(point_V)

    def compute_derivative(self, I_ext=0.0):
        self._raise_if_not_initialized("compute_derivative()")
        self.V.derivative = self.compute_voltage_derivative(self.V.value, I_ext)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.compute_derivative(point_V)

    def compute_membrane_derivative(self, V, I_ext=0.0):
        self._raise_if_not_initialized("compute_membrane_derivative()")
        t = self._resolve_t()
        I_total = currents.total_membrane_current(self, V_cv=V, I_ext=I_ext, t=t)
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

    def compute_voltage_derivative(self, V, I_ext=0.0):
        return (
            self.compute_membrane_derivative(V, I_ext)
            + self.compute_axial_derivative(V)
        )

    def post_integral(self, I_ext=0.0):
        self._raise_if_not_initialized("post_integral()")
        self.V.value = self.sum_delta_inputs(init=self.V.value)
        point_V = self._cv_to_point(self.V.value)
        for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
            if not isinstance(node, IndependentIntegration):
                node.post_integral(point_V)

    def update(self, I_ext=None):
        self._raise_if_not_initialized("update()")

        last_V = self.V.value
        if brainstate.environ.get("dt", None) is None:
            raise ValueError("Cell.update(...) requires brainstate.environ['dt'] to be set.")

        # ``legacy`` preserves the historical step ordering: run the cell
        # solver first, then let ``node.update(...)`` trigger any
        # independently-integrated submodules afterwards.
        #
        # ``family_phased`` uses an explicit NEURON-style family ordering:
        # voltage first, then all channels, then all ions. In that mode
        # ``IndependentIntegration`` only changes how a node advances within
        # its own family phase; it does not change the family ordering.
        if self._update_policy == "legacy":
            if I_ext is None:
                self.solver(self)
            else:
                self.solver(self, I_ext)

            point_V = self._cv_to_point(self.V.value)
            for _, node in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).items():
                node.update(point_V)
        else:
            self._update_family_phased(I_ext)

        spk = self.get_spike(last_V, self.V.value)
        self.spike.value = spk
        return spk

    def _update_family_phased(self, I_ext=None):
        """Run one family-phased update step.

        Order:

        1. Cache a step-start ion-current snapshot for current-driven ion
           dynamics.
        2. Advance voltage only, reusing the existing DHS voltage solve.
        3. Advance every channel (root-level and ion-bound) before any ion
           states.
        4. Advance every ion state after channel states.

        This avoids interleaving ion-bound channels with their owner ions in a
        single step. Current-driven ions consume the cached step-start current
        instead of re-evaluating a newer current after channel / voltage states
        have already moved.
        """
        if self.solver_name != "staggered":
            raise NotImplementedError(
                "update_policy='family_phased' currently requires solver='staggered'."
            )
        point_V_before = self._cv_to_point(self.V.value)
        self._cache_family_phased_currents(point_V_before)

        t = self._resolve_t()
        dt = brainstate.environ.get("dt")
        ext = 0.0 if I_ext is None else I_ext
        dhs_voltage_step(self, t, dt, ext)
        self.V.value = self.sum_delta_inputs(init=self.V.value)

        point_V_after = self._cv_to_point(self.V.value)
        root_nodes = list(self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values())
        ion_nodes = [node for node in root_nodes if isinstance(node, RuntimeIon)]
        root_channel_nodes = [node for node in root_nodes if isinstance(node, RuntimeChannel)]
        regular_ions = [node for node in ion_nodes if not isinstance(node, IndependentIntegration)]
        independent_ions = [node for node in ion_nodes if isinstance(node, IndependentIntegration)]
        regular_channel_calls = [(node, None) for node in root_channel_nodes if not isinstance(node, IndependentIntegration)]
        independent_channel_calls = [(node, None) for node in root_channel_nodes if isinstance(node, IndependentIntegration)]
        # Lift ion-bound channels into the same global channel phase as
        # root-level channels. This avoids channel/ion interleaving caused by
        # the object tree shape (Cell -> Ion -> Channel).
        for ion in ion_nodes:
            regular_child_channels, independent_child_channels = _split_ion_channels(ion)
            regular_channel_calls.extend((node, ion) for node in regular_child_channels)
            independent_channel_calls.extend((node, ion) for node in independent_child_channels)

        if regular_channel_calls:
            active_states, passive_states = _collect_channel_phase_states(regular_channel_calls)
            proxy = _PhaseTarget(
                active_states=active_states,
                passive_states=passive_states,
                pre_fn=lambda calls=regular_channel_calls: [
                    _channel_phase_call(node, "pre_integral", point_V_after, ion) for node, ion in calls
                ],
                compute_fn=lambda calls=regular_channel_calls: [
                    _channel_phase_call(node, "compute_derivative", point_V_after, ion) for node, ion in calls
                ],
                post_fn=lambda calls=regular_channel_calls: [
                    _channel_phase_call(node, "post_integral", point_V_after, ion) for node, ion in calls
                ],
            )
            # ``active_states`` are the channel states integrated in this
            # phase. ``passive_states`` are ion states such as ``Ci`` that
            # channel derivatives may read, but which must stay frozen until
            # the later ion phase.
            if proxy.has_states:
                ind_exp_euler_step(proxy, excluded_paths=proxy.excluded_paths)
            else:
                proxy.pre_integral()
                proxy.compute_derivative()
                proxy.post_integral()

        for node, ion in independent_channel_calls:
            if ion is None:
                node.update(point_V_after)
            else:
                node.update(point_V_after, ion.pack_info())

        # Ion phase advances only ion-owned species / concentration states.
        # Regular child channels are intentionally not re-advanced here because
        # they already ran in the channel phase above.
        for ion in regular_ions:
            ion_proxy = _PhaseTarget(
                active_states=_collect_ion_diffeq_states(ion),
                pre_fn=lambda ion=ion: ion._run_ion_hook("_ion_pre_integral_hook", point_V_after),
                compute_fn=lambda ion=ion: ion._run_ion_hook("_ion_compute_derivative_hook", point_V_after),
                post_fn=lambda ion=ion: ion._run_ion_hook("_ion_post_integral_hook", point_V_after),
            )
            if ion_proxy.has_states:
                ind_exp_euler_step(ion_proxy, excluded_paths=ion_proxy.excluded_paths)
            else:
                ion._run_ion_hook("_ion_post_integral_hook", point_V_after)

        # Independent ions still belong to the ion phase; they differ only in
        # the solver/substep schedule used inside this phase.
        for ion in independent_ions:
            ion_proxy = _PhaseTarget(
                active_states=_collect_ion_diffeq_states(ion),
                pre_fn=lambda ion=ion: ion._run_ion_hook("_ion_pre_integral_hook", point_V_after),
                compute_fn=lambda ion=ion: ion._run_ion_hook("_ion_compute_derivative_hook", point_V_after),
                post_fn=lambda ion=ion: ion._run_ion_hook("_ion_post_integral_hook", point_V_after),
            )
            if ion_proxy.has_states:
                _run_independent_phase_solver(ion, ion_proxy, point_V_after)
            else:
                ion._run_ion_hook("_ion_post_integral_hook", point_V_after)

    def _cache_family_phased_currents(self, point_V):
        runtime = self.runtime
        for ion_name, ion in runtime.ions.items():
            if getattr(type(ion), "uses_total_current", False):
                # Cache a step-start current snapshot for ion dynamics only.
                # The voltage phase still evaluates membrane current through the
                # existing path; this cache merely prevents ion dynamics from
                # re-reading a newer current after channel / voltage states have
                # already advanced.
                ion._cached_total_current = ion.current(point_V, include_external=True)
                ion._cached_total_current_policy = self._update_policy
                ion._cached_total_current_time = self._resolve_t()

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
        self.V.value = braintools.init.param(v_init, self.varshape, batch_size)
        self.spike.value = self.get_spike(self.V.value, self.V.value)
        self._current_time_state.value = 0.0 * u.ms
        point_V = self._cv_to_point(self.V.value)
        for channel in self.nodes(IonChannel, allowed_hierarchy=(1, 1)).values():
            channel.reset_state(point_V, batch_size=batch_size)

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
        point_tree = self._point_tree
        column_ids = tuple(range(len(point_tree.points)))

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
            midpoint_point_id = int(point_tree.cv_midpoint_point_id[cv.id])
            for target_label, mech_list in (("density", cv.density_mech), ("point", cv.point_mech)):
                for mechanism in mech_list:
                    row_key = mechanism_cell_key(mechanism)
                    row_index = ensure_row(mechanism)
                    layout_id = layout_id_by_signature[
                        (target_label,) + mechanism_signature(mechanism)
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


def _resolve_solver(solver):
    if isinstance(solver, str):
        return solver, get_integrator(solver)
    if callable(solver):
        return getattr(solver, "__name__", type(solver).__name__), solver
    raise TypeError(
        f"solver must be str or callable, got {type(solver).__name__!s}."
    )


def _resolve_update_policy(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"update_policy must be a string, got {type(value).__name__!s}."
        )
    if value not in {"legacy", "family_phased"}:
        raise ValueError(
            f"update_policy must be one of ('legacy', 'family_phased'), got {value!r}."
        )
    return value


class _PhaseTarget(brainstate.nn.Module, DiffEqModule):
    """Minimal solver target for one family phase.

    ``active_states`` are the states this phase may integrate.
    ``passive_states`` are visible to derivative evaluation but excluded from
    integration so cross-family reads do not accidentally advance the wrong
    states.
    """

    def __init__(
        self,
        *,
        active_states: dict[str, DiffEqState],
        pre_fn: Callable[[], None],
        compute_fn: Callable[[], None],
        post_fn: Callable[[], None],
        passive_states: dict[str, DiffEqState] | None = None,
    ):
        super().__init__()
        self._pre_fn = pre_fn
        self._compute_fn = compute_fn
        self._post_fn = post_fn
        passive_states = {} if passive_states is None else passive_states
        self.has_states = bool(active_states)
        self.excluded_paths = tuple((name,) for name in passive_states.keys())
        for name, state in active_states.items():
            setattr(self, name, state)
        for name, state in passive_states.items():
            setattr(self, name, state)

    def pre_integral(self, *args, **kwargs):
        self._pre_fn()

    def compute_derivative(self, *args, **kwargs):
        self._compute_fn()

    def post_integral(self, *args, **kwargs):
        self._post_fn()


def _collect_diffeq_states(nodes) -> dict[str, DiffEqState]:
    states: dict[str, DiffEqState] = {}
    seen: set[int] = set()
    for node in nodes:
        for _, state in brainstate.graph.states(node).items():
            if not isinstance(state, DiffEqState):
                continue
            if id(state) in seen:
                continue
            seen.add(id(state))
            states[f"s{len(states)}"] = state
    return states


def _collect_channel_phase_states(calls) -> tuple[dict[str, DiffEqState], dict[str, DiffEqState]]:
    active: dict[str, DiffEqState] = {}
    passive: dict[str, DiffEqState] = {}
    seen_active: set[int] = set()
    seen_passive: set[int] = set()
    for node, ion in calls:
        for _, state in brainstate.graph.states(node).items():
            if not isinstance(state, DiffEqState) or id(state) in seen_active:
                continue
            seen_active.add(id(state))
            active[f"a{len(active)}"] = state
        if ion is None:
            continue
        # Channel derivatives may depend on ion states such as ``Ci``. Expose
        # those states as passive reads so the solver can trace them without
        # advancing them during the channel phase.
        for _, state in brainstate.graph.states(ion).items():
            if not isinstance(state, DiffEqState) or id(state) in seen_active or id(state) in seen_passive:
                continue
            seen_passive.add(id(state))
            passive[f"p{len(passive)}"] = state
    return active, passive


def _collect_ion_diffeq_states(ion: RuntimeIon) -> dict[str, DiffEqState]:
    states: dict[str, DiffEqState] = {}
    for path, state in brainstate.graph.states(ion).items():
        if len(path) != 1 or not isinstance(state, DiffEqState):
            continue
        states[str(path[0])] = state
    return states


def _split_ion_channels(ion: RuntimeIon):
    regular = []
    independent = []
    for node in brainstate.graph.nodes(ion, RuntimeChannel, allowed_hierarchy=(1, 1)).values():
        if isinstance(node, IndependentIntegration):
            independent.append(node)
        else:
            regular.append(node)
    return regular, independent


def _run_independent_phase_solver(owner, proxy: _PhaseTarget, point_V):
    substeps = int(getattr(owner, "substeps", 1))
    if substeps < 1:
        raise ValueError("substeps must be at least 1.")
    with brainstate.environ.context(dt=brainstate.environ.get_dt() / substeps):
        brainstate.transform.for_loop(
            lambda i: owner.solver(proxy, point_V),
            u.math.arange(substeps),
        )


def _channel_phase_call(node, method_name: str, point_V, ion: RuntimeIon | None):
    method = getattr(node, method_name)
    if ion is None:
        return method(point_V)
    return method(point_V, ion.pack_info())
