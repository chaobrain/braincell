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

"""Static discretization types and the base build pipeline.

This module defines the immutable declaration-time objects that sit
between a :class:`~braincell.morph.morphology.Morphology` plus user
declarations and the runtime structures assembled later by
``braincell._compute``.

Two ideas shape the public surface here:

1. ``CV`` and ``CVTree`` are the membrane-oriented physical view of the
   discretization.
2. ``Node`` and ``NodeTree`` are a point-space view derived from those
   CV records for later point-tree assembly and runtime lowering.

The single formal build entry point is :func:`build_discretization`,
which returns a frozen :class:`Discretization` bundling both views.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import brainunit as u
import numpy as np

from braincell.filter import RegionMask
from braincell.mech import CVContext, Density, Point

if TYPE_CHECKING:
    from braincell._discretization.mechanism import PaintRule, PlaceRule
    from braincell._discretization.policy import CVPolicy
    from braincell.morph.morphology import Morphology

Position = Literal["prox", "mid", "dist"]
Half = Literal["prox", "dist"]
NodeKind = Literal["mid", "boundary"]

__all__ = [
    "CV",
    "CVEdge",
    "CVPointMechanism",
    "CVTree",
    "Discretization",
    "Half",
    "Node",
    "NodeEdge",
    "NodeEdgeRole",
    "NodeKind",
    "NodeRole",
    "NodeTree",
    "PointPlacement",
    "Position",
    "build_discretization",
    "locate_cv_on_branch",
]

_EPS_PARAM = 1e-9


@dataclass(frozen=True)
class CVPointMechanism:
    """Unresolved point placement attached to one CV-local position.

    Attributes
    ----------
    placement_id : int
        Stable declaration-order identity assigned during mechanism lowering.
    position : {"prox", "mid", "dist"}
        Which CV-local reference position owns the mechanism.
    mechanism : Point
        Point-mechanism declaration attached at that position.
    display_name : str
        Human-readable continuous morphology location.
    branch_id : int
        Morphology branch on which the placement was declared.
    branch_x : float
        Original normalized coordinate on that branch.
    """

    position: Position
    mechanism: Point
    placement_id: int = -1
    display_name: str = ""
    branch_id: int = -1
    branch_x: float = 0.5
    population_index: int | None = None


@dataclass(frozen=True)
class PointPlacement:
    """One independent point-mechanism placement.

    The original continuous morphology location remains authoritative while
    ``cv_id`` and ``point_id`` record its resolved membrane and electrical
    locations, respectively. Multiple placements may share one ``point_id``.

    Parameters
    ----------
    id : int
        Stable placement index in declaration order.
    mechanism : Point
        Reusable point-mechanism specification for this instance.
    display_name : str
        Human-readable continuous location name.
    branch_id, branch_name, branch_type : int, str, str
        Morphology branch identity retained from the location declaration.
    branch_x : float
        Normalized continuous coordinate on the declared branch.
    cv_id : int
        Unique owning control volume.
    cv_position : {"prox", "mid", "dist"}
        CV-local role used to resolve the electrical node.
    point_id : int
        Electrical node receiving this placement's current.
    """

    id: int
    mechanism: Point
    display_name: str
    branch_id: int
    branch_name: str
    branch_type: str
    branch_x: float
    cv_id: int
    cv_position: Position
    point_id: int
    population_index: int | None = None


@dataclass(frozen=True)
class CV:
    """Immutable per-control-volume record exposed to users.

    Geometry, cable properties, and attached mechanisms are frozen into this
    dataclass by :func:`build_discretization`. ``CV`` is the physical truth
    layer of the declaration-time discretization; node-space structures are
    thin views built from these records.

    Attributes
    ----------
    id : int
        Stable CV index within the discretization.
    branch_id : int
        Index of the morphology branch that owns this CV.
    branch_type : str
        Morphology branch type, for example ``"soma"`` or
        ``"basal_dendrite"``.
    prox : float
        Proximal CV boundary in normalized branch coordinates.
    dist : float
        Distal CV boundary in normalized branch coordinates.
    parent_cv : int or None
        Parent CV id in the CV tree. ``None`` marks the root CV.
    children_cv : tuple of int
        Child CV ids in the CV tree.
    length : brainunit.Quantity
        CV cable length.
    area : brainunit.Quantity
        CV membrane lateral area.
    cm : brainunit.Quantity
        Membrane capacitance density active on this CV.
    ra : brainunit.Quantity
        Axial resistivity active on this CV.
    v : brainunit.Quantity
        Resting potential declaration carried by this CV.
    temp : brainunit.Quantity
        Temperature declaration carried by this CV.
    r_axial : brainunit.Quantity
        End-to-end axial resistance of this CV.
    r_axial_prox : brainunit.Quantity
        Axial resistance from the CV midpoint to the proximal side.
    r_axial_dist : brainunit.Quantity
        Axial resistance from the CV midpoint to the distal side.
    radius_prox : brainunit.Quantity
        Radius at the proximal CV boundary.
    radius_mid : brainunit.Quantity
        Radius at the CV midpoint.
    diam_arc_mean : brainunit.Quantity
        Arc-length-weighted mean diameter across the CV.
    radius_dist : brainunit.Quantity
        Radius at the distal CV boundary.
    density_mech : tuple of Density
        Density-like mechanism declarations active on this CV.
    point_mech : tuple of Point
        Point-mechanism declarations that resolve to this CV, without
        preserving local-role provenance.
    point_mech_roles : tuple of CVPointMechanism
        Point-mechanism declarations together with the CV-local role that
        owns each one.

    Notes
    -----
    ``CV`` is intentionally static. It stores declaration-time geometry
    and mechanism specifications only; runtime ion/channel instances and
    state buffers live later in ``braincell._compute``.
    """

    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length: u.Quantity
    area: u.Quantity
    cm: u.Quantity
    ra: u.Quantity
    v: u.Quantity
    temp: u.Quantity
    r_axial: u.Quantity
    r_axial_prox: u.Quantity
    r_axial_dist: u.Quantity
    radius_prox: u.Quantity
    radius_mid: u.Quantity
    diam_arc_mean: u.Quantity
    radius_dist: u.Quantity
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]
    point_mech_roles: tuple[CVPointMechanism, ...] = ()

    @property
    def region(self) -> RegionMask:
        """Return a one-interval region covering this CV.

        Returns
        -------
        RegionMask
            Region spanning ``(branch_id, prox, dist)``.
        """
        return RegionMask(((self.branch_id, self.prox, self.dist),))

    @property
    def diam_mid(self) -> u.Quantity:
        """Return the diameter at the CV midpoint.

        Returns
        -------
        brainunit.Quantity
            Midpoint diameter, equal to ``2 * radius_mid``.
        """
        return 2.0 * self.radius_mid


@dataclass(frozen=True)
class CVEdge:
    """Directed parent/child relation in the CV tree.

    Attributes
    ----------
    parent_cv_id : int
        Upstream CV id.
    child_cv_id : int
        Downstream CV id.
    """

    parent_cv_id: int
    child_cv_id: int


@dataclass(frozen=True)
class CVTree:
    """Graph-level metadata for one CV discretization.

    Attributes
    ----------
    cvs : tuple of CV
        All CV records in stable id order.
    edges : tuple of CVEdge
        Directed CV-tree edges.
    root_cv_id : int
        Id of the unique root CV.
    branch_to_cv_ids : tuple of tuple of int
        For each morphology branch, the ordered CV ids that tile that
        branch.
    """

    cvs: tuple[CV, ...]
    edges: tuple[CVEdge, ...]
    root_cv_id: int
    branch_to_cv_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class NodeRole:
    """Reference from a node back to one CV-local position.

    Attributes
    ----------
    cv_id : int
        Source CV id.
    position : {"prox", "mid", "dist"}
        CV-local position that collapsed into this node.
    """

    cv_id: int
    position: Position


@dataclass(frozen=True)
class Node:
    """Point-space thin view of one or more CV-local positions.

    Attributes
    ----------
    id : int
        Stable node index within the node tree.
    kind : {"mid", "boundary"}
        Whether this node is a pure midpoint node or a boundary node
        formed by collapsing one or more CV-local boundary roles.
    roles : tuple of NodeRole
        Provenance back to the CV-local positions that define this node.
    density_mech : tuple of Density
        Density declarations visible at this node. In the current
        lowering model these are carried by midpoint nodes.
    point_mech : tuple of Point
        Point-mechanism declarations assigned to this node.
    """

    id: int
    kind: NodeKind
    roles: tuple[NodeRole, ...]
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]

    @property
    def source_cv_ids(self) -> tuple[int, ...]:
        """Return sorted source CV ids contributing to this node.

        Returns
        -------
        tuple of int
            Unique source CV ids in ascending order.
        """
        return tuple(sorted({int(role.cv_id) for role in self.roles}))


@dataclass(frozen=True)
class NodeEdgeRole:
    """One CV half-edge contribution to a node-space edge.

    Attributes
    ----------
    cv_id : int
        Source CV id.
    half : {"prox", "dist"}
        Which half of the source CV contributes to the edge.
    r_axial : brainunit.Quantity
        Axial resistance carried by that half-edge.
    """

    cv_id: int
    half: Half
    r_axial: u.Quantity


@dataclass(frozen=True)
class NodeEdge:
    """Directed edge between two nodes with CV-half provenance.

    Attributes
    ----------
    id : int
        Stable edge index within the node tree.
    parent_node_id : int
        Parent node id.
    child_node_id : int
        Child node id.
    roles : tuple of NodeEdgeRole
        One or more CV-half contributions collapsed into this node edge.
    """

    id: int
    parent_node_id: int
    child_node_id: int
    roles: tuple[NodeEdgeRole, ...]


@dataclass(frozen=True)
class NodeTree:
    """Point-space view of a cell's declaration-time discretization.

    Attributes
    ----------
    nodes : tuple of Node
        All point-space nodes in stable id order.
    edges : tuple of NodeEdge
        Directed node-tree edges.
    root_node_id : int
        Id of the unique root node.
    cv_to_mid_node_id : numpy.ndarray
        ``(n_cv,)`` array mapping each CV id to its midpoint node id.
    branch_endpoint_node_id : numpy.ndarray
        ``(n_branch, 2)`` array mapping each branch to its proximal and
        distal endpoint node ids.

    Notes
    -----
    The node tree is still a static declaration-time object. It is the
    point-space structural view later consumed by scheduling and runtime
    lowering, but it does not itself contain mutable runtime state.
    """

    nodes: tuple[Node, ...]
    edges: tuple[NodeEdge, ...]
    root_node_id: int
    cv_to_mid_node_id: np.ndarray
    branch_endpoint_node_id: np.ndarray

    def __repr__(self) -> str:
        return f"NodeTree(n_nodes={len(self.nodes)!r}, n_edges={len(self.edges)!r}, root_node_id={self.root_node_id!r})"

    def __str__(self) -> str:
        return (
            f"{'-' * 35}\n"
            f"{'n_nodes':<14} | {len(self.nodes)}\n"
            f"{'n_edges':<14} | {len(self.edges)}\n"
            f"{'root_node_id':<14} | {self.root_node_id}\n"
            f"{'-' * 35}\n"
        )


@dataclass(frozen=True)
class Discretization:
    """Static discretization snapshot used before and after runtime init.

    Attributes
    ----------
    cv_tree : CVTree
        Membrane-oriented CV tree.
    node_tree : NodeTree
        Point-space node tree derived from ``cv_tree``.
    cv_contexts : tuple of CVContext
        Read-only geometry and path-distance context for each CV.
    point_placements : tuple of PointPlacement
        Independent point-mechanism instances with continuous and discrete
        location provenance.

    Notes
    -----
    This object is the stable handoff point between declaration-time
    lowering and runtime compilation. ``Cell`` keeps one cached
    ``Discretization`` and exposes convenient projections such as
    ``disc.cvs`` and ``disc.nodes``.
    """

    cv_tree: CVTree
    node_tree: NodeTree
    cv_contexts: tuple[CVContext, ...]
    point_placements: tuple[PointPlacement, ...] = ()

    @property
    def cvs(self) -> tuple[CV, ...]:
        """Return the flat CV tuple carried by :attr:`cv_tree`.

        Returns
        -------
        tuple of CV
            CV records in stable id order.
        """
        return self.cv_tree.cvs

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Return the flat node tuple carried by :attr:`node_tree`.

        Returns
        -------
        tuple of Node
            Node records in stable id order.
        """
        return self.node_tree.nodes

    def locate_cv(self, *, branch_id: int, x: float) -> int:
        """Return the CV that owns one normalized branch location."""
        ids = self.cv_tree.branch_to_cv_ids[int(branch_id)]
        return locate_cv_on_branch(
            ids,
            self.cvs,
            x=float(x),
        )


def locate_cv_on_branch(
    ids: tuple[int, ...],
    cvs: tuple[CV, ...],
    *,
    x: float,
    epsilon: float = _EPS_PARAM,
) -> int:
    """Return the CV owning a normalized coordinate on one branch."""
    if not ids:
        raise ValueError("Cannot locate a CV on an empty branch tiling.")
    if x <= 0.0 + epsilon:
        return int(ids[0])
    if x >= 1.0 - epsilon:
        return int(ids[-1])
    for cv_id in ids:
        cv = cvs[cv_id]
        if float(cv.prox) - epsilon <= x < float(cv.dist) - epsilon:
            return int(cv_id)
    raise ValueError(
        f"x={x!r} lies in no CV interval among ids {list(ids)!r}."
    )


def build_discretization(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]" = (),
    place_rules: "tuple[PlaceRule, ...]" = (),
) -> Discretization:
    """Build one static discretization snapshot from declarations.

    Parameters
    ----------
    morpho : Morphology
        Morphology to discretize.
    policy : CVPolicy
        Policy that resolves branch-wise CV bounds.
    paint_rules : tuple of PaintRule, optional
        Normalized region-based mechanism declarations. The default is an
        empty tuple.
    place_rules : tuple of PlaceRule, optional
        Normalized locset-based point declarations. The default is an
        empty tuple.

    Returns
    -------
    Discretization
        Frozen snapshot containing both the CV tree and the derived node
        tree.

    Notes
    -----
    This is the single formal entry point for declaration-time lowering.
    All higher-level previews such as ``cell.cvs`` and all runtime
    initialization paths should route through the resulting
    :class:`Discretization`.
    """
    cv_tree, node_tree, cv_contexts, point_placements = _build_discretization_parts(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )
    return Discretization(
        cv_tree=cv_tree,
        node_tree=node_tree,
        cv_contexts=cv_contexts,
        point_placements=point_placements,
    )


def _build_discretization_parts(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]",
    place_rules: "tuple[PlaceRule, ...]",
) -> tuple[CVTree, NodeTree, tuple[CVContext, ...], tuple[PointPlacement, ...]]:
    """Assemble both static discretization views in one pass."""
    from .geometry import build_cv_geometry
    from .context import build_cv_contexts
    from .mechanism import build_cv_mechanisms
    from .node_build import build_node_tree_from_cvs
    from .policy import CVPolicy

    if not isinstance(policy, CVPolicy):
        raise TypeError(f"build_discretization(...) expects a CVPolicy, got {type(policy).__name__!s}.")
    bounds = policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)
    geometry = build_cv_geometry(morpho, bounds)
    cv_contexts = build_cv_contexts(morpho, geometry.geos)
    buckets = build_cv_mechanisms(
        morpho,
        geometry,
        paint_rules=paint_rules,
        place_rules=place_rules,
        cv_contexts=cv_contexts,
    )
    cvs = tuple(_assemble_cv(geo, bucket) for geo, bucket in zip(geometry.geos, buckets))
    cv_tree = _build_cv_tree(
        cvs=cvs,
        branch_to_cv_ids=geometry.branch_to_cv_ids,
    )
    node_tree = build_node_tree_from_cvs(morpho, cvs=cv_tree.cvs)
    point_placements = _build_point_placements(
        morpho,
        cvs=cv_tree.cvs,
        node_tree=node_tree,
    )
    return cv_tree, node_tree, cv_contexts, point_placements


def _build_point_placements(
    morpho: "Morphology",
    *,
    cvs: tuple[CV, ...],
    node_tree: NodeTree,
) -> tuple[PointPlacement, ...]:
    """Resolve CV-local placement roles onto electrical point ids."""
    role_to_point_id = {
        (int(role.cv_id), str(role.position)): int(node.id)
        for node in node_tree.nodes
        for role in node.roles
    }
    placements: list[PointPlacement] = []
    for cv in cvs:
        midpoint_id = int(node_tree.cv_to_mid_node_id[cv.id])
        for placement in cv.point_mech_roles:
            point_id = role_to_point_id.get(
                (int(cv.id), str(placement.position)),
                midpoint_id,
            )
            branch = morpho.branch(index=int(placement.branch_id))
            placements.append(
                PointPlacement(
                    id=int(placement.placement_id),
                    mechanism=placement.mechanism,
                    display_name=placement.display_name,
                    branch_id=int(placement.branch_id),
                    branch_name=str(branch.name),
                    branch_type=str(branch.type),
                    branch_x=float(placement.branch_x),
                    cv_id=int(cv.id),
                    cv_position=placement.position,
                    point_id=point_id,
                    population_index=placement.population_index,
                )
            )
    return tuple(sorted(placements, key=lambda item: item.id))


def _build_cv_tree(
    *,
    cvs: tuple[CV, ...],
    branch_to_cv_ids: tuple[tuple[int, ...], ...],
) -> CVTree:
    """Assemble graph metadata around an ordered CV tuple."""
    root_candidates = [cv.id for cv in cvs if cv.parent_cv is None]
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly one root CV, got {root_candidates!r}.")
    edges = tuple(
        CVEdge(parent_cv_id=int(cv.parent_cv), child_cv_id=int(cv.id)) for cv in cvs if cv.parent_cv is not None
    )
    return CVTree(
        cvs=cvs,
        edges=edges,
        root_cv_id=int(root_candidates[0]),
        branch_to_cv_ids=branch_to_cv_ids,
    )


def _assemble_cv(geo, bucket) -> CV:
    """Materialize one public ``CV`` from geometry and mechanism buckets."""
    cable = bucket.cable
    ra = cable.axial_resistivity
    ra_ohm_cm = float(np.asarray(ra.to_decimal(u.ohm * u.cm), dtype=float))
    return CV(
        id=geo.id,
        branch_id=geo.branch_id,
        branch_type=geo.branch_type,
        prox=geo.prox,
        dist=geo.dist,
        parent_cv=geo.parent_cv,
        children_cv=geo.children_cv,
        length=u.Quantity(geo.length_um, u.um),
        area=u.Quantity(geo.lateral_area_um2, u.um**2),
        cm=cable.membrane_capacitance,
        ra=cable.axial_resistivity,
        v=cable.resting_potential,
        temp=cable.temperature,
        r_axial=u.Quantity(ra_ohm_cm * geo.axial_factor_total_per_cm, u.ohm),
        r_axial_prox=u.Quantity(ra_ohm_cm * geo.axial_factor_prox_per_cm, u.ohm),
        r_axial_dist=u.Quantity(ra_ohm_cm * geo.axial_factor_dist_per_cm, u.ohm),
        radius_prox=u.Quantity(geo.r_prox_um, u.um),
        radius_mid=u.Quantity(geo.r_mid_um, u.um),
        diam_arc_mean=u.Quantity(geo.diam_arc_mean_um, u.um),
        radius_dist=u.Quantity(geo.r_dist_um, u.um),
        density_mech=tuple(bucket.density_by_key.values()),
        point_mech=tuple(bucket.points),
        point_mech_roles=tuple(bucket.point_roles),
    )
