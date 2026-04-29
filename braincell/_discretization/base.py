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

"""Static discretization types and the base build pipeline."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import brainunit as u
import numpy as np

from braincell.filter import RegionMask
from braincell.mech import Density, Point

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
    "Position",
    "build_discretization",
]


@dataclass(frozen=True)
class CVPointMechanism:
    """Point mechanism attached to one CV-local position."""

    position: Position
    mechanism: Point


@dataclass(frozen=True)
class CV:
    """Immutable per-control-volume record exposed to users.

    Geometry, cable properties, and attached mechanisms are frozen into this
    dataclass by :func:`build_discretization`. ``CV`` is the physical truth
    layer of the declaration-time discretization; node-space structures are
    thin views built from these records.
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
    radius_dist: u.Quantity
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]
    point_mech_roles: tuple[CVPointMechanism, ...] = ()

    @property
    def region(self) -> RegionMask:
        """Return the ``RegionMask`` covering this CV's branch interval."""
        return RegionMask(((self.branch_id, self.prox, self.dist),))

    @property
    def diam_mid(self) -> u.Quantity:
        """Diameter at the CV midpoint."""
        return 2.0 * self.radius_mid


@dataclass(frozen=True)
class CVEdge:
    """Directed parent/child relation in the CV tree."""

    parent_cv_id: int
    child_cv_id: int


@dataclass(frozen=True)
class CVTree:
    """Graph-level metadata for one CV discretization."""

    cvs: tuple[CV, ...]
    edges: tuple[CVEdge, ...]
    root_cv_id: int
    branch_to_cv_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class NodeRole:
    """Reference from a node back to one CV-local position."""

    cv_id: int
    position: Position


@dataclass(frozen=True)
class Node:
    """Point-space thin view of one or more CV-local positions."""

    id: int
    kind: NodeKind
    roles: tuple[NodeRole, ...]
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]

    @property
    def source_cv_ids(self) -> tuple[int, ...]:
        """CV ids whose local roles collapsed into this node."""
        return tuple(sorted({int(role.cv_id) for role in self.roles}))


@dataclass(frozen=True)
class NodeEdgeRole:
    """One CV half-edge contribution to a node-space edge."""

    cv_id: int
    half: Half
    r_axial: u.Quantity


@dataclass(frozen=True)
class NodeEdge:
    """Directed edge between two nodes with CV-half provenance."""

    id: int
    parent_node_id: int
    child_node_id: int
    roles: tuple[NodeEdgeRole, ...]


@dataclass(frozen=True)
class NodeTree:
    """Point-space view of a cell's declaration-time discretization."""

    nodes: tuple[Node, ...]
    edges: tuple[NodeEdge, ...]
    root_node_id: int
    cv_to_mid_node_id: np.ndarray
    branch_endpoint_node_id: np.ndarray

    def __repr__(self) -> str:
        return (
            f"NodeTree(n_nodes={len(self.nodes)!r}, n_edges={len(self.edges)!r}, "
            f"root_node_id={self.root_node_id!r})"
        )

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
    """Static discretization snapshot used before and after runtime init."""

    cv_tree: CVTree
    node_tree: NodeTree

    @property
    def cvs(self) -> tuple[CV, ...]:
        """Flat CV records carried by :attr:`cv_tree`."""
        return self.cv_tree.cvs

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Flat node records carried by :attr:`node_tree`."""
        return self.node_tree.nodes


def build_discretization(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]" = (),
    place_rules: "tuple[PlaceRule, ...]" = (),
) -> Discretization:
    """Lower a declaration into one static discretization snapshot."""
    cv_tree, node_tree = _build_discretization_parts(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )
    return Discretization(cv_tree=cv_tree, node_tree=node_tree)


def _build_discretization_parts(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]",
    place_rules: "tuple[PlaceRule, ...]",
) -> tuple[CVTree, NodeTree]:
    from .geometry import build_cv_geometry
    from .mechanism import build_cv_mechanisms
    from .node_build import build_node_tree_from_cvs
    from .policy import CVPolicy

    if not isinstance(policy, CVPolicy):
        raise TypeError(
            f"build_discretization(...) expects a CVPolicy, got {type(policy).__name__!s}."
        )
    bounds = policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)
    geometry = build_cv_geometry(morpho, bounds)
    buckets = build_cv_mechanisms(
        morpho,
        geometry,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )
    cvs = tuple(
        _assemble_cv(geo, bucket)
        for geo, bucket in zip(geometry.geos, buckets)
    )
    cv_tree = _build_cv_tree(
        cvs=cvs,
        branch_to_cv_ids=geometry.branch_to_cv_ids,
    )
    node_tree = build_node_tree_from_cvs(morpho, cvs=cv_tree.cvs)
    return cv_tree, node_tree


def _build_cv_tree(
    *,
    cvs: tuple[CV, ...],
    branch_to_cv_ids: tuple[tuple[int, ...], ...],
) -> CVTree:
    root_candidates = [cv.id for cv in cvs if cv.parent_cv is None]
    if len(root_candidates) != 1:
        raise ValueError(
            f"Expected exactly one root CV, got {root_candidates!r}."
        )
    edges = tuple(
        CVEdge(parent_cv_id=int(cv.parent_cv), child_cv_id=int(cv.id))
        for cv in cvs
        if cv.parent_cv is not None
    )
    return CVTree(
        cvs=cvs,
        edges=edges,
        root_cv_id=int(root_candidates[0]),
        branch_to_cv_ids=branch_to_cv_ids,
    )


def _assemble_cv(geo, bucket) -> CV:
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
        area=u.Quantity(geo.lateral_area_um2, u.um ** 2),
        cm=cable.membrane_capacitance,
        ra=cable.axial_resistivity,
        v=cable.resting_potential,
        temp=cable.temperature,
        r_axial=u.Quantity(ra_ohm_cm * geo.axial_factor_total_per_cm, u.ohm),
        r_axial_prox=u.Quantity(ra_ohm_cm * geo.axial_factor_prox_per_cm, u.ohm),
        r_axial_dist=u.Quantity(ra_ohm_cm * geo.axial_factor_dist_per_cm, u.ohm),
        radius_prox=u.Quantity(geo.r_prox_um, u.um),
        radius_mid=u.Quantity(geo.r_mid_um, u.um),
        radius_dist=u.Quantity(geo.r_dist_um, u.um),
        density_mech=tuple(bucket.density_by_key.values()),
        point_mech=tuple(bucket.points),
        point_mech_roles=tuple(bucket.point_roles),
    )
