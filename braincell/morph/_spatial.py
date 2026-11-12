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

"""Shared continuous spatial geometry for morphology-backed contexts.

This module is ``morph``'s shared implementation for three packages:
``braincell._discretization`` (CV contexts), ``braincell.filter`` (sampling
contexts and ``TreeDistanceRegion``) and ``braincell.network`` (synapse
pairing contexts). It is not public API: its fields and signatures may change
without a deprecation, as long as those importers change with them.

Users get the same quantities from public objects: the
``path_distance_to_root``, ``path_distance_from_soma`` and ``radius`` fields of
:class:`braincell.filter.SamplingContext`, the matching fields of
:class:`braincell.mech.CVContext`, the metrics in :mod:`braincell.filter.metric`,
the distance regions of :mod:`braincell.filter`, and
:attr:`braincell.Morphology.max_path_distance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq

import brainunit as u
import numpy as np

from braincell.morph.morphology import Morphology

__all__ = ["MorphologySpatialGeometry", "interpolate_branch"]

_Node = tuple[int, float]

#: How close in ``x`` a site must be to a zero-length segment to take its
#: midpoint (the mean radius of a jump); ``np.isclose``'s ``atol``.
_AT_ZERO_LENGTH = 1e-8


@dataclass(frozen=True)
class MorphologySpatialGeometry:
    """Precomputed tree distances used while materializing spatial contexts.

    Soma-relative distance is the shortest path to the union of all soma
    branches. If the morphology has no soma branch, the whole root branch is
    the reference region instead.
    """

    morpho: Morphology
    branch_lengths_um: np.ndarray
    branch_nodes_x: tuple[np.ndarray, ...]
    root_distance_um: dict[_Node, float]
    soma_distance_um: dict[_Node, float]
    reference_branch_ids: frozenset[int]
    root_bases_um: tuple[np.ndarray, ...]
    soma_bases_um: tuple[np.ndarray, ...]
    #: Each branch's decoded segments, filled as :meth:`interpolate_um` reads them.
    branch_segments: dict[int, "_BranchGeometryUm"] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def build(cls, morpho: Morphology) -> "MorphologySpatialGeometry":
        if not isinstance(morpho, Morphology):
            raise TypeError(f"Expected Morphology, got {type(morpho).__name__!s}.")

        branches = morpho.branches
        n_branches = len(branches)
        lengths = np.asarray(
            [float(branch.branch.length.to_decimal(u.um)) for branch in branches],
            dtype=float,
        )
        # One pass over the edges: collect the endpoint keys while building the
        # node sets, then reuse them once ``adjacency`` exists.
        node_sets = [{0.0, 1.0} for _ in range(n_branches)]
        endpoints: list[tuple[_Node, _Node]] = []
        for edge in morpho.edges:
            parent = (int(edge.parent.index), float(edge.parent_x))
            child = (int(edge.child.index), float(edge.child_x))
            endpoints.append((parent, child))
            node_sets[parent[0]].add(parent[1])
            node_sets[child[0]].add(child[1])

        branch_nodes = tuple(np.asarray(sorted(nodes), dtype=float) for nodes in node_sets)
        adjacency: dict[_Node, list[tuple[_Node, float]]] = {
            (branch_id, float(x)): [] for branch_id, nodes in enumerate(branch_nodes) for x in nodes
        }
        for branch_id, nodes in enumerate(branch_nodes):
            for x0, x1 in zip(nodes[:-1], nodes[1:]):
                first = (branch_id, float(x0))
                second = (branch_id, float(x1))
                distance = float(x1 - x0) * lengths[branch_id]
                adjacency[first].append((second, distance))
                adjacency[second].append((first, distance))
        for parent, child in endpoints:
            adjacency[parent].append((child, 0.0))
            adjacency[child].append((parent, 0.0))

        root_id = int(morpho.root.index)
        root_distance = _distances_from(adjacency, ((root_id, 0.0),))
        soma_ids = {int(branch.index) for branch in branches if str(branch.type) == "soma"}
        reference_ids = soma_ids or {root_id}
        reference_nodes = tuple(
            (branch_id, float(x)) for branch_id in sorted(reference_ids) for x in branch_nodes[branch_id]
        )
        soma_distance = _distances_from(adjacency, reference_nodes)
        return cls(
            morpho=morpho,
            branch_lengths_um=lengths,
            branch_nodes_x=branch_nodes,
            root_distance_um=root_distance,
            soma_distance_um=soma_distance,
            reference_branch_ids=frozenset(reference_ids),
            root_bases_um=_bases_per_branch(branch_nodes, root_distance),
            soma_bases_um=_bases_per_branch(branch_nodes, soma_distance),
        )

    @classmethod
    def of(cls, morpho: Morphology) -> "MorphologySpatialGeometry":
        """Return :meth:`build` of ``morpho``, built once per :attr:`~Morphology.revision`.

        The geometry is kept on ``morpho`` with the revision it was built
        at, as ``Morphology.to_vis_data`` keeps its snapshot; any edit
        drops it and advances the revision, so the next call builds again
        (CELL-18c). A shallow copy shares the attribute but not the tree,
        so it builds its own.
        """
        cached = morpho._spatial_geometry_cache
        if cached is not None and cached[0] == morpho.revision and cached[1].morpho is morpho:
            return cached[1]
        geometry = cls.build(morpho)
        morpho._spatial_geometry_cache = (morpho.revision, geometry)
        return geometry

    def path_distance_to_root(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from root-branch ``x=0``."""
        return u.Quantity(self._distance(branch_id, branch_x, self.root_bases_um), u.um)

    def path_distance_from_soma(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from the soma/root reference region."""
        return u.Quantity(self._soma_distance(branch_id, branch_x), u.um)

    def path_distances_um(self, branch_id: int, branch_x: object) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`path_distance_to_root` and :meth:`path_distance_from_soma` in micrometres."""
        return self._distance(branch_id, branch_x, self.root_bases_um), self._soma_distance(branch_id, branch_x)

    def interpolate_um(self, branch_id: int, branch_x: object) -> tuple[np.ndarray, np.ndarray | None]:
        """:func:`interpolate_branch` in micrometres, decoding each branch once."""
        branch_id = int(branch_id)
        segments = self.branch_segments.get(branch_id)
        if segments is None:
            segments = _BranchGeometryUm(self.morpho.branch(index=branch_id).branch)
            self.branch_segments[branch_id] = segments
        return segments.sample(branch_x)

    def _soma_distance(self, branch_id: int, branch_x: object) -> np.ndarray:
        values = np.asarray(branch_x, dtype=float)
        if int(branch_id) in self.reference_branch_ids:
            return np.zeros_like(values)
        return self._distance(branch_id, values, self.soma_bases_um)

    def _distance(self, branch_id: int, branch_x: object, bases_um: tuple[np.ndarray, ...]) -> np.ndarray:
        branch_id = int(branch_id)
        if not 0 <= branch_id < len(self.branch_nodes_x):
            raise IndexError(f"branch_id {branch_id!r} is out of range.")
        values = np.asarray(branch_x, dtype=float)
        nodes = self.branch_nodes_x[branch_id]
        candidates = (
            np.abs(np.expand_dims(values, axis=-1) - nodes) * self.branch_lengths_um[branch_id] + bases_um[branch_id]
        )
        return np.min(candidates, axis=-1)


def _bases_per_branch(
    branch_nodes: tuple[np.ndarray, ...],
    distances: dict[_Node, float],
) -> tuple[np.ndarray, ...]:
    """Materialize the per-branch node distances ``_distance`` reduces over.

    ``bases`` is fully determined by ``(branch_id, distances)`` on a frozen
    dataclass, so rebuilding it per query cost one dict lookup per node plus
    an ``np.asarray`` on every call.
    """
    return tuple(
        np.asarray([distances[(branch_id, float(x))] for x in nodes], dtype=float)
        for branch_id, nodes in enumerate(branch_nodes)
    )


def _distances_from(adjacency: dict[_Node, list[tuple[_Node, float]]], starts: tuple[_Node, ...]) -> dict[_Node, float]:
    distances = {node: float("inf") for node in adjacency}
    queue: list[tuple[float, _Node]] = []
    for node in starts:
        distances[node] = 0.0
        heapq.heappush(queue, (0.0, node))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def interpolate_branch(morpho: Morphology, branch_id: int, branch_x: object) -> tuple[u.Quantity, u.Quantity | None]:
    """Interpolate radius and optional 3-D position at continuous sites."""
    radii, points = _BranchGeometryUm(morpho.branch(index=int(branch_id)).branch).sample(branch_x)
    return u.Quantity(radii, u.um), None if points is None else u.Quantity(points, u.um)


class _BranchGeometryUm:
    """Micrometre-decoded segment geometry for a single branch.

    :func:`interpolate_branch` evaluates many sites on one branch, and none
    of the five ``to_decimal`` conversions below depend on the site. Decoding
    them once per branch rather than once per site is the reason this class
    exists; :meth:`sample` holds the per-site arithmetic that genuinely varies.
    """

    __slots__ = ("lengths", "r0", "r1", "p0", "p1", "total", "start", "end", "has_length", "long_index", "long_end")

    def __init__(self, branch) -> None:
        self.lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
        self.r0 = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
        self.r1 = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
        points_proximal = branch.points_proximal
        points_distal = branch.points_distal
        has_points = points_proximal is not None and points_distal is not None
        self.p0 = np.asarray(points_proximal.to_decimal(u.um), dtype=float) if has_points else None
        self.p1 = np.asarray(points_distal.to_decimal(u.um), dtype=float) if has_points else None
        self.total = float(np.sum(self.lengths))
        if self.total <= 0.0:
            raise ValueError("Cannot interpolate a zero-length branch.")
        # Where each segment starts and ends as a fraction of the branch:
        # ``cumsum`` adds the lengths in the order a walk along them does.
        self.has_length = ~(self.lengths <= 0.0)
        cursor = np.cumsum(np.where(self.has_length, self.lengths, 0.0))
        self.start = np.concatenate(([0.0], cursor[:-1])) / self.total
        self.end = cursor / self.total
        self.long_index = np.flatnonzero(self.has_length)
        # The ends of segments with a length never decrease, up to a NaN.
        self.long_end = self.end[self.long_index]

    def sample(self, branch_x: object) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the radius, and the 3-D point when available, at each site.

        A site lies in the first segment, walking from ``x=0``, that is
        zero-length and within ``1e-8`` of it in ``x``, or that has a length
        and ends past it (``x=1`` also in one ending at exactly 1); a
        zero-length segment gives its midpoint, and a site in no segment gets
        the distal end. The window is absolute, the same at every ``x``: a
        relative one (``np.isclose``'s default ``rtol``) gave a jump's mean
        radius up to ``1e-5 * x`` away from it. All sites are placed at once (CELL-18c): ``cumsum`` adds the
        lengths in the order a walk does, so every boundary, and so every
        value, is the one the walk gives.
        """
        values = np.asarray(branch_x, dtype=float)
        sites = values.reshape(-1)
        start, end, has_length, long_index, long_end = (
            self.start,
            self.end,
            self.has_length,
            self.long_index,
            self.long_end,
        )
        n_segments = self.lengths.size

        segment = np.full(sites.shape, n_segments)
        if long_index.size:
            at_one = sites == 1.0
            found = np.where(
                at_one,
                np.searchsorted(long_end, sites, side="left"),
                np.searchsorted(long_end, sites, side="right"),
            )
            candidate = long_index[np.minimum(found, long_index.size - 1)]
            inside = (found < long_index.size) & ((sites < end[candidate]) | (at_one & (end[candidate] == 1.0)))
            segment = np.where(inside, candidate, segment)
        for index in np.flatnonzero(~has_length)[::-1].tolist():
            near = np.abs(sites - start[index]) <= _AT_ZERO_LENGTH
            segment = np.where((index < segment) & near, index, segment)

        last_long = int(long_index[-1]) if long_index.size else 0
        beyond = segment == n_segments
        segment = np.where(beyond, last_long, segment)
        midpoint = ~beyond & ~has_length[segment]
        along = ~beyond & has_length[segment]
        fraction = (sites[along] - start[segment[along]]) / (end[segment[along]] - start[segment[along]])

        radii = np.empty(sites.shape, dtype=float)
        radii[beyond] = self.r1[last_long]
        radii[midpoint] = 0.5 * (self.r0[segment[midpoint]] + self.r1[segment[midpoint]])
        r0 = self.r0[segment[along]]
        radii[along] = r0 + fraction * (self.r1[segment[along]] - r0)
        if self.p0 is None or self.p1 is None:
            return radii.reshape(values.shape), None
        points = np.empty(sites.shape + (3,), dtype=float)
        points[beyond] = self.p1[last_long]
        points[midpoint] = 0.5 * (self.p0[segment[midpoint]] + self.p1[segment[midpoint]])
        p0 = self.p0[segment[along]]
        points[along] = p0 + fraction[:, None] * (self.p1[segment[along]] - p0)
        return radii.reshape(values.shape), points.reshape(values.shape + (3,))
