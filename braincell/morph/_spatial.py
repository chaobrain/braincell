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

"""Shared continuous spatial geometry for morphology-backed contexts."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import brainunit as u
import numpy as np

from .morphology import Morphology

__all__ = ["MorphologySpatialGeometry", "interpolate_branch"]

_Node = tuple[int, float]


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

    def path_distance_to_root(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from root-branch ``x=0``."""
        return u.Quantity(self._distance(branch_id, branch_x, self.root_bases_um), u.um)

    def path_distance_from_soma(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from the soma/root reference region."""
        values = np.asarray(branch_x, dtype=float)
        if int(branch_id) in self.reference_branch_ids:
            result = np.zeros_like(values)
        else:
            result = self._distance(branch_id, values, self.soma_bases_um)
        return u.Quantity(result, u.um)

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
    branch = morpho.branch(index=int(branch_id)).branch
    values = np.asarray(branch_x, dtype=float)
    geometry = _BranchGeometryUm(branch)

    radii = np.empty(values.shape, dtype=float)
    flat_radii = radii.reshape(-1)
    sites = values.reshape(-1).tolist()

    if geometry.p0 is None:
        for index, x in enumerate(sites):
            flat_radii[index] = geometry.at(x)[0]
        return u.Quantity(radii, u.um), None

    positions = np.empty(values.shape + (3,), dtype=float)
    flat_positions = positions.reshape(-1, 3)
    for index, x in enumerate(sites):
        flat_radii[index], flat_positions[index] = geometry.at(x)
    return u.Quantity(radii, u.um), u.Quantity(positions, u.um)


class _BranchGeometryUm:
    """Micrometre-decoded segment geometry for a single branch.

    :func:`interpolate_branch` evaluates many sites on one branch, and none
    of the five ``to_decimal`` conversions below depend on the site. Decoding
    them once per branch rather than once per site is the reason this class
    exists; ``at`` holds the per-site arithmetic that genuinely varies.
    """

    __slots__ = ("lengths", "r0", "r1", "p0", "p1", "total")

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

    def at(self, x: float) -> tuple[float, np.ndarray | None]:
        """Return the radius, and the 3-D point when available, at site ``x``."""
        lengths = self.lengths
        r0, r1, p0, p1 = self.r0, self.r1, self.p0, self.p1
        total = self.total

        cursor = 0.0
        last_positive = 0
        for index, segment_length in enumerate(lengths.tolist()):
            if segment_length <= 0.0:
                if np.isclose(x, cursor / total):
                    point = None if p0 is None or p1 is None else 0.5 * (p0[index] + p1[index])
                    return 0.5 * (r0[index] + r1[index]), point
                continue
            start = cursor / total
            cursor += segment_length
            end = cursor / total
            last_positive = index
            if x < end or (x == 1.0 and end == 1.0):
                fraction = (x - start) / (end - start)
                point = None if p0 is None or p1 is None else p0[index] + fraction * (p1[index] - p0[index])
                return r0[index] + fraction * (r1[index] - r0[index]), point
        # ``total > 0`` was enforced in __init__, so at least one segment had a
        # positive length and ``last_positive`` names a real segment here.
        point = None if p1 is None else p1[last_positive]
        return r1[last_positive], point
