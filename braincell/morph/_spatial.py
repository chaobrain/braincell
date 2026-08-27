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

    @classmethod
    def build(cls, morpho: Morphology) -> "MorphologySpatialGeometry":
        if not isinstance(morpho, Morphology):
            raise TypeError(f"Expected Morphology, got {type(morpho).__name__!s}.")

        n_branches = len(morpho.branches)
        lengths = np.asarray(
            [float(morpho.branch(index=i).branch.length.to_decimal(u.um)) for i in range(n_branches)],
            dtype=float,
        )
        node_sets = [{0.0, 1.0} for _ in range(n_branches)]
        for edge in morpho.edges:
            parent_id = int(edge.parent.index)
            child_id = int(edge.child.index)
            node_sets[parent_id].add(float(edge.parent_x))
            node_sets[child_id].add(float(edge.child_x))

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
        for edge in morpho.edges:
            parent = (int(edge.parent.index), float(edge.parent_x))
            child = (int(edge.child.index), float(edge.child_x))
            adjacency[parent].append((child, 0.0))
            adjacency[child].append((parent, 0.0))

        root_id = int(morpho.root.index)
        root_distance = _distances_from(adjacency, ((root_id, 0.0),))
        soma_ids = {int(branch.index) for branch in morpho.branches if str(branch.type) == "soma"}
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
        )

    def path_distance_to_root(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from root-branch ``x=0``."""
        return u.Quantity(self._distance(branch_id, branch_x, self.root_distance_um), u.um)

    def path_distance_from_soma(self, branch_id: int, branch_x: object) -> u.Quantity:
        """Return tree distance from the soma/root reference region."""
        values = np.asarray(branch_x, dtype=float)
        if int(branch_id) in self.reference_branch_ids:
            result = np.zeros_like(values)
        else:
            result = self._distance(branch_id, values, self.soma_distance_um)
        return u.Quantity(result, u.um)

    def _distance(self, branch_id: int, branch_x: object, distances: dict[_Node, float]) -> np.ndarray:
        branch_id = int(branch_id)
        if not 0 <= branch_id < len(self.branch_nodes_x):
            raise IndexError(f"branch_id {branch_id!r} is out of range.")
        values = np.asarray(branch_x, dtype=float)
        nodes = self.branch_nodes_x[branch_id]
        bases = np.asarray([distances[(branch_id, float(x))] for x in nodes], dtype=float)
        candidates = np.abs(np.expand_dims(values, axis=-1) - nodes) * self.branch_lengths_um[branch_id] + bases
        return np.min(candidates, axis=-1)


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
    radii = np.empty(values.shape, dtype=float)
    positions = None
    if branch.points_proximal is not None and branch.points_distal is not None:
        positions = np.empty(values.shape + (3,), dtype=float)

    for index, x in enumerate(values.reshape(-1)):
        radius_um, position_um = _interpolate_scalar(branch, float(x))
        radii.reshape(-1)[index] = radius_um
        if positions is not None:
            assert position_um is not None
            positions.reshape(-1, 3)[index] = position_um
    return (
        u.Quantity(radii, u.um),
        None if positions is None else u.Quantity(positions, u.um),
    )


def _interpolate_scalar(branch, x: float) -> tuple[float, np.ndarray | None]:
    lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    r0 = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    r1 = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
    p0 = None if branch.points_proximal is None else np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)
    p1 = None if branch.points_distal is None else np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)
    total = float(np.sum(lengths))
    if total <= 0.0:
        raise ValueError("Cannot interpolate a zero-length branch.")

    cursor = 0.0
    last_positive = None
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
    if last_positive is None:
        raise ValueError("Cannot interpolate a zero-length branch.")
    point = None if p1 is None else p1[last_positive]
    return r1[last_positive], point
