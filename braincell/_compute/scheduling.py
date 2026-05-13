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

"""Runtime scheduling helpers built from the declaration-time node tree."""

from dataclasses import dataclass

import numpy as np

from braincell._discretization.base import NodeTree

__all__ = [
    "NodeScheduling",
    "_compute_peel_levels",
    "build_node_scheduling",
]


@dataclass(frozen=True)
class NodeScheduling:
    """Execution-oriented grouping derived from a node tree."""

    algorithm: str
    row_to_point_id: np.ndarray
    point_id_to_row: np.ndarray
    groups: tuple[np.ndarray, ...]
    parent_rows: np.ndarray
    edges: np.ndarray
    level_size: np.ndarray
    level_start: np.ndarray


def build_node_scheduling(
    node_tree: NodeTree,
    *,
    max_group_size: int = 32,
    algorithm: str = "dhs",
) -> NodeScheduling:
    """Build an execution schedule from a node tree."""

    if not isinstance(node_tree, NodeTree):
        raise TypeError(
            f"build_node_scheduling(...) expects NodeTree, got {type(node_tree).__name__!s}."
        )
    _validate_algorithm(algorithm)
    _validate_max_group_size(max_group_size)

    node_parent, node_children = _build_node_parent_children(node_tree)
    peel_level_by_point = _compute_peel_levels(
        node_parent=node_parent,
        node_children=node_children,
    )
    row_to_point_id = _build_row_to_point_id(
        node_tree=node_tree,
        peel_level_by_point=peel_level_by_point,
    )
    point_count = len(node_tree.nodes)
    point_id_to_row = np.empty(point_count, dtype=np.int32)
    point_id_to_row[row_to_point_id] = np.arange(point_count, dtype=np.int32)
    parent_rows = np.full(point_count, -1, dtype=np.int32)
    for row, point_id in enumerate(row_to_point_id.tolist()):
        parent_id = int(node_parent[point_id])
        if parent_id >= 0:
            parent_rows[row] = int(point_id_to_row[parent_id])
    groups = _build_groups(
        node_tree=node_tree,
        peel_level_by_point=peel_level_by_point,
        point_id_to_row=point_id_to_row,
        max_group_size=max_group_size,
    )
    if len(groups) == 0:
        edges = np.empty((0, 2), dtype=np.int32)
        level_size = np.empty((0,), dtype=np.int32)
        level_start = np.empty((0,), dtype=np.int32)
    else:
        edge_pairs: list[list[int]] = []
        for group in groups:
            for row in group.tolist():
                parent_row = int(parent_rows[row])
                if parent_row >= 0:
                    edge_pairs.append([int(row), parent_row])
        edges = (
            np.asarray(edge_pairs, dtype=np.int32)
            if edge_pairs
            else np.empty((0, 2), dtype=np.int32)
        )
        level_size = np.asarray([len(group) for group in groups], dtype=np.int32)
        level_start = np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(level_size, dtype=np.int32)]
        )[:-1]

    return NodeScheduling(
        algorithm=algorithm,
        row_to_point_id=row_to_point_id,
        point_id_to_row=point_id_to_row,
        groups=groups,
        parent_rows=parent_rows,
        edges=edges,
        level_size=level_size,
        level_start=level_start,
    )


def _validate_algorithm(algorithm: str) -> None:
    if algorithm != "dhs":
        raise ValueError(f"Unsupported point scheduling algorithm {algorithm!r}.")


def _validate_max_group_size(max_group_size: int) -> None:
    if isinstance(max_group_size, bool) or not isinstance(max_group_size, int):
        raise TypeError(f"max_group_size must be int, got {max_group_size!r}.")
    if max_group_size <= 0:
        raise ValueError(f"max_group_size must be > 0, got {max_group_size!r}.")


def _compute_peel_levels(
    *,
    node_parent: np.ndarray | None = None,
    node_children: tuple[tuple[int, ...], ...] | None = None,
    node_tree: "NodeTree | None" = None,
) -> np.ndarray:
    """Assign each node its distance-to-farthest-leaf peel level."""

    if node_tree is not None:
        if node_parent is not None or node_children is not None:
            raise TypeError(
                "_compute_peel_levels: pass either node_tree or "
                "(node_parent, node_children), not both."
            )
        node_parent, node_children = _build_node_parent_children(node_tree)
    if node_parent is None or node_children is None:
        raise TypeError(
            "_compute_peel_levels: supply node_tree or both "
            "node_parent and node_children."
        )

    n_point = int(len(node_parent))
    levels = np.full(n_point, -1, dtype=np.int32)
    remaining_children = np.asarray(
        [len(children) for children in node_children], dtype=np.int32
    )

    frontier: list[int] = [
        pid for pid, count in enumerate(remaining_children.tolist()) if count == 0
    ]
    for pid in frontier:
        levels[pid] = 0

    cursor = 0
    while cursor < len(frontier):
        pid = frontier[cursor]
        cursor += 1
        parent = int(node_parent[pid])
        if parent < 0:
            continue
        candidate = int(levels[pid]) + 1
        if int(levels[parent]) < candidate:
            levels[parent] = candidate
        remaining_children[parent] -= 1
        if int(remaining_children[parent]) == 0:
            frontier.append(parent)

    if (levels < 0).any():
        raise ValueError(
            "compute_peel_levels: cycle detected or node unreachable from any leaf."
        )
    return levels


def _order_nodes_by_peel_then_id(
    *,
    node_tree: NodeTree,
    peel_levels: np.ndarray,
) -> np.ndarray:
    """Permutation of node ids: peel-descending, then node-id ascending."""

    node_ids = np.asarray([node.id for node in node_tree.nodes], dtype=np.int32)
    peels = np.asarray(peel_levels, dtype=np.int32)
    return np.lexsort((node_ids, -peels)).astype(np.int32)


def _build_row_to_point_id(
    *,
    node_tree: NodeTree,
    peel_level_by_point: np.ndarray,
) -> np.ndarray:
    return _order_nodes_by_peel_then_id(
        node_tree=node_tree,
        peel_levels=peel_level_by_point,
    )


def _build_groups(
    *,
    node_tree: NodeTree,
    peel_level_by_point: np.ndarray,
    point_id_to_row: np.ndarray,
    max_group_size: int,
) -> tuple[np.ndarray, ...]:
    order = _order_nodes_by_peel_then_id(
        node_tree=node_tree,
        peel_levels=peel_level_by_point,
    )
    if len(order) == 0:
        return ()

    peels_ordered = peel_level_by_point[order]
    level_starts: list[int] = [0]
    for i in range(1, len(order)):
        if int(peels_ordered[i]) != int(peels_ordered[i - 1]):
            level_starts.append(i)
    level_starts.append(len(order))

    groups: list[np.ndarray] = []
    for a, b in zip(level_starts[:-1], level_starts[1:]):
        for chunk_start in range(a, b, max_group_size):
            chunk_stop = min(b, chunk_start + max_group_size)
            chunk_point_ids = order[chunk_start:chunk_stop]
            rows = point_id_to_row[chunk_point_ids]
            groups.append(np.asarray(rows, dtype=np.int32))
    return tuple(groups)


def _build_node_parent_children(
    node_tree: NodeTree,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    node_parent = np.full(len(node_tree.nodes), -1, dtype=np.int32)
    node_children_lists: list[list[int]] = [[] for _ in node_tree.nodes]
    for edge in node_tree.edges:
        child_node_id = int(edge.child_node_id)
        parent_node_id = int(edge.parent_node_id)
        if int(node_parent[child_node_id]) != -1:
            raise ValueError(
                f"Node {child_node_id} already has parent {int(node_parent[child_node_id])}."
            )
        node_parent[child_node_id] = parent_node_id
        node_children_lists[parent_node_id].append(child_node_id)
    node_children = tuple(tuple(children) for children in node_children_lists)
    return node_parent, node_children
