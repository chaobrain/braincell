from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .point_tree import PointTree


@dataclass(frozen=True)
class PointScheduling:
    algorithm: str
    row_to_point_id: np.ndarray
    point_id_to_row: np.ndarray
    groups: tuple[np.ndarray, ...]
    parent_rows: np.ndarray
    edges: np.ndarray
    level_size: np.ndarray
    level_start: np.ndarray


def build_point_scheduling(
    point_tree: PointTree,
    *,
    max_group_size: int = 32,
    algorithm: str = "dhs",
) -> PointScheduling:
    if not isinstance(point_tree, PointTree):
        raise TypeError(f"build_point_scheduling(...) expects PointTree, got {type(point_tree).__name__!s}.")
    _validate_algorithm(algorithm)
    _validate_max_group_size(max_group_size)

    peel_level_by_point = _compute_peel_levels(point_tree=point_tree)
    row_to_point_id = _build_row_to_point_id(point_tree=point_tree, peel_level_by_point=peel_level_by_point)
    point_id_to_row = np.empty(point_tree.point_count, dtype=np.int32)
    point_id_to_row[row_to_point_id] = np.arange(point_tree.point_count, dtype=np.int32)
    parent_rows = np.full(point_tree.point_count, -1, dtype=np.int32)
    for row, point_id in enumerate(row_to_point_id.tolist()):
        parent_id = int(point_tree.point_parent[point_id])
        if parent_id >= 0:
            parent_rows[row] = int(point_id_to_row[parent_id])
    groups = _build_groups(
        point_tree=point_tree,
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
        edges = np.asarray(edge_pairs, dtype=np.int32) if edge_pairs else np.empty((0, 2), dtype=np.int32)
        level_size = np.asarray([len(group) for group in groups], dtype=np.int32)
        level_start = np.concatenate(
            [np.asarray([0], dtype=np.int32), np.cumsum(level_size, dtype=np.int32)]
        )[:-1]

    return PointScheduling(
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


def _compute_peel_levels(*, point_tree: PointTree) -> np.ndarray:
    levels = np.zeros(point_tree.point_count, dtype=np.int32)
    for point_id in range(point_tree.point_count - 1, -1, -1):
        children_ids = point_tree.point_children[point_id]
        if len(children_ids) == 0:
            levels[point_id] = 0
        else:
            levels[point_id] = 1 + max(levels[child_id] for child_id in children_ids)
    return levels


def _build_row_to_point_id(
    *,
    point_tree: PointTree,
    peel_level_by_point: np.ndarray,
) -> np.ndarray:
    rows: list[int] = []
    max_level = int(peel_level_by_point.max(initial=0))
    for level in range(max_level, -1, -1):
        point_ids = [point_id for point_id, peel in enumerate(peel_level_by_point) if int(peel) == level]
        point_ids.sort(key=lambda point_id: int(point_tree.point_id_to_matrix_index[point_id]))
        rows.extend(point_ids)
    return np.asarray(rows, dtype=np.int32)


def _build_groups(
    *,
    point_tree: PointTree,
    peel_level_by_point: np.ndarray,
    point_id_to_row: np.ndarray,
    max_group_size: int,
) -> tuple[np.ndarray, ...]:
    groups: list[np.ndarray] = []
    max_level = int(peel_level_by_point.max(initial=0))
    for level in range(max_level, -1, -1):
        point_ids = [point_id for point_id, peel in enumerate(peel_level_by_point) if int(peel) == level]
        point_ids.sort(key=lambda point_id: int(point_tree.point_id_to_matrix_index[point_id]))
        for start in range(0, len(point_ids), max_group_size):
            chunk = point_ids[start:start + max_group_size]
            groups.append(np.asarray([int(point_id_to_row[point_id]) for point_id in chunk], dtype=np.int32))
    return tuple(groups)
