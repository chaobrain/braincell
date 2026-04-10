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


from dataclasses import dataclass

import numpy as np

from braincell.morpho import Morpho
from .cv import CV
from .cv_geo import EPSILON

__all__ = [
    "PointTree",
    "PointScheduling",
    "build_point_tree",
    "build_point_scheduling",
]

# ``PointTree`` is the execution-oriented view of the current CV list. It is
# not another morphology model; it is the merged point-edge graph used by the
# voltage solver, sparse point mechanisms, and scheduling code.

_POSITION_ORDER = {"proximal": 0, "mid": 1, "distal": 2}


@dataclass(frozen=True)
class CVPoint:
    """Reference from a compute point back to one CV-local role.

    A single compute point may coincide with several CV-local positions after
    attachment merging, so :class:`ComputePoint` stores a tuple of these records.

    ``CVPoint`` is intentionally tiny: it only records ``(cv_id, position)``,
    but that is enough for debugging point merges and for tracing a solver-space
    point back to the original CV-local role that produced it.
    """
    cv_id: int
    position: str


@dataclass(frozen=True)
class ComputePoint:
    """Merged point in the compute graph derived from CV geometry.

    ``ComputePoint`` is not tied to just one CV. It represents one unique node
    in the assembled tree after CV proximal/mid/distal locations have been
    merged across attachments and shared boundaries.

    Main payload:

    - one stable compute-point id
    - a tuple of :class:`CVPoint` roles that collapsed into that merged point

    :class:`PointTree` exposes these nodes to solver and inspection code, while
    midpoint lookup tables map each CV back to the compute point used for its
    runtime state.
    """
    id: int
    cv_points: tuple[CVPoint, ...]


@dataclass(frozen=True)
class CVEdge:
    """Reference from a compute edge back to one CV half-edge.

    Like :class:`CVPoint`, this is a provenance record. It lets a merged
    compute-graph edge say which CV-local half-edge contribution it came from.
    """
    cv_id: int
    half: str


@dataclass(frozen=True)
class ComputeEdge:
    """Directed edge in the compute graph between two merged points.

    Like :class:`ComputePoint`, a compute edge may aggregate contributions from
    one or more CV-local half-edges. This is the connectivity that later matrix
    assembly and scheduling logic work with.

    Main payload:

    - parent and child compute-point ids
    - one stable edge id
    - provenance via the contributing :class:`CVEdge` records

    ``PointTree`` stores these edges as the execution graph consumed by solver
    matrix assembly and traversal-oriented scheduling.
    """
    id: int
    parent_point_id: int
    child_point_id: int
    cv_edges: tuple[CVEdge, ...]


@dataclass(frozen=True)
class PointTree:
    """Compute-point view of a cell's CV topology.

    ``PointTree`` is built from the immutable :class:`CV` list and converts
    branch/CV-centered geometry into a point-edge tree better suited for matrix
    assembly and traversal algorithms.

    It records:

    - merged compute points and edges
    - parent/child relations between points
    - mappings between CV ids, point ids, and matrix ordering indices
    - fast lookup of midpoint points and branch terminal points

    ``Cell.point_tree()`` caches this object, and :class:`PointScheduling`
    consumes it to build DHS-style processing groups.

    Key stored mappings:

    - point parent/child relations for traversal
    - midpoint point ids for each CV
    - terminal point ids for each branch
    - matrix-order conversions between point ids and solver row ids

    Typical usage is indirect: ``Cell`` builds and caches one ``PointTree``,
    voltage-solver helpers consume its topology, and notebook/debug code uses it
    to understand how CV-centered declarations were lowered into point space.
    """
    points: tuple[ComputePoint, ...]
    edges: tuple[ComputeEdge, ...]
    point_parent: np.ndarray
    point_children: tuple[tuple[int, ...], ...]
    cv_midpoint_point_id: np.ndarray
    branch_terminal_point_id: np.ndarray
    root_point_id: int
    point_id_to_matrix_index: np.ndarray
    matrix_index_to_point_id: np.ndarray
    cv_id_to_matrix_index: np.ndarray

    def __repr__(self) -> str:
        return (
            f"PointTree(n_points={len(self.points)!r}, n_edges={len(self.edges)!r}, "
            f"root_point_id={self.root_point_id!r})"
        )

    def __str__(self) -> str:
        return (
            f"{'-'*35}\n"
            f"{'n_points':<14} | {len(self.points)}\n"
            f"{'n_edges':<14} | {len(self.edges)}\n"
            f"{'root_point_id':<14} | {self.root_point_id}\n"
            f"{'-'*35}\n"
        )


@dataclass(frozen=True)
class PointScheduling:
    """Execution-oriented grouping derived from a :class:`PointTree`.

    ``PointScheduling`` is the row/group view used by traversal-based solver
    code. It does not redefine topology; instead it reorders the existing point
    tree into batches and dependency arrays that are cheaper to process.

    Main fields:

    - row/point conversion arrays
    - grouped row batches for one scheduling algorithm
    - parent-row and edge arrays for dependency traversal
    - level metadata describing the chunked processing order

    ``Cell.point_scheduling(...)`` builds and caches this object from a
    :class:`PointTree`, and axial-voltage solver code consumes it directly.
    """

    algorithm: str
    row_to_point_id: np.ndarray
    point_id_to_row: np.ndarray
    groups: tuple[np.ndarray, ...]
    parent_rows: np.ndarray
    edges: np.ndarray
    level_size: np.ndarray
    level_start: np.ndarray


@dataclass
class _PointDraft:
    """Mutable build-time draft for one compute point.

    ``build_point_tree`` first accumulates merged point roles into these drafts,
    then freezes them into immutable :class:`ComputePoint` instances.
    """
    id: int
    cv_points: set[tuple[int, str]]


@dataclass(frozen=True)
class _BranchTraversal:
    """Build-time traversal summary for one branch.

    This helper records the CV walk order chosen for one branch plus the
    terminal compute point reached by that walk so later matrix ordering can be
    assembled branch by branch.
    """
    ordered_cv_ids: tuple[int, ...]
    terminal_point_id: int


def build_point_tree(
    morpho: Morpho,
    *,
    cvs: tuple[CV, ...],
) -> PointTree:
    if not isinstance(morpho, Morpho):
        raise TypeError(f"build_point_tree(...) expects Morpho, got {type(morpho).__name__!s}.")

    cv_ids_by_branch = _group_cv_ids_by_branch(cvs=cvs, n_branches=len(morpho.branches))
    edge_by_child_branch = {edge.child.index: edge for edge in morpho.edges}

    drafts: list[_PointDraft] = []
    cv_midpoint_point_id = np.full(len(cvs), -1, dtype=np.int32)
    branch_terminal_point_id = np.full(len(morpho.branches), -1, dtype=np.int32)
    branch_endpoint_point_id_by_x: dict[tuple[int, float], int] = {}
    traversals_by_branch: list[_BranchTraversal | None] = [None for _ in morpho.branches]
    logical_edge_roles: dict[tuple[int, int], list[tuple[int, str]]] = {}
    logical_edge_order: list[tuple[int, int]] = []

    def new_point(*, cv_id: int, position: str) -> int:
        point_id = len(drafts)
        drafts.append(_PointDraft(id=point_id, cv_points={(cv_id, position)}))
        return point_id

    def add_point_role(point_id: int, *, cv_id: int, position: str) -> None:
        drafts[point_id].cv_points.add((cv_id, position))

    def add_edge_role(parent_point_id: int, child_point_id: int, *, cv_id: int, half: str) -> None:
        key = (parent_point_id, child_point_id)
        if key not in logical_edge_roles:
            logical_edge_roles[key] = []
            logical_edge_order.append(key)
        role = (cv_id, half)
        if role not in logical_edge_roles[key]:
            logical_edge_roles[key].append(role)

    root_branch_cv_ids = cv_ids_by_branch[0]
    if len(root_branch_cv_ids) == 0:
        raise ValueError("Root branch has no CVs.")
    root_first_cv_id = root_branch_cv_ids[0]
    root_point_id = new_point(cv_id=root_first_cv_id, position="proximal")
    branch_endpoint_point_id_by_x[(0, 0.0)] = root_point_id

    for branch_id, branch in enumerate(morpho.branches):
        branch_cv_ids = cv_ids_by_branch[branch_id]
        if len(branch_cv_ids) == 0:
            raise ValueError(f"Branch {branch_id} has no CVs.")

        if branch.parent is None:
            attachment_point_id = root_point_id
            attach_x = 0.0
            ordered_cv_ids = branch_cv_ids
        else:
            edge = edge_by_child_branch[branch_id]
            attachment_point_id = _resolve_attachment_point(
                edge.parent.index,
                parent_x=float(edge.parent_x),
                branch_endpoint_point_id_by_x=branch_endpoint_point_id_by_x,
                cv_midpoint_point_id=cv_midpoint_point_id,
                cv_ids_by_branch=cv_ids_by_branch,
                cvs=cvs,
            )
            attach_x = float(edge.child_x)
            ordered_cv_ids = branch_cv_ids if attach_x <= EPSILON else tuple(reversed(branch_cv_ids))

        first_cv_id = ordered_cv_ids[0]
        add_point_role(
            attachment_point_id,
            cv_id=first_cv_id,
            position=_entry_position_for_walk(attach_x),
        )
        branch_endpoint_point_id_by_x[(branch_id, float(attach_x))] = attachment_point_id

        for cv_id in ordered_cv_ids:
            if int(cv_midpoint_point_id[cv_id]) != -1:
                raise ValueError(f"CV {cv_id} already has a midpoint point.")
            cv_midpoint_point_id[cv_id] = new_point(cv_id=cv_id, position="mid")

        terminal_cv_id = ordered_cv_ids[-1]
        terminal_point_id = new_point(
            cv_id=terminal_cv_id,
            position=_exit_position_for_walk(attach_x),
        )
        branch_terminal_point_id[branch_id] = terminal_point_id
        branch_endpoint_point_id_by_x[(branch_id, float(1.0 - attach_x))] = terminal_point_id
        traversals_by_branch[branch_id] = _BranchTraversal(
            ordered_cv_ids=ordered_cv_ids,
            terminal_point_id=terminal_point_id,
        )

        for index, cv_id in enumerate(ordered_cv_ids):
            midpoint_point_id = int(cv_midpoint_point_id[cv_id])
            parent_point_id = attachment_point_id if index == 0 else int(cv_midpoint_point_id[ordered_cv_ids[index - 1]])
            child_point_id = terminal_point_id if index == len(ordered_cv_ids) - 1 else int(
                cv_midpoint_point_id[ordered_cv_ids[index + 1]]
            )
            add_edge_role(
                parent_point_id,
                midpoint_point_id,
                cv_id=cv_id,
                half=_entry_half_for_walk(attach_x) if index == 0 else _entry_half_for_walk(attach_x),
            )
            add_edge_role(
                midpoint_point_id,
                child_point_id,
                cv_id=cv_id,
                half=_exit_half_for_walk(attach_x),
            )

    if np.any(cv_midpoint_point_id < 0):
        raise ValueError("Point tree is missing CV midpoint points.")
    if np.any(branch_terminal_point_id < 0):
        raise ValueError("Point tree is missing branch terminal points.")

    points = tuple(
        ComputePoint(
            id=draft.id,
            cv_points=tuple(
                CVPoint(cv_id=cv_id, position=position)
                for cv_id, position in sorted(
                    draft.cv_points,
                    key=lambda item: (item[0], _POSITION_ORDER[item[1]]),
                )
            ),
        )
        for draft in drafts
    )

    edges = tuple(
        ComputeEdge(
            id=edge_id,
            parent_point_id=parent_point_id,
            child_point_id=child_point_id,
            cv_edges=tuple(
                CVEdge(cv_id=cv_id, half=half)
                for cv_id, half in sorted(cv_roles, key=lambda item: (item[0], item[1]))
            ),
        )
        for edge_id, ((parent_point_id, child_point_id), cv_roles) in enumerate(
            (key, logical_edge_roles[key]) for key in logical_edge_order
        )
    )

    point_parent = np.full(len(points), -1, dtype=np.int32)
    point_children_lists: list[list[int]] = [[] for _ in points]
    for edge in edges:
        child_point_id = edge.child_point_id
        parent_point_id = edge.parent_point_id
        if int(point_parent[child_point_id]) != -1:
            raise ValueError(f"Point {child_point_id} already has parent {int(point_parent[child_point_id])}.")
        point_parent[child_point_id] = parent_point_id
        point_children_lists[parent_point_id].append(child_point_id)
    point_children = tuple(tuple(children) for children in point_children_lists)

    matrix_index_to_point_id = _build_matrix_index_to_point_id(
        traversals_by_branch=traversals_by_branch,
        cv_midpoint_point_id=cv_midpoint_point_id,
        root_point_id=root_point_id,
    )
    point_id_to_matrix_index = np.empty(len(points), dtype=np.int32)
    point_id_to_matrix_index[matrix_index_to_point_id] = np.arange(len(points), dtype=np.int32)
    cv_id_to_matrix_index = point_id_to_matrix_index[cv_midpoint_point_id]

    return PointTree(
        points=points,
        edges=edges,
        point_parent=point_parent,
        point_children=point_children,
        cv_midpoint_point_id=cv_midpoint_point_id,
        branch_terminal_point_id=branch_terminal_point_id,
        root_point_id=root_point_id,
        point_id_to_matrix_index=point_id_to_matrix_index,
        matrix_index_to_point_id=matrix_index_to_point_id,
        cv_id_to_matrix_index=cv_id_to_matrix_index,
    )


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
    point_count = len(point_tree.points)
    point_id_to_row = np.empty(point_count, dtype=np.int32)
    point_id_to_row[row_to_point_id] = np.arange(point_count, dtype=np.int32)
    parent_rows = np.full(point_count, -1, dtype=np.int32)
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


def _group_cv_ids_by_branch(*, cvs: tuple[CV, ...], n_branches: int) -> tuple[tuple[int, ...], ...]:
    grouped: list[list[int]] = [[] for _ in range(n_branches)]
    for cv in cvs:
        grouped[cv.branch_id].append(cv.id)
    return tuple(tuple(ids) for ids in grouped)


def _resolve_attachment_point(
    parent_branch_id: int,
    *,
    parent_x: float,
    branch_endpoint_point_id_by_x: dict[tuple[int, float], int],
    cv_midpoint_point_id: np.ndarray,
    cv_ids_by_branch: tuple[tuple[int, ...], ...],
    cvs: tuple[CV, ...],
) -> int:
    if parent_x <= 0.0 + EPSILON:
        return branch_endpoint_point_id_by_x[(parent_branch_id, 0.0)]
    if parent_x >= 1.0 - EPSILON:
        return branch_endpoint_point_id_by_x[(parent_branch_id, 1.0)]
    cv_id = _locate_branch_cv_by_x(
        cv_ids_by_branch[parent_branch_id],
        cvs,
        x=float(parent_x),
        epsilon=EPSILON,
    )
    return int(cv_midpoint_point_id[cv_id])


def _locate_branch_cv_by_x(
    ids: tuple[int, ...],
    cvs: tuple[CV, ...],
    *,
    x: float,
    epsilon: float,
) -> int:
    if x <= 0.0 + epsilon:
        return ids[0]
    if x >= 1.0 - epsilon:
        return ids[-1]
    for cv_id in ids:
        cv = cvs[cv_id]
        if x >= cv.prox - epsilon and x < cv.dist - epsilon:
            return cv_id
    return ids[-1]


def _entry_half_for_walk(attach_x: float) -> str:
    return "prox" if attach_x <= EPSILON else "dist"


def _exit_half_for_walk(attach_x: float) -> str:
    return "dist" if attach_x <= EPSILON else "prox"


def _entry_position_for_walk(attach_x: float) -> str:
    return "proximal" if attach_x <= EPSILON else "distal"


def _exit_position_for_walk(attach_x: float) -> str:
    return "distal" if attach_x <= EPSILON else "proximal"


def _build_matrix_index_to_point_id(
    *,
    traversals_by_branch: list[_BranchTraversal | None],
    cv_midpoint_point_id: np.ndarray,
    root_point_id: int,
) -> np.ndarray:
    ordered_point_ids: list[int] = [root_point_id]
    seen = {root_point_id}
    for traversal in traversals_by_branch:
        if traversal is None:
            raise ValueError("Point tree is missing branch traversal metadata.")
        for cv_id in traversal.ordered_cv_ids:
            point_id = int(cv_midpoint_point_id[cv_id])
            if point_id not in seen:
                ordered_point_ids.append(point_id)
                seen.add(point_id)
        if traversal.terminal_point_id not in seen:
            ordered_point_ids.append(traversal.terminal_point_id)
            seen.add(traversal.terminal_point_id)
    return np.asarray(ordered_point_ids, dtype=np.int32)


def _validate_algorithm(algorithm: str) -> None:
    if algorithm != "dhs":
        raise ValueError(f"Unsupported point scheduling algorithm {algorithm!r}.")


def _validate_max_group_size(max_group_size: int) -> None:
    if isinstance(max_group_size, bool) or not isinstance(max_group_size, int):
        raise TypeError(f"max_group_size must be int, got {max_group_size!r}.")
    if max_group_size <= 0:
        raise ValueError(f"max_group_size must be > 0, got {max_group_size!r}.")


def _compute_peel_levels(*, point_tree: PointTree) -> np.ndarray:
    point_count = len(point_tree.points)
    levels = np.zeros(point_count, dtype=np.int32)
    for point_id in range(point_count - 1, -1, -1):
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
