from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from braincell.morpho import Morpho
from .cv import CV
from .cv_geo import EPSILON

_POSITION_ORDER = {"proximal": 0, "mid": 1, "distal": 2}


@dataclass(frozen=True)
class CVPoint:
    cv_id: int
    position: str


@dataclass(frozen=True)
class ComputePoint:
    id: int
    cv_points: tuple[CVPoint, ...]


@dataclass(frozen=True)
class CVEdge:
    cv_id: int
    half: str


@dataclass(frozen=True)
class ComputeEdge:
    id: int
    parent_point_id: int
    child_point_id: int
    cv_edges: tuple[CVEdge, ...]


@dataclass(frozen=True)
class PointTree:
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


@dataclass
class _PointDraft:
    id: int
    cv_points: set[tuple[int, str]]


@dataclass(frozen=True)
class _BranchTraversal:
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
