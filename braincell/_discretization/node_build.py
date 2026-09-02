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

"""Node-space construction and lookup helpers.

This module owns the point-space view derived from the static CV list.
It does not define new declaration-time records; instead it converts the
already assembled CV view into a ``NodeTree`` suitable for scheduling,
runtime lowering, and point-topology visualization.
"""

from dataclasses import dataclass

import numpy as np

from braincell.morph.morphology import Morphology
from .base import (
    CV,
    EPS_PARAM,
    Node,
    NodeEdge,
    NodeEdgeRole,
    NodeRole,
    NodeTree,
    Position,
    locate_cv_on_branch,
)

__all__ = ["build_node_tree_from_cvs"]

_POSITION_ORDER = {"prox": 0, "mid": 1, "dist": 2}


@dataclass
class _NodeDraft:
    """Mutable draft used while collapsing CV-local roles into nodes."""

    id: int
    roles: set[tuple[int, str]]


def build_node_tree_from_cvs(
    morpho: Morphology,
    *,
    cvs: tuple[CV, ...],
) -> NodeTree:
    """Build the point-space node tree implied by a CV tuple.

    Parameters
    ----------
    morpho : Morphology
        Morphology whose branch attachments define the point-tree
        topology.
    cvs : tuple of CV
        Flat CV tuple in stable id order.

    Returns
    -------
    NodeTree
        Point-space structural view derived from ``cvs``.

    Raises
    ------
    TypeError
        If ``morpho`` is not a :class:`Morphology`.
    ValueError
        If any branch has no CVs or if midpoint/endpoint nodes cannot be
        assembled consistently.

    Notes
    -----
    The current point tree materializes:

    - one midpoint node per CV
    - one root endpoint node
    - one endpoint node per branch end

    Internal CV boundaries do not receive standalone nodes; point
    mechanisms that land exactly on such a boundary fall back to the
    owning CV midpoint selected by the half-open CV tiling.
    """

    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_node_tree_from_cvs(...) expects Morphology, got {type(morpho).__name__!s}.")

    cv_ids_by_branch = _group_cv_ids_by_branch(
        cvs=cvs,
        n_branches=len(morpho.branches),
    )
    edge_by_child_branch = {edge.child.index: edge for edge in morpho.edges}

    drafts: list[_NodeDraft] = []
    cv_to_mid_node_id = np.full(len(cvs), -1, dtype=np.int32)
    branch_endpoint_node_id_by_x: dict[tuple[int, float], int] = {}
    logical_edge_roles: dict[tuple[int, int], list[tuple[int, str]]] = {}
    logical_edge_order: list[tuple[int, int]] = []

    def new_node(*, cv_id: int, position: str) -> int:
        node_id = len(drafts)
        drafts.append(_NodeDraft(id=node_id, roles={(cv_id, position)}))
        return node_id

    def add_node_role(node_id: int, *, cv_id: int, position: str) -> None:
        drafts[node_id].roles.add((cv_id, position))

    def add_edge_role(
        parent_node_id: int,
        child_node_id: int,
        *,
        cv_id: int,
        half: str,
    ) -> None:
        key = (parent_node_id, child_node_id)
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
    root_node_id = new_node(cv_id=root_first_cv_id, position="prox")
    branch_endpoint_node_id_by_x[(0, 0.0)] = root_node_id

    for branch_id, branch in enumerate(morpho.branches):
        branch_cv_ids = cv_ids_by_branch[branch_id]
        if len(branch_cv_ids) == 0:
            raise ValueError(f"Branch {branch_id} has no CVs.")

        if branch.parent is None:
            attachment_node_id = root_node_id
            attach_x = 0.0
        else:
            edge = edge_by_child_branch[branch_id]
            attachment_node_id = _resolve_attachment_node(
                edge.parent.index,
                parent_x=float(edge.parent_x),
                branch_endpoint_node_id_by_x=branch_endpoint_node_id_by_x,
                cv_to_mid_node_id=cv_to_mid_node_id,
                cv_ids_by_branch=cv_ids_by_branch,
                cvs=cvs,
            )
            attach_x = float(edge.child_x)

        # The branch is walked from ``attach_x`` towards ``1 - attach_x``. That
        # one direction fixes everything downstream: the CV order, which end
        # each endpoint node sits at, and which half of a CV an edge role
        # covers. It is constant for the whole branch, so it is decided here
        # rather than re-derived from ``attach_x`` at each of the four sites.
        walk_from_prox = attach_x <= EPS_PARAM
        ordered_cv_ids = branch_cv_ids if walk_from_prox else tuple(reversed(branch_cv_ids))
        entry_side: Position = "prox" if walk_from_prox else "dist"
        exit_side: Position = "dist" if walk_from_prox else "prox"

        first_cv_id = ordered_cv_ids[0]
        add_node_role(
            attachment_node_id,
            cv_id=first_cv_id,
            position=entry_side,
        )
        branch_endpoint_node_id_by_x[(branch_id, float(attach_x))] = attachment_node_id

        for cv_id in ordered_cv_ids:
            if int(cv_to_mid_node_id[cv_id]) != -1:
                raise ValueError(f"CV {cv_id} already has a midpoint node.")
            cv_to_mid_node_id[cv_id] = new_node(cv_id=cv_id, position="mid")

        terminal_cv_id = ordered_cv_ids[-1]
        terminal_node_id = new_node(
            cv_id=terminal_cv_id,
            position=exit_side,
        )
        branch_endpoint_node_id_by_x[(branch_id, float(1.0 - attach_x))] = terminal_node_id

        for index, cv_id in enumerate(ordered_cv_ids):
            midpoint_node_id = int(cv_to_mid_node_id[cv_id])
            parent_node_id = attachment_node_id if index == 0 else int(cv_to_mid_node_id[ordered_cv_ids[index - 1]])
            child_node_id = (
                terminal_node_id
                if index == len(ordered_cv_ids) - 1
                else int(cv_to_mid_node_id[ordered_cv_ids[index + 1]])
            )
            add_edge_role(
                parent_node_id,
                midpoint_node_id,
                cv_id=cv_id,
                half=entry_side,
            )
            add_edge_role(
                midpoint_node_id,
                child_node_id,
                cv_id=cv_id,
                half=exit_side,
            )

    if np.any(cv_to_mid_node_id < 0):
        raise ValueError("Node tree is missing CV midpoint nodes.")

    node_roles = tuple(
        tuple(
            NodeRole(cv_id=cv_id, position=position)
            for cv_id, position in sorted(
                draft.roles,
                key=lambda item: (item[0], _POSITION_ORDER[item[1]]),
            )
        )
        for draft in drafts
    )
    role_to_node_id: dict[tuple[int, str], int] = {}
    for node_id, roles in enumerate(node_roles):
        for role in roles:
            role_to_node_id[(int(role.cv_id), str(role.position))] = node_id

    node_point_mech_lists: list[list[object]] = [[] for _ in node_roles]
    node_density_mech_lists: list[list[object]] = [[] for _ in node_roles]
    for cv in cvs:
        midpoint_node_id = int(cv_to_mid_node_id[cv.id])
        node_density_mech_lists[midpoint_node_id].extend(cv.density_mech)
        for placement in cv.point_mech_roles:
            node_id = role_to_node_id.get((cv.id, placement.position))
            if node_id is None:
                # The current point tree only materializes CV midpoints plus
                # branch endpoints. A locset that lands exactly on an internal
                # CV boundary has no dedicated boundary node, so it falls back
                # to the owning CV midpoint selected by the half-open tiling.
                node_id = midpoint_node_id
            node_point_mech_lists[node_id].append(placement.mechanism)

    nodes = tuple(
        Node(
            id=node_id,
            kind="mid" if all(role.position == "mid" for role in roles) else "boundary",
            roles=roles,
            density_mech=tuple(node_density_mech_lists[node_id]),
            point_mech=tuple(node_point_mech_lists[node_id]),
        )
        for node_id, roles in enumerate(node_roles)
    )

    edges = tuple(
        NodeEdge(
            id=edge_id,
            parent_node_id=parent_node_id,
            child_node_id=child_node_id,
            roles=tuple(
                NodeEdgeRole(cv_id=cv_id, half=half)
                for cv_id, half in sorted(
                    cv_roles,
                    key=lambda item: (item[0], item[1]),
                )
            ),
        )
        for edge_id, ((parent_node_id, child_node_id), cv_roles) in enumerate(
            (key, logical_edge_roles[key]) for key in logical_edge_order
        )
    )

    return NodeTree(
        nodes=nodes,
        edges=edges,
        root_node_id=root_node_id,
        cv_to_mid_node_id=cv_to_mid_node_id,
    )


def _group_cv_ids_by_branch(
    *,
    cvs: tuple[CV, ...],
    n_branches: int,
) -> tuple[tuple[int, ...], ...]:
    grouped: list[list[int]] = [[] for _ in range(n_branches)]
    for cv in cvs:
        grouped[cv.branch_id].append(cv.id)
    return tuple(tuple(ids) for ids in grouped)


def _resolve_attachment_node(
    parent_branch_id: int,
    *,
    parent_x: float,
    branch_endpoint_node_id_by_x: dict[tuple[int, float], int],
    cv_to_mid_node_id: np.ndarray,
    cv_ids_by_branch: tuple[tuple[int, ...], ...],
    cvs: tuple[CV, ...],
) -> int:
    if parent_x <= 0.0 + EPS_PARAM:
        return branch_endpoint_node_id_by_x[(parent_branch_id, 0.0)]
    if parent_x >= 1.0 - EPS_PARAM:
        return branch_endpoint_node_id_by_x[(parent_branch_id, 1.0)]
    cv_id = locate_cv_on_branch(
        cv_ids_by_branch[parent_branch_id],
        cvs,
        x=float(parent_x),
        epsilon=EPS_PARAM,
    )
    return int(cv_to_mid_node_id[cv_id])
