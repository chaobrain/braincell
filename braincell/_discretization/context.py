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

"""Build public CV spatial contexts from geometry or finalized CV records."""

from collections.abc import Sequence

import brainunit as u

from braincell.mech import CVContext
from braincell.morph.morphology import Morphology

__all__ = ["build_cv_contexts"]


def build_cv_contexts(
    morpho: Morphology,
    cvs: Sequence[object],
) -> tuple[CVContext, ...]:
    """Build one :class:`~braincell.mech.CVContext` per control volume.

    Parameters
    ----------
    morpho : Morphology
        Morphology that owns the control volumes.
    cvs : sequence of object
        Geometry-stage ``_GeoCV`` records or finalized public ``CV`` records.

    Returns
    -------
    tuple of CVContext
        Contexts ordered by stable control-volume id.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(
            f"build_cv_contexts(...) expects Morphology, got {type(morpho).__name__!s}."
        )
    if len(cvs) == 0:
        return ()

    n_branches = len(morpho.branches)
    branch_lengths_um = {
        branch_id: float(
            morpho.branch(index=branch_id).branch.length.to_decimal(u.um)
        )
        for branch_id in range(n_branches)
    }
    edge_by_child = {int(edge.child.index): edge for edge in morpho.edges}
    root_id = int(morpho.root.index)

    entry_x: dict[int, float] = {root_id: 0.0}
    root_base_um: dict[int, float] = {root_id: 0.0}
    soma_base_um: dict[int, float] = {root_id: 0.0}
    resolving: set[int] = set()

    def resolve_branch_base(branch_id: int) -> tuple[float, float]:
        cached_root = root_base_um.get(branch_id)
        if cached_root is not None:
            return cached_root, soma_base_um[branch_id]
        if branch_id in resolving:
            raise ValueError(f"Morphology contains a cycle at branch {branch_id!r}.")
        try:
            edge = edge_by_child[branch_id]
        except KeyError as exc:
            raise ValueError(
                f"Non-root branch {branch_id!r} has no parent edge."
            ) from exc

        resolving.add(branch_id)
        parent_id = int(edge.parent.index)
        parent_root_base, parent_soma_base = resolve_branch_base(parent_id)
        entry_x[branch_id] = float(edge.child_x)
        parent_step_um = (
            abs(float(edge.parent_x) - entry_x[parent_id])
            * branch_lengths_um[parent_id]
        )
        branch_root_base = parent_root_base + parent_step_um
        branch_soma_base = (
            0.0
            if str(edge.parent.type) == "soma"
            else parent_soma_base + parent_step_um
        )
        root_base_um[branch_id] = branch_root_base
        soma_base_um[branch_id] = branch_soma_base
        resolving.remove(branch_id)
        return branch_root_base, branch_soma_base

    contexts: list[CVContext | None] = [None] * len(cvs)
    for source in cvs:
        cv_id = int(getattr(source, "id"))
        if not 0 <= cv_id < len(contexts):
            raise ValueError(f"CV id {cv_id!r} is outside [0, {len(contexts)!r}).")
        if contexts[cv_id] is not None:
            raise ValueError(f"Duplicate CV id {cv_id!r}.")

        branch_id = int(getattr(source, "branch_id"))
        if not 0 <= branch_id < n_branches:
            raise ValueError(f"CV {cv_id!r} has invalid branch id {branch_id!r}.")
        branch_root_base, branch_soma_base = resolve_branch_base(branch_id)
        midpoint = float(
            getattr(
                source,
                "midpoint",
                0.5 * (float(getattr(source, "prox")) + float(getattr(source, "dist"))),
            )
        )
        local_distance_um = abs(midpoint - entry_x[branch_id]) * branch_lengths_um[branch_id]
        branch_type = str(getattr(source, "branch_type"))

        radius_mid = _source_quantity(source, "radius_mid", "r_mid_um", u.um)
        contexts[cv_id] = CVContext(
            cv_id=cv_id,
            branch_id=branch_id,
            branch_name=str(morpho.branch(index=branch_id).name),
            branch_type=branch_type,
            prox=float(getattr(source, "prox")),
            dist=float(getattr(source, "dist")),
            midpoint=midpoint,
            length=_source_quantity(source, "length", "length_um", u.um),
            area=_source_quantity(source, "area", "lateral_area_um2", u.um ** 2),
            radius_prox=_source_quantity(source, "radius_prox", "r_prox_um", u.um),
            radius_mid=radius_mid,
            radius_dist=_source_quantity(source, "radius_dist", "r_dist_um", u.um),
            diam_mid=2.0 * radius_mid,
            diam_arc_mean=_source_quantity(
                source,
                "diam_arc_mean",
                "diam_arc_mean_um",
                u.um,
            ),
            path_distance_to_root=u.Quantity(
                branch_root_base + local_distance_um,
                u.um,
            ),
            path_distance_from_soma=u.Quantity(
                0.0 if branch_type == "soma" else branch_soma_base + local_distance_um,
                u.um,
            ),
        )

    if any(context is None for context in contexts):
        missing = [index for index, context in enumerate(contexts) if context is None]
        raise ValueError(f"Missing CV context records for ids {missing!r}.")
    return tuple(context for context in contexts if context is not None)


def _source_quantity(source: object, quantity_name: str, scalar_name: str, unit):
    value = getattr(source, quantity_name, None)
    if value is not None:
        return u.Quantity(float(value.to_decimal(unit)), unit)
    return u.Quantity(float(getattr(source, scalar_name)), unit)
