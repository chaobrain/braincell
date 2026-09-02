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

"""Build public CV spatial contexts from geometry-stage CV records."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import brainunit as u

from braincell.mech import CVContext
from braincell.morph._spatial import MorphologySpatialGeometry, interpolate_branch
from braincell.morph.morphology import Morphology

if TYPE_CHECKING:
    from .geometry import _GeoCV

__all__ = ["build_cv_contexts"]


def build_cv_contexts(
    morpho: Morphology,
    cvs: "Sequence[_GeoCV]",
) -> tuple[CVContext, ...]:
    """Build one :class:`~braincell.mech.CVContext` per control volume.

    Parameters
    ----------
    morpho : Morphology
        Morphology that owns the control volumes.
    cvs : sequence of _GeoCV
        Geometry-stage records in any order, with distinct ids covering
        ``range(len(cvs))``.

    Returns
    -------
    tuple of CVContext
        Contexts ordered by stable control-volume id.

    Raises
    ------
    TypeError
        If ``morpho`` is not a :class:`Morphology`.
    ValueError
        If a CV id is out of range or repeated, or if a CV names a branch
        the morphology does not have.

    Notes
    -----
    ``_GeoCV`` is the only accepted input. This function used to also claim
    to take finalized public ``CV`` records, and paid for the claim with a
    ``getattr`` shim per quantity that translated between the two field
    spellings -- but both call sites pass ``CVGeometryResult.geos``, and
    ``_GeoCV`` declares only the scalar spelling, so the ``CV`` half of that
    shim was unreachable.
    """
    if not isinstance(morpho, Morphology):
        raise TypeError(f"build_cv_contexts(...) expects Morphology, got {type(morpho).__name__!s}.")
    if len(cvs) == 0:
        return ()

    n_branches = len(morpho.branches)
    geometry = MorphologySpatialGeometry.build(morpho)
    um2 = u.um**2

    contexts: list[CVContext | None] = [None] * len(cvs)
    for source in cvs:
        cv_id = int(source.id)
        if not 0 <= cv_id < len(contexts):
            raise ValueError(f"CV id {cv_id!r} is outside [0, {len(contexts)!r}).")
        if contexts[cv_id] is not None:
            raise ValueError(f"Duplicate CV id {cv_id!r}.")

        branch_id = int(source.branch_id)
        if not 0 <= branch_id < n_branches:
            raise ValueError(f"CV {cv_id!r} has invalid branch id {branch_id!r}.")

        midpoint = source.midpoint
        _, local_position = interpolate_branch(morpho, branch_id, midpoint)
        contexts[cv_id] = CVContext(
            cv_id=cv_id,
            branch_id=branch_id,
            branch_name=str(morpho.branch(index=branch_id).name),
            branch_type=str(source.branch_type),
            prox=float(source.prox),
            dist=float(source.dist),
            midpoint=midpoint,
            length=u.Quantity(float(source.length_um), u.um),
            area=u.Quantity(float(source.lateral_area_um2), um2),
            radius_prox=u.Quantity(float(source.r_prox_um), u.um),
            radius_mid=u.Quantity(float(source.r_mid_um), u.um),
            radius_dist=u.Quantity(float(source.r_dist_um), u.um),
            diam_arc_mean=u.Quantity(float(source.diam_arc_mean_um), u.um),
            path_distance_to_root=geometry.path_distance_to_root(branch_id, midpoint),
            path_distance_from_soma=geometry.path_distance_from_soma(branch_id, midpoint),
            _local_position=local_position,
        )

    # Every iteration either raised or wrote a distinct in-range index, so
    # ``len(cvs)`` distinct indices into ``len(cvs)`` slots leave none empty.
    return tuple(context for context in contexts if context is not None)
