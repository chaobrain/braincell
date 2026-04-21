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

"""Debug / visualization helpers that reconstruct geometry from a CV."""

import brainunit as u
import numpy as np

from braincell._cv.base import CV
from braincell._cv.lower import _build_frusta
from braincell.morph.branch import Branch, branch_class_for_type
from braincell.morph.morphology import Morphology

__all__ = ["cv_to_branch"]


def cv_to_branch(cv: CV, morpho: Morphology) -> Branch:
    """Reconstruct a standalone ``Branch`` for a CV range.

    Slices ``morpho.branches[cv.branch_id]`` from ``cv.prox`` to ``cv.dist``
    using the same frustum math as :func:`braincell.cv._lower._build_frusta`,
    then returns a typed subclass chosen via
    :func:`braincell.morph.branch.branch_class_for_type`.
    """
    source = morpho.branches[cv.branch_id]
    frusta = _build_frusta(source, prox=cv.prox, dist=cv.dist)
    lengths = np.asarray([p.length_um for p in frusta], dtype=float)
    r0 = np.asarray([p.r_prox_um for p in frusta], dtype=float)
    r1 = np.asarray([p.r_dist_um for p in frusta], dtype=float)

    has_points = all(
        p.point_prox_um is not None and p.point_dist_um is not None for p in frusta
    )
    branch_cls = branch_class_for_type(cv.branch_type)
    if has_points:
        first = frusta[0].point_prox_um
        points = [np.asarray(first, dtype=float)]
        for p in frusta:
            points.append(np.asarray(p.point_dist_um, dtype=float))
        radii = np.concatenate((r0[:1], r1), axis=0)
        return branch_cls.from_points(
            points=u.Quantity(np.asarray(points, dtype=float), u.um),
            radii=u.Quantity(radii, u.um),
        )
    return branch_cls.from_lengths(
        lengths=u.Quantity(lengths, u.um),
        radii_proximal=u.Quantity(r0, u.um),
        radii_distal=u.Quantity(r1, u.um),
    )
