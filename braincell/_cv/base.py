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

"""User-facing control-volume records and build entry point."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import brainunit as u

from braincell.filter import RegionMask
from braincell.mech import Density, Point

if TYPE_CHECKING:
    from braincell._cv.policy import CVPolicy
    from braincell._cv.lower import PaintRule, PlaceRule
    from braincell.morph.morphology import Morphology


@dataclass(frozen=True)
class CV:
    """Immutable per-control-volume record exposed to users.

    Geometry, cable properties, and attached mechanisms are all frozen
    into this dataclass by :func:`build_cvs`. CVs carry no references
    back to their source morphology or rules — any post-build analysis
    must re-derive from morpho plus the CV's ``(branch_id, prox, dist)``
    range (see :func:`braincell.cv._debug.cv_to_branch`).
    """

    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length: u.Quantity
    area: u.Quantity
    cm: u.Quantity
    ra: u.Quantity
    v: u.Quantity
    temp: u.Quantity
    r_axial: u.Quantity
    r_axial_prox: u.Quantity
    r_axial_dist: u.Quantity
    radius_prox: u.Quantity
    radius_mid: u.Quantity
    radius_dist: u.Quantity
    density_mech: tuple[Density, ...]
    point_mech: tuple[Point, ...]

    @property
    def region(self) -> RegionMask:
        """Return the ``RegionMask`` covering this CV's branch interval."""
        return RegionMask(((self.branch_id, self.prox, self.dist),))

    @property
    def diam_mid(self) -> u.Quantity:
        """Diameter at the CV midpoint."""
        return 2.0 * self.radius_mid


def build_cvs(
    morpho: "Morphology",
    *,
    policy: "CVPolicy",
    paint_rules: "tuple[PaintRule, ...]" = (),
    place_rules: "tuple[PlaceRule, ...]" = (),
) -> tuple[CV, ...]:
    """Lower a morphology + policy + rules into a frozen ``tuple[CV, ...]``."""
    from braincell._cv.lower import lower
    return lower(
        morpho,
        policy=policy,
        paint_rules=paint_rules,
        place_rules=place_rules,
    )
