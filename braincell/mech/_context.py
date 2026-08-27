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

"""Spatial context passed to callable mechanism parameters."""

from dataclasses import dataclass

import brainunit as u

__all__ = ["CVContext"]


@dataclass(frozen=True)
class CVContext:
    """Read-only geometry context for one control volume.

    Callable cable and density parameters receive one context per control
    volume. All physical fields carry :mod:`brainunit` units, while ``prox``,
    ``dist``, and ``midpoint`` are normalized branch coordinates in ``[0, 1]``.

    Parameters
    ----------
    cv_id : int
        Stable control-volume index.
    branch_id : int
        Owning morphology branch index.
    branch_name : str
        Owning morphology branch name.
    branch_type : str
        Owning morphology branch type.
    prox, dist, midpoint : float
        Normalized control-volume bounds and midpoint.
    length : Quantity
        Control-volume cable length.
    area : Quantity
        Control-volume lateral membrane area.
    radius_prox, radius_mid, radius_dist : Quantity
        Radius at the proximal boundary, midpoint, and distal boundary.
    diam_mid : Quantity
        Diameter at the control-volume midpoint.
    diam_arc_mean : Quantity
        Arc-length-weighted mean diameter over the control volume.
    path_distance_to_root : Quantity
        Tree path distance from root-branch coordinate ``x=0`` to the
        control-volume midpoint.
    path_distance_from_soma : Quantity
        Shortest tree path distance from the union of all soma branches. If
        there is no soma branch, the whole root branch is the reference.
    local_position, position : Quantity
        Morphology-local 3-D midpoint position. Access raises ``ValueError``
        when full point geometry is unavailable.
    """

    cv_id: int
    branch_id: int
    branch_name: str
    branch_type: str
    prox: float
    dist: float
    midpoint: float
    length: u.Quantity
    area: u.Quantity
    radius_prox: u.Quantity
    radius_mid: u.Quantity
    radius_dist: u.Quantity
    diam_mid: u.Quantity
    diam_arc_mean: u.Quantity
    path_distance_to_root: u.Quantity
    path_distance_from_soma: u.Quantity
    _local_position: u.Quantity | None = None

    @property
    def local_position(self) -> u.Quantity:
        """Morphology-local 3-D control-volume midpoint position."""
        if self._local_position is None:
            raise ValueError("CVContext.local_position requires full 3-D point geometry.")
        return self._local_position

    @property
    def position(self) -> u.Quantity:
        """World position, equal to ``local_position`` until transforms land."""
        if self._local_position is None:
            raise ValueError("CVContext.position requires full 3-D point geometry.")
        return self._local_position
