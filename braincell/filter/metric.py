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

"""Composable spatial metrics for morphology-backed callable contexts."""

__all__ = ["branch_x", "radius", "path_distance_from_soma", "position"]


def branch_x(context: object) -> object:
    """Return the normalized branch coordinate represented by ``context``."""
    try:
        return getattr(context, "branch_x")
    except AttributeError:
        try:
            return getattr(context, "midpoint")
        except AttributeError as exc:
            raise TypeError(f"{type(context).__name__} does not expose a branch coordinate.") from exc


def radius(context: object) -> object:
    """Return radius at the location represented by ``context``."""
    try:
        return getattr(context, "radius")
    except AttributeError:
        try:
            return getattr(context, "radius_mid")
        except AttributeError as exc:
            raise TypeError(f"{type(context).__name__} does not expose a radius.") from exc


def path_distance_from_soma(context: object) -> object:
    """Return tree distance from the soma/root reference region."""
    try:
        return getattr(context, "path_distance_from_soma")
    except AttributeError as exc:
        raise TypeError(f"{type(context).__name__} does not expose soma-relative distance.") from exc


def position(context: object) -> object:
    """Return the morphology-local 3-D position represented by ``context``."""
    try:
        return getattr(context, "position")
    except AttributeError as exc:
        raise TypeError(f"{type(context).__name__} does not expose a 3-D position.") from exc
