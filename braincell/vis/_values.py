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

"""Helpers that normalise color-by-values overlay input.

A :class:`~braincell.vis.scene.ValueSpec` (or a bare array) can be
supplied at one of three granularities:

* **per-branch** — ``len(values) == n_branches``. Each branch is shaded
  with a single scalar.
* **per-segment** — ``len(values) == total segment count``. Each
  segment gets its own scalar.
* **per-centerline-point** — ``len(values) == total segment count +
  n_branches``. Each centerline point gets its own scalar.

The helpers in this module turn any of those shapes into a
``dict[int, np.ndarray]`` keyed by branch index where the array is the
per-*point* scalar for that branch's centerline. Scene builders can
then convert per-point → per-segment trivially when building
:class:`~braincell.vis.scene.PolylineValues2D` /
:class:`~braincell.vis.scene.PolygonValuesBatch2D` /
:class:`~braincell.vis.scene.ValueBatch3D` primitives.

Values are unit-aware: :mod:`brainunit` quantities are stripped down to
raw floats and the unit string is reported back so the scene builder
can forward it to the value spec's colour-bar label.
"""


import numpy as np

from braincell.morph.morphology import Morphology
from .scene import BranchValues, ValueSpec

try:  # pragma: no cover - branched off at import
    import brainunit as u
except ModuleNotFoundError:  # pragma: no cover
    u = None  # type: ignore[assignment]


def _strip_quantity(values) -> tuple[np.ndarray, str | None]:
    """Return ``(raw_array, unit_string_or_None)`` for a possibly-unit-carrying input."""
    if u is not None and isinstance(values, u.Quantity):
        unit = u.get_unit(values)
        raw = np.asarray(u.get_mantissa(values), dtype=float)
        return raw, str(unit)
    return np.asarray(values, dtype=float), None


def resolve_values(
    morpho: Morphology,
    spec: ValueSpec,
) -> tuple[dict[int, BranchValues], str | None]:
    """Turn a :class:`ValueSpec` into per-branch per-point scalar arrays.

    Parameters
    ----------
    morpho : Morphology
        Morphology the scalar array is defined against.
    spec : ValueSpec
        Value-and-styling container.

    Returns
    -------
    per_branch : dict[int, BranchValues]
        One entry per branch, keyed by branch index.
    unit_label : str or None
        Unit string extracted from :class:`brainunit` quantities (or
        ``None`` if the caller passed a raw numpy array).

    Raises
    ------
    ValueError
        If the array length does not match any of the supported
        shapes (per-branch, per-segment, per-point).
    """
    raw, unit_label = _strip_quantity(spec.values)
    raw = np.asarray(raw, dtype=float)

    branches = morpho.branches
    n_branches = len(branches)
    segment_counts: list[int] = []
    point_counts: list[int] = []
    for branch_view in branches:
        branch = branch_view.branch
        n_segments = int(branch.radii_distal.size)
        segment_counts.append(n_segments)
        point_counts.append(n_segments + 1)

    total_segments = sum(segment_counts)
    total_points = sum(point_counts)

    if raw.ndim != 1:
        raise ValueError(
            f"ValueSpec.values must be 1-D; got shape {raw.shape!r}. "
            "plot_movie uses a different entry point for (T, N) arrays."
        )

    length = raw.shape[0]
    if length == n_branches:
        return _expand_per_branch(branches, segment_counts, raw), unit_label
    if length == total_segments:
        return _expand_per_segment(branches, segment_counts, raw), unit_label
    if length == total_points:
        return _expand_per_point(branches, point_counts, raw), unit_label

    raise ValueError(
        f"ValueSpec.values has length {length}, but the morphology has "
        f"{n_branches} branches, {total_segments} segments, and "
        f"{total_points} centerline points. Expected one of those shapes."
    )


def _expand_per_branch(
    branches,
    segment_counts: list[int],
    values: np.ndarray,
) -> dict[int, BranchValues]:
    per_branch: dict[int, BranchValues] = {}
    for branch_view, n_segments in zip(branches, segment_counts):
        scalar = float(values[branch_view.index])
        point_values = np.full(n_segments + 1, scalar, dtype=float)
        per_branch[branch_view.index] = BranchValues(
            branch_index=branch_view.index,
            point_values=point_values,
        )
    return per_branch


def _expand_per_segment(
    branches,
    segment_counts: list[int],
    values: np.ndarray,
) -> dict[int, BranchValues]:
    per_branch: dict[int, BranchValues] = {}
    cursor = 0
    for branch_view, n_segments in zip(branches, segment_counts):
        seg_slice = values[cursor: cursor + n_segments]
        cursor += n_segments
        # Promote per-segment scalars to per-point by taking segment-ends:
        # point[i] = mean(seg[i-1], seg[i]) for interior points, segment
        # value at the endpoints. This produces visually continuous
        # interpolation along the centerline.
        if n_segments == 0:
            point_values = np.array([], dtype=float)
        elif n_segments == 1:
            point_values = np.array([seg_slice[0], seg_slice[0]], dtype=float)
        else:
            point_values = np.empty(n_segments + 1, dtype=float)
            point_values[0] = seg_slice[0]
            point_values[-1] = seg_slice[-1]
            point_values[1:-1] = 0.5 * (seg_slice[:-1] + seg_slice[1:])
        per_branch[branch_view.index] = BranchValues(
            branch_index=branch_view.index,
            point_values=point_values,
        )
    return per_branch


def _expand_per_point(
    branches,
    point_counts: list[int],
    values: np.ndarray,
) -> dict[int, BranchValues]:
    per_branch: dict[int, BranchValues] = {}
    cursor = 0
    for branch_view, n_points in zip(branches, point_counts):
        point_values = np.asarray(values[cursor: cursor + n_points], dtype=float)
        cursor += n_points
        per_branch[branch_view.index] = BranchValues(
            branch_index=branch_view.index,
            point_values=point_values,
        )
    return per_branch


def resolved_colorbar_label(spec: ValueSpec, unit_label: str | None) -> str | None:
    """Compose the final colour-bar label.

    Combines the user-supplied ``spec.label`` and ``spec.unit_label``
    (falling back to the brainunit-derived ``unit_label`` from the
    array if the spec's field is ``None``). Returns ``None`` when no
    label information is available at all.
    """
    label = spec.label
    unit = spec.unit_label if spec.unit_label is not None else unit_label
    if label is None and unit is None:
        return None
    if label is None:
        return f"[{unit}]"
    if unit is None:
        return str(label)
    return f"{label} [{unit}]"
