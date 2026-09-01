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

"""Arc-length parameterisation of a polyline, in any number of dimensions.

Overlays address a location on a branch as a fraction ``x`` of its
centerline length, so every renderer needs the same three operations:
build the cumulative arc length of a polyline, find the segment that a
given ``x`` lands in, and interpolate between that segment's endpoints.
The 2D centerlines in :mod:`braincell.vis.scene2d`, the 3D branch
polylines in :mod:`braincell.vis.scene3d`, and the layout sampler in
:mod:`braincell.vis.layout._geometry` all did this independently; this
module is the single implementation they now share.

Nothing here is dimension-specific — the point array is ``(n_points, D)``
and ``D`` is read from its shape, so the same code serves the ``(n, 2)``
layout centerlines and the ``(n, 3)`` reconstructed geometry. Only
:mod:`numpy` is imported, so any module in ``braincell.vis`` can depend
on it without risking an import cycle.
"""

import numpy as np

# Half-open tolerance used when deciding whether an existing polyline
# vertex falls strictly inside a requested span. Vertices within this
# distance of a span endpoint are dropped, because the interpolated
# endpoint already covers them.
_SPAN_EPSILON_UM = 1e-12


def cumulative_arclength_um(points_um: np.ndarray) -> np.ndarray:
    """Return the cumulative arc length along a polyline.

    Parameters
    ----------
    points_um : numpy.ndarray
        Polyline vertices, shape ``(n_points, n_dim)``, in micrometres.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_points,)``. Element ``i`` is the distance from the
        first vertex to vertex ``i``; element ``0`` is always ``0.0``.
    """
    points_um = np.asarray(points_um, dtype=float)
    segment_lengths_um = np.linalg.norm(np.diff(points_um, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths_um)])


def ordered_span(prox: float, dist: float) -> tuple[float, float]:
    """Normalize an interval to ``(lo, hi)`` with both ends clipped to ``[0, 1]``."""
    lo, hi = (float(prox), float(dist)) if prox <= dist else (float(dist), float(prox))
    return float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))


def segment_index_at(
    cumulative_um: np.ndarray,
    *,
    total_um: float,
    n_segments: int,
    x: float,
) -> tuple[int, float]:
    """Locate the segment containing fractional coordinate *x*.

    Parameters
    ----------
    cumulative_um : numpy.ndarray
        Cumulative arc length, as returned by
        :func:`cumulative_arclength_um`.
    total_um : float
        Total length the fraction is taken against. Passed explicitly
        rather than read from ``cumulative_um[-1]`` because a layout
        branch carries its own ``total_length_um``, which is derived
        from the source spec and need not be bit-identical to the
        recomputed sum.
    n_segments : int
        Number of segments, used to clamp the result to a valid index.
    x : float
        Fractional coordinate; values outside ``[0, 1]`` are clipped.

    Returns
    -------
    index : int
        Segment index in ``[0, n_segments - 1]``.
    arc_length_um : float
        Absolute arc length corresponding to *x*.
    """
    arc_length_um = float(np.clip(x, 0.0, 1.0)) * total_um
    index = int(np.searchsorted(cumulative_um[1:], arc_length_um, side="right"))
    return min(max(index, 0), n_segments - 1), arc_length_um


def interpolate_at(
    values: np.ndarray,
    cumulative_um: np.ndarray,
    *,
    total_um: float,
    x: float,
):
    """Linearly interpolate per-vertex *values* at fractional coordinate *x*.

    Works for any trailing shape: ``(n_points, n_dim)`` positions
    interpolate componentwise, ``(n_points,)`` scalars (radii, widths)
    interpolate to a single value.

    Parameters
    ----------
    values : numpy.ndarray
        One entry per polyline vertex, leading axis of length
        ``n_points``.
    cumulative_um : numpy.ndarray
        Cumulative arc length for the same vertices.
    total_um : float
        Total length the fraction is taken against.
    x : float
        Fractional coordinate; values outside ``[0, 1]`` are clipped.

    Returns
    -------
    numpy.ndarray or numpy.float64
        The interpolated entry. A zero-length segment returns a copy of
        its start entry rather than dividing by zero.
    """
    index, arc_length_um = segment_index_at(
        cumulative_um,
        total_um=total_um,
        n_segments=values.shape[0] - 1,
        x=x,
    )
    start_um = float(cumulative_um[index])
    end_um = float(cumulative_um[index + 1])
    span_um = end_um - start_um
    if span_um <= 0.0:
        return values[index].copy()
    alpha = (arc_length_um - start_um) / span_um
    return (1.0 - alpha) * values[index] + alpha * values[index + 1]


class ArcPolyline:
    """A polyline addressed by normalised arc length.

    Wraps a ``(n_points, n_dim)`` vertex array together with its
    cumulative arc length so that overlay code can ask for "the point
    40 % along this branch" without re-deriving the parameterisation.

    Parameters
    ----------
    points_um : numpy.ndarray
        Polyline vertices, shape ``(n_points, n_dim)``, in micrometres.
    cumulative_um : numpy.ndarray or None
        Precomputed cumulative arc length. Supplied by callers that
        already hold one (a layout branch does); recomputed with
        :func:`cumulative_arclength_um` when ``None``.

    Notes
    -----
    Degenerate input is tolerated rather than rejected: a zero-length
    or empty polyline reports :attr:`is_degenerate`, and the samplers
    fall back to the first vertex (or a zero vector) instead of
    dividing by zero. Overlays are user-driven, so an empty branch must
    not raise mid-render.
    """

    __slots__ = ("points_um", "cumulative_um", "total_um")

    def __init__(self, points_um: np.ndarray, cumulative_um: np.ndarray | None = None) -> None:
        self.points_um = np.asarray(points_um, dtype=float)
        self.cumulative_um = (
            cumulative_arclength_um(self.points_um) if cumulative_um is None else np.asarray(cumulative_um, dtype=float)
        )
        self.total_um = float(self.cumulative_um[-1]) if self.cumulative_um.size else 0.0

    @property
    def n_dim(self) -> int:
        """Number of spatial dimensions (2 for layout space, 3 for geometry)."""
        return self.points_um.shape[1] if self.points_um.ndim == 2 else 0

    @property
    def is_degenerate(self) -> bool:
        """Whether the polyline has no length, so arc-length sampling is undefined."""
        return self.total_um <= 0.0 or self.points_um.shape[0] == 0

    def point_at(self, x: float) -> np.ndarray:
        """Return the position at fractional coordinate *x* in ``[0, 1]``."""
        if self.is_degenerate:
            if self.points_um.size:
                return self.points_um[0].copy()
            return np.zeros(self.n_dim, dtype=float)
        return interpolate_at(self.points_um, self.cumulative_um, total_um=self.total_um, x=x)

    def scalar_at(self, values: np.ndarray, x: float, *, empty: float = 1.0) -> float:
        """Return a per-vertex scalar (radius, width) interpolated at *x*.

        Parameters
        ----------
        values : numpy.ndarray
            One scalar per polyline vertex.
        x : float
            Fractional coordinate in ``[0, 1]``.
        empty : float
            Value returned when *values* is empty.
        """
        values = np.asarray(values, dtype=float)
        if self.total_um <= 0.0 or values.size == 0:
            return float(values[0]) if values.size else empty
        return float(interpolate_at(values, self.cumulative_um, total_um=self.total_um, x=x))

    def interior_mask(self, start_arc_um: float, end_arc_um: float) -> np.ndarray:
        """Boolean mask of vertices lying strictly inside an arc-length span."""
        return (self.cumulative_um > start_arc_um + _SPAN_EPSILON_UM) & (
            self.cumulative_um < end_arc_um - _SPAN_EPSILON_UM
        )

    def subspan(self, prox: float, dist: float) -> tuple[float, float, np.ndarray]:
        """Resolve a fractional span into ``(lo, hi, interior_vertex_mask)``.

        The endpoints are ordered and clipped to ``[0, 1]``; the mask
        selects the existing vertices that fall strictly between them,
        which the caller stitches between its own interpolated endpoints.
        """
        lo, hi = ordered_span(prox, dist)
        return lo, hi, self.interior_mask(lo * self.total_um, hi * self.total_um)
