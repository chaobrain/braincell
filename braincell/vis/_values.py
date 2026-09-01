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

:func:`resolve_values` is the one-shot entry point. Callers that
resolve many arrays against the *same* morphology — ``plot_movie``,
which does it once per frame — should build a :class:`ValueLayout`
once and call :meth:`ValueLayout.expand` per array, so the branch walk
that derives segment counts is not repeated.
"""

import dataclasses
from typing import Iterable

import brainunit as u
import numpy as np

from braincell.morph.morphology import Morphology
from .scene import BranchValues, OverlaySpec, ValueSpec


def _strip_quantity(values) -> tuple[np.ndarray, str | None]:
    """Return ``(raw_array, unit_string_or_None)`` for a possibly-unit-carrying input.

    This is the single unit-stripping boundary for the whole ``vis``
    package — ``traces``, ``movie``, and ``point_topology`` all route
    through it so that "what counts as a quantity" is decided in one
    place.
    """
    if isinstance(values, u.Quantity):
        unit = u.get_unit(values)
        raw = np.asarray(u.get_mantissa(values), dtype=float)
        return raw, str(unit)
    return np.asarray(values, dtype=float), None


@dataclasses.dataclass(frozen=True)
class ValueLayout:
    """The frame-invariant structure needed to expand a scalar array.

    Which branch owns which slice of a value array depends only on the
    morphology's branch order and per-branch segment counts — never on
    the values themselves. :func:`plot_movie` re-resolves a new array
    every frame against the *same* morphology, so deriving this once
    and reusing it turns a per-frame walk over every branch into a
    handful of array slices.

    Attributes
    ----------
    branch_indices : tuple of int
        Branch index of each branch, in ``morpho.branches`` order.
    segment_counts : tuple of int
        Segment count of each branch, aligned with *branch_indices*.

    See Also
    --------
    resolve_values : Resolve a single :class:`ValueSpec` in one call.
    """

    branch_indices: tuple[int, ...]
    segment_counts: tuple[int, ...]

    @classmethod
    def from_morphology(cls, morpho: Morphology) -> "ValueLayout":
        """Derive the layout from a morphology in a single pass.

        Branch positions come from :func:`enumerate` rather than
        :attr:`MorphoBranch.index`, which rebuilds the morphology's
        node→index map on every access; ``morpho.branches`` is already
        in that same default order.
        """
        branch_views = morpho.branches
        return cls(
            branch_indices=tuple(range(len(branch_views))),
            segment_counts=tuple(int(branch_view.branch.radii_distal.size) for branch_view in branch_views),
        )

    @property
    def n_branches(self) -> int:
        return len(self.branch_indices)

    @property
    def total_segments(self) -> int:
        return sum(self.segment_counts)

    @property
    def total_points(self) -> int:
        return self.total_segments + self.n_branches

    def expand(self, values) -> dict[int, BranchValues]:
        """Expand a 1-D scalar array into per-branch per-point arrays.

        Parameters
        ----------
        values : ArrayLike
            Scalars at per-branch, per-segment, or per-centerline-point
            granularity; the granularity is inferred from the length.

        Returns
        -------
        dict[int, BranchValues]
            One entry per branch, keyed by branch index.

        Raises
        ------
        ValueError
            If *values* is not 1-D, or its length matches none of the
            three supported shapes.
        """
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 1:
            raise ValueError(
                f"ValueSpec.values must be 1-D; got shape {raw.shape!r}. "
                "plot_movie uses a different entry point for (T, N) arrays."
            )

        length = raw.shape[0]
        if length == self.n_branches:
            return self._expand_per_branch(raw)
        if length == self.total_segments:
            return self._expand_per_segment(raw)
        if length == self.total_points:
            return self._expand_per_point(raw)

        raise ValueError(
            f"ValueSpec.values has length {length}, but the morphology has "
            f"{self.n_branches} branches, {self.total_segments} segments, and "
            f"{self.total_points} centerline points. Expected one of those shapes."
        )

    def resolve(self, spec: ValueSpec) -> tuple[dict[int, BranchValues], str | None]:
        """Strip units off ``spec.values`` and expand it. See :func:`resolve_values`."""
        raw, unit_label = _strip_quantity(spec.values)
        return self.expand(raw), unit_label

    def _expand_per_branch(self, values: np.ndarray) -> dict[int, BranchValues]:
        per_branch: dict[int, BranchValues] = {}
        for branch_index, n_segments in zip(self.branch_indices, self.segment_counts):
            scalar = float(values[branch_index])
            per_branch[branch_index] = BranchValues(
                branch_index=branch_index,
                point_values=np.full(n_segments + 1, scalar, dtype=float),
            )
        return per_branch

    def _expand_per_segment(self, values: np.ndarray) -> dict[int, BranchValues]:
        per_branch: dict[int, BranchValues] = {}
        cursor = 0
        for branch_index, n_segments in zip(self.branch_indices, self.segment_counts):
            seg_slice = values[cursor : cursor + n_segments]
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
            per_branch[branch_index] = BranchValues(branch_index=branch_index, point_values=point_values)
        return per_branch

    def _expand_per_point(self, values: np.ndarray) -> dict[int, BranchValues]:
        per_branch: dict[int, BranchValues] = {}
        cursor = 0
        for branch_index, n_segments in zip(self.branch_indices, self.segment_counts):
            n_points = n_segments + 1
            point_values = np.asarray(values[cursor : cursor + n_points], dtype=float)
            cursor += n_points
            per_branch[branch_index] = BranchValues(branch_index=branch_index, point_values=point_values)
        return per_branch


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

    See Also
    --------
    ValueLayout : Reuse the derived structure across many arrays.
    """
    return ValueLayout.from_morphology(morpho).resolve(spec)


def compose_colorbar_label(label: str | None, unit: str | None) -> str | None:
    """Combine a label and a unit string into one colour-bar title.

    ``("V", "mV")`` → ``"V [mV]"``; ``(None, "mV")`` → ``"[mV]"``;
    ``("V", None)`` → ``"V"``; ``(None, None)`` → ``None``.
    """
    if label is None and unit is None:
        return None
    if label is None:
        return f"[{unit}]"
    if unit is None:
        return str(label)
    return f"{label} [{unit}]"


def resolved_colorbar_label(spec: ValueSpec, unit_label: str | None) -> str | None:
    """Compose the final colour-bar label.

    Combines the user-supplied ``spec.label`` and ``spec.unit_label``
    (falling back to the brainunit-derived ``unit_label`` from the
    array if the spec's field is ``None``). Returns ``None`` when no
    label information is available at all.
    """
    return compose_colorbar_label(
        spec.label,
        spec.unit_label if spec.unit_label is not None else unit_label,
    )


def resolve_value_limits(
    spec: ValueSpec | None,
    value_arrays: Iterable[np.ndarray],
    *,
    degenerate_pad: float = 0.5,
) -> tuple[float, float]:
    """Resolve the ``(vmin, vmax)`` colour scale for a value overlay.

    Honours explicit ``spec.vmin`` / ``spec.vmax`` and derives whichever
    bound is ``None`` from *value_arrays*. A degenerate range (all
    scalars equal, or no data at all) is widened by ``±degenerate_pad``
    so colormap normalisation stays well-defined.

    Every backend resolves its own colour scale from the primitives it
    happens to render, so *value_arrays* is supplied by the caller
    rather than derived here — this helper owns the bound-resolution
    and degenerate-range *policy*, not the choice of which scalars feed
    it.

    Parameters
    ----------
    spec : ValueSpec or None
        Value spec whose explicit bounds take precedence, if any.
    value_arrays : iterable of numpy.ndarray
        Scalar arrays to derive the missing bounds from. Empty arrays
        are skipped.
    degenerate_pad : float
        Half-width applied when the resolved range has zero extent.

    Returns
    -------
    vmin, vmax : float
        The resolved colour-scale bounds, always with ``vmin < vmax``.
    """
    explicit_min = None if spec is None else spec.vmin
    explicit_max = None if spec is None else spec.vmax

    low: float | None = None
    high: float | None = None
    if explicit_min is None or explicit_max is None:
        for array in value_arrays:
            array = np.asarray(array, dtype=float)
            if array.size == 0:
                continue
            array_min = float(array.min())
            array_max = float(array.max())
            low = array_min if low is None else min(low, array_min)
            high = array_max if high is None else max(high, array_max)

    vmin = float(explicit_min) if explicit_min is not None else (0.0 if low is None else low)
    vmax = float(explicit_max) if explicit_max is not None else (1.0 if high is None else high)
    if vmin == vmax:
        vmin -= degenerate_pad
        vmax += degenerate_pad
    return vmin, vmax


def resolve_overlay_values(
    overlay: OverlaySpec | None,
    morpho: Morphology,
) -> tuple[ValueSpec | None, dict[int, BranchValues] | None, str | None]:
    """Return ``(spec, per_branch_values, unit_label)`` or ``(None, None, None)``.

    Shared by the 2D and 3D scene builders — neither step depends on
    the dimensionality of the scene being built.
    """
    if overlay is None:
        return None, None, None
    spec = overlay.values_spec()
    if spec is None:
        return None, None, None
    per_branch, unit_label = resolve_values(morpho, spec)
    return spec, per_branch, unit_label


def with_unit_label(spec: ValueSpec | None, unit_label: str | None) -> ValueSpec | None:
    """Inject the unit label derived from the input array into the spec.

    A unit label already set on the spec always wins. Uses
    :func:`dataclasses.replace` so that a new :class:`ValueSpec` field
    cannot be silently dropped here.
    """
    if spec is None:
        return None
    if unit_label is None or spec.unit_label is not None:
        return spec
    return dataclasses.replace(spec, unit_label=unit_label)
