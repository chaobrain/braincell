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

"""Lazy region expressions for selecting continuous morphology subdomains."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral

import brainunit as u
import numpy as np
from brainunit import Quantity

from braincell.morph._spatial import MorphologySpatialGeometry
from braincell.morph.morphology import MorphoBranch, Morphology
from braincell.filter import helper
from braincell.filter.cache import SelectionCache, evaluate_cached

ClosedSide = str

__all__ = [
    "RegionMask",
    "RegionExpr",
    "AllRegion",
    "EmptyRegion",
    "BranchSlice",
    "BranchInFilter",
    "BranchRangeFilter",
    "RadiusRangeRegion",
    "TreeDistanceRegion",
    "EuclideanDistanceRegion",
    "SubtreeRegion",
    "RegionSetOp",
    "branch_in",
    "branch_range",
]


@dataclass(frozen=True)
class RegionMask:
    """Materialized region selection as ``(branch, prox, dist)`` intervals.

    Parameters
    ----------
    intervals : tuple of tuple of (int, float, float)
        One ``(branch_id, prox, dist)`` row per selected interval, with
        ``prox`` and ``dist`` normalized branch coordinates in ``[0, 1]``.

    Notes
    -----
    A mask is a value: it does not record the morphology it came from. Each
    ``branch_id`` is the default-order index of a branch in the morphology
    at the :attr:`~braincell.Morphology.revision` the mask was evaluated
    on. Attaching a branch, or replacing or transforming one in place,
    keeps every existing default index, so the integers still name the same
    branches (though evaluating the expression again may select other
    intervals). Removing a branch, or replacing a whole subtree, shifts the
    indices after it: evaluate the expression again rather than reuse the
    mask. Keep the expression, not the mask, wherever the tree may change.

    Examples
    --------
    .. code-block:: python

        >>> from braincell.filter import RegionMask
        >>> RegionMask(((0, 0.0, 1.0), (1, 0.0, 0.5))).intervals
        ((0, 0.0, 1.0), (1, 0.0, 0.5))
    """

    intervals: tuple[tuple[int, float, float], ...]


class RegionExpr(ABC):
    """Base class for lazy region selectors."""

    def __or__(self, other: "RegionExpr") -> "RegionExpr":
        if not isinstance(other, RegionExpr):
            return NotImplemented
        return RegionSetOp("union", _operands_of(self, "union") + _operands_of(other, "union"))

    def __and__(self, other: "RegionExpr") -> "RegionExpr":
        if not isinstance(other, RegionExpr):
            return NotImplemented
        return RegionSetOp("intersection", _operands_of(self, "intersection") + _operands_of(other, "intersection"))

    def __sub__(self, other: "RegionExpr") -> "RegionExpr":
        if not isinstance(other, RegionExpr):
            return NotImplemented
        # a - (b - c) keeps c, so only the left operand splices.
        return RegionSetOp("difference", _operands_of(self, "difference") + (other,))

    def complement(self) -> "RegionExpr":
        return RegionSetOp("complement", (self,))

    def evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        """Evaluate this expression against ``morpho``.

        Parameters
        ----------
        morpho : braincell.Morphology
            Morphology to evaluate against.
        cache : braincell.filter.SelectionCache or None
            Shared memoization scratch space, threaded to sub-expressions.

        Returns
        -------
        RegionMask
            The selected interval set.

        Raises
        ------
        TypeError
            If *morpho* is not a :class:`braincell.Morphology`.

        Notes
        -----
        This is the type-checked entry point every caller uses; subclasses
        implement :meth:`_evaluate` instead. Keeping the guard here rather
        than in each leaf is why it cannot be forgotten -- it previously was,
        by ``EmptyRegion``, which accepted arguments every sibling rejected.
        """
        if not isinstance(morpho, Morphology):
            raise TypeError(f"{type(self).__name__} expects Morphology, got {type(morpho).__name__!s}.")
        return self._evaluate(morpho, cache)

    @abstractmethod
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        """Compute the interval set for a morphology already type-checked by :meth:`evaluate`."""


@dataclass(frozen=True)
class AllRegion(RegionExpr):
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        return RegionMask(tuple((index, 0.0, 1.0) for index, _ in enumerate(morpho.branches)))


@dataclass(frozen=True)
class EmptyRegion(RegionExpr):
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        return RegionMask(())


@dataclass(frozen=True)
class BranchSlice(RegionExpr):
    branch_index: object
    prox: object
    dist: object

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        return RegionMask(
            helper.branch_slice_intervals(
                morpho,
                branch_index=self.branch_index,
                prox=self.prox,
                dist=self.dist,
            )
        )


@dataclass(frozen=True)
class BranchInFilter(RegionExpr):
    property: str
    values: object

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        return RegionMask(
            helper.branch_in_intervals(
                morpho,
                property_name=self.property,
                values=self.values,
            )
        )


@dataclass(frozen=True)
class BranchRangeFilter(RegionExpr):
    property: str
    bounds: object
    closed: ClosedSide = "neither"

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        return RegionMask(
            helper.branch_range_intervals(
                morpho,
                property_name=self.property,
                bounds=self.bounds,
                closed=self.closed,
            )
        )


@dataclass(frozen=True, eq=False)
class _LengthRangeRegion(RegionExpr):
    """Shared bounds of the regions that select where a length-valued ``f(x)`` lies in a range.

    On every branch the region is ``{x : minimum <= f(x) <= maximum}``; each
    subclass defines ``f``. ``minimum`` and ``maximum`` are scalar lengths
    (e.g. ``1 * u.um``) or ``None`` for no bound on that side; ``closed`` is
    ``"both"`` (default), ``"left"``, ``"right"`` or ``"neither"``, as in
    :func:`branch_range`. Where ``f`` varies, a bound is met at a single point
    and selects no length either way, so ``closed`` matters only where ``f``
    is constant over a stretch of cable: a cylinder whose radius equals a
    bound, or the soma at distance 0.

    The bounds are checked at construction: a bare number or a quantity that
    is not a length raises ``TypeError``; a non-scalar, negative, infinite or
    NaN bound, ``minimum > maximum`` or an unknown ``closed`` raises
    ``ValueError``. The fields keep the quantities as given, but equality and
    hashing use the bounds in micrometres and ``closed``, so
    ``RadiusRangeRegion(maximum=1.5 * u.um)`` equals
    ``RadiusRangeRegion(maximum=0.0015 * u.mmeter)`` and either can key a
    :class:`SelectionCache`.
    """

    minimum: Quantity | None = None
    maximum: Quantity | None = None
    closed: str = "both"

    def __post_init__(self) -> None:
        name = type(self).__name__
        lo = helper._length_scalar_um(self.minimum, name=f"{name}.minimum", allow_none=True)
        hi = helper._length_scalar_um(self.maximum, name=f"{name}.maximum", allow_none=True)
        closed = helper._parse_closed(self.closed)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{name}: minimum must be <= maximum, got {self.minimum!r} > {self.maximum!r}.")
        object.__setattr__(self, "_bounds_um", (lo, hi, closed))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._bounds_um == other._bounds_um

    def __hash__(self) -> int:
        return hash((type(self), self._bounds_um))

    def _accepts(self, value_um: float) -> bool:
        """Whether a constant ``f`` of ``value_um`` (a plateau) is selected."""
        return helper._value_in_range(value_um, *self._bounds_um)

    def _select_linear_piece(self, x0: float, x1: float, f0: float, f1: float) -> tuple[float, float] | None:
        """The selected part of ``[x0, x1]`` where ``f`` runs linearly from ``f0`` to ``f1`` (um)."""
        return helper._linear_piece_select(x0, x1, f0, f1, *self._bounds_um)


@dataclass(frozen=True, eq=False)
class RadiusRangeRegion(_LengthRangeRegion):
    """Select the cable whose local radius lies in a range.

    Parameters
    ----------
    minimum : brainunit.Quantity or None, optional
        Smallest radius selected, a scalar length; ``None`` (default) for no
        lower bound.
    maximum : brainunit.Quantity or None, optional
        Largest radius selected; ``None`` (default) for no upper bound.
    closed : {'both', 'left', 'right', 'neither'}, optional
        Which bounds are inclusive. The default ``"both"`` differs from
        :func:`branch_range`'s ``"neither"``: it matters only on cable whose
        radius is constant at a bound (a cylinder), and "radius 0.6 to 1 um"
        should include a 0.6 um cylinder.

    Returns
    -------
    RegionMask
        From :meth:`~RegionExpr.evaluate`: on every branch, the stretches
        where the radius is within the range, normalized. With no bounds the
        region is the whole tree, like :class:`AllRegion`.

    Raises
    ------
    TypeError
        At construction, if a bound is a bare number or not a length.
    ValueError
        At construction, if a bound is not a scalar, is negative, infinite
        (use ``None`` for no bound) or NaN, if ``minimum > maximum``, or if
        ``closed`` is not one of the four values.

    See Also
    --------
    branch_range : Select whole branches by a property such as
        ``mean_radius``.

    Notes
    -----
    The radius is linear in arc length along each segment, from the
    segment's proximal to its distal radius, so a tapered branch is cut
    where the radius crosses a bound. A zero-length segment, or a change of
    radius between one segment's distal end and the next one's proximal end,
    is a jump: it has no length and is not selected by itself; the segments
    on either side are judged separately, and the jump's ``x`` can end an
    interval. The radius is read from each branch's segment arrays, so the
    region needs no 3-D points, and moving or rotating the tree does not
    change it (scaling does). ``minimum == maximum`` selects only cable of
    exactly that radius.

    The radius is in the branch's own direction, whichever end faces its
    parent.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import RadiusRangeRegion
        >>> soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
        >>> dend = Branch.from_lengths(
        ...     lengths=np.array([40.0, 40.0]) * u.um, radii=np.array([2.0, 1.0, 1.0]) * u.um, type="basal_dendrite"
        ... )
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.dend = dend
        >>> tree.select(RadiusRangeRegion(maximum=1.5 * u.um)).intervals
        ((1, 0.25, 1.0),)
        >>> tree.select(RadiusRangeRegion(maximum=1.0 * u.um, closed="neither")).intervals
        ()
    """

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        intervals: list[tuple[int, float, float]] = []
        for index, view in enumerate(morpho.branches):
            branch = view.branch
            lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
            proximal = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
            distal = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
            cumulative = np.cumsum(lengths)
            starts = np.concatenate(([0.0], cumulative[:-1])) / cumulative[-1]
            ends = cumulative / cumulative[-1]
            ends[-1] = 1.0
            for i in np.flatnonzero(lengths > 0.0):
                # A zero-length segment is a radius jump: no length to select.
                piece = self._select_linear_piece(
                    float(starts[i]), float(ends[i]), float(proximal[i]), float(distal[i])
                )
                if piece is not None:
                    intervals.append((index, *piece))
        return RegionMask(helper.normalize_region_intervals(intervals))


@dataclass(frozen=True, eq=False)
class TreeDistanceRegion(_LengthRangeRegion):
    """Select the cable whose path distance from the soma lies in a range.

    Parameters
    ----------
    minimum : brainunit.Quantity or None, optional
        Smallest distance selected, a scalar length; ``None`` (default) for
        no lower bound.
    maximum : brainunit.Quantity or None, optional
        Largest distance selected; ``None`` (default) for no upper bound.
    closed : {'both', 'left', 'right', 'neither'}, optional
        Which bounds are inclusive. It matters only on the soma, where the
        distance is 0 over the whole branch: the default ``"both"`` keeps the
        soma in ``TreeDistanceRegion(maximum=100 * u.um)`` and in
        ``TreeDistanceRegion(0 * u.um, 100 * u.um)``.

    Returns
    -------
    RegionMask
        From :meth:`~RegionExpr.evaluate`: on every branch, the stretches
        whose distance is within the range, normalized.

    Raises
    ------
    TypeError
        At construction, if a bound is a bare number or not a length.
    ValueError
        At construction, if a bound is not a scalar, is negative, infinite
        (use ``None`` for no bound) or NaN, if ``minimum > maximum``, or if
        ``closed`` is not one of the four values.

    See Also
    --------
    EuclideanDistanceRegion : The straight-line distance instead.
    RadiusRangeRegion : The same bounds on the local radius.

    Notes
    -----
    The distance is the length of the shortest path along the tree to the
    union of all soma branches, and is 0 on the soma; a tree without a soma
    branch measures from its whole root branch instead. It is the same
    distance a callable cable or density parameter reads as
    ``path_distance_from_soma`` (:class:`braincell.mech.CVContext`,
    :func:`braincell.filter.metric.path_distance_from_soma`), so a region and
    a parameter that use the same bound agree.

    Along a branch the distance changes at a rate of 1 um per um, so the
    region is cut exactly where it crosses a bound. With one soma it grows
    away from the soma; with two or more it can rise and then fall along a
    branch that lies between them, and a single range can then select two
    stretches of one branch. Only lengths and topology are read, so the
    region needs no 3-D points, and moving or rotating the tree does not
    change it (scaling does).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import TreeDistanceRegion, branch_in
        >>> soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
        >>> dend = Branch.from_lengths(lengths=np.array([200.0]) * u.um, radii=np.array([1.0, 1.0]) * u.um, type="basal_dendrite")
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.dend = dend
        >>> tree.select(TreeDistanceRegion(maximum=50 * u.um)).intervals
        ((0, 0.0, 1.0), (1, 0.0, 0.25))
        >>> tree.select(branch_in("type", "basal_dendrite") & TreeDistanceRegion(maximum=100 * u.um)).intervals
        ((1, 0.0, 0.5),)
    """

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        geometry = MorphologySpatialGeometry.build(morpho)
        intervals: list[tuple[int, float, float]] = []
        for index in range(len(morpho.branches)):
            if index in geometry.reference_branch_ids:
                # The distance is 0 over the whole branch: a plateau.
                if self._accepts(0.0):
                    intervals.append((index, 0.0, 1.0))
                continue
            nodes = geometry.branch_nodes_x[index]
            bases = geometry.soma_bases_um[index]
            length = float(geometry.branch_lengths_um[index])
            for j in range(len(nodes) - 1):
                # Between two adjacent nodes the distance is the nearer of the
                # paths through either one: up from b0, then down to b1.
                n0, n1 = float(nodes[j]), float(nodes[j + 1])
                b0, b1 = float(bases[j]), float(bases[j + 1])
                peak = min(n1, max(n0, (b1 - b0 + (n1 + n0) * length) / (2.0 * length)))
                at_peak = b0 + (peak - n0) * length
                for x0, x1, f0, f1 in ((n0, peak, b0, at_peak), (peak, n1, at_peak, b1)):
                    if x1 > x0:
                        piece = self._select_linear_piece(x0, x1, f0, f1)
                        if piece is not None:
                            intervals.append((index, *piece))
        return RegionMask(helper.normalize_region_intervals(intervals))


@dataclass(frozen=True, eq=False)
class EuclideanDistanceRegion(_LengthRangeRegion):
    """Select the cable whose straight-line distance from the soma lies in a range.

    Parameters
    ----------
    minimum : brainunit.Quantity or None, optional
        Smallest distance selected, a scalar length; ``None`` (default) for
        no lower bound.
    maximum : brainunit.Quantity or None, optional
        Largest distance selected; ``None`` (default) for no upper bound.
    closed : {'both', 'left', 'right', 'neither'}, optional
        Which bounds are inclusive. It matters only where the distance is
        constant over a stretch of cable, as on the soma, where it is 0: the
        default ``"both"`` keeps the soma in
        ``EuclideanDistanceRegion(maximum=50 * u.um)``.

    Returns
    -------
    RegionMask
        From :meth:`~RegionExpr.evaluate`: on every branch, the stretches
        whose distance is within the range, normalized. A branch that bends
        back towards the soma can contribute more than one interval.

    Raises
    ------
    TypeError
        At construction, if a bound is a bare number or not a length.
    ValueError
        At construction, if a bound is not a scalar, is negative, infinite
        (use ``None`` for no bound) or NaN, if ``minimum > maximum``, or if
        ``closed`` is not one of the four values. At evaluation (``select``,
        ``Cell.paint``), if any branch of the tree has no 3-D points; the
        message names up to five of them.

    See Also
    --------
    TreeDistanceRegion : The distance along the tree instead.
    RadiusRangeRegion : The same bounds on the local radius.

    Notes
    -----
    The distance from a point is the shortest 3-D distance to the centre
    lines of all soma branches (every segment of every ``"soma"`` branch),
    and is 0 on the soma; a tree without a soma branch measures from its
    root branch instead. This reference differs from
    :meth:`braincell.Morphology.max_euclidean_distance`, which measures from
    one point on the root branch. The straight line is never longer than
    the path along the tree, so this region always contains the
    :class:`TreeDistanceRegion` with the same maximum.

    Each segment is a straight line, and its distance to each soma segment
    is solved exactly (three quadratic pieces, by where the nearest point
    lies), for all segments at once with NumPy; a morphology of about 700
    branches and 40 000 segments takes well under a second. Moving or
    rotating the whole tree does not change the region; scaling does.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import EuclideanDistanceRegion
        >>> soma = Branch.from_points(
        ...     points=np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma"
        ... )
        >>> loop = Branch.from_points(  # out along x, across, and back towards the soma
        ...     points=np.array([[20.0, 0.0, 0.0], [88.0, 0.0, 0.0], [88.0, 24.0, 0.0], [20.0, 24.0, 0.0]]) * u.um,
        ...     radii=np.array([1.0, 1.0, 1.0, 1.0]) * u.um,
        ...     type="basal_dendrite",
        ... )
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.loop = loop
        >>> tree.select(EuclideanDistanceRegion(maximum=40 * u.um)).intervals
        ((0, 0.0, 1.0), (1, 0.0, 0.25), (1, 0.8, 1.0))
    """

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        branches = morpho.branches
        if not morpho.has_full_point_geometry:
            missing = [view.name for view in branches if view.branch.points_proximal is None]
            shown = ", ".join(repr(name) for name in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise ValueError(
                f"EuclideanDistanceRegion needs 3-D points on every branch; "
                f"{len(missing)} branch(es) have none: {shown}{more}."
            )
        lo, hi, closed = self._bounds_um
        reference_ids = sorted({view.index for view in branches if str(view.type) == "soma"} or {morpho.root.index})
        references = []
        for index in reference_ids:
            branch = branches[index].branch
            references.extend(
                zip(
                    np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float),
                    np.asarray(branch.points_distal.to_decimal(u.um), dtype=float),
                )
            )
        intervals: list[tuple[int, float, float]] = []
        if self._accepts(0.0):
            # The distance is 0 over every reference branch: a plateau.
            intervals.extend((index, 0.0, 1.0) for index in reference_ids)

        # Every positive-length segment of every other branch, as one batch.
        segment_branch, segment_x0, segment_x1, starts, steps = [], [], [], [], []
        reference_set = set(reference_ids)
        for index, view in enumerate(branches):
            if index in reference_set:
                continue
            branch = view.branch
            lengths = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
            cumulative = np.cumsum(lengths)
            x0 = np.concatenate(([0.0], cumulative[:-1])) / cumulative[-1]
            x1 = cumulative / cumulative[-1]
            x1[-1] = 1.0
            keep = lengths > 0.0
            proximal = np.asarray(branch.points_proximal.to_decimal(u.um), dtype=float)[keep]
            segment_branch.append(np.full(int(keep.sum()), index))
            segment_x0.append(x0[keep])
            segment_x1.append(x1[keep])
            starts.append(proximal)
            steps.append(np.asarray(branch.points_distal.to_decimal(u.um), dtype=float)[keep] - proximal)
        if not starts:
            return RegionMask(helper.normalize_region_intervals(intervals))
        segment_branch = np.concatenate(segment_branch)
        segment_x0 = np.concatenate(segment_x0)
        segment_x1 = np.concatenate(segment_x1)
        starts = np.concatenate(starts)
        steps = np.concatenate(steps)

        def within(radius: float | None, strict: bool) -> np.ndarray | None:
            # (2, n_references, n_segments): each reference segment's t-interval on each segment.
            if radius is None:
                return None
            return np.stack(
                [
                    np.stack(helper._segment_distance_sublevel(starts, steps, a, b, radius, strict=strict))
                    for a, b in references
                ],
                axis=1,
            )

        # Keep {d <= maximum} (< if the upper end is open); drop {d < minimum} (<= if the lower end is open).
        keep = within(hi, strict=not helper._right_closed(closed))
        drop = within(lo, strict=helper._left_closed(closed))
        eps = helper.EPSILON
        n = len(starts)

        def covers(spans: np.ndarray) -> np.ndarray:
            return np.any((spans[0] <= eps) & (spans[1] >= 1.0 - eps), axis=0)

        def misses(spans: np.ndarray) -> np.ndarray:
            return ~np.any(spans[1] - spans[0] > eps, axis=0)

        whole = (np.ones(n, dtype=bool) if keep is None else covers(keep)) & (
            np.ones(n, dtype=bool) if drop is None else misses(drop)
        )
        empty = (np.zeros(n, dtype=bool) if keep is None else misses(keep)) | (
            np.zeros(n, dtype=bool) if drop is None else covers(drop)
        )
        for i in np.flatnonzero(whole & ~empty):
            intervals.append((int(segment_branch[i]), float(segment_x0[i]), float(segment_x1[i])))
        # Only segments with an interval end strictly inside them need the interval arithmetic.
        for i in np.flatnonzero(~whole & ~empty):
            kept = (
                [(0, 0.0, 1.0)]
                if keep is None
                else [(0, float(t0), float(t1)) for t0, t1 in zip(keep[0, :, i], keep[1, :, i]) if t1 > t0]
            )
            if drop is None:
                pieces = helper.normalize_region_intervals(kept)
            else:
                dropped = [(0, float(t0), float(t1)) for t0, t1 in zip(drop[0, :, i], drop[1, :, i]) if t1 > t0]
                pieces = helper.difference_region_intervals(kept, dropped)
            x0, x1 = float(segment_x0[i]), float(segment_x1[i])
            for _, t0, t1 in pieces:
                intervals.append((int(segment_branch[i]), x0 + t0 * (x1 - x0), x0 + t1 * (x1 - x0)))
        return RegionMask(helper.normalize_region_intervals(intervals))


@dataclass(frozen=True)
class SubtreeRegion(RegionExpr):
    """Select a branch and all of its descendants, each branch whole.

    Parameters
    ----------
    root_branch_index : str or int
        The root of the subtree: a branch name (recommended), or a
        default-order branch index (a Python or NumPy integer).

    Returns
    -------
    RegionMask
        From :meth:`~RegionExpr.evaluate`: one ``(i, 0.0, 1.0)`` interval for
        the root branch and for each of its descendants, in ascending branch
        index. A leaf selects only itself; the morphology's root selects the
        whole tree, like :class:`AllRegion`.

    Raises
    ------
    TypeError
        At construction, if ``root_branch_index`` is neither a ``str`` nor an
        integer (``bool``, ``float`` and ``None`` included), or is a
        :class:`~braincell.MorphoBranch` view; pass the view's name instead.
    KeyError
        At evaluation (``select``, ``Cell.paint``), if no branch has the
        given name.
    IndexError
        At evaluation, if the index is negative or out of range. Negative
        indices do not wrap around.

    See Also
    --------
    braincell.Morphology.subtree : Copy a branch and its descendants into a
        new tree.

    Notes
    -----
    This is a selection on the tree it is evaluated on, not a copy: the
    intervals index that tree. :meth:`braincell.Morphology.subtree` instead
    returns a new morphology whose branch indices start from 0, so its
    indices cannot be used to paint the original tree.

    "Descendants" are topological: every branch whose parent chain reaches
    the root. Which end of a child faces its parent (``child_x``) does not
    matter, because each descendant is selected whole. The region reads the
    topology only, so moving, rotating or scaling the tree does not change
    it. To start partway along the root branch, subtract a slice:
    ``SubtreeRegion("dend") - BranchSlice(1, 0.0, 0.5)``.

    In a :class:`~braincell.Cell` declaration, refer to the root by name. An
    index names whatever branch holds that index when the expression is
    evaluated: evaluated on a morphology after ``Morphology.remove`` or
    ``Morphology.reattach``, it selects another branch. A Cell declaration
    holding an index raises ``ValueError`` at its next rebuild after such an
    edit, and one holding a name raises ``KeyError`` once its branch is gone.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import BranchSlice, SubtreeRegion
        >>> soma = Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um, type="soma")
        >>> dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[2.0, 1.0] * u.um, type="basal_dendrite")
        >>> axon = Branch.from_lengths(lengths=[200.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.dend = dend
        >>> tree.dend.tip = dend
        >>> tree.soma.axon = axon
        >>> [branch.name for branch in tree.branches]
        ['soma', 'dend', 'tip', 'axon']
        >>> tree.select(SubtreeRegion("dend")).intervals
        ((1, 0.0, 1.0), (2, 0.0, 1.0))
        >>> tree.select(SubtreeRegion("dend") - BranchSlice(1, 0.0, 0.5)).intervals
        ((1, 0.5, 1.0), (2, 0.0, 1.0))
    """

    root_branch_index: int | str

    def __post_init__(self) -> None:
        root = self.root_branch_index
        if isinstance(root, MorphoBranch):
            raise TypeError(
                "SubtreeRegion takes a branch name or index, not a MorphoBranch view; "
                f"pass its name, e.g. SubtreeRegion({root.name!r})."
            )
        if isinstance(root, bool) or not isinstance(root, (Integral, str)):
            raise TypeError(f"SubtreeRegion root must be a branch name (str) or index (int), got {root!r}.")

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        root = self.root_branch_index
        if isinstance(root, str):
            view = morpho.branch(name=root)
        else:
            view = morpho.branches[helper._coerce_branch_index(root, len(morpho.branches))]
        # Iterative: a deep reconstruction would overflow Python's recursion limit.
        found: list[int] = []
        stack = [view]
        while stack:
            current = stack.pop()
            found.append(current.index)
            stack.extend(current.children)
        return RegionMask(helper.normalize_region_intervals(tuple((index, 0.0, 1.0) for index in found)))


@dataclass(frozen=True)
class RegionSetOp(RegionExpr):
    """Composite selector produced by set algebra on region expressions.

    Notes
    -----
    The operands are combined from left to right. The ``|``, ``&`` and ``-``
    operators extend a node of the same operation instead of nesting it, so
    ``a | b | c`` is one union of three operands however many are chained;
    only the right operand of ``-`` stays nested, since ``a - (b - c)``
    differs from ``(a - b) - c``.
    """

    op: str
    operands: tuple[RegionExpr, ...]

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        op = self.op
        operands = self.operands

        if op == "complement":
            if len(operands) != 1:
                raise ValueError("complement expects exactly one operand.")
            source = evaluate_cached(operands[0], morpho, cache).intervals
            intervals = helper.complement_region_intervals(
                source,
                n_branches=len(morpho.branches),
            )
            return RegionMask(intervals)

        if op not in {"union", "intersection", "difference"}:
            raise ValueError(f"Unsupported region set operation {op!r}.")
        if len(operands) < 2:
            raise ValueError(f"{op} expects at least two operands.")

        # No normalize_region_intervals() here: all three fold helpers below
        # normalize their own inputs, so an outer pass re-sorts and re-merges
        # data that is about to be sorted and merged again.
        current = evaluate_cached(operands[0], morpho, cache).intervals
        for operand in operands[1:]:
            other = evaluate_cached(operand, morpho, cache).intervals
            if op == "union":
                current = helper.union_region_intervals(current, other)
            elif op == "intersection":
                current = helper.intersect_region_intervals(current, other)
            else:
                current = helper.difference_region_intervals(current, other)
        return RegionMask(current)


def _operands_of(expr: RegionExpr, op: str) -> tuple[RegionExpr, ...]:
    """Return ``expr``'s operands if it is a well-formed ``op`` node, else ``(expr,)``.

    The set operators splice a same-operation operand rather than nest it.
    :meth:`RegionSetOp._evaluate` folds its operands left to right, so
    ``(a - b) - c`` selects what ``RegionSetOp("difference", (a, b, c))``
    does, and union and intersection are associative. A region built one
    operand at a time in a loop is therefore one level deep: hashing,
    comparing, printing and evaluating recurse once per level, and a few
    hundred levels exceeded Python's recursion limit.
    """
    if isinstance(expr, RegionSetOp) and expr.op == op and len(expr.operands) >= 2:
        return tuple(expr.operands)
    return (expr,)


def branch_in(property: str, values: object) -> BranchInFilter:
    """Select whole branches whose ``property`` takes one of ``values``.

    Parameters
    ----------
    property : str
        Branch property to compare, read from each :class:`~braincell.morph.MorphoBranch`:
        e.g. ``"type"``, ``"name"`` or ``"parent_id"`` (the parent's branch
        index, ``-1`` at the root), or a metric such as ``"length"``. An
        unknown property raises ``ValueError`` at evaluation.
    values : object or iterable of object
        One value, or several; a string counts as one value. Quantities
        compare in a common unit.

    Returns
    -------
    BranchInFilter
        A region expression that selects each matching branch from ``x = 0``
        to ``x = 1``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
        >>> tree = Morphology.from_root(
        ...     Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"), name="soma"
        ... )
        >>> tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
        >>> tree.soma.axon = Branch.from_lengths(lengths=[300.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
        >>> from braincell.filter import branch_in
        >>> tree.select(branch_in("type", ("dendrite", "axon"))).intervals
        ((1, 0.0, 1.0), (2, 0.0, 1.0))
        >>> tree.select(branch_in("name", "dend")).intervals
        ((1, 0.0, 1.0),)
    """
    return BranchInFilter(property=property, values=values)


def branch_range(
    property: str,
    bounds: object,
    *,
    closed: ClosedSide = "neither",
) -> BranchRangeFilter:
    """Select whole branches whose numeric ``property`` lies within ``bounds``.

    Parameters
    ----------
    property : str
        Branch property to compare, as in :func:`branch_in`; usually a metric
        such as ``"length"``, ``"mean_radius"``, ``"area"`` or ``"volume"``.
    bounds : tuple or Quantity
        ``(low, high)``, either of which may be ``None`` for no bound on that
        side, or a quantity vector of two values. Anything else raises
        ``TypeError`` at evaluation.
    closed : {"neither", "left", "right", "both"}, default "neither"
        Which ends of the range are included. Another value raises
        ``ValueError`` at evaluation.

    Returns
    -------
    BranchRangeFilter
        A region expression that selects each matching branch from ``x = 0``
        to ``x = 1``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from braincell import Branch, Morphology
        >>> tree = Morphology.from_root(
        ...     Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma"), name="soma"
        ... )
        >>> tree.soma.dend = Branch.from_lengths(lengths=[100.0] * u.um, radii=[1.0, 1.0] * u.um, type="dendrite")
        >>> tree.soma.axon = Branch.from_lengths(lengths=[300.0] * u.um, radii=[0.5, 0.5] * u.um, type="axon")
        >>> from braincell.filter import branch_range
        >>> tree.select(branch_range("length", (50 * u.um, None))).intervals
        ((1, 0.0, 1.0), (2, 0.0, 1.0))
        >>> tree.select(branch_range("length", (100 * u.um, 300 * u.um))).intervals
        ()
        >>> tree.select(branch_range("length", (100 * u.um, 300 * u.um), closed="both")).intervals
        ((1, 0.0, 1.0), (2, 0.0, 1.0))
    """
    return BranchRangeFilter(property=property, bounds=bounds, closed=closed)
