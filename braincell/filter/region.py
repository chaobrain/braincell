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
        return RegionSetOp("union", (self, other))

    def __and__(self, other: "RegionExpr") -> "RegionExpr":
        if not isinstance(other, RegionExpr):
            return NotImplemented
        return RegionSetOp("intersection", (self, other))

    def __sub__(self, other: "RegionExpr") -> "RegionExpr":
        if not isinstance(other, RegionExpr):
            return NotImplemented
        return RegionSetOp("difference", (self, other))

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
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class EuclideanDistanceRegion(_LengthRangeRegion):
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        raise NotImplementedError


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
    evaluated, so after a ``remove`` it can silently select another branch;
    a name raises ``KeyError`` once its branch is gone.

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
    """Composite selector produced by set algebra on region expressions."""

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


def branch_in(property: str, values: object) -> BranchInFilter:
    return BranchInFilter(property=property, values=values)


def branch_range(
    property: str,
    bounds: object,
    *,
    closed: ClosedSide = "neither",
) -> BranchRangeFilter:
    return BranchRangeFilter(property=property, bounds=bounds, closed=closed)
