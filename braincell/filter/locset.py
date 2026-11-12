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


from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real

import numpy as np

from brainunit import Quantity

from braincell._random import draw_seed
from braincell.morph.morphology import Morphology
from braincell.filter import helper
from braincell.filter._sampling import sample_locations_from_region
from braincell.filter.cache import SelectionCache, evaluate_cached
from braincell.filter.region import RegionExpr

Location = tuple[int, float]

__all__ = [
    "LocsetBatch",
    "LocsetMask",
    "LocsetExpr",
    "AtLocation",
    "at",
    "RootLocation",
    "ForkPoints",
    "BranchPoints",
    "Terminals",
    "RegionAnchors",
    "UniformSamples",
    "RandomSamples",
    "SampleLocations",
    "sample",
    "StepSamples",
    "LocsetConcatOp",
    "LocsetSetOp",
    "LocsetUniqueOp",
]


def _normalize_location_columns(
    branch_id: object,
    branch_x: object,
    *,
    ndim: int,
    type_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    branch_values = np.asarray(branch_id)
    x_values = np.asarray(branch_x)
    if branch_values.ndim != ndim or x_values.ndim != ndim:
        dimension = "one-dimensional" if ndim == 1 else "two-dimensional"
        raise ValueError(f"{type_name} columns must be {dimension}.")
    if branch_values.shape != x_values.shape:
        raise ValueError(
            f"{type_name}.branch_id and branch_x must have the same shape, "
            f"got {branch_values.shape!r} and {x_values.shape!r}."
        )
    if branch_values.dtype.kind not in "iu":
        raise TypeError(f"{type_name}.branch_id must contain integers.")
    if x_values.dtype.kind not in "iuf":
        raise TypeError(f"{type_name}.branch_x must contain real numbers.")
    if branch_values.dtype.kind == "u" and np.any(branch_values > np.iinfo(np.int64).max):
        raise ValueError(f"{type_name}.branch_id values must fit in int64.")
    if np.any(branch_values < 0):
        raise ValueError(f"{type_name}.branch_id values must be non-negative.")

    normalized_x = np.asarray(x_values, dtype=np.float64)
    if np.any(~np.isfinite(normalized_x)):
        raise ValueError(f"{type_name}.branch_x values must be finite.")
    if np.any(normalized_x < -helper.EPSILON) or np.any(normalized_x > 1.0 + helper.EPSILON):
        raise ValueError(f"{type_name}.branch_x values must be within [0, 1].")
    return (
        np.asarray(branch_values, dtype=np.int64),
        np.round(np.clip(normalized_x, 0.0, 1.0), decimals=helper._round_digits_from_epsilon(helper.EPSILON)),
    )


class LocsetBatch:
    """A read-only batch of aligned resolved locsets.

    The leading dimension identifies population members and the trailing
    dimension contains the locations assigned to each member.

    Instances are built through :meth:`from_columns`; the class itself takes
    no constructor arguments.

    Like :class:`LocsetMask`, a batch holds default-order branch indices
    valid for the morphology revision it was resolved on; evaluate again
    after a branch is removed or moved.

    See Also
    --------
    from_columns : The constructor, which documents the columns.
    LocsetMask : The single-population equivalent.
    """

    __slots__ = ("_branch_id", "_branch_x", "_display_names")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__!s} is immutable.")

    @classmethod
    def from_columns(
        cls,
        branch_id: object,
        branch_x: object,
        display_names: tuple[tuple[str, ...], ...] | list[list[str]] | None = None,
    ) -> "LocsetBatch":
        """Construct a batch from two aligned location matrices.

        Parameters
        ----------
        branch_id : array-like
            Two-dimensional integer branch identifiers.
        branch_x : array-like
            Two-dimensional normalized branch coordinates in ``[0, 1]``.
        display_names : iterable of iterable of str, optional
            Human-readable names aligned with the location matrix.

        Returns
        -------
        LocsetBatch
            Immutable batch over the normalized columns.

        Raises
        ------
        ValueError
            If *display_names* does not match the column shape, or a
            coordinate falls outside ``[0, 1]``.
        """
        branch_values, x_values = _normalize_location_columns(
            branch_id,
            branch_x,
            ndim=2,
            type_name=cls.__name__,
        )
        names = None
        if display_names is not None:
            names = tuple(tuple(row) for row in display_names)
            if len(names) != branch_values.shape[0] or any(len(row) != branch_values.shape[1] for row in names):
                raise ValueError("LocsetBatch.display_names must have the same shape as its columns.")
            if any(not isinstance(name, str) for row in names for name in row):
                raise TypeError("LocsetBatch.display_names must contain strings.")

        obj = cls.__new__(cls)
        branch_column = np.array(branch_values, dtype=np.int64, copy=True)
        x_column = np.array(x_values, dtype=np.float64, copy=True)
        branch_column.flags.writeable = False
        x_column.flags.writeable = False
        object.__setattr__(obj, "_branch_id", branch_column)
        object.__setattr__(obj, "_branch_x", x_column)
        object.__setattr__(obj, "_display_names", names)
        return obj

    @property
    def branch_id(self) -> np.ndarray:
        """Read-only branch identifier matrix."""
        return self._branch_id

    @property
    def branch_x(self) -> np.ndarray:
        """Read-only normalized branch-coordinate matrix."""
        return self._branch_x

    @property
    def display_names(self) -> tuple[tuple[str, ...], ...] | None:
        """Optional human-readable names aligned with the location matrix."""
        return self._display_names

    @property
    def shape(self) -> tuple[int, int]:
        """Batch and per-member location dimensions."""
        return self._branch_id.shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: object) -> "LocsetMask | LocsetBatch":
        branch_values = np.asarray(self._branch_id[index])
        x_values = np.asarray(self._branch_x[index])
        names = None
        if self._display_names is not None:
            names_array = np.asarray(self._display_names, dtype=object)[index]
            names = names_array.tolist()
        if branch_values.ndim == 1:
            return LocsetMask.from_columns(branch_values, x_values, names)
        if branch_values.ndim == 2:
            return LocsetBatch.from_columns(branch_values, x_values, names)
        raise IndexError("LocsetBatch indexing must preserve its location dimension.")

    def __repr__(self) -> str:
        return (
            f"LocsetBatch(branch_id={self._branch_id.tolist()!r}, "
            f"branch_x={self._branch_x.tolist()!r}, display_names={self._display_names!r})"
        )


class LocsetMask:
    """Resolved ordered locations in read-only columnar storage.

    Parameters
    ----------
    points : iterable of tuple of int and float, optional
        Location rows in ``(branch_id, branch_x)`` form. Row order and
        duplicate rows are preserved.
    display_names : iterable of str, optional
        Human-readable labels aligned with ``points``. Names may be omitted
        when a mask is constructed independently of a morphology.

    Notes
    -----
    A mask is a value: it does not record the morphology it came from. Each
    ``branch_id`` is the default-order index of a branch in the morphology
    at the :attr:`~braincell.Morphology.revision` the mask was evaluated
    on. Attaching a branch, or replacing or transforming one in place,
    keeps every existing default index, so the integers still name the same
    branches (though evaluating the expression again may select other
    points). Removing a branch, or replacing a whole subtree, shifts the
    indices after it: evaluate the expression again rather than reuse the
    mask. Keep the expression (``at("axon", 0.5)``), not the mask, wherever
    the tree may change.

    Examples
    --------
    .. code-block:: python

        >>> from braincell.filter import LocsetMask
        >>> LocsetMask(points=((0, 0.5), (1, 1.0))).points
        ((0, 0.5), (1, 1.0))
    """

    __slots__ = ("_branch_id", "_branch_x", "_display_names", "_points_cache")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__!s} is immutable.")

    def __init__(
        self,
        points: tuple[Location, ...] | list[Location] = (),
        display_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        normalized = helper.normalize_locset_points(points)
        branch_id = np.asarray([point[0] for point in normalized], dtype=np.int64)
        branch_x = np.asarray([point[1] for point in normalized], dtype=np.float64)
        self._initialize(branch_id, branch_x, display_names, points_cache=normalized)

    @classmethod
    def from_columns(
        cls,
        branch_id: object,
        branch_x: object,
        display_names: tuple[str, ...] | list[str] | None = None,
    ) -> "LocsetMask":
        """Construct a mask directly from aligned location columns.

        Parameters
        ----------
        branch_id : array-like
            One-dimensional integer branch identifiers.
        branch_x : array-like
            One-dimensional normalized branch coordinates in ``[0, 1]``.
        display_names : iterable of str, optional
            Human-readable names aligned with the two location columns.

        Returns
        -------
        LocsetMask
            A mask whose columns are immutable copies of the input data.
        """
        branch_values, normalized_x = _normalize_location_columns(
            branch_id,
            branch_x,
            ndim=1,
            type_name=cls.__name__,
        )

        obj = cls.__new__(cls)
        obj._initialize(
            np.asarray(branch_values, dtype=np.int64),
            normalized_x,
            display_names,
            points_cache=None,
        )
        return obj

    def _initialize(
        self,
        branch_id: np.ndarray,
        branch_x: np.ndarray,
        display_names: tuple[str, ...] | list[str] | None,
        *,
        points_cache: tuple[Location, ...] | None,
    ) -> None:
        branch_column = np.array(branch_id, dtype=np.int64, copy=True)
        x_column = np.array(branch_x, dtype=np.float64, copy=True)
        branch_column.flags.writeable = False
        x_column.flags.writeable = False

        names = None if display_names is None else tuple(display_names)
        if names is not None and len(branch_column) != len(names):
            raise ValueError(
                "LocsetMask.points and display_names must have the same length, "
                f"got {len(branch_column)!r} and {len(names)!r}."
            )
        if names is not None and any(not isinstance(name, str) for name in names):
            raise TypeError("LocsetMask.display_names must contain strings.")

        object.__setattr__(self, "_branch_id", branch_column)
        object.__setattr__(self, "_branch_x", x_column)
        object.__setattr__(self, "_display_names", names)
        object.__setattr__(self, "_points_cache", points_cache)

    @property
    def branch_id(self) -> np.ndarray:
        """Read-only branch identifier column."""
        return self._branch_id

    @property
    def branch_x(self) -> np.ndarray:
        """Read-only normalized branch-coordinate column."""
        return self._branch_x

    @property
    def display_names(self) -> tuple[str, ...] | None:
        """Optional human-readable names aligned with location rows."""
        return self._display_names

    @property
    def points(self) -> tuple[Location, ...]:
        """Compatibility row view in ``(branch_id, branch_x)`` form."""
        cached = self._points_cache
        if cached is None:
            cached = tuple(zip(self._branch_id.tolist(), self._branch_x.tolist()))
            object.__setattr__(self, "_points_cache", cached)
        return cached

    def resolved_display_names(self, morpho) -> tuple[str, ...]:
        """Return display names, deriving ``branch(x)`` labels when unset.

        Locations carry optional author-supplied names; when they are absent
        every consumer needs the same ``branch_name(x)`` fallback, so the
        label format is defined here rather than at each call site.

        Parameters
        ----------
        morpho : braincell.Morphology
            Morphology used to resolve branch names.

        Returns
        -------
        tuple of str
            One label per location row.
        """
        if self._display_names is not None:
            return self._display_names
        return _display_names_for_points(morpho, self.points)

    def __len__(self) -> int:
        return len(self._branch_id)

    def __getitem__(self, index: object) -> "LocsetMask | LocsetBatch":
        """Select locations, preserving duplicates and index order.

        One-dimensional selection returns a :class:`LocsetMask`. A
        two-dimensional integer index returns a :class:`LocsetBatch`, whose
        rows can be aligned with population members during placement.
        """
        branch_values = np.asarray(self._branch_id[index])
        x_values = np.asarray(self._branch_x[index])
        names = None
        if self._display_names is not None:
            selected_names = np.asarray(self._display_names, dtype=object)[index]
            names = selected_names.tolist() if isinstance(selected_names, np.ndarray) else selected_names
        if branch_values.ndim == 0:
            branch_values = branch_values.reshape(1)
            x_values = x_values.reshape(1)
            if names is not None:
                names = [names]
        if branch_values.ndim == 1:
            return LocsetMask.from_columns(branch_values, x_values, names)
        if branch_values.ndim == 2:
            return LocsetBatch.from_columns(branch_values, x_values, names)
        raise IndexError("LocsetMask indexing supports at most two dimensions.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocsetMask):
            return NotImplemented
        return self.points == other.points and self.display_names == other.display_names

    def __hash__(self) -> int:
        return hash((self.points, self.display_names))

    def __repr__(self) -> str:
        return f"LocsetMask(points={self.points!r}, display_names={self.display_names!r})"

    def __reduce__(self) -> tuple[object, tuple[tuple[Location, ...], tuple[str, ...] | None]]:
        return type(self), (self.points, self.display_names)

    def unique(self) -> "LocsetMask":
        """Return stable first-occurrence unique location rows."""
        indices: list[int] = []
        seen: set[Location] = set()
        for index, point in enumerate(self.points):
            if point in seen:
                continue
            seen.add(point)
            indices.append(index)
        names = None
        if self._display_names is not None:
            names = tuple(self._display_names[index] for index in indices)
        return LocsetMask.from_columns(self._branch_id[indices], self._branch_x[indices], names)


class LocsetExpr(ABC):
    """Base class of locset expressions: lazy, ordered selections of points on a morphology.

    An expression names points without a morphology at hand. :meth:`evaluate`
    (or ``Morphology.select``) resolves it against one and returns a
    :class:`LocsetMask` of ``(branch_id, branch_x)`` rows.

    Expressions combine with ``+``, which concatenates and keeps duplicates and
    order, and with the set operations ``|`` (union), ``&`` (intersection) and
    ``-`` (difference). :meth:`unique` drops duplicates after evaluation.
    Subclasses implement ``_evaluate``.

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
        >>> from braincell.filter import Terminals, at
        >>> (at("dend", 1.0) + Terminals()).evaluate(tree).branch_id
        array([1, 1, 2])
        >>> (at("dend", 1.0) + Terminals()).unique().evaluate(tree).branch_id
        array([1, 2])
    """

    def __add__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetConcatOp(_operands_of(self, "concat") + _operands_of(other, "concat"))

    def __or__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetSetOp("union", _operands_of(self, "union") + _operands_of(other, "union"))

    def __and__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetSetOp("intersection", _operands_of(self, "intersection") + _operands_of(other, "intersection"))

    def __sub__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        # a - (b - c) keeps c, so only the left operand splices.
        return LocsetSetOp("difference", _operands_of(self, "difference") + (other,))

    def unique(self) -> "LocsetExpr":
        """Return an expression that deduplicates locations after evaluation."""
        return LocsetUniqueOp(self)

    def evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        """Evaluate this expression against ``morpho``.

        Parameters
        ----------
        morpho : braincell.Morphology
            Morphology to evaluate against.
        cache : braincell.filter.SelectionCache or None
            Shared memoization scratch space, threaded to sub-expressions.

        Returns
        -------
        LocsetMask
            The selected point set.

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
    ) -> LocsetMask:
        """Compute the point set for a morphology already type-checked by :meth:`evaluate`."""


@dataclass(frozen=True)
class AtLocation(LocsetExpr):
    branch: int | str
    x: float

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if isinstance(self.branch, bool):
            raise TypeError("AtLocation.branch expects int or str, got bool.")
        if isinstance(self.branch, int):
            branch_view = morpho.branch(index=self.branch)
        elif isinstance(self.branch, str):
            branch_view = morpho.branch(name=self.branch)
        else:
            raise TypeError(f"AtLocation.branch expects int or str, got {type(self.branch).__name__!s}.")
        branch_id = branch_view.index
        points = helper.normalize_locset_points(((branch_id, self.x),))
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


def at(branch: int | str, x: float) -> AtLocation:
    """Select one location: the point at fraction ``x`` along one branch.

    Parameters
    ----------
    branch : int or str
        Branch index or branch name. It is looked up when the expression is
        evaluated: an unknown name raises ``KeyError`` and an index out of
        range raises ``IndexError``.
    x : float
        Position along the branch, from 0 (proximal end) to 1 (distal end).
        A value outside ``[0, 1]`` raises ``ValueError`` at evaluation.

    Returns
    -------
    AtLocation
        A locset expression that selects one point.

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
        >>> from braincell.filter import at
        >>> points = tree.select(at("dend", 0.5))
        >>> points.branch_id, points.branch_x
        (array([1]), array([0.5]))
    """
    return AtLocation(branch=branch, x=x)


@dataclass(frozen=True)
class RootLocation(LocsetExpr):
    x: float

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        points = helper.normalize_locset_points(((0, self.x),))
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class ForkPoints(LocsetExpr):
    """Select topology junctions incident to at least three branches."""

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        points = helper.fork_points_locations(morpho)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


BranchPoints = ForkPoints


@dataclass(frozen=True)
class Terminals(LocsetExpr):
    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        points = helper.terminal_locations(morpho)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class RegionAnchors(LocsetExpr):
    """Place one location in each interval of a region, at a relative position.

    Parameters
    ----------
    region : RegionExpr
        The region whose intervals receive a location each.
    x : float
        Relative position inside each interval, in ``[0, 1]``: the fraction
        of the interval's arc length from its proximal end, measured along
        the branch's own ``x`` direction. ``0`` is the interval's ``prox``,
        ``1`` its ``dist``. Values within ``1e-12`` of ``[0, 1]`` are clipped
        to it, as in :func:`at`.

    Returns
    -------
    LocsetMask
        From :meth:`~LocsetExpr.evaluate`: for each interval ``(b, p, d)`` of
        the normalized region, in ascending branch index and then ``p``, one
        row ``(b, p + x * (d - p))``. The number of rows equals the number of
        normalized intervals, and rows are never merged.

    Raises
    ------
    TypeError
        At construction, if ``region`` is not a :class:`RegionExpr`, or ``x``
        is a ``bool``, a :class:`~brainunit.Quantity` or not a real number.
    ValueError
        At construction, if ``x`` is NaN or outside ``[0, 1]``. At evaluation
        (``select``, ``Cell.place``), if the region is empty once normalized.

    See Also
    --------
    at : One location on one named branch.
    UniformSamples : A given number of locations spread over a region.

    Notes
    -----
    Normalization sorts the region's intervals by branch index and merges
    the ones that touch or overlap on the same branch, so
    ``BranchSlice(1, 0.0, 0.5) | BranchSlice(1, 0.5, 1.0)`` is one interval
    and gives one location. ``RegionAnchors(branch_in("name", ["dend"]), x)``
    gives the same location as ``at("dend", x)``.

    Intervals on two branches that meet at a junction can give two rows for
    the same physical point when ``x`` is ``0`` or ``1``: ``(soma, 1.0)``
    and ``(dend, 0.0)`` for a dendrite attached at the soma's distal end.
    :meth:`~LocsetExpr.unique` removes only identical rows, so it keeps
    both. ``Cell.place`` puts both at the same junction point, each on its
    own branch.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import AllRegion, BranchSlice, RegionAnchors
        >>> soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
        >>> dend = Branch.from_lengths(
        ...     lengths=np.array([40.0, 60.0]) * u.um, radii=np.array([2.0, 1.5, 1.0]) * u.um, type="basal_dendrite"
        ... )
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.dend = dend
        >>> tree.select(RegionAnchors(AllRegion(), 0.25)).points
        ((0, 0.25), (1, 0.25))
        >>> tree.select(RegionAnchors(AllRegion(), 0.25)).display_names
        ('soma(0.25)', 'dend(0.25)')
        >>> tree.select(RegionAnchors(BranchSlice(1, 0.2, 0.6) | BranchSlice(0, 0.5, 1.0), 0.5)).points
        ((0, 0.75), (1, 0.4))
    """

    region: RegionExpr
    x: float

    def __post_init__(self) -> None:
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"RegionAnchors.region expects RegionExpr, got {type(self.region).__name__!s}.")
        x = self.x
        if isinstance(x, Quantity):
            raise TypeError(f"RegionAnchors.x is a relative position in [0, 1], not a length; got {x!r}.")
        if isinstance(x, bool) or not isinstance(x, Real):
            raise TypeError(f"RegionAnchors.x must be a real number in [0, 1], got {x!r}.")
        if not (-helper.EPSILON <= float(x) <= 1.0 + helper.EPSILON):
            # NaN fails both comparisons and lands here too.
            raise ValueError(f"RegionAnchors.x must be within [0, 1], got {x!r}.")

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        mask = evaluate_cached(self.region, morpho, cache)
        # Leaf masks (e.g. a multi-branch BranchSlice) keep their input order,
        # and a slice shorter than the tolerance survives until normalized.
        intervals = helper.normalize_region_intervals(mask.intervals)
        if not intervals:
            raise ValueError(f"RegionAnchors cannot anchor on an empty region: {self.region!r} selects nothing.")
        x = min(1.0, max(0.0, float(self.x)))
        points = helper.normalize_locset_points(
            tuple((branch, prox + x * (dist - prox)) for branch, prox, dist in intervals)
        )
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class UniformSamples(LocsetExpr):
    region: RegionExpr
    count: int

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"UniformSamples.region expects RegionExpr, got {type(self.region).__name__!s}.")
        mask = evaluate_cached(self.region, morpho, cache)
        points = helper.uniform_samples_from_region(
            morpho,
            intervals=mask.intervals,
            count=self.count,
        )
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class RandomSamples(LocsetExpr):
    """Random locations, uniform in length over the intervals of a region.

    Each sample picks one interval of the region with probability
    proportional to its physical length, then a uniform position inside it.
    The draws use a NumPy ``default_rng(seed)``; for an explicit measure, a
    density or a preserved draw order use :func:`sample`.

    Parameters
    ----------
    region : RegionExpr
        Region to sample; an empty or zero-length region raises ``ValueError``
        when evaluated.
    count : int
        Number of locations.
    seed : int or None, optional
        Local seed of the draws. ``None`` (the default) draws an integer once,
        when the expression is built, from the BrainState default random
        stream, so an enclosing ``brainstate.random.seed_context(s)`` controls
        it; each such expression advances that stream by one draw. The drawn
        integer is stored in ``seed``, so every evaluation gives the same
        locations and ``RandomSamples(region, count, expr.seed)`` reproduces
        them. Inside ``brainstate.transform.jit``, ``for_loop`` or another
        transformation ``None`` raises ``TypeError``; build the expression
        outside it or pass an integer.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> import braincell
    >>> from braincell.filter import AllRegion, RandomSamples
    >>> soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    >>> morpho = braincell.Morphology.from_root(soma, name="soma")
    >>> with brainstate.random.seed_context(42):
    ...     sites = RandomSamples(AllRegion(), 3)
    >>> sites.evaluate(morpho).points == RandomSamples(AllRegion(), 3, sites.seed).evaluate(morpho).points
    True
    """

    region: RegionExpr
    count: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.seed is None:
            object.__setattr__(self, "seed", draw_seed("RandomSamples(seed=None)"))

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"RandomSamples.region expects RegionExpr, got {type(self.region).__name__!s}.")
        mask = evaluate_cached(self.region, morpho, cache)
        points = helper.random_samples_from_region(
            morpho,
            intervals=mask.intervals,
            count=self.count,
            seed=self.seed,
        )
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class SampleLocations(LocsetExpr):
    """Lazy continuous random sampling over a morphology region.

    Parameters
    ----------
    region : RegionExpr
        Continuous morphology support.
    number : int
        Number of locations to draw. Duplicates are retained.
    seed : int or None
        Local random seed. ``None`` draws one from the BrainState default
        random stream when the expression is built (see :func:`sample`).
    measure : {'normalized', 'length', 'lateral_area', 'area'}, optional
        Base geometry measure multiplied by ``density``. ``area`` includes
        discrete annular area at zero-length radius jumps.
    density : callable, optional
        Non-negative, dimensionless preference evaluated with a
        :class:`~braincell.filter.SamplingContext`.
    u_resolution : float, optional
        Target inverse-CDF error for custom densities.
    """

    region: RegionExpr
    number: int
    seed: int | None
    measure: str = "length"
    density: object | None = None
    u_resolution: float = 1e-10

    def __post_init__(self) -> None:
        if self.seed is None:
            object.__setattr__(self, "seed", draw_seed("sample(seed=None)"))

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"sample().region expects RegionExpr, got {type(self.region).__name__!s}.")
        mask = evaluate_cached(self.region, morpho, cache)
        branch_id, branch_x = sample_locations_from_region(
            morpho,
            intervals=mask.intervals,
            number=self.number,
            seed=self.seed,
            measure=self.measure,
            density=self.density,
            u_resolution=self.u_resolution,
        )
        return LocsetMask.from_columns(branch_id, branch_x)


def sample(
    region: RegionExpr,
    *,
    number: int,
    seed: int | None = None,
    measure: str = "length",
    density: object | None = None,
    u_resolution: float = 1e-10,
) -> SampleLocations:
    """Create a lazy continuous random-location expression.

    Parameters
    ----------
    region : RegionExpr
        Continuous morphology support.
    number : int
        Number of samples. Draw order and duplicate locations are preserved.
    seed : int or None, optional
        Seed local to this sampling rule. ``None`` (the default) draws an
        integer once, here, from the BrainState default random stream, so an
        enclosing ``brainstate.random.seed_context(s)`` (or an earlier
        ``brainstate.random.seed(s)``) controls the locations together with
        the other default-stream draws; each such call advances that stream
        by one draw. The drawn integer is stored in the returned expression's
        ``seed``, so every evaluation gives the same locations and
        ``sample(region, number=n, seed=expr.seed)`` reproduces them. Inside
        ``brainstate.transform.jit``, ``for_loop`` or another transformation
        ``None`` raises ``TypeError``; call it outside or pass an integer.
    measure : {'normalized', 'length', 'lateral_area', 'area'}, optional
        Geometry measure beneath the preference density. Default is physical
        branch length.
    density : callable, optional
        Non-negative dimensionless preference over ``SamplingContext``.
    u_resolution : float, optional
        Numerical inverse-CDF target in ``[1e-12, 1e-5]``.

    Returns
    -------
    SampleLocations
        Lazy locset expression resolved against a morphology later.

    Examples
    --------
    >>> import brainstate
    >>> import brainunit as u
    >>> import braincell
    >>> from braincell.filter import AllRegion, sample
    >>> soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    >>> morpho = braincell.Morphology.from_root(soma, name="soma")
    >>> with brainstate.random.seed_context(42):
    ...     sites = sample(AllRegion(), number=3)
    >>> again = sample(AllRegion(), number=3, seed=sites.seed)
    >>> sites.evaluate(morpho).points == again.evaluate(morpho).points
    True
    """
    return SampleLocations(
        region=region,
        number=number,
        seed=seed,
        measure=measure,
        density=density,
        u_resolution=u_resolution,
    )


@dataclass(frozen=True, eq=False)
class StepSamples(LocsetExpr):
    """Place locations a fixed physical step apart along a region.

    Parameters
    ----------
    region : RegionExpr
        The region to sample.
    step : brainunit.Quantity
        Arc length between consecutive locations, a positive, finite scalar
        length such as ``25 * u.um`` (a one-element array is accepted).

    Returns
    -------
    LocsetMask
        From :meth:`~LocsetExpr.evaluate`: with ``T`` the region's total
        length and ``h`` the step, ``n = floor(T / h)`` locations at arc
        lengths ``(k + 1/2) * h``, ``k = 0 ... n - 1``, along the
        concatenated region, in increasing arc length and without repeats.
        The first location is half a step in; a final partial step gets no
        location.

    Raises
    ------
    TypeError
        At construction, if ``region`` is not a :class:`RegionExpr`, or
        ``step`` is ``None``, a bare number or not a length.
    ValueError
        At construction, if ``step`` is not a scalar, or is zero, negative,
        infinite or NaN. At evaluation (``select``, ``Cell.place``), if the
        region is empty or has zero length once normalized, or if the step
        is longer than the region (``n = 0``).

    See Also
    --------
    UniformSamples : A given number of locations, evenly spaced.

    Notes
    -----
    The region's intervals are concatenated as in :class:`UniformSamples`:
    in ascending branch index and then ``prox``, dropping intervals of zero
    length. The step count runs on across branches, so a location can fall
    on any branch the region touches, and arc lengths map back to each
    branch the same way. When the step divides ``T`` into ``n`` equal parts,
    the result equals ``UniformSamples(region, n)``. The count ``floor(T / h)``
    allows a relative slack of ``1e-9``, so that ``0.3 um`` sampled every
    ``0.1 um`` gives three locations although ``0.3 / 0.1`` is
    ``2.9999999999999996`` in floating point.

    ``step`` keeps the quantity it was given, but equality and hashing use
    its value in micrometres: ``StepSamples(r, 30 * u.um)`` equals
    ``StepSamples(r, 0.03 * u.mmeter)``, and both can key a
    :class:`SelectionCache`.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> import numpy as np
        >>> from braincell import Branch, Morphology
        >>> from braincell.filter import AllRegion, StepSamples
        >>> soma = Branch.from_lengths(lengths=np.array([20.0]) * u.um, radii=np.array([5.0, 5.0]) * u.um, type="soma")
        >>> dend = Branch.from_lengths(
        ...     lengths=np.array([40.0, 60.0]) * u.um, radii=np.array([2.0, 1.5, 1.0]) * u.um, type="basal_dendrite"
        ... )
        >>> tree = Morphology.from_root(soma, name="soma")
        >>> tree.soma.dend = dend
        >>> tree.select(StepSamples(AllRegion(), 25 * u.um)).points
        ((0, 0.625), (1, 0.175), (1, 0.425), (1, 0.675))
        >>> tree.select(StepSamples(AllRegion(), 30 * u.um)).points
        ((0, 0.75), (1, 0.25), (1, 0.55), (1, 0.85))
    """

    region: RegionExpr
    step: Quantity

    def __post_init__(self) -> None:
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"StepSamples.region expects RegionExpr, got {type(self.region).__name__!s}.")
        object.__setattr__(
            self, "_step_um", helper._length_scalar_um(self.step, name="StepSamples.step", positive=True)
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.region == other.region and self._step_um == other._step_um

    def __hash__(self) -> int:
        return hash((type(self), self.region, self._step_um))

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        mask = evaluate_cached(self.region, morpho, cache)
        points = helper._step_samples_from_region(morpho, intervals=mask.intervals, step_um=self._step_um)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class LocsetConcatOp(LocsetExpr):
    """Deferred duplicate-preserving concatenation of locset expressions.

    Parameters
    ----------
    operands : tuple of LocsetExpr
        Expressions evaluated and concatenated from left to right. ``+``
        extends a concatenation instead of nesting it, so ``a + b + c`` is
        one node of three operands.
    """

    operands: tuple[LocsetExpr, ...]

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if len(self.operands) < 2:
            raise ValueError("concatenation expects at least two operands.")
        # Accumulate raw, normalize once. concat_locset_points() normalizes
        # left + right on every call, so folding through it made the pass
        # O(k^2 * m) over data that was already validated by each operand.
        collected: list[Location] = []
        for operand in self.operands:
            collected.extend(evaluate_cached(operand, morpho, cache).points)
        points = helper.normalize_locset_points(tuple(collected))
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class LocsetSetOp(LocsetExpr):
    """Composite selector produced by set algebra on locset expressions.

    Notes
    -----
    The operands are combined from left to right. The ``|``, ``&`` and ``-``
    operators extend a node of the same operation instead of nesting it, so
    ``a | b | c`` is one union of three operands however many are chained;
    only the right operand of ``-`` stays nested, since ``a - (b - c)``
    differs from ``(a - b) - c``.
    """

    op: str
    operands: tuple[LocsetExpr, ...]

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        if self.op not in {"union", "intersection", "difference"}:
            raise ValueError(f"Unsupported locset set operation {self.op!r}.")
        if len(self.operands) < 2:
            raise ValueError(f"{self.op} expects at least two operands.")

        # As in RegionSetOp: union_/intersect_/difference_locset_points each
        # normalize through unique_locset_points, and LocsetMask.points is
        # already canonical, so an outer pass is the third of three.
        current = evaluate_cached(self.operands[0], morpho, cache).points
        for operand in self.operands[1:]:
            other = evaluate_cached(operand, morpho, cache).points
            if self.op == "union":
                current = helper.union_locset_points(current, other)
            elif self.op == "intersection":
                current = helper.intersect_locset_points(current, other)
            else:
                current = helper.difference_locset_points(current, other)
        return LocsetMask(points=current, display_names=_display_names_for_points(morpho, current))


def _operands_of(expr: LocsetExpr, op: str) -> tuple[LocsetExpr, ...]:
    """Return ``expr``'s operands if it is a well-formed ``op`` node, else ``(expr,)``.

    ``op`` is ``"concat"`` for :class:`LocsetConcatOp`. As in the region
    operators, splicing keeps a locset built one operand at a time in a loop
    one level deep: concatenation, union and intersection are associative,
    and :meth:`LocsetSetOp._evaluate` folds a difference left to right.
    Nesting recursed once per level and exceeded Python's recursion limit
    after a few hundred.
    """
    if op == "concat":
        same = isinstance(expr, LocsetConcatOp)
    else:
        same = isinstance(expr, LocsetSetOp) and expr.op == op
    if same and len(expr.operands) >= 2:
        return tuple(expr.operands)
    return (expr,)


@dataclass(frozen=True)
class LocsetUniqueOp(LocsetExpr):
    """Deferred stable deduplication of a locset expression.

    Parameters
    ----------
    operand : LocsetExpr
        Expression whose repeated canonical location rows are removed.
    """

    operand: LocsetExpr

    def _evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        return evaluate_cached(self.operand, morpho, cache).unique()


def _display_names_for_points(morpho: Morphology, points: tuple[Location, ...]) -> tuple[str, ...]:
    # Hoist the branch tuple out of the loop: this runs once per sampled
    # point, and ``sample(number=10000)`` would otherwise resolve the branch
    # index ten thousand times.
    branches = morpho.branches
    return tuple(f"{branches[int(branch_id)].name}({float(x):g})" for branch_id, x in points)
