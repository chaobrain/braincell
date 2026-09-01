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

import numpy as np

from brainunit import Quantity

from braincell.morph.morphology import Morphology
from . import helper
from .cache import SelectionCache
from .region import RegionExpr

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
        np.round(np.clip(normalized_x, 0.0, 1.0), decimals=12),
    )


@dataclass(init=False, eq=False, repr=False)
class LocsetBatch:
    """A read-only batch of aligned resolved locsets.

    The leading dimension identifies population members and the trailing
    dimension contains the locations assigned to each member.

    Parameters
    ----------
    branch_id : array-like
        Two-dimensional integer branch identifiers.
    branch_x : array-like
        Two-dimensional normalized branch coordinates in ``[0, 1]``.
    display_names : iterable of iterable of str, optional
        Human-readable names aligned with the location matrix.
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
        """Construct a batch from two aligned location matrices."""
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


@dataclass(init=False, eq=False, repr=False)
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
        morpho : braincell.morph.Morphology
            Morphology used to resolve branch names.

        Returns
        -------
        tuple of str
            One label per location row.
        """
        if self._display_names is not None:
            return self._display_names
        return tuple(
            f"{morpho.branch(index=int(branch_id)).name}({float(branch_x):g})" for branch_id, branch_x in self.points
        )

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
    def __add__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetConcatOp((self, other))

    def __or__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetSetOp("union", (self, other))

    def __and__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetSetOp("intersection", (self, other))

    def __sub__(self, other: "LocsetExpr") -> "LocsetExpr":
        if not isinstance(other, LocsetExpr):
            return NotImplemented
        return LocsetSetOp("difference", (self, other))

    def unique(self) -> "LocsetExpr":
        """Return an expression that deduplicates locations after evaluation."""
        return LocsetUniqueOp(self)

    @abstractmethod
    def evaluate(
        self,
        morpho: Morphology,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        raise NotImplementedError


@dataclass(frozen=True)
class AtLocation(LocsetExpr):
    branch: int | str
    x: float

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        _ = cache
        if not isinstance(morpho, Morphology):
            raise TypeError(f"AtLocation expects Morpho, got {type(morpho).__name__!s}.")
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
    return AtLocation(branch=branch, x=x)


@dataclass(frozen=True)
class RootLocation(LocsetExpr):
    x: float

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"RootLocation expects Morpho, got {type(morpho).__name__!s}.")
        points = helper.normalize_locset_points(((0, self.x),))
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class ForkPoints(LocsetExpr):
    """Select topology junctions incident to at least three branches."""

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"ForkPoints expects Morpho, got {type(morpho).__name__!s}.")
        points = helper.fork_points_locations(morpho)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


BranchPoints = ForkPoints


@dataclass(frozen=True)
class Terminals(LocsetExpr):
    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"Terminals expects Morpho, got {type(morpho).__name__!s}.")
        points = helper.terminal_locations(morpho)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class RegionAnchors(LocsetExpr):
    region: RegionExpr
    x: float

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        raise NotImplementedError("RegionAnchors is not implemented in this version.")


@dataclass(frozen=True)
class UniformSamples(LocsetExpr):
    region: RegionExpr
    count: int

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"UniformSamples expects Morpho, got {type(morpho).__name__!s}.")
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"UniformSamples.region expects RegionExpr, got {type(self.region).__name__!s}.")
        mask = self.region.evaluate(morpho, cache=cache)
        points = helper.uniform_samples_from_region(
            morpho,
            intervals=mask.intervals,
            count=self.count,
        )
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class RandomSamples(LocsetExpr):
    region: RegionExpr
    count: int
    seed: int

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"RandomSamples expects Morpho, got {type(morpho).__name__!s}.")
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"RandomSamples.region expects RegionExpr, got {type(self.region).__name__!s}.")
        mask = self.region.evaluate(morpho, cache=cache)
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
    seed : int
        Explicit local random seed.
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
    seed: int
    measure: str = "length"
    density: object | None = None
    u_resolution: float = 1e-10

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"SampleLocations expects Morpho, got {type(morpho).__name__!s}.")
        if not isinstance(self.region, RegionExpr):
            raise TypeError(f"sample().region expects RegionExpr, got {type(self.region).__name__!s}.")
        from ._sampling import sample_locations_from_region

        mask = self.region.evaluate(morpho, cache=cache)
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
    seed: int,
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
    seed : int
        Explicit seed local to this sampling rule.
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
    """
    return SampleLocations(
        region=region,
        number=number,
        seed=seed,
        measure=measure,
        density=density,
        u_resolution=u_resolution,
    )


@dataclass(frozen=True)
class StepSamples(LocsetExpr):
    region: RegionExpr
    step: Quantity

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        raise NotImplementedError


@dataclass(frozen=True)
class LocsetConcatOp(LocsetExpr):
    """Deferred duplicate-preserving concatenation of locset expressions.

    Parameters
    ----------
    operands : tuple of LocsetExpr
        Expressions evaluated and concatenated from left to right.
    """

    operands: tuple[LocsetExpr, ...]

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"LocsetConcatOp expects Morpho, got {type(morpho).__name__!s}.")
        if len(self.operands) < 2:
            raise ValueError("concatenation expects at least two operands.")
        points: tuple[Location, ...] = ()
        for operand in self.operands:
            points = helper.concat_locset_points(points, operand.evaluate(morpho, cache=cache).points)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class LocsetSetOp(LocsetExpr):
    op: str
    operands: tuple[LocsetExpr, ...]

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"LocsetSetOp expects Morpho, got {type(morpho).__name__!s}.")
        if self.op not in {"union", "intersection", "difference"}:
            raise ValueError(f"Unsupported locset set operation {self.op!r}.")
        if len(self.operands) < 2:
            raise ValueError(f"{self.op} expects at least two operands.")

        current = helper.normalize_locset_points(self.operands[0].evaluate(morpho, cache=cache).points)
        for operand in self.operands[1:]:
            other = helper.normalize_locset_points(operand.evaluate(morpho, cache=cache).points)
            if self.op == "union":
                current = helper.union_locset_points(current, other)
            elif self.op == "intersection":
                current = helper.intersect_locset_points(current, other)
            else:
                current = helper.difference_locset_points(current, other)
        return LocsetMask(points=current, display_names=_display_names_for_points(morpho, current))


@dataclass(frozen=True)
class LocsetUniqueOp(LocsetExpr):
    """Deferred stable deduplication of a locset expression.

    Parameters
    ----------
    operand : LocsetExpr
        Expression whose repeated canonical location rows are removed.
    """

    operand: LocsetExpr

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"LocsetUniqueOp expects Morpho, got {type(morpho).__name__!s}.")
        return self.operand.evaluate(morpho, cache=cache).unique()


def _display_names_for_points(morpho: Morphology, points: tuple[Location, ...]) -> tuple[str, ...]:
    return tuple(_display_name_for_point(morpho, point) for point in points)


def _display_name_for_point(morpho: Morphology, point: Location) -> str:
    branch_id, x = point
    branch_name = morpho.branch(index=int(branch_id)).name
    return f"{branch_name}({float(x):g})"
