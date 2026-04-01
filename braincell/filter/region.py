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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..morpho import Morpho
from . import helper
from .cache import SelectionCache

Quantity = Any
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
    """Materialized region selection as `(branch, prox, dist)` intervals."""

    intervals: tuple[tuple[int, float, float], ...]


class RegionExpr(ABC):
    """Base class for lazy region selectors."""

    def __or__(self, other: "RegionExpr") -> "RegionExpr":
        return RegionSetOp("union", (self, other))

    def __and__(self, other: "RegionExpr") -> "RegionExpr":
        return RegionSetOp("intersection", (self, other))

    def __sub__(self, other: "RegionExpr") -> "RegionExpr":
        return RegionSetOp("difference", (self, other))

    def complement(self) -> "RegionExpr":
        return RegionSetOp("complement", (self,))

    @abstractmethod
    def evaluate(
        self,
        morpho: Morpho,
        cache: SelectionCache | None = None,
    ) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True)
class AllRegion(RegionExpr):
    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"AllRegion expects Morpho, got {type(morpho).__name__!s}.")
        return RegionMask(tuple((index, 0.0, 1.0) for index, _ in enumerate(morpho.branches)))


@dataclass(frozen=True)
class EmptyRegion(RegionExpr):
    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        return RegionMask(())


@dataclass(frozen=True)
class BranchSlice(RegionExpr):
    branch_index: object
    prox: object
    dist: object

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"BranchSlice expects Morpho, got {type(morpho).__name__!s}.")
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

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"BranchInFilter expects Morpho, got {type(morpho).__name__!s}.")
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

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"BranchRangeFilter expects Morpho, got {type(morpho).__name__!s}.")
        return RegionMask(
            helper.branch_range_intervals(
                morpho,
                property_name=self.property,
                bounds=self.bounds,
                closed=self.closed,
            )
        )


@dataclass(frozen=True)
class RadiusRangeRegion(RegionExpr):
    minimum: Quantity
    maximum: Quantity

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True)
class TreeDistanceRegion(RegionExpr):
    minimum: Quantity
    maximum: Quantity

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True)
class EuclideanDistanceRegion(RegionExpr):
    minimum: Quantity
    maximum: Quantity

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True)
class SubtreeRegion(RegionExpr):
    root_branch_index: int

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        raise NotImplementedError


@dataclass(frozen=True)
class RegionSetOp(RegionExpr):
    """Composite selector produced by set algebra on region expressions."""

    op: str
    operands: tuple[RegionExpr, ...]

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> RegionMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"RegionSetOp expects Morpho, got {type(morpho).__name__!s}.")
        op = self.op
        operands = self.operands

        if op == "complement":
            if len(operands) != 1:
                raise ValueError("complement expects exactly one operand.")
            source = operands[0].evaluate(morpho, cache).intervals
            intervals = helper.complement_region_intervals(
                source,
                n_branches=len(morpho.branches),
            )
            return RegionMask(intervals)

        if op not in {"union", "intersection", "difference"}:
            raise ValueError(f"Unsupported region set operation {op!r}.")
        if len(operands) < 2:
            raise ValueError(f"{op} expects at least two operands.")

        current = helper.normalize_region_intervals(operands[0].evaluate(morpho, cache).intervals)
        for operand in operands[1:]:
            other = helper.normalize_region_intervals(operand.evaluate(morpho, cache).intervals)
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
