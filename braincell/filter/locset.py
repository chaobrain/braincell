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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..morpho import Morpho
from . import helper
from .cache import SelectionCache
from .region import RegionExpr

Quantity = Any
Location = tuple[int, float]

__all__ = [
    "LocsetMask",
    "LocsetExpr",
    "RootLocation",
    "BranchPoints",
    "Terminals",
    "RegionAnchors",
    "UniformSamples",
    "RandomSamples",
    "StepSamples",
    "LocsetSetOp",
]


@dataclass(frozen=True)
class LocsetMask:
    points: tuple[Location, ...]


class LocsetExpr(ABC):
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

    @abstractmethod
    def evaluate(
        self,
        morpho: Morpho,
        cache: SelectionCache | None = None,
    ) -> LocsetMask:
        raise NotImplementedError


@dataclass(frozen=True)
class RootLocation(LocsetExpr):
    x: float

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"RootLocation expects Morpho, got {type(morpho).__name__!s}.")
        return LocsetMask(helper.normalize_locset_points(((0, self.x),)))


@dataclass(frozen=True)
class BranchPoints(LocsetExpr):
    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"BranchPoints expects Morpho, got {type(morpho).__name__!s}.")
        return LocsetMask(helper.branch_points_locations(morpho))


@dataclass(frozen=True)
class Terminals(LocsetExpr):
    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"Terminals expects Morpho, got {type(morpho).__name__!s}.")
        return LocsetMask(helper.terminal_locations(morpho))


@dataclass(frozen=True)
class RegionAnchors(LocsetExpr):
    region: RegionExpr
    x: float

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        raise NotImplementedError("RegionAnchors is not implemented in this version.")


@dataclass(frozen=True)
class UniformSamples(LocsetExpr):
    region: RegionExpr
    count: int

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"UniformSamples expects Morpho, got {type(morpho).__name__!s}.")
        if not isinstance(self.region, RegionExpr):
            raise TypeError(
                "UniformSamples.region expects RegionExpr, "
                f"got {type(self.region).__name__!s}."
            )
        mask = self.region.evaluate(morpho, cache=cache)
        return LocsetMask(
            helper.uniform_samples_from_region(
                morpho,
                intervals=mask.intervals,
                count=self.count,
            )
        )


@dataclass(frozen=True)
class RandomSamples(LocsetExpr):
    region: RegionExpr
    count: int
    seed: int

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
            raise TypeError(f"RandomSamples expects Morpho, got {type(morpho).__name__!s}.")
        if not isinstance(self.region, RegionExpr):
            raise TypeError(
                "RandomSamples.region expects RegionExpr, "
                f"got {type(self.region).__name__!s}."
            )
        mask = self.region.evaluate(morpho, cache=cache)
        return LocsetMask(
            helper.random_samples_from_region(
                morpho,
                intervals=mask.intervals,
                count=self.count,
                seed=self.seed,
            )
        )


@dataclass(frozen=True)
class StepSamples(LocsetExpr):
    region: RegionExpr
    step: Quantity

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        raise NotImplementedError


@dataclass(frozen=True)
class LocsetSetOp(LocsetExpr):
    op: str
    operands: tuple[LocsetExpr, ...]

    def evaluate(self, morpho: Morpho, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morpho):
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
        return LocsetMask(current)
