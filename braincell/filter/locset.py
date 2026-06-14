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

from brainunit import Quantity

from braincell.morph.morphology import Morphology
from . import helper
from .cache import SelectionCache
from .region import RegionExpr

Location = tuple[int, float]

__all__ = [
    "LocsetMask",
    "LocsetExpr",
    "AtLocation",
    "at",
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
    display_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.display_names):
            raise ValueError(
                "LocsetMask.points and display_names must have the same length, "
                f"got {len(self.points)!r} and {len(self.display_names)!r}."
            )


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
            raise TypeError(
                "AtLocation.branch expects int or str, "
                f"got {type(self.branch).__name__!s}."
            )
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
class BranchPoints(LocsetExpr):
    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        if not isinstance(morpho, Morphology):
            raise TypeError(f"BranchPoints expects Morpho, got {type(morpho).__name__!s}.")
        points = helper.branch_points_locations(morpho)
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


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
            raise TypeError(
                "UniformSamples.region expects RegionExpr, "
                f"got {type(self.region).__name__!s}."
            )
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
            raise TypeError(
                "RandomSamples.region expects RegionExpr, "
                f"got {type(self.region).__name__!s}."
            )
        mask = self.region.evaluate(morpho, cache=cache)
        points = helper.random_samples_from_region(
            morpho,
            intervals=mask.intervals,
            count=self.count,
            seed=self.seed,
        )
        return LocsetMask(points=points, display_names=_display_names_for_points(morpho, points))


@dataclass(frozen=True)
class StepSamples(LocsetExpr):
    region: RegionExpr
    step: Quantity

    def evaluate(self, morpho: Morphology, cache: SelectionCache | None = None) -> LocsetMask:
        raise NotImplementedError


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


def _display_names_for_points(morpho: Morphology, points: tuple[Location, ...]) -> tuple[str, ...]:
    return tuple(_display_name_for_point(morpho, point) for point in points)


def _display_name_for_point(morpho: Morphology, point: Location) -> str:
    branch_id, x = point
    branch_name = morpho.branch(index=int(branch_id)).name
    return f"{branch_name}({float(x):g})"
