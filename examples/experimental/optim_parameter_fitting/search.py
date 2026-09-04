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

"""Derivative-free stage contracts and candidate handoff helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateSet:
    """Carry physical parameter rows and provenance between optimization stages."""

    physical: np.ndarray
    candidate_id: np.ndarray
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        physical = np.asarray(self.physical)
        identifiers = np.asarray(self.candidate_id)
        if physical.ndim != 2 or physical.shape[0] < 1:
            raise ValueError(f"physical candidates must have shape (candidate,parameter), got {physical.shape!r}.")
        if identifiers.shape != (physical.shape[0],) or len(self.provenance) != physical.shape[0]:
            raise ValueError("candidate_id and provenance must have one entry per candidate.")
        if not np.all(np.isfinite(physical)):
            raise ValueError("Physical candidates must be finite.")

    @property
    def size(self) -> int:
        """Return the number of candidates."""
        return int(self.physical.shape[0])

    def replace(self, physical, *, stage_name: str) -> "CandidateSet":
        """Return changed physical values with explicit stage provenance."""
        values = np.asarray(physical, dtype=np.float64)
        if values.shape != self.physical.shape:
            raise ValueError("A replacement stage must preserve candidate shape.")
        return CandidateSet(values, self.candidate_id.copy(), tuple(stage_name for _ in range(self.size)))


@dataclass(frozen=True)
class SearchStageResult:
    """Return derivative-free candidates plus explicit forward-evaluation accounting."""

    candidates: CandidateSet
    forward_evaluations: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.forward_evaluations < 0:
            raise ValueError("forward_evaluations must be non-negative.")


@dataclass(frozen=True)
class ForwardSelectionStage:
    """Define the minimal forward-only search-stage interface for later methods."""

    name: str = "forward_selection"
    kind: str = "derivative_free"
    resets_optimizer_state: bool = True

    def run(self, _context, candidates: CandidateSet) -> SearchStageResult:
        """Pass candidates through while exercising the stage handoff contract."""
        return SearchStageResult(
            candidates=candidates.replace(candidates.physical, stage_name=self.name),
            forward_evaluations=0,
            metadata={"status": "pass_through_contract_only"},
        )

    def describe(self) -> dict[str, object]:
        """Return serializable stage metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "coordinate": "normalized_bounded_u",
            "uses_gradient": False,
            "ranking_split": "train",
            "resets_optimizer_state": self.resets_optimizer_state,
        }
