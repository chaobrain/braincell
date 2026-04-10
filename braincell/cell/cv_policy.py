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

import brainunit as u
import numpy as np

from braincell.morpho import Morpho

EPSILON = 1e-12
Bounds = tuple[tuple[float, float], ...]
BoundsByBranch = tuple[Bounds, ...]


@dataclass(frozen=True)
class CVPolicy(ABC):
    """Base class for morphology-to-CV discretization policies.

    A ``CVPolicy`` answers one question: for each branch in a morphology, which
    normalized intervals should become control volumes. The rest of the cell
    stack treats policy output as the source of truth for CV splitting.

    ``Cell`` owns one policy instance. During rebuild it calls
    :meth:`resolve_cv_bounds`, passes the result into :func:`build_cv_geo`, and
    then applies paint/place rules on top of those intervals.
    """

    @abstractmethod
    def resolve_cv_bounds(self, morpho: Morpho) -> BoundsByBranch:
        """Return normalized CV intervals for each branch."""


@dataclass(frozen=True)
class CVPerBranch(CVPolicy):
    """Assign the same number of CVs to every branch.

    This is the default policy used by :class:`Cell`. ``cv_per_branch=1`` means
    one CV per branch; larger values split every branch uniformly in normalized
    branch coordinates.
    """

    cv_per_branch: int = 1

    def resolve_cv_bounds(self, morpho: Morpho) -> BoundsByBranch:
        cv_per_branch = self.cv_per_branch
        if isinstance(cv_per_branch, bool) or not isinstance(cv_per_branch, int):
            raise TypeError(f"cv_per_branch must be integer, got {cv_per_branch!r}.")
        if cv_per_branch <= 0:
            raise ValueError(f"cv_per_branch must be > 0, got {cv_per_branch!r}.")
        n_per_branch = int(cv_per_branch)
        return tuple(
            tuple(
                (float(offset) / float(n_per_branch), float(offset + 1) / float(n_per_branch))
                for offset in range(n_per_branch)
            )
            for _ in morpho.branches
        )


@dataclass(frozen=True)
class MaxCVLen(CVPolicy):
    """Split branches so each CV stays below a target physical length.

    The policy computes the number of CVs independently for each branch from its
    real path length and then splits the branch uniformly in normalized branch
    coordinates.
    """

    max_cv_len: u.Quantity[u.um]

    def resolve_cv_bounds(self, morpho: Morpho) -> BoundsByBranch:
        max_cv_len = self.max_cv_len
        if not hasattr(max_cv_len, "to_decimal"):
            raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.")
        try:
            max_len_um = float(np.asarray(max_cv_len.to_decimal(u.um), dtype=float))
        except Exception as exc:  # pragma: no cover - defensive for foreign quantity types
            raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.") from exc
        if not np.isfinite(max_len_um) or max_len_um <= 0.0:
            raise ValueError(f"max_cv_len must be > 0, got {max_cv_len!r}.")

        return tuple(_bounds_from_max_len_um(branch, max_len_um=max_len_um) for branch in morpho.branches)


@dataclass(frozen=True)
class DLambda(CVPolicy):
    """Placeholder for future d-lambda-based discretization.

    The type exists so higher-level APIs can already speak in terms of
    ``CVPolicy`` variants, but it intentionally raises
    ``NotImplementedError`` today.
    """

    def resolve_cv_bounds(self, morpho: Morpho) -> BoundsByBranch:
        raise NotImplementedError("DLambda cv policy is not implemented yet.")


def _bounds_from_max_len_um(branch, *, max_len_um: float) -> Bounds:
    branch_len_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    if branch_len_um <= max_len_um + EPSILON:
        return ((0.0, 1.0),)
    n_cv = int(np.ceil((branch_len_um / max_len_um) - EPSILON))
    n_cv = max(1, n_cv)
    return tuple(
        (float(offset) / float(n_cv), float(offset + 1) / float(n_cv))
        for offset in range(n_cv)
    )
