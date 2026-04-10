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
from typing import TYPE_CHECKING

import brainunit as u
import numpy as np

from braincell.mech import CableProperties
from braincell.morph import Morphology, branch_class_for_type

if TYPE_CHECKING:
    from .cv_mech import PaintRule

EPSILON = 1e-12
Bounds = tuple[tuple[float, float], ...]
BoundsByBranch = tuple[Bounds, ...]
_DEFAULT_D_LAMBDA_CABLE = CableProperties(
    resting_potential=-65.0 * u.mV,
    membrane_capacitance=1.0 * (u.uF / u.cm ** 2),
    axial_resistivity=100.0 * (u.ohm * u.cm),
)


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
    def resolve_cv_bounds(
        self,
        morpho: Morphology,
        *,
        paint_rules: tuple["PaintRule", ...] | None = None,
    ) -> BoundsByBranch:
        """Return normalized CV intervals for each branch."""


@dataclass(frozen=True)
class CVPerBranch(CVPolicy):
    """Assign the same number of CVs to every branch.

    This is the default policy used by :class:`Cell`. ``cv_per_branch=1`` means
    one CV per branch; larger values split every branch uniformly in normalized
    branch coordinates.
    """

    cv_per_branch: int = 1

    def resolve_cv_bounds(
        self,
        morpho: Morphology,
        *,
        paint_rules: tuple["PaintRule", ...] | None = None,
    ) -> BoundsByBranch:
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
    keep_odd: bool = True

    def resolve_cv_bounds(
        self,
        morpho: Morphology,
        *,
        paint_rules: tuple["PaintRule", ...] | None = None,
    ) -> BoundsByBranch:
        max_cv_len = self.max_cv_len
        if not hasattr(max_cv_len, "to_decimal"):
            raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.")
        try:
            max_len_um = float(np.asarray(max_cv_len.to_decimal(u.um), dtype=float))
        except Exception as exc:  # pragma: no cover - defensive for foreign quantity types
            raise TypeError(f"max_cv_len must be a length Quantity, got {max_cv_len!r}.") from exc
        if not np.isfinite(max_len_um) or max_len_um <= 0.0:
            raise ValueError(f"max_cv_len must be > 0, got {max_cv_len!r}.")

        return tuple(
            _bounds_from_max_len_um(branch, max_len_um=max_len_um, keep_odd=self.keep_odd)
            for branch in morpho.branches
        )


@dataclass(frozen=True)
class DLambda(CVPolicy):
    """Placeholder for future d-lambda-based discretization.

    The type exists so higher-level APIs can already speak in terms of
    ``CVPolicy`` variants, but it intentionally raises
    ``NotImplementedError`` today.
    """

    d_lambda: float
    frequency: u.Quantity[u.Hz] = 100.0 * u.Hz
    keep_odd: bool = True

    def resolve_cv_bounds(
        self,
        morpho: Morphology,
        *,
        paint_rules: tuple["PaintRule", ...] | None = None,
    ) -> BoundsByBranch:
        d_lambda = float(self.d_lambda)
        if not np.isfinite(d_lambda) or d_lambda <= 0.0:
            raise ValueError(f"d_lambda must be > 0, got {self.d_lambda!r}.")
        frequency = self.frequency
        if not hasattr(frequency, "to_decimal"):
            raise TypeError(f"frequency must be a frequency Quantity, got {frequency!r}.")
        try:
            freq_hz = float(np.asarray(frequency.to_decimal(u.Hz), dtype=float))
        except Exception as exc:  # pragma: no cover - defensive for foreign quantity types
            raise TypeError(f"frequency must be a frequency Quantity, got {frequency!r}.") from exc
        if not np.isfinite(freq_hz) or freq_hz <= 0.0:
            raise ValueError(f"frequency must be > 0, got {frequency!r}.")

        branch_cables = _resolve_branch_cable_properties(
            morpho,
            paint_rules=paint_rules,
        )
        return tuple(
            _bounds_from_d_lambda(
                morpho.branches[branch_id],
                ra_ohm_cm=ra_ohm_cm,
                cm_uF_per_cm2=cm_uF_per_cm2,
                frequency_hz=freq_hz,
                d_lambda=d_lambda,
                keep_odd=self.keep_odd,
            )
            for branch_id, (ra_ohm_cm, cm_uF_per_cm2) in enumerate(branch_cables)
        )


@dataclass(frozen=True)
class CVPolicyByTypeRule:
    branch_types: tuple[str, ...]
    policy: CVPolicy

    def __post_init__(self) -> None:
        if len(self.branch_types) == 0:
            raise ValueError("CVPolicyByTypeRule.branch_types must be non-empty.")
        normalized: list[str] = []
        for branch_type in self.branch_types:
            if not isinstance(branch_type, str):
                raise TypeError(f"branch_types entries must be str, got {branch_type!r}.")
            branch_class_for_type(branch_type)
            normalized.append(branch_type)
        if not isinstance(self.policy, CVPolicy):
            raise TypeError(f"policy must be CVPolicy, got {type(self.policy).__name__!s}.")
        object.__setattr__(self, "branch_types", tuple(normalized))


@dataclass(frozen=True)
class CompositeByTypePolicy(CVPolicy):
    rules: tuple[CVPolicyByTypeRule, ...]
    default_policy: CVPolicy

    def __post_init__(self) -> None:
        for rule in self.rules:
            if not isinstance(rule, CVPolicyByTypeRule):
                raise TypeError(
                    "CompositeByTypePolicy.rules must contain CVPolicyByTypeRule instances."
                )
        if not isinstance(self.default_policy, CVPolicy):
            raise TypeError(
                f"default_policy must be CVPolicy, got {type(self.default_policy).__name__!s}."
            )

    def resolve_cv_bounds(
        self,
        morpho: Morphology,
        *,
        paint_rules: tuple["PaintRule", ...] | None = None,
    ) -> BoundsByBranch:
        effective_policies: list[CVPolicy] = [self.default_policy for _ in morpho.branches]
        for rule in self.rules:
            targets = set(rule.branch_types)
            for branch_id, branch in enumerate(morpho.branches):
                if branch.type in targets:
                    effective_policies[branch_id] = rule.policy

        bounds_cache: dict[CVPolicy, BoundsByBranch] = {}
        out: list[Bounds] = []
        for branch_id, policy in enumerate(effective_policies):
            branch_bounds = bounds_cache.get(policy)
            if branch_bounds is None:
                branch_bounds = policy.resolve_cv_bounds(morpho, paint_rules=paint_rules)
                bounds_cache[policy] = branch_bounds
            out.append(branch_bounds[branch_id])
        return tuple(out)


def _bounds_from_max_len_um(branch, *, max_len_um: float, keep_odd: bool) -> Bounds:
    branch_len_um = float(np.asarray(branch.length.to_decimal(u.um), dtype=float))
    if branch_len_um <= max_len_um + EPSILON:
        return ((0.0, 1.0),)
    n_cv = int(np.ceil((branch_len_um / max_len_um) - EPSILON))
    n_cv = max(1, n_cv)
    n_cv = _promote_to_odd(n_cv, keep_odd=keep_odd)
    return tuple(
        (float(offset) / float(n_cv), float(offset + 1) / float(n_cv))
        for offset in range(n_cv)
    )


def _promote_to_odd(n_cv: int, *, keep_odd: bool) -> int:
    if not keep_odd:
        return int(n_cv)
    if int(n_cv) % 2 == 0:
        return int(n_cv) + 1
    return int(n_cv)


def _bounds_from_d_lambda(
    branch,
    *,
    ra_ohm_cm: float,
    cm_uF_per_cm2: float,
    frequency_hz: float,
    d_lambda: float,
    keep_odd: bool,
) -> Bounds:
    lengths_um = np.asarray(branch.lengths.to_decimal(u.um), dtype=float)
    r0_um = np.asarray(branch.radii_proximal.to_decimal(u.um), dtype=float)
    r1_um = np.asarray(branch.radii_distal.to_decimal(u.um), dtype=float)
    diam_um = r0_um + r1_um
    if np.any(diam_um <= 0.0):
        raise ValueError("DLambda requires strictly positive branch diameters.")

    lambda_f_um = 1.0e5 * np.sqrt(diam_um / (4.0 * np.pi * frequency_hz * ra_ohm_cm * cm_uF_per_cm2))
    electrotonic_length = float(np.sum(lengths_um / lambda_f_um))
    n_cv = int(np.ceil((electrotonic_length / d_lambda) - EPSILON))
    n_cv = max(1, n_cv)
    n_cv = _promote_to_odd(n_cv, keep_odd=keep_odd)
    return tuple(
        (float(offset) / float(n_cv), float(offset + 1) / float(n_cv))
        for offset in range(n_cv)
    )


def _resolve_branch_cable_properties(
    morpho: Morphology,
    *,
    paint_rules: tuple["PaintRule", ...] | None,
) -> tuple[tuple[float, float], ...]:
    branch_intervals: list[list[tuple[float, float, CableProperties]]] = [
        [(0.0, 1.0, _DEFAULT_D_LAMBDA_CABLE)] for _ in morpho.branches
    ]
    for rule in paint_rules or ():
        mechanism = getattr(rule, "mechanism", None)
        if not isinstance(mechanism, CableProperties):
            continue
        region = getattr(rule, "region", None)
        if region is None or not hasattr(region, "evaluate"):
            continue
        intervals = region.evaluate(morpho).intervals
        for branch_id, prox, dist in intervals:
            branch_intervals[int(branch_id)].append((float(prox), float(dist), mechanism))

    out: list[tuple[float, float]] = []
    for branch_id, rules in enumerate(branch_intervals):
        boundaries = [0.0, 1.0]
        for prox, dist, _ in rules:
            boundaries.append(float(prox))
            boundaries.append(float(dist))
        pieces = _sorted_unique_coords(boundaries)
        effective: list[tuple[float, float]] = []
        for left, right in zip(pieces[:-1], pieces[1:]):
            if right - left <= EPSILON:
                continue
            x = 0.5 * (left + right)
            cable = _last_cable_covering(rules, x=x)
            effective.append(_cable_signature(cable))
        first = effective[0]
        for signature in effective[1:]:
            if not _same_cable_signature(first, signature):
                raise ValueError(
                    "DLambda requires branch-wise uniform cable properties, "
                    f"but branch {branch_id} has multiple Ra/cm values. "
                    "Unify Ra/cm within the branch or use another cv_policy."
                )
        out.append(first)
    return tuple(out)


def _sorted_unique_coords(values: list[float]) -> list[float]:
    coords = sorted(float(value) for value in values)
    out: list[float] = []
    for value in coords:
        if not out or abs(value - out[-1]) > EPSILON:
            out.append(value)
    if out[0] > 0.0:
        out.insert(0, 0.0)
    if out[-1] < 1.0:
        out.append(1.0)
    out[0] = 0.0
    out[-1] = 1.0
    return out


def _last_cable_covering(
    rules: list[tuple[float, float, CableProperties]],
    *,
    x: float,
) -> CableProperties:
    selected = rules[0][2]
    for prox, dist, cable in rules:
        if prox - EPSILON <= x <= dist + EPSILON:
            selected = cable
    return selected


def _cable_signature(cable: CableProperties) -> tuple[float, float]:
    ra = float(np.asarray(cable.axial_resistivity.to_decimal(u.ohm * u.cm), dtype=float))
    cm = float(np.asarray(cable.membrane_capacitance.to_decimal(u.uF / u.cm ** 2), dtype=float))
    return (ra, cm)


def _same_cable_signature(lhs: tuple[float, float], rhs: tuple[float, float]) -> bool:
    return bool(
        np.isclose(lhs[0], rhs[0], atol=EPSILON, rtol=EPSILON)
        and np.isclose(lhs[1], rhs[1], atol=EPSILON, rtol=EPSILON)
    )
