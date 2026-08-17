"""Declaration-time construction of population-specific synapse pools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import brainunit as u

from braincell.filter import AllRegion, RegionExpr, at
from braincell.filter.helper import normalize_region_intervals
@dataclass(frozen=True)
class SynapsePoolContext:
    """Context supplied to per-post synapse-count rules."""

    post_size: int
    active_post_index: np.ndarray
    indegree: np.ndarray
    edge_pre_index: np.ndarray
    edge_post_index: np.ndarray
    morphology: object
    cv_contexts: tuple


@dataclass(frozen=True)
class RandomLocations:
    """Sample continuous cable locations uniformly by cable length."""

    region: RegionExpr
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region, RegionExpr):
            raise TypeError("random_locations region must be a RegionExpr.")

    def sample(self, morphology, *, post_index: np.ndarray, counts: np.ndarray):
        intervals = normalize_region_intervals(self.region.evaluate(morphology).intervals)
        entries = []
        for branch_id, prox, dist in intervals:
            branch = morphology.branch(index=int(branch_id))
            length_um = float(np.asarray(branch.length.to_decimal(u.um))) * (float(dist) - float(prox))
            if length_um > 0.0:
                entries.append((int(branch_id), float(prox), float(dist), length_um))
        if not entries and int(np.sum(counts)) > 0:
            raise ValueError("Synapse placement region contains no positive-length cable.")
        lengths = np.asarray([entry[3] for entry in entries], dtype=float)
        probabilities = lengths / np.sum(lengths) if lengths.size else lengths
        rng = np.random.default_rng(self.seed)
        rows = []
        for post, count in zip(post_index.tolist(), counts.tolist()):
            if int(count) <= 0:
                continue
            selected = rng.choice(len(entries), size=int(count), p=probabilities)
            fractions = rng.random(int(count))
            for entry_id, fraction in zip(selected.tolist(), fractions.tolist()):
                branch_id, prox, dist, _ = entries[int(entry_id)]
                rows.append((int(post), at(branch_id, prox + fraction * (dist - prox))))
        return tuple(rows)


@dataclass(frozen=True)
class SynapsePool:
    """Rule for creating a packed synapse-instance pool before initialization.

    ``number`` may be a scalar, a full ``(post_size,)`` integer array, an
    active-post integer array, or a callable receiving
    :class:`SynapsePoolContext`. ``placement`` supplies the continuous
    locations and defaults to length-weighted uniform cable sampling.
    """

    number: object = 1
    placement: RandomLocations | None = None

    def __post_init__(self) -> None:
        if not callable(getattr(self.placement, "sample", None)):
            raise TypeError("SynapsePool placement must provide sample(morphology, post_index, counts).")

    def counts(self, context: SynapsePoolContext) -> np.ndarray:
        value = self.number(context) if callable(self.number) else self.number
        array = np.asarray(value)
        if array.shape == ():
            array = np.full(context.active_post_index.shape, array, dtype=np.int32)
        elif array.shape == (context.post_size,):
            array = array[context.active_post_index]
        elif array.shape != context.active_post_index.shape:
            raise ValueError(
                "synapse_pool number must be scalar, shape (post_size,), or shape "
                f"{context.active_post_index.shape!r}; got {array.shape!r}."
            )
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("synapse_pool number must contain integers.")
        array = array.astype(np.int32, copy=False)
        if np.any(array < 0):
            raise ValueError("synapse_pool number must be >= 0.")
        return array


@dataclass(frozen=True)
class SynapseInstanceTable:
    """Materialized packed synapse instances for one automatic pool."""

    placement_index: np.ndarray
    synapse_index: np.ndarray
    post_index: np.ndarray
    branch_id: np.ndarray
    branch_x: np.ndarray
    cv_id: np.ndarray
    point_id: np.ndarray
    synapse: str

    @property
    def n_instance(self) -> int:
        return int(self.synapse_index.shape[0])

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "synapse_index": int(self.synapse_index[i]),
                "placement_id": int(self.placement_index[i]),
                "post_index": int(self.post_index[i]),
                "branch_id": int(self.branch_id[i]),
                "branch_x": float(self.branch_x[i]),
                "cv_id": int(self.cv_id[i]),
                "point_id": int(self.point_id[i]),
                "synapse": self.synapse,
            }
            for i in range(self.n_instance)
        ]


def synapse_pool(*, number=1, placement=None) -> SynapsePool:
    if placement is None:
        placement = random_locations()
    return SynapsePool(number=number, placement=placement)


def random_locations(*, region=None, seed=None) -> RandomLocations:
    return RandomLocations(region=AllRegion() if region is None else region, seed=seed)
