"""Immutable geometry primitives for the layered architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from braincell._units import (
    mantissa,
    normalize_param,
    segment_lengths_from_points,
    u,
)

_ALLOWED_BRANCH_TYPES = {
    "apical_dend",
    "apical_dendrite",
    "axon",
    "basal_dend",
    "basal_dendrite",
    "custom",
    "dend",
    "soma",
}


def _normalize_segment_vector(
    value: object,
    *,
    name: str,
    unit: Any,
    bounds: dict[str, object] | None = None,
) -> Any:
    """Normalize a 1D segment parameter and allow scalar sugar for one segment."""

    quantity = normalize_param(value, name=name, unit=unit, bounds=bounds)
    if quantity.ndim == 0:
        return quantity.reshape(1)
    if quantity.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {quantity.shape!r}.")
    return quantity


@dataclass(frozen=True)
class Branch:
    """A named anatomical branch with segment-wise cable geometry.

    All geometric quantities are stored as `brainunit` values and normalized to
    `u.um` internally. The constructor accepts raw lists, NumPy arrays, JAX
    arrays, or existing `brainunit` quantities; bare numeric inputs are assumed
    to be in micrometers.
    """

    lengths: u.Quantity[u.um]
    radii_prox: u.Quantity[u.um]
    radii_dist: u.Quantity[u.um]
    proximal_points: u.Quantity[u.um] | None = None
    distal_points: u.Quantity[u.um] | None = None
    name: str | None = None
    type: str = "custom"

    def __post_init__(self) -> None:
        lengths = _normalize_segment_vector(
            self.lengths,
            name="lengths",
            unit=u.um,
            bounds={"gt": 0},
        )
        radii_prox = _normalize_segment_vector(
            self.radii_prox,
            name="radii_prox",
            unit=u.um,
            bounds={"ge": 0},
        )
        radii_dist = _normalize_segment_vector(
            self.radii_dist,
            name="radii_dist",
            unit=u.um,
            bounds={"ge": 0},
        )
        proximal_points = normalize_param(
            self.proximal_points,
            name="proximal_points",
            unit=u.um,
            shape=(None, 3),
            allow_none=True,
        )
        distal_points = normalize_param(
            self.distal_points,
            name="distal_points",
            unit=u.um,
            shape=(None, 3),
            allow_none=True,
        )

        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "radii_prox", radii_prox)
        object.__setattr__(self, "radii_dist", radii_dist)
        object.__setattr__(self, "proximal_points", proximal_points)
        object.__setattr__(self, "distal_points", distal_points)

        if self.type not in _ALLOWED_BRANCH_TYPES:
            raise ValueError(
                f"type must be one of {sorted(_ALLOWED_BRANCH_TYPES)!r}, got {self.type!r}."
            )

        n_segments = len(lengths)
        if n_segments == 0:
            raise ValueError("Branch must contain at least one segment.")
        if len(radii_prox) != n_segments:
            raise ValueError("radii_prox must match lengths segment count.")
        if len(radii_dist) != n_segments:
            raise ValueError("radii_dist must match lengths segment count.")
        if proximal_points is not None and len(proximal_points) != n_segments:
            raise ValueError("proximal_points must match lengths segment count.")
        if distal_points is not None and len(distal_points) != n_segments:
            raise ValueError("distal_points must match lengths segment count.")

    @classmethod
    def lengths_shared(
        cls,
        lengths: u.Quantity[u.um],
        radii: u.Quantity[u.um],
        *,
        name: str | None = None,
        type: str = "custom",
    ) -> "Branch":
        """Build from segment lengths plus a continuous radius sequence."""

        lengths_q = _normalize_segment_vector(
            lengths,
            name="lengths",
            unit=u.um,
            bounds={"gt": 0},
        )
        radii_q = normalize_param(
            radii,
            name="radii",
            unit=u.um,
            shape=(None,),
            bounds={"ge": 0},
        )
        if len(radii_q) != len(lengths_q) + 1:
            raise ValueError("lengths_shared requires one more radius than segment length.")
        return cls(
            lengths=lengths_q,
            radii_prox=radii_q[:-1],
            radii_dist=radii_q[1:],
            name=name,
            type=type,
        )

    @classmethod
    def lengths_paired(
        cls,
        lengths: u.Quantity[u.um],
        radii_pairs: u.Quantity[u.um],
        *,
        name: str | None = None,
        type: str = "custom",
    ) -> "Branch":
        """Build from segment lengths plus per-segment `(r_prox, r_dist)` pairs."""

        lengths_q = _normalize_segment_vector(
            lengths,
            name="lengths",
            unit=u.um,
            bounds={"gt": 0},
        )
        radii_pairs_q = normalize_param(
            radii_pairs,
            name="radii_pairs",
            unit=u.um,
            shape=(None, 2),
            bounds={"ge": 0},
        )
        if len(radii_pairs_q) != len(lengths_q):
            raise ValueError("radii_pairs must match lengths segment count.")
        return cls(
            lengths=lengths_q,
            radii_prox=radii_pairs_q[:, 0],
            radii_dist=radii_pairs_q[:, 1],
            name=name,
            type=type,
        )

    @classmethod
    def xyz_shared(
        cls,
        points: u.Quantity[u.um],
        radii: u.Quantity[u.um],
        *,
        name: str | None = None,
        type: str = "custom",
    ) -> "Branch":
        """Build from 3D points plus a continuous radius sequence."""

        points_q = normalize_param(
            points,
            name="points",
            unit=u.um,
            shape=(None, 3),
        )
        radii_q = normalize_param(
            radii,
            name="radii",
            unit=u.um,
            shape=(None,),
            bounds={"ge": 0},
        )
        if len(points_q) < 2:
            raise ValueError("xyz_shared requires at least two points.")
        if len(radii_q) != len(points_q):
            raise ValueError("radii must contain one value per point.")
        return cls(
            lengths=segment_lengths_from_points(points_q),
            radii_prox=radii_q[:-1],
            radii_dist=radii_q[1:],
            proximal_points=points_q[:-1],
            distal_points=points_q[1:],
            name=name,
            type=type,
        )

    @classmethod
    def xyz_paired(
        cls,
        points: u.Quantity[u.um],
        radii_pairs: u.Quantity[u.um],
        *,
        name: str | None = None,
        type: str = "custom",
    ) -> "Branch":
        """Build from 3D points plus per-segment `(r_prox, r_dist)` pairs."""

        points_q = normalize_param(
            points,
            name="points",
            unit=u.um,
            shape=(None, 3),
        )
        radii_pairs_q = normalize_param(
            radii_pairs,
            name="radii_pairs",
            unit=u.um,
            shape=(None, 2),
            bounds={"ge": 0},
        )
        if len(points_q) < 2:
            raise ValueError("xyz_paired requires at least two points.")
        if len(radii_pairs_q) != len(points_q) - 1:
            raise ValueError("radii_pairs must match the number of point-to-point segments.")
        return cls(
            lengths=segment_lengths_from_points(points_q),
            radii_prox=radii_pairs_q[:, 0],
            radii_dist=radii_pairs_q[:, 1],
            proximal_points=points_q[:-1],
            distal_points=points_q[1:],
            name=name,
            type=type,
        )

    @property
    def n_segments(self) -> int:
        return len(self.lengths)

    @property
    def total_length(self) -> u.Quantity[u.um]:
        return u.math.sum(self.lengths)

    @property
    def radius_proximal(self) -> u.Quantity[u.um]:
        return self.radii_prox[0]

    @property
    def radius_distal(self) -> u.Quantity[u.um]:
        return self.radii_dist[-1]

    def lateral_areas(self) -> tuple[object, ...]:
        lengths_um = np.asarray(self.lengths.to_decimal(u.um), dtype=float)
        radii_prox_um = np.asarray(self.radii_prox.to_decimal(u.um), dtype=float)
        radii_dist_um = np.asarray(self.radii_dist.to_decimal(u.um), dtype=float)
        values = []
        for length_um, r0_um, r1_um in zip(lengths_um, radii_prox_um, radii_dist_um):
            slant_um = float(np.sqrt(length_um * length_um + (r1_um - r0_um) ** 2))
            values.append(u.Quantity(float(np.pi * (r0_um + r1_um) * slant_um), u.um ** 2))
        return tuple(values)

    def total_lateral_area(self):
        return sum(self.lateral_areas())

    def volumes(self) -> tuple[object, ...]:
        lengths_um = np.asarray(self.lengths.to_decimal(u.um), dtype=float)
        radii_prox_um = np.asarray(self.radii_prox.to_decimal(u.um), dtype=float)
        radii_dist_um = np.asarray(self.radii_dist.to_decimal(u.um), dtype=float)
        values = []
        for length_um, r0_um, r1_um in zip(lengths_um, radii_prox_um, radii_dist_um):
            volume_um3 = float(np.pi * length_um * (r0_um * r0_um + r0_um * r1_um + r1_um * r1_um) / 3.0)
            values.append(u.Quantity(volume_um3, u.um ** 3))
        return tuple(values)
