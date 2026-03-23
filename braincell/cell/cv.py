from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from .._units import u
from ..filter import RegionMask
from ..mech import DensityMechanism
from ..morpho import Branch
from .cv_geo import CVFrustum, CVGeo, axial_resistance_from_factor

if TYPE_CHECKING:
    from .cv_mech import CVMech


@dataclass(frozen=True)
class CVPolicy:
    mode: str = "cv_per_branch"
    cv_per_branch: int = 1
    max_cv_len: Any | None = None


@dataclass(frozen=True)
class CV:
    id: int
    branch_id: int
    branch_type: str
    prox: float
    dist: float
    parent_cv: int | None
    children_cv: tuple[int, ...]
    length: Any
    lateral_area: Any
    cm: Any
    ra: Any
    v: Any
    temp: u.Quantity[u.kelvin]
    r_axial: Any
    r_axial_prox: Any
    r_axial_dist: Any
    density_mech: tuple[DensityMechanism, ...]
    point_mech: tuple[object, ...]
    _frusta: tuple[CVFrustum, ...]

    @property
    def region(self) -> RegionMask:
        return RegionMask(((self.branch_id, self.prox, self.dist),))

    def as_branch(self) -> Branch:
        if len(self._frusta) == 0:
            raise ValueError("Cannot convert empty CV geometry into Branch.")

        lengths_um = np.asarray(
            [float(np.asarray(piece.length.to_decimal(u.um), dtype=float)) for piece in self._frusta],
            dtype=float,
        )
        r0_um = np.asarray(
            [float(np.asarray(piece.radius_prox.to_decimal(u.um), dtype=float)) for piece in self._frusta],
            dtype=float,
        )
        r1_um = np.asarray(
            [float(np.asarray(piece.radius_dist.to_decimal(u.um), dtype=float)) for piece in self._frusta],
            dtype=float,
        )

        has_points = all(
            piece.point_prox is not None and piece.point_dist is not None for piece in self._frusta
        )
        if has_points:
            first = self._frusta[0].point_prox
            if first is None:
                raise ValueError("CV frusta are missing proximal point geometry.")
            points: list[np.ndarray] = [np.asarray(first.to_decimal(u.um), dtype=float)]
            for piece in self._frusta:
                point_dist = piece.point_dist
                if point_dist is None:
                    raise ValueError("CV frusta are missing distal point geometry.")
                points.append(np.asarray(point_dist.to_decimal(u.um), dtype=float))
            radii_um = np.concatenate((r0_um[:1], r1_um), axis=0)
            return Branch.xyz_shared(
                points=u.Quantity(np.asarray(points, dtype=float), u.um),
                radii=u.Quantity(np.asarray(radii_um, dtype=float), u.um),
                type=self.branch_type,
            )

        pairs = np.stack((r0_um, r1_um), axis=1)
        return Branch.lengths_paired(
            lengths=u.Quantity(lengths_um, u.um),
            radii_pairs=u.Quantity(pairs, u.um),
            type=self.branch_type,
        )


def assemble_cv(*, cv_geo: CVGeo, mech: CVMech) -> CV:
    return CV(
        id=cv_geo.id,
        branch_id=cv_geo.branch_id,
        branch_type=cv_geo.branch_type,
        prox=cv_geo.prox,
        dist=cv_geo.dist,
        parent_cv=cv_geo.parent_cv,
        children_cv=cv_geo.children_cv,
        length=cv_geo.length,
        lateral_area=cv_geo.lateral_area,
        cm=mech.cm,
        ra=mech.ra,
        v=mech.v,
        temp=mech.temp,
        r_axial=axial_resistance_from_factor(mech.ra, factor=cv_geo.axial_factor_total),
        r_axial_prox=axial_resistance_from_factor(mech.ra, factor=cv_geo.axial_factor_prox),
        r_axial_dist=axial_resistance_from_factor(mech.ra, factor=cv_geo.axial_factor_dist),
        density_mech=tuple(mech.density_mech),
        point_mech=tuple(mech.point_mech),
        _frusta=cv_geo.frusta,
    )
