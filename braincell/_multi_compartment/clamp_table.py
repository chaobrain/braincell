"""Pre-computed active-clamp-point table attached to :class:`CellRuntimeState`.

Building this once at compile time replaces the per-step filter walk
the legacy ``Cell._point_clamp_input`` used, and lets
``currents._clamp_density`` run without iterating over layouts.
"""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell.compute._point_tree import PointTree
from braincell.compute._runtime import MechanismLayout

__all__ = ["CLAMP_KINDS", "ClampActiveTable", "build_clamp_active_table"]

#: Clamp layout kinds that contribute point-space current via
#: :meth:`CellRuntimeState.evaluate_point_clamps`.
CLAMP_KINDS = frozenset({"CurrentClamp", "SineClamp", "FunctionClamp"})


@dataclass(frozen=True)
class ClampActiveTable:
    """Active clamp points and their membrane areas.

    Attributes
    ----------
    ids : np.ndarray
        ``(n_active,)`` ``int32`` sorted unique point ids that carry a
        clamp layout.
    area : np.ndarray
        ``(n_active,)`` ``float64`` membrane area in ``cm^2`` at those
        points.
    """

    ids: np.ndarray
    area: np.ndarray


def build_clamp_active_table(
    *,
    layouts: "tuple[MechanismLayout, ...]",
    cvs,
    point_tree: "PointTree",
    n_point: int,
) -> ClampActiveTable | None:
    """Return a :class:`ClampActiveTable` or ``None`` if no clamps placed.

    Parameters
    ----------
    layouts : tuple[MechanismLayout, ...]
        All mechanism layouts from :class:`CellRuntimeState`.
    cvs : Sequence[CV]
        The cell's control volumes — source of per-CV membrane area.
    point_tree : PointTree
        Carries ``cv_midpoint_point_id`` for CV→point mapping.
    n_point : int
        Number of points in ``point_tree``.

    Raises
    ------
    ValueError
        If any active clamp point has non-positive membrane area
        (would produce NaN in ``I_total / area`` division).
    """
    active: set[int] = set()
    for layout in layouts:
        if layout.target != "point" or layout.point_index is None:
            continue
        if layout.kind not in CLAMP_KINDS:
            continue
        active.update(int(pid) for pid in layout.point_index.tolist())

    if not active:
        return None

    ids = np.asarray(sorted(active), dtype=np.int32)

    point_area = np.zeros((n_point,), dtype=float)
    for cv in cvs:
        pid = int(point_tree.cv_midpoint_point_id[cv.id])
        point_area[pid] = float(np.asarray(cv.area.to_decimal(u.cm ** 2), dtype=float))

    area = point_area[ids]
    if np.any(area <= 0.0):
        bad = ids[area <= 0.0].tolist()
        raise ValueError(
            "Point clamp active points must have positive membrane area; "
            f"got non-positive area at point ids {bad!r}."
        )
    return ClampActiveTable(ids=ids, area=area.astype(np.float64, copy=False))
