"""CV ↔ point-space conversion helpers.

Thin, named wrappers around
:func:`braincell.compute._runtime.scatter_midpoint_values` /
:func:`braincell.compute._runtime.gather_midpoint_values` so the
surrounding pipeline reads naturally. The originals stay for other
call sites.
"""

from braincell.compute._runtime import (
    CellRuntimeState,
    gather_midpoint_values,
    scatter_midpoint_values,
)

__all__ = ["cv_to_point", "point_to_cv"]


def cv_to_point(values, runtime: CellRuntimeState):
    """Scatter a ``(..., n_cv)`` array onto CV midpoints in point space.

    Returned array has shape ``(..., n_point)`` with zeros at every
    non-midpoint point.
    """
    return scatter_midpoint_values(
        values=values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
        n_point=runtime.n_point,
    )


def point_to_cv(values, runtime: CellRuntimeState):
    """Gather a ``(..., n_point)`` array at CV midpoints.

    Returned array has shape ``(..., n_cv)``.
    """
    return gather_midpoint_values(
        values,
        point_ids=runtime.point_tree.cv_midpoint_point_id,
    )
