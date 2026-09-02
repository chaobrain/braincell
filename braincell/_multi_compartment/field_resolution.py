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

"""Resolve user-facing selectors against a ``Cell`` into spatial vectors.

Three different surfaces ask a ``Cell`` the same shape of question — given
a *selector*, which CVs or points does it name, and what value does each
one carry?

- :func:`braincell.vis.plot_cell_topology` colours a topology graph by a
  region, a locset, or a runtime field.
- ``Cell.on(...)`` narrows a selection scope to the CVs a region covers
  (see :mod:`braincell._multi_compartment.selection`).
- ``cell.runtime_cvs[i].ions[...]`` reads one runtime field through a
  local CV or node view.

The answer is the same computation in all three cases, so it lives here
as free functions taking ``cell`` rather than as methods on ``Cell``.
Keeping it out of the class is what lets :mod:`braincell.vis` import a
module instead of reaching for private attributes across a package
boundary.

There are two families:

**Coverage** — :func:`region_intervals`, :func:`cv_coverage_fractions`,
:func:`branch_coverage_fractions`, :func:`locset_cv_ids`,
:func:`node_highlight_fractions`, :func:`cv_highlight_fractions`. These
turn a region or locset into per-CV / per-point / per-branch overlap
fractions.

**Values** — :func:`resolve_node_field_values`,
:func:`resolve_cv_field_values` and the coercers beneath them. These turn
a value selector (a raw array, ``"V"``, or an ``("ion" | "channel" |
"layout_id", name, field)`` tuple) into a point-space or CV-space vector.

Every function that can raise a user-facing message takes a ``caller``
string so the message names the entry point the user actually invoked,
rather than whichever internal helper happened to detect the problem.
"""

import brainunit as u
import numpy as np

from braincell._discretization.base import locate_cv_on_branch
from braincell.filter import LocsetExpr, LocsetMask, RegionExpr, RegionMask
from braincell.filter.helper import normalize_region_intervals

__all__ = [
    "branch_coverage_fractions",
    "coerce_cv_values",
    "coerce_named_cv_values",
    "coerce_named_node_values",
    "coerce_node_values",
    "coerce_runtime_point_values",
    "cv_coverage_fractions",
    "cv_highlight_fractions",
    "cv_to_node_values",
    "cv_voltage",
    "layout_field_to_cv_values",
    "layout_field_to_point_values",
    "layout_values_to_cv_space",
    "layout_values_to_point_space",
    "locset_cv_ids",
    "mask_non_midpoint_points",
    "node_highlight_fractions",
    "region_intervals",
    "require_initialized",
    "resolve_cv_field_values",
    "resolve_node_field_values",
    "single_population_view",
    "split_unit",
    "unique_layout_by_kind",
]


def require_initialized(cell, action: str) -> None:
    """Raise unless ``cell`` has been initialized.

    Lets callers outside :mod:`braincell._multi_compartment` demand a
    runtime without reaching for ``Cell``'s private guard.

    Parameters
    ----------
    cell : braincell.Cell
        Cell to check.
    action : str
        Description of what needed the runtime, used in the message.

    Raises
    ------
    RuntimeError
        If ``cell.init_state()`` has not been called.
    """
    cell._raise_if_not_initialized(action)


# ----------------------------------------------------------------------
# Units


def split_unit(value):
    """Separate a possibly-united field into the parts a coercer needs.

    Every coercer below decides what a field means purely from its
    length, then either returns it untouched or maps it between point and
    CV space. Only two steps care about units: reading the length needs a
    bare mantissa, and synthesizing an array from a scalar needs the unit
    put back. Splitting those out here lets each coercer state its length
    rules once instead of once per storage flavour.

    Parameters
    ----------
    value : array-like or brainunit.Quantity
        A field taken from a runtime buffer or supplied by a caller.

    Returns
    -------
    mantissa : numpy.ndarray
        Unitless values, for shape and length tests.
    original : array-like or brainunit.Quantity
        ``value`` itself when it carried a unit, else ``mantissa``. This
        is what to return or hand to a spatial mapper, so the unit
        survives.
    rewrap : Callable[[numpy.ndarray], object]
        Puts the unit back on a freshly built array; identity when there
        was no unit.
    """
    if hasattr(value, "to_decimal") and hasattr(value, "unit"):
        unit = value.unit
        mantissa = np.asarray(value.to_decimal(unit), dtype=float)
        return mantissa, value, lambda array: u.Quantity(array, unit)
    mantissa = np.asarray(value, dtype=float)
    return mantissa, mantissa, lambda array: array


def single_population_view(cell, values, *, caller: str, field: str = "value"):
    """Reduce a ``pop_size + (n,)`` field to the single-member ``(n,)`` view.

    Inspection and visualization both answer questions about *one*
    morphology, so the leading population axes have to be singleton
    before the spatial mapping helpers can interpret the trailing axis.
    Collapsing them here keeps a default ``pop_size=1`` cell behaving
    exactly as a rank-0 population used to.

    Parameters
    ----------
    cell : braincell.Cell
        Cell the field was read from; supplies ``pop_size`` for the error
        message.
    values : array-like or brainunit.Quantity
        Field values shaped ``pop_size + (n,)``. Scalars and 1-D values
        pass through untouched.
    caller : str
        Entry point description, used only in the error message.
    field : str, default 'value'
        Field name, used only in the error message. The default suits the
        coercers, which are handed whatever the caller passed as
        ``value=``; named-field callers pass the real name.

    Returns
    -------
    array-like or brainunit.Quantity
        ``values`` with every leading singleton axis removed.

    Raises
    ------
    ValueError
        If any leading population axis holds more than one member, since
        there is then no single morphology to answer for.
    """
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        return values
    leading = shape[:-1]
    if any(int(dim) != 1 for dim in leading):
        raise ValueError(
            f"{caller} addresses a single morphology, but field {field!r} has "
            f"population shape {tuple(int(d) for d in leading)!r} from "
            f"pop_size={tuple(int(d) for d in cell.pop_size)!r}. Index the "
            f"population axis first."
        )
    return values[(0,) * len(leading)]


# ----------------------------------------------------------------------
# Coverage: regions and locsets


def region_intervals(cell, region, *, caller: str) -> dict[int, tuple[tuple[float, float], ...]]:
    """Normalize ``region`` into per-branch ``(prox, dist)`` intervals.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose morphology the region is evaluated against.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask
        Continuous morphology selection.
    caller : str
        Entry point description, used only in the error message.

    Returns
    -------
    dict
        Branch index mapped to a tuple of normalized ``(prox, dist)``
        interval pairs.

    Raises
    ------
    TypeError
        If ``region`` is neither a ``RegionExpr`` nor a ``RegionMask``.
    """
    if isinstance(region, RegionExpr):
        mask = region.evaluate(cell.morpho)
    elif isinstance(region, RegionMask):
        mask = region
    else:
        raise TypeError(f"{caller} expects RegionExpr or RegionMask, got {type(region).__name__!s}.")
    normalized = normalize_region_intervals(mask.intervals)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for branch_id, prox, dist in normalized:
        grouped.setdefault(int(branch_id), []).append((float(prox), float(dist)))
    return {branch_id: tuple(intervals) for branch_id, intervals in grouped.items()}


def cv_coverage_fractions(cell, region, *, caller: str) -> dict[int, float]:
    """Return each CV's fractional overlap with ``region``.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose CVs are measured.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask
        Continuous morphology selection.
    caller : str
        Entry point description, used only in the error message.

    Returns
    -------
    dict
        CV id mapped to a fraction in ``[0, 1]``.
    """
    branch_intervals = region_intervals(cell, region, caller=caller)
    fractions: dict[int, float] = {}
    for cv in cell.cvs:
        intervals = branch_intervals.get(int(cv.branch_id), ())
        total = max(float(cv.dist) - float(cv.prox), 1e-12)
        overlap = 0.0
        for left, right in intervals:
            start = max(float(cv.prox), float(left))
            end = min(float(cv.dist), float(right))
            if end - start <= 1e-9:
                continue
            overlap += end - start
        fractions[int(cv.id)] = float(np.clip(overlap / total, 0.0, 1.0))
    return fractions


def branch_coverage_fractions(cell, region, *, caller: str) -> dict[int, float]:
    """Return each branch's fractional coverage by ``region``.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose morphology branches are measured.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask
        Continuous morphology selection.
    caller : str
        Entry point description, used only in the error message.

    Returns
    -------
    dict
        Branch index mapped to a fraction in ``[0, 1]``.
    """
    branch_intervals = region_intervals(cell, region, caller=caller)
    fractions: dict[int, float] = {}
    for branch in cell.morpho.branches:
        intervals = branch_intervals.get(int(branch.index), ())
        covered = sum(max(0.0, float(right) - float(left)) for left, right in intervals)
        fractions[int(branch.index)] = float(np.clip(covered, 0.0, 1.0))
    return fractions


def locset_cv_ids(cell, locset, *, caller: str) -> set[int]:
    """Return the ids of the CVs that own each location in ``locset``.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose CVs are searched.
    locset : braincell.filter.LocsetExpr or braincell.filter.LocsetMask
        Discrete morphology locations.
    caller : str
        Entry point description, used only in the error message.

    Returns
    -------
    set of int
        Owning CV ids, deduplicated.

    Raises
    ------
    TypeError
        If ``locset`` is neither a ``LocsetExpr`` nor a ``LocsetMask``.
    """
    if isinstance(locset, LocsetExpr):
        mask = locset.evaluate(cell.morpho)
    elif isinstance(locset, LocsetMask):
        mask = locset
    else:
        raise TypeError(f"{caller} expects LocsetExpr or LocsetMask, got {type(locset).__name__!s}.")

    grouped: dict[int, list[int]] = {}
    for cv in cell.cvs:
        grouped.setdefault(int(cv.branch_id), []).append(int(cv.id))
    cv_ids_by_branch = {branch_id: tuple(ids) for branch_id, ids in grouped.items()}

    cv_ids: set[int] = set()
    for branch_id, x in mask.points:
        ids = cv_ids_by_branch.get(int(branch_id))
        if not ids:
            continue
        cv_id = locate_cv_on_branch(ids, cell.cvs, x=float(x))
        cv_ids.add(int(cv_id))
    return cv_ids


def node_highlight_fractions(cell, *, region, locset, caller: str) -> dict[int, float]:
    """Map ``region`` / ``locset`` onto CV-midpoint node highlight strengths.

    Both selections are reduced to CV granularity first, then attributed
    to each CV's midpoint node — the mapping the runtime lowering model
    uses. Locset-backed highlights are always full intensity.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose node tree is highlighted.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask or None
        Continuous selection contributing overlap fractions.
    locset : braincell.filter.LocsetExpr or braincell.filter.LocsetMask or None
        Discrete selection contributing full-intensity highlights.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    dict
        Point id mapped to a highlight fraction in ``[0, 1]``.
    """
    fractions: dict[int, float] = {}
    node_tree = cell.node_tree
    if region is not None:
        for cv_id, fraction in cv_coverage_fractions(cell, region, caller=caller).items():
            point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
            fractions[point_id] = max(fractions.get(point_id, 0.0), float(fraction))
    if locset is not None:
        for cv_id in locset_cv_ids(cell, locset, caller=caller):
            point_id = int(node_tree.cv_to_mid_node_id[int(cv_id)])
            fractions[point_id] = max(fractions.get(point_id, 0.0), 1.0)
    return fractions


def cv_highlight_fractions(cell, *, region, locset, caller: str) -> dict[int, float]:
    """Map ``region`` / ``locset`` onto per-CV highlight strengths.

    Parameters
    ----------
    cell : braincell.Cell
        Cell whose CVs are highlighted.
    region : braincell.filter.RegionExpr or braincell.filter.RegionMask or None
        Continuous selection contributing overlap fractions.
    locset : braincell.filter.LocsetExpr or braincell.filter.LocsetMask or None
        Discrete selection contributing full-intensity highlights.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    dict
        CV id mapped to a highlight fraction in ``[0, 1]``.
    """
    fractions: dict[int, float] = {}
    if region is not None:
        fractions.update(cv_coverage_fractions(cell, region, caller=caller))
    if locset is not None:
        for cv_id in locset_cv_ids(cell, locset, caller=caller):
            fractions[int(cv_id)] = max(fractions.get(int(cv_id), 0.0), 1.0)
    return fractions


# ----------------------------------------------------------------------
# Spatial mapping


def cv_to_node_values(cell, cv_values):
    """Scatter a ``(n_cv,)`` vector onto the CV midpoint nodes.

    Non-midpoint points receive ``NaN``, so a renderer draws only the
    points the value is actually defined at.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the node tree and the CV / point counts.
    cv_values : array-like or brainunit.Quantity
        One value per CV.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector, ``NaN`` away from CV midpoints.

    Raises
    ------
    ValueError
        If ``cv_values`` is not shaped ``(n_cv,)``.
    """
    node_tree = cell.node_tree
    mantissa, _, rewrap = split_unit(cv_values)
    raw = mantissa.reshape(-1)
    if raw.shape != (cell.n_cv,):
        raise ValueError(f"cv_to_node_values expects shape ({cell.n_cv},), got {raw.shape!r}.")
    point_values = np.full((cell.n_point,), np.nan, dtype=float)
    point_values[np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)] = raw
    return rewrap(point_values)


def mask_non_midpoint_points(cell, point_values):
    """Blank every point that is not a CV midpoint.

    A *named* state or parameter field is defined per CV, so only the
    midpoint carries it; the remaining points are set to ``NaN``.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the node tree and the point count.
    point_values : array-like or brainunit.Quantity
        One value per point.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector, ``NaN`` away from CV midpoints.

    Raises
    ------
    ValueError
        If ``point_values`` is not shaped ``(n_point,)``.
    """
    node_tree = cell.node_tree
    midpoint_ids = np.asarray(node_tree.cv_to_mid_node_id, dtype=np.int32)
    midpoint_mask = np.zeros((cell.n_point,), dtype=bool)
    midpoint_mask[midpoint_ids] = True
    mantissa, _, rewrap = split_unit(point_values)
    raw = mantissa.reshape(-1)
    if raw.shape != (cell.n_point,):
        raise ValueError(f"mask_non_midpoint_points expects shape ({cell.n_point},), got {raw.shape!r}.")
    masked = raw.copy()
    masked[~midpoint_mask] = np.nan
    return rewrap(masked)


# ----------------------------------------------------------------------
# Value coercion


def coerce_node_values(cell, value, *, caller: str):
    """Coerce a caller-supplied field into unmasked point-space values.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the CV-to-point bridge.
    value : array-like or brainunit.Quantity
        Scalar, point-space, or CV-space values.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector.

    Raises
    ------
    ValueError
        If ``value`` is neither scalar nor 1-D, or its length matches
        neither ``n_point`` nor ``n_cv``.
    """
    raw, original, rewrap = split_unit(single_population_view(cell, value, caller=caller))
    if raw.ndim == 0:
        return rewrap(np.full((cell.n_point,), float(raw), dtype=float))
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports scalar or 1-D value arrays.")
    if raw.shape[0] == cell.n_point:
        return original
    if raw.shape[0] == cell.n_cv:
        return cv_to_node_values(cell, original)
    raise ValueError(
        f"{caller} expects a point array of length {cell.n_point} "
        f"or a CV array of length {cell.n_cv}, got length {raw.shape[0]}."
    )


def coerce_runtime_point_values(cell, value):
    """Coerce one runtime field into unmasked point-space values.

    Unlike :func:`coerce_node_values`, a CV-space input is *broadcast*
    across every point of its CV rather than scattered to the midpoint —
    this backs ``cell.runtime_nodes[i].ions[...]``, where each point wants
    the value its CV holds.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the CV-to-point bridge.
    value : array-like or brainunit.Quantity
        Scalar, point-space, or CV-space values.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector.

    Raises
    ------
    ValueError
        If ``value`` is neither scalar nor 1-D, or its length matches
        neither ``n_point`` nor ``n_cv``.
    """
    caller = "Runtime point inspection"
    raw, original, rewrap = split_unit(single_population_view(cell, value, caller=caller))
    if raw.ndim == 0:
        return rewrap(np.full((cell.n_point,), float(raw), dtype=float))
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports scalar or 1-D value arrays.")
    if raw.shape[0] == cell.n_point:
        return original
    if raw.shape[0] == cell.n_cv:
        return cell._cv_to_point(original)
    raise ValueError(
        f"{caller} expects a point array of length {cell.n_point} "
        f"or a CV array of length {cell.n_cv}, got length {raw.shape[0]}."
    )


def coerce_cv_values(cell, value, *, caller: str):
    """Coerce a caller-supplied field into CV-space values.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the point-to-CV bridge.
    value : array-like or brainunit.Quantity
        Scalar, CV-space, or point-space values.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_cv,)`` vector.

    Raises
    ------
    ValueError
        If ``value`` is neither scalar nor 1-D, or its length matches
        neither ``n_cv`` nor ``n_point``.
    """
    raw, original, rewrap = split_unit(single_population_view(cell, value, caller=caller))
    if raw.ndim == 0:
        return rewrap(np.full((cell.n_cv,), float(raw), dtype=float))
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports scalar or 1-D value arrays.")
    if raw.shape[0] == cell.n_cv:
        return original
    if raw.shape[0] == cell.n_point:
        return cell._point_to_cv(original)
    raise ValueError(
        f"{caller} expects a CV array of length {cell.n_cv} "
        f"or a point array of length {cell.n_point}, got length {raw.shape[0]}."
    )


def coerce_named_node_values(cell, value, *, caller: str):
    """Coerce a *named* state/parameter field into point-space values.

    A named field already in point space is masked down to midpoints,
    which is the difference from :func:`coerce_node_values`: a named value
    is defined per CV, so only the midpoint carries it.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the node tree.
    value : array-like or brainunit.Quantity
        Scalar, point-space, or CV-space values.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector, ``NaN`` away from CV midpoints.

    Raises
    ------
    ValueError
        If ``value`` is not scalar or 1-D, or cannot be mapped into point
        space.
    """
    raw, original, rewrap = split_unit(single_population_view(cell, value, caller=caller))
    if raw.ndim == 0:
        return cv_to_node_values(cell, rewrap(np.full((cell.n_cv,), float(raw), dtype=float)))
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports scalar or 1-D named value arrays.")
    if raw.shape[0] == cell.n_point:
        return mask_non_midpoint_points(cell, original)
    if raw.shape[0] == cell.n_cv:
        return cv_to_node_values(cell, original)
    raise ValueError(f"{caller} cannot map the named value into point space.")


def coerce_named_cv_values(cell, value, *, caller: str):
    """Coerce a *named* state/parameter field into CV-space values.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the point-to-CV bridge.
    value : array-like or brainunit.Quantity
        Scalar, CV-space, or point-space values.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_cv,)`` vector.

    Raises
    ------
    ValueError
        If ``value`` is not scalar or 1-D, or cannot be mapped into CV
        space.
    """
    raw, original, rewrap = split_unit(single_population_view(cell, value, caller=caller))
    if raw.ndim == 0:
        return rewrap(np.full((cell.n_cv,), float(raw), dtype=float))
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports scalar or 1-D named value arrays.")
    if raw.shape[0] == cell.n_cv:
        return original
    if raw.shape[0] == cell.n_point:
        return cell._point_to_cv(mask_non_midpoint_points(cell, original))
    raise ValueError(f"{caller} cannot map the named value into CV space.")


# ----------------------------------------------------------------------
# Runtime layout lookup


def unique_layout_by_kind(cell, kind: str, *, caller: str):
    """Return the single runtime layout of ``kind``.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cell whose runtime layouts are searched.
    kind : str
        Layout kind, e.g. ``"channel:IL"``.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    object
        The matching runtime layout.

    Raises
    ------
    ValueError
        If no layout matches, or if more than one does — in which case the
        message lists the candidates and points at the ``layout_id``
        selector.
    """
    matches = [layout for layout in cell.layouts if layout.kind == kind]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(f"{caller} found no runtime layout with kind {kind!r}.")
    details = ", ".join(f"id={layout.id}:{layout.kind}" for layout in matches)
    raise ValueError(
        f"{caller} found multiple runtime layouts for {kind!r}: {details}. "
        "Use ('layout_id', id, field) to select one exact layout."
    )


def _layout_by_id(cell, layout_id: int):
    layout = next((candidate for candidate in cell.layouts if candidate.id == int(layout_id)), None)
    if layout is None:
        raise KeyError(f"Unknown layout id {layout_id!r}.")
    return layout


def _layout_raw_values(cell, layout, field: str):
    try:
        return cell.get_state(layout.id, field)
    except KeyError:
        node = cell.get_runtime_node(layout.id)
        if not hasattr(node, field):
            raise AttributeError(f"Runtime layout {layout.id!r} has no field {field!r}.")
        return getattr(node, field)


def layout_field_to_point_values(cell, layout_id: int, field: str, *, caller: str):
    """Read one field off a runtime layout and map it into point space.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cell owning the layout.
    layout_id : int
        Runtime layout id.
    field : str
        State or attribute name on the layout's runtime node.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector, ``NaN`` outside the layout's points.

    Raises
    ------
    KeyError
        If ``layout_id`` names no layout.
    AttributeError
        If the layout's runtime node has no such field.
    """
    layout = _layout_by_id(cell, layout_id)
    raw_values = _layout_raw_values(cell, layout, field)
    return layout_values_to_point_space(cell, layout, raw_values, field=field, caller=caller)


def layout_field_to_cv_values(cell, layout_id: int, field: str, *, caller: str):
    """Read one field off a runtime layout and map it into CV space.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cell owning the layout.
    layout_id : int
        Runtime layout id.
    field : str
        State or attribute name on the layout's runtime node.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_cv,)`` vector, ``NaN`` outside the layout's CVs.

    Raises
    ------
    KeyError
        If ``layout_id`` names no layout.
    AttributeError
        If the layout's runtime node has no such field.
    """
    layout = _layout_by_id(cell, layout_id)
    raw_values = _layout_raw_values(cell, layout, field)
    return layout_values_to_cv_space(cell, layout, raw_values, field=field, caller=caller)


def layout_values_to_point_space(cell, layout, raw_values, *, field: str, caller: str):
    """Scatter a layout-local field onto the cell's full point vector.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the point count.
    layout : object
        Runtime layout, read for ``id`` and ``point_index``.
    raw_values : array-like or brainunit.Quantity
        Scalar, point-space, or layout-local values.
    field : str
        Field name, used only in error messages.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector, ``NaN`` outside the layout's points.

    Raises
    ------
    ValueError
        If the layout has no ``point_index``, or the field is not 1-D, or
        its length matches neither ``n_point`` nor the layout's own point
        count.
    """
    n_point = cell.n_point
    raw_values = single_population_view(cell, raw_values, field=field, caller=caller)
    raw, _, rewrap = split_unit(raw_values)
    point_values = np.full((n_point,), np.nan, dtype=float)

    if raw.ndim == 0:
        if layout.point_index is None:
            raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
        point_values[np.asarray(layout.point_index, dtype=np.int32)] = float(raw)
        return rewrap(point_values)
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports 1-D value fields; {field!r} is not 1-D.")

    array = raw.reshape(-1)
    if array.shape[0] == n_point:
        if layout.point_index is None:
            raise ValueError(f"Layout {layout.id!r} has no point_index for field {field!r}.")
        index = np.asarray(layout.point_index, dtype=np.int32)
        point_values[index] = array[index]
        return rewrap(point_values)
    if layout.point_index is None or array.shape[0] != len(layout.point_index):
        raise ValueError(
            f"{caller} cannot map field {field!r} from layout {layout.id!r} "
            f"with shape {array.shape!r} into point space."
        )
    point_values[layout.point_index] = array
    return rewrap(point_values)


def layout_values_to_cv_space(cell, layout, raw_values, *, field: str, caller: str):
    """Gather a layout-local field onto the cell's full CV vector.

    Parameters
    ----------
    cell : braincell.Cell
        Cell supplying the CV / point counts and the node tree.
    layout : object
        Runtime layout, read for ``id``, ``point_index``, and
        ``source_cv_ids``.
    raw_values : array-like or brainunit.Quantity
        Scalar, CV-space, point-space, or layout-local values.
    field : str
        Field name, used only in error messages.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    numpy.ndarray or brainunit.Quantity
        A ``(n_cv,)`` vector, ``NaN`` outside the layout's CVs.

    Raises
    ------
    ValueError
        If the field is not 1-D, or its length matches none of ``n_cv``,
        ``n_point``, or the layout's own point count.
    """
    n_cv = cell.n_cv
    raw_values = single_population_view(cell, raw_values, field=field, caller=caller)
    source_cv_ids = tuple(int(cv_id) for cv_id in layout.source_cv_ids)
    midpoint_by_cv = {cv_id: int(cell.node_tree.cv_to_mid_node_id[cv_id]) for cv_id in source_cv_ids}
    raw, original, rewrap = split_unit(raw_values)
    cv_values = np.full((n_cv,), np.nan, dtype=float)

    if raw.ndim == 0:
        for cv_id in source_cv_ids:
            cv_values[cv_id] = float(raw)
        return rewrap(cv_values)
    if raw.ndim != 1:
        raise ValueError(f"{caller} only supports 1-D value fields; {field!r} is not 1-D.")

    array = raw.reshape(-1)
    if array.shape[0] == n_cv:
        return original
    if array.shape[0] == cell.n_point:
        for cv_id, point_id in midpoint_by_cv.items():
            cv_values[cv_id] = array[point_id]
        return rewrap(cv_values)
    if layout.point_index is None or array.shape[0] != len(layout.point_index):
        raise ValueError(
            f"{caller} cannot map field {field!r} from layout {layout.id!r} "
            f"with shape {array.shape!r} into CV space."
        )
    value_by_point = {
        int(point_id): float(array[index])
        for index, point_id in enumerate(np.asarray(layout.point_index, dtype=np.int32))
    }
    for cv_id, point_id in midpoint_by_cv.items():
        if point_id in value_by_point:
            cv_values[cv_id] = value_by_point[point_id]
    return rewrap(cv_values)


# ----------------------------------------------------------------------
# Value selectors


def cv_voltage(cell, *, caller: str):
    """Return ``V`` as a plain ``(n_cv,)`` CV vector.

    Parameters
    ----------
    cell : braincell.Cell
        Initialized cell.
    caller : str
        Entry point description, used only in the error message.

    Returns
    -------
    array-like or brainunit.Quantity
        Membrane voltage, one value per CV.
    """
    return single_population_view(cell, cell.V.value, field="V", caller=caller)


def resolve_node_field_values(cell, value, *, caller: str) -> tuple[object, str | None]:
    """Resolve a value selector into point-space values and a label.

    Parameters
    ----------
    cell : braincell.Cell
        Cell the selector is resolved against.
    value : object
        One of: a point-space array, a CV-space array, ``"V"`` /
        ``"voltage"``, ``("ion", ion_name, field)``,
        ``("channel", class_name, field)``, or
        ``("layout_id", layout_id, field)``.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    values : numpy.ndarray or brainunit.Quantity
        A ``(n_point,)`` vector.
    label : str or None
        Inferred colorbar label, or ``None`` for a raw array.

    Raises
    ------
    ValueError
        If a string or tuple selector is not recognised.
    AttributeError
        If a named ion has no such field.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"v", "voltage"}:
            return cv_to_node_values(cell, cv_voltage(cell, caller=caller)), "V"
        raise ValueError(f"Unsupported {caller} value string {value!r}.")

    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
        mode = value[0]
        if mode == "ion":
            ion_name, field = str(value[1]), str(value[2])
            ion = cell.get_ion(ion_name)
            if not hasattr(ion, field):
                raise AttributeError(f"Ion {ion_name!r} has no field {field!r}.")
            return coerce_named_node_values(cell, getattr(ion, field), caller=caller), f"{ion_name}.{field}"
        if mode == "channel":
            class_name, field = str(value[1]), str(value[2])
            layout = unique_layout_by_kind(cell, f"channel:{class_name}", caller=caller)
            return layout_field_to_point_values(cell, layout.id, field, caller=caller), f"{class_name}.{field}"
        if mode == "layout_id":
            layout_id, field = int(value[1]), str(value[2])
            return (
                layout_field_to_point_values(cell, layout_id, field, caller=caller),
                f"layout_{layout_id}.{field}",
            )
        raise ValueError(f"Unsupported {caller} value tuple selector {mode!r}.")

    return coerce_node_values(cell, value, caller=caller), None


def resolve_cv_field_values(cell, value, *, caller: str) -> tuple[object, str | None]:
    """Resolve a value selector into CV-space values and a label.

    Parameters
    ----------
    cell : braincell.Cell
        Cell the selector is resolved against.
    value : object
        One of: a CV-space array, a point-space array, ``"V"`` /
        ``"voltage"``, ``("ion", ion_name, field)``,
        ``("channel", class_name, field)``, or
        ``("layout_id", layout_id, field)``.
    caller : str
        Entry point description, used only in error messages.

    Returns
    -------
    values : numpy.ndarray or brainunit.Quantity
        A ``(n_cv,)`` vector.
    label : str or None
        Inferred colorbar label, or ``None`` for a raw array.

    Raises
    ------
    ValueError
        If a string or tuple selector is not recognised.
    AttributeError
        If a named ion has no such field.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"v", "voltage"}:
            return cv_voltage(cell, caller=caller), "V"
        raise ValueError(f"Unsupported {caller} value string {value!r}.")

    if isinstance(value, tuple) and len(value) == 3 and isinstance(value[0], str):
        mode = value[0]
        if mode == "ion":
            ion_name, field = str(value[1]), str(value[2])
            ion = cell.get_ion(ion_name)
            if not hasattr(ion, field):
                raise AttributeError(f"Ion {ion_name!r} has no field {field!r}.")
            return coerce_named_cv_values(cell, getattr(ion, field), caller=caller), f"{ion_name}.{field}"
        if mode == "channel":
            class_name, field = str(value[1]), str(value[2])
            layout = unique_layout_by_kind(cell, f"channel:{class_name}", caller=caller)
            return layout_field_to_cv_values(cell, layout.id, field, caller=caller), f"{class_name}.{field}"
        if mode == "layout_id":
            layout_id, field = int(value[1]), str(value[2])
            return (
                layout_field_to_cv_values(cell, layout_id, field, caller=caller),
                f"layout_{layout_id}.{field}",
            )
        raise ValueError(f"Unsupported {caller} value tuple selector {mode!r}.")

    return coerce_cv_values(cell, value, caller=caller), None
