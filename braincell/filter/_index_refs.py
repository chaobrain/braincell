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


"""Which default branch indices a region or locset expression names by number.

An expression is evaluated again on every rebuild of a Cell's grid, so one
that finds its branches by name, type or geometry follows an edit of the
morphology. A leaf that holds an integer branch index does not: once a
branch is removed or moved (``Morphology.remove``, ``Morphology.reattach``)
the same integer names another branch. A Cell asks :func:`held_branch_indices`
at declaration time whether to watch the morphology's index watermark for
the declaration (MORPH-17).
"""

from collections.abc import Callable, Iterable
from dataclasses import fields, is_dataclass
import functools
from numbers import Integral, Real

import numpy as np

from braincell.filter.locset import AtLocation, LocsetExpr
from braincell.filter.region import BranchInFilter, BranchRangeFilter, BranchSlice, RegionExpr, SubtreeRegion

__all__ = ["held_branch_indices"]

#: Branch properties whose value is a default branch index. ``parent_id`` is
#: the parent's index, with ``-1`` at the root (see ``helper._resolve_branch_property``).
INDEX_PROPERTIES = frozenset({"index", "branch_id", "parent_id"})


def _numbers(value: object, *, nonnegative: bool = False) -> tuple[int | float, ...]:
    """The real numbers in ``value``: a scalar, a sequence or an array of them."""
    if type(value) is int:  # the common case, without the slower ABC checks below
        return () if nonnegative and value < 0 else (value,)
    if isinstance(value, np.ndarray):
        items: Iterable[object] = value.reshape(-1).tolist()
    elif isinstance(value, (tuple, list, range)):
        items = value
    else:
        items = (value,)
    found = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            continue
        number = int(item) if isinstance(item, Integral) or float(item).is_integer() else float(item)
        if not (nonnegative and number < 0):
            found.append(number)
    return tuple(found)


def _property_bounds(node: BranchRangeFilter) -> tuple[object, ...]:
    bounds = node.bounds
    return tuple(bounds) if isinstance(bounds, (tuple, list)) else ()


#: The leaves that can hold a default branch index, and how to read it. A
#: property predicate holds one only when it filters on an index-valued
#: property; ``branch_in("index", [1])`` is what ``cell.branch[1]`` declares.
#: ``_index_refs_test`` checks that every other leaf's fields are known not to.
INDEX_LEAVES: dict[type, Callable[[object], Iterable[int | float]]] = {
    BranchSlice: lambda node: _numbers(node.branch_index),
    SubtreeRegion: lambda node: _numbers(node.root_branch_index),
    AtLocation: lambda node: _numbers(node.branch),
    BranchInFilter: lambda node: _numbers(node.values, nonnegative=True) if node.property in INDEX_PROPERTIES else (),
    BranchRangeFilter: lambda node: (
        _numbers(_property_bounds(node), nonnegative=True) if node.property in INDEX_PROPERTIES else ()
    ),
}


#: Per expression class: its index reader (``None`` if it holds none) and its
#: field names, looked up once; the walk runs at every declaration.
_CLASS_INFO: dict[type, tuple[Callable[[object], Iterable[int | float]] | None, tuple[str, ...]]] = {}


def _class_info(cls: type) -> tuple[Callable[[object], Iterable[int | float]] | None, tuple[str, ...]]:
    info = _CLASS_INFO.get(cls)
    if info is None:
        reader = next((read for leaf, read in INDEX_LEAVES.items() if issubclass(cls, leaf)), None)
        names = tuple(item.name for item in fields(cls)) if is_dataclass(cls) else ()
        info = _CLASS_INFO[cls] = (reader, names)
    return info


def held_branch_indices(expression: object) -> tuple[int | float, ...]:
    """Return the default branch indices that ``expression`` names by number.

    Parameters
    ----------
    expression : RegionExpr or LocsetExpr
        A region or locset expression, possibly composite (set algebra,
        ``RegionAnchors``, samplers over a region). Anything else, such as
        a resolved ``RegionMask`` or ``LocsetMask``, gives ``()``.

    Returns
    -------
    tuple of int or float
        Every index held by a leaf -- ``BranchSlice(1, ...)``,
        ``SubtreeRegion(1)``, ``at(1, x)``, or a ``branch_in``/``branch_range``
        filter on ``index``, ``branch_id`` or ``parent_id`` (non-negative
        values only: ``-1`` is the root's ``parent_id``) -- in order of first
        appearance, without repeats. Empty when the expression finds its
        branches by name, type or geometry only.

    Notes
    -----
    Leaves are recognized by type, so a user subclass of ``RegionExpr`` or
    ``LocsetExpr`` that holds indices in fields of its own is not seen.

    Results are memoized by expression value: per-member placement asks
    once per member, often with equal expressions. An expression with an
    unhashable field (a list, an array) is walked every time.
    """
    if not isinstance(expression, (RegionExpr, LocsetExpr)):
        return ()
    try:
        return _held_by_value(expression)
    except TypeError:  # unhashable: BranchSlice([1, 2], ...), Quantity fields
        return _walk(expression)


@functools.lru_cache(maxsize=4096)
def _held_by_value(expression: RegionExpr | LocsetExpr) -> tuple[int | float, ...]:
    return _walk(expression)


def _walk(expression: RegionExpr | LocsetExpr) -> tuple[int | float, ...]:
    found: dict[int | float, None] = {}
    stack = [expression]
    while stack:
        node = stack.pop()
        read, names = _class_info(type(node))
        if read is not None:
            found.update(dict.fromkeys(read(node)))
        operands = []
        for name in names:
            value = getattr(node, name)
            if isinstance(value, (RegionExpr, LocsetExpr)):
                operands.append(value)
            elif isinstance(value, (tuple, list)):
                operands.extend(part for part in value if isinstance(part, (RegionExpr, LocsetExpr)))
        stack.extend(reversed(operands))
    return tuple(found)
