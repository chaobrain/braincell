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

"""Parse and validate the morphology of one NeuroML2 cell (private).

:func:`parse_document` reads a NeuroML2 document with the standard library's
``xml.etree.ElementTree`` and returns a :class:`NeuroMlDocument`: the validated
segment table of one ``<cell>`` and the members of its segment groups. It does
not build a :class:`~braincell.Morphology`; the reader does.

Every malformed input raises ``ValueError("NeuroML2 import failed for <path>:
<reason>")``, with the element and id in the reason. Documents that declare a
DTD or entities are refused before parsing, so entity expansion cannot run.
"""

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from braincell._typing import FilePath

__all__ = [
    "CABLE_NEUROLEX_ID",
    "NEUROML2_NAMESPACE",
    "NeuroMlDocument",
    "Segment",
    "TYPE_GROUP_IDS",
    "TYPE_NEUROLEX_IDS",
    "parse_document",
]

NEUROML2_NAMESPACE = "http://www.neuroml.org/schema/neuroml2"

TYPE_GROUP_IDS = {"soma": "soma_group", "axon": "axon_group", "dendrite": "dendrite_group"}
"""Branch type -> the segment group id NeuroML2 reserves for it."""

TYPE_NEUROLEX_IDS = {
    "soma": ("GO:0043025", "sao1044911821"),
    "axon": ("GO:0030424",),
    "dendrite": ("GO:0030425", "sao1211023249"),
}
"""Branch type -> the ``neuroLexId`` values that mark a group of that type."""

CABLE_NEUROLEX_ID = "sao864921383"
"""``neuroLexId`` of an unbranched cable group, which names one branch."""

_CELL_ELEMENTS = ("cell", "cell2CaPools")
_GROUP_CONTENT = ("member", "include", "path", "subTree")
_SEGMENT_CONTENT = ("parent", "proximal", "distal")
_NON_NEGATIVE_INTEGER = re.compile(r"[0-9]+")

Point = tuple[float, float, float, float]
"""``(x, y, z, diameter)`` in um."""


@dataclass(frozen=True)
class Segment:
    """One validated ``<segment>``.

    ``proximal`` is the actual proximal point: the ``<proximal>`` element when
    given, otherwise the point at ``fraction_along`` on the parent segment
    (as libNeuroML's ``get_actual_proximal``).
    """

    id: int
    name: str | None
    parent: int | None
    fraction_along: float
    proximal: Point
    distal: Point

    @property
    def is_sphere(self) -> bool:
        """Whether the proximal and distal points coincide (a spherical segment)."""
        return self.proximal[:3] == self.distal[:3]


@dataclass(frozen=True)
class NeuroMlDocument:
    """The validated morphology of one NeuroML2 cell.

    Attributes
    ----------
    path : Path
    cell_id, morphology_id : str or None
    segments : Mapping[int, Segment]
        Every segment by id, in document order.
    root : int
        Id of the only segment without a parent.
    segment_groups : Mapping[str, tuple[int, ...]]
        Each group's segments after expanding ``member``, ``include``, ``path``
        and ``subTree``, without duplicates, in ascending order.
    segment_types : Mapping[int, str]
        ``"soma"``, ``"axon"``, ``"dendrite"`` from the type groups, or
        ``"custom"`` for a segment in none of them.
    cable_groups : tuple[str, ...]
        Ids of the unbranched cable groups, in document order.
    segment_cables : Mapping[int, str]
        The cable group of each segment that belongs to one.
    ignored_elements : tuple[str, ...]
        Names of the cell's elements that carry no morphology (such as
        ``biophysicalProperties`` or ``notes``), sorted.
    """

    path: Path
    cell_id: str | None
    morphology_id: str | None
    segments: Mapping[int, Segment]
    root: int
    segment_groups: Mapping[str, tuple[int, ...]]
    segment_types: Mapping[int, str]
    cable_groups: tuple[str, ...]
    segment_cables: Mapping[int, str]
    ignored_elements: tuple[str, ...]

    def children(self) -> dict[int, list[int]]:
        """Return each segment's child segment ids, in document order."""
        children: dict[int, list[int]] = {segment_id: [] for segment_id in self.segments}
        for segment in self.segments.values():
            if segment.parent is not None:
                children[segment.parent].append(segment.id)
        return children


class _Failure(Exception):
    """Internal: a reason, turned into the public ``ValueError`` with the path."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _one_or_none(element: ET.Element, name: str, what: str) -> ET.Element | None:
    found = _children(element, name)
    if len(found) > 1:
        raise _Failure(f"{what} has {len(found)} <{name}> elements; at most one is allowed")
    return found[0] if found else None


def _segment_reference(raw: str | None, what: str) -> int:
    if raw is None or not _NON_NEGATIVE_INTEGER.fullmatch(raw):
        raise _Failure(f"{what} {raw!r} is not a non-negative integer")
    return int(raw)


def _number(element: ET.Element, attribute: str, what: str) -> float:
    raw = element.get(attribute)
    if raw is None:
        raise _Failure(f"{what}: missing attribute {attribute!r}")
    try:
        value = float(raw)
    except ValueError:
        raise _Failure(f"{what}: {attribute}={raw!r} is not a number") from None
    if not math.isfinite(value):
        raise _Failure(f"{what}: {attribute}={raw!r} is not finite")
    return value


def _point(element: ET.Element, what: str) -> Point:
    x, y, z, diameter = (_number(element, attribute, what) for attribute in ("x", "y", "z", "diameter"))
    if diameter <= 0.0:
        raise _Failure(f"{what}: diameter={element.get('diameter')!r} must be > 0")
    return (x, y, z, diameter)


def _load_root(path: Path) -> ET.Element:
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise _Failure("the document declares a DTD or entities (<!DOCTYPE or <!ENTITY), which are not supported")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise _Failure(f"not well-formed XML: {exc}") from None
    if _local(root.tag) != "neuroml" or _namespace(root.tag) not in (NEUROML2_NAMESPACE, ""):
        raise _Failure(
            f"the root element is <{_local(root.tag)}> (namespace {_namespace(root.tag)!r}), "
            f"expected <neuroml> in namespace {NEUROML2_NAMESPACE}"
        )
    return root


def _select_cell(root: ET.Element, cell_id: str | None) -> ET.Element:
    cells = [child for child in root if _local(child.tag) in _CELL_ELEMENTS]
    if not cells:
        includes = [include.get("href") for include in _children(root, "include")]
        detail = f"; includes are not followed: {', '.join(map(str, includes))}" if includes else ""
        raise _Failure(f"the document has no <cell>{detail}")
    ids = [cell.get("id") for cell in cells]
    if cell_id is None:
        if len(cells) > 1:
            listed = ", ".join(repr(cid) for cid in ids)
            raise _Failure(f"the document has {len(cells)} cells ({listed}); pass cell_id= to choose one")
        return cells[0]
    for cell in cells:
        if cell.get("id") == cell_id:
            return cell
    raise _Failure(f"no cell {cell_id!r}; the document has {', '.join(repr(cid) for cid in ids)}")


def _select_morphology(root: ET.Element, cell: ET.Element) -> ET.Element:
    what = f"cell {cell.get('id')!r}"
    morphology = _one_or_none(cell, "morphology", what)
    if morphology is not None:
        return morphology
    reference = cell.get("morphology")
    if reference is None:
        raise _Failure(f"{what} has no <morphology> and no morphology= attribute")
    for candidate in _children(root, "morphology"):
        if candidate.get("id") == reference:
            return candidate
    raise _Failure(f"{what} refers to morphology {reference!r}, which the document does not define")


def _read_segments(morphology: ET.Element) -> dict[int, dict]:
    raw_segments: dict[int, dict] = {}
    for element in _children(morphology, "segment"):
        segment_id = _segment_reference(element.get("id"), "segment id")
        if segment_id in raw_segments:
            raise _Failure(f"duplicate segment id {segment_id}")
        what = f"segment {segment_id}"
        parent_element = _one_or_none(element, "parent", what)
        parent, fraction = None, 1.0
        if parent_element is not None:
            parent = _segment_reference(parent_element.get("segment"), f"{what}: parent segment")
            raw_fraction = parent_element.get("fractionAlong", "1")
            try:
                fraction = float(raw_fraction)
            except ValueError:
                raise _Failure(f"{what}: fractionAlong={raw_fraction!r} is not a number") from None
            if not 0.0 <= fraction <= 1.0:
                raise _Failure(f"{what}: fractionAlong={raw_fraction} is outside [0, 1]")
        proximal = _one_or_none(element, "proximal", what)
        distal = _one_or_none(element, "distal", what)
        if distal is None:
            raise _Failure(f"{what}: missing <distal>")
        raw_segments[segment_id] = {
            "name": element.get("name"),
            "parent": parent,
            "fraction": fraction,
            "proximal": None if proximal is None else _point(proximal, f"{what} <proximal>"),
            "distal": _point(distal, f"{what} <distal>"),
        }
    if not raw_segments:
        raise _Failure(f"morphology {morphology.get('id')!r} has no segments")
    return raw_segments


def _check_tree(raw_segments: Mapping[int, dict]) -> int:
    for segment_id, raw in raw_segments.items():
        parent = raw["parent"]
        if parent is not None and parent not in raw_segments:
            raise _Failure(f"segment {segment_id}: parent segment {parent} does not exist")
    reaches_root: set[int] = set()
    for start in raw_segments:
        walk: list[int] = []
        current: int | None = start
        while current is not None and current not in reaches_root:
            if current in walk:
                cycle = sorted(walk[walk.index(current) :])
                raise _Failure(f"parent cycle through segments {cycle}")
            walk.append(current)
            current = raw_segments[current]["parent"]
        reaches_root.update(walk)
    roots = [segment_id for segment_id, raw in raw_segments.items() if raw["parent"] is None]
    if len(roots) != 1:
        raise _Failure(f"{len(roots)} root segments {roots}; a morphology has exactly one")
    return roots[0]


def _resolve_points(raw_segments: Mapping[int, dict], root: int) -> dict[int, Segment]:
    proximal: dict[int, Point] = {}

    def actual_proximal(segment_id: int) -> Point:
        # Iterative: walk up to the nearest known point, then fill in downwards.
        chain = []
        current = segment_id
        while current not in proximal:
            raw = raw_segments[current]
            if raw["proximal"] is not None:
                proximal[current] = raw["proximal"]
                break
            if raw["parent"] is None:
                raise _Failure(f"root segment {current} has no <proximal>")
            chain.append(current)
            current = raw["parent"]
        for child in reversed(chain):
            raw = raw_segments[child]
            parent = raw_segments[raw["parent"]]
            fraction = raw["fraction"]
            if fraction == 1.0:
                point = parent["distal"]
            elif fraction == 0.0:
                point = proximal[raw["parent"]]
            else:
                start = proximal[raw["parent"]]
                point = tuple((1.0 - fraction) * a + fraction * b for a, b in zip(start, parent["distal"], strict=True))
            proximal[child] = point
        return proximal[segment_id]

    actual_proximal(root)
    segments = {}
    for segment_id, raw in raw_segments.items():
        segment = Segment(
            id=segment_id,
            name=raw["name"],
            parent=raw["parent"],
            fraction_along=raw["fraction"],
            proximal=actual_proximal(segment_id),
            distal=raw["distal"],
        )
        if segment.is_sphere and segment.proximal[3] != segment.distal[3]:
            raise _Failure(
                f"segment {segment_id}: proximal and distal points coincide (a spherical segment) but their "
                f"diameters differ ({segment.proximal[3]!r} and {segment.distal[3]!r})"
            )
        segments[segment_id] = segment
    return segments


def _expand_groups(morphology: ET.Element, segments: Mapping[int, Segment]) -> dict[str, tuple[int, ...]]:
    elements: dict[str, ET.Element] = {}
    for element in _children(morphology, "segmentGroup"):
        group_id = element.get("id")
        if group_id is None:
            raise _Failure("a <segmentGroup> has no id")
        if group_id in elements:
            raise _Failure(f"duplicate segmentGroup id {group_id!r}")
        elements[group_id] = element

    children: dict[int, list[int]] = {segment_id: [] for segment_id in segments}
    for segment in segments.values():
        if segment.parent is not None:
            children[segment.parent].append(segment.id)

    def ancestors(segment_id: int) -> list[int]:
        out = []
        current: int | None = segment_id
        while current is not None:
            out.append(current)
            current = segments[current].parent
        return out

    def subtree(segment_id: int) -> list[int]:
        out, stack = [], [segment_id]
        while stack:
            current = stack.pop()
            out.append(current)
            stack.extend(children[current])
        return out

    def endpoint(group_id: str, element: ET.Element | None, kind: str) -> int | None:
        if element is None:
            return None
        segment_id = _segment_reference(element.get("segment"), f"segmentGroup {group_id!r}: {kind} endpoint")
        if segment_id not in segments:
            raise _Failure(f"segmentGroup {group_id!r}: {kind} endpoint {segment_id} does not exist")
        return segment_id

    resolved: dict[str, tuple[int, ...]] = {}

    def members(group_id: str, stack: tuple[str, ...]) -> tuple[int, ...]:
        if group_id in resolved:
            return resolved[group_id]
        if group_id in stack:
            raise _Failure(f"segmentGroup include cycle {list((*stack, group_id))}")
        element = elements[group_id]
        what = f"segmentGroup {group_id!r}"
        out: list[int] = []
        for member in _children(element, "member"):
            segment_id = _segment_reference(member.get("segment"), f"{what}: member segment")
            if segment_id not in segments:
                raise _Failure(f"{what}: member segment {segment_id} does not exist")
            out.append(segment_id)
        for include in _children(element, "include"):
            included = include.get("segmentGroup")
            if included not in elements:
                raise _Failure(f"{what} includes missing group {included!r}")
            out.extend(members(included, (*stack, group_id)))
        for path_element in _children(element, "path"):
            start = endpoint(group_id, _one_or_none(path_element, "from", f"{what} <path>"), "path")
            end = endpoint(group_id, _one_or_none(path_element, "to", f"{what} <path>"), "path")
            if start is None or end is None:
                raise _Failure(f"{what}: a <path> needs both <from> and <to>")
            route = ancestors(end)
            if start not in route:
                raise _Failure(f"{what}: path from segment {start} to {end} does not run away from the root")
            out.extend(route[: route.index(start) + 1])
        for subtree_element in _children(element, "subTree"):
            start = endpoint(group_id, _one_or_none(subtree_element, "from", f"{what} <subTree>"), "subTree")
            end = endpoint(group_id, _one_or_none(subtree_element, "to", f"{what} <subTree>"), "subTree")
            if (start is None) == (end is None):
                raise _Failure(f"{what}: a <subTree> needs exactly one of <from> and <to>")
            out.extend(subtree(start) if start is not None else ancestors(end))
        resolved[group_id] = tuple(sorted(set(out)))
        return resolved[group_id]

    return {group_id: members(group_id, ()) for group_id in elements}


def _segment_types(
    morphology: ET.Element,
    segments: Mapping[int, Segment],
    groups: Mapping[str, tuple[int, ...]],
) -> dict[int, str]:
    neurolex = {element.get("id"): element.get("neuroLexId") for element in _children(morphology, "segmentGroup")}
    types: dict[int, str] = {}
    for branch_type, reserved_id in TYPE_GROUP_IDS.items():
        for group_id, members in groups.items():
            if group_id != reserved_id and neurolex[group_id] not in TYPE_NEUROLEX_IDS[branch_type]:
                continue
            for segment_id in members:
                previous = types.setdefault(segment_id, branch_type)
                if previous != branch_type:
                    raise _Failure(
                        f"segment {segment_id} is in both a {previous} group and a {branch_type} group ({group_id!r})"
                    )
    return {segment_id: types.get(segment_id, "custom") for segment_id in segments}


def _cables(morphology: ET.Element, groups: Mapping[str, tuple[int, ...]]) -> tuple[tuple[str, ...], dict[int, str]]:
    cable_groups = tuple(
        element.get("id")
        for element in _children(morphology, "segmentGroup")
        if element.get("neuroLexId") == CABLE_NEUROLEX_ID
    )
    owner: dict[int, str] = {}
    for group_id in cable_groups:
        for segment_id in groups[group_id]:
            previous = owner.setdefault(segment_id, group_id)
            if previous != group_id:
                raise _Failure(f"segment {segment_id} is in two unbranched cable groups, {previous!r} and {group_id!r}")
    return cable_groups, owner


def _ignored_elements(cell: ET.Element, morphology: ET.Element) -> tuple[str, ...]:
    ignored = {_local(child.tag) for child in cell if child is not morphology}
    for child in morphology:
        name = _local(child.tag)
        used = {"segment": _SEGMENT_CONTENT, "segmentGroup": _GROUP_CONTENT}.get(name)
        if used is None:
            ignored.add(name)
        else:
            ignored.update(_local(grandchild.tag) for grandchild in child if _local(grandchild.tag) not in used)
    return tuple(sorted(ignored))


def parse_document(path: FilePath, *, cell_id: str | None = None) -> NeuroMlDocument:
    """Parse one cell's morphology from a NeuroML2 document.

    Parameters
    ----------
    path : str or os.PathLike
        NeuroML2 document. ``<include>`` elements are not followed.
    cell_id : str, optional
        Id of the ``<cell>`` to read; required when the document has several.

    Returns
    -------
    NeuroMlDocument

    Raises
    ------
    ValueError
        ``NeuroML2 import failed for <path>: <reason>`` for any malformed or
        unsupported input.
    """

    source = Path(path)
    try:
        root = _load_root(source)
        cell = _select_cell(root, cell_id)
        morphology = _select_morphology(root, cell)
        raw_segments = _read_segments(morphology)
        root_segment = _check_tree(raw_segments)
        segments = _resolve_points(raw_segments, root_segment)
        groups = _expand_groups(morphology, segments)
        segment_types = _segment_types(morphology, segments, groups)
        cable_groups, segment_cables = _cables(morphology, groups)
    except _Failure as failure:
        raise ValueError(f"NeuroML2 import failed for {source}: {failure}") from None
    return NeuroMlDocument(
        path=source,
        cell_id=cell.get("id"),
        morphology_id=morphology.get("id"),
        segments=segments,
        root=root_segment,
        segment_groups=groups,
        segment_types=segment_types,
        cable_groups=cable_groups,
        segment_cables=segment_cables,
        ignored_elements=_ignored_elements(cell, morphology),
    )
