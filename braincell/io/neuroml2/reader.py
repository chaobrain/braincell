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

"""Build a :class:`~braincell.Morphology` from one NeuroML2 cell.

The private ``document`` module validates the file and returns the segment
table; this module groups the segments into branches. A segment continues
its parent's branch when it hangs off the parent's distal end as the only
child there, has the same type and cable group, and starts where the parent
ends. A child attached part-way along a segment cuts that branch in two, and
so does a child attached at the start of a segment that is not the first of
its branch. A spherical soma becomes the same three-point stand-in as an SWC
one-point soma.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping

import brainunit as u
import numpy as np

from braincell._typing import FilePath
from braincell.io._geometry import MIN_SYNTHETIC_LENGTH_UM, synthetic_soma_geometry
from braincell.io.neuroml2.document import NeuroMlDocument, Point, Segment, parse_document
from braincell.io.neuroml2.types import NeuroMlMetadata, NeuroMlReport
from braincell.morph.branch import Branch, branch_class_for_type
from braincell.morph.morphology import MorphoBranch, Morphology

__all__ = ["NeuroMlReader"]

# The part of one segment between two fractions along it: (segment id, f0, f1).
_Item = tuple[int, float, float]
# A branch to build: the index of its chain and the segment parts it covers.
_Piece = tuple[int, list[_Item]]


class NeuroMlReader:
    """Read the morphology of one NeuroML2 cell.

    Only the morphology is read: ``<segment>`` geometry and ``<segmentGroup>``
    membership. Channels, biophysical properties and annotations are skipped
    and listed in the report. ``<include>`` elements are not followed, so the
    cell and its morphology must be in the file itself.

    How segments become branches:

    - Consecutive segments form one branch while they are unbranched and of
      the same type and unbranched cable group (``neuroLexId="sao864921383"``).
      A diameter change at the joint becomes a zero-length jump segment.
    - A child attached at ``0 < fractionAlong < 1`` cuts its parent's branch
      in two at that point and attaches at the end of the first part. A child
      at ``fractionAlong="0"`` attaches at ``parent_x=0`` when its parent is
      the first segment of a branch, and cuts the branch at that segment's
      start otherwise. A child at the middle of a one-segment soma attaches
      at ``parent_x=0.5`` without a cut.
    - The type comes from ``soma_group``, ``axon_group`` and
      ``dendrite_group`` or from their NeuroLex ids; other segments are
      ``"custom"``.
    - A spherical segment (proximal and distal at the same point) is read as
      a soma of length and diameter equal to the sphere's diameter, so its
      area is the sphere's; its children attach at ``parent_x=0.5``.
    - A branch is named after its cable group, with ``_0``, ``_1``, ... added
      when the group was cut into several branches. A group id that cannot
      name a branch gives an automatic name and a ``naming.group_id_unusable``
      warning in the report. Branches outside cable groups are named
      automatically, the root ``"soma"`` (or its type) when it has no group.

    The report's metadata maps each segment and segment group back onto the
    branches, since segment groups do not become regions.

    Examples
    --------

    .. code-block:: python

        >>> from braincell.io import NeuroMlReader
        >>> morpho, report = NeuroMlReader().read(
        ...     "data/morphology/neuroml2/minimal.cell.nml", return_report=True
        ... )  # doctest: +SKIP
        >>> [branch.name for branch in morpho.branches]  # doctest: +SKIP
        ['soma', 'trunk_0', 'trunk_1', 'axon_0', 'dendrite_0', 'dendrite_1']
        >>> report.metadata.segment_branches[2]  # doctest: +SKIP
        (('trunk_0', 0.8, 1.0), ('trunk_1', 0.0, 1.0))
    """

    def read(self, path: FilePath, *, cell_id: str | None = None, return_report: bool = False):
        """Read one cell of a NeuroML2 document into a :class:`~braincell.Morphology`.

        Parameters
        ----------
        path : str or os.PathLike
            NeuroML2 document, such as a ``.cell.nml`` file.
        cell_id : str, optional
            Id of the ``<cell>`` to read. Required when the document has more
            than one cell.
        return_report : bool
            If ``True``, return ``(morphology, report)``.

        Returns
        -------
        Morphology or (Morphology, NeuroMlReport)
            The morphology, and with ``return_report=True`` the report whose
            :class:`NeuroMlMetadata` holds the segment groups, the position of
            every segment on the branches and the skipped elements.

        Raises
        ------
        ValueError
            ``NeuroML2 import failed for <path>: <reason>`` for input that is
            not well-formed, not NeuroML2 or not a valid tree, and for a
            spherical segment outside the soma.
        """
        document = parse_document(path, cell_id=cell_id)
        report = NeuroMlReport()
        morpho, segment_branches = _build(document, report)
        report.metadata = NeuroMlMetadata(
            cell_id=document.cell_id,
            morphology_id=document.morphology_id,
            segment_groups=dict(document.segment_groups),
            segment_branches=segment_branches,
            segment_names={sid: seg.name for sid, seg in document.segments.items() if seg.name is not None},
            ignored_elements=document.ignored_elements,
        )
        if return_report:
            return morpho, report
        return morpho


def _build(
    document: NeuroMlDocument, report: NeuroMlReport
) -> tuple[Morphology, dict[int, tuple[tuple[str, float, float], ...]]]:
    segments, types, cables = document.segments, document.segment_types, document.segment_cables
    for segment in segments.values():
        if segment.is_sphere and types[segment.id] != "soma":
            raise ValueError(
                f"NeuroML2 import failed for {document.path}: segment {segment.id}: a spherical segment "
                f"(proximal and distal points coincide) is only supported as the soma; it is {types[segment.id]}"
            )

    children = document.children()
    order = [document.root]
    for segment_id in order:  # breadth first, so every parent comes before its children
        order.extend(sorted(children[segment_id]))
    merged = _merged_segments(document, children)

    chains: list[list[int]] = []
    chain_of: dict[int, int] = {}
    for segment_id in order:
        if segment_id in merged:
            chain_of[segment_id] = chain_of[segments[segment_id].parent]
            chains[chain_of[segment_id]].append(segment_id)
        else:
            chain_of[segment_id] = len(chains)
            chains.append([segment_id])

    cuts, midpoint = _cuts(document, order, merged, chains, chain_of)
    pieces = _pieces(chains, cuts)
    segment_items: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for index, (_, items) in enumerate(pieces):
        for segment_id, f0, f1 in items:
            segment_items[segment_id].append((index, f0, f1))

    def parent_of(index: int) -> tuple[int, float]:
        chain, items = pieces[index]
        if pieces[index - 1][0] == chain:  # the rest of a branch cut in two
            return index - 1, 1.0
        child = segments[items[0][0]]
        parent, fraction = segments[child.parent], child.fraction_along
        first = segment_items[parent.id][0][0]
        if parent.is_sphere or (parent.id in midpoint and fraction == 0.5):
            return first, 0.5
        if fraction == 0.0:
            if first > 0 and pieces[first - 1][0] == pieces[first][0]:
                return first - 1, 1.0  # the start of a cut-off part is the end of the part before it
            return first, 0.0
        return next(i for i, _, f1 in segment_items[parent.id] if f1 == fraction), 1.0

    branches = [_piece_branch(items, segments, types) for _, items in pieces]
    morpho: Morphology | None = None
    views: list[MorphoBranch] = []

    def add(index: int, name: str | None) -> MorphoBranch:
        nonlocal morpho
        if index == 0:
            morpho = Morphology.from_root(branches[0], name=name)
            return morpho.root
        parent_index, parent_x = parent_of(index)
        return morpho.attach(
            parent=views[parent_index], child_branch=branches[index], child_name=name, parent_x=parent_x
        )

    in_cable = Counter(cables.get(items[0][0]) for _, items in pieces)
    seen: Counter = Counter()
    for index, (_, items) in enumerate(pieces):
        first = items[0][0]
        cable = cables.get(first)
        fallback = types[first] if index == 0 else None  # the root is named after its type
        view, reason = None, None
        if cable is not None:
            wanted = cable if in_cable[cable] == 1 else f"{cable}_{seen[cable]}"
            seen[cable] += 1
            try:
                view = add(index, wanted)
            except ValueError as exc:  # braincell's own check decides what can name a branch
                reason = str(exc)
        if view is None:
            view = add(index, fallback)
        if reason is not None:
            report.add_warning(
                "naming.group_id_unusable",
                f"segment group {cable!r} cannot name a branch ({reason}); the branch is named {view.name!r}.",
                segment_group=cable,
            )
        views.append(view)

    segment_branches: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    for (_, items), view in zip(pieces, views, strict=True):
        for (segment_id, _, _), x0, x1 in _x_intervals(items, segments):
            segment_branches[segment_id].append((view.name, x0, x1))
    return morpho, {segment_id: tuple(segment_branches[segment_id]) for segment_id in segments}


def _merged_segments(document: NeuroMlDocument, children: Mapping[int, list[int]]) -> set[int]:
    """Ids of the segments that continue their parent's branch."""
    segments, types, cables = document.segments, document.segment_types, document.segment_cables
    merged = set()
    for parent_id, child_ids in children.items():
        at_distal = [child_id for child_id in child_ids if segments[child_id].fraction_along == 1.0]
        if len(at_distal) != 1:
            continue
        parent, child = segments[parent_id], segments[at_distal[0]]
        if (
            types[child.id] == types[parent_id]
            and cables.get(child.id) == cables.get(parent_id)
            and not parent.is_sphere
            and not child.is_sphere
            and child.proximal[:3] == parent.distal[:3]
        ):
            merged.add(child.id)
    return merged


def _cuts(
    document: NeuroMlDocument,
    order: list[int],
    merged: set[int],
    chains: list[list[int]],
    chain_of: Mapping[int, int],
) -> tuple[dict[int, set[float]], set[int]]:
    """Where each segment is cut, and the one-segment somas left whole for a midpoint child."""
    segments, types = document.segments, document.segment_types
    cuts: dict[int, set[float]] = defaultdict(set)
    midpoint: set[int] = set()
    for child_id in order[1:]:
        if child_id in merged:
            continue
        child = segments[child_id]
        parent, fraction = segments[child.parent], child.fraction_along
        if parent.is_sphere:  # every point of a sphere is its centre
            continue
        chain = chains[chain_of[parent.id]]
        if fraction == 0.5 and types[parent.id] == "soma" and len(chain) == 1:
            midpoint.add(parent.id)
        elif 0.0 < fraction < 1.0 or (fraction == 0.0 and chain[0] != parent.id):
            cuts[parent.id].add(fraction)
    for soma_id in midpoint:
        if cuts.get(soma_id):  # cut anyway, so the midpoint child goes at the end of a part
            cuts[soma_id].add(0.5)
    midpoint.difference_update(cuts)
    return cuts, midpoint


def _pieces(chains: list[list[int]], cuts: Mapping[int, set[float]]) -> list[_Piece]:
    pieces: list[_Piece] = []
    for chain_index, chain in enumerate(chains):
        current: list[_Item] = []
        for segment_id in chain:
            start = 0.0
            for fraction in sorted(cuts.get(segment_id, ())):
                if fraction > 0.0:
                    current.append((segment_id, start, fraction))
                    start = fraction
                if current:
                    pieces.append((chain_index, current))
                    current = []
            current.append((segment_id, start, 1.0))
        pieces.append((chain_index, current))
    return pieces


def _at(segment: Segment, fraction: float) -> Point:
    if fraction == 0.0:
        return segment.proximal
    if fraction == 1.0:
        return segment.distal
    return tuple((1.0 - fraction) * a + fraction * b for a, b in zip(segment.proximal, segment.distal, strict=True))


def _piece_branch(items: list[_Item], segments: Mapping[int, Segment], types: Mapping[int, str]) -> Branch:
    first = segments[items[0][0]]
    branch_cls = branch_class_for_type(types[first.id])
    if first.is_sphere:  # never merged or cut, so it is the only item
        radius = max(first.distal[3] / 2.0, MIN_SYNTHETIC_LENGTH_UM)
        points, radii = synthetic_soma_geometry(np.asarray(first.distal[:3], dtype=float), radius)
        return branch_cls.from_points(points=points * u.um, radii=radii * u.um)
    points, radii_proximal, radii_distal = [], [], []
    for segment_id, f0, f1 in items:
        start, end = _at(segments[segment_id], f0), _at(segments[segment_id], f1)
        if not points:
            points.append(start[:3])
        points.append(end[:3])
        radii_proximal.append(start[3] / 2.0)
        radii_distal.append(end[3] / 2.0)
    # A diameter change between items becomes a zero-length jump segment in
    # canonicalization; giving the radii per segment keeps from_points quiet.
    return branch_cls.from_points(
        points=np.asarray(points, dtype=float) * u.um,
        radii_proximal=np.asarray(radii_proximal, dtype=float) * u.um,
        radii_distal=np.asarray(radii_distal, dtype=float) * u.um,
    )


def _x_intervals(items: list[_Item], segments: Mapping[int, Segment]) -> list[tuple[_Item, float, float]]:
    if segments[items[0][0]].is_sphere:
        return [(items[0], 0.0, 1.0)]
    lengths = []
    for segment_id, f0, f1 in items:
        start, end = _at(segments[segment_id], f0), _at(segments[segment_id], f1)
        lengths.append(float(np.linalg.norm(np.subtract(end[:3], start[:3]))))
    total = sum(lengths)
    out, done = [], 0.0
    for position, (item, length) in enumerate(zip(items, lengths, strict=True)):
        x0 = done / total
        done += length
        out.append((item, x0, 1.0 if position == len(items) - 1 else done / total))
    return out
