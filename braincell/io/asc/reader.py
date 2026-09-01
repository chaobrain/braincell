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


from dataclasses import dataclass
from pathlib import Path

import brainunit as u
import numpy as np

from .types import AscMetadata, AscReport, AscSpineRecord
from braincell._typing import FilePath
from braincell.io import _geometry
from braincell.io._geometry import MIN_SYNTHETIC_LENGTH_UM
from braincell.morph.morphology import Branch, Morphology, MorphoBranch
from braincell.morph.branch import Soma, branch_class_for_type

_PIPE = object()
#: Single characters that tokenize to one punctuation token each.
_PUNCTUATION_KINDS = {"(": "lparen", ")": "rparen", "<": "leftsp", ">": "rightsp", "|": "pipe"}
_NEURITE_TYPE_MAP = {
    "axon": "axon",
    "dendrite": "dendrite",
    "dend": "dendrite",
    "apical": "apical_dendrite",
    "apicaldendrite": "apical_dendrite",
    "apicaldend": "apical_dendrite",
}
_IGNORED_SYMBOLS = {"normal"}
_ANNOTATION_KEYS = {
    "cellbody",
    "color",
    "sections",
    "imagecoords",
    "rgb",
    *set(_NEURITE_TYPE_MAP),
}


@dataclass(frozen=True)
class _AscToken:
    kind: str
    value: object
    line_number: int


@dataclass(frozen=True)
class _AscPoint:
    x: float
    y: float
    z: float
    diameter: float
    line_number: int

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def radius(self) -> float:
        return 0.5 * float(self.diameter)


@dataclass(frozen=True)
class _AscSegment:
    points: tuple[_AscPoint, ...]
    children: tuple["_AscSegment", ...]
    branch_type: str


@dataclass(frozen=True)
class _AscSpineBlock:
    items: tuple[object, ...]
    line_number: int


@dataclass(frozen=True)
class AscReader:
    def read(self, path: FilePath, return_report: bool = False):
        source_path = Path(path)
        report = AscReport()
        try:
            expressions = self._parse_document(source_path.read_text(), report)
            contours, neurites = self._extract_blocks(expressions, report)
            morpho = self._build_morpho(contours, neurites, report, path=source_path)
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"ASC import failed for {source_path}: {exc}") from exc

        if return_report:
            return morpho, report
        return morpho

    def _parse_document(self, text: str, report: AscReport) -> tuple[object, ...]:
        tokens = self._tokenize(text, report.metadata)
        expressions, index = self._parse_expressions(tokens, 0)
        if index != len(tokens):
            token = tokens[index]
            raise ValueError(f"Unexpected token {token.value!r} at line {token.line_number}.")
        return expressions

    def _tokenize(self, text: str, metadata: AscMetadata) -> tuple[_AscToken, ...]:
        tokens: list[_AscToken] = []
        index = 0
        line_number = 1
        while index < len(text):
            char = text[index]
            if char in " \t\r,":
                index += 1
                continue
            if char == "\n":
                line_number += 1
                index += 1
                continue
            if char == ";":
                end = text.find("\n", index)
                if end == -1:
                    comment = text[index + 1 :].strip()
                    if comment:
                        metadata.comments.append(comment)
                    break
                comment = text[index + 1 : end].strip()
                if comment:
                    metadata.comments.append(comment)
                index = end
                continue
            punctuation_kind = _PUNCTUATION_KINDS.get(char)
            if punctuation_kind is not None:
                tokens.append(_AscToken(punctuation_kind, char, line_number))
                index += 1
                continue
            if char == '"':
                end = index + 1
                while end < len(text) and text[end] != '"':
                    if text[end] == "\n":
                        line_number += 1
                    end += 1
                if end >= len(text):
                    raise ValueError(f"Unterminated string literal at line {line_number}.")
                tokens.append(_AscToken("string", text[index + 1 : end], line_number))
                index = end + 1
                continue

            end = index
            while end < len(text) and text[end] not in "()<>|;, \t\r\n":
                end += 1
            raw = text[index:end]
            tokens.append(_AscToken("atom", self._coerce_atom(raw), line_number))
            index = end
        return tuple(tokens)

    def _coerce_atom(self, raw: str) -> object:
        try:
            return float(raw)
        except ValueError:
            return raw

    def _parse_expressions(self, tokens: tuple[_AscToken, ...], index: int) -> tuple[tuple[object, ...], int]:
        expressions: list[object] = []
        while index < len(tokens):
            token = tokens[index]
            if token.kind == "rparen":
                break
            expr, index = self._parse_expression(tokens, index)
            expressions.append(expr)
        return tuple(expressions), index

    def _parse_group(
        self,
        tokens: tuple[_AscToken, ...],
        index: int,
        *,
        closer: str,
        opener: str,
    ) -> tuple[tuple[object, ...], int]:
        """Collect expressions until *closer*, returning them and the next index."""

        opener_token = tokens[index]
        items: list[object] = []
        index += 1
        while index < len(tokens) and tokens[index].kind != closer:
            if tokens[index].kind == "pipe":
                items.append(_PIPE)
                index += 1
                continue
            item, index = self._parse_expression(tokens, index)
            items.append(item)
        if index >= len(tokens):
            raise ValueError(f"Unclosed {opener!r} at line {opener_token.line_number}.")
        return tuple(items), index + 1

    def _parse_expression(self, tokens: tuple[_AscToken, ...], index: int) -> tuple[object, int]:
        token = tokens[index]
        if token.kind == "lparen":
            return self._parse_group(tokens, index, closer="rparen", opener="(")
        if token.kind == "leftsp":
            items, index = self._parse_group(tokens, index, closer="rightsp", opener="<")
            return _AscSpineBlock(items=items, line_number=token.line_number), index
        if token.kind in {"string", "atom"}:
            return token.value, index + 1
        if token.kind == "pipe":
            return _PIPE, index + 1
        raise ValueError(f"Unexpected token {token.value!r} at line {token.line_number}.")

    def _extract_blocks(
        self,
        expressions: tuple[object, ...],
        report: AscReport,
    ) -> tuple[tuple[tuple[_AscPoint, ...], ...], tuple[_AscSegment, ...]]:
        contours: list[tuple[_AscPoint, ...]] = []
        neurites: list[_AscSegment] = []
        for expr in expressions:
            self._collect_metadata(expr, report.metadata)
            if not isinstance(expr, tuple):
                continue

            block_kind = self._block_kind(expr)
            if block_kind == "soma":
                points = tuple(self._iter_points(expr))
                if points:
                    contours.append(points)
                continue
            if block_kind is None:
                continue

            for segment in self._parse_segments(expr, branch_type=block_kind, report=report):
                normalized = self._normalize_segment(segment)
                if normalized is not None:
                    neurites.append(normalized)

        return tuple(contours), tuple(neurites)

    def _block_kind(self, expr: tuple[object, ...]) -> str | None:
        for item in expr:
            if not isinstance(item, tuple):
                continue
            key = self._head_key(item)
            if key == "cellbody":
                return "soma"
            if key in _NEURITE_TYPE_MAP:
                return _NEURITE_TYPE_MAP[key]
        return None

    def _parse_segments(
        self,
        expr: tuple[object, ...],
        *,
        branch_type: str,
        report: AscReport,
    ) -> tuple[_AscSegment, ...]:
        segments = []
        for arm in self._split_arms(expr):
            segment = self._parse_arm(arm, branch_type=branch_type, report=report)
            if segment.points or segment.children:
                segments.append(segment)
        return tuple(segments)

    def _parse_arm(
        self,
        items: tuple[object, ...],
        *,
        branch_type: str,
        report: AscReport,
    ) -> _AscSegment:
        points: list[_AscPoint] = []
        seen_children = False
        children: list[_AscSegment] = []

        for item in items:
            if self._is_annotation(item):
                continue
            if isinstance(item, _AscSpineBlock):
                if seen_children:
                    report.add_warning(
                        "syntax.spine_after_children",
                        "Ignoring ASC spine block that appeared after child branches had started.",
                        line_number=item.line_number,
                    )
                    continue
                if not points:
                    report.add_warning(
                        "syntax.spine_before_point",
                        "Ignoring ASC spine block that appeared before any parent branch point.",
                        line_number=item.line_number,
                    )
                    continue
                spine_record = self._spine_record_from_block(base_point=points[-1], block=item, report=report)
                if spine_record is not None:
                    report.metadata.spines.append(spine_record)
                continue
            if self._is_point_expr(item):
                if seen_children:
                    report.add_warning(
                        "syntax.point_after_children",
                        "Ignoring ASC point that appeared after child branches had started.",
                    )
                    continue
                points.append(self._point_from_expr(item))
                continue
            if isinstance(item, tuple):
                seen_children = True
                children.extend(self._parse_segments(item, branch_type=branch_type, report=report))
                continue
            if isinstance(item, str) and self._normalize_name(item) in _IGNORED_SYMBOLS:
                continue

        return _AscSegment(points=tuple(points), children=tuple(children), branch_type=branch_type)

    def _spine_record_from_block(
        self,
        *,
        base_point: _AscPoint,
        block: _AscSpineBlock,
        report: AscReport,
    ) -> AscSpineRecord | None:
        property_exprs: list[tuple[object, ...]] = []
        tip_points: list[_AscPoint] = []

        for item in block.items:
            if self._is_point_expr(item):
                tip_points.append(self._point_from_expr(item))
                continue
            if isinstance(item, tuple):
                property_exprs.append(item)
                continue
            if item is _PIPE:
                report.add_warning(
                    "syntax.spine_pipe",
                    "Ignoring unexpected '|' inside ASC spine block.",
                    line_number=block.line_number,
                )
                continue
            report.add_warning(
                "syntax.spine_item",
                f"Ignoring unexpected ASC spine item {item!r}.",
                line_number=block.line_number,
            )

        if len(tip_points) == 0:
            report.add_warning(
                "syntax.spine_missing_tip",
                "Ignoring ASC spine block with no tip point.",
                line_number=block.line_number,
            )
            return None
        if len(tip_points) > 1:
            report.add_warning(
                "syntax.spine_multiple_tips",
                "Ignoring ASC spine block with multiple tip points.",
                line_number=block.line_number,
            )
            return None

        class_type: int | float | None = None
        class_label: str | None = None
        for expr in property_exprs:
            key = self._head_key(expr)
            if key != "class" or len(expr) < 3:
                continue
            raw_type = expr[1]
            raw_label = expr[2]
            if isinstance(raw_type, (int, float)):
                value = float(raw_type)
                class_type = int(value) if value.is_integer() else value
            if isinstance(raw_label, str):
                class_label = raw_label

        tip_point = tip_points[0]
        return AscSpineRecord(
            base_xyz=(float(base_point.x), float(base_point.y), float(base_point.z)),
            base_diameter=float(base_point.diameter),
            tip_xyz=(float(tip_point.x), float(tip_point.y), float(tip_point.z)),
            tip_diameter=float(tip_point.diameter),
            class_type=class_type,
            class_label=class_label,
            properties=tuple(property_exprs),
            line_number=block.line_number,
        )

    def _split_arms(self, expr: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
        arms: list[list[object]] = [[]]
        for item in expr:
            if item is _PIPE:
                arms.append([])
                continue
            arms[-1].append(item)
        return tuple(tuple(arm) for arm in arms if arm)

    def _normalize_segment(self, segment: _AscSegment | None) -> _AscSegment | None:
        if segment is None:
            return None

        children = tuple(
            child for child in (self._normalize_segment(child) for child in segment.children) if child is not None
        )
        # A pointless node with exactly one child is pure nesting: splice it out.
        if not segment.points and len(children) == 1:
            return children[0]
        return _AscSegment(points=segment.points, children=children, branch_type=segment.branch_type)

    def _build_morpho(
        self,
        contours: tuple[tuple[_AscPoint, ...], ...],
        neurites: tuple[_AscSegment, ...],
        report: AscReport,
        *,
        path: Path,
    ) -> Morphology:
        if not contours and not neurites:
            raise ValueError(f"ASC import failed for {path}: no soma contour or neurites were found.")

        if contours:
            # Cross the boundary into format-independent geometry once:
            # everything below works on (N, 3) xyz arrays, not _AscPoint.
            stacks = _geometry.group_contour_stacks(
                tuple(np.asarray([point.xyz for point in contour], dtype=float) for contour in contours)
            )
            if len(stacks) != 1:
                raise ValueError(
                    f"ASC import failed for {path}: found {len(stacks)} disjoint CellBody contour groups; "
                    "Braincell currently supports exactly one soma."
                )
            stack = stacks[0]
            soma_branch, center, radius = self._soma_branch_from_stack(stack, path=path)
            soma_bbox_xy = _geometry.loose_bbox_xy(stack)
        else:
            first_point = self._first_point(neurites)
            if first_point is None:
                raise ValueError(f"ASC import failed for {path}: no geometry points were found.")
            center = first_point.xyz
            radius = max(float(first_point.radius), MIN_SYNTHETIC_LENGTH_UM)
            report.add_warning(
                "topology.synthetic_soma",
                "ASC file has no CellBody contour; synthesized a soma from the first neurite root point.",
            )
            points, radii = _geometry.synthetic_soma_geometry(center, radius)
            soma_branch = Soma.from_points(points=points * u.um, radii=radii * u.um)
            soma_bbox_xy = None

        morpho = Morphology.from_root(soma_branch, name="soma")
        for neurite in neurites:
            if soma_bbox_xy is not None:
                root_point = self._first_point((neurite,))
                if root_point is not None and not _geometry.point_inside_bbox_xy(root_point.xyz, soma_bbox_xy):
                    report.add_warning(
                        "topology.root_outside_soma_bbox",
                        "Main branch root is outside the soma bounding box; connected to the nearest soma center.",
                        line_number=None,
                    )
            self._attach_segment(
                parent=morpho.root,
                segment=neurite,
                parent_branch_type="soma",
                attach_point=center,
                attach_radius=radius,
                parent_x=0.5,
                report=report,
            )
        return morpho

    def _attach_segment(
        self,
        *,
        parent: MorphoBranch,
        segment: _AscSegment,
        parent_branch_type: str,
        attach_point: np.ndarray,
        attach_radius: float,
        parent_x: float,
        report: AscReport,
    ) -> None:
        if not segment.points:
            for child in segment.children:
                self._attach_segment(
                    parent=parent,
                    segment=child,
                    parent_branch_type=parent_branch_type,
                    attach_point=attach_point,
                    attach_radius=attach_radius,
                    parent_x=parent_x,
                    report=report,
                )
            return

        branch = self._segment_branch(
            segment,
            parent_branch_type=parent_branch_type,
            attach_point=attach_point,
            parent_x=parent_x,
        )
        tail_point = segment.points[-1].xyz
        tail_radius = float(segment.points[-1].radius)

        if branch is None:
            report.add_warning(
                "geometry.degenerate_branch",
                "Dropped a zero-length ASC branch and reattached its children to the parent.",
                line_number=segment.points[0].line_number,
            )
            for child in segment.children:
                self._attach_segment(
                    parent=parent,
                    segment=child,
                    parent_branch_type=parent_branch_type,
                    attach_point=tail_point,
                    attach_radius=tail_radius,
                    parent_x=parent_x,
                    report=report,
                )
            return

        child = parent.attach(branch, parent_x=parent_x, child_x=0.0)
        for grandchild in segment.children:
            self._attach_segment(
                parent=child,
                segment=grandchild,
                parent_branch_type=segment.branch_type,
                attach_point=tail_point,
                attach_radius=tail_radius,
                parent_x=1.0,
                report=report,
            )

    # NEURON Import3d_Neurolucida3() parity notes for ASC geometry:
    # - Column 4 in Neurolucida ASC points is treated as diameter, not radius.
    # - Single-contour CellBody is converted with NEURON-style 21-point principal-axis sampling.
    # - Multi-contour CellBody stacks follow the NEURON stack centroid/diameter path instead of forcing 21 points.
    # - For non-soma parent/child attachments, if child-first xyz differs from the parent terminal xyz, copy the
    #   parent terminal xyz into the child branch; the copied point keeps the child's diameter, not the parent's.
    # - If child-first xyz already matches the parent terminal xyz, do not inject another attachment point.
    # - Angle-bracket spine blocks are metadata attached to the preceding branch point; they do not become branches
    #   or section pt3d points, but they must not terminate the parent point stream.
    # - Preserve repeated consecutive points and one-point sections; NEURON read_nlcda3.hoc can keep both as real
    #   section geometry, and import3d_gui.hoc instantiate() then emits them through pt3dadd()/pt3dstyle().
    # - Root soma attachment remains a logical parent_x=0.5 rule; it is not modeled by inserting a soma midpoint
    #   into child pt3d geometry.
    def _segment_branch(
        self,
        segment: _AscSegment,
        *,
        parent_branch_type: str,
        attach_point: np.ndarray,
        parent_x: float,
    ) -> Branch | None:
        points = [point.xyz for point in segment.points]
        radii = [float(point.radius) for point in segment.points]
        if points:
            # ``keep_radius_jump=False``: unlike the SWC reader, a coincident
            # first point is never duplicated here, whatever its diameter.
            # See braincell.io._geometry.should_copy_attach_point.
            allow_copy = not (parent_branch_type == "soma" and abs(parent_x - 0.5) <= 1e-9 and len(points) > 1)
            if _geometry.should_copy_attach_point(
                allow_copy=allow_copy,
                same_xyz=bool(np.allclose(points[0], attach_point)),
                same_radius=True,
                keep_radius_jump=False,
            ):
                points.insert(0, np.asarray(attach_point, dtype=float))
                radii.insert(0, radii[0])

        if len(points) < 2:
            return None

        lengths_um = np.linalg.norm(np.asarray(points[1:], dtype=float) - np.asarray(points[:-1], dtype=float), axis=1)
        if float(np.sum(lengths_um)) <= 0.0:
            return None

        return branch_class_for_type(segment.branch_type).from_points(
            points=np.asarray(points, dtype=float) * u.um,
            radii=np.asarray(radii, dtype=float) * u.um,
        )

    def _first_point(self, segments: tuple[_AscSegment, ...]) -> _AscPoint | None:
        for segment in segments:
            if segment.points:
                return segment.points[0]
            first = self._first_point(segment.children)
            if first is not None:
                return first
        return None

    def _soma_branch_from_stack(
        self,
        stack: tuple[np.ndarray, ...],
        *,
        path: Path,
    ) -> tuple[Branch, np.ndarray, float]:
        if len(stack) == 1:
            points, radii, center = _geometry.contour_to_centroid(stack[0])
        else:
            # A CellBody stack whose z layers are duplicated or non-monotonic is
            # malformed input, so _validate_soma_stack's ValueError propagates.
            # It used to sit inside the try below, which meant every validation
            # failure silently degraded the soma to its first contour and made
            # the validation itself unreachable.
            self._validate_soma_stack(stack, path=path)
            try:
                points, radii = _geometry.contour_stack_to_centroid(stack)
                center = _geometry.contour_stack_center(stack)
            except ValueError:
                points, radii, center = _geometry.contour_to_centroid(stack[0])

        branch = Soma.from_points(points=points * u.um, radii=radii * u.um)
        return branch, center, float(radii[len(radii) // 2])

    def _validate_soma_stack(
        self,
        stack: tuple[np.ndarray, ...],
        *,
        path: Path,
        tol: float = 1e-6,
    ) -> None:
        """Reject a CellBody stack whose z layers are not strictly monotonic.

        The geometric test lives in :func:`braincell.io._geometry.constant_z`;
        the wording, the offending file, and the contour index are ASC
        reader business and stay here.
        """
        direction = 0
        previous_z = self._contour_constant_z(stack[0], path=path, contour_index=0)
        for index, contour in enumerate(stack[1:], start=1):
            current_z = self._contour_constant_z(contour, path=path, contour_index=index)
            delta_z = current_z - previous_z
            if abs(delta_z) <= tol:
                raise ValueError(
                    f"ASC import failed for {path}: adjacent CellBody contours share the same z value "
                    f"({current_z:.6g}); NEURON-style soma stacks require strictly monotonic z."
                )
            current_direction = 1 if delta_z > 0.0 else -1
            if direction == 0:
                direction = current_direction
            elif direction != current_direction:
                raise ValueError(f"ASC import failed for {path}: CellBody contour stack is not monotonic in z.")
            previous_z = current_z

    def _contour_constant_z(
        self,
        contour: np.ndarray,
        *,
        path: Path,
        contour_index: int,
        tol: float = 1e-6,
    ) -> float:
        z_value = _geometry.constant_z(contour, tol=tol)
        if z_value is None:
            raise ValueError(
                f"ASC import failed for {path}: CellBody contour {contour_index} does not have constant z."
            )
        return z_value

    def _collect_metadata(self, expr: object, metadata: AscMetadata) -> None:
        if isinstance(expr, str):
            if expr.strip() and self._normalize_name(expr) not in _IGNORED_SYMBOLS:
                metadata.source_labels.append(expr)
            return
        if isinstance(expr, _AscSpineBlock):
            for item in expr.items:
                self._collect_metadata(item, metadata)
            return
        if not isinstance(expr, tuple):
            return
        if self._is_point_expr(expr):
            return

        key = self._head_key(expr)
        if key == "color":
            metadata.colors.append(expr)
        elif key == "spine":
            metadata.spine_annotations.append(expr)
        elif key == "marker":
            metadata.markers.append(expr)
        elif key == "filledcircle":
            metadata.filled_circles.append(expr)

        for item in expr:
            self._collect_metadata(item, metadata)

    def _is_annotation(self, expr: object) -> bool:
        if isinstance(expr, str):
            return self._normalize_name(expr) in _IGNORED_SYMBOLS
        if isinstance(expr, _AscSpineBlock):
            return False
        if not isinstance(expr, tuple):
            return False
        if self._is_point_expr(expr):
            return False
        if self._is_property_expr(expr):
            return True
        key = self._head_key(expr)
        return key in _ANNOTATION_KEYS or key in {"spine", "marker", "filledcircle"}

    def _is_property_expr(self, expr: object) -> bool:
        if not isinstance(expr, tuple) or len(expr) == 0:
            return False
        if self._is_point_expr(expr):
            return False
        key = self._head_key(expr)
        if key is None:
            return False
        for item in expr[1:]:
            if item is _PIPE or isinstance(item, _AscSpineBlock):
                return False
            if isinstance(item, tuple) and not self._is_point_expr(item):
                return False
        return True

    def _head_key(self, expr: tuple[object, ...]) -> str | None:
        if not expr:
            return None
        head = expr[0]
        if not isinstance(head, str):
            return None
        return self._normalize_name(head)

    def _normalize_name(self, value: str) -> str:
        return "".join(char for char in value.lower() if char.isalnum())

    def _is_point_expr(self, expr: object) -> bool:
        return isinstance(expr, tuple) and len(expr) == 4 and all(isinstance(item, (float, int)) for item in expr)

    def _point_from_expr(self, expr: object) -> _AscPoint:
        if not isinstance(expr, tuple):
            raise TypeError(f"_point_from_expr: expected tuple, got {type(expr).__name__!r}")
        return _AscPoint(
            x=float(expr[0]),
            y=float(expr[1]),
            z=float(expr[2]),
            diameter=max(float(expr[3]), 0.0),
            line_number=0,
        )

    def _iter_points(self, expr: object):
        if self._is_point_expr(expr):
            yield self._point_from_expr(expr)
            return
        if not isinstance(expr, tuple):
            return
        for item in expr:
            yield from self._iter_points(item)
