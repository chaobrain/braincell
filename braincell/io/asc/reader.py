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
from os import PathLike
from pathlib import Path

import numpy as np

from .types import AscMetadata, AscReport, AscSpineRecord
from braincell.io.swc.types import MIN_SYNTHETIC_LENGTH_UM
from braincell._misc import u
from braincell.morph._morphology import Branch, Morphology, MorphoBranch
from braincell.morph._branch import Soma, branch_class_for_type

_PIPE = object()
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
    def read(self, path: str | PathLike[str], return_report: bool = False):
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
                    comment = text[index + 1:].strip()
                    if comment:
                        metadata.comments.append(comment)
                    break
                comment = text[index + 1: end].strip()
                if comment:
                    metadata.comments.append(comment)
                index = end
                continue
            if char == "(":
                tokens.append(_AscToken("lparen", char, line_number))
                index += 1
                continue
            if char == ")":
                tokens.append(_AscToken("rparen", char, line_number))
                index += 1
                continue
            if char == "<":
                tokens.append(_AscToken("leftsp", char, line_number))
                index += 1
                continue
            if char == ">":
                tokens.append(_AscToken("rightsp", char, line_number))
                index += 1
                continue
            if char == "|":
                tokens.append(_AscToken("pipe", char, line_number))
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
                tokens.append(_AscToken("string", text[index + 1: end], line_number))
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

    def _parse_expression(self, tokens: tuple[_AscToken, ...], index: int) -> tuple[object, int]:
        token = tokens[index]
        if token.kind == "lparen":
            items: list[object] = []
            index += 1
            while index < len(tokens) and tokens[index].kind != "rparen":
                if tokens[index].kind == "pipe":
                    items.append(_PIPE)
                    index += 1
                    continue
                item, index = self._parse_expression(tokens, index)
                items.append(item)
            if index >= len(tokens):
                raise ValueError(f"Unclosed '(' at line {token.line_number}.")
            return tuple(items), index + 1
        if token.kind == "leftsp":
            items: list[object] = []
            index += 1
            while index < len(tokens) and tokens[index].kind != "rightsp":
                if tokens[index].kind == "pipe":
                    items.append(_PIPE)
                    index += 1
                    continue
                item, index = self._parse_expression(tokens, index)
                items.append(item)
            if index >= len(tokens):
                raise ValueError(f"Unclosed '<' at line {token.line_number}.")
            return _AscSpineBlock(items=tuple(items), line_number=token.line_number), index + 1
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
            child
            for child in (self._normalize_segment(child) for child in segment.children)
            if child is not None
        )
        points = segment.points

        if len(points) == 0:
            if len(children) == 1:
                return children[0]
            return _AscSegment(points=(), children=children, branch_type=segment.branch_type)

        if len(points) == 1:
            return _AscSegment(points=points, children=children, branch_type=segment.branch_type)

        return _AscSegment(points=points, children=children, branch_type=segment.branch_type)

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
            soma_branch, center, radius = self._soma_branch_from_contours(contours, path=path)
            stack = self._merge_soma_contours(contours)[0]
            soma_bbox_xy = self._soma_loose_bbox_xy(stack)
        else:
            first_point = self._first_point(neurites)
            if first_point is None:
                raise ValueError(f"ASC import failed for {path}: no geometry points were found.")
            center = first_point.xyz
            radius = max(float(first_point.radius), MIN_SYNTHETIC_LENGTH_UM)
            report.add_warning("topology.synthetic_soma",
                               "ASC file has no CellBody contour; synthesized a soma from the first neurite root point.")
            soma_branch = self._synthetic_soma_branch(center=center, radius=radius)
            soma_bbox_xy = None

        morpho = Morphology.from_root(soma_branch, name="soma")
        for neurite in neurites:
            if soma_bbox_xy is not None:
                root_point = self._first_point((neurite,))
                if root_point is not None and not self._point_inside_bbox_xy(root_point.xyz, soma_bbox_xy):
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
            attach_radius=attach_radius,
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
        attach_radius: float,
        parent_x: float,
    ) -> Branch | None:
        points = [point.xyz for point in segment.points]
        radii = [float(point.radius) for point in segment.points]
        if points:
            should_copy_attach = True
            if parent_branch_type == "soma" and abs(parent_x - 0.5) <= 1e-9 and len(points) > 1:
                should_copy_attach = False
            if np.allclose(points[0], attach_point):
                should_copy_attach = False

            attach_radius_for_child = radii[0]
            if should_copy_attach:
                points.insert(0, np.asarray(attach_point, dtype=float))
                radii.insert(0, attach_radius_for_child)

        if len(points) < 2:
            return None

        lengths_um = np.linalg.norm(np.asarray(points[1:], dtype=float) - np.asarray(points[:-1], dtype=float), axis=1)
        if float(np.sum(lengths_um)) <= 0.0:
            return None

        return branch_class_for_type(segment.branch_type).from_points(
            points=np.asarray(points, dtype=float) * u.um,
            radii=np.asarray(radii, dtype=float) * u.um,
        )

    def _dedupe_shared_points(
        self,
        points: list[np.ndarray],
        radii: list[float],
    ) -> tuple[list[np.ndarray], list[float]]:
        if not points:
            return [], []
        dedup_points = [np.asarray(points[0], dtype=float)]
        dedup_radii = [float(radii[0])]
        for point, radius in zip(points[1:], radii[1:]):
            xyz = np.asarray(point, dtype=float)
            if np.allclose(dedup_points[-1], xyz) and np.isclose(dedup_radii[-1], float(radius)):
                continue
            dedup_points.append(xyz)
            dedup_radii.append(float(radius))
        return dedup_points, dedup_radii

    def _merge_point_sequences(
        self,
        prefix: tuple[_AscPoint, ...],
        suffix: tuple[_AscPoint, ...],
    ) -> tuple[_AscPoint, ...]:
        if not prefix:
            return suffix
        if not suffix:
            return prefix
        if self._same_point(prefix[-1], suffix[0]):
            return prefix + suffix[1:]
        return prefix + suffix

    def _same_point(self, left: _AscPoint, right: _AscPoint) -> bool:
        return np.allclose(left.xyz, right.xyz) and np.isclose(left.diameter, right.diameter)

    def _first_point(self, segments: tuple[_AscSegment, ...]) -> _AscPoint | None:
        for segment in segments:
            if segment.points:
                return segment.points[0]
            first = self._first_point(segment.children)
            if first is not None:
                return first
        return None

    def _soma_branch_from_contours(
        self,
        contours: tuple[tuple[_AscPoint, ...], ...],
        *,
        path: Path,
    ) -> tuple[Branch, np.ndarray, float]:
        stacks = self._merge_soma_contours(contours)
        if len(stacks) != 1:
            raise ValueError(
                f"ASC import failed for {path}: found {len(stacks)} disjoint CellBody contour groups; "
                "Braincell currently supports exactly one soma."
            )

        stack = stacks[0]
        if len(stack) == 1:
            points, radii, center = self._contour2centroid(stack[0])
        else:
            try:
                self._validate_soma_stack(stack, path=path)
                points, radii = self._contourstack2centroid(stack)
                center = self._soma_stack_center(stack)
            except ValueError:
                points, radii, center = self._contour2centroid(stack[0])

        branch = Soma.from_points(points=points * u.um, radii=radii * u.um)
        return branch, center, float(radii[len(radii) // 2])

    def _merge_soma_contours(
        self,
        contours: tuple[tuple[_AscPoint, ...], ...],
    ) -> tuple[tuple[tuple[_AscPoint, ...], ...], ...]:
        if not contours:
            return tuple()

        stacks: list[tuple[tuple[_AscPoint, ...], ...]] = []
        current_stack: list[tuple[_AscPoint, ...]] = [contours[0]]
        previous_bbox = self._contour_bbox_xy(contours[0])
        for contour in contours[1:]:
            bbox = self._contour_bbox_xy(contour)
            if self._xy_intersect(previous_bbox, bbox):
                current_stack.append(contour)
            else:
                stacks.append(tuple(current_stack))
                current_stack = [contour]
            previous_bbox = bbox
        stacks.append(tuple(current_stack))
        return tuple(stacks)

    def _contour_bbox_xy(self, contour: tuple[_AscPoint, ...]) -> tuple[float, float, float, float]:
        xs = [point.x for point in contour]
        ys = [point.y for point in contour]
        return min(xs), max(xs), min(ys), max(ys)

    def _xy_intersect(
        self,
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> bool:
        xmin1, xmax1, ymin1, ymax1 = left
        xmin2, xmax2, ymin2, ymax2 = right
        return not (xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1)

    def _soma_loose_bbox_xy(
        self,
        stack: tuple[tuple[_AscPoint, ...], ...],
    ) -> tuple[float, float, float, float]:
        if len(stack) == 1:
            xmin, xmax, ymin, ymax = self._contour_bbox_xy(stack[0])
            return xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5

        bboxes = [self._contour_bbox_xy(contour) for contour in stack]
        return (
            min(bbox[0] for bbox in bboxes),
            max(bbox[1] for bbox in bboxes),
            min(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    def _point_inside_bbox_xy(
        self,
        xyz: np.ndarray,
        bbox_xy: tuple[float, float, float, float],
    ) -> bool:
        xmin, xmax, ymin, ymax = bbox_xy
        x, y = float(xyz[0]), float(xyz[1])
        return xmin <= x <= xmax and ymin <= y <= ymax

    def _validate_soma_stack(
        self,
        stack: tuple[tuple[_AscPoint, ...], ...],
        *,
        path: Path,
        tol: float = 1e-6,
    ) -> None:
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
                raise ValueError(
                    f"ASC import failed for {path}: CellBody contour stack is not monotonic in z."
                )
            previous_z = current_z

    def _contour_constant_z(
        self,
        contour: tuple[_AscPoint, ...],
        *,
        path: Path,
        contour_index: int,
        tol: float = 1e-6,
    ) -> float:
        z0 = float(contour[0].z)
        for point in contour[1:]:
            if abs(float(point.z) - z0) > tol:
                raise ValueError(
                    f"ASC import failed for {path}: CellBody contour {contour_index} does not have constant z."
                )
        return z0

    def _contour_center_radius(self, points: tuple[_AscPoint, ...]) -> tuple[np.ndarray, float]:
        xyz = np.array([point.xyz for point in points], dtype=float)
        center = xyz.mean(axis=0)
        radius = max(np.linalg.norm(point.xyz - center) + float(point.radius) for point in points)
        return center, max(float(radius), MIN_SYNTHETIC_LENGTH_UM)

    def _contourcenter(
        self,
        contour: tuple[_AscPoint, ...],
        *,
        num: int = 101,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.array([point.x for point in contour], dtype=float)
        y = np.array([point.y for point in contour], dtype=float)
        z = np.array([point.z for point in contour], dtype=float)
        seg_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        perimeter = np.zeros(len(x), dtype=float)
        perimeter[1:] = np.cumsum(seg_lengths)
        d_uniform = np.linspace(0.0, perimeter[-1], num)
        x_new = np.interp(d_uniform, perimeter, x)
        y_new = np.interp(d_uniform, perimeter, y)
        z_new = np.interp(d_uniform, perimeter, z)
        mean = np.array([x_new.mean(), y_new.mean(), z_new.mean()], dtype=float)
        return mean, x_new, y_new, z_new

    def _soma_axis_sampling(
        self,
        contour: tuple[_AscPoint, ...],
        *,
        n_samples: int = 21,
        arclength_resample: int = 101,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean, x_new, y_new, _ = self._contourcenter(contour, num=arclength_resample)
        mean_xy = mean[:2]

        pts = np.stack([x_new, y_new], axis=1)
        pts_centered = pts - mean_xy
        cov = np.cov(pts_centered, rowvar=False)
        _, eigvecs = np.linalg.eigh(cov)
        major = eigvecs[:, 1]
        minor = eigvecs[:, 0]
        neuron_major = self._neuron_major_xy_for_contour(contour)
        if neuron_major is not None:
            if float(np.dot(major, neuron_major)) < 0.0:
                major = -major
        else:
            if major[np.argmax(np.abs(major))] < 0.0:
                major = -major
            first_projection = (np.array([contour[0].x, contour[0].y], dtype=float) - mean_xy) @ major
            last_projection = (np.array([contour[-1].x, contour[-1].y], dtype=float) - mean_xy) @ major
            if first_projection > 0.0 and last_projection < 0.0:
                major = -major
        major = major / np.linalg.norm(major)
        minor = minor / np.linalg.norm(minor)

        d = (pts - mean_xy) @ major
        rad = (pts - mean_xy) @ minor

        def _rotate(values: np.ndarray, k: int) -> np.ndarray:
            return np.concatenate([values[k:], values[:k]])

        def _keep_strictly_monotonic(
            x_values: np.ndarray,
            y_values: np.ndarray,
            *,
            increasing: bool,
            tol: float = 1e-8,
        ) -> tuple[np.ndarray, np.ndarray]:
            keep_indices = [0]
            for index in range(1, len(x_values)):
                if increasing:
                    if x_values[index] > x_values[keep_indices[-1]] + tol:
                        keep_indices.append(index)
                elif x_values[index] < x_values[keep_indices[-1]] - tol:
                    keep_indices.append(index)
            keep = np.asarray(keep_indices, dtype=int)
            return x_values[keep], y_values[keep]

        def _interp_strict(xp: np.ndarray, fp: np.ndarray, x_values: np.ndarray) -> np.ndarray:
            if len(xp) == 1:
                return np.full_like(x_values, fp[0], dtype=float)
            if xp[0] > xp[-1]:
                xp = xp[::-1]
                fp = fp[::-1]
            return np.interp(x_values, xp, fp)

        index_max = int(np.argmax(d))
        index_min = int(np.argmin(d))
        d_rot = _rotate(d, index_max)
        rad_rot = _rotate(rad, index_max)
        index_min_rot = int(np.where(d_rot == d[index_min])[0][0])

        d_side1 = d_rot[:index_min_rot][::-1]
        rad_side1 = rad_rot[:index_min_rot][::-1]
        d_side2 = d_rot[index_min_rot:]
        rad_side2 = rad_rot[index_min_rot:]

        inc1 = len(d_side1) > 1 and bool(d_side1[1] > d_side1[0])
        inc2 = len(d_side2) > 1 and bool(d_side2[1] > d_side2[0])
        d_side1_new, rad_side1_new = _keep_strictly_monotonic(d_side1, rad_side1, increasing=inc1)
        d_side2_new, rad_side2_new = _keep_strictly_monotonic(d_side2, rad_side2, increasing=inc2)

        d_all_sorted = np.sort(np.concatenate([d_side1_new, d_side2_new]))
        d_min = float(d_all_sorted[1])
        d_max = float(d_all_sorted[-2])
        d_interp = np.linspace(d_min, d_max, n_samples)
        xy_interp = mean_xy[None, :] + d_interp[:, None] * major[None, :]

        rad1_interp = _interp_strict(d_side1_new, rad_side1_new, d_interp)
        rad2_interp = _interp_strict(d_side2_new, rad_side2_new, d_interp)
        diam_interp = np.abs(rad1_interp - rad2_interp)
        diam_interp[0] = 0.5 * (diam_interp[0] + diam_interp[1])
        diam_interp[-1] = 0.5 * (diam_interp[-1] + diam_interp[-2])
        return xy_interp, diam_interp

    def _neuron_major_xy_for_contour(
        self,
        contour: tuple[_AscPoint, ...],
    ) -> np.ndarray | None:
        try:
            from neuron import h
        except ImportError:
            return None

        h.load_file("stdlib.hoc")
        h.load_file("import3d.hoc")

        helper = h.Import3d_Section(0, 1)
        xv = h.Vector([float(point.x) for point in contour])
        yv = h.Vector([float(point.y) for point in contour])
        zv = h.Vector([float(point.z) for point in contour])
        mean = helper.contourcenter(xv, yv, zv)

        pts = h.Matrix(3, int(xv.size()))
        row0 = xv.c()
        row1 = yv.c()
        row2 = zv.c()
        row0.sub(mean.x[0])
        row1.sub(mean.x[1])
        row2.sub(mean.x[2])
        pts.setrow(0, row0)
        pts.setrow(1, row1)
        pts.setrow(2, row2)

        matrix = h.Matrix(3, 3)
        for row in range(3):
            for col in range(row, 3):
                value = pts.getrow(row).mul(pts.getrow(col)).sum()
                matrix.x[row][col] = value
                matrix.x[col][row] = value

        eigenvalues = matrix.symmeig(matrix)
        major = matrix.getcol(int(eigenvalues.max_ind()))
        major_xy = np.array([float(major.x[0]), float(major.x[1])], dtype=float)
        if np.allclose(major_xy, 0.0):
            return None
        return major_xy / np.linalg.norm(major_xy)

    def _contour2centroid(
        self,
        contour: tuple[_AscPoint, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy, diameters = self._soma_axis_sampling(contour, n_samples=21)
        mean, _, _, _ = self._contourcenter(contour)
        z_value = float(contour[0].z) if contour else 0.0
        points = np.column_stack([xy, np.full(len(diameters), z_value, dtype=float)])
        radii = 0.5 * np.asarray(diameters, dtype=float)
        return points, radii, mean

    def _approximate_contour_by_circle(
        self,
        contour: tuple[_AscPoint, ...],
        *,
        num: int = 101,
    ) -> tuple[np.ndarray, float]:
        center, x_new, y_new, z_new = self._contourcenter(contour, num=num)
        xyz = np.array([point.xyz for point in contour], dtype=float)
        perimeter = float(np.sum(np.linalg.norm(np.roll(xyz, -1, axis=0) - xyz, axis=1)))
        resampled = np.stack([x_new, y_new, z_new], axis=1)
        mean_radius = float(np.mean(np.linalg.norm(resampled - center[None, :], axis=1)))
        diameter = mean_radius + perimeter / (2.0 * np.pi)
        return center, diameter

    def _contourstack2centroid(
        self,
        stack: tuple[tuple[_AscPoint, ...], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        points = []
        radii = []
        for contour in stack:
            center, diameter = self._approximate_contour_by_circle(contour)
            points.append(center)
            radii.append(0.5 * float(diameter))
        return np.asarray(points, dtype=float), np.asarray(radii, dtype=float)

    def _soma_stack_center(
        self,
        stack: tuple[tuple[_AscPoint, ...], ...],
    ) -> np.ndarray:
        centers = np.asarray([self._contourcenter(contour)[0] for contour in stack], dtype=float)
        if len(centers) == 1:
            return centers[0]

        lengths = [0.0]
        total_length = 0.0
        for index in range(1, len(centers)):
            total_length += float(np.linalg.norm(centers[index] - centers[index - 1]))
            lengths.append(total_length)

        if total_length <= 0.0:
            return centers[0]

        target = 0.5 * total_length
        for index in range(1, len(lengths)):
            if lengths[index] > target:
                fraction = (target - lengths[index - 1]) / (lengths[index] - lengths[index - 1])
                return fraction * centers[index] + (1.0 - fraction) * centers[index - 1]
        return centers[-1]

    def _synthetic_soma_branch(self, *, center: np.ndarray, radius: float) -> Branch:
        offset = np.array([radius, 0.0, 0.0], dtype=float)
        points = np.array((center - offset, center, center + offset), dtype=float) * u.um
        radii = np.array((radius, radius, radius), dtype=float) * u.um
        return Soma.from_points(points=points, radii=radii)

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
