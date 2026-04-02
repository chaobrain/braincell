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

from .types import AscMetadata, AscReport
from ..swc.types import MIN_SYNTHETIC_LENGTH_UM
from ..._misc import u
from ...morpho import Branch, Morpho, MorphoBranch
from ...morpho.branch import Soma, branch_class_for_type

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
    radius: float
    line_number: int

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


@dataclass(frozen=True)
class _AscSegment:
    points: tuple[_AscPoint, ...]
    children: tuple["_AscSegment", ...]
    branch_type: str


@dataclass(frozen=True)
class AscReader:
    def read(self, path: str | PathLike[str], return_report: bool = False):
        source_path = Path(path)
        report = AscReport()
        try:
            expressions = self._parse_document(source_path.read_text(), report)
            contours, neurites = self._extract_blocks(expressions, report)
            morpho = self._build_morpho(contours, neurites, report, path=source_path)
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
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
            while end < len(text) and text[end] not in "()|;, \t\r\n":
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
            if not children:
                return None
            if len(children) == 1:
                child = children[0]
                merged = _AscSegment(
                    points=self._merge_point_sequences(points, child.points),
                    children=child.children,
                    branch_type=segment.branch_type,
                )
                return self._normalize_segment(merged)

            pushed_children = []
            for child in children:
                merged = _AscSegment(
                    points=self._merge_point_sequences(points, child.points),
                    children=child.children,
                    branch_type=segment.branch_type,
                )
                normalized = self._normalize_segment(merged)
                if normalized is not None:
                    pushed_children.append(normalized)
            if len(pushed_children) == 1:
                return pushed_children[0]
            return _AscSegment(points=(), children=tuple(pushed_children), branch_type=segment.branch_type)

        return _AscSegment(points=points, children=children, branch_type=segment.branch_type)

    def _build_morpho(
        self,
        contours: tuple[tuple[_AscPoint, ...], ...],
        neurites: tuple[_AscSegment, ...],
        report: AscReport,
        *,
        path: Path,
    ) -> Morpho:
        if not contours and not neurites:
            raise ValueError(f"ASC import failed for {path}: no soma contour or neurites were found.")

        if contours:
            all_points = [point for contour in contours for point in contour]
            center, radius = self._contour_center_radius(tuple(all_points))
        else:
            first_point = self._first_point(neurites)
            if first_point is None:
                raise ValueError(f"ASC import failed for {path}: no geometry points were found.")
            center = first_point.xyz
            radius = max(float(first_point.radius), MIN_SYNTHETIC_LENGTH_UM)
            report.add_warning("topology.synthetic_soma",
                               "ASC file has no CellBody contour; synthesized a soma from the first neurite root point.")

        soma_branch = self._synthetic_soma_branch(center=center, radius=radius)
        morpho = Morpho.from_root(soma_branch, name="soma")
        for neurite in neurites:
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

            attach_radius_for_child = radii[0] if parent_branch_type == "soma" else float(attach_radius)
            if should_copy_attach:
                points.insert(0, np.asarray(attach_point, dtype=float))
                radii.insert(0, attach_radius_for_child)
            elif np.allclose(points[0], attach_point):
                radii[0] = attach_radius_for_child

        points, radii = self._dedupe_shared_points(points, radii)
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
        return np.allclose(left.xyz, right.xyz) and np.isclose(left.radius, right.radius)

    def _first_point(self, segments: tuple[_AscSegment, ...]) -> _AscPoint | None:
        for segment in segments:
            if segment.points:
                return segment.points[0]
            first = self._first_point(segment.children)
            if first is not None:
                return first
        return None

    def _contour_center_radius(self, points: tuple[_AscPoint, ...]) -> tuple[np.ndarray, float]:
        xyz = np.array([point.xyz for point in points], dtype=float)
        center = xyz.mean(axis=0)
        radius = max(np.linalg.norm(point.xyz - center) + float(point.radius) for point in points)
        return center, max(float(radius), MIN_SYNTHETIC_LENGTH_UM)

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
        if not isinstance(expr, tuple):
            return
        if self._is_point_expr(expr):
            return

        key = self._head_key(expr)
        if key == "color":
            metadata.colors.append(expr)
        elif key == "spine":
            metadata.spines.append(expr)
        elif key == "marker":
            metadata.markers.append(expr)
        elif key == "filledcircle":
            metadata.filled_circles.append(expr)

        for item in expr:
            self._collect_metadata(item, metadata)

    def _is_annotation(self, expr: object) -> bool:
        if isinstance(expr, str):
            return self._normalize_name(expr) in _IGNORED_SYMBOLS
        if not isinstance(expr, tuple):
            return False
        if self._is_point_expr(expr):
            return False
        key = self._head_key(expr)
        return key in _ANNOTATION_KEYS or key in {"spine", "marker", "filledcircle"}

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
        assert isinstance(expr, tuple)
        return _AscPoint(
            x=float(expr[0]),
            y=float(expr[1]),
            z=float(expr[2]),
            radius=max(float(expr[3]), 0.0),
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
