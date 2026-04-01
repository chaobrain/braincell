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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SWC_TYPE_MAP = {
    0: "custom",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
}
MIN_SYNTHETIC_LENGTH_UM = 1e-6


def map_swc_type_code(type_code: int) -> str:
    return SWC_TYPE_MAP.get(type_code, "custom")


@dataclass(frozen=True)
class SwcReadOptions:
    standardize_safe_fixes: bool = True
    unknown_type_as_custom: bool = True
    require_root_type_soma: bool = False


@dataclass(frozen=True)
class SwcIssue:
    level: str
    code: str
    message: str
    line_number: int | None = None
    node_id: int | None = None
    fix_message: str | None = None
    fix_applied: bool = False

    def format_block(self) -> str:
        lines = [f"[{self.level.upper()}] {self.code}"]
        location_parts = []
        if self.line_number is not None:
            location_parts.append(f"line={self.line_number}")
        if self.node_id is not None:
            location_parts.append(f"node={self.node_id}")
        if location_parts:
            lines.append(", ".join(location_parts))
        lines.append(self.message)
        if self.fix_message:
            suffix = " (applied)" if self.fix_applied else ""
            lines.append(f"fix: {self.fix_message}{suffix}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_block()


@dataclass
class SwcReport:
    issues: list[SwcIssue] = field(default_factory=list)

    def add_error(
        self,
        code: str,
        message: str,
        *,
        line_number: int | None = None,
        node_id: int | None = None,
        fix_message: str | None = None,
        fix_applied: bool = False,
    ) -> None:
        self.issues.append(
            SwcIssue(
                level="error",
                code=code,
                message=message,
                line_number=line_number,
                node_id=node_id,
                fix_message=fix_message,
                fix_applied=fix_applied,
            )
        )

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        line_number: int | None = None,
        node_id: int | None = None,
        fix_message: str | None = None,
        fix_applied: bool = False,
    ) -> None:
        self.issues.append(
            SwcIssue(
                level="warning",
                code=code,
                message=message,
                line_number=line_number,
                node_id=node_id,
                fix_message=fix_message,
                fix_applied=fix_applied,
            )
        )

    @property
    def error_count(self) -> int:
        return sum(issue.level == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.level == "warning" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        return self.warning_count > 0

    def error_messages(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues if issue.level == "error")

    def format(self, *, errors_only: bool = False) -> str:
        levels = ("error",) if errors_only else ("error", "warning")
        sections: list[str] = []
        summary_parts = []
        if self.error_count:
            summary_parts.append(f"{self.error_count} error{'s' if self.error_count != 1 else ''}")
        if not errors_only and self.warning_count:
            summary_parts.append(f"{self.warning_count} warning{'s' if self.warning_count != 1 else ''}")
        if not summary_parts:
            summary_parts.append("0 issues")
        sections.append(f"SWC report: {', '.join(summary_parts)}")

        grouped = {
            "error": [issue for issue in self.issues if issue.level == "error"],
            "warning": [issue for issue in self.issues if issue.level == "warning"],
        }
        for level in levels:
            issues = grouped[level]
            if not issues:
                continue
            title = "Errors" if level == "error" else "Warnings"
            body = "\n\n".join(issue.format_block() for issue in issues)
            sections.append(f"{title}\n{'-' * len(title)}\n{body}")
        return "\n\n".join(sections)

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class _SwcRawRow:
    fields: tuple[str, ...]
    line_number: int


@dataclass
class _SwcRow:
    line_number: int
    fields: tuple[str, ...]
    node_id: int | None = None
    type_code: int | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    radius: float | None = None
    parent_id: int | None = None


@dataclass
class _SwcContext:
    path: Path
    options: SwcReadOptions
    report: SwcReport = field(default_factory=SwcReport)
    raw_rows: list[_SwcRawRow] = field(default_factory=list)
    rows: list[_SwcRow] = field(default_factory=list)
    use_corrections: bool = True
    mark_fix_applied: bool = False
    stop_processing: bool = False
    nodes: dict[int, _SwcRow] = field(default_factory=dict)
    children: dict[int, list[int]] = field(default_factory=dict)
    root_ids: list[int] = field(default_factory=list)
    root_id: int | None = None
    contour_soma_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class _SwcAttach:
    node_id: int | None = None
    point: tuple[float, float, float] | None = None
    radius: float | None = None
    parent_x: float | None = None


@dataclass(frozen=True)
class _SwcBranch:
    point_ids: tuple[int, ...]
    branch_type: str
    parent_index: int | None
    start_node_id: int
    attach: _SwcAttach | None = None
    override_points: tuple[tuple[float, float, float], ...] | None = None
    override_radii: tuple[float, ...] | None = None
