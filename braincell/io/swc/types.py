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


from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from braincell.io._report import Issue, Report

SWC_TYPE_MAP = {
    0: "custom",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
}
MIN_SYNTHETIC_LENGTH_UM = 1e-6
SWC_IMPORT_MODES = ("neuron", "neuromorpho")


def map_swc_type_code(type_code: int) -> str:
    return SWC_TYPE_MAP.get(type_code, "custom")


@dataclass(frozen=True)
class SwcReadOptions:
    standardize_safe_fixes: bool = True
    unknown_type_as_custom: bool = True
    require_root_type_soma: bool = False
    mode: str = "neuron"

    def __post_init__(self) -> None:
        if self.mode not in SWC_IMPORT_MODES:
            raise ValueError(f"mode must be one of {SWC_IMPORT_MODES!r}, got {self.mode!r}.")


@dataclass(frozen=True)
class SwcIssue(Issue):
    """An :class:`~braincell.io._report.Issue` carrying SWC node context.

    Adds the offending ``node_id`` to the block header and, when the rule
    offers a repair, a trailing ``fix:`` line marked ``(applied)`` if the
    reader actually performed it.
    """

    node_id: int | None = None
    fix_message: str | None = None
    fix_applied: bool = False

    def location_parts(self) -> list[str]:
        parts = super().location_parts()
        if self.node_id is not None:
            parts.append(f"node={self.node_id}")
        return parts

    def trailing_lines(self) -> list[str]:
        if not self.fix_message:
            return []
        suffix = " (applied)" if self.fix_applied else ""
        return [f"fix: {self.fix_message}{suffix}"]


@dataclass
class SwcReport(Report):
    """Diagnostics collected while reading one SWC file."""

    label: ClassVar[str] = "SWC"
    issue_cls: ClassVar[type[Issue]] = SwcIssue

    issues: list[SwcIssue] = field(default_factory=list)


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
