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
from typing import ClassVar

from braincell.io._report import Issue, Report


@dataclass(frozen=True)
class AscIssue(Issue):
    """An :class:`~braincell.io._report.Issue` raised while reading ASC.

    Neurolucida files carry no node identifiers, so the base line-number
    location and block layout are used unchanged.
    """


@dataclass(frozen=True)
class AscSpineRecord:
    base_xyz: tuple[float, float, float]
    base_diameter: float
    tip_xyz: tuple[float, float, float]
    tip_diameter: float
    class_type: int | float | None = None
    class_label: str | None = None
    properties: tuple[tuple[object, ...], ...] = ()
    line_number: int | None = None


@dataclass
class AscMetadata:
    spines: list[AscSpineRecord] = field(default_factory=list)
    spine_annotations: list[object] = field(default_factory=list)
    markers: list[object] = field(default_factory=list)
    filled_circles: list[object] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    colors: list[object] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)


@dataclass
class AscReport(Report):
    """Diagnostics and side-channel metadata from one ASC file."""

    label: ClassVar[str] = "ASC"
    issue_cls: ClassVar[type[Issue]] = AscIssue

    issues: list[AscIssue] = field(default_factory=list)
    metadata: AscMetadata = field(default_factory=AscMetadata)
