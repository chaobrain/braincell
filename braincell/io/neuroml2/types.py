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

"""Report types of the NeuroML2 reader."""

from dataclasses import dataclass, field
from typing import ClassVar

from braincell.io._report import Issue, Report

__all__ = ["NeuroMlIssue", "NeuroMlMetadata", "NeuroMlReport"]


@dataclass(frozen=True)
class NeuroMlIssue(Issue):
    """An :class:`~braincell.io._report.Issue` raised while reading NeuroML2.

    ElementTree keeps no line numbers, so an issue names its element instead:
    the segment group or segment it concerns, when there is one.
    """

    segment_group: str | None = None
    segment_id: int | None = None

    def location_parts(self) -> list[str]:
        parts = super().location_parts()
        if self.segment_group is not None:
            parts.append(f"segmentGroup={self.segment_group}")
        if self.segment_id is not None:
            parts.append(f"segment={self.segment_id}")
        return parts


@dataclass
class NeuroMlMetadata:
    """What a NeuroML2 cell carries besides the geometry.

    Attributes
    ----------
    cell_id, morphology_id : str or None
        Ids of the ``<cell>`` read and of its ``<morphology>``.
    segment_groups : dict[str, tuple[int, ...]]
        Each segment group's segment ids after expanding ``member``,
        ``include``, ``path`` and ``subTree``, without duplicates, ascending.
    segment_branches : dict[int, tuple[tuple[str, float, float], ...]]
        Where each segment lies: ``(branch name, x0, x1)`` intervals of the
        branch's normalized length. A segment cut in two has two entries.
    segment_names : dict[int, str]
        The ``name`` attribute of each segment that has one.
    ignored_elements : tuple[str, ...]
        Names of the cell's elements that were not read, such as
        ``biophysicalProperties`` or ``notes``, sorted.
    """

    cell_id: str | None = None
    morphology_id: str | None = None
    segment_groups: dict[str, tuple[int, ...]] = field(default_factory=dict)
    segment_branches: dict[int, tuple[tuple[str, float, float], ...]] = field(default_factory=dict)
    segment_names: dict[int, str] = field(default_factory=dict)
    ignored_elements: tuple[str, ...] = ()


@dataclass
class NeuroMlReport(Report):
    """Diagnostics and side-channel metadata from one NeuroML2 cell."""

    label: ClassVar[str] = "NeuroML2"
    issue_cls: ClassVar[type[Issue]] = NeuroMlIssue

    issues: list[NeuroMlIssue] = field(default_factory=list)
    metadata: NeuroMlMetadata = field(default_factory=NeuroMlMetadata)
