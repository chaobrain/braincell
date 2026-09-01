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

"""Format-neutral diagnostic report shared by the morphology readers.

Every reader accumulates issues while parsing and renders them the same way:
a one-line summary, then an ``Errors`` section and a ``Warnings`` section of
blocks. Keeping that rendering here means a new reader inherits it instead of
forking it, and the SWC and ASC reports cannot drift apart again.

Subclasses supply the format label and the concrete issue type; issue
subclasses extend the block layout through :meth:`Issue.location_parts` and
:meth:`Issue.trailing_lines`.
"""

from dataclasses import dataclass, field
from typing import ClassVar

__all__ = ["Issue", "Report"]


@dataclass(frozen=True)
class Issue:
    """One diagnostic emitted while reading a morphology file.

    Parameters
    ----------
    level : str
        Either ``"error"`` or ``"warning"``.
    code : str
        Stable machine-readable identifier, e.g. ``"geometry.degenerate_branch"``.
    message : str
        Human-readable description.
    line_number : int or None
        1-based line in the source file, when known.
    """

    level: str
    code: str
    message: str
    line_number: int | None = None

    def location_parts(self) -> list[str]:
        """Return the comma-joined location fragments for the block header.

        Returns
        -------
        list of str
        """
        return [] if self.line_number is None else [f"line={self.line_number}"]

    def trailing_lines(self) -> list[str]:
        """Return extra lines rendered after the message.

        Returns
        -------
        list of str
        """
        return []

    def format_block(self) -> str:
        """Render this issue as a multi-line block.

        Returns
        -------
        str
        """
        lines = [f"[{self.level.upper()}] {self.code}"]
        location_parts = self.location_parts()
        if location_parts:
            lines.append(", ".join(location_parts))
        lines.append(self.message)
        lines.extend(self.trailing_lines())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_block()


@dataclass
class Report:
    """Ordered collection of :class:`Issue` records for one source file.

    Subclasses set :attr:`label` (the format name shown in the summary line)
    and :attr:`issue_cls` (the concrete issue type :meth:`add` constructs).

    Attributes
    ----------
    issues : list of Issue
        Every issue recorded, in the order it was found.
    """

    #: Format name used in the summary line, e.g. ``"SWC"``.
    label: ClassVar[str] = ""
    #: Concrete issue type constructed by :meth:`add`.
    issue_cls: ClassVar[type[Issue]] = Issue

    issues: list[Issue] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, **kwargs) -> None:
        """Append an issue of ``level``.

        Parameters
        ----------
        level : str
            Either ``"error"`` or ``"warning"``.
        code : str
            Stable machine-readable identifier.
        message : str
            Human-readable description.
        **kwargs : Any
            Extra fields accepted by :attr:`issue_cls`, such as
            ``line_number`` or ``node_id``.
        """
        self.issues.append(self.issue_cls(level=level, code=code, message=message, **kwargs))

    def add_error(self, code: str, message: str, **kwargs) -> None:
        """Append an error. See :meth:`add` for the parameters."""
        self.add("error", code, message, **kwargs)

    def add_warning(self, code: str, message: str, **kwargs) -> None:
        """Append a warning. See :meth:`add` for the parameters."""
        self.add("warning", code, message, **kwargs)

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
        """Return the message of every error, in order.

        Returns
        -------
        tuple of str
        """
        return tuple(issue.message for issue in self.issues if issue.level == "error")

    def format(self, *, errors_only: bool = False) -> str:
        """Render the summary line and the per-level issue sections.

        Parameters
        ----------
        errors_only : bool
            Omit warnings from both the summary and the sections.

        Returns
        -------
        str
        """
        summary_parts = []
        if self.error_count:
            summary_parts.append(f"{self.error_count} error{'s' if self.error_count != 1 else ''}")
        if not errors_only and self.warning_count:
            summary_parts.append(f"{self.warning_count} warning{'s' if self.warning_count != 1 else ''}")
        if not summary_parts:
            summary_parts.append("0 issues")
        sections = [f"{self.label} report: {', '.join(summary_parts)}"]

        levels = ("error",) if errors_only else ("error", "warning")
        for level in levels:
            issues = [issue for issue in self.issues if issue.level == level]
            if not issues:
                continue
            title = "Errors" if level == "error" else "Warnings"
            body = "\n\n".join(issue.format_block() for issue in issues)
            sections.append(f"{title}\n{'-' * len(title)}\n{body}")
        return "\n\n".join(sections)

    def __str__(self) -> str:
        return self.format()
