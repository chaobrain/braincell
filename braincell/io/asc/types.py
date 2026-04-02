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


@dataclass(frozen=True)
class AscIssue:
    level: str
    code: str
    message: str
    line_number: int | None = None

    def format_block(self) -> str:
        lines = [f"[{self.level.upper()}] {self.code}"]
        if self.line_number is not None:
            lines.append(f"line={self.line_number}")
        lines.append(self.message)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format_block()


@dataclass
class AscMetadata:
    spines: list[object] = field(default_factory=list)
    markers: list[object] = field(default_factory=list)
    filled_circles: list[object] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)
    colors: list[object] = field(default_factory=list)
    source_labels: list[str] = field(default_factory=list)


@dataclass
class AscReport:
    issues: list[AscIssue] = field(default_factory=list)
    metadata: AscMetadata = field(default_factory=AscMetadata)

    def add_error(self, code: str, message: str, *, line_number: int | None = None) -> None:
        self.issues.append(AscIssue(level="error", code=code, message=message, line_number=line_number))

    def add_warning(self, code: str, message: str, *, line_number: int | None = None) -> None:
        self.issues.append(AscIssue(level="warning", code=code, message=message, line_number=line_number))

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

    def format(self, *, errors_only: bool = False) -> str:
        levels = ("error",) if errors_only else ("error", "warning")
        parts: list[str] = []
        summary: list[str] = []
        if self.error_count:
            summary.append(f"{self.error_count} error{'s' if self.error_count != 1 else ''}")
        if not errors_only and self.warning_count:
            summary.append(f"{self.warning_count} warning{'s' if self.warning_count != 1 else ''}")
        if not summary:
            summary.append("0 issues")
        parts.append(f"ASC report: {', '.join(summary)}")

        for level in levels:
            issues = [issue for issue in self.issues if issue.level == level]
            if not issues:
                continue
            title = "Errors" if level == "error" else "Warnings"
            parts.append(f"{title}\n{'-' * len(title)}\n" + "\n\n".join(issue.format_block() for issue in issues))
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return self.format()
