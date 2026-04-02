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



from pathlib import Path

from .soma import is_contour_soma
from .types import SWC_TYPE_MAP, SwcReport, _SwcContext, _SwcRawRow, _SwcRow


def apply_swc_rules(context: _SwcContext) -> None:
    for rule in SWC_RULES:
        if context.stop_processing:
            break
        rule(context)


def raise_for_swc_errors(report: SwcReport, path: Path) -> None:
    if not report.has_errors:
        return
    raise ValueError(f"SWC validation failed for {path}:\n\n{report.format(errors_only=True)}")


def _add_warning(
    context: _SwcContext,
    code: str,
    message: str,
    *,
    line_number: int | None = None,
    node_id: int | None = None,
    fix_message: str | None = None,
) -> None:
    context.report.add_warning(
        code,
        message,
        line_number=line_number,
        node_id=node_id,
        fix_message=fix_message,
        fix_applied=bool(fix_message) and context.mark_fix_applied and context.use_corrections,
    )


def _add_error(
    context: _SwcContext,
    code: str,
    message: str,
    *,
    line_number: int | None = None,
    node_id: int | None = None,
    fix_message: str | None = None,
) -> None:
    context.report.add_error(
        code,
        message,
        line_number=line_number,
        node_id=node_id,
        fix_message=fix_message,
        fix_applied=bool(fix_message) and context.mark_fix_applied and context.use_corrections,
    )


def _set_attr(context: _SwcContext, row: _SwcRow, attr: str, value) -> None:
    if context.use_corrections:
        setattr(row, attr, value)


def _parse_integer_token(token: str) -> tuple[int | None, bool]:
    normalized = token.strip()
    try:
        numeric = float(normalized)
    except ValueError:
        return None, False
    if not numeric.is_integer():
        return None, False
    value = int(numeric)
    return value, normalized != str(value)


def _parse_float_token(token: str) -> tuple[float | None, bool]:
    normalized = token.strip()
    if normalized.lower() in {"n/a", "na", "nan"}:
        return None, True
    try:
        numeric = float(normalized)
    except ValueError:
        return None, False
    if numeric != numeric:
        return None, True
    return numeric, False


def rule_missing_field_columns(context: _SwcContext) -> None:
    bad_rows = False
    context.rows = []
    for raw_row in context.raw_rows:
        if len(raw_row.fields) < 7:
            _add_error(
                context,
                "format.column_count",
                f"SWC line {raw_row.line_number} must have at least 7 columns, got {len(raw_row.fields)}.",
                line_number=raw_row.line_number,
            )
            bad_rows = True
            continue
        context.rows.append(_SwcRow(line_number=raw_row.line_number, fields=tuple(raw_row.fields[:7])))
    if bad_rows:
        context.stop_processing = True


def rule_tree_sample_count(context: _SwcContext) -> None:
    sample_count = len(context.rows)
    if sample_count == 0:
        _add_error(context, "format.empty_file", "SWC file is empty.")
        context.stop_processing = True
    elif sample_count < 20:
        _add_warning(
            context,
            "format.low_sample_count",
            f"SWC contains only {sample_count} samples; morphology may be underspecified.",
        )


def rule_itp_int(context: _SwcContext) -> None:
    has_fatal = False
    seen_ids: set[int] = set()
    for row in context.rows:
        node_token, type_token, parent_token = row.fields[0], row.fields[1], row.fields[6]

        node_id, node_corrected = _parse_integer_token(node_token)
        if node_id is None or node_id <= 0:
            _add_error(
                context,
                "identity.invalid_id",
                f"SWC index must be a positive integer, got {node_token!r}.",
                line_number=row.line_number,
            )
            has_fatal = True
        else:
            if node_corrected:
                _add_warning(
                    context,
                    "format.index_int",
                    f"SWC index {node_token!r} was normalized to integer {node_id}.",
                    line_number=row.line_number,
                    node_id=node_id,
                    fix_message=f"normalize index to {node_id}",
                )
            row.node_id = node_id
            if node_id in seen_ids:
                _add_error(
                    context,
                    "identity.duplicate_id",
                    f"Duplicate SWC node id {node_id!r}.",
                    line_number=row.line_number,
                    node_id=node_id,
                )
                has_fatal = True
            else:
                seen_ids.add(node_id)

        type_code, type_corrected = _parse_integer_token(type_token)
        if type_code is None:
            _add_warning(
                context,
                "format.type_int",
                f"SWC type id {type_token!r} is not an integer; using 0 (custom).",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="set type id to 0",
            )
            _set_attr(context, row, "type_code", 0)
        else:
            if type_corrected:
                _add_warning(
                    context,
                    "format.type_int",
                    f"SWC type id {type_token!r} was normalized to integer {type_code}.",
                    line_number=row.line_number,
                    node_id=row.node_id,
                    fix_message=f"normalize type id to {type_code}",
                )
            row.type_code = type_code

        parent_id, parent_corrected = _parse_integer_token(parent_token)
        if parent_id is None:
            _add_error(
                context,
                "identity.invalid_parent_id",
                f"SWC parent index must be an integer, got {parent_token!r}.",
                line_number=row.line_number,
                node_id=row.node_id,
            )
            has_fatal = True
        else:
            if parent_corrected:
                _add_warning(
                    context,
                    "format.parent_int",
                    f"SWC parent index {parent_token!r} was normalized to integer {parent_id}.",
                    line_number=row.line_number,
                    node_id=row.node_id,
                    fix_message=f"normalize parent index to {parent_id}",
                )
            row.parent_id = parent_id

    if has_fatal:
        context.stop_processing = True


def rule_xyz_double(context: _SwcContext) -> None:
    for row in context.rows:
        for attr, token in (("x", row.fields[2]), ("y", row.fields[3]), ("z", row.fields[4])):
            value, placeholder = _parse_float_token(token)
            if value is None:
                _add_warning(
                    context,
                    "geometry.xyz_double",
                    f"SWC {attr}-coordinate {token!r} was replaced with 0.0.",
                    line_number=row.line_number,
                    node_id=row.node_id,
                    fix_message=f"set {attr} to 0.0",
                )
                _set_attr(context, row, attr, 0.0)
            else:
                setattr(row, attr, value)


def rule_radius_positive_double(context: _SwcContext) -> None:
    for row in context.rows:
        value, placeholder = _parse_float_token(row.fields[5])
        if value is None or value <= 0.0:
            _add_warning(
                context,
                "geometry.radius_positive_double",
                f"SWC radius {row.fields[5]!r} was replaced with 0.5.",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="set radius to 0.5",
            )
            _set_attr(context, row, "radius", 0.5)
        else:
            row.radius = value


def rule_nonstandard_type_id(context: _SwcContext) -> None:
    for row in context.rows:
        if row.type_code in SWC_TYPE_MAP:
            continue
        message = f"SWC type code {row.type_code!r} is not mapped."
        if context.options.unknown_type_as_custom:
            _add_warning(
                context,
                "semantics.unknown_type",
                f"{message} Using 0 (custom).",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="set type id to 0",
            )
            _set_attr(context, row, "type_code", 0)
        else:
            _add_error(
                context,
                "semantics.unknown_type",
                message,
                line_number=row.line_number,
                node_id=row.node_id,
            )


def rule_invalid_parent_index(context: _SwcContext) -> None:
    node_ids = {row.node_id for row in context.rows if row.node_id is not None}
    for row in context.rows:
        if row.parent_id is None:
            continue
        if row.parent_id == 0:
            _add_warning(
                context,
                "topology.invalid_parent",
                "SWC parent index 0 is invalid and was treated as root (-1).",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="set parent index to -1",
            )
            _set_attr(context, row, "parent_id", -1)
            continue
        if row.parent_id == -1:
            continue
        if row.parent_id == row.node_id or row.parent_id not in node_ids:
            _add_warning(
                context,
                "topology.invalid_parent",
                f"SWC parent index {row.parent_id!r} is invalid and was replaced with -1.",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="set parent index to -1",
            )
            _set_attr(context, row, "parent_id", -1)


def rule_duplicate_xyzr_parent_child(context: _SwcContext) -> None:
    if not context.use_corrections:
        rows_by_id = {row.node_id: row for row in context.rows if row.node_id is not None}
        for row in context.rows:
            if row.node_id is None or row.parent_id in (None, -1):
                continue
            parent = rows_by_id.get(row.parent_id)
            if parent is None:
                continue
            if (row.x, row.y, row.z, row.radius) != (parent.x, parent.y, parent.z, parent.radius):
                continue
            _add_warning(
                context,
                "geometry.duplicate_xyzr_node",
                f"SWC node {row.node_id} duplicates parent {parent.node_id} in xyzr and would be merged into the parent.",
                line_number=row.line_number,
                node_id=row.node_id,
                fix_message="merge duplicate xyzr node into parent",
            )
        return

    while True:
        rows_by_id = {row.node_id: row for row in context.rows if row.node_id is not None}
        changed = False
        for row in list(context.rows):
            if row.node_id is None or row.parent_id in (None, -1):
                continue
            parent = rows_by_id.get(row.parent_id)
            if parent is None:
                continue
            if (row.x, row.y, row.z, row.radius) != (parent.x, parent.y, parent.z, parent.radius):
                continue
            duplicate_id = row.node_id
            keep_id = parent.node_id
            _add_warning(
                context,
                "geometry.duplicate_xyzr_node",
                f"SWC node {duplicate_id} duplicates parent {keep_id} in xyzr and was merged into the parent.",
                line_number=row.line_number,
                node_id=duplicate_id,
                fix_message="merge duplicate xyzr node into parent",
            )
            for candidate in context.rows:
                if candidate.parent_id == duplicate_id:
                    candidate.parent_id = keep_id
            context.rows.remove(row)
            changed = True
            break
        if not changed:
            return


def rule_no_soma_samples(context: _SwcContext) -> None:
    if any(row.type_code == 1 for row in context.rows):
        return
    _add_warning(context, "semantics.no_soma_samples", "SWC contains no soma samples.")


def rule_contour(context: _SwcContext) -> None:
    soma_rows = [row for row in context.rows if row.type_code == 1]
    if not soma_rows:
        return
    if is_contour_soma(soma_rows):
        context.contour_soma_ids = {row.node_id for row in soma_rows if row.node_id is not None}
        _add_warning(
            context,
            "semantics.contour",
            "Soma samples were interpreted as a contour and will be converted to an equivalent cylinder.",
        )


def rule_sorted_index_order(context: _SwcContext) -> None:
    if not context.rows:
        return
    sorted_rows = sorted(context.rows, key=lambda row: (row.parent_id != -1, row.parent_id or -1, row.node_id or -1))
    if [row.line_number for row in sorted_rows] != [row.line_number for row in context.rows]:
        _add_warning(
            context,
            "topology.sorted_order",
            "SWC rows were reordered so parents appear before children.",
            fix_message="sort rows into parent-before-child order",
        )
        if context.use_corrections:
            context.rows = sorted_rows


def rule_index_sequential(context: _SwcContext) -> None:
    expected_ids = list(range(1, len(context.rows) + 1))
    actual_ids = [row.node_id for row in context.rows]
    if actual_ids == expected_ids:
        return
    old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate((row.node_id for row in context.rows), start=1)
        if old_id is not None
    }
    _add_warning(
        context,
        "identity.sequential_index",
        "SWC node ids were renumbered to a sequential 1..N scheme.",
        fix_message="renumber indices sequentially",
    )
    if not context.use_corrections:
        return
    contour_old_to_new = {
        old_id: new_id
        for new_id, old_id in enumerate((row.node_id for row in context.rows), start=1)
        if old_id is not None and old_id in context.contour_soma_ids
    }
    for new_id, row in enumerate(context.rows, start=1):
        old_id = row.node_id
        row.node_id = new_id
        if row.parent_id == -1:
            continue
        row.parent_id = old_to_new.get(row.parent_id, -1)
    if context.contour_soma_ids:
        context.contour_soma_ids = set(contour_old_to_new.values())


def rule_root_parent_index(context: _SwcContext) -> None:
    root_rows = [row for row in context.rows if row.parent_id == -1]
    if len(root_rows) != 1:
        _add_error(
            context,
            "topology.root_count",
            f"SWC must contain exactly one root after corrections, got {len(root_rows)}.",
        )
    elif context.options.require_root_type_soma and root_rows[0].type_code != 1:
        _add_error(
            context,
            "topology.root_type",
            "SWC root must have soma type (1).",
            line_number=root_rows[0].line_number,
            node_id=root_rows[0].node_id,
        )


def rule_tree_integrity(context: _SwcContext) -> None:
    rows_by_id = {row.node_id: row for row in context.rows if row.node_id is not None}
    if not rows_by_id:
        return

    child_map = {node_id: [] for node_id in rows_by_id}
    for row in rows_by_id.values():
        if row.parent_id == -1:
            continue
        if row.parent_id not in child_map:
            _add_error(
                context,
                "topology.orphan",
                f"SWC node {row.node_id} references missing parent {row.parent_id}.",
                line_number=row.line_number,
                node_id=row.node_id,
            )
            continue
        child_map[row.parent_id].append(row.node_id)

    roots = [row.node_id for row in rows_by_id.values() if row.parent_id == -1]
    if len(roots) != 1:
        return

    root_id = roots[0]
    visited: set[int] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(child_map[node_id])

    if len(visited) != len(rows_by_id):
        _add_error(
            context,
            "topology.disconnected",
            "SWC graph is disconnected after corrections.",
        )


SWC_RULES = (
    rule_missing_field_columns,
    rule_tree_sample_count,
    rule_itp_int,
    rule_xyz_double,
    rule_radius_positive_double,
    rule_nonstandard_type_id,
    rule_invalid_parent_index,
    rule_duplicate_xyzr_parent_child,
    rule_no_soma_samples,
    rule_contour,
    rule_sorted_index_order,
    rule_index_sequential,
    rule_root_parent_index,
    rule_tree_integrity,
)
