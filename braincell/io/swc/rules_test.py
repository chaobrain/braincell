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

"""Tests for :mod:`braincell.io.swc.rules`.

``reader_test.py`` drives these rules end-to-end through :class:`SwcReader`,
which is the right home for behaviour observed in the parsed morphology.
This file covers each rule's own contract instead: the exact issue code it
emits, whether it halts the pipeline, and what it does to the row when
corrections are disabled.

The rules are pure functions over a :class:`_SwcContext`, which is a plain
dataclass, so every case here is built directly rather than round-tripped
through a temporary file.
"""

import unittest
from pathlib import Path

from braincell.io.swc.rules import (
    SWC_RULES,
    _add_error,
    _add_warning,
    _parse_float_token,
    _parse_integer_token,
    _set_attr,
    apply_swc_rules,
    raise_for_swc_errors,
    rule_contour,
    rule_duplicate_xyzr_parent_child,
    rule_index_sequential,
    rule_invalid_parent_index,
    rule_itp_int,
    rule_missing_field_columns,
    rule_no_soma_samples,
    rule_nonstandard_type_id,
    rule_radius_positive_double,
    rule_root_parent_index,
    rule_sorted_index_order,
    rule_tree_integrity,
    rule_tree_sample_count,
    rule_xyz_double,
)
from braincell.io.swc.types import (
    SwcReadOptions,
    SwcReport,
    _SwcContext,
    _SwcRawRow,
    _SwcRow,
)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _context(*, rows=(), raw_rows=(), options=None, use_corrections=True, mark_fix_applied=False) -> _SwcContext:
    """Assemble a context directly, skipping the file-reading front end."""
    context = _SwcContext(path=Path("memory.swc"), options=options or SwcReadOptions())
    context.raw_rows = list(raw_rows)
    context.rows = list(rows)
    context.use_corrections = use_corrections
    context.mark_fix_applied = mark_fix_applied
    return context


def _raw(line: str, line_number: int) -> _SwcRawRow:
    """A raw row from a whitespace-separated SWC line."""
    return _SwcRawRow(fields=tuple(line.split()), line_number=line_number)


def _unparsed(*fields, line_number: int = 1) -> _SwcRow:
    """A row carrying only its raw tokens, as the column rule leaves it."""
    return _SwcRow(line_number=line_number, fields=tuple(str(f) for f in fields))


def _parsed(node_id, type_code, x, y, z, radius, parent_id, *, line_number=None) -> _SwcRow:
    """A fully parsed row, as the rules downstream of ``rule_itp_int`` see it."""
    return _SwcRow(
        line_number=line_number if line_number is not None else node_id,
        fields=(str(node_id), str(type_code), str(x), str(y), str(z), str(radius), str(parent_id)),
        node_id=node_id,
        type_code=type_code,
        x=float(x),
        y=float(y),
        z=float(z),
        radius=float(radius),
        parent_id=parent_id,
    )


def _codes(context: _SwcContext) -> list[str]:
    return [issue.code for issue in context.report.issues]


# ---------------------------------------------------------------------------
# Token parsing helpers
# ---------------------------------------------------------------------------


class ParseIntegerTokenTest(unittest.TestCase):
    def test_plain_integer_needs_no_correction(self) -> None:
        self.assertEqual(_parse_integer_token("3"), (3, False))

    def test_surrounding_whitespace_is_stripped_without_correction(self) -> None:
        self.assertEqual(_parse_integer_token("  3  "), (3, False))

    def test_float_spelling_of_an_integer_is_flagged_as_corrected(self) -> None:
        self.assertEqual(_parse_integer_token("3.0"), (3, True))

    def test_negative_integer_is_accepted(self) -> None:
        self.assertEqual(_parse_integer_token("-1"), (-1, False))

    def test_non_integral_float_is_rejected(self) -> None:
        self.assertEqual(_parse_integer_token("3.5"), (None, False))

    def test_non_numeric_token_is_rejected(self) -> None:
        self.assertEqual(_parse_integer_token("abc"), (None, False))
        self.assertEqual(_parse_integer_token(""), (None, False))


class ParseFloatTokenTest(unittest.TestCase):
    def test_plain_float_needs_no_correction(self) -> None:
        self.assertEqual(_parse_float_token("1.5"), (1.5, False))

    def test_scientific_notation_is_accepted(self) -> None:
        value, placeholder = _parse_float_token("1e-3")
        self.assertAlmostEqual(value, 0.001)
        self.assertFalse(placeholder)

    def test_placeholder_spellings_are_recognised(self) -> None:
        for token in ("n/a", "N/A", "na", "nan", "NaN"):
            with self.subTest(token=token):
                self.assertEqual(_parse_float_token(token), (None, True))

    def test_non_numeric_token_is_not_a_placeholder(self) -> None:
        self.assertEqual(_parse_float_token("abc"), (None, False))


# ---------------------------------------------------------------------------
# Issue recording
# ---------------------------------------------------------------------------


class IssueRecordingTest(unittest.TestCase):
    """``fix_applied`` is a three-way conjunction; cover each falsifying leg."""

    def test_fix_is_marked_applied_when_all_three_conditions_hold(self) -> None:
        context = _context(use_corrections=True, mark_fix_applied=True)

        _add_warning(context, "some.code", "message", fix_message="do the thing")

        self.assertTrue(context.report.issues[0].fix_applied)

    def test_fix_is_not_applied_when_the_context_is_not_marking(self) -> None:
        context = _context(use_corrections=True, mark_fix_applied=False)

        _add_warning(context, "some.code", "message", fix_message="do the thing")

        self.assertFalse(context.report.issues[0].fix_applied)

    def test_fix_is_not_applied_when_corrections_are_disabled(self) -> None:
        context = _context(use_corrections=False, mark_fix_applied=True)

        _add_warning(context, "some.code", "message", fix_message="do the thing")

        self.assertFalse(context.report.issues[0].fix_applied)

    def test_no_fix_message_means_no_fix_applied(self) -> None:
        context = _context(use_corrections=True, mark_fix_applied=True)

        _add_warning(context, "some.code", "message")

        self.assertFalse(context.report.issues[0].fix_applied)

    def test_errors_and_warnings_land_at_the_right_level(self) -> None:
        context = _context()

        _add_warning(context, "w.code", "warned", line_number=4, node_id=2)
        _add_error(context, "e.code", "failed", line_number=5, node_id=3)

        warning, error = context.report.issues
        self.assertEqual((warning.level, warning.line_number, warning.node_id), ("warning", 4, 2))
        self.assertEqual((error.level, error.line_number, error.node_id), ("error", 5, 3))
        self.assertTrue(context.report.has_errors)


class SetAttrTest(unittest.TestCase):
    def test_attribute_is_written_when_corrections_are_enabled(self) -> None:
        context = _context(use_corrections=True)
        row = _unparsed(1, 1, 0, 0, 0, 1, -1)

        _set_attr(context, row, "radius", 0.5)

        self.assertEqual(row.radius, 0.5)

    def test_attribute_is_left_alone_in_report_only_mode(self) -> None:
        context = _context(use_corrections=False)
        row = _unparsed(1, 1, 0, 0, 0, 1, -1)

        _set_attr(context, row, "radius", 0.5)

        self.assertIsNone(row.radius)


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


class RuleMissingFieldColumnsTest(unittest.TestCase):
    def test_seven_columns_are_accepted(self) -> None:
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0 -1", 1)])

        rule_missing_field_columns(context)

        self.assertEqual(_codes(context), [])
        self.assertEqual(len(context.rows), 1)
        self.assertFalse(context.stop_processing)

    def test_too_few_columns_is_a_fatal_error(self) -> None:
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0", 1)])

        rule_missing_field_columns(context)

        self.assertEqual(_codes(context), ["format.column_count"])
        self.assertEqual(context.rows, [])
        self.assertTrue(context.stop_processing)

    def test_extra_columns_are_truncated_to_seven(self) -> None:
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0 -1 99 100", 1)])

        rule_missing_field_columns(context)

        self.assertEqual(len(context.rows[0].fields), 7)
        self.assertEqual(context.rows[0].fields[-1], "-1")

    def test_good_rows_survive_alongside_a_bad_one(self) -> None:
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0 -1", 1), _raw("2 3", 2)])

        rule_missing_field_columns(context)

        self.assertEqual(len(context.rows), 1)
        self.assertTrue(context.stop_processing)


class RuleTreeSampleCountTest(unittest.TestCase):
    def test_empty_file_is_fatal(self) -> None:
        context = _context(rows=[])

        rule_tree_sample_count(context)

        self.assertEqual(_codes(context), ["format.empty_file"])
        self.assertTrue(context.stop_processing)

    def test_a_short_file_only_warns(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1)])

        rule_tree_sample_count(context)

        self.assertEqual(_codes(context), ["format.low_sample_count"])
        self.assertFalse(context.stop_processing)

    def test_twenty_samples_is_the_quiet_threshold(self) -> None:
        rows = [_parsed(i, 1, 0, 0, 0, 1.0, -1 if i == 1 else i - 1) for i in range(1, 21)]
        context = _context(rows=rows)

        rule_tree_sample_count(context)

        self.assertEqual(_codes(context), [])


class RuleItpIntTest(unittest.TestCase):
    def test_clean_row_parses_all_three_integer_columns(self) -> None:
        context = _context(rows=[_unparsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1)])

        rule_itp_int(context)

        row = context.rows[0]
        self.assertEqual((row.node_id, row.type_code, row.parent_id), (1, 1, -1))
        self.assertEqual(_codes(context), [])
        self.assertFalse(context.stop_processing)

    def test_non_positive_node_id_is_fatal(self) -> None:
        for bad in ("0", "-1"):
            with self.subTest(node_id=bad):
                context = _context(rows=[_unparsed(bad, 1, 0.0, 0.0, 0.0, 1.0, -1)])

                rule_itp_int(context)

                self.assertIn("identity.invalid_id", _codes(context))
                self.assertTrue(context.stop_processing)

    def test_non_numeric_node_id_is_fatal(self) -> None:
        context = _context(rows=[_unparsed("abc", 1, 0.0, 0.0, 0.0, 1.0, -1)])

        rule_itp_int(context)

        self.assertIn("identity.invalid_id", _codes(context))
        self.assertTrue(context.stop_processing)

    def test_duplicate_node_ids_are_fatal(self) -> None:
        context = _context(
            rows=[
                _unparsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1, line_number=1),
                _unparsed(1, 3, 1.0, 0.0, 0.0, 1.0, 1, line_number=2),
            ]
        )

        rule_itp_int(context)

        self.assertIn("identity.duplicate_id", _codes(context))
        self.assertTrue(context.stop_processing)

    def test_float_spelled_index_is_normalized_with_a_warning(self) -> None:
        context = _context(rows=[_unparsed("2.0", 1, 0.0, 0.0, 0.0, 1.0, -1)])

        rule_itp_int(context)

        self.assertIn("format.index_int", _codes(context))
        self.assertEqual(context.rows[0].node_id, 2)

    def test_non_integer_type_falls_back_to_custom(self) -> None:
        context = _context(rows=[_unparsed(1, "abc", 0.0, 0.0, 0.0, 1.0, -1)])

        rule_itp_int(context)

        self.assertIn("format.type_int", _codes(context))
        self.assertEqual(context.rows[0].type_code, 0)

    def test_float_spelled_type_is_normalized(self) -> None:
        context = _context(rows=[_unparsed(1, "3.0", 0.0, 0.0, 0.0, 1.0, -1)])

        rule_itp_int(context)

        self.assertIn("format.type_int", _codes(context))
        self.assertEqual(context.rows[0].type_code, 3)

    def test_float_spelled_parent_is_normalized(self) -> None:
        context = _context(
            rows=[
                _unparsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1, line_number=1),
                _unparsed(2, 3, 1.0, 0.0, 0.0, 1.0, "1.0", line_number=2),
            ]
        )

        rule_itp_int(context)

        self.assertIn("format.parent_int", _codes(context))
        self.assertEqual(context.rows[1].parent_id, 1)

    def test_non_numeric_parent_is_fatal(self) -> None:
        context = _context(rows=[_unparsed(1, 1, 0.0, 0.0, 0.0, 1.0, "abc")])

        rule_itp_int(context)

        self.assertIn("identity.invalid_parent_id", _codes(context))
        self.assertTrue(context.stop_processing)


class RuleXyzDoubleTest(unittest.TestCase):
    def test_valid_coordinates_are_parsed(self) -> None:
        context = _context(rows=[_unparsed(1, 1, "1.5", "-2.5", "3.0", 1.0, -1)])

        rule_xyz_double(context)

        row = context.rows[0]
        self.assertEqual((row.x, row.y, row.z), (1.5, -2.5, 3.0))
        self.assertEqual(_codes(context), [])

    def test_placeholder_coordinate_is_replaced_with_zero(self) -> None:
        context = _context(rows=[_unparsed(1, 1, "n/a", "0.0", "0.0", 1.0, -1)])

        rule_xyz_double(context)

        self.assertEqual(_codes(context), ["geometry.xyz_double"])
        self.assertEqual(context.rows[0].x, 0.0)

    def test_report_only_mode_warns_without_writing_the_row(self) -> None:
        context = _context(rows=[_unparsed(1, 1, "n/a", "0.0", "0.0", 1.0, -1)], use_corrections=False)

        rule_xyz_double(context)

        self.assertEqual(_codes(context), ["geometry.xyz_double"])
        self.assertIsNone(context.rows[0].x)


class RuleRadiusPositiveDoubleTest(unittest.TestCase):
    def test_positive_radius_is_kept(self) -> None:
        context = _context(rows=[_unparsed(1, 1, 0.0, 0.0, 0.0, "2.5", -1)])

        rule_radius_positive_double(context)

        self.assertEqual(context.rows[0].radius, 2.5)
        self.assertEqual(_codes(context), [])

    def test_non_positive_and_unparseable_radii_fall_back_to_half(self) -> None:
        for token in ("0.0", "-1.0", "n/a", "abc"):
            with self.subTest(radius=token):
                context = _context(rows=[_unparsed(1, 1, 0.0, 0.0, 0.0, token, -1)])

                rule_radius_positive_double(context)

                self.assertEqual(_codes(context), ["geometry.radius_positive_double"])
                self.assertEqual(context.rows[0].radius, 0.5)


class RuleNonstandardTypeIdTest(unittest.TestCase):
    def test_mapped_type_codes_pass_silently(self) -> None:
        rows = [_parsed(i + 1, code, 0, 0, 0, 1.0, -1) for i, code in enumerate((0, 1, 2, 3, 4))]
        context = _context(rows=rows)

        rule_nonstandard_type_id(context)

        self.assertEqual(_codes(context), [])

    def test_unknown_type_becomes_custom_by_default(self) -> None:
        context = _context(rows=[_parsed(1, 7, 0, 0, 0, 1.0, -1)])

        rule_nonstandard_type_id(context)

        self.assertEqual(_codes(context), ["semantics.unknown_type"])
        self.assertEqual(context.report.issues[0].level, "warning")
        self.assertEqual(context.rows[0].type_code, 0)

    def test_unknown_type_is_an_error_when_the_option_is_off(self) -> None:
        context = _context(
            rows=[_parsed(1, 7, 0, 0, 0, 1.0, -1)],
            options=SwcReadOptions(unknown_type_as_custom=False),
        )

        rule_nonstandard_type_id(context)

        self.assertEqual(_codes(context), ["semantics.unknown_type"])
        self.assertEqual(context.report.issues[0].level, "error")
        self.assertEqual(context.rows[0].type_code, 7)


class RuleInvalidParentIndexTest(unittest.TestCase):
    def test_parent_zero_is_rewritten_to_root(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, 0)])

        rule_invalid_parent_index(context)

        self.assertEqual(_codes(context), ["topology.invalid_parent"])
        self.assertEqual(context.rows[0].parent_id, -1)

    def test_self_parent_is_rewritten_to_root(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, 1)])

        rule_invalid_parent_index(context)

        self.assertEqual(_codes(context), ["topology.invalid_parent"])
        self.assertEqual(context.rows[0].parent_id, -1)

    def test_dangling_parent_is_rewritten_to_root(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, 99)])

        rule_invalid_parent_index(context)

        self.assertEqual(_codes(context), ["topology.invalid_parent"])
        self.assertEqual(context.rows[0].parent_id, -1)

    def test_a_real_root_and_a_real_parent_pass_silently(self) -> None:
        context = _context(
            rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 3, 1, 0, 0, 1.0, 1)],
        )

        rule_invalid_parent_index(context)

        self.assertEqual(_codes(context), [])

    def test_report_only_mode_warns_without_rewriting(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, 99)], use_corrections=False)

        rule_invalid_parent_index(context)

        self.assertEqual(_codes(context), ["topology.invalid_parent"])
        self.assertEqual(context.rows[0].parent_id, 99)


class RuleDuplicateXyzrParentChildTest(unittest.TestCase):
    def _tree_with_duplicate(self):
        return [
            _parsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1),
            _parsed(2, 3, 0.0, 0.0, 0.0, 1.0, 1),  # identical xyzr to its parent
            _parsed(3, 3, 1.0, 0.0, 0.0, 1.0, 2),
        ]

    def test_duplicate_is_merged_and_grandchild_is_reparented(self) -> None:
        context = _context(rows=self._tree_with_duplicate())

        rule_duplicate_xyzr_parent_child(context)

        self.assertEqual(_codes(context), ["geometry.duplicate_xyzr_node"])
        self.assertEqual([row.node_id for row in context.rows], [1, 3])
        self.assertEqual(context.rows[1].parent_id, 1)

    def test_report_only_mode_warns_and_keeps_the_row(self) -> None:
        context = _context(rows=self._tree_with_duplicate(), use_corrections=False)

        rule_duplicate_xyzr_parent_child(context)

        self.assertEqual(_codes(context), ["geometry.duplicate_xyzr_node"])
        self.assertEqual([row.node_id for row in context.rows], [1, 2, 3])

    def test_distinct_geometry_is_left_alone(self) -> None:
        context = _context(
            rows=[_parsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1), _parsed(2, 3, 1.0, 0.0, 0.0, 1.0, 1)],
        )

        rule_duplicate_xyzr_parent_child(context)

        self.assertEqual(_codes(context), [])
        self.assertEqual(len(context.rows), 2)

    def test_a_chain_of_duplicates_collapses_onto_one_surviving_ancestor(self) -> None:
        # Merges cascade: 2 and 3 both repeat the root, so both disappear and
        # the first genuinely distinct node reparents all the way up to 1.
        context = _context(
            rows=[
                _parsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1),
                _parsed(2, 3, 0.0, 0.0, 0.0, 1.0, 1),
                _parsed(3, 3, 0.0, 0.0, 0.0, 1.0, 2),
                _parsed(4, 3, 1.0, 0.0, 0.0, 1.0, 3),
            ]
        )

        rule_duplicate_xyzr_parent_child(context)

        self.assertEqual([row.node_id for row in context.rows], [1, 4])
        self.assertEqual(context.rows[1].parent_id, 1)
        self.assertEqual([issue.node_id for issue in context.report.issues], [2, 3])

    def test_merges_are_reported_in_row_order_not_node_id_order(self) -> None:
        # The single-pass implementation replaces a restart-the-whole-scan
        # loop, whose observable contract was "lowest surviving row position
        # first". Rows here are deliberately out of node-id order so that a
        # by-id traversal would report 3, 5, 7 instead of 7, 3, 5.
        context = _context(
            rows=[
                _parsed(1, 1, 0.0, 0.0, 0.0, 1.0, -1, line_number=1),
                _parsed(6, 3, 5.0, 0.0, 0.0, 1.0, 1, line_number=2),
                _parsed(7, 3, 5.0, 0.0, 0.0, 1.0, 6, line_number=3),
                _parsed(2, 3, 2.0, 0.0, 0.0, 1.0, 1, line_number=4),
                _parsed(3, 3, 2.0, 0.0, 0.0, 1.0, 2, line_number=5),
                _parsed(4, 3, 8.0, 0.0, 0.0, 1.0, 1, line_number=6),
                _parsed(5, 3, 8.0, 0.0, 0.0, 1.0, 4, line_number=7),
            ]
        )

        rule_duplicate_xyzr_parent_child(context)

        self.assertEqual([issue.node_id for issue in context.report.issues], [7, 3, 5])
        self.assertEqual([row.node_id for row in context.rows], [1, 6, 2, 4])


class RuleNoSomaSamplesTest(unittest.TestCase):
    def test_a_soma_sample_passes_silently(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1)])

        rule_no_soma_samples(context)

        self.assertEqual(_codes(context), [])

    def test_no_soma_sample_warns(self) -> None:
        context = _context(rows=[_parsed(1, 3, 0, 0, 0, 1.0, -1)])

        rule_no_soma_samples(context)

        self.assertEqual(_codes(context), ["semantics.no_soma_samples"])


class RuleContourTest(unittest.TestCase):
    def test_contour_detection_is_disabled_but_the_rule_still_exists(self) -> None:
        # Deliberately a no-op: contour-soma classification is off for SWC
        # import, and the rule is kept only so the pipeline shape is stable.
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1)])

        rule_contour(context)

        self.assertEqual(_codes(context), [])
        self.assertEqual(context.contour_soma_ids, set())

    def test_the_rule_is_still_registered_in_the_pipeline(self) -> None:
        self.assertIn(rule_contour, SWC_RULES)


class RuleSortedIndexOrderTest(unittest.TestCase):
    def test_parent_before_child_passes_silently(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 3, 1, 0, 0, 1.0, 1)])

        rule_sorted_index_order(context)

        self.assertEqual(_codes(context), [])

    def test_child_before_parent_is_reordered(self) -> None:
        context = _context(rows=[_parsed(2, 3, 1, 0, 0, 1.0, 1), _parsed(1, 1, 0, 0, 0, 1.0, -1)])

        rule_sorted_index_order(context)

        self.assertEqual(_codes(context), ["topology.sorted_order"])
        self.assertEqual([row.node_id for row in context.rows], [1, 2])

    def test_report_only_mode_warns_without_reordering(self) -> None:
        context = _context(
            rows=[_parsed(2, 3, 1, 0, 0, 1.0, 1), _parsed(1, 1, 0, 0, 0, 1.0, -1)],
            use_corrections=False,
        )

        rule_sorted_index_order(context)

        self.assertEqual(_codes(context), ["topology.sorted_order"])
        self.assertEqual([row.node_id for row in context.rows], [2, 1])

    def test_an_empty_row_list_is_a_no_op(self) -> None:
        context = _context(rows=[])

        rule_sorted_index_order(context)

        self.assertEqual(_codes(context), [])


class RuleIndexSequentialTest(unittest.TestCase):
    def test_already_sequential_ids_pass_silently(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 3, 1, 0, 0, 1.0, 1)])

        rule_index_sequential(context)

        self.assertEqual(_codes(context), [])

    def test_sparse_ids_are_renumbered_and_parents_remapped(self) -> None:
        context = _context(
            rows=[
                _parsed(10, 1, 0, 0, 0, 1.0, -1, line_number=1),
                _parsed(20, 3, 1, 0, 0, 1.0, 10, line_number=2),
                _parsed(30, 3, 2, 0, 0, 1.0, 20, line_number=3),
            ]
        )

        rule_index_sequential(context)

        self.assertEqual(_codes(context), ["identity.sequential_index"])
        self.assertEqual([row.node_id for row in context.rows], [1, 2, 3])
        self.assertEqual([row.parent_id for row in context.rows], [-1, 1, 2])

    def test_report_only_mode_warns_without_renumbering(self) -> None:
        context = _context(
            rows=[_parsed(10, 1, 0, 0, 0, 1.0, -1, line_number=1)],
            use_corrections=False,
        )

        rule_index_sequential(context)

        self.assertEqual(_codes(context), ["identity.sequential_index"])
        self.assertEqual(context.rows[0].node_id, 10)

    def test_contour_soma_ids_follow_the_renumbering(self) -> None:
        context = _context(
            rows=[
                _parsed(10, 1, 0, 0, 0, 1.0, -1, line_number=1),
                _parsed(20, 1, 1, 0, 0, 1.0, 10, line_number=2),
            ]
        )
        context.contour_soma_ids = {20}

        rule_index_sequential(context)

        self.assertEqual(context.contour_soma_ids, {2})


class RuleRootParentIndexTest(unittest.TestCase):
    def test_exactly_one_root_passes_silently(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 3, 1, 0, 0, 1.0, 1)])

        rule_root_parent_index(context)

        self.assertEqual(_codes(context), [])

    def test_no_root_is_an_error(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, 2), _parsed(2, 3, 1, 0, 0, 1.0, 1)])

        rule_root_parent_index(context)

        self.assertEqual(_codes(context), ["topology.root_count"])

    def test_two_roots_is_an_error(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 1, 1, 0, 0, 1.0, -1)])

        rule_root_parent_index(context)

        self.assertEqual(_codes(context), ["topology.root_count"])

    def test_non_soma_root_is_rejected_only_when_required(self) -> None:
        rows = [_parsed(1, 3, 0, 0, 0, 1.0, -1)]

        permissive = _context(rows=list(rows))
        rule_root_parent_index(permissive)
        self.assertEqual(_codes(permissive), [])

        strict = _context(rows=list(rows), options=SwcReadOptions(require_root_type_soma=True))
        rule_root_parent_index(strict)
        self.assertEqual(_codes(strict), ["topology.root_type"])


class RuleTreeIntegrityTest(unittest.TestCase):
    def test_a_connected_tree_passes_silently(self) -> None:
        context = _context(
            rows=[
                _parsed(1, 1, 0, 0, 0, 1.0, -1),
                _parsed(2, 3, 1, 0, 0, 1.0, 1),
                _parsed(3, 3, 2, 0, 0, 1.0, 2),
            ]
        )

        rule_tree_integrity(context)

        self.assertEqual(_codes(context), [])

    def test_an_empty_row_list_is_a_no_op(self) -> None:
        context = _context(rows=[])

        rule_tree_integrity(context)

        self.assertEqual(_codes(context), [])

    def test_a_reference_to_a_missing_parent_is_an_orphan(self) -> None:
        context = _context(rows=[_parsed(1, 1, 0, 0, 0, 1.0, -1), _parsed(2, 3, 1, 0, 0, 1.0, 99)])

        rule_tree_integrity(context)

        self.assertIn("topology.orphan", _codes(context))

    def test_a_component_unreachable_from_the_root_is_disconnected(self) -> None:
        # 2 and 3 parent each other, so neither is reachable from root 1.
        context = _context(
            rows=[
                _parsed(1, 1, 0, 0, 0, 1.0, -1),
                _parsed(2, 3, 1, 0, 0, 1.0, 3),
                _parsed(3, 3, 2, 0, 0, 1.0, 2),
            ]
        )

        rule_tree_integrity(context)

        self.assertEqual(_codes(context), ["topology.disconnected"])


# ---------------------------------------------------------------------------
# Pipeline driver and error surfacing
# ---------------------------------------------------------------------------


class ApplySwcRulesTest(unittest.TestCase):
    def test_a_prior_stop_flag_prevents_every_rule_from_running(self) -> None:
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0", 1)])
        context.stop_processing = True

        apply_swc_rules(context)

        # rule_missing_field_columns would otherwise have flagged the short row.
        self.assertEqual(_codes(context), [])

    def test_a_fatal_rule_halts_the_remaining_rules(self) -> None:
        # A 6-column row makes the first rule fatal; without the halt,
        # rule_tree_sample_count would add format.empty_file on top.
        context = _context(raw_rows=[_raw("1 1 0 0 0 1.0", 1)])

        apply_swc_rules(context)

        self.assertEqual(_codes(context), ["format.column_count"])
        self.assertTrue(context.stop_processing)

    def test_a_clean_file_runs_the_whole_pipeline(self) -> None:
        raw_rows = [_raw("1 1 0 0 0 1.0 -1", 1), _raw("2 3 1 0 0 1.0 1", 2)]
        context = _context(raw_rows=raw_rows)

        apply_swc_rules(context)

        self.assertFalse(context.stop_processing)
        self.assertFalse(context.report.has_errors)
        # A 2-sample file is under the 20-sample threshold.
        self.assertEqual(_codes(context), ["format.low_sample_count"])


class RaiseForSwcErrorsTest(unittest.TestCase):
    def test_a_clean_report_does_not_raise(self) -> None:
        self.assertIsNone(raise_for_swc_errors(SwcReport(), Path("memory.swc")))

    def test_warnings_alone_do_not_raise(self) -> None:
        report = SwcReport()
        report.add_warning("some.code", "just a warning")

        self.assertIsNone(raise_for_swc_errors(report, Path("memory.swc")))

    def test_errors_raise_with_the_path_and_the_error_only_report(self) -> None:
        report = SwcReport()
        report.add_warning("noisy.warning", "should not appear")
        report.add_error("format.column_count", "too few columns")

        with self.assertRaises(ValueError) as caught:
            raise_for_swc_errors(report, Path("broken.swc"))

        message = str(caught.exception)
        self.assertIn("broken.swc", message)
        self.assertIn("format.column_count", message)
        self.assertNotIn("noisy.warning", message)


if __name__ == "__main__":
    unittest.main()
