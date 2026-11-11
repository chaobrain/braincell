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


import codecs
import re
import tempfile
import textwrap
import unittest
from pathlib import Path

import brainunit as u
import numpy as np

from braincell import Morphology
from braincell.io._testing import CEREBELLUM_FIXTURES, ALLOWED_TYPES, FIXTURE_DIR, LATIN1_ASC_FIXTURE
from braincell.io.asc import AscReader, AscSpineRecord

try:
    from neuron import h as _NEURON_H
except Exception:  # pragma: no cover
    _NEURON_H = None


# Soma contours shared by the fixtures below. The indentation is baked in so
# that ``textwrap.dedent`` in ``_write_asc`` still strips a consistent prefix
# once these are interpolated into a test body.
_SOMA = """("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )"""

_SOMA_Z5 = """("Cell Body"
              (CellBody)
              (0 0 5 0)
              (5 0 5 0)
              (0 5 5 0)
              (-5 0 5 0)
            )"""


# Hand-written inputs for the ASC import gaps (MORPH-5). Each is one CellBody
# contour, (0,0), (4,0), (0,4), (-4,0), and a 10 um tree, with one thing
# changed; the line numbers the tests expect refer to these texts as written.
_GAP_FIXTURES = {
    "nonnumeric_point": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 abc 0 1)
  (0 10 0 1)
)
""",
    "three_number_point": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "nan_point": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 nan 0 1)
  (0 10 0 1)
)
""",
    "negative_diameter": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 -1)
  (0 10 0 1)
)
""",
    "zero_diameter": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 0)
  (0 10 0 1)
)
""",
    "point_after_children": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (
    (0 10 0 1)
  |
    (5 10 0 1)
  )
  (0 20 0 1)
)
""",
    "collinear_contour": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (2 0 0 0)
  (4 0 0 0)
  (6 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "two_point_contour": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "nonplanar_contour": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 1 0)
  (0 4 2 0)
  (-4 0 1 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "vertical_closed_contour": """\
("CellBody"
  (CellBody)
  (-4 0 0 0)
  (0 0 1 0)
  (4 0 0 0)
  (0 0 -1 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "untyped_tree": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Color Red)
  (0 0 0 1)
  (5 0 0 1)
  (10 0 0 1)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "soma_name_no_tag": """\
("Soma"
  (Closed)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "cellbody_space_name_no_tag": """\
("Cell Body"
  (Closed)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "tag_other_name": """\
("Outline"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
    "no_soma_two_trees": """\
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
( (Axon)
  (50 0 0 1)
  (50 -5 0 1)
  (50 -10 0 1)
)
""",
    "two_cellbody_groups": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
("CellBody"
  (CellBody)
  (100 0 0 0)
  (104 0 0 0)
  (100 4 0 0)
  (96 0 0 0)
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
( (Dendrite)
  (100 0 0 1)
  (100 5 0 1)
  (100 10 0 1)
)
""",
    "text_label": """\
("CellBody"
  (CellBody)
  (0 0 0 0)
  (4 0 0 0)
  (0 4 0 0)
  (-4 0 0 0)
)
( (Color Red)
  (Font "Arial" 10)
  (20 20 0 1)
  "label"
)
( (Dendrite)
  (0 0 0 1)
  (0 5 0 1)
  (0 10 0 1)
)
""",
}


class _AscTestMixin:
    def _write_asc(self, body: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)  # type: ignore[attr-defined]
        path = Path(temp_dir.name) / "sample.asc"
        path.write_text(textwrap.dedent(body).strip() + "\n")
        return path


class AscReaderTest(_AscTestMixin, unittest.TestCase):
    @staticmethod
    def _issue_codes(report) -> list[str]:
        return [issue.code for issue in report.issues]

    def test_reader_imports_simple_neurolucida_tree_and_metadata(self) -> None:
        path = self._write_asc(
            f'''
            ; example asc
            {_SOMA}

            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (
                (0 10 0 1)
                (0 15 0 1)
                (Spine "s1")
                Normal
              |
                (5 5 0 1)
                (10 5 0 1)
                (Marker "m1")
                (FilledCircle 1 2 3)
                Normal
              )
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertIsInstance(tree, Morphology)
        self.assertFalse(report.has_errors)
        self.assertEqual(tree.root.type, "soma")
        self.assertEqual(len(tree.branches), 4)
        self.assertEqual(tree.root.branch.points.shape[0], 21)
        self.assertEqual(tree.root.n_children, 1)
        self.assertEqual(tree.branch(index=1).type, "dendrite")
        self.assertEqual(tree.branch(index=1).parent_x, 0.5)
        self.assertEqual(tree.branch(index=1).n_children, 2)
        self.assertEqual(tree.branch(index=2).parent_x, 1.0)
        self.assertEqual(tree.branch(index=3).parent_x, 1.0)
        self.assertEqual(len(report.metadata.spines), 0)
        self.assertEqual(len(report.metadata.spine_annotations), 1)
        self.assertEqual(len(report.metadata.markers), 1)
        self.assertEqual(len(report.metadata.filled_circles), 1)
        self.assertGreaterEqual(len(report.metadata.comments), 1)
        self.assertIn("Cell Body", report.metadata.source_labels)

    def test_morpho_from_asc_supports_return_report(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Axon)
              (0 0 0 1)
              (10 0 0 1)
            )
            '''
        )

        tree, report = Morphology.from_asc(path, return_report=True)

        self.assertIsInstance(tree, Morphology)
        self.assertFalse(report.has_errors)
        self.assertEqual(tree.root.type, "soma")
        self.assertEqual(tree.branch(index=1).type, "axon")

    def test_reader_converts_valid_cellbody_stack_to_layer_centroids(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            {_SOMA_Z5}
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertFalse(report.has_errors)
        self.assertEqual(tree.root.branch.points.shape[0], 2)
        self.assertEqual(tree.branch(index=1).parent_x, 0.5)

    def test_reader_rejects_cellbody_stack_with_duplicate_z_layers(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (5 0 0 0)
              (0 5 0 0)
              (-5 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        with self.assertRaisesRegex(ValueError, "strictly monotonic z"):
            AscReader().read(path)

    def test_reader_rejects_cellbody_stack_with_non_monotonic_z_order(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            {_SOMA_Z5}
            ("Cell Body"
              (CellBody)
              (0 0 2 0)
              (5 0 2 0)
              (0 5 2 0)
              (-5 0 2 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        with self.assertRaisesRegex(ValueError, "not monotonic in z"):
            AscReader().read(path)

    def test_reader_keeps_single_point_terminal_branch(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (
                (0 10 0 1)
              |
                (5 5 0 1)
                (10 5 0 1)
              )
            )
            '''
        )

        tree = AscReader().read(path)

        self.assertEqual(len(tree.branches), 4)
        self.assertEqual(tree.branch(index=1).n_children, 2)
        self.assertEqual(tree.branch(index=2).branch.points.shape[0], 2)

    def test_reader_keeps_child_diameter_when_child_starts_at_parent_endpoint(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 2)
              (0 5 0 2)
              (
                (0 5 0 1)
                (0 10 0 1)
              )
            )
            '''
        )

        tree = AscReader().read(path)
        child = tree.branch(index=2).branch
        diameters = 2.0 * np.asarray(child.radii.to_decimal(u.um), dtype=float)

        self.assertEqual(child.points.shape[0], 2)
        self.assertTrue(np.allclose(diameters, np.array([1.0, 1.0])))

    def test_reader_copies_parent_xyz_with_child_diameter_when_child_starts_away_from_parent(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 2)
              (0 5 0 2)
              (
                (0 7 0 1)
                (0 10 0 1)
              )
            )
            '''
        )

        tree = AscReader().read(path)
        child = tree.branch(index=2).branch
        points = np.asarray(child.points.to_decimal(u.um), dtype=float)
        diameters = 2.0 * np.asarray(child.radii.to_decimal(u.um), dtype=float)

        self.assertEqual(child.points.shape[0], 3)
        self.assertTrue(np.allclose(points[0], np.array([0.0, 5.0, 0.0])))
        self.assertTrue(np.allclose(diameters, np.array([1.0, 1.0, 1.0])))

    def test_reader_preserves_duplicate_points_inside_branch(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (0 5 0 1)
              (0 10 0 1)
            )
            '''
        )

        tree = AscReader().read(path)
        branch = tree.branch(index=1).branch
        points = np.asarray(branch.points.to_decimal(u.um), dtype=float)

        self.assertEqual(branch.points.shape[0], 4)
        self.assertTrue(np.allclose(points[1], points[2]))

    def test_reader_supports_angle_spine_block_and_keeps_parent_branch_geometry(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 4 "none")
                (Color Red)
                (Generated 0)
                (1 1 0 0.1)
              >
              (0 5 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertFalse(report.has_errors)
        self.assertNotIn("syntax.point_after_children", self._issue_codes(report))
        self.assertEqual(len(tree.branches), 2)
        dendrite = tree.branch(index=1).branch
        dend_points = np.asarray(dendrite.points.to_decimal(u.um), dtype=float)
        dend_diameters = 2.0 * np.asarray(dendrite.radii.to_decimal(u.um), dtype=float)
        self.assertEqual(dendrite.points.shape[0], 2)
        self.assertTrue(np.allclose(dend_points, np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])))
        self.assertTrue(np.allclose(dend_diameters, np.array([1.0, 1.0])))
        self.assertEqual(len(report.metadata.spines), 1)
        spine = report.metadata.spines[0]
        self.assertIsInstance(spine, AscSpineRecord)
        self.assertEqual(spine.base_xyz, (0.0, 0.0, 0.0))
        self.assertEqual(spine.base_diameter, 1.0)
        self.assertEqual(spine.tip_xyz, (1.0, 1.0, 0.0))
        self.assertEqual(spine.tip_diameter, 0.1)
        self.assertEqual(spine.class_type, 4)
        self.assertEqual(spine.class_label, "none")

    def test_reader_supports_multiple_angle_spine_blocks_after_one_point(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 1 "a")
                (1 1 0 0.1)
              >
              <
                (Class 2 "b")
                (1 -1 0 0.2)
              >
              (0 5 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertFalse(report.has_errors)
        self.assertEqual(len(tree.branches), 2)
        self.assertNotIn("syntax.point_after_children", self._issue_codes(report))
        self.assertEqual(len(report.metadata.spines), 2)
        self.assertEqual(report.metadata.spines[0].class_type, 1)
        self.assertEqual(report.metadata.spines[1].class_type, 2)

    def test_reader_treats_generic_property_tuples_as_annotations_not_children(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Color Red)
              (zSmear 1.0 0.0)
              (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
              (0 10 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertFalse(report.has_errors)
        self.assertNotIn("syntax.point_after_children", self._issue_codes(report))
        self.assertEqual(len(tree.branches), 2)
        dendrite = tree.branch(index=1).branch
        self.assertEqual(dendrite.points.shape[0], 3)

    def test_reader_warns_for_spine_before_first_point(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              <
                (Class 4 "none")
                (1 1 0 0.1)
              >
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        _, report = AscReader().read(path, return_report=True)

        self.assertIn("syntax.spine_before_point", self._issue_codes(report))

    def test_reader_warns_for_spine_with_missing_tip(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 4 "none")
                (Color Red)
              >
              (0 5 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertEqual(len(tree.branches), 2)
        self.assertIn("syntax.spine_missing_tip", self._issue_codes(report))
        self.assertEqual(len(report.metadata.spines), 0)

    def test_reader_warns_for_spine_with_multiple_tip_points(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 4 "none")
                (1 1 0 0.1)
                (2 2 0 0.1)
              >
              (0 5 0 1)
            )
            '''
        )

        tree, report = AscReader().read(path, return_report=True)

        self.assertEqual(len(tree.branches), 2)
        self.assertIn("syntax.spine_multiple_tips", self._issue_codes(report))
        self.assertEqual(len(report.metadata.spines), 0)

    def test_sc_fixture_drops_point_after_children_warning_after_spine_support(self) -> None:
        path = CEREBELLUM_FIXTURES["SC"]

        _, report = AscReader().read(path, return_report=True)

        self.assertNotIn("syntax.point_after_children", self._issue_codes(report))

    def test_bc_fixture_reports_root_outside_soma_bbox_warnings(self) -> None:
        path = CEREBELLUM_FIXTURES["BC"]

        _, report = AscReader().read(path, return_report=True)

        codes = self._issue_codes(report)
        self.assertNotIn("syntax.point_after_children", codes)
        self.assertEqual(sum(code == "topology.root_outside_soma_bbox" for code in codes), 7)


class AscMalformedInputTest(_AscTestMixin, unittest.TestCase):
    """Malformed points and soma contours raise, naming the line (MORPH-5b)."""

    def _assert_read_fails(self, text: str, message: str) -> None:
        path = self._write_asc(text)
        pattern = f"^ASC import failed for {re.escape(str(path))}: {message}"
        with self.assertRaisesRegex(ValueError, pattern):
            AscReader().read(path)

    def test_invalid_tree_point_raises_with_its_line(self) -> None:
        cases = {
            "nonnumeric_point": r"invalid point at line 10: expected 4 numbers \(x y z diameter\), found 'abc' at position 2",
            "three_number_point": r"invalid point at line 9: expected 4 numbers \(x y z diameter\), found 3",
            "nan_point": r"invalid point at line 10: x, y, z and diameter must be finite, found \(0 nan 0 1\)",
            "negative_diameter": r"invalid point at line 10: diameter must be positive, found -1",
            "zero_diameter": r"invalid point at line 10: diameter must be positive, found 0",
        }
        for name, message in cases.items():
            with self.subTest(fixture=name):
                self._assert_read_fails(_GAP_FIXTURES[name], message)

    def test_point_after_child_branches_raises_with_its_line(self) -> None:
        self._assert_read_fails(_GAP_FIXTURES["point_after_children"], "point at line 16 follows child branches")

    def test_degenerate_cellbody_contour_raises_with_its_line(self) -> None:
        cases = {
            "collinear_contour": "CellBody contour at line 1 is degenerate: its points are collinear",
            "two_point_contour": "CellBody contour at line 1 has 2 points; a contour needs at least 3",
        }
        for name, message in cases.items():
            with self.subTest(fixture=name):
                self._assert_read_fails(_GAP_FIXTURES[name], message)

    def test_point_with_more_than_one_trailing_item_raises(self) -> None:
        for point in ("(0 5 0 1 2)", "(0 5 0 1 S1 S2)", "(0 5 0 1 (Color Red))"):
            with self.subTest(point=point):
                text = _GAP_FIXTURES["zero_diameter"].replace("(0 5 0 0)", point)
                self._assert_read_fails(
                    text, r"invalid point at line 10: expected 4 numbers \(x y z diameter\) and at most one label"
                )

    def test_point_interrupted_by_a_non_number_names_it(self) -> None:
        cases = {
            "(0 5 | 0 1)": "'\\|' at position 3",
            "(0 5 0 (Color Red))": "a nested list at position 4",
            "(0 < (1 1 0 0.1) > 0 1)": "a spine block at position 2",
        }
        for point, found in cases.items():
            with self.subTest(point=point):
                text = _GAP_FIXTURES["zero_diameter"].replace("(0 5 0 0)", point)
                self._assert_read_fails(
                    text, rf"invalid point at line 10: expected 4 numbers \(x y z diameter\), found {found}"
                )

    def test_labelled_point_reads_like_the_unlabelled_point(self) -> None:
        plain = _GAP_FIXTURES["zero_diameter"].replace("(0 5 0 0)", "(0 5 0 1)")
        labelled = _GAP_FIXTURES["zero_diameter"].replace("(0 5 0 0)", "(0 5 0 1 S1)")

        plain_tree = AscReader().read(self._write_asc(plain))
        labelled_tree, report = AscReader().read(self._write_asc(labelled), return_report=True)

        self.assertEqual(report.issues, [])
        self.assertEqual(len(labelled_tree.branches), len(plain_tree.branches))
        for got, want in zip(labelled_tree.branches, plain_tree.branches):
            self.assertEqual(got.type, want.type)
            np.testing.assert_array_equal(got.branch.points.to_decimal(u.um), want.branch.points.to_decimal(u.um))
            np.testing.assert_array_equal(got.branch.radii.to_decimal(u.um), want.branch.radii.to_decimal(u.um))

    def test_non_utf8_file_falls_back_to_latin1_with_one_report_warning(self) -> None:
        tree, report = AscReader().read(LATIN1_ASC_FIXTURE, return_report=True)
        sc_tree = AscReader().read(CEREBELLUM_FIXTURES["SC"])

        fallbacks = [issue for issue in report.issues if issue.code == "syntax.encoding_fallback"]
        self.assertEqual(len(fallbacks), 1)
        self.assertEqual(fallbacks[0].level, "warning")
        self.assertIn("byte 0xe9 at offset", fallbacks[0].message)
        self.assertEqual(tree.n_branches, 120)
        self.assertEqual(tree.n_branches, sc_tree.n_branches)
        self.assertEqual(float(tree.total_length.to_decimal(u.um)), float(sc_tree.total_length.to_decimal(u.um)))
        self.assertEqual(float(tree.total_area.to_decimal(u.um**2)), float(sc_tree.total_area.to_decimal(u.um**2)))

    def test_encoding_fallback_names_the_first_invalid_byte(self) -> None:
        body = '("CellBody"\n  (CellBody)\n  (0 0 0 0)\n  (4 0 0 0)\n  (0 4 0 0)\n  (-4 0 0 0)\n)\n'
        body += '; caf\xe9\n( (Dendrite)\n  (0 0 0 1)\n  (0 10 0 1)\n)\n'
        path = self._write_asc("")
        path.write_bytes(body.encode("latin-1"))

        tree, report = AscReader().read(path, return_report=True)

        offset = body.index("\xe9")
        self.assertEqual([issue.code for issue in report.issues], ["syntax.encoding_fallback"])
        self.assertEqual(report.issues[0].line_number, 8)
        self.assertIn(f"byte 0xe9 at offset {offset}", report.issues[0].message)
        self.assertIn("caf\xe9", report.metadata.comments)
        self.assertEqual(tree.n_branches, 2)

    def test_utf8_bom_and_crlf_line_endings_read_like_plain_text(self) -> None:
        text = _GAP_FIXTURES["zero_diameter"].replace("(0 5 0 0)", "(0 5 0 1)")
        plain_tree = AscReader().read(self._write_asc(text))
        path = self._write_asc("")
        path.write_bytes(codecs.BOM_UTF8 + text.replace("\n", "\r\n").encode("utf-8"))

        tree, report = AscReader().read(path, return_report=True)

        self.assertEqual(report.issues, [])
        self.assertEqual(float(tree.total_area.to_decimal(u.um**2)), float(plain_tree.total_area.to_decimal(u.um**2)))
        bad = path.parent / "bad.asc"
        bad.write_bytes(text.replace("(0 5 0 1)", "(0 5 0 -1)").replace("\n", "\r").encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "invalid point at line 10"):
            AscReader().read(bad)

    def test_shipped_asc_fixtures_keep_their_geometry_and_issues(self) -> None:
        # Values from main before MORPH-5b; the change must leave them alone.
        expected = {
            "BC": (114, 2496.9785587068873, 4294.063957896643, {"topology.root_outside_soma_bbox"}),
            "GoC": (227, 4985.442723872502, 9806.738726884687, set()),
            "GrC": (5, 65.78856724516834, 243.8172886274238, set()),
            "PC": (461, 4464.311149963656, 14475.189250368849, set()),
            "SC": (120, 1388.0434735239396, 2317.9950713977005, set()),
        }
        paths = {name: CEREBELLUM_FIXTURES[name] for name in expected}
        paths["stellate"] = LATIN1_ASC_FIXTURE
        expected["stellate"] = (*expected["SC"][:3], {"syntax.encoding_fallback"})
        for name, path in paths.items():
            with self.subTest(fixture=name):
                tree, report = AscReader().read(path, return_report=True)
                n_branches, total_length, total_area, codes = expected[name]

                self.assertEqual(tree.n_branches, n_branches)
                np.testing.assert_allclose(float(tree.total_length.to_decimal(u.um)), total_length, rtol=1e-12)
                np.testing.assert_allclose(float(tree.total_area.to_decimal(u.um**2)), total_area, rtol=1e-12)
                self.assertEqual({issue.code for issue in report.issues}, codes)


class AscStructureTest(_AscTestMixin, unittest.TestCase):
    """Untyped trees, soma contours found by name, and the no-soma and many-soma messages (MORPH-5c)."""

    @staticmethod
    def _types(tree) -> list[str]:
        return [branch.type for branch in tree.branches]

    def test_untyped_tree_is_imported_as_custom_and_reported(self) -> None:
        tree, report = AscReader().read(self._write_asc(_GAP_FIXTURES["untyped_tree"]), return_report=True)

        self.assertEqual(self._types(tree), ["soma", "custom", "dendrite"])
        untyped = [issue for issue in report.issues if issue.code == "topology.untyped_tree"]
        self.assertEqual(len(untyped), 1)
        self.assertEqual(untyped[0].level, "warning")
        self.assertEqual(untyped[0].line_number, 8)
        self.assertEqual([issue.code for issue in report.issues], ["topology.untyped_tree"])
        # NEURON 9.0.2: 3 sections, total_length 27.63631.
        self.assertAlmostEqual(float(tree.total_length.to_decimal(u.um)), 27.63631, places=5)

    def test_text_label_is_not_a_tree(self) -> None:
        # ( Properties Point "text" ) is Neurolucida's text label, as in NEURON's grammar.
        tree, report = AscReader().read(self._write_asc(_GAP_FIXTURES["text_label"]), return_report=True)

        self.assertEqual(self._types(tree), ["soma", "dendrite"])
        self.assertEqual(report.issues, [])
        self.assertIn("label", report.metadata.source_labels)

    def test_contour_named_like_a_soma_is_the_soma_without_a_marker(self) -> None:
        for name in ("soma_name_no_tag", "cellbody_space_name_no_tag"):
            with self.subTest(fixture=name):
                tree, report = AscReader().read(self._write_asc(_GAP_FIXTURES[name]), return_report=True)

                self.assertEqual(tree.root.type, "soma")
                self.assertEqual(tree.root.branch.points.shape[0], 21)
                self.assertNotIn("topology.synthetic_soma", [issue.code for issue in report.issues])
                # NEURON 9.0.2: total_length 17.63631, total_area 88.218759.
                self.assertAlmostEqual(float(tree.total_length.to_decimal(u.um)), 17.63631, places=5)
                self.assertAlmostEqual(float(tree.total_area.to_decimal(u.um**2)), 88.218759, places=6)

    def test_soma_names_are_matched_exactly(self) -> None:
        for name in ('"soma"', '"Cell body"', '"CellBody "'):
            with self.subTest(name=name):
                text = _GAP_FIXTURES["soma_name_no_tag"].replace('"Soma"', name)
                _, report = AscReader().read(self._write_asc(text), return_report=True)

                self.assertIn("topology.synthetic_soma", [issue.code for issue in report.issues])

    def test_cellbody_marker_with_another_name_still_makes_the_soma(self) -> None:
        # A known difference: NEURON only goes by the name and drops this soma.
        tree = AscReader().read(self._write_asc(_GAP_FIXTURES["tag_other_name"]))

        self.assertEqual(tree.root.type, "soma")
        self.assertEqual(round(float(tree.total_area.to_decimal(u.um**2)), 6), 88.218759)

    def test_synthetic_soma_message_counts_the_trees_and_the_farthest_root(self) -> None:
        tree, report = AscReader().read(self._write_asc(_GAP_FIXTURES["no_soma_two_trees"]), return_report=True)

        synthetic = [issue for issue in report.issues if issue.code == "topology.synthetic_soma"]
        self.assertEqual(len(synthetic), 1)
        self.assertIn("2 trees", synthetic[0].message)
        self.assertIn("50 um", synthetic[0].message)
        self.assertEqual(float(tree.total_length.to_decimal(u.um)), 21.0)

    def test_two_cellbody_groups_ask_for_one_cell_per_file(self) -> None:
        path = self._write_asc(_GAP_FIXTURES["two_cellbody_groups"])

        with self.assertRaisesRegex(ValueError, "found 2 disjoint CellBody contour groups.*one cell per file"):
            AscReader().read(path)


class AscContourPrincipalAxisTest(_AscTestMixin, unittest.TestCase):
    """A single CellBody contour is sampled along NEURON's 3D principal axis (MORPH-5d)."""

    def test_nonplanar_contour_soma_follows_the_3d_major_axis(self) -> None:
        tree = AscReader().read(self._write_asc(_GAP_FIXTURES["nonplanar_contour"]))
        soma = tree.root.branch
        points = np.asarray(soma.points.to_decimal(u.um), dtype=float)
        diameters = 2.0 * np.asarray(soma.radii.to_decimal(u.um), dtype=float)

        # float64 values. NEURON 9.0.2 stores pt3d in float32 and gives
        # (-3.358562707901001, 2.315037965774536, 1.5604015588760376),
        # (4.026669502258301, 0.6830273866653442, 0.9303712844848633) and a
        # soma area of 56.888621768205184.
        self.assertEqual(points.shape, (21, 3))
        np.testing.assert_allclose(points[0], [-3.3585626261135237, 2.3150378826073803, 1.5604016078799239], rtol=1e-12)
        np.testing.assert_allclose(points[-1], [4.026669602396029, 0.6830273875075027, 0.9303712659878458], rtol=1e-12)
        np.testing.assert_allclose(diameters[[0, -1]], [0.3776128883207476, 0.26736422871776877], rtol=1e-12)
        np.testing.assert_allclose(float(soma.area.to_decimal(u.um**2)), 56.88862183431629, rtol=1e-12)
        np.testing.assert_allclose(float(tree.total_area.to_decimal(u.um**2)), 88.30454837021422, rtol=1e-12)

    def test_contour_whose_minor_axis_is_vertical_raises_like_neuron(self) -> None:
        # Traced in the xz plane: the middle principal axis is z, which has no
        # width in the xy plane. NEURON 9.0.2 stops with "Failed to compute
        # soma centroid from contour."
        path = self._write_asc(_GAP_FIXTURES["vertical_closed_contour"])

        with self.assertRaisesRegex(
            ValueError, "CellBody contour at line 1 is degenerate: its minor axis is parallel to z"
        ):
            AscReader().read(path)


class AscRealFileSmokeTest(unittest.TestCase):
    def test_valid_real_asc_fixtures_pass_smoke_checks(self) -> None:
        reader = AscReader()
        for fixture_name in ("goc.asc", "pc.asc"):
            with self.subTest(fixture=fixture_name):
                tree, report = reader.read(FIXTURE_DIR / fixture_name, return_report=True)

                self.assertIsInstance(tree, Morphology)
                self.assertFalse(report.has_errors)
                self.assertGreater(len(tree.branches), 0)
                self.assertEqual(tree.root.type, "soma")
                self.assertEqual(tree.root.branch.points.shape[0], 21)
                self.assertTrue(tree.topo())
                self.assertTrue(all(branch.type in ALLOWED_TYPES for branch in tree.branches))

    def test_valid_real_asc_fixtures_support_morpho_from_asc(self) -> None:
        for fixture_name in ("goc.asc", "pc.asc"):
            with self.subTest(fixture=fixture_name):
                path = FIXTURE_DIR / fixture_name
                tree = Morphology.from_asc(path)
                tree_with_report, report = Morphology.from_asc(path, return_report=True)

                self.assertIsInstance(tree, Morphology)
                self.assertIsInstance(tree_with_report, Morphology)
                self.assertFalse(report.has_errors)
                self.assertGreater(len(tree.branches), 0)
                self.assertEqual(tree.root.type, "soma")
                self.assertEqual(tree.root.branch.points.shape[0], 21)
                self.assertTrue(tree.topo())


@unittest.skipIf(_NEURON_H is None, "NEURON import3d is not available")
class AscNeuronParityTest(_AscTestMixin, unittest.TestCase):
    def _instantiate_neuron_sections(self, path: Path):
        assert _NEURON_H is not None
        _NEURON_H.load_file("stdlib.hoc")
        _NEURON_H.load_file("import3d.hoc")
        reader = _NEURON_H.Import3d_Neurolucida3()
        reader.input(str(path))
        existing_sections = tuple(_NEURON_H.allsec())
        _NEURON_H.Import3d_GUI(reader, 0).instantiate(None)
        return tuple(sec for sec in _NEURON_H.allsec() if sec not in existing_sections)

    def _neuron_root_soma_pt3d(self, path: Path) -> np.ndarray:
        assert _NEURON_H is not None
        new_sections = self._instantiate_neuron_sections(path)
        try:
            soma_roots = []
            for sec in new_sections:
                ref = _NEURON_H.SectionRef(sec=sec)
                if not ref.has_parent() and sec.name().startswith("soma"):
                    soma_roots.append(sec)
            self.assertEqual(len(soma_roots), 1)
            soma = soma_roots[0]
            n3d = int(_NEURON_H.n3d(sec=soma))
            return np.asarray(
                [
                    [
                        float(_NEURON_H.x3d(index, sec=soma)),
                        float(_NEURON_H.y3d(index, sec=soma)),
                        float(_NEURON_H.z3d(index, sec=soma)),
                        float(_NEURON_H.diam3d(index, sec=soma)),
                    ]
                    for index in range(n3d)
                ],
                dtype=float,
            )
        finally:
            for sec in new_sections:
                try:
                    _NEURON_H.delete_section(sec=sec)
                except Exception:
                    pass

    def _braincell_root_soma_pt3d(self, tree: Morphology) -> np.ndarray:
        points_um = np.asarray(tree.root.branch.points.to_decimal(u.um), dtype=float)
        diam_um = 2.0 * np.asarray(tree.root.branch.radii.to_decimal(u.um), dtype=float)
        return np.column_stack([points_um, diam_um])

    def test_simple_cellbody_matches_neuron_pt3d(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-6, rtol=0.0))

    def test_multicontour_cellbody_matches_neuron_pt3d(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            {_SOMA_Z5}
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            '''
        )

        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (2, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-6, rtol=0.0))

    def test_nonplanar_cellbody_matches_neuron_pt3d(self) -> None:
        path = self._write_asc(_GAP_FIXTURES["nonplanar_contour"])

        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-6, rtol=0.0))

    def test_real_fixture_goc_soma_matches_neuron_pt3d(self) -> None:
        path = FIXTURE_DIR / "goc.asc"
        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-5, rtol=0.0))

    def test_real_fixture_grc_soma_matches_neuron_pt3d(self) -> None:
        path = CEREBELLUM_FIXTURES["GrC"]
        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-5, rtol=0.0))

    def test_angle_spine_demo_matches_neuron_parent_section_geometry(self) -> None:
        path = self._write_asc(
            f'''
            {_SOMA}
            ( (Dendrite)
              (0 0 0 1)
              <
                (Class 4 "none")
                (Color Red)
                (Generated 0)
                (1 1 0 0.1)
              >
              (0 5 0 1)
            )
            '''
        )

        tree = Morphology.from_asc(path)
        neuron_sections = self._instantiate_neuron_sections(path)
        try:
            self.assertEqual(tuple(sec.name() for sec in neuron_sections), ("soma[0]", "dend[0]"))
            dend = next(sec for sec in neuron_sections if sec.name() == "dend[0]")
            neuron_pt3d = np.asarray(
                [
                    [
                        float(_NEURON_H.x3d(index, sec=dend)),
                        float(_NEURON_H.y3d(index, sec=dend)),
                        float(_NEURON_H.z3d(index, sec=dend)),
                        float(_NEURON_H.diam3d(index, sec=dend)),
                    ]
                    for index in range(int(_NEURON_H.n3d(sec=dend)))
                ],
                dtype=float,
            )
        finally:
            for sec in neuron_sections:
                try:
                    _NEURON_H.delete_section(sec=sec)
                except Exception:
                    pass

        braincell_dend = tree.branch(index=1).branch
        braincell_pt3d = np.column_stack(
            [
                np.asarray(braincell_dend.points.to_decimal(u.um), dtype=float),
                2.0 * np.asarray(braincell_dend.radii.to_decimal(u.um), dtype=float),
            ]
        )
        self.assertEqual(braincell_pt3d.shape, (2, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
