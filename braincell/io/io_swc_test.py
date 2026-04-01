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

import tempfile
import textwrap
import unittest
from pathlib import Path

from braincell._test_support import np, u

from braincell import Morpho, SwcReadOptions, SwcReader
from braincell.io.swc.soma import is_contour_soma, is_special_three_point_soma
from braincell.io.swc.types import _SwcAttach, _SwcBranch, _SwcRow


class SwcReaderTest(unittest.TestCase):
    def _morpho_file(self, name: str) -> Path:
        return Path(__file__).with_name("morpho_files") / name

    def _write_swc(self, body: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.swc"
        path.write_text(textwrap.dedent(body).strip() + "\n")
        return path

    def _issue_codes(self, report) -> set[str]:
        return {issue.code for issue in report.issues}

    def _soma_row(
        self,
        node_id: int,
        x: float,
        y: float,
        z: float = 0.0,
        radius: float = 5.0,
        parent_id: int = -1,
    ) -> _SwcRow:
        return _SwcRow(
            line_number=node_id,
            fields=tuple(),
            node_id=node_id,
            type_code=1,
            x=x,
            y=y,
            z=z,
            radius=radius,
            parent_id=parent_id,
        )

    def _branch_points_um(self, branch) -> np.ndarray:
        if branch.points_proximal is None or branch.points_distal is None:
            raise AssertionError("Branch does not expose point geometry.")
        points = [branch.points_proximal[0].to_decimal(u.um)]
        points.extend(point.to_decimal(u.um) for point in branch.points_distal)
        return np.array(points, dtype=float)

    def _branch_point_radii_um(self, branch) -> np.ndarray:
        radii = [branch.radii_proximal[0].to_decimal(u.um)]
        radii.extend(radius.to_decimal(u.um) for radius in branch.radii_distal)
        return np.array(radii, dtype=float)

    def test_contour_rule_uses_start_mid_end_curvature_angle(self) -> None:
        acute_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 5.0, 1.0),
            self._soma_row(3, 0.0, 2.0),
            self._soma_row(4, -5.0, 1.0),
        )
        right_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 5.0, 5.0),
            self._soma_row(3, 8.0, 1.0),
            self._soma_row(4, 10.0, 0.0),
        )
        obtuse_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 3.0, 4.0),
            self._soma_row(3, 7.0, 4.0),
            self._soma_row(4, 10.0, 0.0),
        )

        self.assertTrue(is_contour_soma(acute_rows))
        self.assertTrue(is_contour_soma(right_rows))
        self.assertFalse(is_contour_soma(obtuse_rows))

    def test_special_three_point_rule_uses_center_first_topology(self) -> None:
        special_rows = (
            self._soma_row(1, 0.0, 0.0, radius=5.0, parent_id=-1),
            self._soma_row(2, -5.0, 0.0, radius=5.0, parent_id=1),
            self._soma_row(3, 0.0, 5.0, radius=5.0, parent_id=1),
        )
        chain_rows = (
            self._soma_row(1, -5.0, 0.0, radius=5.0, parent_id=-1),
            self._soma_row(2, 0.0, 0.0, radius=5.0, parent_id=1),
            self._soma_row(3, 5.0, 0.0, radius=5.0, parent_id=2),
        )

        is_special, ordered = is_special_three_point_soma(special_rows)
        self.assertTrue(is_special)
        self.assertIsNotNone(ordered)
        self.assertEqual(ordered[0].node_id, 1)
        self.assertFalse(is_special_three_point_soma(chain_rows)[0])

    def test_reader_builds_compressed_tree_and_maps_known_types(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 0 10 0 2 1
            3 3 0 20 0 1 2
            4 2 10 0 0 1 1
            5 2 20 0 0 0.5 4
            """
        )

        tree = SwcReader().read(path)

        self.assertEqual(len(tree.branches), 3)
        self.assertEqual(tree.root.name, "soma")
        self.assertEqual(tree.branch(name="basal_dendrite_0").type, "basal_dendrite")
        self.assertEqual(tree.branch(name="axon_0").type, "axon")
        self.assertAlmostEqual(tree.root.length.to_decimal(u.um), 20.0)
        self.assertEqual(tree.edges[0].parent_x, 0.5)
        self.assertEqual(tree.edges[1].parent_x, 0.5)
        self.assertEqual(tree.edges[0].child_x, 0.0)
        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "├── basal_dendrite_0",
                    "└── axon_0",
                )
            ),
        )

    def test_reader_extends_root_branch_when_generic_root_immediately_branches(self) -> None:
        path = self._write_swc(
            """
            1 2 0 0 0 5 -1
            2 2 10 0 0 4 1
            3 2 -10 0 0 4 1
            4 3 20 0 0 2 2
            """
        )

        tree = SwcReader().read(path)
        axon_children = [branch for branch in tree.branches if branch.parent is not None and branch.type == "axon"]

        self.assertEqual(len(axon_children), 1)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(tree.root.branch),
                np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                self._branch_points_um(axon_children[0].branch),
                np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
            )
        )
        self.assertEqual(axon_children[0].parent_x, 0.0)

    def test_reader_applies_con2prox_for_first_non_soma_branch_under_soma(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 10 0 0 2 1
            3 3 20 0 0 1 2
            4 3 10 10 0 1 2
            """
        )

        tree = SwcReader().read(path)
        main = tree.branch(name="basal_dendrite_0")
        sibling = tree.branch(name="basal_dendrite_1")

        self.assertEqual(main.parent.name, "soma")
        self.assertEqual(main.parent_x, 0.5)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(main.branch),
                np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            )
        )
        self.assertEqual(sibling.parent.name, "basal_dendrite_0")
        self.assertEqual(sibling.parent_x, 0.0)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(sibling.branch),
                np.array([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]]),
            )
        )
        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "└── basal_dendrite_0",
                    "    └── basal_dendrite_1",
                )
            ),
        )

    def test_reader_applies_con2prox_to_mixed_type_siblings_under_soma(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 10 0 0 2 1
            3 3 20 0 0 1 2
            4 4 10 10 0 1 2
            """
        )

        tree = SwcReader().read(path)
        main = tree.branch(name="basal_dendrite_0")
        side = tree.branch(name="apical_dendrite_0")

        self.assertEqual(main.parent.name, "soma")
        self.assertEqual(main.parent_x, 0.5)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(main.branch),
                np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            )
        )
        self.assertEqual(side.parent.name, "basal_dendrite_0")
        self.assertEqual(side.parent_x, 0.0)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(side.branch),
                np.array([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]]),
            )
        )

    def test_single_point_soma_expands_to_three_points_and_connects_at_midpoint(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 0 10 0 2 1
            """
        )

        tree = SwcReader().read(path)
        soma_points = self._branch_points_um(tree.root.branch)
        dend_points = self._branch_points_um(tree.branch(name="basal_dendrite_0").branch)
        dend_radii = self._branch_point_radii_um(tree.branch(name="basal_dendrite_0").branch)

        self.assertTrue(np.allclose(soma_points[1], np.array([0.0, 0.0, 0.0])))
        self.assertEqual({tuple(point) for point in soma_points[[0, 2]]}, {(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)})
        self.assertTrue(np.allclose(dend_points[0], np.array([0.0, 0.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], dend_radii[1])
        self.assertNotAlmostEqual(dend_radii[0], 10.0)
        self.assertEqual(tree.edges[0].parent_x, 0.5)

    def test_special_three_point_soma_uses_center_point_as_attachment_site(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 1 -10 0 0 10 1
            3 1 0 10 0 10 1
            4 3 0 10 0 2 1
            """
        )

        tree = SwcReader().read(path)
        soma_points = self._branch_points_um(tree.root.branch)
        dend_points = self._branch_points_um(tree.branch(name="basal_dendrite_0").branch)
        dend_radii = self._branch_point_radii_um(tree.branch(name="basal_dendrite_0").branch)

        self.assertTrue(np.allclose(soma_points[1], np.array([0.0, 0.0, 0.0])))
        self.assertEqual(
            {tuple(point) for point in soma_points[[0, 2]]},
            {(-10.0, 0.0, 0.0), (10.0, 0.0, 0.0)},
        )
        self.assertTrue(np.allclose(dend_points[0], np.array([0.0, 0.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], dend_radii[1])
        self.assertNotAlmostEqual(dend_radii[0], 10.0)
        self.assertEqual(tree.edges[0].parent_x, 0.5)

    def test_nonbranching_multi_point_soma_uses_midpoint_attachment_for_internal_children(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 5 -1
            2 1 10 0 0 5 1
            3 1 20 0 0 5 2
            4 3 10 10 0 2 2
            5 3 20 10 0 2 3
            """
        )

        tree = SwcReader().read(path)
        self.assertEqual(len(tree.branches), 3)
        self.assertEqual(tree.edges[0].parent_x, 0.5)
        self.assertEqual(tree.edges[1].parent_x, 1.0)

    def test_branched_multi_point_soma_splits_into_multiple_soma_sections(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 5 -1
            2 1 10 0 0 5 1
            3 1 20 0 0 5 2
            4 1 10 10 0 5 2
            5 3 20 10 0 2 4
            """
        )

        tree = SwcReader().read(path)
        soma_names = [branch.name for branch in tree.branches if branch.type == "soma"]
        self.assertEqual(len(soma_names), 3)
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent_x, 1.0)

    def test_branched_multi_point_soma_uses_section_attach_positions(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 5 -1
            2 1 10 0 0 5 1
            3 1 20 0 0 5 2
            4 1 30 0 0 5 3
            5 1 10 10 0 5 2
            6 3 -10 0 0 2 1
            7 3 10 20 0 2 5
            8 3 40 0 0 2 4
            """
        )

        tree = SwcReader().read(path)
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent_x, 0.0)
        self.assertEqual(tree.branch(name="basal_dendrite_1").parent_x, 1.0)
        self.assertEqual(tree.branch(name="basal_dendrite_2").parent_x, 1.0)

    def test_single_point_original_branched_soma_section_uses_distal_attachment_after_copy(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 5 -1
            2 1 10 0 0 5 1
            3 1 20 0 0 5 2
            4 1 10 10 0 5 2
            5 3 20 10 0 2 4
            6 3 10 20 0 2 4
            """
        )

        tree = SwcReader().read(path)
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent_x, 1.0)
        self.assertEqual(tree.branch(name="basal_dendrite_1").parent_x, 1.0)

    def test_branched_soma_child_sections_copy_parent_branch_point_before_attachment_x(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 100 10 -1
            2 1 0 0 0 10 1
            3 1 10 0 0 10 2
            4 1 20 0 0 10 3
            5 1 -5 0 0 10 2
            6 1 -10 0 0 10 5
            7 3 100 0 0 1 2
            """
        )

        tree = SwcReader().read(path)
        soma_branches = [branch.branch for branch in tree.branches if branch.type == "soma"]

        self.assertEqual(len(soma_branches), 3)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(soma_branches[0]),
                np.array([[0.0, 0.0, 100.0], [0.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                self._branch_points_um(soma_branches[1]),
                np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                self._branch_points_um(soma_branches[2]),
                np.array([[0.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
            )
        )
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent_x, 1.0)

    def test_reader_imports_root_immediate_branched_soma_file(self) -> None:
        tree = SwcReader().read(self._morpho_file("branched_soma_1.swc"))
        soma_children = [branch for branch in tree.branches if branch.parent is not None and branch.type == "soma"]

        self.assertEqual(len(tree.branches), 3)
        self.assertEqual(len(soma_children), 1)
        self.assertTrue(
            np.allclose(
                self._branch_points_um(tree.root.branch),
                np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            )
        )
        self.assertTrue(
            np.allclose(
                self._branch_points_um(soma_children[0].branch),
                np.array([[0.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]),
            )
        )
        self.assertEqual(soma_children[0].parent_x, 0.0)
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent_x, 0.0)

    def test_reader_imports_bundled_io_file_without_single_point_root_error(self) -> None:
        tree = SwcReader().read(self._morpho_file("io.swc"))

        self.assertGreater(len(tree.branches), 1)

    def test_regular_three_point_soma_uses_file_attachment_point(self) -> None:
        path = self._write_swc(
            """
            1 1 -10 0 0 10 -1
            2 1 0 0 0 5 1
            3 1 10 0 0 10 2
            4 3 10 10 0 2 3
            """
        )

        tree = SwcReader().read(path)
        dend_points = self._branch_points_um(tree.branch(name="basal_dendrite_0").branch)
        dend_radii = self._branch_point_radii_um(tree.branch(name="basal_dendrite_0").branch)

        self.assertTrue(np.allclose(dend_points[0], np.array([10.0, 0.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], dend_radii[1])
        self.assertNotAlmostEqual(dend_radii[0], 10.0)
        self.assertEqual(tree.edges[0].parent_x, 1.0)

    def test_contour_soma_becomes_equivalent_single_point_then_midpoint_attachment(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 5 -1
            2 1 5 1 0 5 1
            3 1 0 2 0 5 2
            4 1 -5 1 0 5 3
            5 3 0 12 0 2 2
            """
        )

        tree, report = SwcReader().read(path, return_report=True)
        soma_points = self._branch_points_um(tree.root.branch)
        dend_points = self._branch_points_um(tree.branch(name="basal_dendrite_0").branch)
        dend_radii = self._branch_point_radii_um(tree.branch(name="basal_dendrite_0").branch)

        self.assertIn("semantics.contour", self._issue_codes(report))
        self.assertTrue(np.allclose(soma_points[1], np.array([0.0, 1.0, 0.0])))
        self.assertTrue(np.allclose(dend_points[0], np.array([0.0, 1.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], dend_radii[1])
        self.assertNotAlmostEqual(dend_radii[0], 10.0)
        self.assertEqual(tree.edges[0].parent_x, 0.5)

    def test_make_branch_reuses_same_xyz_attach_point_without_duplication(self) -> None:
        reader = SwcReader()
        nodes = {
            2: _SwcRow(
                line_number=2,
                fields=tuple(),
                node_id=2,
                type_code=2,
                x=10.0,
                y=0.0,
                z=0.0,
                radius=4.0,
                parent_id=1,
            ),
            3: _SwcRow(
                line_number=3,
                fields=tuple(),
                node_id=3,
                type_code=3,
                x=10.0,
                y=0.0,
                z=0.0,
                radius=2.0,
                parent_id=2,
            ),
            4: _SwcRow(
                line_number=4,
                fields=tuple(),
                node_id=4,
                type_code=3,
                x=20.0,
                y=0.0,
                z=0.0,
                radius=1.0,
                parent_id=3,
            ),
        }
        branch = _SwcBranch(
            point_ids=(3, 4),
            branch_type="basal_dendrite",
            parent_index=0,
            start_node_id=3,
            attach=_SwcAttach(node_id=2),
        )

        dend = reader._make_branch(branch, nodes, parent_branch_type="axon")
        dend_points = self._branch_points_um(dend)
        dend_radii = self._branch_point_radii_um(dend)

        self.assertEqual(dend_points.shape, (2, 3))
        self.assertTrue(np.allclose(dend_points[0], np.array([10.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(dend_points[1], np.array([20.0, 0.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], 4.0)
        self.assertAlmostEqual(dend_radii[1], 1.0)
        self.assertGreater(np.linalg.norm(dend_points[1] - dend_points[0]), 0.0)

    def test_reader_merges_duplicate_xyzr_parent_child_nodes(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 10 0 0 2 1
            3 3 10 0 0 2 2
            4 3 20 0 0 1 3
            """
        )

        tree, report = SwcReader().read(path, return_report=True)
        dend = tree.branch(name="basal_dendrite_0").branch
        dend_points = self._branch_points_um(dend)

        self.assertIn("geometry.duplicate_xyzr_node", self._issue_codes(report))
        duplicate_issue = next(issue for issue in report.issues if issue.code == "geometry.duplicate_xyzr_node")
        self.assertTrue(duplicate_issue.fix_applied)
        self.assertEqual(dend_points.shape, (2, 3))
        self.assertTrue(np.allclose(dend_points[0], np.array([10.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(dend_points[1], np.array([20.0, 0.0, 0.0])))

    def test_check_reports_duplicate_xyzr_parent_child_without_applying_fix(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 10 0 0 2 1
            3 3 10 0 0 2 2
            4 3 20 0 0 1 3
            """
        )

        report = SwcReader().check(path)

        self.assertFalse(report.has_errors)
        duplicate_issue = next(issue for issue in report.issues if issue.code == "geometry.duplicate_xyzr_node")
        self.assertFalse(duplicate_issue.fix_applied)

    def test_reader_merges_duplicate_xyzr_chains_before_branch_building(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 3 10 0 0 2 1
            3 3 10 0 0 2 2
            4 3 10 0 0 2 3
            5 3 25 0 0 1 4
            """
        )

        tree, report = SwcReader().read(path, return_report=True)
        dend = tree.branch(name="basal_dendrite_0").branch
        dend_points = self._branch_points_um(dend)

        self.assertEqual(sum(issue.code == "geometry.duplicate_xyzr_node" for issue in report.issues), 2)
        self.assertEqual(dend_points.shape, (2, 3))
        self.assertTrue(np.allclose(dend_points[0], np.array([10.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(dend_points[1], np.array([25.0, 0.0, 0.0])))

    def test_make_branch_keeps_same_xyz_different_radius_as_zero_length_segment(self) -> None:
        reader = SwcReader()
        nodes = {
            2: _SwcRow(
                line_number=2,
                fields=tuple(),
                node_id=2,
                type_code=3,
                x=10.0,
                y=0.0,
                z=0.0,
                radius=2.0,
                parent_id=1,
            ),
            3: _SwcRow(
                line_number=3,
                fields=tuple(),
                node_id=3,
                type_code=3,
                x=10.0,
                y=0.0,
                z=0.0,
                radius=1.5,
                parent_id=2,
            ),
            4: _SwcRow(
                line_number=4,
                fields=tuple(),
                node_id=4,
                type_code=3,
                x=20.0,
                y=0.0,
                z=0.0,
                radius=1.0,
                parent_id=3,
            ),
        }
        branch = _SwcBranch(
            point_ids=(2, 3, 4),
            branch_type="basal_dendrite",
            parent_index=0,
            start_node_id=2,
        )

        built = reader._make_branch(branch, nodes)

        self.assertTrue(u.math.array_equal(built.lengths, np.array([0.0, 10.0]) * u.um))
        self.assertTrue(u.math.array_equal(built.radii, np.array([2.0, 1.5, 1.0]) * u.um))
        self.assertTrue(
            np.allclose(
                self._branch_points_um(built),
                np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
            )
        )

    def test_non_soma_parent_keeps_parent_attachment_radius(self) -> None:
        path = self._write_swc(
            """
            1 2 0 0 0 5 -1
            2 2 10 0 0 4 1
            3 3 20 0 0 2 2
            """
        )

        tree = SwcReader().read(path)
        dend = tree.branch(name="basal_dendrite_0").branch
        dend_points = self._branch_points_um(dend)
        dend_radii = self._branch_point_radii_um(dend)

        self.assertTrue(np.allclose(dend_points[0], np.array([10.0, 0.0, 0.0])))
        self.assertAlmostEqual(dend_radii[0], 4.0)
        self.assertAlmostEqual(dend_radii[1], 2.0)

    def test_reader_splits_branches_when_swc_type_changes(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 1 10 0 0 10 1
            3 3 20 0 0 2 2
            4 3 30 0 0 1 3
            """
        )

        tree = SwcReader().read(path)

        self.assertEqual(len(tree.branches), 2)
        self.assertEqual(tree.root.name, "soma")
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent.name, "soma")
        self.assertEqual(tree.edges[0].parent_x, 1.0)
        self.assertEqual(
            tree.topo(),
            "\n".join(
                (
                    "soma",
                    "└── basal_dendrite_0",
                )
            ),
        )

    def test_reader_downgrades_unknown_types_to_custom(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 5 0 10 0 2 1
            """
        )

        report = SwcReader().check(path)
        tree = SwcReader().read(path)

        self.assertFalse(report.has_errors)
        self.assertIn("semantics.unknown_type", self._issue_codes(report))
        self.assertEqual(len(tree.branches), 2)
        self.assertEqual(tree.branch(name="custom_0").type, "custom")
        self.assertEqual(tree.branch(name="custom_0").parent.name, "soma")

    def test_check_reports_available_fixes_without_marking_them_applied(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 0
            2 3 0 10 0 2 1
            """
        )

        report = SwcReader().check(path)

        self.assertFalse(report.has_errors)
        self.assertIn("topology.invalid_parent", self._issue_codes(report))
        invalid_parent_issue = next(issue for issue in report.issues if issue.code == "topology.invalid_parent")
        self.assertEqual(invalid_parent_issue.fix_message, "set parent index to -1")
        self.assertFalse(invalid_parent_issue.fix_applied)

    def test_reader_can_return_report_with_applied_fixes(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 0
            2 3 0 10 0 2 1
            """
        )

        tree, report = SwcReader().read(path, return_report=True)

        self.assertEqual(tree.root.name, "soma")
        self.assertEqual(tree.branch(name="basal_dendrite_0").parent.name, "soma")
        self.assertIn("topology.invalid_parent", self._issue_codes(report))
        invalid_parent_issue = next(issue for issue in report.issues if issue.code == "topology.invalid_parent")
        self.assertTrue(invalid_parent_issue.fix_applied)

    def test_report_string_groups_warnings_and_separates_issue_blocks(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 0
            2 3 0 10 0 2 1
            """
        )

        report = SwcReader().check(path)
        text = str(report)

        self.assertIn("SWC report: 2 warnings", text)
        self.assertIn("Warnings\n--------", text)
        self.assertIn("[WARNING] format.low_sample_count", text)
        self.assertIn("[WARNING] topology.invalid_parent", text)
        self.assertIn("\n\n[WARNING] topology.invalid_parent", text)
        self.assertIn("fix: set parent index to -1", text)

    def test_reader_can_reject_unknown_types_when_configured(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            2 5 0 10 0 2 1
            """
        )

        reader = SwcReader(options=SwcReadOptions(unknown_type_as_custom=False))
        report = reader.check(path)

        self.assertTrue(report.has_errors)
        self.assertEqual(report.error_count, 1)
        self.assertIn("semantics.unknown_type", self._issue_codes(report))
        with self.assertRaises(ValueError):
            reader.read(path)

    def test_morpho_from_swc_supports_return_report(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 0
            2 3 0 10 0 2 1
            """
        )

        tree, report = Morpho.from_swc(path, return_report=True)

        self.assertIsInstance(tree, Morpho)
        self.assertFalse(report.has_errors)
        self.assertIn("topology.invalid_parent", self._issue_codes(report))

    def test_reader_error_uses_pretty_error_report(self) -> None:
        path = self._write_swc(
            """
            1 1 0 0 0 10 -1
            1 3 0 10 0 2 1
            """
        )

        with self.assertRaises(ValueError) as exc_info:
            SwcReader().read(path)

        text = str(exc_info.exception)
        self.assertIn(f"SWC validation failed for {path}:", text)
        self.assertIn("SWC report: 1 error", text)
        self.assertIn("Errors\n------", text)
        self.assertIn("[ERROR] identity.duplicate_id", text)
        self.assertNotIn("Warnings\n--------", text)

    def test_reader_reports_invalid_swc_graphs(self) -> None:
        bad_cases = {
            "duplicate_id": """
                1 1 0 0 0 10 -1
                1 3 0 10 0 2 1
            """,
            "missing_parent": """
                1 1 0 0 0 10 -1
                2 3 0 10 0 2 99
            """,
            "multiple_roots": """
                1 1 0 0 0 10 -1
                2 3 0 10 0 2 -1
            """,
        }

        for case_name, body in bad_cases.items():
            with self.subTest(case=case_name):
                path = self._write_swc(body)
                report = SwcReader().check(path)
                self.assertTrue(report.has_errors)
                with self.assertRaises(ValueError):
                    SwcReader().read(path)
