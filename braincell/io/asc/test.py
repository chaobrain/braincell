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


import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

from braincell import Morphology
from braincell._misc import u
from braincell.io.asc import AscReader, AscSpineRecord

try:
    from neuron import h as _NEURON_H
except Exception:  # pragma: no cover
    _NEURON_H = None

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "examples" / "multi_compartment" / "morpho_files"
ALLOWED_TYPES = {
    "soma",
    "axon",
    "dendrite",
    "basal_dendrite",
    "apical_dendrite",
    "custom",
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
            '''
            ; example asc
            ("Cell Body"
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ("Cell Body"
              (CellBody)
              (0 0 5 0)
              (5 0 5 0)
              (0 5 5 0)
              (-5 0 5 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ("Cell Body"
              (CellBody)
              (0 0 5 0)
              (5 0 5 0)
              (0 5 5 0)
              (-5 0 5 0)
            )
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
            '''
            ("Cell Body"
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
        path = FIXTURE_DIR / "Cerebellum_morph" / "SC.asc"

        _, report = AscReader().read(path, return_report=True)

        self.assertNotIn("syntax.point_after_children", self._issue_codes(report))

    def test_bc_fixture_reports_root_outside_soma_bbox_warnings(self) -> None:
        path = FIXTURE_DIR / "Cerebellum_morph" / "BC.asc"

        _, report = AscReader().read(path, return_report=True)

        codes = self._issue_codes(report)
        self.assertNotIn("syntax.point_after_children", codes)
        self.assertEqual(sum(code == "topology.root_outside_soma_bbox" for code in codes), 7)


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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ("Cell Body"
              (CellBody)
              (0 0 5 0)
              (5 0 5 0)
              (0 5 5 0)
              (-5 0 5 0)
            )
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

    def test_real_fixture_goc_soma_matches_neuron_pt3d(self) -> None:
        path = FIXTURE_DIR / "goc.asc"
        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-5, rtol=0.0))

    def test_real_fixture_grc_soma_matches_neuron_pt3d(self) -> None:
        path = FIXTURE_DIR / "Cerebellum_morph" / "GrC.asc"
        tree = Morphology.from_asc(path)
        neuron_pt3d = self._neuron_root_soma_pt3d(path)
        braincell_pt3d = self._braincell_root_soma_pt3d(tree)

        self.assertEqual(braincell_pt3d.shape, (21, 4))
        self.assertEqual(braincell_pt3d.shape, neuron_pt3d.shape)
        self.assertTrue(np.allclose(braincell_pt3d, neuron_pt3d, atol=1e-5, rtol=0.0))

    def test_angle_spine_demo_matches_neuron_parent_section_geometry(self) -> None:
        path = self._write_asc(
            '''
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
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
