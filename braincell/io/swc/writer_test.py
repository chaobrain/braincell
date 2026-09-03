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
import unittest
from pathlib import Path

import brainunit as u

from braincell import Branch, Morphology
from braincell.io._testing import FIXTURE_DIR, VALID_SWC_FIXTURES
from braincell.io.swc import SwcReader
from braincell.io.swc.writer import write_swc


def _data_rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]


class SwcWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_writes_complete_branch_paths_with_duplicate_attachment_samples(self) -> None:
        soma = Branch.from_points(
            points=[[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0, 5.0] * u.um,
            type="soma",
        )
        basal = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 1.0] * u.um,
            type="basal_dendrite",
        )
        axon = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [0.0, -10.0, 0.0]] * u.um,
            radii=[5.0, 0.5] * u.um,
            type="axon",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.attach(parent="soma", child_branch=basal, child_name="basal", parent_x=1.0)
        morpho.attach(parent="soma", child_branch=axon, child_name="axon", parent_x=0.5)

        path = write_swc(morpho, self.root / "tree.swc")

        self.assertEqual(
            _data_rows(path),
            [
                ["1", "1", "-10", "0", "0", "5", "-1"],
                ["2", "1", "0", "0", "0", "5", "1"],
                ["3", "1", "10", "0", "0", "5", "2"],
                ["4", "3", "10", "0", "0", "5", "3"],
                ["5", "3", "20", "0", "0", "1", "4"],
                ["6", "2", "0", "0", "0", "5", "2"],
                ["7", "2", "0", "-10", "0", "0.5", "6"],
            ],
        )
        report = SwcReader().check(path)
        self.assertFalse(report.has_errors)

    def test_rejects_soma_midpoint_without_existing_internal_sample(self) -> None:
        soma = Branch.from_points(
            points=[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[4.0, 6.0] * u.um,
            type="soma",
        )
        dend = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]] * u.um,
            radii=[5.0, 1.0] * u.um,
            type="apical_dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.attach(parent="soma", child_branch=dend, child_name="dend", parent_x=0.5)

        destination = self.root / "midpoint.swc"

        with self.assertRaisesRegex(ValueError, "existing internal soma sample"):
            write_swc(morpho, destination)

        self.assertFalse(destination.exists())

    def test_reverses_child_attached_at_distal_end(self) -> None:
        soma = Branch.from_points(
            points=[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        dend = Branch.from_points(
            points=[[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 5.0] * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.attach(
            parent="soma",
            child_branch=dend,
            child_name="dend",
            parent_x=1.0,
            child_x=1.0,
        )

        rows = _data_rows(write_swc(morpho, self.root / "reversed.swc"))

        self.assertEqual(rows[-1], ["4", "3", "20", "0", "0", "1", "3"])

    def test_round_trip_resolves_con2prox_attachment_by_matching_point(self) -> None:
        source = self.root / "con2prox.swc"
        source.write_text(
            "\n".join(
                (
                    "1 1 0 0 -1 1 -1",
                    "2 1 0 0 0 1 1",
                    "3 3 0 0 1 1 2",
                    "4 3 0 0 2 1 3",
                    "5 3 1 0 1 1 3",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        original = Morphology.from_swc(source)

        path = write_swc(original, self.root / "con2prox-roundtrip.swc")
        rows = _data_rows(path)
        restored = Morphology.from_swc(path)

        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[2], ["3", "3", "0", "0", "0", "1", "2"])
        self.assertEqual(rows[5], ["6", "3", "0", "0", "1", "1", "4"])
        self.assertEqual(restored, original)

    def test_soma_midpoint_does_not_require_matching_child_endpoint(self) -> None:
        source = self.root / "four-point-soma.swc"
        source.write_text(
            "\n".join(
                (
                    "1 1 0 0 0 2 -1",
                    "2 1 10 0 0 2 1",
                    "3 1 20 0 0 2 2",
                    "4 1 30 0 0 2 3",
                    "5 3 10 10 0 1 2",
                    "6 3 10 20 0 0.5 5",
                    "7 2 20 -10 0 1 3",
                    "8 2 20 -20 0 0.5 7",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        original = Morphology.from_swc(source)

        path = write_swc(original, self.root / "four-point-soma-roundtrip.swc")
        restored = Morphology.from_swc(path)

        self.assertEqual(restored, original)
        self.assertTrue(all(float(child.parent_x) == 0.5 for child in original.soma.children))

    def test_rejects_internal_connection_without_soma_upstream(self) -> None:
        parent = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 2.0, 2.0] * u.um,
            type="axon",
        )
        child = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(parent, name="axon")
        morpho.attach(parent="axon", child_branch=child, child_name="dend", parent_x=0.0)

        with self.assertRaisesRegex(ValueError, "does not match the declared anchor"):
            write_swc(morpho, self.root / "invalid-internal.swc")

    def test_rejects_con2prox_when_parent_start_cannot_collapse_into_soma(self) -> None:
        soma = Branch.from_points(
            points=[[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 2.0, 2.0] * u.um,
            type="soma",
        )
        main = Branch.from_points(
            points=[[10.0, 10.0, 0.0], [20.0, 10.0, 0.0], [30.0, 10.0, 0.0]] * u.um,
            radii=[1.0, 1.0, 1.0] * u.um,
            type="dendrite",
        )
        side = Branch.from_points(
            points=[[20.0, 10.0, 0.0], [20.0, 20.0, 0.0]] * u.um,
            radii=[1.0, 0.5] * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.attach(parent="soma", child_branch=main, child_name="main", parent_x=0.5)
        morpho.attach(parent="main", child_branch=side, child_name="side", parent_x=0.0)

        with self.assertRaisesRegex(ValueError, "does not match the declared anchor"):
            write_swc(morpho, self.root / "invalid-con2prox.swc")

    def test_round_trip_preserves_radius_jump_at_attachment(self) -> None:
        source = self.root / "attachment-radius-jump-source.swc"
        source.write_text(
            "\n".join(
                (
                    "1 2 0 0 0 2 -1",
                    "2 2 10 0 0 2 1",
                    "3 3 10 0 0 1 2",
                    "4 3 20 0 0 0.5 3",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        morpho = Morphology.from_swc(source)

        path = write_swc(morpho, self.root / "attachment-radius-jump.swc")
        restored = Morphology.from_swc(path)

        self.assertEqual(_data_rows(path)[2][2:6], ["10", "0", "0", "2"])
        self.assertEqual(_data_rows(path)[3][2:6], ["10", "0", "0", "1"])
        self.assertEqual(restored, morpho)

    def test_rejects_connection_without_matching_parent_sample_before_writing(self) -> None:
        parent = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 2.0] * u.um,
            type="axon",
        )
        child = Branch.from_points(
            points=[[11.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 0.5] * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(parent, name="axon")
        morpho.attach(parent="axon", child_branch=child, child_name="dend", parent_x=1.0)
        destination = self.root / "not-created" / "disconnected.swc"

        with self.assertRaisesRegex(ValueError, "does not match the declared anchor on parent 'axon'"):
            write_swc(morpho, destination)

        self.assertFalse(destination.parent.exists())

    def test_rejects_connection_with_matching_xyz_but_different_radius(self) -> None:
        parent = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[2.0, 2.0] * u.um,
            type="axon",
        )
        child = Branch.from_points(
            points=[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 0.5] * u.um,
            type="dendrite",
        )
        morpho = Morphology.from_root(parent, name="axon")
        morpho.attach(parent="axon", child_branch=child, child_name="dend", parent_x=1.0)

        with self.assertRaisesRegex(ValueError, "does not match the declared anchor"):
            write_swc(morpho, self.root / "radius-mismatch.swc")

    def test_maps_every_branch_type_to_standard_code(self) -> None:
        expected = {
            "custom": "0",
            "soma": "1",
            "axon": "2",
            "dendrite": "3",
            "basal_dendrite": "3",
            "apical_dendrite": "4",
        }
        for branch_type, type_code in expected.items():
            with self.subTest(branch_type=branch_type):
                branch = Branch.from_points(
                    points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * u.um,
                    radii=[1.0, 0.5] * u.um,
                    type=branch_type,
                )
                morpho = Morphology.from_root(branch, name="cable")
                rows = _data_rows(write_swc(morpho, self.root / f"{branch_type}.swc"))
                self.assertTrue(all(row[1] == type_code for row in rows))

    def test_preserves_radius_jumps_as_repeated_xyz_samples(self) -> None:
        branch = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]] * u.um,
            radii_proximal=[2.0, 1.0] * u.um,
            radii_distal=[2.0, 1.0] * u.um,
            type="axon",
        )
        morpho = Morphology.from_root(branch, name="cable")

        rows = _data_rows(write_swc(morpho, self.root / "jump.swc"))

        self.assertEqual(rows[1][2:6], ["10", "0", "0", "2"])
        self.assertEqual(rows[2][2:6], ["10", "0", "0", "1"])

    def test_missing_points_fail_before_destination_is_touched(self) -> None:
        soma = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]] * u.um,
            radii=[5.0, 5.0] * u.um,
            type="soma",
        )
        dend = Branch.from_lengths(
            lengths=[10.0] * u.um,
            radii=[1.0, 0.5] * u.um,
            type="basal_dendrite",
        )
        morpho = Morphology.from_root(soma, name="soma")
        morpho.soma.dend = dend
        destination = self.root / "not-created" / "cell.swc"

        with self.assertRaisesRegex(ValueError, "missing points on: 'dend'"):
            write_swc(morpho, destination)

        self.assertFalse(destination.parent.exists())

    def test_suffixless_nested_path_is_created_and_existing_file_is_replaced(self) -> None:
        branch = Branch.from_points(
            points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * u.um,
            radii=[1.0, 1.0] * u.um,
            type="custom",
        )
        morpho = Morphology.from_root(branch, name="cable")
        destination = self.root / "nested" / "cell.swc"
        destination.parent.mkdir(parents=True)
        destination.write_text("old data", encoding="utf-8")

        written = write_swc(morpho, destination.with_suffix(""))

        self.assertEqual(written, destination)
        self.assertNotIn("old data", destination.read_text(encoding="utf-8"))
        self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_existing_swc_fixtures_export_and_read_back(self) -> None:
        for fixture_name in VALID_SWC_FIXTURES:
            with self.subTest(fixture=fixture_name):
                morpho = Morphology.from_swc(FIXTURE_DIR / fixture_name)
                path = morpho.to_swc(self.root / f"roundtrip-{fixture_name}")
                report = SwcReader().check(path)
                loaded = Morphology.from_swc(path)

                self.assertFalse(report.has_errors)
                self.assertTrue(loaded.has_full_point_geometry)
                self.assertEqual(len(loaded.edges), len(loaded.branches) - 1)
                self.assertEqual(loaded, morpho)


if __name__ == "__main__":
    unittest.main()
