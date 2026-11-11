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

"""Tests for :class:`braincell.io.NeuroMlReader` and ``Morphology.from_neuroml``.

The reference totals are computed from each file, not copied: the length is
the sum of ``|distal - proximal|`` over the segments (``2r`` for a sphere), the
area the sum of libNeuroML's segment areas (frustum side, ``4 pi r^2`` for a
sphere) plus the annulus ``pi |r1^2 - r2^2|`` where a merged joint changes
diameter.
"""

import math
import re
import tempfile
import unittest
import warnings
from pathlib import Path

import brainunit as u

import braincell.io
from braincell import Morphology
from braincell.io import NeuroMlIssue, NeuroMlMetadata, NeuroMlReader, NeuroMlReport
from braincell.io._testing import FIXTURE_DIR
from braincell.io.neuroml2.document import parse_document

DATA_DIR = FIXTURE_DIR / "neuroml2"
RTOL = 1e-12

_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="d">\n'
_SOMA = (
    '<segment id="0"><proximal x="0" y="0" z="0" diameter="10"/><distal x="0" y="10" z="0" diameter="10"/></segment>'
)
_TYPES = '<segmentGroup id="soma_group"><member segment="0"/></segmentGroup>'


def _cell(body: str) -> str:
    return f'{_HEADER}  <cell id="c"><morphology id="m">\n    {body}\n  </morphology></cell>\n</neuroml>\n'


def _segment(
    segment_id: int, parent: int, fraction: float, proximal: str | None, distal: str, diameter: float = 2.0
) -> str:
    proximal_element = "" if proximal is None else f'<proximal {proximal} diameter="{diameter}"/>'
    return (
        f'<segment id="{segment_id}"><parent segment="{parent}" fractionAlong="{fraction}"/>'
        f'{proximal_element}<distal {distal} diameter="{diameter}"/></segment>'
    )


def _length(proximal, distal) -> float:
    return math.dist(proximal[:3], distal[:3])


def reference_totals(path: Path, report: NeuroMlReport, cell_id: str | None = None) -> tuple[float, float]:
    """Total length and area of the cell computed from the file's segments."""
    document = parse_document(path, cell_id=cell_id)
    branches = report.metadata.segment_branches
    total_length = total_area = 0.0
    for segment in document.segments.values():
        r1, r2 = segment.proximal[3] / 2.0, segment.distal[3] / 2.0
        if segment.is_sphere:
            total_length += 2.0 * r1
            total_area += 4.0 * math.pi * r1**2
            continue
        length = _length(segment.proximal, segment.distal)
        total_length += length
        total_area += math.pi * (r1 + r2) * math.sqrt(length**2 + (r1 - r2) ** 2)
        if segment.parent is None:
            continue
        # A merged joint: the segment goes on in its parent's branch from where the parent ends.
        parent_end, start = branches[segment.parent][-1], branches[segment.id][0]
        if parent_end[0] == start[0] and parent_end[2] == start[1]:
            parent_radius = document.segments[segment.parent].distal[3] / 2.0
            total_area += math.pi * abs(r1**2 - parent_radius**2)
    return total_length, total_area


def branch_table(morpho: Morphology) -> list[tuple]:
    return [
        (
            branch.name,
            branch.type,
            None if branch.parent is None else branch.parent.name,
            None if branch.parent is None else float(branch.parent_x),
            branch.branch.n_segments,
            float(branch.length.to_decimal(u.um)),
        )
        for branch in morpho.branches
    ]


class _ReaderTestCase(unittest.TestCase):
    def write(self, text: str, name: str = "cell.nml") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def assert_branch_table(self, morpho: Morphology, expected: list[tuple]) -> None:
        table = branch_table(morpho)
        self.assertEqual([row[:5] for row in table], [row[:5] for row in expected])
        for row, want in zip(table, expected, strict=True):
            self.assertTrue(math.isclose(row[5], want[5], rel_tol=RTOL), (row, want))

    def assert_totals(self, path: Path, cell_id: str | None = None) -> None:
        morpho, report = NeuroMlReader().read(path, cell_id=cell_id, return_report=True)
        length, area = reference_totals(path, report, cell_id)
        self.assertTrue(math.isclose(float(morpho.total_length.to_decimal(u.um)), length, rel_tol=RTOL))
        self.assertTrue(math.isclose(float(morpho.total_area.to_decimal(u.um**2)), area, rel_tol=RTOL))


class BranchTableTest(_ReaderTestCase):
    def test_minimal(self) -> None:
        morpho = Morphology.from_neuroml(DATA_DIR / "minimal.cell.nml")

        # trunk_0 is segment 1, the diameter jump at its joint with segment 2,
        # and the first quarter of segment 2, where segment 3 attaches.
        self.assert_branch_table(
            morpho,
            [
                ("soma", "soma", None, None, 1, 10.0),
                ("trunk_0", "dendrite", "soma", 1.0, 3, 25.0),
                ("trunk_1", "dendrite", "trunk_0", 1.0, 1, 15.0),
                ("axon_0", "axon", "soma", 0.0, 1, 40.0),
                ("dendrite_0", "dendrite", "soma", 0.5, 1, 20.0),
                ("dendrite_1", "dendrite", "trunk_0", 1.0, 1, 10.0),
            ],
        )

    def test_branched(self) -> None:
        morpho = Morphology.from_neuroml(DATA_DIR / "branched.cell.nml")

        # Segment 5 attaches at the start of segment 3, the end of trunk_1.
        tip = math.hypot(5.0, 10.0)
        self.assert_branch_table(
            morpho,
            [
                ("soma", "soma", None, None, 1, 10.0),
                ("trunk_0", "dendrite", "soma", 1.0, 3, 25.0),
                ("trunk_1", "dendrite", "trunk_0", 1.0, 1, 15.0),
                ("trunk_2", "dendrite", "trunk_1", 1.0, 1, 20.0),
                ("axon_0", "axon", "soma", 0.0, 1, 40.0),
                ("dendrite_0", "dendrite", "soma", 0.5, 1, 20.0),
                ("dendrite_1", "dendrite", "trunk_0", 1.0, 1, 10.0),
                ("dendrite_2", "dendrite", "trunk_1", 1.0, 1, 10.0),
                ("dendrite_3", "dendrite", "trunk_2", 1.0, 1, tip),
                ("dendrite_4", "dendrite", "trunk_2", 1.0, 1, tip),
            ],
        )

    def test_spherical_soma(self) -> None:
        morpho = Morphology.from_neuroml(DATA_DIR / "sphere_soma.cell.nml")

        self.assert_branch_table(
            morpho,
            [
                ("soma", "soma", None, None, 2, 12.0),
                ("dendrite_0", "dendrite", "soma", 0.5, 1, 40.0),
                ("dendrite_1", "dendrite", "soma", 0.5, 1, 40.0),
            ],
        )
        # Centred on the sphere's centre.
        points = morpho.soma.branch.points.to_decimal(u.um)
        self.assertEqual(points.tolist(), [[-5.0, 2.0, 3.0], [1.0, 2.0, 3.0], [7.0, 2.0, 3.0]])

    def test_pyr_4_sym_branches_are_its_cable_groups(self) -> None:
        path = DATA_DIR / "pyr_4_sym.cell.nml"
        morpho = Morphology.from_neuroml(path)

        names = [branch.name for branch in morpho.branches]
        self.assertEqual(len(names), 9)
        self.assertEqual(sorted(names), sorted(parse_document(path).cable_groups))

    def test_two_cells_need_a_cell_id(self) -> None:
        path = DATA_DIR / "two_cells.nml"

        with self.assertRaisesRegex(ValueError, r"pass cell_id= to choose one"):
            Morphology.from_neuroml(path)
        self.assert_branch_table(
            Morphology.from_neuroml(path, cell_id="a"),
            [("soma", "soma", None, None, 1, 8.0), ("custom_0", "custom", "soma", 1.0, 1, 22.0)],
        )
        # No type groups: the root is custom and named after its type.
        self.assert_branch_table(Morphology.from_neuroml(path, cell_id="b"), [("custom", "custom", None, None, 1, 6.0)])


class TotalsTest(_ReaderTestCase):
    def test_totals_match_the_file(self) -> None:
        cases = [
            ("minimal.cell.nml", None),
            ("branched.cell.nml", None),
            ("sphere_soma.cell.nml", None),
            ("two_cells.nml", "a"),
            ("two_cells.nml", "b"),
            ("pyr_4_sym.cell.nml", None),
        ]
        for name, cell_id in cases:
            with self.subTest(name=name, cell_id=cell_id):
                self.assert_totals(DATA_DIR / name, cell_id)

    def test_merged_joint_adds_the_annulus(self) -> None:
        # Segments 1 and 2 of minimal.cell.nml meet at diameters 3 and 2.
        _, report = NeuroMlReader().read(DATA_DIR / "minimal.cell.nml", return_report=True)
        _, area = reference_totals(DATA_DIR / "minimal.cell.nml", report)

        self.assertTrue(math.isclose(area, 958.3133390423366 + math.pi * (1.5**2 - 1.0**2), rel_tol=RTOL))


class ReportTest(_ReaderTestCase):
    def test_minimal_report(self) -> None:
        morpho, report = Morphology.from_neuroml(DATA_DIR / "minimal.cell.nml", return_report=True)

        self.assertIsInstance(report, NeuroMlReport)
        self.assertIsInstance(report.metadata, NeuroMlMetadata)
        self.assertEqual(report.issues, [])
        self.assertEqual(str(report), "NeuroML2 report: 0 issues")
        metadata = report.metadata
        self.assertEqual((metadata.cell_id, metadata.morphology_id), ("minimal", "minimal_morph"))
        self.assertEqual(metadata.segment_branches[1], (("trunk_0", 0.0, 0.8),))
        self.assertEqual(metadata.segment_branches[2], (("trunk_0", 0.8, 1.0), ("trunk_1", 0.0, 1.0)))
        self.assertEqual(metadata.segment_branches[5], (("dendrite_0", 0.0, 1.0),))
        self.assertEqual(sorted(metadata.segment_branches), [0, 1, 2, 3, 4, 5])
        self.assertEqual(metadata.segment_names[3], "side")
        self.assertEqual(metadata.segment_groups["trunk"], (1, 2))
        self.assertEqual(metadata.segment_groups["distal_part"], (2, 3))
        self.assertEqual(metadata.ignored_elements, ())
        self.assertEqual(morpho.n_branches, 6)

    def test_ignored_elements_of_a_full_cell(self) -> None:
        path = self.write(
            _HEADER + '  <cell id="full"><notes>a cell</notes><morphology id="m">'
            f"{_SOMA}{_TYPES}</morphology>"
            '<biophysicalProperties id="bio"><membraneProperties>'
            '<channelDensity id="leak" ionChannel="pas" condDensity="3.0 S_per_m2" erev="-70mV" ion="non_specific"/>'
            "</membraneProperties></biophysicalProperties></cell>\n</neuroml>\n"
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assertEqual(report.metadata.ignored_elements, ("biophysicalProperties", "notes"))
        self.assertEqual(report.issues, [])
        self.assertEqual(morpho.n_branches, 1)

    def test_pyr_4_sym_report(self) -> None:
        _, report = Morphology.from_neuroml(DATA_DIR / "pyr_4_sym.cell.nml", return_report=True)

        self.assertEqual(len(report.metadata.segment_groups), 18)
        self.assertEqual(report.metadata.ignored_elements, ("biophysicalProperties", "notes"))
        self.assertEqual(report.issues, [])

    def test_reading_emits_no_python_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for name in ("minimal.cell.nml", "branched.cell.nml", "sphere_soma.cell.nml", "pyr_4_sym.cell.nml"):
                Morphology.from_neuroml(DATA_DIR / name)


class NamingTest(_ReaderTestCase):
    def test_reserved_cable_group_id_is_auto_named_with_a_warning(self) -> None:
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 1, 'x="0" y="10" z="0"', 'x="0" y="30" z="0"')
                + _TYPES
                + '<segmentGroup id="dendrite_group"><member segment="1"/></segmentGroup>'
                '<segmentGroup id="length" neuroLexId="sao864921383"><member segment="1"/></segmentGroup>'
            )
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assertEqual([branch.name for branch in morpho.branches], ["soma", "dendrite_0"])
        self.assertEqual([issue.code for issue in report.issues], ["naming.group_id_unusable"])
        issue = report.issues[0]
        self.assertIsInstance(issue, NeuroMlIssue)
        self.assertEqual((issue.level, issue.segment_group), ("warning", "length"))
        self.assertIn("reserved", issue.message)
        self.assertIn("'dendrite_0'", issue.message)
        self.assertEqual(report.metadata.segment_branches[1], (("dendrite_0", 0.0, 1.0),))

    def test_unusable_root_group_falls_back_to_the_type_name(self) -> None:
        path = self.write(
            _cell(
                _SOMA
                + _TYPES
                + '<segmentGroup id="2soma" neuroLexId="sao864921383"><member segment="0"/></segmentGroup>'
            )
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assertEqual(morpho.root.name, "soma")
        self.assertEqual(
            [(issue.code, issue.segment_group) for issue in report.issues], [("naming.group_id_unusable", "2soma")]
        )

    def test_taken_name_is_auto_named(self) -> None:
        # The auto-named branch of segment 1 takes "dendrite_0" before the cable group asks for it.
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 1, 'x="0" y="10" z="0"', 'x="0" y="30" z="0"')
                + _segment(2, 0, 0, 'x="0" y="0" z="0"', 'x="0" y="-30" z="0"')
                + _TYPES
                + '<segmentGroup id="dendrite_group"><member segment="1"/><member segment="2"/></segmentGroup>'
                '<segmentGroup id="dendrite_0" neuroLexId="sao864921383"><member segment="2"/></segmentGroup>'
            )
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assertEqual([branch.name for branch in morpho.branches], ["soma", "dendrite_0", "dendrite_1"])
        self.assertIn("already exists", report.issues[0].message)

    def test_cut_root_cable_group_numbers_every_part(self) -> None:
        # A two-segment soma in cable group "body", cut by a child at its middle segment's midpoint.
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 1, 'x="0" y="10" z="0"', 'x="0" y="20" z="0"', diameter=10.0)
                + _segment(2, 1, 0.5, 'x="0" y="15" z="0"', 'x="10" y="15" z="0"')
                + '<segmentGroup id="soma_group"><member segment="0"/><member segment="1"/></segmentGroup>'
                '<segmentGroup id="body" neuroLexId="sao864921383"><member segment="0"/><member segment="1"/></segmentGroup>'
            )
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assert_branch_table(
            morpho,
            [
                ("body_0", "soma", None, None, 2, 15.0),
                ("body_1", "soma", "body_0", 1.0, 1, 5.0),
                ("custom_0", "custom", "body_0", 1.0, 1, 10.0),
            ],
        )
        self.assertEqual(report.metadata.segment_branches[1], (("body_0", 2 / 3, 1.0), ("body_1", 0.0, 1.0)))
        self.assertEqual(report.issues, [])


class AttachmentTest(_ReaderTestCase):
    def test_one_segment_soma_cut_elsewhere_is_cut_at_its_midpoint_too(self) -> None:
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 0.5, 'x="0" y="5" z="0"', 'x="10" y="5" z="0"')
                + _segment(2, 0, 0.25, None, 'x="-10" y="2.5" z="0"')
                + _TYPES
            )
        )

        morpho, report = NeuroMlReader().read(path, return_report=True)

        self.assert_branch_table(
            morpho,
            [
                ("soma", "soma", None, None, 1, 2.5),
                ("soma_0", "soma", "soma", 1.0, 1, 2.5),
                ("soma_1", "soma", "soma_0", 1.0, 1, 5.0),
                ("custom_0", "custom", "soma_0", 1.0, 1, 10.0),
                ("custom_1", "custom", "soma", 1.0, 1, 10.0),
            ],
        )
        self.assertEqual(
            report.metadata.segment_branches[0], (("soma", 0.0, 1.0), ("soma_0", 0.0, 1.0), ("soma_1", 0.0, 1.0))
        )
        self.assert_totals(path)

    def test_children_anywhere_on_a_sphere_attach_at_its_centre(self) -> None:
        sphere = '<segment id="0"><proximal x="0" y="0" z="0" diameter="6"/><distal x="0" y="0" z="0" diameter="6"/></segment>'
        path = self.write(
            _cell(
                sphere
                + _segment(1, 0, 0.3, 'x="0" y="0" z="0"', 'x="10" y="0" z="0"')
                + _segment(2, 0, 0, 'x="0" y="0" z="0"', 'x="-10" y="0" z="0"')
                + _TYPES
            )
        )

        morpho = NeuroMlReader().read(path)

        self.assertEqual([float(branch.parent_x) for branch in morpho.branches[1:]], [0.5, 0.5])
        self.assert_totals(path)

    def test_explicit_proximal_away_from_the_parent_starts_a_branch(self) -> None:
        # Same type, at the distal end, but the proximal point is not the parent's distal point.
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 1, 'x="0" y="12" z="0"', 'x="0" y="30" z="0"', diameter=10.0)
                + '<segmentGroup id="soma_group"><member segment="0"/><member segment="1"/></segmentGroup>'
            )
        )

        morpho = NeuroMlReader().read(path)

        self.assertEqual([branch.name for branch in morpho.branches], ["soma", "soma_0"])
        self.assert_totals(path)


class EntryPointTest(_ReaderTestCase):
    def test_from_neuroml_matches_the_reader(self) -> None:
        path = DATA_DIR / "branched.cell.nml"

        self.assertEqual(branch_table(Morphology.from_neuroml(path)), branch_table(NeuroMlReader().read(path)))

    def test_from_neuroml_raises_for_invalid_documents(self) -> None:
        segment_1 = (
            '<segment id="1"><parent segment="{parent}"/>{proximal}<distal x="20" y="0" z="0" diameter="2"/></segment>'
        )
        cases = {
            "truncated XML": (
                f'{_HEADER}  <cell id="c"><morphology id="m">\n    {_SOMA[:40]}\n',
                "not well-formed XML",
            ),
            "missing parent": (
                _cell(_SOMA + segment_1.format(parent=7, proximal="")),
                "parent segment 7 does not exist",
            ),
            "parent cycle": (
                _cell(
                    _SOMA
                    + segment_1.format(parent=2, proximal='<proximal x="8" y="0" z="0" diameter="2"/>')
                    + '<segment id="2"><parent segment="1"/><distal x="30" y="0" z="0" diameter="2"/></segment>'
                ),
                r"parent cycle through segments \[1, 2\]",
            ),
        }
        for name, (text, message) in cases.items():
            with self.subTest(name):
                path = self.write(text)
                with self.assertRaisesRegex(
                    ValueError, rf"^NeuroML2 import failed for {re.escape(str(path))}: .*{message}"
                ):
                    Morphology.from_neuroml(path)

    def test_spherical_dendrite_raises_on_read(self) -> None:
        path = self.write(
            _cell(
                _SOMA
                + _segment(1, 0, 1, 'x="0" y="20" z="0"', 'x="0" y="20" z="0"', diameter=4.0)
                + _TYPES
                + '<segmentGroup id="dendrite_group"><member segment="1"/></segmentGroup>'
            )
        )

        # The document itself is valid; the reader refuses it.
        self.assertTrue(parse_document(path).segments[1].is_sphere)
        with self.assertRaisesRegex(
            ValueError,
            rf"^NeuroML2 import failed for {re.escape(str(path))}: segment 1: a spherical segment .* only supported as the soma",
        ):
            Morphology.from_neuroml(path)

    def test_report_types_are_exported(self) -> None:
        for name in ("NeuroMlIssue", "NeuroMlMetadata", "NeuroMlReader", "NeuroMlReport"):
            self.assertIn(name, braincell.io.__all__)
        self.assertIs(braincell.io.NeuroMlReport, NeuroMlReport)


if __name__ == "__main__":
    unittest.main()
