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

"""Tests for :mod:`braincell.io.neuroml2.document`."""

import tempfile
import textwrap
import unittest
from pathlib import Path

from braincell.io._testing import FIXTURE_DIR
from braincell.io.neuroml2.document import parse_document
from braincell.io.neuroml2.types import NeuroMlIssue, NeuroMlMetadata, NeuroMlReport

_HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="doc">\n'
_SOMA = '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/></segment>'
_TWO_SEGMENTS = (
    '<segment id="0"><proximal x="0" y="0" z="0" diameter="10"/><distal x="0" y="10" z="0" diameter="10"/></segment>'
    '<segment id="1"><parent segment="0"/><proximal x="0" y="10" z="0" diameter="2"/>'
    '<distal x="0" y="30" z="0" diameter="2"/></segment>'
)


def _cell(body: str) -> str:
    return f'{_HEADER}  <cell id="c"><morphology id="m">\n    {body}\n  </morphology></cell>\n</neuroml>\n'


MINIMAL = """\
<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="minimal_doc">
  <cell id="minimal">
    <morphology id="minimal_morph">
      <segment id="0" name="soma">
        <proximal x="0" y="0" z="0" diameter="10"/>
        <distal x="0" y="10" z="0" diameter="10"/>
      </segment>
      <segment id="1" name="trunk_a">
        <parent segment="0"/>
        <proximal x="0" y="10" z="0" diameter="4"/>
        <distal x="0" y="30" z="0" diameter="3"/>
      </segment>
      <segment id="2" name="trunk_b">
        <parent segment="1"/>
        <proximal x="0" y="30" z="0" diameter="2"/>
        <distal x="0" y="50" z="0" diameter="2"/>
      </segment>
      <segment id="3" name="side">
        <parent segment="2" fractionAlong="0.25"/>
        <distal x="10" y="35" z="0" diameter="1"/>
      </segment>
      <segment id="4" name="axon">
        <parent segment="0" fractionAlong="0"/>
        <proximal x="0" y="0" z="0" diameter="1"/>
        <distal x="0" y="-40" z="0" diameter="1"/>
      </segment>
      <segment id="5" name="mid">
        <parent segment="0" fractionAlong="0.5"/>
        <proximal x="5" y="5" z="0" diameter="2"/>
        <distal x="25" y="5" z="0" diameter="2"/>
      </segment>
      <segmentGroup id="soma_group" neuroLexId="GO:0043025"><member segment="0"/></segmentGroup>
      <segmentGroup id="axon_group" neuroLexId="GO:0030424"><member segment="4"/></segmentGroup>
      <segmentGroup id="trunk" neuroLexId="sao864921383"><member segment="1"/><member segment="2"/></segmentGroup>
      <segmentGroup id="dendrite_group" neuroLexId="GO:0030425">
        <include segmentGroup="trunk"/><member segment="3"/><member segment="5"/>
      </segmentGroup>
      <segmentGroup id="distal_part"><subTree><from segment="2"/></subTree></segmentGroup>
    </morphology>
  </cell>
</neuroml>
"""

TWO_CELLS = """\
<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="doc">
  <morphology id="shared">
    <segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/></segment>
    <segment id="1"><parent segment="0"/><distal x="30" y="0" z="0" diameter="2"/></segment>
    <segmentGroup id="soma_group"><member segment="0"/></segmentGroup>
  </morphology>
  <cell id="a" morphology="shared"/>
  <cell id="b">
    <morphology id="b_morph">
      <segment id="0"><proximal x="0" y="0" z="0" diameter="6"/><distal x="6" y="0" z="0" diameter="6"/></segment>
    </morphology>
  </cell>
</neuroml>
"""

MALFORMED = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="doc">\n'
    '  <cell id="c"><morphology id="m">\n'
    '    <segment id="0"><proximal x="0" y="0" z="0" diameter="8"/>\n'
)

DOCTYPE_ENTITIES = (
    '<?xml version="1.0"?>\n'
    '<!DOCTYPE neuroml [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">]>\n'
    '<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"><cell id="&b;"/></neuroml>\n'
)

NEUROML1_MORPHML = '<?xml version="1.0"?>\n<morphml xmlns="http://morphml.org/morphml/schema"><cells/></morphml>\n'

NO_CELL = (
    f'{_HEADER}  <izhikevichCell id="iz" v0="-70mV" thresh="30mV" a="0.02" b="0.2" c="-65" d="6"/>\n'
    '  <include href="other.cell.nml"/>\n</neuroml>\n'
)

# Each malformed document, the cell_id passed, and a fragment of the message.
COUNTER_EXAMPLES = {
    "malformed": (MALFORMED, None, ("not well-formed XML",)),
    "doctype_entities": (DOCTYPE_ENTITIES, None, ("DTD",)),
    "neuroml1_morphml": (NEUROML1_MORPHML, None, ("expected <neuroml>",)),
    "no_cell": (NO_CELL, None, ("no <cell>", "other.cell.nml")),
    "two_cells_without_cell_id": (TWO_CELLS, None, ("'a', 'b'",)),
    "unknown_cell_id": (TWO_CELLS, "zz", ("no cell 'zz'",)),
    "empty_morphology": (_cell(""), None, ("no segments",)),
    "bad_segment_id": (
        _cell(
            '<segment id="a0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/></segment>'
        ),
        None,
        ("not a non-negative integer",),
    ),
    "duplicate_id": (
        _cell(_SOMA + '<segment id="0"><parent segment="0"/><distal x="20" y="0" z="0" diameter="2"/></segment>'),
        None,
        ("duplicate segment id 0",),
    ),
    "missing_distal": (
        _cell('<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/></segment>'),
        None,
        ("missing <distal>",),
    ),
    "nan_coord": (
        _cell(_SOMA + '<segment id="1"><parent segment="0"/><distal x="NaN" y="0" z="0" diameter="2"/></segment>'),
        None,
        ("not finite",),
    ),
    "zero_diameter": (
        _cell(_SOMA + '<segment id="1"><parent segment="0"/><distal x="20" y="0" z="0" diameter="0"/></segment>'),
        None,
        ("must be > 0",),
    ),
    "missing_parent": (
        _cell(_SOMA + '<segment id="1"><parent segment="7"/><distal x="20" y="0" z="0" diameter="2"/></segment>'),
        None,
        ("parent segment 7 does not exist",),
    ),
    "parent_cycle": (
        _cell(
            _SOMA + '<segment id="1"><parent segment="2"/><proximal x="8" y="0" z="0" diameter="2"/>'
            '<distal x="20" y="0" z="0" diameter="2"/></segment>'
            '<segment id="2"><parent segment="1"/><distal x="30" y="0" z="0" diameter="2"/></segment>'
        ),
        None,
        ("parent cycle through segments [1, 2]",),
    ),
    "self_parent": (
        _cell(
            _SOMA + '<segment id="1"><parent segment="1"/><proximal x="8" y="0" z="0" diameter="2"/>'
            '<distal x="20" y="0" z="0" diameter="2"/></segment>'
        ),
        None,
        ("parent cycle through segments [1]",),
    ),
    "two_roots": (
        _cell(
            _SOMA
            + '<segment id="1"><proximal x="50" y="0" z="0" diameter="2"/><distal x="60" y="0" z="0" diameter="2"/></segment>'
        ),
        None,
        ("2 root segments",),
    ),
    "root_no_proximal": (
        _cell('<segment id="0"><distal x="8" y="0" z="0" diameter="8"/></segment>'),
        None,
        ("root segment 0 has no <proximal>",),
    ),
    "bad_fraction": (
        _cell(
            _SOMA
            + '<segment id="1"><parent segment="0" fractionAlong="1.5"/><distal x="20" y="0" z="0" diameter="2"/></segment>'
        ),
        None,
        ("outside [0, 1]",),
    ),
    "sphere_unequal": (
        _cell(
            '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="0" y="0" z="0" diameter="6"/></segment>'
        ),
        None,
        ("diameters differ",),
    ),
    "conflicting_types": (
        _cell(
            _SOMA + '<segmentGroup id="soma_group"><member segment="0"/></segmentGroup>'
            '<segmentGroup id="dendrite_group"><member segment="0"/></segmentGroup>'
        ),
        None,
        ("both a soma group and a dendrite group",),
    ),
    "member_missing": (
        _cell(_SOMA + '<segmentGroup id="g"><member segment="9"/></segmentGroup>'),
        None,
        ("member segment 9 does not exist",),
    ),
    "include_missing": (
        _cell(_SOMA + '<segmentGroup id="g"><include segmentGroup="nope"/></segmentGroup>'),
        None,
        ("missing group 'nope'",),
    ),
    "include_cycle": (
        _cell(
            _SOMA + '<segmentGroup id="g"><include segmentGroup="h"/></segmentGroup>'
            '<segmentGroup id="h"><include segmentGroup="g"/></segmentGroup>'
        ),
        None,
        ("include cycle",),
    ),
    "path_missing": (
        _cell(_TWO_SEGMENTS + '<segmentGroup id="g"><path><from segment="0"/><to segment="9"/></path></segmentGroup>'),
        None,
        ("path endpoint 9 does not exist",),
    ),
    "subtree_missing": (
        _cell(_TWO_SEGMENTS + '<segmentGroup id="g"><subTree><from segment="9"/></subTree></segmentGroup>'),
        None,
        ("subTree endpoint 9 does not exist",),
    ),
}


class _DocumentTestCase(unittest.TestCase):
    def write(self, text: str, name: str = "cell.nml") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(text, encoding="utf-8")
        return path


class ParseDocumentCounterExampleTest(_DocumentTestCase):
    def test_every_counter_example_raises_value_error(self) -> None:
        self.assertEqual(len(COUNTER_EXAMPLES), 25)
        for name, (text, cell_id, fragments) in COUNTER_EXAMPLES.items():
            with self.subTest(name=name):
                path = self.write(text, f"{name}.nml")
                with self.assertRaises(ValueError) as caught:
                    parse_document(path, cell_id=cell_id)
                message = str(caught.exception)
                self.assertTrue(message.startswith(f"NeuroML2 import failed for {path}: "), message)
                for fragment in fragments:
                    self.assertIn(fragment, message)

    def test_entities_are_refused_before_expansion(self) -> None:
        path = self.write(DOCTYPE_ENTITIES)

        with self.assertRaises(ValueError) as caught:
            parse_document(path)

        self.assertNotIn("aaaaaaaaaa", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_malformed_xml_reports_the_parser_position(self) -> None:
        with self.assertRaisesRegex(ValueError, r"not well-formed XML: no element found: line 5, column 0"):
            parse_document(self.write(MALFORMED))

    def test_other_malformed_elements(self) -> None:
        cases = {
            "two distal": (
                _cell(
                    '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/>'
                    '<distal x="9" y="0" z="0" diameter="8"/></segment>'
                ),
                "segment 0 has 2 <distal> elements",
            ),
            "missing attribute": (
                _cell(
                    '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" diameter="8"/></segment>'
                ),
                "segment 0 <distal>: missing attribute 'z'",
            ),
            "not a number": (
                _cell(
                    '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="a" diameter="8"/></segment>'
                ),
                "segment 0 <distal>: z='a' is not a number",
            ),
            "negative id": (
                _cell(
                    '<segment id="-1"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/></segment>'
                ),
                "segment id '-1' is not a non-negative integer",
            ),
            "fraction not a number": (
                _cell(
                    _SOMA
                    + '<segment id="1"><parent segment="0" fractionAlong="half"/><distal x="20" y="0" z="0" diameter="2"/></segment>'
                ),
                "segment 1: fractionAlong='half' is not a number",
            ),
            "group without id": (
                _cell(_SOMA + '<segmentGroup><member segment="0"/></segmentGroup>'),
                "a <segmentGroup> has no id",
            ),
            "duplicate group": (
                _cell(_SOMA + '<segmentGroup id="g"><member segment="0"/></segmentGroup><segmentGroup id="g"/>'),
                "duplicate segmentGroup id 'g'",
            ),
            "path without to": (
                _cell(_TWO_SEGMENTS + '<segmentGroup id="g"><path><from segment="0"/></path></segmentGroup>'),
                "segmentGroup 'g': a <path> needs both <from> and <to>",
            ),
            "path towards the root": (
                _cell(
                    _TWO_SEGMENTS
                    + '<segmentGroup id="g"><path><from segment="1"/><to segment="0"/></path></segmentGroup>'
                ),
                "path from segment 1 to 0 does not run away from the root",
            ),
            "subTree with both ends": (
                _cell(
                    _TWO_SEGMENTS
                    + '<segmentGroup id="g"><subTree><from segment="0"/><to segment="1"/></subTree></segmentGroup>'
                ),
                "a <subTree> needs exactly one of <from> and <to>",
            ),
            "two cable groups": (
                _cell(
                    _TWO_SEGMENTS
                    + '<segmentGroup id="p" neuroLexId="sao864921383"><member segment="1"/></segmentGroup>'
                    '<segmentGroup id="q" neuroLexId="sao864921383"><member segment="1"/></segmentGroup>'
                ),
                "segment 1 is in two unbranched cable groups, 'p' and 'q'",
            ),
            "cell without morphology": (f'{_HEADER}  <cell id="c"/>\n</neuroml>\n', "cell 'c' has no <morphology>"),
            "unknown morphology reference": (
                f'{_HEADER}  <cell id="c" morphology="m"/>\n</neuroml>\n',
                "cell 'c' refers to morphology 'm', which the document does not define",
            ),
        }
        for name, (text, fragment) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as caught:
                    parse_document(self.write(text))
                self.assertIn(fragment, str(caught.exception))


class ParseDocumentTest(_DocumentTestCase):
    def test_minimal_cell(self) -> None:
        document = parse_document(self.write(MINIMAL))

        self.assertEqual(document.cell_id, "minimal")
        self.assertEqual(document.morphology_id, "minimal_morph")
        self.assertEqual(tuple(document.segments), (0, 1, 2, 3, 4, 5))
        self.assertEqual(document.root, 0)
        # Segment 3 omits <proximal>: the point a quarter along segment 2.
        self.assertEqual(document.segments[3].proximal, (0.0, 35.0, 0.0, 2.0))
        self.assertEqual(document.segments[3].fraction_along, 0.25)
        # Segment 4 starts at fractionAlong 0 but gives its own <proximal>.
        self.assertEqual(document.segments[4].proximal, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(document.segments[1].proximal, (0.0, 10.0, 0.0, 4.0))
        self.assertEqual(document.segments[1].distal, (0.0, 30.0, 0.0, 3.0))
        self.assertEqual(
            dict(document.segment_groups),
            {
                "soma_group": (0,),
                "axon_group": (4,),
                "trunk": (1, 2),
                "dendrite_group": (1, 2, 3, 5),
                "distal_part": (2, 3),
            },
        )
        self.assertEqual(
            [document.segment_types[i] for i in range(6)],
            ["soma", "dendrite", "dendrite", "dendrite", "axon", "dendrite"],
        )
        self.assertEqual(document.cable_groups, ("trunk",))
        self.assertEqual(dict(document.segment_cables), {1: "trunk", 2: "trunk"})
        self.assertEqual(document.segments[2].name, "trunk_b")
        self.assertEqual(document.children(), {0: [1, 4, 5], 1: [2], 2: [3], 3: [], 4: [], 5: []})
        self.assertEqual(document.ignored_elements, ())
        self.assertFalse(document.segments[0].is_sphere)

    def test_cell_referring_to_a_top_level_morphology(self) -> None:
        path = self.write(TWO_CELLS)

        a = parse_document(path, cell_id="a")
        b = parse_document(path, cell_id="b")

        self.assertEqual((a.cell_id, a.morphology_id, len(a.segments)), ("a", "shared", 2))
        # Segment 1 omits <proximal> and fractionAlong: the parent's distal point.
        self.assertEqual(a.segments[1].proximal, (8.0, 0.0, 0.0, 8.0))
        self.assertEqual(dict(a.segment_types), {0: "soma", 1: "custom"})
        self.assertEqual((b.cell_id, b.morphology_id, len(b.segments)), ("b", "b_morph", 1))
        self.assertEqual(dict(b.segment_groups), {})

    def test_fraction_along_zero_and_interpolated_chains(self) -> None:
        # Segments 2 and 3 omit <proximal>: 2 starts where 1 starts, 3 halfway along 2.
        path = self.write(
            _cell(
                _SOMA + '<segment id="1"><parent segment="0"/><distal x="18" y="0" z="0" diameter="4"/></segment>'
                '<segment id="2"><parent segment="1" fractionAlong="0"/><distal x="8" y="10" z="0" diameter="2"/></segment>'
                '<segment id="3"><parent segment="2" fractionAlong="0.5"/><distal x="0" y="5" z="0" diameter="1"/></segment>'
            )
        )

        document = parse_document(path)

        self.assertEqual(document.segments[1].proximal, (8.0, 0.0, 0.0, 8.0))
        self.assertEqual(document.segments[2].proximal, (8.0, 0.0, 0.0, 8.0))
        self.assertEqual(document.segments[3].proximal, (8.0, 5.0, 0.0, 5.0))

    def test_type_groups_by_neurolex_id_and_path_groups(self) -> None:
        path = self.write(
            _cell(
                _TWO_SEGMENTS
                + '<segment id="2"><parent segment="1"/><distal x="0" y="40" z="0" diameter="1"/></segment>'
                '<segmentGroup id="body" neuroLexId="sao1044911821"><member segment="0"/></segmentGroup>'
                '<segmentGroup id="dends" neuroLexId="sao1211023249"><subTree><from segment="1"/></subTree></segmentGroup>'
                '<segmentGroup id="trunk_path"><path><from segment="0"/><to segment="2"/></path></segmentGroup>'
                '<segmentGroup id="to_root"><subTree><to segment="1"/></subTree></segmentGroup>'
                '<segmentGroup id="both" neuroLexId="sao864921383"><include segmentGroup="dends"/><member segment="2"/>'
                "<notes>cable</notes></segmentGroup>"
            )
        )

        document = parse_document(path)

        self.assertEqual(dict(document.segment_types), {0: "soma", 1: "dendrite", 2: "dendrite"})
        self.assertEqual(document.segment_groups["trunk_path"], (0, 1, 2))
        self.assertEqual(document.segment_groups["to_root"], (0, 1))
        self.assertEqual(document.segment_groups["both"], (1, 2))
        self.assertEqual(dict(document.segment_cables), {1: "both", 2: "both"})
        self.assertEqual(document.ignored_elements, ("notes",))

    def test_spherical_segment(self) -> None:
        path = self.write(
            _cell(
                '<segment id="0"><proximal x="1" y="2" z="3" diameter="12"/><distal x="1" y="2" z="3" diameter="12"/></segment>'
            )
        )

        document = parse_document(path)

        self.assertTrue(document.segments[0].is_sphere)

    def test_ignored_elements_are_listed(self) -> None:
        # One element name at each level: cell, morphology, segment, segmentGroup.
        path = self.write(
            f"{_HEADER}"
            '  <cell id="c">\n'
            '    <biophysicalProperties id="bio"><membraneProperties/></biophysicalProperties>\n'
            '    <morphology id="m">\n'
            "      <notes>a morphology</notes>\n"
            '      <segment id="0"><property tag="color" value="red"/><proximal x="0" y="0" z="0" diameter="8"/>'
            '<distal x="8" y="0" z="0" diameter="8"/></segment>\n'
            '      <segmentGroup id="soma_group"><annotation/><member segment="0"/></segmentGroup>\n'
            "    </morphology>\n"
            "  </cell>\n"
            "</neuroml>\n"
        )

        document = parse_document(path)

        self.assertEqual(document.ignored_elements, ("annotation", "biophysicalProperties", "notes", "property"))
        self.assertEqual(document.segment_groups["soma_group"], (0,))

    def test_document_without_namespace_and_cell2capools(self) -> None:
        path = self.write(
            '<neuroml><cell2CaPools id="pools"><morphology id="m">'
            '<segment id="0"><proximal x="0" y="0" z="0" diameter="8"/><distal x="8" y="0" z="0" diameter="8"/></segment>'
            "</morphology></cell2CaPools></neuroml>"
        )

        document = parse_document(path)

        self.assertEqual((document.cell_id, tuple(document.segments)), ("pools", (0,)))

    def test_deep_chain_is_linear(self) -> None:
        segments = [_SOMA] + [
            f'<segment id="{i}"><parent segment="{i - 1}"/><distal x="{8 + i}" y="0" z="0" diameter="1"/></segment>'
            for i in range(1, 5001)
        ]

        document = parse_document(self.write(_cell("".join(segments))))

        self.assertEqual(len(document.segments), 5001)
        self.assertEqual(document.segments[5000].proximal, (5007.0, 0.0, 0.0, 1.0))

    def test_pyr_4_sym(self) -> None:
        # libNeuroML's pyramidal cell: 9 segments, 18 groups, of which 9 are cables.
        document = parse_document(FIXTURE_DIR / "neuroml2" / "pyr_4_sym.cell.nml")

        self.assertEqual((document.cell_id, document.root), ("pyr_4_sym", 0))
        self.assertEqual((len(document.segments), len(document.segment_groups), len(document.cable_groups)), (9, 18, 9))
        self.assertEqual(set(document.segment_cables.values()), set(document.cable_groups))
        self.assertEqual(document.ignored_elements, ("biophysicalProperties", "notes"))


class NeuroMlReportTest(unittest.TestCase):
    def test_report_renders_issue_location(self) -> None:
        report = NeuroMlReport()
        report.add_warning("naming.group_id_unusable", "group 'length' cannot name a branch", segment_group="length")
        report.add_error("geometry.example", "segment message", segment_id=3)

        self.assertIsInstance(report.issues[0], NeuroMlIssue)
        self.assertEqual(
            report.format(),
            textwrap.dedent(
                """\
                NeuroML2 report: 1 error, 1 warning

                Errors
                ------
                [ERROR] geometry.example
                segment=3
                segment message

                Warnings
                --------
                [WARNING] naming.group_id_unusable
                segmentGroup=length
                group 'length' cannot name a branch"""
            ),
        )
        self.assertEqual(NeuroMlMetadata().segment_groups, {})
        self.assertEqual(NeuroMlReport().metadata.ignored_elements, ())


if __name__ == "__main__":
    unittest.main()
