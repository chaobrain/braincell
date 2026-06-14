from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest

import numpy as np
import brainunit as u

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dcn_native import (
    DCN_REGION_NAMES,
    dcn_region,
    load_dcn_morphology,
    parse_dcn_hoc,
    resolve_source_hoc,
)


class DcnNativeMorphologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.source_hoc = resolve_source_hoc()
        except FileNotFoundError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls.native = load_dcn_morphology()
        cls.specs = cls.native.specs
        cls.morpho = cls.native.morpho

    def test_parser_preserves_source_counts_and_regions(self) -> None:
        specs = parse_dcn_hoc()

        self.assertEqual(len(specs), 517)
        self.assertEqual(
            Counter(spec.region for spec in specs),
            {
                "soma": 1,
                "axHillock": 1,
                "axIniSeg": 10,
                "axNode": 20,
                "proxDend": 83,
                "distDend": 402,
            },
        )
        self.assertEqual(specs[0].source_name, "soma")
        self.assertEqual(specs[0].branch_name, "soma")
        self.assertEqual(specs[0].region, "soma")

    def test_builder_preserves_topology_geometry_and_names(self) -> None:
        morpho = self.morpho

        self.assertEqual(morpho.n_branches, 517)
        self.assertEqual(len(morpho.edges), 516)
        self.assertTrue(morpho.has_full_point_geometry)
        self.assertEqual(morpho.root.name, "soma")
        self.assertEqual(morpho.branch(index=1).name, "axHillock__axHill")
        self.assertEqual(morpho.branch(index=2).name, "proxDend__p3__0")

        branch_type_counts = Counter(branch.type for branch in morpho.branches)
        self.assertEqual(branch_type_counts, {"soma": 1, "axon": 31, "dendrite": 485})

    def test_parent_attachments_match_hoc(self) -> None:
        by_source = {spec.source_name: spec for spec in self.specs}
        ax_hill = by_source["axHill"]
        p0 = by_source["p0[0]"]
        ax_is0 = by_source["axIS[0]"]

        self.assertEqual(ax_hill.parent_branch_name, "soma")
        self.assertEqual(ax_hill.parent_x, 0.5)
        self.assertEqual(p0.parent_branch_name, "soma")
        self.assertEqual(p0.parent_x, 0.5)
        self.assertEqual(ax_is0.parent_branch_name, "axHillock__axHill")
        self.assertEqual(ax_is0.parent_x, 1.0)

        ax_hill_node = self.morpho.branch(name=ax_hill.branch_name)
        self.assertEqual(ax_hill_node.parent.name, "soma")
        self.assertEqual(ax_hill_node.parent_x, 0.5)

    def test_total_length_matches_source_hoc_geometry(self) -> None:
        expected_um = sum(spec.length_um for spec in self.specs)
        actual_um = float(np.asarray(self.morpho.total_length.to_decimal(u.um), dtype=float))

        self.assertAlmostEqual(actual_um, expected_um, places=9)
        self.assertAlmostEqual(actual_um, 4851.247756975878, places=6)

    def test_region_selectors_use_name_prefixes(self) -> None:
        expected = Counter(spec.region for spec in self.specs)
        for region_name in DCN_REGION_NAMES:
            mask = self.morpho.select(dcn_region(self.morpho, region_name))
            self.assertEqual(len(mask.intervals), expected[region_name])

        prox_names = self.native.regions["proxDend"]
        dist_names = self.native.regions["distDend"]
        self.assertIn("proxDend__p1b2__1", prox_names)
        self.assertIn("distDend__p1b2b2__0", dist_names)

    def test_invalid_region_name_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown DCN region"):
            dcn_region(self.morpho, "dend")


if __name__ == "__main__":
    unittest.main()
