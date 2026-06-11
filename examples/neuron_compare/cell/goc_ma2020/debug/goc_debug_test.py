from __future__ import annotations

import unittest

from .goc_braincell_debug import GoC as BrainCellGoC
from .goc_parameters import (
    GoCConfig,
    GoCToggles,
    dend_region_name,
    load_goc20_params,
)


class GoCRegionMapTest(unittest.TestCase):
    def test_dend_region_map_covers_source_indices(self) -> None:
        basal = [index for index in range(151) if dend_region_name(index) == "dend_basal"]
        apical = [index for index in range(151) if dend_region_name(index) == "dend_apical"]
        self.assertEqual(len(basal), 62)
        self.assertEqual(len(apical), 89)
        self.assertEqual(sorted(basal + apical), list(range(151)))
        self.assertEqual(basal[:6], [0, 1, 2, 3, 16, 17])
        self.assertEqual(apical[:4], [4, 5, 6, 7])


class GoCBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_leak_only_with_source_region_counts(self) -> None:
        params = load_goc20_params()
        config = GoCConfig(toggles=GoCToggles(
            leak=True,
            nav=False,
            kv1p1=False,
            kv3p4=False,
            kv4p3=False,
            km=False,
            kca1p1=False,
            kca2p2=False,
            kca3p1=False,
            cahva=False,
            cav2p3=False,
            cav3p1=False,
            hcn1=False,
            hcn2=False,
            cdp=False,
        ))
        cell = BrainCellGoC(params=params, config=config).build()
        summary = cell.summary()
        self.assertEqual(summary["branch_counts"], {
            "n_soma": 1,
            "n_dend": 151,
            "n_axon": 75,
            "n_total": 227,
        })
        self.assertEqual(summary["region_counts"]["dend_basal"], 62)
        self.assertEqual(summary["region_counts"]["dend_apical"], 89)
        self.assertEqual(summary["region_counts"]["axon_ais"], 1)
        self.assertEqual(summary["region_counts"]["axon_regular"], 74)


if __name__ == "__main__":
    unittest.main()
