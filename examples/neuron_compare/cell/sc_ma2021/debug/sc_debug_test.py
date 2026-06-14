from __future__ import annotations

import unittest

from .sc_braincell_debug import SC as BrainCellSC
from .sc_parameters import (
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    SCConfig,
    SCToggles,
    axon_region_name,
    dend_region_name,
    load_sc21_params,
)


class SCRegionMapTest(unittest.TestCase):
    def test_dend_region_map(self) -> None:
        prox = [index for index in range(EXPECTED_DEND_COUNT) if dend_region_name(index) == "dendprox"]
        dist = [index for index in range(EXPECTED_DEND_COUNT) if dend_region_name(index) == "denddist"]
        self.assertEqual(prox, [2, 3, 15, 16, 20, 31, 34, 35, 36, 50, 66, 67, 81, 103])
        self.assertEqual(len(prox), 14)
        self.assertEqual(len(dist), 90)
        self.assertEqual(sorted(prox + dist), list(range(EXPECTED_DEND_COUNT)))

    def test_axon_region_map(self) -> None:
        ais = [index for index in range(EXPECTED_AXON_COUNT) if axon_region_name(index) == "axon_ais"]
        regular = [index for index in range(EXPECTED_AXON_COUNT) if axon_region_name(index) == "axon_regular"]
        self.assertEqual(ais, [0])
        self.assertEqual(regular, list(range(1, EXPECTED_AXON_COUNT)))


class SCModCompileLayoutTest(unittest.TestCase):
    def test_top_level_nrnmech_contains_cdp(self) -> None:
        mod_func = DEFAULT_NRNMECH_PATH.parents[1] / "mod_func.cpp"
        self.assertTrue(DEFAULT_NRNMECH_PATH.exists())
        self.assertTrue(mod_func.exists())
        text = mod_func.read_text()
        self.assertIn("CdpStC_RI21_SC", text)
        self.assertIn("Nav1p6_RI21_SC", text)


class SCBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_leak_only_with_source_region_counts(self) -> None:
        params = load_sc21_params()
        config = SCConfig(
            toggles=SCToggles(
                leak=True,
                nav1p1=False,
                nav1p6=False,
                cav2p1=False,
                cav3p2=False,
                cav3p3=False,
                kir2p3=False,
                kv1p1=False,
                kv3p4=False,
                kv4p3=False,
                km=False,
                kca1p1=False,
                kca2p2=False,
                hcn1=False,
                cdp=False,
            )
        )
        cell = BrainCellSC(params=params, config=config).build()
        summary = cell.summary()
        self.assertEqual(
            summary["branch_counts"],
            {
                "n_soma": EXPECTED_SOMA_COUNT,
                "n_dend": EXPECTED_DEND_COUNT,
                "n_axon": EXPECTED_AXON_COUNT,
                "n_total": EXPECTED_SOMA_COUNT + EXPECTED_DEND_COUNT + EXPECTED_AXON_COUNT,
            },
        )
        self.assertEqual(summary["region_counts"]["soma"], EXPECTED_SOMA_COUNT)
        self.assertEqual(summary["region_counts"]["dendprox"], 14)
        self.assertEqual(summary["region_counts"]["denddist"], 90)
        self.assertEqual(summary["region_counts"]["axon_ais"], 1)
        self.assertEqual(summary["region_counts"]["axon_regular"], EXPECTED_AXON_COUNT - 1)

    def test_cav2p1_frozen_switch_selects_braincell_channel_class(self) -> None:
        params = load_sc21_params()
        config = SCConfig(
            toggles=SCToggles(
                leak=False,
                nav1p1=False,
                nav1p6=False,
                cav2p1=True,
                cav3p2=False,
                cav3p3=False,
                kir2p3=False,
                kv1p1=False,
                kv3p4=False,
                kv4p3=False,
                km=False,
                kca1p1=False,
                kca2p2=False,
                hcn1=False,
                cdp=False,
            )
        )

        frozen_cell = BrainCellSC(params=params, config=config, frozen=True).build()
        unfrozen_cell = BrainCellSC(params=params, config=config, frozen=False).build()

        def cav2p1_classes(cell: BrainCellSC) -> set[str]:
            return {
                mech.class_name
                for cv in cell.cell.cvs
                for mech in cv.density_mech
                if "Cav2p1" in mech.class_name
            }

        self.assertEqual(cav2p1_classes(frozen_cell), {"Cav2p1_RI2021_SC_Frozen"})
        self.assertEqual(cav2p1_classes(unfrozen_cell), {"Cav2p1_RI2021_SC"})
        self.assertTrue(frozen_cell.summary()["frozen"])
        self.assertFalse(unfrozen_cell.summary()["frozen"])


if __name__ == "__main__":
    unittest.main()
