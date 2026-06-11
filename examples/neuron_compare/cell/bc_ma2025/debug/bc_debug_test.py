from __future__ import annotations

import unittest

from .bc_braincell_debug import BC as BrainCellBC
from .bc_parameters import (
    BCConfig,
    BCToggles,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    axon_region_name,
    load_bc25_params,
)


class BCRegionMapTest(unittest.TestCase):
    def test_axon_region_map(self) -> None:
        ais = [index for index in range(EXPECTED_AXON_COUNT) if axon_region_name(index) == "axon_ais"]
        regular = [index for index in range(EXPECTED_AXON_COUNT) if axon_region_name(index) == "axon_regular"]
        self.assertEqual(ais, [0])
        self.assertEqual(regular, list(range(1, EXPECTED_AXON_COUNT)))


class BCModCompileLayoutTest(unittest.TestCase):
    def test_top_level_nrnmech_contains_cdp(self) -> None:
        mod_func = DEFAULT_NRNMECH_PATH.parents[1] / "mod_func.cpp"
        self.assertTrue(DEFAULT_NRNMECH_PATH.exists())
        self.assertTrue(mod_func.exists())
        text = mod_func.read_text()
        self.assertIn("CdpStC_MA25_BC", text)
        self.assertIn("Nav1p6_MA25_BC", text)


class BCBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_leak_only_with_source_region_counts(self) -> None:
        params = load_bc25_params()
        config = BCConfig(
            toggles=BCToggles(
                leak=True,
                nav1p1=False,
                nav1p6=False,
                cav1p2=False,
                cav1p3=False,
                cav2p1=False,
                cav3p2=False,
                kir2p3=False,
                kv1p1=False,
                kv3p4=False,
                kv4p3=False,
                kca1p1=False,
                kca2p2=False,
                kca3p1=False,
                hcn1=False,
                cdp=False,
            )
        )
        cell = BrainCellBC(params=params, config=config).build()
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
        self.assertEqual(summary["region_counts"]["dend"], EXPECTED_DEND_COUNT)
        self.assertEqual(summary["region_counts"]["axon_ais"], 1)
        self.assertEqual(summary["region_counts"]["axon_regular"], EXPECTED_AXON_COUNT - 1)

    def test_cav2p1_frozen_switch_selects_braincell_channel_class(self) -> None:
        params = load_bc25_params()
        config = BCConfig(
            toggles=BCToggles(
                leak=False,
                nav1p1=False,
                nav1p6=False,
                cav1p2=False,
                cav1p3=False,
                cav2p1=True,
                cav3p2=False,
                kir2p3=False,
                kv1p1=False,
                kv3p4=False,
                kv4p3=False,
                kca1p1=False,
                kca2p2=False,
                kca3p1=False,
                hcn1=False,
                cdp=False,
            )
        )

        frozen_cell = BrainCellBC(params=params, config=config, frozen=True).build()
        unfrozen_cell = BrainCellBC(params=params, config=config, frozen=False).build()

        def cav2p1_classes(cell: BrainCellBC) -> set[str]:
            return {
                mech.class_name
                for cv in cell.cell.cvs
                for mech in cv.density_mech
                if "Cav2p1" in mech.class_name
            }

        self.assertEqual(cav2p1_classes(frozen_cell), {"Cav2p1_MA2025_BC_Frozen"})
        self.assertEqual(cav2p1_classes(unfrozen_cell), {"Cav2p1_MA2025_BC"})
        self.assertTrue(frozen_cell.summary()["frozen"])
        self.assertFalse(unfrozen_cell.summary()["frozen"])


if __name__ == "__main__":
    unittest.main()
