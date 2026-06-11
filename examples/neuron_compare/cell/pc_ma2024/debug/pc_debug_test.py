from __future__ import annotations

import unittest

from .pc_braincell_debug import PC as BrainCellPC
from .pc_parameters import (
    DEFAULT_NRNMECH_PATH,
    PCConfig,
    PCToggles,
    load_pc24_params,
)


class PCModCompileLayoutTest(unittest.TestCase):
    def test_top_level_nrnmech_contains_pc_mechanisms(self) -> None:
        mod_func = DEFAULT_NRNMECH_PATH.parents[1] / "mod_func.cpp"
        self.assertTrue(DEFAULT_NRNMECH_PATH.exists())
        self.assertTrue(mod_func.exists())
        text = mod_func.read_text()
        self.assertIn("CdpCAM_MA24_PC", text)
        self.assertIn("Nav1p6_MA24_PC", text)


class PCBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_leak_only_with_source_morphology(self) -> None:
        params = load_pc24_params()
        config = PCConfig(
            toggles=PCToggles(
                leak=True,
                nav=False,
                kv1p1=False,
                kv1p5=False,
                kv3p3=False,
                kv3p4=False,
                kv4p3=False,
                kir2p3=False,
                kca1p1=False,
                kca2p2=False,
                kca3p1=False,
                cav21=False,
                cav31=False,
                cav32=False,
                cav33=False,
                hcn1=False,
                cdp=False,
            )
        )
        cell = BrainCellPC(params=params, config=config).build()
        summary = cell.summary()
        branch_counts = summary["branch_counts"]

        self.assertGreater(branch_counts["n_soma"], 0)
        self.assertGreater(branch_counts["n_dend"], 0)
        self.assertEqual(branch_counts["n_total"], branch_counts["n_soma"] + branch_counts["n_dend"])
        self.assertGreater(summary["compartment_counts"]["n_total_cv"], 0)
        self.assertFalse(cell.branch_table().empty)
        self.assertFalse(cell.compartment_table().empty)


if __name__ == "__main__":
    unittest.main()
