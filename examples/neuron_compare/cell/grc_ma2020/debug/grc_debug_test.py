from __future__ import annotations

import unittest

from .grc_braincell_debug import GrC as BrainCellGrC
from .grc_full_braincell_debug import GrCFull as BrainCellFullGrC
from .grc_full_neuron_debug import GrCFull as NeuronFullGrC
from .grc_full_parameters import (
    EXPECTED_FULL_AA_COUNT,
    EXPECTED_FULL_AIS_COUNT,
    EXPECTED_FULL_AXON_COUNT,
    EXPECTED_FULL_DEND_COUNT,
    EXPECTED_FULL_HILOCK_COUNT,
    EXPECTED_FULL_PF1_COUNT,
    EXPECTED_FULL_PF2_COUNT,
    EXPECTED_FULL_SOMA_COUNT,
    GrCFullConfig,
    GrCFullToggles,
    load_grc20_full_params,
)
from .grc_neuron_debug import GrC as NeuronGrC
from .grc_parameters import (
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_SOMA_COUNT,
    GrCConfig,
    GrCToggles,
    load_grc20_params,
)


class GrCModCompileLayoutTest(unittest.TestCase):
    def test_top_level_nrnmech_contains_grc_cell_mechanisms(self) -> None:
        mod_func = DEFAULT_NRNMECH_PATH.parents[1] / "mod_func.cpp"
        self.assertTrue(DEFAULT_NRNMECH_PATH.exists())
        self.assertTrue(mod_func.exists())
        text = mod_func.read_text()
        self.assertIn("CdpCR_MA20_GrC", text)
        self.assertIn("CaHVA_MA20_GrC", text)
        self.assertIn("Kv1p1_MA20_GrC", text)
        self.assertIn("Nav_MA20_GrC", text)
        self.assertIn("NaFHF_MA20_GrC", text)
        self.assertIn("KM_MA20_GrC", text)


class GrCNeuronDebugBuildTest(unittest.TestCase):
    def test_builds_asc_only_with_source_region_counts(self) -> None:
        cell = NeuronGrC(params=load_grc20_params()).build()
        try:
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
            self.assertTrue(summary["asc_only"])
            self.assertFalse(any(name.startswith("aa_") for name in cell.branch_table()["branch_name"]))
        finally:
            cell.cleanup()


class GrCBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_leak_only_with_source_region_counts(self) -> None:
        config = GrCConfig(
            toggles=GrCToggles(
                leak=True,
                kv3p4=False,
                kv4p3=False,
                kir2p3=False,
                cahva=False,
                kv1p1=False,
                kv1p5=False,
                kv2p2=False,
                kca1p1=False,
                cdp=False,
            )
        )
        cell = BrainCellGrC(params=load_grc20_params(), config=config).build()
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
        self.assertTrue(summary["asc_only"])
        self.assertGreater(len(cell.compartment_table()), 0)


class GrCFullNeuronDebugBuildTest(unittest.TestCase):
    def test_builds_full_manual_morphology_with_pf(self) -> None:
        cell = NeuronFullGrC(params=load_grc20_full_params()).build()
        try:
            summary = cell.summary()
            self.assertEqual(
                summary["branch_counts"],
                {
                    "n_soma": EXPECTED_FULL_SOMA_COUNT,
                    "n_dend": EXPECTED_FULL_DEND_COUNT,
                    "n_axon": EXPECTED_FULL_AXON_COUNT,
                    "n_total": EXPECTED_FULL_SOMA_COUNT + EXPECTED_FULL_DEND_COUNT + EXPECTED_FULL_AXON_COUNT,
                },
            )
            self.assertEqual(summary["region_counts"]["hilock"], EXPECTED_FULL_HILOCK_COUNT)
            self.assertEqual(summary["region_counts"]["ais"], EXPECTED_FULL_AIS_COUNT)
            self.assertEqual(summary["region_counts"]["aa"], EXPECTED_FULL_AA_COUNT)
            self.assertEqual(summary["region_counts"]["pf"], EXPECTED_FULL_PF1_COUNT + EXPECTED_FULL_PF2_COUNT)
            self.assertFalse(summary["asc_only"])
            self.assertTrue(summary["manual_pf"])
            names = set(cell.branch_table()["branch_name"])
            self.assertIn("aa_3", names)
            self.assertIn("pf1_141", names)
            self.assertIn("pf2_141", names)
        finally:
            cell.cleanup()


class GrCFullBrainCellDebugBuildTest(unittest.TestCase):
    def test_builds_full_manual_morphology_with_unique_branch_names(self) -> None:
        config = GrCFullConfig(
            toggles=GrCFullToggles(
                leak=True,
                nav=False,
                nafhhf=False,
                kv3p4=False,
                kv4p3=False,
                kir2p3=False,
                cahva=False,
                kv1p1=False,
                kv1p5=False,
                kv2p2=False,
                kca1p1=False,
                km=False,
                cdp=False,
            )
        )
        cell = BrainCellFullGrC(params=load_grc20_full_params(), config=config).build()
        summary = cell.summary()
        self.assertEqual(
            summary["branch_counts"],
            {
                "n_soma": EXPECTED_FULL_SOMA_COUNT,
                "n_dend": EXPECTED_FULL_DEND_COUNT,
                "n_axon": EXPECTED_FULL_AXON_COUNT,
                "n_total": EXPECTED_FULL_SOMA_COUNT + EXPECTED_FULL_DEND_COUNT + EXPECTED_FULL_AXON_COUNT,
            },
        )
        self.assertEqual(summary["region_counts"]["hilock"], EXPECTED_FULL_HILOCK_COUNT)
        self.assertEqual(summary["region_counts"]["ais"], EXPECTED_FULL_AIS_COUNT)
        self.assertEqual(summary["region_counts"]["aa"], EXPECTED_FULL_AA_COUNT)
        self.assertEqual(summary["region_counts"]["pf"], EXPECTED_FULL_PF1_COUNT + EXPECTED_FULL_PF2_COUNT)
        self.assertFalse(summary["asc_only"])
        self.assertTrue(summary["manual_pf"])
        branch_names = list(cell.branch_table()["branch_name"])
        self.assertEqual(len(branch_names), len(set(branch_names)))
        self.assertIn("pf1_141", branch_names)
        self.assertIn("pf2_141", branch_names)
        self.assertGreater(len(cell.compartment_table()), 0)


if __name__ == "__main__":
    unittest.main()
