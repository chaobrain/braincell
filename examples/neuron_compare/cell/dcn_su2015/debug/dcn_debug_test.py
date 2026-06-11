from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

import brainunit as u

from .dcn_braincell_debug import DCN as BrainCellDCN
from .dcn_braincell_debug import _dend_shell_depth_um, _section_diam_um, _soma_shell_depth_um
from .dcn_neuron_debug import DCN as NeuronDCN
from .dcn_parameters import (
    DcnConfig,
    DcnToggles,
    DEFAULT_NRNMECH_PATH,
    EXPECTED_AXON_COUNT,
    EXPECTED_DEND_COUNT,
    EXPECTED_REGION_COUNTS,
    EXPECTED_SOMA_COUNT,
    EXPECTED_TOTAL_COUNT,
    SOURCE_MORPH_PATH,
    SOURCE_TEMPLATE_PATH,
)


class DcnModCompileLayoutTest(unittest.TestCase):
    def test_top_level_nrnmech_contains_dcn_cell_mechanisms(self) -> None:
        mod_func = DEFAULT_NRNMECH_PATH.parents[1] / "mod_func.cpp"
        self.assertTrue(DEFAULT_NRNMECH_PATH.exists())
        self.assertTrue(mod_func.exists())
        text = mod_func.read_text()
        self.assertIn("NaF_SU15_DCN", text)
        self.assertIn("NaP_SU15_DCN", text)
        self.assertIn("CaHVA_SU15_DCN", text)
        self.assertIn("CdpHVA_SU15_DCN", text)


class DcnNeuronDebugBuildTest(unittest.TestCase):
    def test_builds_source_template_with_sectionlist_region_counts(self) -> None:
        if not Path(SOURCE_TEMPLATE_PATH).exists() or not Path(SOURCE_MORPH_PATH).exists():
            raise unittest.SkipTest("Missing source DCN HOC files.")
        cell = NeuronDCN().build()
        try:
            summary = cell.summary()
            self.assertEqual(
                summary["branch_counts"],
                {
                    "n_soma": EXPECTED_SOMA_COUNT,
                    "n_dend": EXPECTED_DEND_COUNT,
                    "n_axon": EXPECTED_AXON_COUNT,
                    "n_total": EXPECTED_TOTAL_COUNT,
                },
            )
            self.assertEqual(summary["region_counts"], dict(sorted(EXPECTED_REGION_COUNTS.items())))
            self.assertTrue(summary["native_hoc"])
            self.assertEqual(summary["nrnmech_path"], str(DEFAULT_NRNMECH_PATH))
            self.assertIn("tnc", summary["enabled_mechanisms"]["soma"])
            self.assertEqual(len(cell.branch_table()), EXPECTED_TOTAL_COUNT)
            self.assertGreater(len(cell.compartment_table()), 0)
        finally:
            cell.cleanup()

    def test_calva_external_calcium_survives_finitialize(self) -> None:
        if not Path(SOURCE_TEMPLATE_PATH).exists() or not Path(SOURCE_MORPH_PATH).exists():
            raise unittest.SkipTest("Missing source DCN HOC files.")
        from neuron import h

        config = DcnConfig(
            toggles=DcnToggles(
                leak=False,
                naf=False,
                nap=False,
                fkdr=False,
                skdr=False,
                sk=False,
                hcn=False,
                tnc=False,
                calva=True,
                cahva=False,
                ca_conc=False,
                cal_conc=True,
            )
        )
        cell = NeuronDCN(config=config).build()
        try:
            h.finitialize(config.v_init_mV)
            soma_seg = cell.root_soma(0.5)
            self.assertAlmostEqual(float(soma_seg.calo), cell.params.calcium_co, places=12)
        finally:
            cell.cleanup()


class DcnBrainCellDebugBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not Path(SOURCE_MORPH_PATH).exists():
            raise unittest.SkipTest(f"Missing source HOC morphology: {SOURCE_MORPH_PATH}")
        cls.cell = BrainCellDCN().build()

    def test_builds_native_hoc_with_region_counts(self) -> None:
        summary = self.cell.summary()
        self.assertEqual(
            summary["branch_counts"],
            {
                "n_soma": EXPECTED_SOMA_COUNT,
                "n_dend": EXPECTED_DEND_COUNT,
                "n_axon": EXPECTED_AXON_COUNT,
                "n_total": EXPECTED_TOTAL_COUNT,
            },
        )
        self.assertEqual(summary["region_counts"], dict(sorted(EXPECTED_REGION_COUNTS.items())))
        self.assertTrue(summary["native_hoc"])
        self.assertEqual(len(self.cell.branch_table()), EXPECTED_TOTAL_COUNT)
        self.assertEqual(len(self.cell.compartment_table()), EXPECTED_TOTAL_COUNT)

    def test_native_branch_names_are_unique_and_region_prefixed(self) -> None:
        branch_names = list(self.cell.branch_table()["branch_name"])
        self.assertEqual(len(branch_names), len(set(branch_names)))
        self.assertIn("soma", branch_names)
        self.assertTrue(any(name.startswith("proxDend__") for name in branch_names))
        self.assertTrue(any(name.startswith("distDend__") for name in branch_names))
        self.assertTrue(any(name.startswith("axIniSeg__") for name in branch_names))

    def test_expected_channel_classes_are_painted(self) -> None:
        classes = Counter(
            getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__)
            for rule in self.cell.cell.paint_rules
        )
        self.assertGreaterEqual(classes["CableProperty"], 2)
        self.assertEqual(classes["IL"], 6)
        self.assertEqual(classes["SodiumFixed"], 1)
        self.assertEqual(classes["PotassiumFixed"], 1)
        self.assertEqual(classes["CdpHVA_SU2015_DCN"], EXPECTED_REGION_COUNTS["soma"] + EXPECTED_REGION_COUNTS["proxDend"] + EXPECTED_REGION_COUNTS["distDend"])
        self.assertEqual(classes["CdpLVA_SU2015_DCN"], EXPECTED_REGION_COUNTS["soma"] + EXPECTED_REGION_COUNTS["proxDend"] + EXPECTED_REGION_COUNTS["distDend"])
        self.assertEqual(classes["NaF_SU2015_DCN"], 4)
        self.assertEqual(classes["NaP_SU2015_DCN"], 1)
        self.assertEqual(classes["fKdr_SU2015_DCN"], 4)
        self.assertEqual(classes["sKdr_SU2015_DCN"], 4)
        self.assertEqual(classes["SK_SU2015_DCN"], 3)
        self.assertEqual(classes["HCN_SU2015_DCN"], 3)
        self.assertEqual(classes["CaLVA_SU2015_DCN"], 3)
        self.assertEqual(classes["CaHVA_SU2015_DCN"], 3)

    def test_axnode_region_has_only_passive_paints(self) -> None:
        axnode_mask = self.cell.morpho.select(self.cell.regions["axNode"])
        self.assertEqual(len(axnode_mask.intervals), EXPECTED_REGION_COUNTS["axNode"])
        axnode_rules = [
            rule
            for rule in self.cell.cell.paint_rules
            if len(self.cell.morpho.select(rule.region).intervals) == EXPECTED_REGION_COUNTS["axNode"]
            and all(
                self.cell.morpho.branch(index=index).name.startswith("axNode__")
                for index, _, _ in self.cell.morpho.select(rule.region).intervals
            )
        ]
        classes = {
            getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__)
            for rule in axnode_rules
        }
        self.assertEqual(classes, {"CableProperty", "IL"})

    def test_tnc_is_represented_as_named_il_instances(self) -> None:
        names = [
            getattr(rule.mechanism, "name", "")
            for rule in self.cell.cell.paint_rules
            if getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__) == "IL"
        ]
        self.assertIn("TNC_soma", names)
        self.assertIn("TNC_axHillock", names)
        self.assertIn("TNC_axIniSeg", names)
        self.assertIn("TNC_proxDend", names)

    def test_cdp_hva_is_branch_painted_with_neuron_depth_params(self) -> None:
        specs_by_branch = {spec.branch_name: spec for spec in self.cell.native.specs}
        active_regions = {"soma", "proxDend", "distDend"}
        rules = [
            rule
            for rule in self.cell.cell.paint_rules
            if getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__) == "CdpHVA_SU2015_DCN"
        ]
        self.assertEqual(len(rules), EXPECTED_REGION_COUNTS["soma"] + EXPECTED_REGION_COUNTS["proxDend"] + EXPECTED_REGION_COUNTS["distDend"])
        seen = set()
        for rule in rules:
            intervals = self.cell.morpho.select(rule.region).intervals
            self.assertEqual(len(intervals), 1)
            branch_index, prox, dist = intervals[0]
            self.assertEqual((prox, dist), (0.0, 1.0))
            branch = self.cell.morpho.branch(index=branch_index)
            spec = specs_by_branch[branch.name]
            self.assertIn(spec.region, active_regions)
            seen.add(spec.branch_name)

            diam_um = _section_diam_um(spec)
            expected_depth = (
                _soma_shell_depth_um(diam_um, self.cell.params.shell_thick)
                if spec.region == "soma"
                else _dend_shell_depth_um(diam_um, self.cell.params.shell_thick)
            )
            expected_k = self.cell.params.k_ca_ca_conc_soma if spec.region == "soma" else self.cell.params.k_ca_ca_conc_dend
            params = rule.mechanism.params
            self.assertAlmostEqual(float(params["depth"].to_decimal(u.um)), expected_depth, places=12)
            self.assertAlmostEqual(float(params["kCa"] / (1 / u.coulomb)), expected_k, places=18)
        self.assertEqual(len(seen), len(rules))

    def test_cahva_without_ca_conc_uses_fixed_calcium(self) -> None:
        config = DcnConfig(
            toggles=DcnToggles(
                leak=False,
                naf=False,
                nap=False,
                fkdr=False,
                skdr=False,
                sk=False,
                hcn=False,
                tnc=False,
                calva=False,
                cahva=True,
                ca_conc=False,
                cal_conc=False,
            )
        )
        cell = BrainCellDCN(config=config).build()
        classes = Counter(
            getattr(rule.mechanism, "class_name", type(rule.mechanism).__name__)
            for rule in cell.cell.paint_rules
        )
        self.assertEqual(classes["CdpHVA_SU2015_DCN"], 0)
        self.assertEqual(classes["CalciumInitNernst"], 1)


class DcnDebugTablePairingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not Path(SOURCE_MORPH_PATH).exists():
            raise unittest.SkipTest(f"Missing source HOC morphology: {SOURCE_MORPH_PATH}")
        cls.neuron_cell = NeuronDCN().build()
        cls.braincell_cell = BrainCellDCN().build()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "neuron_cell"):
            cls.neuron_cell.cleanup()

    def test_compartment_tables_pair_by_source_section(self) -> None:
        nrn = self.neuron_cell.compartment_table()
        bc = self.braincell_cell.compartment_table()

        self.assertIn("source_section", nrn.columns)
        self.assertIn("source_section", bc.columns)

        paired = bc.merge(
            nrn,
            on=["source_section", "local_index"],
            suffixes=("_braincell", "_neuron"),
            validate="one_to_one",
        )
        self.assertEqual(len(paired), EXPECTED_TOTAL_COUNT)
        self.assertIn("p0b2b2b1b2b2[25]", set(paired["source_section"]))

        row = paired.loc[paired["source_section"] == "p0b2b2b1b2b2[25]"].iloc[0]
        self.assertEqual(row["branch_name_braincell"], "distDend__p0b2b2b1b2b2__25")
        self.assertEqual(row["branch_name_neuron"], "p0b2b2b1b2b2[25]")


if __name__ == "__main__":
    unittest.main()
