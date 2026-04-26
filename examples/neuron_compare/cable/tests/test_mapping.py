

import unittest

from ._helpers import TEMPLATES_ROOT, load_module


case_schema = load_module(TEMPLATES_ROOT / "case_schema.py", "cable_mapping_case_schema")
braincell_runner = load_module(TEMPLATES_ROOT / "braincell_runner.py", "cable_mapping_bc_runner")
neuron_runner = load_module(TEMPLATES_ROOT / "neuron_runner.py", "cable_mapping_nrn_runner")
mapping = load_module(TEMPLATES_ROOT / "mapping.py", "cable_mapping_helper")
fixtures = load_module(TEMPLATES_ROOT / "fixtures.py", "cable_mapping_fixtures")


class MappingTest(unittest.TestCase):
    def test_mapping_normalizes_neuron_extended_section_prefixes(self) -> None:
        self.assertEqual(mapping._normalized_branch_type("apic"), "dend")
        self.assertEqual(mapping._normalized_branch_type("dend_6"), "custom")
        self.assertEqual(mapping._normalized_branch_type("minus_3"), "custom")
        self.assertEqual(mapping._normalized_branch_type("custom"), "custom")

    def test_mapping_corrects_simple_axon_dend_order_difference(self) -> None:
        swc_path = fixtures.write_temp_swc(
            self,
            """
            1 1 0 0 0 10 -1
            2 3 0 10 0 2 1
            3 3 0 20 0 1 2
            4 2 10 0 0 1 1
            5 2 20 0 0 0.5 4
            """
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(morphology_path=str(swc_path), cv_per_branch=1)
        )
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )

        branch_pairs = {
            pair["braincell_branch_name"]: pair["neuron_section_name"]
            for pair in result.branch_pairs
        }
        self.assertEqual(branch_pairs["soma"], "soma[0]")
        self.assertEqual(branch_pairs["basal_dendrite_0"], "dend[0]")
        self.assertEqual(branch_pairs["axon_0"], "axon[0]")

    def test_mapping_finds_root_soma_middle_target_for_branched_soma(self) -> None:
        payload = fixtures.base_case_payload(
            morphology_path=fixtures.BRANCHED_SOMA_SWC,
            cv_per_branch=3,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )
        self.assertEqual(result.stimulus_target_pair["braincell_branch_name"], "soma")
        self.assertEqual(result.stimulus_target_pair["neuron_section_name"], "soma[0]")
        self.assertEqual(result.stimulus_target_pair["local_index"], 1)

    def test_mapping_assigns_canonical_soma_names(self) -> None:
        payload = fixtures.base_case_payload(
            morphology_path=fixtures.BRANCHED_SOMA_SWC,
            cv_per_branch=1,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )
        soma_pairs = [
            pair
            for pair in result.branch_pairs
            if pair["braincell_branch_type"] == "soma"
        ]
        canonical_names = [pair["braincell_canonical_name"] for pair in soma_pairs]
        self.assertEqual(canonical_names, ["soma[0]", "soma[1]", "soma[2]"])

    def test_mapping_smoke_real_io_tree(self) -> None:
        payload = fixtures.base_case_payload(
            morphology_path=fixtures.IO_SWC,
            cv_per_branch=1,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )
        self.assertEqual(len(result.branch_pairs), len(braincell_result["branch_order"]))
        self.assertEqual(len(result.branch_pairs), len(neuron_result["section_order"]))
        self.assertIn("braincell_canonical_name", result.branch_pairs[0])
        self.assertIn("match_score", result.branch_pairs[0])

    def test_mapping_supports_simple_asc_tree(self) -> None:
        asc_path = fixtures.write_temp_asc(
            self,
            """
            ("Cell Body"
              (CellBody)
              (0 0 0 0)
              (4 0 0 0)
              (0 4 0 0)
              (-4 0 0 0)
            )
            ( (Dendrite)
              (0 0 0 1)
              (0 5 0 1)
            )
            """,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(
            fixtures.base_case_payload(
                morphology_kind="asc",
                morphology_path=str(asc_path),
                cv_per_branch=1,
            )
        )
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )

        branch_pairs = {
            pair["braincell_branch_name"]: pair["neuron_section_name"]
            for pair in result.branch_pairs
        }
        self.assertEqual(branch_pairs["soma"], "soma[0]")
        self.assertEqual(branch_pairs["dendrite_0"], "dend[0]")

    def test_mapping_supports_real_bc_tree_with_unknown_swc_type(self) -> None:
        payload = fixtures.base_case_payload(
            morphology_path=fixtures.BC_SWC,
            cv_per_branch=1,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_runner.run_case(case)
        neuron_result = neuron_runner.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )

        branch_pairs = {
            pair["braincell_branch_name"]: pair
            for pair in result.branch_pairs
        }
        self.assertIn("custom_0", branch_pairs)
        self.assertEqual(branch_pairs["custom_0"]["neuron_section_name"], "dend_6[0]")
        self.assertEqual(branch_pairs["custom_0"]["braincell_branch_type"], "custom")
        self.assertEqual(branch_pairs["custom_0"]["neuron_section_type"], "dend_6")


if __name__ == "__main__":
    unittest.main()
