from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ROOT = Path(__file__).resolve().parent
case_schema = _load_module(_ROOT / "case_schema.py", "mc_cable_mapping_case_schema")
braincell_single_case = _load_module(_ROOT / "braincell_single_case.py", "mc_cable_mapping_bc_runner")
neuron_single_case = _load_module(_ROOT / "neuron_single_case.py", "mc_cable_mapping_nrn_runner")
mapping = _load_module(_ROOT / "mapping.py", "mc_cable_mapping_helper")
fixtures = _load_module(_ROOT / "fixtures.py", "mc_cable_mapping_fixtures")


class MappingTest(unittest.TestCase):
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
            fixtures.base_case_payload(swc_path=str(swc_path), cv_per_branch=1)
        )
        braincell_result = braincell_single_case.run_case(case)
        neuron_result = neuron_single_case.run_case(case)
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
            swc_path=fixtures.BRANCHED_SOMA_SWC,
            cv_per_branch=3,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_single_case.run_case(case)
        neuron_result = neuron_single_case.run_case(case)
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
            swc_path=fixtures.BRANCHED_SOMA_SWC,
            cv_per_branch=1,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_single_case.run_case(case)
        neuron_result = neuron_single_case.run_case(case)
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
            swc_path=fixtures.IO_SWC,
            cv_per_branch=1,
        )
        case = case_schema.MultiCompartmentCableCase.from_dict(payload)
        braincell_result = braincell_single_case.run_case(case)
        neuron_result = neuron_single_case.run_case(case)
        result = mapping.build_mapping(
            case,
            braincell_result=braincell_result,
            neuron_result=neuron_result,
        )
        self.assertEqual(len(result.branch_pairs), len(braincell_result["branch_order"]))
        self.assertEqual(len(result.branch_pairs), len(neuron_result["section_order"]))
        self.assertIn("braincell_canonical_name", result.branch_pairs[0])
        self.assertIn("match_score", result.branch_pairs[0])


if __name__ == "__main__":
    unittest.main()
