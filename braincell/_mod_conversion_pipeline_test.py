from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NMODL_ROOT = ROOT / "examples" / "convert_mod" / "nmodl"
if str(NMODL_ROOT) not in sys.path:
    sys.path.insert(0, str(NMODL_ROOT))

from steps import ARTIFACT_FILENAMES
from steps import run_pipeline_from_path
from steps import save_pipeline_artifacts
from steps.inspect_ast import resolve_mod_file
from steps.inspect_ast import run as run_inspect_ast
from steps.inspect_ir import run as run_inspect_ir


def _mod_path(name: str) -> str:
    return str(NMODL_ROOT / "mod_files" / name)


class TestModConversionPipeline:
    def test_kv_mod_builds_typed_ast_semantic_ir_and_render_validation(self) -> None:
        result = run_pipeline_from_path(_mod_path("kv.mod"))

        step1_result = result["step1_result"]
        step2_result = result["step2_result"]
        step3_result = result["step3_result"]

        assert step1_result["bc_ast"]["mechanism_name"] == "Kv"
        assert step2_result["semantic_ir"]["module_kind"] == "density_channel"
        assert step2_result["target_ir"]["supported"] is True
        assert step2_result["target_ir"]["target_family"] == "hh_ohmic_inf_tau"
        assert step3_result["validation"]["compiled"] is True
        assert step3_result["validation"]["imported"] is True
        assert "class IK_Kv" in step3_result["rendered_text"]

    def test_alpha_beta_mod_lowers_to_alpha_beta_target_family(self) -> None:
        result = run_pipeline_from_path(_mod_path("na_alpha_beta.mod"))
        target_ir = result["step2_result"]["target_ir"]

        assert target_ir["supported"] is True
        assert target_ir["target_family"] == "hh_ohmic_alpha_beta"
        assert sorted(gate["name"] for gate in target_ir["gates"]) == ["h", "m"]
        assert {gate["source_form"] for gate in target_ir["gates"]} == {"alpha_beta"}

    def test_multi_ion_mod_is_semantically_preserved_but_rejected_for_density_lowering(self) -> None:
        step1_result = run_inspect_ast(resolve_mod_file(_mod_path("hh.mod")))
        step2_result = run_inspect_ir(step1_result)
        semantic_ir = step2_result["semantic_ir"]
        target_ir = step2_result["target_ir"]

        assert len(semantic_ir["useions"]) == 2
        assert "multi_useion" in semantic_ir["unsupported_features"]
        assert target_ir["supported"] is False
        assert any("exactly one USEION" in reason for reason in target_ir["rejection_reasons"])

    def test_save_pipeline_artifacts_writes_current_contract_and_removes_stale_files(self, tmp_path) -> None:
        result = run_pipeline_from_path(_mod_path("kv.mod"))
        artifact_dir = tmp_path / "kv"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stale_file = artifact_dir / "braincell_ir.json"
        stale_file.write_text("stale", encoding="utf-8")

        metadata = save_pipeline_artifacts(result, artifact_dir)

        assert metadata["saved_files"] == list(ARTIFACT_FILENAMES)
        assert stale_file.exists() is False
        assert sorted(path.name for path in artifact_dir.iterdir() if path.is_file()) == sorted(ARTIFACT_FILENAMES)
