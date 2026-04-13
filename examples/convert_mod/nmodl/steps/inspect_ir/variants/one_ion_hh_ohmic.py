from __future__ import annotations

from typing import Any

from ...model import to_payload
from ...semantic_ir import build_semantic_ir
from ...target_ir import lower_density_channel_ir
from ...target_ir import summarize_density_channel_ir

VARIANT_NAME = "one_ion_hh_ohmic"


def build_one_ion_hh_ohmic_ir(step1_result: dict[str, Any]) -> dict[str, Any]:
    semantic_ir = build_semantic_ir(step1_result["bc_ast_model"])
    target_ir = lower_density_channel_ir(semantic_ir)
    return {
        "semantic_ir_model": semantic_ir,
        "semantic_ir": to_payload(semantic_ir),
        "target_ir_model": target_ir,
        "target_ir": to_payload(target_ir),
        "ir": to_payload(target_ir),
    }


def summarize_one_ion_hh_ohmic_ir(target_ir_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": target_ir_payload["source_file"],
        "supported": target_ir_payload["supported"],
        "class_name": target_ir_payload["class_name"],
        "registry_name": target_ir_payload["registry_name"],
        "base_class_name": target_ir_payload["base_class_name"],
        "ion_name": target_ir_payload["ion_name"],
        "target_family": target_ir_payload["target_family"],
        "g_max_source_name": target_ir_payload["g_max_param"]["source_name"],
        "gate_summary": [
            {
                "name": gate["name"],
                "power": gate["power"],
                "q10": gate["q10_expression"],
                "source_form": gate["source_form"],
            }
            for gate in target_ir_payload.get("gates", [])
        ],
        "current_model": target_ir_payload.get("current_model"),
        "rejection_reasons": target_ir_payload.get("rejection_reasons", []),
    }


def run(step1_result: dict[str, Any]) -> dict[str, Any]:
    compiled = build_one_ion_hh_ohmic_ir(step1_result)
    semantic_ir_payload = compiled["semantic_ir"]
    target_ir_payload = compiled["target_ir"]
    return {
        "variant": VARIANT_NAME,
        "family": target_ir_payload["target_family"],
        "source_file": step1_result["source_file"],
        "supported": target_ir_payload["supported"],
        "rejection_reasons": target_ir_payload.get("rejection_reasons", []),
        "summary": summarize_density_channel_ir(compiled["target_ir_model"]),
        "semantic_ir_model": compiled["semantic_ir_model"],
        "semantic_ir": semantic_ir_payload,
        "target_ir_model": compiled["target_ir_model"],
        "target_ir": target_ir_payload,
        "ir": target_ir_payload,
    }
