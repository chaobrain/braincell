from __future__ import annotations

from typing import Any

from ..jinja_env import render_template

VARIANT_NAME = "braincell_one_ion_hh_ohmic"


def render_braincell_one_ion_hh_ohmic(ir: dict[str, Any]) -> str:
    if not ir.get("supported", False):
        raise SystemExit(
            "Cannot render braincell_one_ion_hh_ohmic for this MOD file. Reasons:\n- "
            + "\n- ".join(ir.get("rejection_reasons", ["Unsupported IR"]))
        )
    return render_template(ir, template_name=ir.get("template_name", "one_ion_hh_ohmic.py"))


def run(step2_result: dict[str, Any]) -> dict[str, Any]:
    rendered = render_braincell_one_ion_hh_ohmic(step2_result["ir"])
    return {
        "variant": VARIANT_NAME,
        "source_file": step2_result["source_file"],
        "summary": step2_result["summary"],
        "rendered_text": rendered,
    }
