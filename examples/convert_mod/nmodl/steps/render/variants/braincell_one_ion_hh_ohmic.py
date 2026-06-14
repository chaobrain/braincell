

import types
from typing import Any

from braincell.mech import get_registry

from ...model import RenderValidation
from ...target_ir import attach_validation
from ..jinja_env import render_template

VARIANT_NAME = "braincell_one_ion_hh_ohmic"


def _validate_rendered_module(rendered: str, *, class_name: str, registry_name: str) -> RenderValidation:
    registry = get_registry()
    already_registered = registry.contains("channel", registry_name)
    try:
        compiled = compile(rendered, f"<generated:{registry_name}>", "exec")
    except Exception as exc:
        return RenderValidation(compiled=False, imported=False, error=str(exc))

    module = types.ModuleType(f"_generated_{registry_name}")
    try:
        exec(compiled, module.__dict__)
        klass = getattr(module, class_name)
    except Exception as exc:
        return RenderValidation(compiled=True, imported=False, error=str(exc))
    finally:
        if not already_registered and registry.contains("channel", registry_name):
            try:
                registry.unregister("channel", registry_name)
            except Exception:
                pass

    return RenderValidation(compiled=True, imported=True, class_name=getattr(klass, "__name__", class_name))


def render_braincell_one_ion_hh_ohmic(ir: dict[str, Any]) -> str:
    if not ir.get("supported", False):
        raise SystemExit(
            "Cannot render braincell_one_ion_hh_ohmic for this MOD file. Reasons:\n- "
            + "\n- ".join(ir.get("rejection_reasons", ["Unsupported IR"]))
        )
    return render_template(ir, template_name=ir.get("render_metadata", {}).get("template_name", "density_channel.py"))


def run(step2_result: dict[str, Any]) -> dict[str, Any]:
    target_ir_model = step2_result["target_ir_model"]
    rendered = render_braincell_one_ion_hh_ohmic(step2_result["target_ir"])
    validation = _validate_rendered_module(
        rendered,
        class_name=target_ir_model.class_name,
        registry_name=target_ir_model.registry_name,
    )
    target_ir_model = attach_validation(target_ir_model, validation)
    return {
        "variant": VARIANT_NAME,
        "source_file": step2_result["source_file"],
        "summary": {**step2_result["summary"], "validation": validation.__dict__},
        "target_ir": target_ir_model,
        "rendered_text": rendered,
        "validation": validation.__dict__,
    }
