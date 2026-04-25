

import importlib.util
import json
from pathlib import Path
import sys


CABLE_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = CABLE_ROOT / "engine"
TEMPLATES_ROOT = ENGINE_ROOT
TEMPLATE_JSON_ROOT = CABLE_ROOT / "templates"
WORKFLOWS_ROOT = CABLE_ROOT / "workflows"
MORPHO_FILES = Path("/home/swl/braincell/examples/multi_compartment/morpho_files")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_case_payload(
    *,
    case_id: str = "smoke",
    morphology_kind: str = "swc",
    morphology_path: str | None = None,
    stimulus: dict | None = None,
) -> dict:
    return {
        "template_family": "multi_compartment_cable",
        "case_id": case_id,
        "morphology": {
            "path": morphology_path or str(MORPHO_FILES / "unbranched_soma.swc"),
        },
        "simulation": {
            "dt_ms": 0.025,
            "duration_ms": 2.0,
            "v_init_mV": -65.0,
        },
        "cable": {
            "ra_ohm_cm": 100.0,
            "cm_uF_cm2": 1.0,
        },
        "cv_policy": {
            "kind": "CVPerBranch",
            "cv_per_branch": 3,
        },
        "stimulus": stimulus
        or {
            "kind": "dc_step",
            "target": "root_soma_midpoint",
            "delay_ms": 0.0,
            "dur_ms": 2.0,
            "amp_nA": 0.0,
        },
    }
    if morphology_kind is not None:
        payload["morphology"]["kind"] = morphology_kind
    return payload


def build_config_defaults_payload(
    *,
    morphology_kind: str | None = None,
    morphology_path: str | None = None,
) -> dict:
    payload = {
        "morphology": {
            "path": morphology_path or str(MORPHO_FILES / "unbranched_soma.swc"),
        },
    }
    if morphology_kind is not None:
        payload["morphology"]["kind"] = morphology_kind
    return payload


def build_scan_template_payload(
    *,
    group_id: str = "smoke",
    base: dict | None = None,
    sweep_axes: dict | None = None,
    plot: bool = False,
) -> dict:
    return {
        "meta": {"label": f"{group_id} template"},
        "base": base
        or {
            "simulation": build_case_payload()["simulation"],
            "cable": build_case_payload()["cable"],
            "cv_policy": build_case_payload()["cv_policy"],
            "stimulus": build_case_payload()["stimulus"],
        },
        "group": {
            "group_id": group_id,
            "description": f"{group_id} template",
            "sweep_axes": sweep_axes or {},
        },
        "outputs": {"plot": plot},
    }


def build_model_config_payload(
    template_paths: list[str],
    *,
    meta: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    return {
        "meta": meta or {"label": "cable test config"},
        "defaults": defaults or build_config_defaults_payload(),
        "templates": template_paths,
    }


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path
