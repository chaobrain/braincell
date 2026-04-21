

import importlib.util
import json
from pathlib import Path
import sys


CHANNEL_NO_CONC_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = CHANNEL_NO_CONC_ROOT / "engine"
MOD_VALIDATE_MOD_DIR = "/home/swl/braincell/examples/convert_mod/mod_validate/mods"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_mapping_payload(
    *,
    current_kind: str = "k",
    current: str | None = None,
    impl_name: dict | None = None,
    gate_names: dict | None = None,
    channel_params: dict | None = None,
    params: dict | None = None,
) -> dict:
    current_var = current or {
        "na": "ina",
        "k": "ik",
        "ca": "ica",
    }[current_kind]
    return {
        "current": current_var,
        "impl_name": impl_name or {"neuron": "Kv", "braincell": "IK_Kv_test"},
        "gate_names": gate_names or {"common": ["n"]},
        "channel_params": channel_params or params or {
            "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
        },
    }


def build_identity_payload(*, mod_dir: str = MOD_VALIDATE_MOD_DIR) -> dict:
    return {
        "mod_dir": mod_dir,
    }


def build_case_payload(
    *,
    case_id: str = "kv_case",
    config_name: str = "kv_test",
    template_name: str = "smoke",
    run_id: str | None = None,
    identity: dict | None = None,
    mapping: dict | None = None,
    stimulus: dict | None = None,
) -> dict:
    return {
        "case_id": case_id,
        "config_name": config_name,
        "template_name": template_name,
        "run_id": run_id or f"{config_name}__{template_name}",
        "identity": identity or build_identity_payload(),
        "mapping": mapping or build_mapping_payload(),
        "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
        "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
        "stimulus": stimulus or {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.2},
        **(
            {"ion_state": {"E_mV": -80.0}}
            if (mapping or build_mapping_payload())["current"] in {"ina", "ik", "ica"}
            else {}
        ),
        "channel_params": {"g_max_S_cm2": 0.0},
    }


def build_scan_template_payload(
    *,
    group_id: str = "smoke",
    base: dict | None = None,
    sweep_axes: dict | None = None,
    plot: bool = False,
) -> dict:
    base_payload = base or {
        "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
        "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
        "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
        "ion_state": {"E_mV": -80.0},
        "channel_params": {"g_max_S_cm2": 0.0},
    }
    return {
        "base": base_payload,
        "group": {
            "group_id": group_id,
            "description": f"{group_id} template",
            "sweep_axes": sweep_axes or {},
        },
        "outputs": {"plot": plot},
    }


def build_main_config_payload(
    template_paths: list[str],
    *,
    meta: dict | None = None,
    identity: dict | None = None,
    mapping: dict | None = None,
    defaults: dict | None = None,
) -> dict:
    return {
        "meta": meta or {"label": "test config"},
        "identity": identity or build_identity_payload(),
        **({"defaults": defaults} if defaults is not None else {}),
        "mapping": mapping or build_mapping_payload(),
        "templates": template_paths,
    }


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path
