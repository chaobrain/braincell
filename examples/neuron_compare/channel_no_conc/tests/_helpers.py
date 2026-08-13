

import importlib.util
import json
from pathlib import Path
import sys

import brainstate

# NEURON integrates in double precision, and these suites compare against it at
# tolerances as tight as 2e-5 mV. Under the float32 default, rounding accumulates
# through the capacitive voltage update -- roughly 0.03 mV of monotonic drift over
# a 4 ms run, growing with step count -- which swamps those tolerances. Set it
# here, in the module every test in this package imports, rather than relying on
# `cable/tests/test_runner.py` having happened to set the same process-global
# first: run this package alone and the comparison tests would otherwise fail.
brainstate.environ.set(precision=64)


CHANNEL_NO_CONC_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = CHANNEL_NO_CONC_ROOT / "engine"

# Resolved from this file rather than hardcoded: `workflow_api.load_workflow_inputs`
# raises FileNotFoundError when `identity.mod_dir` does not exist, so an absolute
# path into one developer's home directory fails everywhere else.
EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
MOD_VALIDATE_MOD_DIR = str(EXAMPLES_ROOT / "convert_mod" / "mod_validate" / "mods")

_SUITES_ROOT = CHANNEL_NO_CONC_ROOT.parent


def load_module(path: Path, name: str):
    """Load a suite module under ``name`` with its real package attached.

    The engine modules import their siblings relatively (``from .compare
    import ...``) and fall back to bare absolute names on ImportError. Loading
    them with no package makes the relative import fail every time, so the
    fallback registers ``compare`` / ``experiment_schema`` / ``outputs`` as
    top-level modules -- names the sibling ``cable`` suite also uses. Whichever
    suite ran first then owned the name and the other imported the wrong file.

    Loading under a dotted name inside ``channel_no_conc.<subpackage>`` lets the
    relative imports resolve, so the fallback never runs and nothing lands in
    ``sys.modules`` under an unqualified name. The package is taken from the
    file's own directory, so this works for ``engine/`` and ``workflows/``
    alike. The caller's ``name`` is kept as the leaf, so callers that load the
    same file twice still get independent module objects and cannot leak state
    into each other.
    """
    if str(_SUITES_ROOT) not in sys.path:
        sys.path.insert(0, str(_SUITES_ROOT))

    package = f"{CHANNEL_NO_CONC_ROOT.name}.{path.resolve().parent.name}"
    qualified = f"{package}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
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
        "cal": "ical",
    }[current_kind]
    return {
        "current": current_var,
        "impl_name": impl_name or {"neuron": "Kv", "braincell": "K_Kv_test"},
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
            if (mapping or build_mapping_payload())["current"] in {"ina", "ik", "ica", "ical"}
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
    identity = payload.get("identity") if isinstance(payload, dict) else None
    if isinstance(identity, dict):
        mod_dir = identity.get("mod_dir")
        if isinstance(mod_dir, str) and mod_dir and not Path(mod_dir).is_absolute():
            (path.parent / mod_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path
