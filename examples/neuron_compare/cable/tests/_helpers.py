# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
from pathlib import Path

from examples.neuron_compare._suite_loader import load_suite_module
from examples.neuron_compare._suite_loader import use_double_precision

use_double_precision()


CABLE_ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = CABLE_ROOT / "engine"
TEMPLATES_ROOT = ENGINE_ROOT
TEMPLATE_JSON_ROOT = CABLE_ROOT / "templates"
WORKFLOWS_ROOT = CABLE_ROOT / "workflows"
MORPHO_FILES = Path(__file__).resolve().parents[4] / "data" / "morphology"


def load_module(path: Path, name: str):
    """Load a ``cable`` module by path. See :func:`load_suite_module`."""
    return load_suite_module(CABLE_ROOT, path, name)


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
