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

import pprint
from pathlib import Path
from typing import Any


def template_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "templates"


def render_template(
    context: dict[str, Any],
    *,
    template_name: str,
) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError as exc:
        message = "Unable to import `jinja2`. Run `python -m pip install jinja2`."
        raise SystemExit(message) from exc

    environment = Environment(
        loader=FileSystemLoader(str(template_dir())),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template(template_name)
    return template.render(
        context=context,
        context_python=pprint.pformat(context, sort_dicts=False, width=100),
    )
