from __future__ import annotations

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
