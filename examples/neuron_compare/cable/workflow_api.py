"""Compatibility wrapper for cable workflow helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SOURCE = Path(__file__).resolve().parent / "workflows" / "workflow_api.py"
_SPEC = spec_from_file_location("cable_workflow_api_compat", _SOURCE)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Could not load cable workflow API from {_SOURCE!s}.")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in getattr(_MODULE, "__all__", ()):
    globals()[_name] = getattr(_MODULE, _name)

__all__ = tuple(getattr(_MODULE, "__all__", ()))
