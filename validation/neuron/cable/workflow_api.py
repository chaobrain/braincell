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
