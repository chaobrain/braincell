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

"""Mechanism layouts + clamp active table.

Part of the ARCH-02 logical partition of :mod:`braincell._compute.runtime`.
Exposes the layout-family symbols (layout groupings, clamp evaluation,
fingerprint and state-buffer helpers) through a dedicated module name.

At present all bodies still live in :mod:`braincell._compute.runtime`;
this module re-exports them so callers and tests can address the
layout responsibility group explicitly. A follow-up can physically move
the bodies without changing any call site that imports via
``braincell._compute.layouts``.
"""

from .runtime import (  # noqa: F401
    CLAMP_KINDS,
    ClampActiveTable,
    MechanismLayout,
    _allocate_clamp_ragged_buffer,
    _allocate_state_buffer,
    _eval_current_clamp,
    _eval_function_clamp,
    _eval_sine_clamp,
    _evaluate_clamp_layout,
    _extract_point_value,
    _fn_fingerprint,
    _is_ragged_param,
    _mechanism_var_names,
    _mechanism_var_value,
    _opaque_warned,
    _quantity_sequence_to_decimal_vector,
    _scalar_state_value,
    _write_state_buffer,
    build_clamp_active_table,
    choose_layout,
    mechanism_kind,
    mechanism_signature,
)

__all__ = [
    "CLAMP_KINDS",
    "ClampActiveTable",
    "MechanismLayout",
    "build_clamp_active_table",
    "choose_layout",
    "mechanism_kind",
    "mechanism_signature",
]
