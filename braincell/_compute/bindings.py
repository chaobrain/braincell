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

"""Channel-to-ion runtime binding resolution + node instantiation.

Part of the ARCH-02 logical partition of :mod:`braincell._compute.runtime`.
Exposes the channel-binding and node-instantiation helpers through a
dedicated module name.
"""

from .runtime import (  # noqa: F401
    _BoundIonChannelRuntime,
    _build_runtime_nodes,
    _channel_current_owner_family,
    _channel_current_owner_specs,
    _channel_family_slots,
    _instantiate_runtime_node,
    _is_root_level_runtime_node,
    _quantity_full,
    _resolve_channel_runtime_bindings,
    _resolve_ion_instance_key,
    _root_type_to_family,
    _runtime_constructor_params,
    _runtime_param_value,
    _sync_runtime_node_param,
)
