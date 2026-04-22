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

"""Runtime ion-instance construction + param normalization.

Part of the ARCH-02 logical partition of :mod:`braincell._compute.runtime`.
Exposes the ion-runtime family helpers through a dedicated module name.
"""

from .runtime import (  # noqa: F401
    _build_default_ions,
    _build_ion_alias_map,
    _build_runtime_ions,
    _collect_runtime_ion_instances,
    _instantiate_runtime_ion_instance,
    _ion_param_broadcast,
    _ion_param_scatter,
    _ion_runtime_attr_name,
    _normalize_ion_runtime_param_value,
    _runtime_ion_family,
    _runtime_ion_species_key,
    _supported_ion_runtime_params,
    _sync_runtime_ion,
)
