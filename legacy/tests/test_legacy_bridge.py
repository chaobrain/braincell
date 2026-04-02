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



import warnings

import braincell


def test_legacy_morphology_symbol_still_accessible_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        morphology_cls = braincell.Morphology
    assert morphology_cls.__name__ == "Morphology"
    assert any("deprecated" in str(item.message).lower() for item in caught)


def test_legacy_morph_module_still_accessible_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        morph_module = braincell.morph
    assert hasattr(morph_module, "from_swc")
    if caught:
        assert any("braincell.morph" in str(item.message) for item in caught)


def test_multi_compartment_still_available() -> None:
    assert braincell.MultiCompartment.__name__ == "MultiCompartment"
