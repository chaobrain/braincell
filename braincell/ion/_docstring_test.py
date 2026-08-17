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

"""Docstring conformance guard for :mod:`braincell.ion`."""

import unittest

from braincell._testing import DocstringConformanceTests
from braincell.ion import _base, calcium, nonspecific, potassium, sodium

# Extended by one module per docstring task. A module is listed only once
# every one of its public symbols satisfies the shared assertions.
_COVERED_MODULES = (_base, calcium, nonspecific, potassium, sodium)

# Public symbols with no primary literature source. Membership must be a
# deliberate decision: a new ion that lands undocumented fails instead of
# silently inheriting an exemption.
_NO_PRIMARY_SOURCE = frozenset(
    {
        "NonSpecific",
        "NonSpecificFixed",
        "Potassium",
        "PotassiumFixed",
        "PotassiumInitNernst",
        "Sodium",
        "SodiumFixed",
        "SodiumInitNernst",
        "Factor",
        "Species",
        "Reaction",
        "Source",
        "Conserve",
        "FixedIon",
        "InitNernstIon",
        "DynamicNernstIon",
        "Calcium",
        "CalciumFixed",
        "CalciumInitNernst",
        "CalciumFirstOrder",
        "ToyCaBindingKinetic_SU2015_DCN",
        "ToyCaBindingSourceKinetic_SU2015_DCN",
        "ToyCaBindingIcaSourceKinetic_SU2015_DCN",
        "ToyDiamFactorKinetic_SU2015_DCN",
        "ToyCaPumpFactorKinetic_SU2015_DCN",
    }
)


class IonDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES
    no_primary_source = _NO_PRIMARY_SOURCE


if __name__ == "__main__":
    unittest.main()
