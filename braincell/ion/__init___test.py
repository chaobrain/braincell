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

"""Package-level guards for :mod:`braincell.ion`.

Three concerns live here because all are properties of the package as a
whole: the docstring conformance sweep over every covered module, the
re-export completeness of ``braincell/ion/__init__.py``, and the static
unit-safety sweep.
"""

import unittest
from pathlib import Path

import braincell.ion as ion
from braincell._testing import DocstringConformanceTests, ReExportTests, UnitSafetyGuardTests
from braincell.ion import _base, calcium, chloride, nonspecific, potassium, sodium

# Extended by one module per docstring task. A module is listed only once
# every one of its public symbols satisfies the shared assertions.
_COVERED_MODULES = (_base, calcium, chloride, nonspecific, potassium, sodium)

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
        "Chloride",
        "ChlorideFixed",
        "ChlorideInitNernst",
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


class IonReExportTest(ReExportTests, unittest.TestCase):
    """Guard the explicit re-export block against drift.

    ``braincell/ion/__init__.py`` builds ``__all__`` by concatenating the
    submodules' own ``__all__`` but imports the names one by one. Adding a
    name to a submodule's ``__all__`` without adding it to the matching
    import block would leave ``braincell.ion.__all__`` naming an attribute
    the package does not have -- which only fails at ``import *`` time,
    far from the edit that caused it.
    """

    package = ion
    reexport_sources = (calcium, chloride, nonspecific, potassium, sodium)
    require_sorted_all = True


class IonUnitSafetyTest(UnitSafetyGuardTests, unittest.TestCase):
    """No bare-number reads or ``TypeError`` fallbacks (MECH-27)."""

    package_dir = Path(__file__).resolve().parent
    allowlist = (("_testing.py", "make_shell_ion", "get_mantissa", "fixture: the same unit is put back."),)


if __name__ == "__main__":
    unittest.main()
