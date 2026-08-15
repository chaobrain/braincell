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
