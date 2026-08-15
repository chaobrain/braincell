"""Docstring conformance guard for :mod:`braincell.channel`."""

import unittest

from braincell._testing import DocstringConformanceTests
from braincell.channel import (
    _base,
    hyperpolarization_activated,
    leaky,
    potassium_sodium,
    sodium,
)

# Extended by one module per docstring task. A module is listed only once
# every one of its public symbols satisfies the shared assertions.
_COVERED_MODULES = (
    _base,
    hyperpolarization_activated,
    leaky,
    potassium_sodium,
    sodium,
)

# Public symbols with no primary literature source. Membership must be a
# deliberate decision: a new channel that lands undocumented fails instead of
# silently inheriting an exemption.
_NO_PRIMARY_SOURCE = frozenset({
    "LeakageChannel",
    "IL",
    "Gate",
    "Transition",
    "HH",
    "OhmicHH",
    "Markov",
})


class ChannelDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES
    no_primary_source = _NO_PRIMARY_SOURCE


if __name__ == "__main__":
    unittest.main()
