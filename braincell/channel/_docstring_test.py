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

"""Docstring conformance guard for :mod:`braincell.channel`."""

import unittest

from braincell._testing import DocstringConformanceTests
from braincell.channel import (
    _base,
    calcium,
    hyperpolarization_activated,
    leaky,
    potassium,
    potassium_calcium,
    potassium_sodium,
    sodium,
)

# Extended by one module per docstring task. A module is listed only once
# every one of its public symbols satisfies the shared assertions.
_COVERED_MODULES = (
    _base,
    calcium,
    hyperpolarization_activated,
    leaky,
    potassium,
    potassium_calcium,
    potassium_sodium,
    sodium,
)

# Public symbols with no primary literature source. Membership must be a
# deliberate decision: a new channel that lands undocumented fails instead of
# silently inheriting an exemption.
_NO_PRIMARY_SOURCE = frozenset(
    {
        "LeakageChannel",
        "IL",
        "Gate",
        "Transition",
        "HH",
        "OhmicHH",
        "Markov",
        "CaN_IS2008",
        "CaL_IS2008",
        "K_Leak",
        "K_Kv_test",
    }
)


class ChannelDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES
    no_primary_source = _NO_PRIMARY_SOURCE


if __name__ == "__main__":
    unittest.main()
