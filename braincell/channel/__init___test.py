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

"""Package-level guards for :mod:`braincell.channel`.

Two concerns live here because both are properties of the package as a whole
rather than of any single module: the docstring conformance sweep over every
covered module, and the ``__getattr__`` deprecation shim declared in
``braincell/channel/__init__.py``.
"""

import unittest

import pytest

import braincell.channel as channel
from braincell._testing import DocstringConformanceTests
from braincell.channel import (
    _DEPRECATED_ALIASES,
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


@pytest.mark.parametrize("old_name, new_name", sorted(_DEPRECATED_ALIASES.items()))
def test_deprecated_alias_resolves_with_warning(old_name, new_name):
    with pytest.warns(DeprecationWarning, match=new_name):
        resolved = getattr(channel, old_name)
    assert resolved is getattr(channel, new_name)


def test_deprecated_names_absent_from_all():
    for old_name in _DEPRECATED_ALIASES:
        assert old_name not in channel.__all__


@pytest.mark.parametrize(
    "name",
    [
        "ICav12_Ma2020",  # ambiguous: split into region variants
        "Ih_HM1992",  # ambiguous: renamed to HCN_HM1992 family
        "INa_Rsg",  # removed, no successor
        "INa_p3q_markov",  # removed, no successor
        "CalciumChannel",  # removed base class
        "DoesNotExist",
    ],
)
def test_non_aliased_names_raise_attribute_error(name):
    with pytest.raises(AttributeError):
        getattr(channel, name)


if __name__ == "__main__":
    unittest.main()
