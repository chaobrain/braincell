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

Three concerns live here because all are properties of the package as a whole
rather than of any single module: the docstring conformance sweep over every
covered module, the re-export completeness of the explicit import block, and
the agreement between each channel's class name and the key it is registered
under.
"""

import unittest

import pytest

import braincell.channel as channel
from braincell._base_channel import Channel
from braincell._testing import DocstringConformanceTests, ReExportTests
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
from braincell.mech import get_registry

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
        "GhkHH",
        "Markov",
        "OhmicMarkov",
        "freeze_gradient",
        "q10_factor",
        "CaN_IS2008",
        "CaL_IS2008",
        "K_Leak",
        "K_Kv_test",
    }
)


class ChannelDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES
    no_primary_source = _NO_PRIMARY_SOURCE


class ChannelReExportTest(ReExportTests, unittest.TestCase):
    """Guard the explicit re-export block against drift.

    ``braincell/channel/__init__.py`` builds ``__all__`` by concatenating
    the submodules' own ``__all__`` but imports the names one by one. The
    two lists are the reason this guard exists.
    Adding a channel to a submodule's ``__all__`` without adding it to the
    matching import block would leave ``braincell.channel.__all__`` naming
    an attribute the package does not have -- which only fails at
    ``import *`` time, far from the edit that caused it.
    """

    package = channel
    reexport_sources = _COVERED_MODULES
    require_sorted_all = True


def _registered_channel_classes():
    """Return every channel class this package contributes to the registry."""
    return [(name, cls) for name, cls in get_registry().items("channel") if getattr(channel, cls.__name__, None) is cls]


@pytest.mark.parametrize(
    "name",
    [
        "INa_HH1952",  # renamed: the leading ``I`` was dropped
        "IKDR_Ba2002",
        "ICaN_IS2008",
        "ICav12_Ma2020",  # ambiguous: split into region variants
        "Ih_HM1992",  # ambiguous: renamed to HCN_HM1992 family
        "INa_Rsg",  # removed, no successor
        "INa_p3q_markov",  # removed, no successor
        "CalciumChannel",  # removed base class
        "DoesNotExist",
    ],
)
def test_pre_normalization_names_raise_attribute_error(name):
    """The old ``I``-prefixed spellings resolve to nothing.

    They were carried by a ``__getattr__`` shim until this package's
    simplification pass; the shim covered module attribute access only and
    never the mechanism registry, which is the path the documented
    ``mech.Channel("...")`` API uses.
    """
    with pytest.raises(AttributeError):
        getattr(channel, name)


class ChannelRegistryKeyTest(unittest.TestCase):
    """Guard the ``@register_channel("Name")`` argument against drift.

    Every channel in the catalogue is registered under its own class name, so
    the string is pure duplication -- and a rename that updates the class but
    not the decorator would leave ``mech.Channel("...")`` resolving to the
    wrong mechanism, or to nothing, with no other signal.
    """

    def test_every_registry_key_matches_its_class_name(self):
        mismatched = [(key, cls.__name__) for key, cls in _registered_channel_classes() if key != cls.__name__]
        self.assertEqual(mismatched, [], f"registry keys that do not match their class: {mismatched}")

    def test_every_public_channel_class_is_registered(self):
        registered = {cls for _, cls in _registered_channel_classes()}
        # The templates in ``_base`` and the abstract leak base are extension
        # points, not mechanisms, so they are deliberately unregistered.
        templates = set(_base.__all__) | {"LeakageChannel"}
        missing = sorted(
            name
            for name in channel.__all__
            if name not in templates
            and isinstance(getattr(channel, name), type)
            and issubclass(getattr(channel, name), Channel)
            and getattr(channel, name) not in registered
        )
        self.assertEqual(missing, [], f"unregistered public channels: {missing}")


if __name__ == "__main__":
    unittest.main()
