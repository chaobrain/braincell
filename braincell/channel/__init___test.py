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

Four concerns live here because all are properties of the package as a whole
rather than of any single module: the docstring conformance sweep over every
covered module, the re-export completeness of the explicit import block, the
agreement between each channel's class name and the key it is registered
under, and the static unit-safety sweep.
"""

import unittest
from pathlib import Path

import pytest

import braincell.channel as channel
from braincell._base_channel import Channel
from braincell._testing import DocstringConformanceTests, ReExportTests, UnitSafetyGuardTests
from braincell.channel import (
    _base,
    calcium,
    chloride,
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
    chloride,
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
        "K_Leak",
        "Cl_Leak",
        "Na_Leak",
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


class ChannelUnitSafetyTest(UnitSafetyGuardTests, unittest.TestCase):
    """No bare-number reads, ``TypeError`` fallbacks or NMODL unit factors (MECH-27)."""

    package_dir = Path(__file__).resolve().parent
    check_factors = True
    allowlist = (
        ("_base.py", "freeze_gradient", "get_mantissa", "stop_gradient needs an array; the same unit is put back."),
        ("_base.py", "is_disabled", "get_mantissa", "tests a switch for exact zero, which is zero in any unit."),
        ("_base.py", "_steady_state_inputs.flatten", "get_magnitude", "rates come bare (1/ms) from _transition_rate."),
        ("_base.py", "Markov._solve_steady_state_jax", "get_magnitude", "uses only the zero template's shape."),
        ("_base.py", "Markov._solve_steady_state_host", "get_magnitude", "uses only the zero template's shape."),
        ("potassium.py", "Kv1p5_MA2024_PC.f_n_tau", "factor 1e-3", "the .mod rate coefficient 0.001/ms, not a unit."),
    )


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


class RenamedChannelTest(unittest.TestCase):
    """braincell 0.3 renamed ``CaHT_HM1992`` (MECH-42) and ``CaL_IS2008`` (MECH-49), keeping no alias."""

    def test_access_names_the_new_name_and_the_release(self):
        with self.assertRaises(AttributeError) as caught:
            channel.CaHT_HM1992
        message = str(caught.exception)
        self.assertTrue(message.startswith("braincell.channel.CaHT_HM1992 was renamed in braincell 0.3."), message)
        self.assertIn("braincell.channel.CaT_HM1992_Shifted25", message)
        self.assertIn("mech.Channel('CaT_HM1992_Shifted25'", message)
        self.assertFalse(hasattr(channel, "CaHT_HM1992"))
        self.assertNotIn("CaHT_HM1992", channel.__all__)

    def test_from_import_raises_import_error_with_the_message(self):
        with self.assertRaisesRegex(ImportError, r"^braincell\.channel\.CaHT_HM1992 was renamed in braincell 0\.3\."):
            from braincell.channel import CaHT_HM1992  # noqa: F401

    def test_cal_is2008_names_cal_li2017(self):
        # MECH-49: the class reproduces Li et al. (2017); Inoue & Strowbridge (2008) has no L-type current.
        with self.assertRaises(AttributeError) as caught:
            channel.CaL_IS2008
        message = str(caught.exception)
        self.assertTrue(message.startswith("braincell.channel.CaL_IS2008 was renamed in braincell 0.3."), message)
        self.assertIn("braincell.channel.CaL_Li2017", message)
        self.assertIn("mech.Channel('CaL_Li2017'", message)
        self.assertFalse(hasattr(channel, "CaL_IS2008"))
        self.assertNotIn("CaL_IS2008", channel.__all__)
        with self.assertRaisesRegex(ImportError, r"^braincell\.channel\.CaL_IS2008 was renamed in braincell 0\.3\."):
            from braincell.channel import CaL_IS2008  # noqa: F401

    def test_other_missing_names_keep_the_generic_message(self):
        with self.assertRaisesRegex(AttributeError, "module 'braincell.channel' has no attribute 'NoSuchChannel'"):
            channel.NoSuchChannel


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
