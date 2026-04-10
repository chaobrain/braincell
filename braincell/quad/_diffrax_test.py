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

"""Tests for :mod:`braincell.quad._diffrax`.

The diffrax-backed step functions are gated on whether ``diffrax`` is
importable. These tests verify that:

* the ``diffrax_*_step`` symbols are always importable from
  :mod:`braincell.quad`,
* the integrator registry contains the canonical names if-and-only-if
  ``diffrax`` is installed,
* the lazy ``__getattr__`` on the submodule either returns the diffrax
  package or raises ``ModuleNotFoundError`` cleanly.
"""

import importlib.util
import unittest

import braincell.quad as quad
from braincell.quad import get_integrator, get_registry
from braincell.quad._diffrax import diffrax_installed


DIFFRAX_STEP_NAMES = [
    "diffrax_euler_step",
    "diffrax_heun_step",
    "diffrax_midpoint_step",
    "diffrax_ralston_step",
    "diffrax_bosh3_step",
    "diffrax_tsit5_step",
    "diffrax_dopri5_step",
    "diffrax_dopri8_step",
    "diffrax_bwd_euler_step",
    "diffrax_kvaerno3_step",
    "diffrax_kvaerno4_step",
    "diffrax_kvaerno5_step",
]

DIFFRAX_REGISTRY_NAMES = [name[:-5] for name in DIFFRAX_STEP_NAMES]  # strip "_step"


class DiffraxAvailabilityTest(unittest.TestCase):

    def test_diffrax_installed_matches_find_spec(self):
        spec = importlib.util.find_spec("diffrax")
        self.assertEqual(diffrax_installed, spec is not None)

    def test_step_functions_are_importable(self):
        # The functions should always be present on the package, even when
        # diffrax itself is missing.
        for name in DIFFRAX_STEP_NAMES:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(quad, name)))

    def test_registry_membership_matches_install_state(self):
        registry = get_registry()
        for name in DIFFRAX_REGISTRY_NAMES:
            with self.subTest(name=name):
                if diffrax_installed:
                    self.assertIn(name, registry)
                    self.assertEqual(
                        registry.entry(name).category, "diffrax"
                    )
                else:
                    self.assertNotIn(name, registry)
                    with self.assertRaises(ValueError):
                        get_integrator(name)

    def test_lazy_import_via_module_getattr(self):
        # Force-import the submodule so its module-level ``__getattr__`` is
        # in play. Looking up the ``diffrax`` attribute should either
        # succeed (when installed) or raise ``ModuleNotFoundError``.
        from braincell.quad import _diffrax
        if diffrax_installed:
            mod = _diffrax.diffrax  # noqa: F841 - exercise the lazy import
        else:
            with self.assertRaises(ModuleNotFoundError):
                _diffrax.diffrax  # noqa: B018

    def test_unknown_diffrax_attribute_raises_attribute_error(self):
        from braincell.quad import _diffrax
        with self.assertRaises(AttributeError):
            _diffrax.totally_made_up  # noqa: B018


if __name__ == "__main__":
    unittest.main()
