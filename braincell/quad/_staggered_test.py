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

"""Tests for :mod:`braincell.quad._staggered`.

The full ``staggered_step`` integrator chains a ``dhs_voltage_step``
implicit voltage solve with an ``ind_exp_euler_step`` gating update and
therefore requires a point-tree-aware Cell target. End-to-end behaviour
is exercised by the cell-level test suite. The tests in this file
guard against misuse and verify the registry metadata.
"""

import unittest

import brainstate

from braincell.quad import (
    DiffEqModule,
    get_registry,
    staggered_step,
)


class StaggeredStepGuardTest(unittest.TestCase):

    def test_rejects_plain_module(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaises(AssertionError):
            staggered_step(Plain())

    def test_error_message_mentions_diffeq_module(self):
        class Plain(brainstate.nn.Module):
            pass

        with self.assertRaises(AssertionError) as ctx:
            staggered_step(Plain())
        self.assertIn(DiffEqModule.__name__, str(ctx.exception))


class StaggeredRegistryMetadataTest(unittest.TestCase):

    def test_canonical_name_and_alias(self):
        registry = get_registry()
        self.assertIn("staggered", registry)
        self.assertIn("stagger", registry)
        # ``stagger`` is an alias of the canonical ``staggered`` entry.
        self.assertIs(registry["stagger"], registry["staggered"])
        self.assertIs(registry["staggered"], staggered_step)

    def test_category_and_description(self):
        entry = get_registry().entry("staggered")
        self.assertEqual(entry.category, "staggered")
        self.assertEqual(entry.aliases, ("stagger",))
        self.assertTrue(entry.description)


if __name__ == "__main__":
    unittest.main()
