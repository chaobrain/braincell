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

import dataclasses
import unittest

import braincell
import brainunit as u

from braincell.filter import AllRegion, at
from braincell.reduction.dbnn import signature
from braincell.reduction.dbnn.signature import build_cell_signature


def _cell(*synapses, length=20.0, cv_per_branch=1):
    soma = braincell.Branch.from_lengths(
        lengths=[length] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dendrite = braincell.Branch.from_lengths(
        lengths=[40.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="dendrite",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    morphology.soma.dendrite = dendrite
    cell = braincell.Cell(
        morphology,
        cv_policy=braincell.CVPerBranch(cv_per_branch),
        pop_size=(1,),
    )
    if synapses:
        cell.place(at("dendrite", 0.25), *synapses)
    return cell


class CellSignatureTest(unittest.TestCase):
    def test_module_exports_only_structural_signature_api(self):
        self.assertEqual(signature.__all__, ["CellStructuralSignature", "build_cell_signature"])

    def test_explicit_cell_type_is_preferred(self):
        first = build_cell_signature(_cell(), cell_type="SC_MA2021")
        second = build_cell_signature(_cell(length=200.0, cv_per_branch=3), cell_type="SC_MA2021")
        self.assertEqual(first.key, ("explicit", "SC_MA2021"))
        self.assertEqual(first, second)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.cell_type = "other"

    def test_inferred_key_is_canonical_and_structural(self):
        synapse = braincell.mech.Synapse("ExpSyn", name="E")
        first = build_cell_signature(_cell(synapse, length=20.0))
        equivalent = build_cell_signature(_cell(synapse, length=200.0))
        different_cvs = build_cell_signature(_cell(synapse, cv_per_branch=2))
        different_synapses = build_cell_signature(_cell(synapse, synapse))

        self.assertEqual(first.key[0], "inferred")
        self.assertEqual(len(first.digest), 64)
        self.assertEqual(first, equivalent)
        self.assertNotEqual(first.key, different_cvs.key)
        self.assertNotEqual(first.key, different_synapses.key)
        self.assertNotIn("object at", first.canonical_json)

    def test_inferred_key_includes_installed_mechanism_kinds_not_parameters(self):
        first = _cell()
        first.paint(
            AllRegion(),
            braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-70.0 * u.mV),
        )
        same_kind = _cell()
        same_kind.paint(
            AllRegion(),
            braincell.mech.Channel("IL", g_max=0.2 * u.mS / u.cm**2, E=-60.0 * u.mV),
        )
        different_kind = _cell()
        different_kind.paint(
            AllRegion(),
            braincell.mech.Channel("Na_HH1952", g_max=12.0 * u.mS / u.cm**2),
        )
        self.assertEqual(build_cell_signature(first).key, build_cell_signature(same_kind).key)
        self.assertNotEqual(build_cell_signature(first).key, build_cell_signature(different_kind).key)

    def test_bad_cell_type_and_non_cell_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_cell_signature(_cell(), cell_type="")
        with self.assertRaisesRegex(TypeError, "Cell"):
            build_cell_signature(object())
        cell = _cell()
        cell.init_state()
        with self.assertRaisesRegex(RuntimeError, "before Cell.init_state"):
            build_cell_signature(cell, cell_type="explicit")


if __name__ == "__main__":
    unittest.main()
