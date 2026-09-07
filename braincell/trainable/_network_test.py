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

"""Live Network aggregation preserves original optimizer roots."""

import gc
import unittest

import braincell
import brainstate
import brainunit as u

from braincell.filter import at
from braincell.network._testing import make_soma_tree
from braincell.trainable._network import NetworkTrainables


class NetworkTrainablesTest(unittest.TestCase):
    def test_live_collection_shared_roots_and_bindings(self):
        net = braincell.Network("shared")
        parameters = net.trainables.parameters()
        root = brainstate.nn.Param(1.0)
        for name in ("b", "a"):
            cell = braincell.Cell(make_soma_tree(), pop_size=(1,))
            cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn"))
            cell.synapses["syn"].trainable(tau=braincell.trainable.scale(root, name="factor"))
            net.add_population(name, cell)
        self.assertEqual(tuple(parameters.states()), ("a.factor",))
        self.assertEqual(len(net.trainables.roots), 1)
        self.assertIs(net.trainables.roots["a.factor"], root)
        self.assertIs(parameters.states()["a.factor"], root.val)
        self.assertEqual(len(net.trainables.bindings()), 2)
        self.assertTrue(all(b.root_names == ("a.factor",) for b in net.trainables.bindings()))
        net.prepare_run(dt=0.1 * u.ms)
        parameters.set_physical_values({"a.factor": 2.0})
        net.reset_state()
        for population in net.populations.values():
            self.assertAlmostEqual(float(population.cell.synapses["syn"].tau[0] / u.ms), 0.2)

    def test_collected_owner_cannot_be_used(self):
        net = braincell.Network("temporary")
        manager = NetworkTrainables(net)
        del net
        gc.collect()
        with self.assertRaisesRegex(RuntimeError, "no longer exists"):
            list(manager.roots)

    def test_ambiguous_qualified_names_are_rejected(self):
        net = braincell.Network("ambiguous")
        for population, parameter in (("a.b", "c"), ("a", "b.c")):
            cell = braincell.Cell(make_soma_tree(), pop_size=(1,))
            cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn"))
            cell.synapses["syn"].trainable(tau=braincell.trainable.scale(name=parameter))
            net.add_population(population, cell)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            net.trainables.parameters().states()
