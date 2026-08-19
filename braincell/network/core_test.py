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

"""Tests for :mod:`braincell.network.core`."""

import unittest

import numpy as np

from braincell import CVPerBranch, Cell
from braincell.network import (
    Connection,
    Network,
    Population,
)
from braincell.network._testing import (
    make_probe_cell,
    make_soma_tree,
    make_spiking_cell,
)


class PopulationTest(unittest.TestCase):
    def test_population_accepts_one_dimensional_cell_pop_size(self) -> None:
        pop = Population("E", make_spiking_cell(size=3))
        self.assertEqual(pop.size, 3)

    def test_population_accepts_default_single_cell_pop_size(self) -> None:
        # A Cell now always carries a population axis, so the default is the
        # one-dimensional ``(1,)`` that Population requires.
        pop = Population("one", Cell(make_soma_tree(), cv_policy=CVPerBranch()))
        self.assertEqual(pop.size, 1)

    def test_cell_rejects_rank_zero_pop_size(self) -> None:
        # The guard that used to live in Population now lives in Cell itself.
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=())

    def test_population_rejects_multi_dimensional_pop_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            Population("grid", Cell(make_soma_tree(), cv_policy=CVPerBranch(), pop_size=(2, 2)))

    def test_population_repr_is_compact(self) -> None:
        population = net_population = Network().add_population("E", make_probe_cell(size=3))

        text = repr(population)

        self.assertIs(population, net_population)
        self.assertIn("Population", text)
        self.assertIn("size=3", text)
        self.assertIn("initialized=False", text)
        self.assertNotIn("Cell(root=", text)


class ConnectionTest(unittest.TestCase):
    def test_connection_normalizes_index_arrays(self) -> None:
        conn = Connection(
            pre_population="E",
            post_population="I",
            pre_index=[0, 2],
            post_index=[1, 0],
            synapse="ampa",
        )

        self.assertEqual(conn.n_contact, 2)
        self.assertEqual(conn.n_edge, 2)
        np.testing.assert_array_equal(conn.pre_index, np.asarray([0, 2], dtype=np.int32))
        np.testing.assert_array_equal(conn.post_index, np.asarray([1, 0], dtype=np.int32))

    def test_connection_rejects_mismatched_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            Connection(
                pre_population="E",
                post_population="I",
                pre_index=[0, 1],
                post_index=[0],
                synapse="ampa",
            )

    def test_connection_rejects_non_integer_indices(self) -> None:
        with self.assertRaisesRegex(TypeError, "integers"):
            Connection(
                pre_population="E",
                post_population="I",
                pre_index=[0.5],
                post_index=[0],
                synapse="ampa",
            )

    def test_connection_repr_is_compact(self) -> None:
        conn = Connection("E", "I", [0, 2], [1, 0], "ampa")

        text = repr(conn)

        self.assertIn("Connection", text)
        self.assertIn("n_contact=2", text)
        self.assertNotIn("array(", text)


if __name__ == "__main__":
    unittest.main()
