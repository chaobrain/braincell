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

"""Tests for :mod:`braincell.network.edges`."""

import unittest

import brainunit as u
import numpy as np

import braincell.network as network
from braincell.network import (
    EdgeSet,
    Network,
    all_pairs,
    dense,
    pairs,
    probability,
)
from braincell.network._testing import (
    make_post_cell,
    make_probe_cell,
    make_spiking_cell,
)


class EdgeSetTest(unittest.TestCase):
    def test_all_to_all_builder_is_not_exported(self) -> None:
        self.assertFalse(hasattr(network, "all_to_all"))

    def test_low_level_build_helper_is_not_exported(self) -> None:
        self.assertFalse(hasattr(network, "build"))

    def test_edge_set_stores_cell_level_edges_without_synapse(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [2, 3])

        self.assertEqual(edges.n_edge, 2)
        np.testing.assert_array_equal(edges.pre_index, [0, 1])
        np.testing.assert_array_equal(edges.post_index, [2, 3])

    def test_edge_set_repr_is_compact(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [2, 3])

        text = repr(edges)

        self.assertIn("EdgeSet", text)
        self.assertIn("n_edge=2", text)
        self.assertNotIn("array(", text)

    def test_network_add_edges_accepts_probability_and_callable_methods(self) -> None:
        net = Network(name="edge_method_demo")
        net.add_population("E", make_probe_cell(size=4))
        net.add_population("I", make_probe_cell(size=4))

        sampled = net.add_edges(
            name="sampled",
            pre="E",
            post="I",
            method=network.probability(p=0.5, seed=1),
        )

        def diagonal(*, n_pre: int, n_post: int):
            index = np.arange(min(n_pre, n_post), dtype=np.int32)
            return index, index

        custom = net.add_edges(
            name="diagonal",
            pre="E",
            post="I",
            method=diagonal,
        )

        self.assertGreaterEqual(sampled.n_edge, 0)
        np.testing.assert_array_equal(custom.pre_index, [0, 1, 2, 3])
        np.testing.assert_array_equal(custom.post_index, [0, 1, 2, 3])


class NetworkAddEdgesTest(unittest.TestCase):
    def test_add_edges_and_project_store_named_objects(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network(name="demo")
        net.add_population("E", pre)
        net.add_population("I", post)

        edges = net.add_edges(
            name="E_to_I",
            pre="E",
            post="I",
            method=all_pairs(pre_indices=[0], post_indices=[1]),
        )
        projection = net.project(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            weight=0.5 * u.uS,
        )

        self.assertEqual(net.name, "demo")
        self.assertIs(net.edge_sets["E_to_I"], edges)
        self.assertIs(net.proj["E_to_I_exp"], projection)
        self.assertIs(net.projections["E_to_I_exp"], projection)

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[1, 1]), 0.5, places=6)

    def test_add_edges_pairs_dense_all_pairs_and_probability_builders(self) -> None:
        pre = make_spiking_cell(size=3)
        post = make_post_cell(size=3)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        explicit = net.add_edges(
            name="explicit",
            pre="E",
            post="I",
            method=pairs([(2, 1), (0, 2)]),
        )
        dense_edges = net.add_edges(
            name="dense",
            pre="E",
            post="I",
            method=dense(
                [
                    [False, True, False],
                    [True, False, True],
                    [False, False, False],
                ]
            ),
        )
        all_pair_edges = net.add_edges(
            name="all_pair",
            pre="E",
            post="I",
            method=all_pairs(pre_indices=[0, 2], post_indices=[1, 2]),
        )
        sampled_a = net.add_edges(
            name="sampled_a",
            pre="E",
            post="I",
            method=probability(p=0.5, seed=11),
        )
        sampled_b = net.add_edges(
            name="sampled_b",
            pre="E",
            post="I",
            method=probability(p=0.5, seed=11),
        )

        np.testing.assert_array_equal(explicit.pre_index, [2, 0])
        np.testing.assert_array_equal(explicit.post_index, [1, 2])
        np.testing.assert_array_equal(dense_edges.pre_index, [0, 1, 1])
        np.testing.assert_array_equal(dense_edges.post_index, [1, 0, 2])
        np.testing.assert_array_equal(all_pair_edges.pre_index, [0, 0, 2, 2])
        np.testing.assert_array_equal(all_pair_edges.post_index, [1, 2, 1, 2])
        np.testing.assert_array_equal(sampled_a.pre_index, sampled_b.pre_index)
        np.testing.assert_array_equal(sampled_a.post_index, sampled_b.post_index)

    def test_add_edges_accepts_custom_callable_method(self) -> None:
        pre = make_spiking_cell(size=3)
        post = make_post_cell(size=3)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        def ring(*, n_pre: int, n_post: int):
            pre_index = list(range(n_pre))
            post_index = [(index + 1) % n_post for index in pre_index]
            return pre_index, post_index

        edges = net.add_edges(name="ring", pre="E", post="I", method=ring)

        self.assertEqual(edges.pre_index.dtype, np.dtype(np.int32))
        self.assertEqual(edges.post_index.dtype, np.dtype(np.int32))
        np.testing.assert_array_equal(edges.pre_index, [0, 1, 2])
        np.testing.assert_array_equal(edges.post_index, [1, 2, 0])

    def test_add_edges_validates_custom_callable_bounds(self) -> None:
        pre = make_spiking_cell(size=2)
        post = make_post_cell(size=2)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        def out_of_range(*, n_pre: int, n_post: int):
            _ = (n_pre, n_post)
            return [0, 2], [0, 1]

        with self.assertRaisesRegex(IndexError, "pre_index out of range"):
            net.add_edges(name="bad", pre="E", post="I", method=out_of_range)


if __name__ == "__main__":
    unittest.main()
