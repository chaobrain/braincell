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

"""Tests for :mod:`braincell.network.projections`."""

import unittest

import brainunit as u
import numpy as np

import braincell.network as network
from braincell.network import (
    ContactTable,
    EdgeSet,
    Network,
    Projection,
    by_post,
    explicit_contacts,
    pairs,
    per_edge,
)
from braincell.network._testing import (
    make_post_cell,
    make_post_cell_with_synapse_pool,
    make_spiking_cell,
)


class ProjectionTest(unittest.TestCase):
    def test_projection_reuses_edges_with_explicit_contacts(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [2, 3])
        projection = Projection(
            name="E_to_I_ampa",
            edges="E_to_I",
            synapse="ampa",
            method=explicit_contacts(source_edge=[0, 0, 1], synapse_index=[0, 1, 1]),
            weight=[0.1, 0.2, 0.3] * u.uS,
        )

        conns = projection.to_connections(edges, pool_size=2)

        self.assertEqual(len(conns), 1)
        self.assertEqual(conns[0].synapse, "ampa")
        np.testing.assert_array_equal(conns[0].pre_index, [0, 0, 1])
        np.testing.assert_array_equal(conns[0].post_index, [2, 2, 3])
        np.testing.assert_array_equal(conns[0].synapse_index, [0, 1, 1])
        np.testing.assert_allclose(conns[0].weight.to_decimal(u.uS), [0.1, 0.2, 0.3])

    def test_projection_and_contact_table_repr_are_compact(self) -> None:
        contacts = explicit_contacts(source_edge=[0, 0, 1], synapse_index=[0, 1, 1])(
            network.ProjectionEdgeContext(
                edge_index=np.asarray([0, 1], dtype=np.int32),
                edge_pre_index=np.asarray([0, 1], dtype=np.int32),
                edge_post_index=np.asarray([2, 3], dtype=np.int32),
                pre_size=2,
                post_size=4,
                pool_size=2,
                synapse="ampa",
            )
        )
        projection = Projection(
            name="E_to_I_ampa",
            edges="E_to_I",
            synapse="ampa",
            method=explicit_contacts(source_edge=[0], synapse_index=[0]),
            edge_index=[0],
            weight=0.1 * u.uS,
        )

        self.assertEqual(repr(contacts), "ContactTable(n_contact=3)")
        text = repr(projection)
        self.assertIn("Projection", text)
        self.assertIn("edges='E_to_I'", text)
        self.assertIn("edge_index=True", text)
        self.assertNotIn("<function", text)


class NetworkProjectionTest(unittest.TestCase):
    def test_projection_reuses_unused_edge_set_without_error(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edge_set(EdgeSet("unused", "E", "I", [0], [1]))

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertTrue(np.allclose(g, 0.0))

    def test_projection_delivers_weighted_payload_to_shared_synapse(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edge_set(EdgeSet("E_to_I", "E", "I", [0, 1], [1, 1]))
        net.add_projection(
            Projection(
                "E_to_I_exp",
                edges="E_to_I",
                synapse="exp",
                weight=[0.25, 0.75] * u.uS,
            )
        )

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 1.0, places=6)

    def test_add_projection_accepts_direct_arguments(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edges(
            name="E_to_I",
            pre="E",
            post="I",
            method=pairs([(0, 1)]),
        )

        projection = net.add_projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            weight=0.4 * u.uS,
        )

        self.assertIs(net.proj["E_to_I_exp"], projection)
        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[1, 1]), 0.4, places=6)

    def test_projection_per_edge_expands_contacts_over_synapse_pool(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [1, 1])
        projection = Projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            method=per_edge(number=2, replace=False, seed=1),
            weight=[0.1, 0.2] * u.uS,
        )

        conns = projection.to_connections(edges, pool_size=2)

        self.assertEqual(len(conns), 1)
        np.testing.assert_array_equal(conns[0].pre_index, [0, 0, 1, 1])
        np.testing.assert_array_equal(conns[0].post_index, [1, 1, 1, 1])
        np.testing.assert_allclose(conns[0].weight.to_decimal(u.uS), [0.1, 0.1, 0.2, 0.2])
        for start in (0, 2):
            self.assertEqual(set(conns[0].synapse_index[start : start + 2].tolist()), {0, 1})

    def test_projection_by_post_without_replacement_requires_enough_targets(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1, 0], [1, 1, 0])
        projection = Projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            method=by_post(replace=False, seed=1),
        )

        conns = projection.to_connections(edges, pool_size=2)

        post_one_targets = conns[0].synapse_index[conns[0].post_index == 1]
        self.assertEqual(set(post_one_targets.tolist()), {0, 1})

        too_small = Projection(
            name="too_small",
            edges="E_to_I",
            synapse="exp",
            method=by_post(number=2, replace=False, seed=1),
        )
        with self.assertRaisesRegex(ValueError, "total contacts per post"):
            too_small.to_connections(edges, pool_size=3)

    def test_projection_accepts_callable_number_and_weight_rules(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1, 2, 3], [0, 0, 1, 1])

        def number(ctx):
            return np.where(ctx.edge_pre_index < ctx.pre_size // 2, 1, 2)

        def weight(ctx):
            return np.where(ctx.edge_pre_index < ctx.pre_size // 2, 0.1, 0.3) * u.uS

        projection = Projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            method=per_edge(number=number, replace=True, seed=1),
            weight=weight,
        )

        conns = projection.to_connections(edges, pre_size=4, post_size=2, pool_size=3)

        self.assertEqual(len(conns), 1)
        np.testing.assert_array_equal(conns[0].pre_index, [0, 1, 2, 2, 3, 3])
        np.testing.assert_allclose(
            conns[0].weight.to_decimal(u.uS),
            [0.1, 0.1, 0.3, 0.3, 0.3, 0.3],
        )

    def test_projection_accepts_custom_contact_method(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [1, 1])

        def duplicate_first_edge(ctx):
            self.assertEqual(ctx.synapse, "exp")
            return ContactTable(source_edge=[0, 0, 1], synapse_index=[0, 1, 0])

        projection = Projection(
            name="custom",
            edges="E_to_I",
            synapse="exp",
            method=duplicate_first_edge,
            weight=[0.1, 0.2, 0.3] * u.uS,
        )

        conns = projection.to_connections(edges, pool_size=2)

        np.testing.assert_array_equal(conns[0].pre_index, [0, 0, 1])
        np.testing.assert_array_equal(conns[0].post_index, [1, 1, 1])
        np.testing.assert_array_equal(conns[0].synapse_index, [0, 1, 0])
        np.testing.assert_allclose(conns[0].weight.to_decimal(u.uS), [0.1, 0.2, 0.3])

    def test_network_delivers_to_selected_synapse_pool_indices(self) -> None:
        pre = make_spiking_cell()
        post = make_post_cell_with_synapse_pool()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edge_set(EdgeSet("E_to_I", "E", "I", [0, 1], [1, 1]))
        net.add_projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse="exp",
            method=per_edge(number=2, replace=False, seed=1),
            weight=[0.25, 0.75] * u.uS,
        )

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertIn("I", result.traces)
        layout = next(layout for layout, _ in post.runtime.iter_synapse_layouts())
        node = post.runtime.get_runtime_node(layout.id)
        g = np.asarray(node.g.value.to_decimal(u.uS))
        self.assertEqual(g.shape, (2, 2))
        np.testing.assert_allclose(g[1], [1.0, 1.0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
