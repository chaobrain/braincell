import unittest

import brainunit as u
import numpy as np

from braincell.network import Connection
from braincell.network import EdgeSet, Projection
from braincell.network import connectors


class ConnectionTest(unittest.TestCase):
    def test_connection_normalizes_index_arrays(self) -> None:
        conn = Connection(
            pre_population="E",
            post_population="I",
            pre_index=[0, 2],
            post_index=[1, 0],
            synapse="ampa",
        )

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

    def test_pairs_connector_preserves_order(self) -> None:
        conn = connectors.pairs(
            "E",
            "I",
            [(2, 1), (0, 3)],
            synapse="ampa",
            weight=0.5 * u.uS,
            delay=0.0 * u.ms,
        )

        np.testing.assert_array_equal(conn.pre_index, [2, 0])
        np.testing.assert_array_equal(conn.post_index, [1, 3])
        np.testing.assert_allclose(conn.weight.to_decimal(u.uS), 0.5)

    def test_probability_connector_is_seeded(self) -> None:
        a = connectors.probability(
            "E",
            "E",
            n_pre=4,
            n_post=4,
            p=0.5,
            synapse="ampa",
            seed=7,
            allow_self=False,
        )
        b = connectors.probability(
            "E",
            "E",
            n_pre=4,
            n_post=4,
            p=0.5,
            synapse="ampa",
            seed=7,
            allow_self=False,
        )

        np.testing.assert_array_equal(a.pre_index, b.pre_index)
        np.testing.assert_array_equal(a.post_index, b.post_index)
        self.assertFalse(np.any(a.pre_index == a.post_index))


class EdgeSetProjectionTest(unittest.TestCase):
    def test_edge_set_stores_cell_level_edges_without_synapse(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [2, 3])

        self.assertEqual(edges.n_edge, 2)
        np.testing.assert_array_equal(edges.pre_index, [0, 1])
        np.testing.assert_array_equal(edges.post_index, [2, 3])

    def test_projection_reuses_edges_and_groups_by_synapse(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1], [2, 3])
        projection = Projection(
            name="E_to_I_ampa",
            edges="E_to_I",
            edge_index=[0, 0, 1],
            synapse=["ampa_0", "ampa_1", "ampa_1"],
            weight=[0.1, 0.2, 0.3] * u.uS,
        )

        conns = projection.to_connections(edges)

        self.assertEqual(len(conns), 2)
        self.assertEqual(conns[0].synapse, "ampa_0")
        np.testing.assert_array_equal(conns[0].pre_index, [0])
        np.testing.assert_array_equal(conns[0].post_index, [2])
        np.testing.assert_allclose(conns[0].weight.to_decimal(u.uS), [0.1])
        self.assertEqual(conns[1].synapse, "ampa_1")
        np.testing.assert_array_equal(conns[1].pre_index, [0, 1])
        np.testing.assert_array_equal(conns[1].post_index, [2, 3])
        np.testing.assert_allclose(conns[1].weight.to_decimal(u.uS), [0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
