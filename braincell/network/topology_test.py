import unittest

import brainunit as u
import numpy as np

import braincell
import braincell.network as network
from braincell import CVPerBranch, Cell, Morphology
from braincell.filter import at
from braincell.network import Connection
from braincell.network import EdgeSet, Projection
from braincell.network import explicit_contacts
from braincell.network import Network
from braincell.network import pairs


def _build_cell(size: int = 2) -> Cell:
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = Cell(
        Morphology.from_root(soma, name="soma"),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    return cell


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


class EdgeSetProjectionTest(unittest.TestCase):
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


class NetworkDisplayTest(unittest.TestCase):
    def test_network_add_edges_accepts_probability_and_callable_methods(self) -> None:
        net = Network(name="edge_method_demo")
        net.add_population("E", _build_cell(size=4))
        net.add_population("I", _build_cell(size=4))

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

    def test_network_repr_and_str_show_topology_summary(self) -> None:
        net = Network(name="demo")
        net.add_population("E", _build_cell(size=3))
        net.add_population("I", _build_cell(size=2))
        net.add_edges(name="E_to_I", pre="E", post="I", method=pairs([(0, 1)]))
        net.add_projection(name="E_to_I_exp", edges="E_to_I", synapse="exp")
        net.add_connection(Connection("E", "I", [1], [0], "exp"))

        compact = repr(net)
        detailed = str(net)

        self.assertEqual(
            compact,
            "Network(name='demo', populations=2, edge_sets=1, projections=1, connections=1)",
        )
        self.assertIn("populations:", detailed)
        self.assertIn("E: size=3", detailed)
        self.assertIn("edge_sets:", detailed)
        self.assertIn("E_to_I: E -> I, n_edge=1", detailed)
        self.assertIn("projections:", detailed)
        self.assertIn("E_to_I_exp: edges='E_to_I', synapse='exp'", detailed)
        self.assertIn("direct connections:", detailed)
        self.assertIn("[0]: E -> I, synapse='exp', n_contact=1", detailed)
        self.assertNotIn("array(", detailed)

    def test_population_repr_is_compact(self) -> None:
        population = net_population = Network().add_population("E", _build_cell(size=3))

        text = repr(population)

        self.assertIs(population, net_population)
        self.assertIn("Population", text)
        self.assertIn("size=3", text)
        self.assertIn("initialized=False", text)
        self.assertNotIn("Cell(root=", text)


if __name__ == "__main__":
    unittest.main()
