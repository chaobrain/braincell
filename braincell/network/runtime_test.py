import unittest

import brainunit as u
import numpy as np

import braincell
from braincell import CVPerBranch, Cell, Morphology
from braincell.filter import RootLocation, at
from braincell.network import (
    Connection,
    EdgeSet,
    Network,
    Population,
    Projection,
    all_to_all,
    pairs,
    probability,
    lower_connections,
)


def _build_tree() -> Morphology:
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    return Morphology.from_root(soma, name="soma")


def _build_two_point_tree() -> Morphology:
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    dend = braincell.Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.dend = dend
    return tree


def _step_up_solver(cell):
    cell.V.value = cell.V.value + 40.0 * u.mV


def _step_down_solver(cell):
    cell.V.value = cell.V.value - 1.0 * u.mV


def _spiking_cell(size: int = 2) -> Cell:
    cell = Cell(
        _build_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-10.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=_step_up_solver,
    )
    cell.place(at("soma", 0.5), braincell.mech.StateProbe(name="v", field="v"))
    return cell


def _post_cell(size: int = 2) -> Cell:
    cell = Cell(
        _build_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=_step_down_solver,
    )
    cell.place(at("soma", 0.5), braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"))
    cell.place(
        at("soma", 0.5),
        braincell.mech.Synapse(
            "ExpSyn",
            tau=2.0 * u.ms,
            e=0.0 * u.mV,
            weight=1.0 * u.uS,
            name="exp",
        ),
    )
    return cell


def _post_cell_with_synapse_pool(size: int = 2) -> Cell:
    cell = Cell(
        _build_two_point_tree(),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=_step_down_solver,
    )
    cell.place(
        at("soma", 0.5) | at(1, 0.5),
        braincell.mech.Synapse(
            "ExpSyn",
            tau=2.0 * u.ms,
            e=0.0 * u.mV,
            weight=1.0 * u.uS,
            name="exp",
        ),
    )
    return cell


class PopulationTest(unittest.TestCase):
    def test_population_accepts_one_dimensional_cell_pop_size(self) -> None:
        pop = Population("E", _spiking_cell(size=3))
        self.assertEqual(pop.size, 3)

    def test_population_rejects_missing_pop_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            Population("one", Cell(_build_tree(), cv_policy=CVPerBranch()))

    def test_population_rejects_multi_dimensional_pop_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            Population("grid", Cell(_build_tree(), cv_policy=CVPerBranch(), pop_size=(2, 2)))


class LoweringTest(unittest.TestCase):
    def test_lowering_validates_unknown_population(self) -> None:
        post = _post_cell()
        post.init_state()
        populations = {"I": Population("I", post)}
        conn = Connection("E", "I", [0], [0], "exp")

        with self.assertRaisesRegex(KeyError, "Unknown pre_population"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_validates_index_range(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [2], [0], "exp")

        with self.assertRaisesRegex(IndexError, "pre_index out of range"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_requires_named_synapse_on_post_cell(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [0], "missing")

        with self.assertRaisesRegex(KeyError, "no placed synapse"):
            lower_connections(populations, (conn,), dt=0.1 * u.ms)

    def test_lowering_converts_zero_delay_to_next_step(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.0 * u.ms)

        block = lower_connections(populations, (conn,), dt=0.1 * u.ms)[0]

        np.testing.assert_array_equal(block.delay_steps, [1])


class NetworkRuntimeTest(unittest.TestCase):
    def test_population_spike_reduces_multicompartment_spike_to_cell_level_events(self) -> None:
        cell = _spiking_cell(size=2)
        spike = np.asarray([[False, True, False], [False, False, False]])

        reduced = braincell.network.runtime._population_spike(spike)

        np.testing.assert_array_equal(np.asarray(reduced), [True, False])

    def test_cross_population_delivery_arrives_on_next_step(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertEqual(g.shape, (3, 2))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 0]), 0.0)
        self.assertIn("E", result.spikes)
        self.assertEqual(result.spikes["E"].shape, (3, 2, 1))

    def test_same_population_recurrent_delivery_uses_next_step(self) -> None:
        cell = _post_cell(size=2)
        cell.solver = _step_up_solver
        cell.V_init = -10.0 * u.mV
        net = Network()
        net.add_population("E", cell)
        net.add_connection(Connection("E", "E", [0], [1], "exp"))

        result = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        g = np.asarray(result.traces["E"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertGreater(float(g[1, 1]), 0.0)

    def test_multi_step_delay_reports_scan_limitation(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection("E", "I", [0], [1], "exp", delay=0.2 * u.ms)
        )

        with self.assertRaisesRegex(NotImplementedError, "Multi-step delays"):
            net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

    def test_projection_reuses_unused_edge_set_without_error(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edge_set(EdgeSet("unused", "E", "I", [0], [1]))

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertTrue(np.allclose(g, 0.0))

    def test_add_edges_and_project_store_named_objects(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network(name="demo")
        net.add_population("E", pre)
        net.add_population("I", post)

        edges = net.add_edges(
            name="E_to_I",
            pre="E",
            post="I",
            method=all_to_all(pre_indices=[0], post_indices=[1]),
        )
        projection = net.project(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse_pool="exp",
            weight=0.5 * u.uS,
        )

        self.assertEqual(net.name, "demo")
        self.assertIs(net.edge_sets["E_to_I"], edges)
        self.assertIs(net.proj["E_to_I_exp"], projection)
        self.assertIs(net.projections["E_to_I_exp"], projection)

        result = net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[1, 1]), 0.5, places=6)

    def test_add_edges_pairs_and_probability_builders(self) -> None:
        pre = _spiking_cell(size=3)
        post = _post_cell(size=3)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        explicit = net.add_edges(
            name="explicit",
            pre="E",
            post="I",
            method=pairs([(2, 1), (0, 2)]),
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
        np.testing.assert_array_equal(sampled_a.pre_index, sampled_b.pre_index)
        np.testing.assert_array_equal(sampled_a.post_index, sampled_b.post_index)

    def test_projection_delivers_weighted_payload_to_shared_synapse(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
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
        pre = _spiking_cell()
        post = _post_cell()
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
            synapse_pool="exp",
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
            synapse_pool="exp",
            number=2,
            replace=False,
            seed=1,
            weight=[0.1, 0.2] * u.uS,
        )

        conns = projection.to_connections(edges, pool_size=2)

        self.assertEqual(len(conns), 1)
        np.testing.assert_array_equal(conns[0].pre_index, [0, 0, 1, 1])
        np.testing.assert_array_equal(conns[0].post_index, [1, 1, 1, 1])
        np.testing.assert_allclose(conns[0].weight.to_decimal(u.uS), [0.1, 0.1, 0.2, 0.2])
        for start in (0, 2):
            self.assertEqual(set(conns[0].synapse_index[start:start + 2].tolist()), {0, 1})

    def test_projection_by_post_without_replacement_requires_enough_targets(self) -> None:
        edges = EdgeSet("E_to_I", "E", "I", [0, 1, 0], [1, 1, 0])
        projection = Projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse_pool="exp",
            target_policy="by_post",
            replace=False,
            seed=1,
        )

        conns = projection.to_connections(edges, pool_size=2)

        post_one_targets = conns[0].synapse_index[conns[0].post_index == 1]
        self.assertEqual(set(post_one_targets.tolist()), {0, 1})

        too_small = Projection(
            name="too_small",
            edges="E_to_I",
            synapse_pool="exp",
            target_policy="by_post",
            number=2,
            replace=False,
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "incoming_edge_count"):
            too_small.to_connections(edges, pool_size=3)

    def test_network_delivers_to_selected_synapse_pool_indices(self) -> None:
        pre = _spiking_cell()
        post = _post_cell_with_synapse_pool()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_edge_set(EdgeSet("E_to_I", "E", "I", [0, 1], [1, 1]))
        net.add_projection(
            name="E_to_I_exp",
            edges="E_to_I",
            synapse_pool="exp",
            target_policy="per_edge",
            number=2,
            replace=False,
            seed=1,
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
