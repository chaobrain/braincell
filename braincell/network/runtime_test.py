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
    ContactTable,
    all_pairs,
    by_post,
    dense,
    per_edge,
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

    def test_lowering_quantizes_delay_with_ceil_floor_and_strict(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms)

        ceil_block = lower_connections(
            populations,
            (conn,),
            dt=0.1 * u.ms,
            delay_quantization="ceil",
        )[0]
        floor_block = lower_connections(
            populations,
            (conn,),
            dt=0.1 * u.ms,
            delay_quantization="floor",
        )[0]

        np.testing.assert_array_equal(ceil_block.delay_steps, [2])
        np.testing.assert_array_equal(floor_block.delay_steps, [1])
        strict_block = lower_connections(
            populations,
            (Connection("E", "I", [0], [1], "exp", delay=0.2 * u.ms),),
            dt=0.1 * u.ms,
            delay_quantization="strict",
        )[0]
        np.testing.assert_array_equal(strict_block.delay_steps, [2])
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            lower_connections(
                populations,
                (conn,),
                dt=0.1 * u.ms,
                delay_quantization="strict",
            )

    def test_lowering_zero_delay_is_next_step_for_all_quantization_modes(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp", delay=0.0 * u.ms)

        for mode in ("ceil", "floor", "strict"):
            with self.subTest(mode=mode):
                block = lower_connections(
                    populations,
                    (conn,),
                    dt=0.1 * u.ms,
                    delay_quantization=mode,
                )[0]
                np.testing.assert_array_equal(block.delay_steps, [1])

    def test_lowering_rejects_unknown_delay_quantization(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        pre.init_state()
        post.init_state()
        populations = {"E": Population("E", pre), "I": Population("I", post)}
        conn = Connection("E", "I", [0], [1], "exp")

        with self.assertRaisesRegex(ValueError, "delay_quantization"):
            lower_connections(
                populations,
                (conn,),
                dt=0.1 * u.ms,
                delay_quantization="nearest",
            )


class NetworkRuntimeTest(unittest.TestCase):
    def test_init_state_initializes_population_cells_and_is_idempotent(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        returned = net.add_population("E", pre)
        net.add_population("I", post)

        self.assertIs(returned.cell, pre)
        self.assertFalse(pre._initialized)
        self.assertFalse(post._initialized)

        self.assertIs(net.init_state(), net)
        self.assertTrue(pre._initialized)
        self.assertTrue(post._initialized)

        pre_runtime = pre.runtime
        post_runtime = post.runtime
        self.assertIs(net.init_state(), net)
        self.assertIs(pre.runtime, pre_runtime)
        self.assertIs(post.runtime, post_runtime)

    def test_reset_state_initializes_uninitialized_cells_and_preserves_topology(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        edges = net.add_edges(
            name="E_to_I",
            pre="E",
            post="I",
            method=pairs([(0, 1)]),
        )
        projection = net.add_projection(name="E_to_I_exp", edges="E_to_I", synapse="exp")
        connection = net.add_connection(Connection("E", "I", [1], [0], "exp"))

        self.assertFalse(pre._initialized)
        self.assertFalse(post._initialized)

        self.assertIs(net.reset_state(), net)

        self.assertTrue(pre._initialized)
        self.assertTrue(post._initialized)
        self.assertIs(net.populations["E"].cell, pre)
        self.assertIs(net.edge_sets["E_to_I"], edges)
        self.assertIs(net.projections["E_to_I_exp"], projection)
        self.assertIs(net.connections[0], connection)

    def test_reset_state_restarts_run_from_initial_cell_state(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        first = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)
        net.reset_state()
        second = net.run(dt=0.1 * u.ms, duration=0.3 * u.ms)

        np.testing.assert_allclose(
            np.asarray(first.traces["I"]["g"].to_decimal(u.uS)),
            np.asarray(second.traces["I"]["g"].to_decimal(u.uS)),
            rtol=1e-9,
            atol=1e-9,
        )
        np.testing.assert_array_equal(
            np.asarray(first.spikes["E"]),
            np.asarray(second.spikes["E"]),
        )

    def test_population_spike_reduces_multicompartment_spike_to_cell_level_events(self) -> None:
        cell = _spiking_cell(size=2)
        spike = np.asarray([[False, True, False], [False, False, False]])

        reduced = braincell.network.delivery.population_spike(spike)

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

    def test_spike_recording_population_returns_cell_level_spikes(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            spike_recording="population",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertEqual(result.spikes["E"].shape, (3, 2))
        self.assertGreater(int(np.asarray(result.spikes["E"]).sum()), 0)

    def test_spike_recording_none_omits_spike_traces_but_delivers_events(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            spike_recording="none",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)
        self.assertEqual(result.spikes, {})

    def test_spike_recording_rejects_unknown_value(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        with self.assertRaisesRegex(ValueError, "spike_recording"):
            net.run(
                dt=0.1 * u.ms,
                duration=0.1 * u.ms,
                spike_recording="cell",
            )

    def test_run_setup_cache_reuses_repeated_configuration(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))

        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(net._run_setup_cache), 1)
        first_setup = next(iter(net._run_setup_cache.values()))

        pre.reset_state()
        post.reset_state()
        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)

        self.assertEqual(len(net._run_setup_cache), 1)
        self.assertIs(next(iter(net._run_setup_cache.values())), first_setup)

    def test_run_setup_cache_is_cleared_on_topology_change(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(Connection("E", "I", [0], [1], "exp"))
        net.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(net._run_setup_cache), 1)

        net.add_connection(Connection("E", "I", [1], [0], "exp"))

        self.assertEqual(len(net._run_setup_cache), 0)

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

    def test_multi_step_delay_arrives_after_requested_steps(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection("E", "I", [0], [1], "exp", delay=0.2 * u.ms)
        )

        result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 0.0)
        self.assertGreater(float(g[2, 1]), 0.0)

    def test_non_integer_delay_uses_ceil_by_default(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms)
        )

        result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[1, 1]), 0.0)
        self.assertGreater(float(g[2, 1]), 0.0)

    def test_non_integer_delay_can_use_floor_quantization(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection("E", "I", [0], [1], "exp", delay=0.15 * u.ms)
        )

        result = net.run(
            dt=0.1 * u.ms,
            duration=0.3 * u.ms,
            delay_quantization="floor",
        )

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertGreater(float(g[1, 1]), 0.0)

    def test_per_edge_heterogeneous_delays_arrive_at_different_steps(self) -> None:
        pre = _spiking_cell(size=2)
        post = _post_cell(size=2)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)
        net.add_connection(
            Connection(
                "E",
                "I",
                [0, 1],
                [1, 1],
                "exp",
                weight=[0.25, 0.75] * u.uS,
                delay=[0.0, 0.2] * u.ms,
            )
        )

        result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms)

        g = np.asarray(result.traces["I"]["g"].to_decimal(u.uS))
        self.assertAlmostEqual(float(g[0, 1]), 0.0)
        self.assertAlmostEqual(float(g[1, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(g[2, 1]), 0.25, places=6)
        self.assertAlmostEqual(float(g[3, 1]), 1.0, places=6)

    def test_event_backend_auto_matches_scatter(self) -> None:
        def run(backend):
            pre = _spiking_cell(size=2)
            post = _post_cell(size=2)
            net = Network()
            net.add_population("E", pre)
            net.add_population("I", post)
            net.add_connection(
                Connection(
                    "E",
                    "I",
                    [0, 1],
                    [1, 1],
                    "exp",
                    weight=[0.25, 0.75] * u.uS,
                    delay=[0.0, 0.2] * u.ms,
                )
            )
            result = net.run(dt=0.1 * u.ms, duration=0.4 * u.ms, event_backend=backend)
            return np.asarray(result.traces["I"]["g"].to_decimal(u.uS))

        np.testing.assert_allclose(run("auto"), run("scatter"), rtol=1e-9, atol=1e-9)

    def test_event_backend_brainevent_jax_raw_matches_scatter_when_available(self) -> None:
        try:
            import brainevent
        except Exception:
            return
        if not hasattr(brainevent, "coomv"):
            return

        def run(backend, *, brainevent_backend="jax_raw"):
            pre = _spiking_cell(size=2)
            post = _post_cell(size=2)
            net = Network()
            net.add_population("E", pre)
            net.add_population("I", post)
            net.add_connection(
                Connection(
                    "E",
                    "I",
                    [0, 1],
                    [1, 1],
                    "exp",
                    weight=[0.25, 0.75] * u.uS,
                    delay=[0.0, 0.2] * u.ms,
                )
            )
            result = net.run(
                dt=0.1 * u.ms,
                duration=0.4 * u.ms,
                event_backend=backend,
                brainevent_backend=brainevent_backend,
            )
            return np.asarray(result.traces["I"]["g"].to_decimal(u.uS))

        np.testing.assert_allclose(
            run("brainevent", brainevent_backend="jax_raw"),
            run("scatter"),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_event_backend_rejects_unknown_value(self) -> None:
        pre = _spiking_cell()
        post = _post_cell()
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        with self.assertRaisesRegex(ValueError, "event_backend"):
            net.run(dt=0.1 * u.ms, duration=0.1 * u.ms, event_backend="dense")

    def test_event_backend_brainevent_requires_coomv(self) -> None:
        import braincell.network.delivery as delivery

        try:
            import brainevent
        except Exception:
            return
        if hasattr(brainevent, "coomv"):
            return

        with self.assertRaisesRegex(RuntimeError, "brainevent.coomv"):
            delivery.resolve_event_backend("brainevent")

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
        pre = _spiking_cell(size=3)
        post = _post_cell(size=3)
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
        pre = _spiking_cell(size=2)
        post = _post_cell(size=2)
        net = Network()
        net.add_population("E", pre)
        net.add_population("I", post)

        def out_of_range(*, n_pre: int, n_post: int):
            _ = (n_pre, n_post)
            return [0, 2], [0, 1]

        with self.assertRaisesRegex(IndexError, "pre_index out of range"):
            net.add_edges(name="bad", pre="E", post="I", method=out_of_range)

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

    def test_multiple_pre_populations_sum_arrivals_to_same_synapse_layout(self) -> None:
        pre_a = _spiking_cell(size=1)
        pre_b = _spiking_cell(size=1)
        post = _post_cell()
        net = Network()
        net.add_population("A", pre_a)
        net.add_population("B", pre_b)
        net.add_population("I", post)
        net.add_connection(Connection("A", "I", [0], [1], "exp", weight=0.25 * u.uS))
        net.add_connection(Connection("B", "I", [0], [1], "exp", weight=0.75 * u.uS))

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
            self.assertEqual(set(conns[0].synapse_index[start:start + 2].tolist()), {0, 1})

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
        pre = _spiking_cell()
        post = _post_cell_with_synapse_pool()
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
