import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.filter import AllRegion, RootLocation, at
from braincell.network import Network, Population
from braincell.network.lowering import resolve_source_cv


def _tree(two_branches=False):
    soma = braincell.Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morphology = braincell.Morphology.from_root(soma, name="soma")
    if two_branches:
        morphology.soma.dend = braincell.Branch.from_lengths(
            lengths=[100.0] * u.um,
            radii=[2.0, 1.0] * u.um,
            type="dendrite",
        )
    return morphology


def _step_up(cell):
    cell.V.value = cell.V.value + 40.0 * u.mV


def _step_down(cell):
    cell.V.value = cell.V.value - 1.0 * u.mV


def _pre(size=2):
    return braincell.Cell(
        _tree(),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(size,),
        V_init=-10.0 * u.mV,
        V_th=0.0 * u.mV,
        solver=_step_up,
    )


def _post(size=2):
    cell = braincell.Cell(
        _tree(),
        cv_policy=braincell.CVPerBranch(),
        pop_size=(size,),
        V_init=-65.0 * u.mV,
        solver=_step_down,
    )
    cell.place(
        at("soma", 0.5),
        braincell.mech.SynapseSpec("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV),
    )
    cell.soma.record("g", braincell.observe.synapse(name="exp").state("g"))
    return cell


def _network(*, delay=0.0 * u.ms, weight=0.2 * u.uS):
    network = Network("runtime")
    pre = network.add_population("pre", _pre())
    post = network.add_population("post", _post())
    network.connect(
        "drive",
        source=pre.event_outputs["spike"],
        synapse=post.synapses["exp"],
        weight=weight,
        delay=delay,
    )
    return network


class PopulationTest(unittest.TestCase):
    def test_population_requires_one_dimensional_cell_population(self) -> None:
        self.assertEqual(Population("pre", _pre(3)).size, 3)
        self.assertEqual(
            Population("scalar", braincell.Cell(_tree(), cv_policy=braincell.CVPerBranch())).size,
            1,
        )
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            Population(
                "grid",
                braincell.Cell(_tree(), cv_policy=braincell.CVPerBranch(), pop_size=(2, 2)),
            )

    def test_population_forwards_cell_event_outputs(self) -> None:
        population = Population("pre", _pre(3))
        self.assertIs(population.event_outputs["spike"].owner, population.cell.event_outputs["spike"].owner)


class LoweringTest(unittest.TestCase):
    def test_source_location_resolves_to_canonical_cv(self) -> None:
        cell = braincell.Cell(
            _tree(two_branches=True),
            cv_policy=braincell.CVPerBranch(),
            pop_size=(2,),
        )
        self.assertEqual(resolve_source_cv(cell, RootLocation(0.5)), 0)
        self.assertEqual(resolve_source_cv(cell, at("dend", 0.5)), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            resolve_source_cv(cell, at("soma", 0.5) | at("dend", 0.5))


class NetworkRuntimeTest(unittest.TestCase):
    def test_cell_with_trainable_bindings_is_rejected_until_network_aggregation_exists(self) -> None:
        cell = _pre()
        cell.paint(AllRegion(), braincell.mech.Channel("IL", name="leak"))
        cell.channels["leak"].trainable(g_max=braincell.trainable.scale(name="factor"))
        with self.assertRaisesRegex(NotImplementedError, "Network aggregation"):
            Network("network").add_population("cell", cell)

    def test_cell_has_one_network_execution_owner(self) -> None:
        cell = _pre()
        first = Network("first")
        second = Network("second")
        first.add_population("pre", cell)
        with self.assertRaisesRegex(ValueError, "more than one Network population"):
            first.add_population("alias", cell)
        with self.assertRaisesRegex(RuntimeError, "already belongs"):
            second.add_population("pre", cell)

    def test_initialization_freezes_topology(self) -> None:
        network = _network()
        network.init_state()
        self.assertIs(network.init_state(), network)
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.add_population("extra", braincell.NetStim())
        with self.assertRaisesRegex(RuntimeError, "after Network initialization"):
            network.connect(
                "late",
                source=network.populations["pre"],
                synapse=network.populations["post"].synapses["exp"],
            )

    def test_batch_size_is_explicitly_unsupported(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "batch execution"):
            _network().init_state(batch_size=2)

    def test_zero_delay_delivers_on_source_boundary(self) -> None:
        result = _network().run(dt=0.1 * u.ms, duration=0.4 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        first_nonzero = int(np.flatnonzero(conductance[:, 0] > 0.0)[0])
        self.assertEqual(first_nonzero, 1)

    def test_heterogeneous_delay_routes_rows_independently(self) -> None:
        network = _network(delay=[0.0, 0.3] * u.ms)
        result = network.run(dt=0.1 * u.ms, duration=0.6 * u.ms)
        conductance = np.asarray(result.samples["post"]["g"].values.to_decimal(u.uS))
        self.assertEqual(int(np.flatnonzero(conductance[:, 0] > 0.0)[0]), 1)
        self.assertEqual(int(np.flatnonzero(conductance[:, 1] > 0.0)[0]), 4)

    def test_split_run_preserves_events_in_flight(self) -> None:
        continuous = _network(delay=0.3 * u.ms).run(dt=0.1 * u.ms, duration=0.7 * u.ms)
        split_network = _network(delay=0.3 * u.ms)
        split = braincell.NetworkResult.concat(
            (
                split_network.run(dt=0.1 * u.ms, duration=0.2 * u.ms),
                split_network.run(dt=0.1 * u.ms, duration=0.5 * u.ms),
            )
        )
        np.testing.assert_allclose(
            split.samples["post"]["g"].values.to_decimal(u.uS),
            continuous.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_reset_state_restarts_and_discards_events_in_flight(self) -> None:
        network = _network(delay=0.3 * u.ms)
        first = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        network.reset_state()
        second = network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        np.testing.assert_allclose(
            first.samples["post"]["g"].values.to_decimal(u.uS),
            second.samples["post"]["g"].values.to_decimal(u.uS),
        )

    def test_event_backend_auto_matches_scatter(self) -> None:
        auto = _network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="auto",
        )
        scatter = _network(delay=[0.0, 0.2] * u.ms).run(
            dt=0.1 * u.ms,
            duration=0.5 * u.ms,
            event_backend="scatter",
        )
        np.testing.assert_allclose(
            auto.samples["post"]["g"].values.to_decimal(u.uS),
            scatter.samples["post"]["g"].values.to_decimal(u.uS),
            rtol=1e-6,
        )

    def test_run_setup_and_compiled_loop_are_reused(self) -> None:
        network = _network()
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        setup_count = len(network._run_setup_cache)
        loop_count = len(network._network_run_loop_cache)
        network.run(dt=0.1 * u.ms, duration=0.2 * u.ms)
        self.assertEqual(len(network._run_setup_cache), setup_count)
        self.assertEqual(len(network._network_run_loop_cache), loop_count)


if __name__ == "__main__":
    unittest.main()
