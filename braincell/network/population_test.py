import unittest
from types import MappingProxyType

import brainunit as u
import numpy as np

import braincell
from braincell._multi_compartment.selection_test import _cell
from braincell.filter import at


class PopulationTest(unittest.TestCase):
    def test_cell_and_source_models_share_one_population_api(self) -> None:
        network = braincell.network.Network(seed=12)
        cells = network.add_population("pc", _cell(2), layer="P")
        stimuli = network.add_population(
            "stim",
            braincell.NetStim(size=2, start=[1.0, 2.0] * u.ms),
            cohort=np.asarray([3, 4]),
        )

        self.assertEqual(cells.kind, "cell")
        self.assertEqual(stimuli.kind, "netstim")
        self.assertIs(cells.model, cells.cell)
        self.assertEqual(tuple(cells.event_outputs), ("spike",))
        self.assertEqual(tuple(stimuli.event_outputs), ("spike",))
        self.assertEqual(tuple(cells.metadata), ("layer",))
        np.testing.assert_array_equal(cells.metadata["layer"], ["P", "P"])
        np.testing.assert_array_equal(cells.layer, ["P", "P"])
        self.assertFalse(hasattr(cells.model, "layer"))
        np.testing.assert_array_equal(stimuli["cohort"], [3, 4])
        self.assertIsInstance(cells.metadata, MappingProxyType)
        self.assertFalse(hasattr(cells, "sources"))
        self.assertFalse(hasattr(cells, "fields"))

    def test_cell_population_forwards_synapses_and_connections(self) -> None:
        network = braincell.Network("forwarding")
        source = network.add_population("stim", braincell.NetStim(size=2))
        cell = _cell(2)
        cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        target = network.add_population("post", cell)

        connection = network.connect(
            "drive",
            source=source,
            synapse=target.synapses["ampa"],
            weight=0.1 * u.uS,
        )

        self.assertEqual(len(target.synapses), 2)
        np.testing.assert_array_equal(target.connections.id, connection.id)
        self.assertIs(network.connections["post"].cell, cell)
        np.testing.assert_array_equal(network.connections["post", "drive"].id, connection.id)

    def test_network_summary_counts_named_connections_and_rows(self) -> None:
        network = braincell.Network("recording_demo")
        source = network.add_population("stim", braincell.NetStim(size=4))
        cell = _cell(4)
        cell[0:2].place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="fast"))
        cell[2:4].place(at("soma", 0.5), braincell.mech.SynapseSpec("Exp2Syn", name="slow"))
        target = network.add_population("post", cell)
        network.connect("stim_fast", source=source.event_outputs["spike"][0:2], synapse=target.synapses["fast"])
        network.connect("stim_slow", source=source.event_outputs["spike"][2:4], synapse=target.synapses["slow"])

        self.assertEqual(
            repr(network),
            "Network(name='recording_demo', populations=2, connections=2, rows=4)",
        )
        self.assertEqual(len(network.connections), 2)
        self.assertEqual(network.connections.n_rows, 4)
        self.assertIn("stim_fast: NetStim(stim) -> ExpSyn(fast), rows=2", str(network))

    def test_network_connect_can_atomically_place_new_synapses(self) -> None:
        network = braincell.Network("quick")
        source = network.add_population("stim", braincell.NetStim(size=2))
        target = network.add_population("post", _cell(2))

        connection = network.connect(
            "new_ampa",
            source=source,
            target=target,
            locations=at("soma", 0.5),
            synapse=braincell.mech.SynapseSpec("ExpSyn", name="ampa", tau=2.0 * u.ms),
            weight=[0.1, 0.2] * u.uS,
        )

        self.assertEqual(len(connection), 2)
        np.testing.assert_array_equal(connection.synapse.id, target.synapses["ampa"].id)

        bad_network = braincell.Network("rollback")
        bad_source = bad_network.add_population("stim", braincell.NetStim(size=2))
        bad_target = bad_network.add_population("post", _cell(3))
        with self.assertRaisesRegex(ValueError, "duplicate-preserving"):
            bad_network.connect(
                "bad",
                source=bad_source,
                target=bad_target,
                locations=at("soma", 0.5),
                synapse=braincell.mech.SynapseSpec("ExpSyn", name="ampa"),
            )
        self.assertEqual(len(bad_target.synapses), 0)
        self.assertEqual(len(bad_target.connections), 0)

    def test_network_connect_requires_registered_owners(self) -> None:
        network = braincell.Network()
        target_cell = _cell(1)
        target_cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        target = network.add_population("post", target_cell)

        with self.assertRaisesRegex(RuntimeError, "source owner is not registered"):
            network.connect(
                "outside",
                source=braincell.NetStim(),
                synapse=target.synapses["ampa"],
            )

    def test_custom_metadata_cannot_shadow_reserved_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved names"):
            braincell.network.Population("pc", _cell(1), size=3)
        population = braincell.network.Population("pc", _cell(2))
        with self.assertRaisesRegex(ValueError, "leading dimension"):
            population.set(layer=np.asarray([1, 2, 3]))
        with self.assertRaises(TypeError):
            population.metadata["layer"] = np.asarray([1, 2])

    def test_provider_is_eager_and_atomic(self) -> None:
        calls = []
        network = braincell.network.Network()

        def provider():
            calls.append(1)
            return braincell.NetStim(size=2)

        population = network.add_population("stim", provider)
        self.assertEqual(calls, [1])
        self.assertEqual(population.size, 2)

    def test_implicit_netstim_seed_depends_on_network_and_name_not_order(self) -> None:
        def schedules(order):
            network = braincell.network.Network(seed=7)
            result = {}
            for name in order:
                source = braincell.NetStim(size=2, number=4, noise=1.0)
                result[name] = network.add_population(name, source).model
            return {name: source.event_times.to_decimal(u.ms) for name, source in result.items()}

        first = schedules(("a", "b"))
        second = schedules(("b", "a"))
        np.testing.assert_allclose(first["a"], second["a"])
        np.testing.assert_allclose(first["b"], second["b"])
        self.assertFalse(np.allclose(first["a"], first["b"]))

    def test_explicit_netstim_seed_overrides_network_seed(self) -> None:
        left = braincell.network.Network(seed=1).add_population("stim", braincell.NetStim(number=4, noise=1.0, seed=9))
        right = braincell.network.Network(seed=999).add_population(
            "other", braincell.NetStim(number=4, noise=1.0, seed=9)
        )
        np.testing.assert_allclose(
            left.model.event_times.to_decimal(u.ms),
            right.model.event_times.to_decimal(u.ms),
        )

    def test_population_is_single_port_connect_source_and_results_include_events(self) -> None:
        network = braincell.network.Network(seed=3)
        source = network.add_population(
            "stim",
            braincell.NetStim(size=2, start=[0.05, 0.15] * u.ms, number=1),
        )
        cell = _cell(2)
        cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        braincell.connect("drive", source=source, synapse=cell.synapses["ampa"], weight=0.1 * u.uS)
        cell[0].soma.record("v", braincell.observe.state("v"), period=0.1 * u.ms)
        network.add_population("post", cell)

        result = network.run(dt=0.05 * u.ms, duration=0.2 * u.ms)

        self.assertEqual(result.samples["post"]["v"].values.shape, (2, 1))
        np.testing.assert_allclose(result.events["stim"]["spike"].time.to_decimal(u.ms), [0.05, 0.15])
        np.testing.assert_array_equal(result.events["stim"]["spike"].source_id, [0, 1])

    def test_network_result_concat_matches_split_recording(self) -> None:
        cell = _cell(1)
        cell.soma.record("v", braincell.observe.state("v"))
        network = braincell.network.Network()
        network.add_population("cell", cell)
        parts = (
            network.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
            network.run(dt=0.05 * u.ms, duration=0.1 * u.ms),
        )

        joined = braincell.network.NetworkResult.concat(parts)

        np.testing.assert_allclose(joined.time.to_decimal(u.ms), [0.0, 0.05, 0.1, 0.15])
        self.assertEqual(joined.samples["cell"]["v"].values.shape[0], 4)

    def test_network_rejects_unregistered_direct_source(self) -> None:
        cell = _cell(1)
        cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        braincell.connect(
            "drive",
            source=braincell.NetStim(),
            synapse=cell.synapses["ampa"],
            weight=0.1 * u.uS,
        )
        network = braincell.network.Network()
        network.add_population("post", cell)

        with self.assertRaisesRegex(RuntimeError, "register it"):
            network.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

    def test_cell_population_manual_event_output_is_reported_without_connection(self) -> None:
        cell = _cell(2)
        network = braincell.network.Network()
        population = network.add_population("cell", cell)
        detector = braincell.VoltageCrossingSource(
            cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
        )
        registered = population.register_event_output(detector, name="local_spike")

        result = network.run(dt=0.05 * u.ms, duration=0.1 * u.ms)

        self.assertIs(registered.owner, detector)
        self.assertEqual(tuple(population.event_outputs), ("spike", "local_spike"))
        self.assertIn("local_spike", result.events["cell"])

    def test_multilocation_event_output_reports_readonly_endpoint_mapping(self) -> None:
        cell = _cell(2)
        network = braincell.network.Network()
        population = network.add_population("cell", cell)
        detector = braincell.VoltageCrossingSource(
            cell,
            location=cell.cv_midpoints,
            name="all_cv_spikes",
        )
        population.register_event_output(detector)

        result = network.run(dt=0.05 * u.ms, duration=0.1 * u.ms)
        metadata = result.events["cell"]["all_cv_spikes"].metadata

        np.testing.assert_array_equal(metadata["population_index"], detector.population_index)
        np.testing.assert_array_equal(metadata["location_index"], detector.location_index)
        np.testing.assert_array_equal(metadata["cv_id"], detector.cv_id)
        self.assertFalse(metadata["population_index"].flags.writeable)
        self.assertFalse(metadata["location_index"].flags.writeable)
        self.assertFalse(metadata["cv_id"].flags.writeable)

    def test_network_connect_auto_registers_full_named_event_source_owner(self) -> None:
        pre_cell = _cell(2)
        post_cell = _cell(2)
        post_cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        network = braincell.Network("auto_output")
        pre = network.add_population("pre", pre_cell)
        post = network.add_population("post", post_cell)
        detector = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
            name="dendritic_spike",
        )

        network.connect(
            "first",
            source=detector[0],
            synapse=post.synapses["ampa"][0],
            weight=0.1 * u.uS,
        )
        network.connect(
            "second",
            source=detector[1],
            synapse=post.synapses["ampa"][1],
            weight=0.1 * u.uS,
        )

        self.assertEqual(tuple(pre.event_outputs), ("spike", "dendritic_spike"))
        self.assertIs(pre.event_outputs["dendritic_spike"].owner, detector)
        self.assertEqual(len(pre.event_outputs["dendritic_spike"]), 2)
        result = network.run(dt=0.05 * u.ms, duration=0.1 * u.ms)
        self.assertIn("dendritic_spike", result.events["pre"])

    def test_network_connect_requires_name_for_additional_event_source(self) -> None:
        pre_cell = _cell(1)
        post_cell = _cell(1)
        post_cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        network = braincell.Network("unnamed_output")
        pre = network.add_population("pre", pre_cell)
        post = network.add_population("post", post_cell)
        detector = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
        )

        with self.assertRaisesRegex(ValueError, "non-empty source name"):
            network.connect("invalid", source=detector, synapse=post.synapses["ampa"])

        self.assertEqual(tuple(pre.event_outputs), ("spike",))
        self.assertEqual(len(post.connections), 0)

    def test_network_connect_rejects_event_output_name_collision_before_mutation(self) -> None:
        pre_cell = _cell(1)
        post_cell = _cell(1)
        post_cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        network = braincell.Network("output_collision")
        pre = network.add_population("pre", pre_cell)
        post = network.add_population("post", post_cell)
        first = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
            name="local_spike",
        )
        second = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-50.0 * u.mV,
            name="local_spike",
        )
        pre.register_event_output(first)

        with self.assertRaisesRegex(ValueError, "different event output"):
            network.connect("invalid", source=second, synapse=post.synapses["ampa"])

        self.assertIs(pre.event_outputs["local_spike"].owner, first)
        self.assertEqual(len(post.connections), 0)

    def test_failed_existing_synapse_connect_does_not_register_event_output(self) -> None:
        pre_cell = _cell(2)
        post_cell = _cell(3)
        post_cell.place(at("soma", 0.5), braincell.mech.SynapseSpec("ExpSyn", name="ampa"))
        network = braincell.Network("existing_output_rollback")
        pre = network.add_population("pre", pre_cell)
        post = network.add_population("post", post_cell)
        detector = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
            name="local_spike",
        )

        with self.assertRaisesRegex(ValueError, "equal lengths"):
            network.connect("invalid", source=detector, synapse=post.synapses["ampa"])

        self.assertEqual(tuple(pre.event_outputs), ("spike",))
        self.assertEqual(len(post.connections), 0)

    def test_failed_place_and_connect_does_not_register_event_output(self) -> None:
        pre_cell = _cell(2)
        post_cell = _cell(3)
        network = braincell.Network("output_rollback")
        pre = network.add_population("pre", pre_cell)
        post = network.add_population("post", post_cell)
        detector = braincell.VoltageCrossingSource(
            pre_cell,
            location=at("soma", 0.5),
            threshold=-60.0 * u.mV,
            name="local_spike",
        )

        with self.assertRaisesRegex(ValueError, "duplicate-preserving"):
            network.connect(
                "invalid",
                source=detector,
                target=post,
                locations=at("soma", 0.5),
                synapse=braincell.mech.SynapseSpec("ExpSyn", name="ampa"),
            )

        self.assertEqual(tuple(pre.event_outputs), ("spike",))
        self.assertEqual(len(post.synapses), 0)
        self.assertEqual(len(post.connections), 0)

    def test_scheduled_only_network_advances_time_and_reports_event_sequence(self) -> None:
        sequence = braincell.EventSequence(
            size=2,
            events=braincell.EventTable(
                source_index=np.asarray([1, 0, 1]),
                time=np.asarray([0.15, 0.05, 0.25]) * u.ms,
            ),
        )
        network = braincell.network.Network()
        network.add_population("input", sequence)

        first = network.run(dt=0.05 * u.ms, duration=0.2 * u.ms)
        second = network.run(dt=0.05 * u.ms, duration=0.1 * u.ms)
        joined = braincell.NetworkResult.concat((first, second))

        np.testing.assert_allclose(first.events["input"]["event"].time.to_decimal(u.ms), [0.05, 0.15])
        np.testing.assert_array_equal(first.events["input"]["event"].source_id, [0, 1])
        np.testing.assert_allclose(second.events["input"]["event"].time.to_decimal(u.ms), [0.25])
        np.testing.assert_allclose(
            joined.events["input"]["event"].time.to_decimal(u.ms),
            [0.05, 0.15, 0.25],
        )


if __name__ == "__main__":
    unittest.main()
