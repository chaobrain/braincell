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

import unittest

import brainunit as u
import numpy as np

import braincell
from braincell import Branch, Cell, CVPerBranch, Morphology, NetStim, Network, connect
from braincell.filter import at


def _population(size=3, *, name="post"):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    cell = Cell(
        Morphology.from_root(soma, name="soma"),
        cv_policy=CVPerBranch(),
        pop_size=(size,),
        name=name,
    )
    exp = braincell.mech.SynapseSpec("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(at("soma", 0.5), exp)
    return cell, exp


class ConnectionSamplingTest(unittest.TestCase):
    def test_independent_has_fixed_rows_and_is_reproducible(self):
        first_cell, first_exp = _population()
        second_cell, second_exp = _population()
        source = NetStim(size=5, start=1.0 * u.ms)
        pairing = braincell.network.connection.independent(20, seed=7)

        first = connect("first", source=source, synapse=first_cell.synapses[first_exp], pairing=pairing)
        second = connect("second", source=source, synapse=second_cell.synapses[second_exp], pairing=pairing)

        self.assertEqual(len(first), 20)
        np.testing.assert_array_equal(first.source_index, second.source_index)
        np.testing.assert_array_equal(first.target_index, second.target_index)

    def test_candidate_views_must_have_unique_ids(self):
        cell, exp = _population()
        source = NetStim(size=2, start=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "source view must contain unique IDs"):
            connect(
                "duplicate_source",
                source=source[[0, 0]],
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.independent(2),
            )
        with self.assertRaisesRegex(ValueError, "synapse view must contain unique IDs"):
            connect(
                "duplicate_synapse",
                source=source,
                synapse=cell.synapses[exp][[0, 0]],
                pairing=braincell.network.connection.independent(2),
            )

    def test_source_first_score_is_conditional(self):
        cell, exp = _population(3)
        source = NetStim(size=5, start=1.0 * u.ms)
        seen_shapes = []

        def score(ctx):
            seen_shapes.append((ctx.source.id.shape, ctx.synapse.population_index.shape))
            return ctx.synapse.population_index == (ctx.source.id % 3)

        result = connect(
            "conditional",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.source_first(30, synapse_score=score, seed=3),
        )

        np.testing.assert_array_equal(result.synapse.population_index, result.source_index % 3)
        self.assertEqual(seen_shapes[0][1], (1, 3))
        self.assertEqual(seen_shapes[0][0][1], 1)

    def test_synapse_first_score_is_conditional(self):
        cell, exp = _population(3)
        source = NetStim(size=3, start=1.0 * u.ms)

        def score(ctx):
            return ctx.source.id == ctx.synapse.population_index

        result = connect(
            "reverse_conditional",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.synapse_first(20, source_score=score, seed=4),
        )
        np.testing.assert_array_equal(result.source_index, result.synapse.population_index)

    def test_by_source_uses_exact_degree_counts(self):
        cell, exp = _population(4)
        source = NetStim(size=4, start=1.0 * u.ms)
        result = connect(
            "out_degree",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.by_source([0, 2, 1, 3], seed=2),
        )
        np.testing.assert_array_equal(np.bincount(result.source_index, minlength=4), [0, 2, 1, 3])

    def test_by_synapse_uses_exact_degree_counts(self):
        cell, exp = _population(4)
        source = NetStim(size=3, start=1.0 * u.ms)
        result = connect(
            "in_degree",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.by_synapse([1, 0, 2, 3], seed=2),
        )
        np.testing.assert_array_equal(
            np.bincount(result.synapse.population_index, minlength=4),
            [1, 0, 2, 3],
        )

    def test_degree_callable_receives_one_dimensional_context(self):
        cell, exp = _population(3)
        source = NetStim(size=4, start=1.0 * u.ms)
        observed = []

        def degree(ctx, rng):
            observed.append(ctx.source.id.shape)
            return np.asarray(rng.binomial(1, 1.0, size=ctx.source.id.shape), dtype=np.int64)

        result = connect(
            "callable_degree",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.by_source(degree, seed=8),
        )
        self.assertEqual(observed, [(4,)])
        self.assertEqual(len(result), 4)

    def test_match_degrees_preserves_both_stub_sequences(self):
        cell, exp = _population(3)
        source = NetStim(size=3, start=1.0 * u.ms)
        result = connect(
            "stub_match",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.match_degrees([1, 2, 0], [0, 1, 2], seed=5),
        )
        np.testing.assert_array_equal(np.bincount(result.source_index, minlength=3), [1, 2, 0])
        np.testing.assert_array_equal(
            np.bincount(result.synapse.population_index, minlength=3),
            [0, 1, 2],
        )

    def test_match_degrees_rejects_unequal_stub_sums(self):
        cell, exp = _population(2)
        source = NetStim(size=2, start=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "equal stub sums"):
            connect(
                "bad_stubs",
                source=source,
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.match_degrees([1, 1], [1, 0]),
            )

    def test_target_cell_groups_are_sorted_and_concatenated(self):
        cell, exp = _population(3)
        source = NetStim(size=4, start=1.0 * u.ms)
        result = connect(
            "grouped",
            source=source,
            synapse=cell.synapses[exp][[2, 0, 1]],
            pairing=braincell.network.connection.independent(
                [1, 2, 3],
                group_by="target_cell",
                seed=11,
            ),
        )
        np.testing.assert_array_equal(result.synapse.population_index, [0, 1, 1, 2, 2, 2])

    def test_target_cell_group_can_explicitly_produce_zero_rows(self):
        cell, exp = _population(3)
        source = NetStim(size=2, start=1.0 * u.ms)
        result = connect(
            "grouped_with_zero",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.independent(
                [1, 0, 2],
                group_by="target_cell",
                seed=11,
            ),
        )
        np.testing.assert_array_equal(result.synapse.population_index, [0, 2, 2])

    def test_grouped_by_source_repeats_degree_for_each_target_cell(self):
        cell, exp = _population(3)
        source = NetStim(size=2, start=1.0 * u.ms)
        result = connect(
            "grouped_degree",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.by_source(1, group_by="target_cell", seed=9),
        )
        self.assertEqual(len(result), 6)
        for population_index in range(3):
            selected = result.source_index[result.synapse.population_index == population_index]
            np.testing.assert_array_equal(np.sort(selected), [0, 1])

    def test_without_replacement_is_local_to_each_fixed_endpoint(self):
        cell, exp = _population(3)
        source = NetStim(size=2, start=1.0 * u.ms)
        result = connect(
            "local_unique",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.by_source(3, replace=False, seed=2),
        )
        for source_id in range(2):
            self.assertEqual(len(np.unique(result.synapse_id[result.source_index == source_id])), 3)
        with self.assertRaisesRegex(ValueError, "positive candidates"):
            connect(
                "too_many",
                source=source,
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.by_source(4, replace=False),
            )

    def test_score_validation_and_zero_rows_do_not_modify_store(self):
        cell, exp = _population(2)
        source = NetStim(size=2, start=1.0 * u.ms)
        with self.assertRaisesRegex(ValueError, "positive support"):
            connect(
                "zero_score",
                source=source,
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.independent(2, source_score=[0.0, 0.0]),
            )
        with self.assertRaisesRegex(ValueError, "zero connection rows"):
            connect(
                "zero_degree",
                source=source,
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.by_source([0, 0]),
            )
        self.assertEqual(len(cell.connections), 0)

    def test_context_exposes_geometry_and_synapse_parameters(self):
        cell, exp = _population(2)
        source = NetStim(size=2, start=[1.0, 2.0] * u.ms)
        observed = {}

        def score(ctx):
            observed["radius"] = ctx.synapse.radius
            observed["distance"] = ctx.synapse.path_distance_to_root
            observed["tau"] = ctx.synapse.get("tau")
            observed["start"] = ctx.source.get("start")
            return np.ones((2, 2))

        connect(
            "context",
            source=source,
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.source_first(
                2,
                synapse_score=score,
                source_replace=False,
                seed=1,
            ),
        )
        self.assertEqual(observed["radius"].shape, (1, 2))
        self.assertEqual(observed["distance"].shape, (1, 2))
        self.assertEqual(observed["tau"].shape, (1, 2))
        self.assertEqual(observed["start"].shape, (2, 1))

    def test_degree_distribution_helpers_return_valid_counts(self):
        cell, exp = _population(3)
        source = NetStim(size=20, start=1.0 * u.ms)
        helpers = (
            braincell.network.connection.degree.poisson(1.0),
            braincell.network.connection.degree.binomial(2, 0.5),
            braincell.network.connection.degree.negative_binomial(2, 0.7),
            braincell.network.connection.degree.empirical([0, 2], [0.25, 0.75]),
        )
        for index, helper in enumerate(helpers):
            result = connect(
                f"distribution_{index}",
                source=source,
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.by_source(helper, seed=index + 1),
            )
            self.assertTrue(np.all(np.bincount(result.source_index, minlength=20) >= 0))


class NetworkConnectionSamplingTest(unittest.TestCase):
    def _build(self, *, network_seed, source_first_order, pairing_seed=None):
        cell, exp = _population(8)
        source = NetStim(size=8, start=1.0 * u.ms)
        network = Network(seed=network_seed)
        if source_first_order:
            network.add_population("pre", source)
            network.add_population("post", cell)
        else:
            network.add_population("post", cell)
            network.add_population("pre", source)
        result = network.connect(
            "ampa_drive",
            source=network.populations["pre"],
            synapse=cell.synapses[exp],
            pairing=braincell.network.connection.independent(40, seed=pairing_seed),
        )
        return result

    def test_implicit_seed_is_order_independent(self):
        first = self._build(network_seed=13, source_first_order=True)
        second = self._build(network_seed=13, source_first_order=False)
        np.testing.assert_array_equal(first.source_index, second.source_index)
        np.testing.assert_array_equal(first.target_index, second.target_index)

    def test_network_seed_changes_implicit_pairing(self):
        first = self._build(network_seed=13, source_first_order=True)
        second = self._build(network_seed=14, source_first_order=True)
        self.assertFalse(
            np.array_equal(first.source_index, second.source_index)
            and np.array_equal(first.target_index, second.target_index)
        )

    def test_explicit_pairing_seed_overrides_network_seed(self):
        first = self._build(network_seed=13, source_first_order=True, pairing_seed=99)
        second = self._build(network_seed=14, source_first_order=True, pairing_seed=99)
        np.testing.assert_array_equal(first.source_index, second.source_index)
        np.testing.assert_array_equal(first.target_index, second.target_index)

    def test_pairing_rejects_place_and_connect_shortcut(self):
        cell, _ = _population(2)
        source = NetStim(size=2, start=1.0 * u.ms)
        network = Network(seed=1)
        network.add_population("pre", source)
        network.add_population("post", cell)
        spec = braincell.mech.SynapseSpec("ExpSyn", name="other", tau=2.0 * u.ms, e=0.0 * u.mV)
        with self.assertRaisesRegex(TypeError, "existing SynapseView"):
            network.connect(
                "invalid",
                source=source,
                target=network.populations["post"],
                locations=at("soma", 0.5),
                synapse=spec,
                pairing=braincell.network.connection.independent(2),
            )


if __name__ == "__main__":
    unittest.main()
