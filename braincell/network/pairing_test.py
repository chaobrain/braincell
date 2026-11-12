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
import warnings

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
    exp = braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV)
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

    def test_omitted_score_draws_exactly_like_an_explicit_uniform_score(self):
        # _score returns None instead of a dense ones matrix for the default
        # case, so the two spellings must remain indistinguishable.
        def rows(score):
            cell, exp = _population(6)
            result = connect(
                "drive",
                source=NetStim(size=4, start=1.0 * u.ms),
                synapse=cell.synapses[exp],
                pairing=braincell.network.connection.by_source(3, seed=5, synapse_score=score),
            )
            return np.asarray(result.source_index), np.asarray(result.target_index)

        default_source, default_target = rows(None)
        uniform_source, uniform_target = rows(np.ones(6))

        np.testing.assert_array_equal(default_source, uniform_source)
        np.testing.assert_array_equal(default_target, uniform_target)

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
        spec = braincell.mech.Synapse("ExpSyn", name="other", tau=2.0 * u.ms, e=0.0 * u.mV)
        with self.assertRaisesRegex(TypeError, "existing SynapseView"):
            network.connect(
                "invalid",
                source=source,
                target=network.populations["post"],
                locations=at("soma", 0.5),
                synapse=spec,
                pairing=braincell.network.connection.independent(2),
            )


def _reference_synapse_geometry(view):
    """``_synapse_geometry`` as written before CELL-18c: one geometry build and one pass per row."""
    from braincell.morph._spatial import MorphologySpatialGeometry, interpolate_branch

    morpho = view.cell.morpho
    geometry = MorphologySpatialGeometry.build(morpho)
    names, types, radius, root_distance, soma_distance, positions = [], [], [], [], [], []
    has_positions = True
    for branch_id, x in zip(view.branch_id.tolist(), view.branch_x.tolist()):
        branch_view = morpho.branch(index=int(branch_id))
        names.append(str(branch_view.name))
        types.append(str(branch_view.type))
        radius_value, point_value = interpolate_branch(morpho, int(branch_id), float(x))
        radius.append(float(radius_value.to_decimal(u.um)))
        root_distance.append(float(geometry.path_distance_to_root(int(branch_id), float(x)).to_decimal(u.um)))
        soma_distance.append(float(geometry.path_distance_from_soma(int(branch_id), float(x)).to_decimal(u.um)))
        if point_value is None:
            has_positions = False
            positions.append(None)
        else:
            positions.append(np.asarray(point_value.to_decimal(u.um), dtype=float))
    local_position = None
    if has_positions:
        local_position = u.Quantity(np.asarray(positions, dtype=float), u.um)
    return (
        np.asarray(names, dtype=object),
        np.asarray(types, dtype=object),
        u.Quantity(np.asarray(radius), u.um),
        u.Quantity(np.asarray(root_distance), u.um),
        u.Quantity(np.asarray(soma_distance), u.um),
        local_position,
    )


def _geometry_cell(*, points: bool, soma: bool = True):
    """Soma (or a dendrite root), two tapering dendrites and an axon, sites on every branch."""
    from braincell.filter import AllRegion, sample

    def branch(lengths, radii, kind, start=None):
        lengths = np.asarray(lengths)
        if not points:
            return Branch.from_lengths(lengths=lengths * u.um, radii=np.asarray(radii) * u.um, type=kind)
        steps = np.cumsum(np.concatenate([[0.0], lengths]))
        xyz = np.stack([steps, 0.3 * steps, np.zeros_like(steps)], axis=1) + (start if start is not None else 0.0)
        with warnings.catch_warnings():
            # The zero-length middle segment of "dend" is deliberate.
            warnings.filterwarnings("ignore", "from_points.. produced 1 zero-length", UserWarning)
            return Branch.from_points(points=xyz * u.um, radii=np.asarray(radii) * u.um, type=kind)

    tree = Morphology.from_root(
        branch([20.0], [10.0, 10.0], "soma" if soma else "basal_dendrite"), name="soma" if soma else "trunk"
    )
    root = tree.root
    root.attach(branch([30.0, 0.0, 45.0], [2.0, 1.5, 1.5, 0.75], "basal_dendrite"), name="dend", parent_x=1.0)
    root.attach(branch([60.0], [1.0, 0.5], "apical_dendrite"), name="apical", parent_x=0.5 if soma else 1.0)
    tree.branch(name="dend").attach(branch([25.0, 25.0], [0.6, 0.5, 0.4], "basal_dendrite"), name="tip")
    root.attach(branch([80.0], [0.5, 0.5], "axon"), name="axon", parent_x=0.0)
    cell = Cell(tree, cv_policy=CVPerBranch())
    syn = braincell.mech.Synapse("ExpSyn", name="syn", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(sample(AllRegion(), number=40, seed=18), syn)
    cell.place(at("soma" if soma else "trunk", 0.0) | at("tip", 1.0) | at("dend", 0.4) | at("dend", 0.4), syn)
    return cell, syn


class SynapseGeometryTest(unittest.TestCase):
    """CELL-18c: the geometry columns of a pairing context, value for value."""

    def _assert_same_columns(self, view) -> None:
        from braincell.network.pairing import _synapse_geometry

        actual, expected = _synapse_geometry(view), _reference_synapse_geometry(view)
        self.assertEqual(len(actual), len(expected))
        for name, got, want in zip(("name", "type", "radius", "root", "soma", "position"), actual, expected):
            with self.subTest(column=name):
                if want is None:
                    self.assertIsNone(got)
                    continue
                if isinstance(want, u.Quantity):
                    self.assertEqual(u.get_unit(got), u.get_unit(want))
                    got, want = got.mantissa, want.mantissa
                self.assertEqual(np.asarray(got).dtype, np.asarray(want).dtype)
                self.assertEqual(np.shape(got), np.shape(want))
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_columns_match_one_pass_per_row(self) -> None:
        for points in (True, False):
            for soma in (True, False):
                with self.subTest(points=points, soma=soma):
                    cell, syn = _geometry_cell(points=points, soma=soma)
                    view = cell.synapses[syn]
                    self.assertGreater(len(np.unique(view.branch_id)), 3)
                    self._assert_same_columns(view)
                    self._assert_same_columns(view[np.arange(len(view))[::-3]])

    def test_geometry_scored_connects_build_the_geometry_once(self) -> None:
        from unittest import mock

        from braincell.morph._spatial import MorphologySpatialGeometry

        cell, syn = _geometry_cell(points=True)
        source = NetStim(size=4, start=1.0 * u.ms)

        def nearer_soma(ctx):
            return 1.0 / (1.0 + ctx.synapse.path_distance_from_soma.to_decimal(u.um))

        build = MorphologySpatialGeometry.build
        with mock.patch.object(MorphologySpatialGeometry, "build", side_effect=build) as counted:
            for index in range(3):
                pairing = braincell.network.connection.independent(5, synapse_score=nearer_soma, seed=index)
                connect(f"c{index}", source=source, synapse=cell.synapses[syn], pairing=pairing)
        self.assertEqual(counted.call_count, 1)

    def test_an_empty_view_has_empty_columns(self) -> None:
        cell, syn = _geometry_cell(points=True)
        self._assert_same_columns(cell.synapses[syn][np.zeros(0, dtype=np.int64)])


def _reference_conditional_sample(
    *, fixed_side, fixed, candidates, score, replace, source_data, synapse_data, group, seed
):
    """``_conditional_sample`` before CELL-18f: rows filled one at a time."""
    from braincell.network.pairing import (
        PairingContext,
        SourceContext,
        SynapseContext,
        _derived_seed,
        _sample,
        _score,
        _score_row,
    )

    fixed = np.asarray(fixed, dtype=np.int64)
    if fixed.size == 0:
        return np.asarray([], dtype=np.int64)
    unique, inverse, counts = np.unique(fixed, return_inverse=True, return_counts=True)
    if fixed_side == "source":
        ctx = PairingContext(
            SourceContext(source_data, unique, fixed=True), SynapseContext(synapse_data, candidates, fixed=False), group
        )
        side = "synapse"
    else:
        ctx = PairingContext(
            SourceContext(source_data, candidates, fixed=False), SynapseContext(synapse_data, unique, fixed=True), group
        )
        side = "source"
    weights = _score(score, ctx, (len(unique), len(candidates)), side=side, positions=candidates)
    sampled_by_fixed = []
    for row, (endpoint, count) in enumerate(zip(unique.tolist(), counts.tolist())):
        endpoint_id = source_data.id[int(endpoint)] if fixed_side == "source" else synapse_data.id[int(endpoint)]
        sampled_by_fixed.append(
            _sample(
                candidates,
                _score_row(weights, row),
                count,
                replace,
                _derived_seed(seed, fixed_side, int(endpoint_id)),
            )
        )
    offsets = np.zeros(len(unique), dtype=np.int64)
    result = np.empty(len(fixed), dtype=np.int64)
    for row, unique_index in enumerate(inverse.tolist()):
        result[row] = sampled_by_fixed[unique_index][offsets[unique_index]]
        offsets[unique_index] += 1
    return result


class ConditionalSampleTest(unittest.TestCase):
    """CELL-18f: each fixed endpoint's partners land on its rows in draw order."""

    def test_partners_match_filling_one_row_at_a_time(self) -> None:
        from unittest import mock

        from braincell.network import pairing
        from braincell.network.event import EventSourceView

        cell, exp = _population(3)
        cell.place(at("soma", 0.25), braincell.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms, e=0.0 * u.mV))
        synapses = cell.synapses[exp]
        source = EventSourceView(NetStim(size=5, start=1.0 * u.ms), np.arange(5))
        n_synapses = len(synapses)
        specs = {
            "source first": pairing.source_first(40, seed=1),
            "source first, scored": pairing.source_first(31, synapse_score=np.linspace(0.1, 1.0, n_synapses), seed=2),
            "by source, without replacement": pairing.by_source(2, replace=False, seed=3),
            "synapse first, scored": pairing.synapse_first(
                27, source_score=lambda ctx: np.arange(1.0, 6.0)[None, :] * np.ones((len(ctx.synapse.id), 1)), seed=4
            ),
            "by source": pairing.by_source(np.asarray([3, 0, 2, 5, 1]), seed=5),
            "by synapse": pairing.by_synapse(2, seed=6),
            "by source, per target cell": pairing.by_source(2, group_by="target_cell", seed=7),
            "source first, per target cell": pairing.source_first(
                np.asarray([7, 0, 4]), group_by="target_cell", seed=8
            ),
        }
        compared = 0
        for name, spec in specs.items():
            with self.subTest(name):
                actual = pairing.materialize_pairing(spec, source, synapses)
                with mock.patch.object(pairing, "_conditional_sample", _reference_conditional_sample):
                    expected = pairing.materialize_pairing(spec, source, synapses)
                np.testing.assert_array_equal(actual.source_position, expected.source_position)
                np.testing.assert_array_equal(actual.synapse_position, expected.synapse_position)
                self.assertEqual(actual.synapse_position.dtype, expected.synapse_position.dtype)
                compared += len(actual.source_position)
        self.assertGreater(compared, 100)

    def test_many_endpoints_match_one_random_state_each(self) -> None:
        # CELL-18h: endpoints that draw the same number of partners, and
        # endpoints that draw different numbers, as one ``RandomState`` each.
        from unittest import mock

        from braincell.network import pairing
        from braincell.network.event import EventSourceView

        rng = np.random.default_rng(18)
        cell, exp = _population(30)
        synapses = cell.synapses[exp]
        source = EventSourceView(NetStim(size=40, start=1.0 * u.ms), np.arange(40))
        sparse = rng.random(30) * (rng.random(30) < 0.6)
        sparse[:8] += 0.5
        specs = {
            "mixed degrees": pairing.by_source(rng.integers(0, 7, 40), seed=11),
            "mixed degrees, without replacement": pairing.by_source(rng.integers(1, 8, 40), replace=False, seed=12),
            "sparse scores": pairing.by_source(3, synapse_score=sparse, seed=13),
            "sparse scores, without replacement": pairing.by_source(
                rng.integers(1, 9, 40), synapse_score=sparse, replace=False, seed=14
            ),
            "by synapse, mixed degrees": pairing.by_synapse(rng.integers(0, 5, 30), seed=2**31 + 5),
            "synapse first": pairing.synapse_first(200, seed=0),
            "source first, scored": pairing.source_first(
                150, synapse_score=lambda ctx: (1.0 + np.asarray(ctx.source.id) % 3) * np.ones((1, 30)), seed=15
            ),
            "one candidate": pairing.by_source(2, synapse_score=np.eye(30)[4], seed=16),
        }
        for name, spec in specs.items():
            with self.subTest(name):
                actual = pairing.materialize_pairing(spec, source, synapses)
                with mock.patch.object(pairing, "_conditional_sample", _reference_conditional_sample):
                    expected = pairing.materialize_pairing(spec, source, synapses)
                np.testing.assert_array_equal(actual.source_position, expected.source_position)
                np.testing.assert_array_equal(actual.synapse_position, expected.synapse_position)

    def test_too_few_candidates_without_replacement_name_the_first_endpoint(self) -> None:
        from unittest import mock

        from braincell.network import pairing
        from braincell.network.event import EventSourceView

        cell, exp = _population(4)
        source = EventSourceView(NetStim(size=3, start=1.0 * u.ms), np.arange(3))
        spec = pairing.by_source(np.asarray([2, 5, 6]), replace=False, seed=1)
        messages = []
        for conditional in (pairing._conditional_sample, _reference_conditional_sample):
            with mock.patch.object(pairing, "_conditional_sample", conditional):
                with self.assertRaises(ValueError) as raised:
                    pairing.materialize_pairing(spec, source, cell.synapses[exp])
            messages.append(str(raised.exception))
        self.assertEqual(messages[0], messages[1])
        self.assertIn("at least 5 positive candidates; got 4", messages[0])

    def _assert_mixed_degrees_match_one_random_state_each(self) -> None:
        from unittest import mock

        from braincell.network import pairing
        from braincell.network.event import EventSourceView

        rng = np.random.default_rng(81)
        cell, exp = _population(30)
        synapses = cell.synapses[exp]
        source = EventSourceView(NetStim(size=40, start=1.0 * u.ms), np.arange(40))
        specs = {
            "with replacement": pairing.by_source(rng.integers(0, 12, 40), seed=31),
            "without replacement": pairing.by_source(rng.integers(1, 12, 40), replace=False, seed=32),
            "scored, with replacement": pairing.by_source(
                rng.integers(1, 7, 40), synapse_score=rng.random(30) + 0.1, seed=33
            ),
        }
        for name, spec in specs.items():
            with self.subTest(name):
                actual = pairing.materialize_pairing(spec, source, synapses)
                with mock.patch.object(pairing, "_conditional_sample", _reference_conditional_sample):
                    expected = pairing.materialize_pairing(spec, source, synapses)
                np.testing.assert_array_equal(actual.synapse_position, expected.synapse_position)

    def test_draws_split_into_blocks_match_one_random_state_each(self) -> None:
        # CELL-18h: a block of 64 probabilities holds two rows of 30 candidates,
        # so each width's rows are drawn two at a time.
        from unittest import mock

        from braincell.network import pairing

        with mock.patch.object(pairing, "_DRAW_BLOCK", 64):
            self._assert_mixed_degrees_match_one_random_state_each()

    def test_replace_flags_are_checked_as_in_one_draw(self) -> None:
        from braincell.network import pairing

        candidates = np.arange(5)
        counts = np.asarray([2, 3])
        drawn = pairing._sample_rows(candidates, None, counts, np.bool_(False), [7, 8])
        for row, seed in enumerate((7, 8)):
            np.testing.assert_array_equal(drawn[row], pairing._sample(candidates, None, counts[row], False, seed))
        for sample in (
            lambda: pairing._sample_rows(candidates, None, counts, 1, [7, 8]),
            lambda: pairing._sample(candidates, None, 2, 1, 7),
        ):
            with self.assertRaisesRegex(TypeError, "replace flags must be bool values"):
                sample()

    def test_without_partitionable_threefry_draws_exactly_count(self) -> None:
        # CELL-18h: with replacement a wider draw is not a prefix-preserving
        # draw under other generator configurations, so each count draws itself.
        import jax

        from braincell.network import pairing

        self.assertEqual(pairing._draw_width(5, 30, True), 8)
        self.assertEqual(pairing._draw_width(5, 6, False), 6)
        previous = jax.config.jax_threefry_partitionable
        jax.config.update("jax_threefry_partitionable", False)
        try:
            self.assertEqual(pairing._draw_width(5, 30, True), 5)
            self.assertEqual(pairing._draw_width(5, 30, False), 8)
            self._assert_mixed_degrees_match_one_random_state_each()
        finally:
            jax.config.update("jax_threefry_partitionable", previous)


if __name__ == "__main__":
    unittest.main()
