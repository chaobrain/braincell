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
from collections import Counter
from unittest import mock

import brainunit as u
import brainstate
import numpy as np

import braincell
from braincell import Branch, Cell, CVPerBranch, Morphology, NetStim, SynapseView, connect
from braincell._multi_compartment.synapses import _BRANCH_NAME_CODES, _SynapseStore
from braincell.filter import at
from braincell.filter import LocsetMask, Terminals
from braincell.morph._testing import make_tree_s


@braincell.mech.register_synapse("_SynapseViewTrainableExpSyn")
class _SynapseViewTrainableExpSyn(braincell.synapse.ExpSyn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = brainstate.nn.Param(self.tau)


@braincell.mech.register_synapse("_DistinctTauExpSyn")
class _DistinctTauExpSyn(braincell.synapse.ExpSyn):
    """A check that relates rows: no declaration fails it alone."""

    @classmethod
    def validate_parameter_values(cls, parameters):
        super().validate_parameter_values(parameters)
        tau = np.asarray(parameters["tau"].to_decimal(u.ms)).reshape(-1)
        if np.unique(tau).size != tau.size:
            raise ValueError("_DistinctTauExpSyn requires distinct tau values.")


def _population(size=2):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    return Cell(morpho, cv_policy=CVPerBranch(), pop_size=(size,))


class SynapseViewTest(unittest.TestCase):
    def test_repr_aggregates_names_by_type_without_reading_values(self):
        cell = _population()
        fast = braincell.mech.Synapse("ExpSyn", name="fast")
        slow = braincell.mech.Synapse("ExpSyn", name="slow")
        nmda = braincell.mech.Synapse("Exp2Syn", name="nmda")
        cell.place(at("soma", 0.2), fast)
        cell.place(at("soma", 0.4), slow)
        cell.place(at("soma", 0.6), nmda)

        display = repr(cell.synapses)
        self.assertIn("ExpSyn  instances=4  names={'fast': 2, 'slow': 2}", display)
        self.assertIn("Exp2Syn  instances=2  names={'nmda': 2}", display)
        selected = repr(cell.synapses["fast"])
        self.assertIn("synapse_type=ExpSyn", selected)
        self.assertIn("names={'fast': 2}", selected)
        single = repr(cell.synapses["fast"][0])
        self.assertIn("parameters={'e':", single)
        self.assertIn("'tau':", single)
        np.testing.assert_array_equal(cell.synapses.by_name("fast").id, cell.synapses["fast"].id)

    def test_root_name_type_and_numeric_selections_are_views(self):
        cell = _population()
        pf = braincell.mech.Synapse("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.Synapse("ExpSyn", name="aa", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.31), pf)
        cell.place(at("soma", 0.39), aa)

        self.assertIsInstance(cell.synapses, SynapseView)
        self.assertIsInstance(cell.synapses[0], SynapseView)
        self.assertIsInstance(cell.synapses["pf"], SynapseView)
        self.assertIsInstance(cell.synapses.by_type("ExpSyn"), SynapseView)
        self.assertEqual(len(cell.synapses), 4)
        self.assertEqual(len(cell.synapses[0]), 1)
        self.assertEqual(cell.synapses.name.tolist(), ["pf", "aa", "pf", "aa"])
        self.assertEqual(cell.synapses["pf"].population_index.tolist(), [0, 1])
        self.assertEqual(cell[1].synapses.id.tolist(), [1, 3])

    def test_same_name_cannot_identify_different_types(self):
        cell = _population()
        cell.place(at("soma", 0.31), braincell.mech.Synapse("ExpSyn", name="shared"))

        with self.assertRaisesRegex(ValueError, "same name.*different synapse types"):
            cell.place(at("soma", 0.39), braincell.mech.Synapse("Exp2Syn", name="shared"))

    def test_one_flat_runtime_per_type_preserves_logical_instances(self):
        cell = _population()
        pf = braincell.mech.Synapse("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.Synapse("ExpSyn", name="aa", tau=2.0 * u.ms, e=-5.0 * u.mV)
        cell.place(at("soma", 0.31), pf)
        cell.place(at("soma", 0.39), aa)

        cell.init_state()

        layouts = [layout for layout in cell.layouts if layout.kind == "synapse:ExpSyn"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual(layouts[0].n_active, 4)
        np.testing.assert_array_equal(layouts[0].population_index, [0, 0, 1, 1])
        np.testing.assert_allclose(cell.synapses.tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(cell.synapses.e.to_decimal(u.mV), [0.0, -5.0, 0.0, -5.0])

    def test_post_init_parameter_and_state_updates_are_separate(self):
        cell = _population()
        exp = braincell.mech.Synapse("ExpSyn", name="exp", tau=1.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)
        cell.init_state()
        view = cell.synapses["exp"]
        cell.reset_state()

        view.set(tau=np.asarray([2.0, 3.0]) * u.ms)
        view.set_state(g=np.asarray([0.1, 0.2]) * u.uS)

        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [2.0, 3.0])
        np.testing.assert_allclose(view.g.to_decimal(u.uS), [0.1, 0.2])
        with self.assertRaisesRegex(KeyError, "state"):
            view.set(g=0.0 * u.uS)
        with self.assertRaisesRegex(KeyError, "parameter"):
            view.set_state(tau=1.0 * u.ms)

    def test_post_init_unit_errors_name_the_synapse(self):
        cell = _population()
        cell.place(at("soma", 0.2), braincell.mech.Synapse("ExpSyn", name="fast"))
        cell.place(at("soma", 0.4), braincell.mech.Synapse("ExpSyn", name="slow"))
        cell.init_state()
        with self.assertRaisesRegex(ValueError, r"^Synapse 'fast' \(ExpSyn\) parameter 'tau' has units .* got 5 mV\.$"):
            cell.synapses["fast"].set(tau=5 * u.mV)
        with self.assertRaisesRegex(TypeError, r"^Synapse 'ExpSyn' parameter 'tau' requires .* got 5.0 \(no unit\)\.$"):
            cell.synapses.set(tau=5.0)

    def test_init_state_names_the_declaration_that_fails_a_type_check(self):
        # CELL-44: the positive and tau1 < tau2 checks see every synapse of a
        # type at once; the error names the declaration whose rows fail.
        cell = _population()
        cell.place(at("soma", 0.2), braincell.mech.Synapse("ExpSyn", name="fast", tau=1.0 * u.ms))
        cell.place(at("soma", 0.4), braincell.mech.Synapse("ExpSyn", name="slow", tau=[2.0, -2.0] * u.ms))
        with self.assertRaisesRegex(
            ValueError,
            r"^Synapse 'slow' \(ExpSyn\) parameter 'tau' must be > 0\. Placed on AtLocation\(branch='soma', x=0\.4\)",
        ):
            cell.init_state()

    def test_init_state_names_the_first_failing_of_many_declarations(self):
        for invalid, named in (({1, 3}, "s1"), ({4}, "s4"), ({0, 1, 2, 3, 4}, "s0")):
            with self.subTest(invalid=sorted(invalid)):
                cell = _population(size=1)
                for index in range(5):
                    tau = (-1.0 if index in invalid else 1.0 + index) * u.ms
                    cell.place(
                        at("soma", 0.1 * (index + 1)), braincell.mech.Synapse("ExpSyn", name=f"s{index}", tau=tau)
                    )
                with self.assertRaisesRegex(
                    ValueError, rf"^Synapse '{named}' \(ExpSyn\) parameter 'tau' must be > 0\."
                ):
                    cell.init_state()

    def test_a_check_no_declaration_fails_alone_keeps_its_message(self):
        cell = _population(size=1)
        cell.place(at("soma", 0.2), braincell.mech.Synapse("_DistinctTauExpSyn", name="a", tau=1.0 * u.ms))
        cell.place(at("soma", 0.4), braincell.mech.Synapse("_DistinctTauExpSyn", name="b", tau=1.0 * u.ms))
        with self.assertRaisesRegex(ValueError, r"^_DistinctTauExpSyn requires distinct tau values\.$"):
            cell.init_state()

    def test_post_init_parameter_validation_is_atomic(self):
        cell = _population()
        exp2 = braincell.mech.Synapse(
            "Exp2Syn",
            name="exp2",
            tau1=1.0 * u.ms,
            tau2=2.0 * u.ms,
        )
        cell.place(at("soma", 0.5), exp2)
        cell.init_state()
        view = cell.synapses["exp2"]
        cell.reset_state()

        view.set(tau1=3.0 * u.ms, tau2=4.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])
        np.testing.assert_allclose(view.tau2.to_decimal(u.ms), [4.0, 4.0])

        with self.assertRaisesRegex(ValueError, "must be > 0"):
            view.set(tau1=-1.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])

        with self.assertRaisesRegex(ValueError, "tau1 < tau2"):
            view.set(tau1=5.0 * u.ms)
        np.testing.assert_allclose(view.tau1.to_decimal(u.ms), [3.0, 3.0])

    def test_reset_state_clears_pending_event_payload(self):
        cell = _population()
        exp = braincell.mech.Synapse("ExpSyn", name="exp")
        cell.place(at("soma", 0.5), exp)
        cell.init_state()
        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        cell.runtime.event_buffers[layout.id].value = np.asarray([0.1, 0.2]) * u.uS

        cell.reset_state()

        np.testing.assert_array_equal(
            cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS),
            [0.0, 0.0],
        )

    def test_connection_materializes_stable_synapse_ids(self):
        cell = _population()
        exp = braincell.mech.Synapse("ExpSyn", name="exp", tau=1.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.5), exp)

        connection = connect(
            "exp_input",
            source=NetStim(size=2),
            synapse=cell.synapses["exp"],
            weight=0.1 * u.uS,
        )

        np.testing.assert_array_equal(connection.synapse_id, cell.synapses["exp"].id)

    def test_logical_ids_remain_stable_when_later_placements_change_view_order(self):
        cell = _population()
        first = braincell.mech.Synapse("ExpSyn", name="first", tau=1.0 * u.ms, e=0.0 * u.mV)
        later = braincell.mech.Synapse("ExpSyn", name="later", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.3), first)
        original_ids = cell.synapses[first].id
        connection = connect(
            "first_input",
            source=NetStim(size=2),
            synapse=cell.synapses[first],
            weight=0.1 * u.uS,
        )

        cell.place(at("soma", 0.7), later)

        np.testing.assert_array_equal(cell.synapses[first].id, original_ids)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            _ = connection.synapse_id
        np.testing.assert_array_equal(cell.connections["first_input"].synapse_id, original_ids)
        self.assertEqual(cell.synapses.name.tolist(), ["first", "later", "first", "later"])
        self.assertEqual(cell.synapses.id.tolist(), [0, 2, 1, 3])

    def test_per_cell_ragged_locsets_accept_flat_and_per_cell_parameters(self):
        cell = _population()
        locations = (
            LocsetMask.from_columns([0, 0], [0.2, 0.4]),
            LocsetMask.from_columns([0], [0.7]),
        )
        exp = braincell.mech.Synapse(
            "ExpSyn",
            name="ragged",
            tau=np.asarray([1.0, 2.0, 3.0]) * u.ms,
            e=np.asarray([[-70.0], [-60.0]]) * u.mV,
        )

        cell.place(locations, exp)
        cell.init_state()
        view = cell.synapses[exp]
        cell.reset_state()

        self.assertEqual(view.population_index.tolist(), [0, 0, 1])
        np.testing.assert_allclose(view.branch_x, [0.2, 0.4, 0.7])
        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(view.e.to_decimal(u.mV), [-70.0, -70.0, -60.0])

    def test_rectangular_per_cell_locsets_support_location_axis_broadcast(self):
        cell = _population()
        locations = (
            LocsetMask.from_columns([0, 0], [0.2, 0.4]),
            LocsetMask.from_columns([0, 0], [0.6, 0.8]),
        )
        exp = braincell.mech.Synapse(
            "ExpSyn",
            name="rectangular",
            tau=np.asarray([1.0, 2.0]) * u.ms,
            e=0.0 * u.mV,
        )

        cell.place(locations, exp)
        cell.init_state()

        np.testing.assert_allclose(cell.synapses[exp].tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])

    def test_locset_batch_supports_location_and_population_parameter_axes(self):
        cell = _population()
        locations = cell.cv_midpoints[np.asarray([[0, 0], [0, 0]])]
        exp = braincell.mech.Synapse(
            "ExpSyn",
            name="batch",
            tau=np.asarray([1.0, 2.0]) * u.ms,
            e=np.asarray([[-70.0], [-60.0]]) * u.mV,
        )

        cell.place(locations, exp)
        cell.init_state()

        view = cell.synapses[exp]
        np.testing.assert_allclose(view.tau.to_decimal(u.ms), [1.0, 2.0, 1.0, 2.0])
        np.testing.assert_allclose(view.e.to_decimal(u.mV), [-70.0, -70.0, -60.0, -60.0])

    def test_per_cell_locsets_allow_empty_rows(self):
        cell = _population()
        locations = (
            LocsetMask(),
            LocsetMask.from_columns([0], [0.5]),
        )
        exp = braincell.mech.Synapse("ExpSyn", name="sparse", tau=2.0 * u.ms, e=0.0 * u.mV)

        cell.place(locations, exp)

        self.assertEqual(cell.synapses[exp].population_index.tolist(), [1])

    def test_connection_targets_one_name_inside_shared_type_runtime(self):
        cell = _population()
        pf = braincell.mech.Synapse("ExpSyn", name="pf", tau=1.0 * u.ms, e=0.0 * u.mV)
        aa = braincell.mech.Synapse("ExpSyn", name="aa", tau=2.0 * u.ms, e=0.0 * u.mV)
        cell.place(at("soma", 0.3), pf)
        cell.place(at("soma", 0.7), aa)
        braincell.connect(
            "drive",
            source=braincell.NetStim(start=0.0 * u.ms, number=1),
            synapse=cell.synapses["pf"],
            weight=np.asarray([1.0, 2.0]) * 0.1 * u.uS,
        )
        cell.init_state()

        with brainstate.environ.context(t=0.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        cell.runtime.get_runtime_node(layout.id)
        np.testing.assert_allclose(
            cell.runtime.get_event_buffer(layout.id).to_decimal(u.uS),
            [0.1, 0.0, 0.2, 0.0],
        )

    def test_post_init_set_preserves_trainable_parameter_wrapper(self):
        cell = _population()
        spec = braincell.mech.Synapse(
            "_SynapseViewTrainableExpSyn",
            name="trainable",
            tau=1.0 * u.ms,
            e=0.0 * u.mV,
        )
        cell.place(at("soma", 0.5), spec)
        cell.init_state()
        layout = next(layout for layout, _ in cell.runtime.iter_synapse_layouts())
        node = cell.runtime.get_runtime_node(layout.id)
        parameter = node.tau

        cell.synapses["trainable"].set(tau=np.asarray([2.0, 3.0]) * u.ms)

        self.assertIs(node.tau, parameter)
        self.assertIsInstance(node.tau, brainstate.nn.Param)
        np.testing.assert_allclose(node.tau.value().to_decimal(u.ms), [2.0, 3.0])


def _declare_every_parameter_form(n):
    """Tree S with ``n`` members and one synapse declaration per parameter form.

    Returns the cell and, per declaration name, the declared value of each
    parameter as a function of ``(member, k, i)``: ``k`` numbers that
    declaration's synapses within the member, ``i`` across all members in
    row order. A parameter left out takes the type's default.
    """
    cell = Cell(make_tree_s(), pop_size=(n,))
    member = np.arange(n)
    tau_a = (1.0 + 0.001 * member).reshape(n, 1)
    tau_c = 2.0 + 0.001 * member
    tau_d = (2500.1 + 0.37 * member).reshape(n, 1)
    tau_rec = 500.0 + member[:, None] + 100.0 * np.arange(2)[None, :]
    counts = 1 + member % 2
    tau1 = 0.5 + 1e-4 * np.arange(int(counts.sum()))
    Synapse = braincell.mech.Synapse
    # Terminals() is the dend and axon ends: two synapses per member.
    cell.place(Terminals(), Synapse("ExpSyn", name="a", tau=tau_a * u.ms))
    cell.place(at("soma", 0.5), Synapse("ExpSyn", name="b", tau=1500.0 * u.us, e=-0.005 * u.volt))
    cell.place(at("dend", 0.5), Synapse("ExpSyn", name="c", tau=tau_c * u.ms, e=np.asarray([-7.0]) * u.mV))
    # An array in another unit than the column's (ms, from "a").
    cell.place(at("axon", 0.5), Synapse("ExpSyn", name="d", tau=tau_d * u.us))
    cell.place(
        Terminals(),
        Synapse("ExpSynSTD", name="std", tau=2.0 * u.ms, U=np.asarray([0.25, 0.75]), tau_rec=tau_rec * u.ms),
    )
    cell.place(at("soma", 0.25), Synapse("ExpSynSTD", name="std1", tau=3.0 * u.ms, U=1))
    # One or two synapses per member at the same site: repeat 0 and 1.
    locations = tuple(LocsetMask.from_columns(np.zeros(count, dtype=int), np.full(count, 0.2)) for count in counts)
    cell.place(locations, Synapse("Exp2Syn", name="x", tau1=tau1 * u.ms, tau2=10 * u.ms))
    declared = {
        "a": {"tau": lambda m, k, i: u.Quantity(tau_a[m, 0], u.ms)},
        "b": {"tau": lambda m, k, i: 1500.0 * u.us, "e": lambda m, k, i: -0.005 * u.volt},
        "c": {"tau": lambda m, k, i: u.Quantity(tau_c[m], u.ms), "e": lambda m, k, i: u.Quantity(-7.0, u.mV)},
        "d": {"tau": lambda m, k, i: u.Quantity(tau_d[m, 0], u.us)},
        "std": {
            "tau": lambda m, k, i: 2.0 * u.ms,
            "U": lambda m, k, i: np.float64([0.25, 0.75][k]),
            "tau_rec": lambda m, k, i: u.Quantity(tau_rec[m, k], u.ms),
        },
        "std1": {"tau": lambda m, k, i: 3.0 * u.ms, "U": lambda m, k, i: 1},
        "x": {"tau1": lambda m, k, i: u.Quantity(tau1[i], u.ms), "tau2": lambda m, k, i: 10 * u.ms},
    }
    return cell, declared


def _reference_store(cell, declared):
    """The store's columns built one row at a time from the cell's placements.

    This is the construction the store did before CELL-38, written out
    without shortcuts: rows are cell-major and, within a member, in
    placement id order; each parameter column stacks one value per row
    in the unit of the type's first row.
    """
    n = int(cell.pop_size[0])
    placements = [p for p in cell.point_placements if isinstance(p.mechanism, braincell.mech.Synapse)]
    rows = sorted(
        (
            (owner, p)
            for p in placements
            for owner in (range(n) if p.population_index is None else (int(p.population_index),))
        ),
        key=lambda row: (row[0], row[1].id),
    )
    seen, repeat, sites = Counter(), {}, {}
    for p in placements:
        member = -1 if p.population_index is None else int(p.population_index)
        key = (p._rule_index, p._mechanism_index, member, p.branch_name, float(p.branch_x))
        repeat[p.id] = seen[key]
        seen[key] += 1
        sites.setdefault(key[:2], []).append(key[2:])
    expected = {
        "id": [p.id * n + owner for owner, p in rows],
        "placement_id": [p.id for _, p in rows],
        "population_index": [owner for owner, _ in rows],
        "point_id": [p.point_id for _, p in rows],
        "cv_id": [p.cv_id for _, p in rows],
        "branch_id": [p.branch_id for _, p in rows],
        "branch_x": [float(p.branch_x) for _, p in rows],
        "branch_name_code": [_BRANCH_NAME_CODES[p.branch_name] for _, p in rows],
        "rule_index": [p._rule_index for _, p in rows],
        "mechanism_index": [p._mechanism_index for _, p in rows],
        "repeat_index": [repeat[p.id] for _, p in rows],
        "name": [p.mechanism.instance_name for _, p in rows],
        "synapse_type": [p.mechanism.synapse_type for _, p in rows],
    }
    k_seen, i_seen, values = Counter(), Counter(), []
    for owner, p in rows:
        name = p.mechanism.instance_name
        values.append((owner, k_seen[(owner, name)], i_seen[name]))
        k_seen[(owner, name)] += 1
        i_seen[name] += 1
    columns, type_rows, local_by_id = {}, {}, {}
    for synapse_type in dict.fromkeys(expected["synapse_type"]):
        schema = braincell.mech.get_registry().get("synapse", synapse_type).parameter_info()
        selected = [row for row, item in enumerate(expected["synapse_type"]) if item == synapse_type]
        type_rows[synapse_type] = selected
        local_by_id.update((expected["id"][row], local) for local, row in enumerate(selected))
        columns[synapse_type] = {}
        for parameter, spec in schema.items():
            items = []
            for row in selected:
                given = declared[expected["name"][row]].get(parameter)
                items.append(spec.default if given is None else given(*values[row]))
            if isinstance(items[0], u.Quantity):
                unit = items[0].unit
                column = u.Quantity(np.asarray([np.asarray(item.to_decimal(unit)) for item in items]), unit)
            else:
                column = np.asarray(items)
            columns[synapse_type][parameter] = column
    sites = {key: tuple(sorted(value)) for key, value in sites.items()}
    return rows, expected, sites, columns, type_rows, local_by_id


class SynapseStoreBuildTest(unittest.TestCase):
    """CELL-38: the store matches its row-by-row construction, element by element and in order."""

    def _check_store_matches_reference(self, n):
        cell, declared = _declare_every_parameter_form(n)
        store = _SynapseStore(cell)
        rows, expected, sites, columns, type_rows, local_by_id = _reference_store(cell, declared)

        self.assertEqual(len(store.id), len(rows))
        for field, values in expected.items():
            with self.subTest(field=field):
                actual = getattr(store, field)
                self.assertIsInstance(actual, np.ndarray)
                self.assertEqual(actual.dtype, np.asarray(values, dtype=actual.dtype).dtype)
                np.testing.assert_array_equal(actual, np.asarray(values, dtype=actual.dtype))
        self.assertEqual(store.id.dtype, np.int64)
        self.assertEqual(store.branch_x.dtype, np.float64)
        self.assertEqual(store.name.dtype, object)
        self.assertIsInstance(store.mechanism, tuple)
        self.assertIsInstance(store.declaration, tuple)
        self.assertEqual(len(store.mechanism), len(rows))
        self.assertTrue(all(item is p.mechanism for item, (_, p) in zip(store.mechanism, rows)))
        origins = cell._synapse_origins
        self.assertTrue(
            all(item is origins.get(id(p.mechanism), p.mechanism) for item, (_, p) in zip(store.declaration, rows))
        )
        self.assertEqual(store.sites, sites)
        self.assertEqual(list(store.parameter_columns), list(columns))
        for synapse_type, type_columns in columns.items():
            self.assertEqual(list(store.parameter_columns[synapse_type]), list(type_columns))
            for parameter, column in type_columns.items():
                with self.subTest(synapse_type=synapse_type, parameter=parameter):
                    actual = store.parameter_columns[synapse_type][parameter]
                    self.assertIs(type(actual), type(column))
                    if isinstance(column, u.Quantity):
                        self.assertEqual(actual.unit, column.unit)
                        actual, column = np.asarray(actual.mantissa), np.asarray(column.mantissa)
                    self.assertEqual(actual.dtype, column.dtype)
                    np.testing.assert_array_equal(actual, column)
        self.assertEqual(list(store._type_rows), list(type_rows))
        for synapse_type, selected in type_rows.items():
            self.assertEqual(store._type_rows[synapse_type].dtype, np.int64)
            np.testing.assert_array_equal(store._type_rows[synapse_type], selected)
        self.assertEqual(store._type_local_by_id, local_by_id)
        self.assertEqual(store._row_by_id, {logical_id: row for row, logical_id in enumerate(expected["id"])})
        return store, columns

    def test_store_matches_row_by_row_construction_for_10_members(self):
        store, columns = self._check_store_matches_reference(10)
        # The forms the declarations exercise, read back in members 0 and 1.
        tau = columns["ExpSyn"]["tau"].mantissa
        np.testing.assert_array_equal(tau[[0, 1, 2, 3, 5, 6, 7, 8]], [1.0, 1.0, 1.5, 2.0, 1.001, 1.001, 1.5, 2.001])
        np.testing.assert_allclose(tau[[4, 9]], [2.5001, 2.50047], rtol=1e-15)
        np.testing.assert_array_equal(columns["ExpSyn"]["e"].mantissa[:5], [0.0, 0.0, -5.0, -7.0, 0.0])
        np.testing.assert_array_equal(columns["ExpSynSTD"]["U"][:6], [0.25, 0.75, 1.0, 0.25, 0.75, 1.0])
        self.assertEqual(columns["Exp2Syn"]["tau2"].mantissa.dtype, np.int64)
        np.testing.assert_array_equal(store.repeat_index[store.name == "x"][:3], [0, 0, 1])

    def test_store_matches_row_by_row_construction_for_1000_members(self):
        self._check_store_matches_reference(1000)

    def test_valid_declarations_skip_the_row_by_row_build(self):
        # Every parameter form above takes the per-declaration build; only a
        # failing value rebuilds its column row by row to name the row.
        cell, _ = _declare_every_parameter_form(10)
        with mock.patch.object(_SynapseStore, "_column_by_row", side_effect=AssertionError("row-by-row build")):
            store = _SynapseStore(cell)
        self.assertEqual(len(store.id), 10 * 8 + 15)
        cell = Cell(make_tree_s(), pop_size=(2,))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", tau=np.nan * u.ms, e=0.0 * u.mV))
        by_row = _SynapseStore._column_by_row
        with mock.patch.object(_SynapseStore, "_column_by_row", autospec=True, side_effect=by_row) as spy:
            with self.assertRaisesRegex(ValueError, r"^Synapse 'ExpSyn' parameter 'tau' must contain only finite"):
                _SynapseStore(cell)
        self.assertEqual([call.args[1] for call in spy.call_args_list], ["tau"])

    def test_invalid_values_name_the_first_failing_row(self):
        Synapse = braincell.mech.Synapse
        soma = at("soma", 0.5)
        cases = (
            # "a" first fails in member 1; "b" already in member 0.
            (
                lambda: (
                    (Terminals(), Synapse("ExpSyn", name="a", tau=[[1.0], [np.nan], [3.0]] * u.ms)),
                    (soma, Synapse("ExpSyn", name="b", tau=np.inf * u.ms)),
                ),
                ValueError,
                r"^Synapse 'b' \(ExpSyn\) parameter 'tau' must contain only finite values; got inf ms\. Placed on ",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", tau=[1.0, np.nan, 3.0] * u.ms)),),
                ValueError,
                r"^Synapse 's' \(ExpSyn\) parameter 'tau' must contain only finite values; got nan ms\. Placed on ",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", tau=[1.0, 2.0] * u.ms)),),
                ValueError,
                r"^Synapse parameter shape \(2,\) cannot broadcast to 3 logical instances\.$",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", tau=2 * u.mV)),),
                ValueError,
                r"^Synapse 's' \(ExpSyn\) parameter 'tau' has units incompatible with ms; got 2 mV\. Placed on ",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", tau=2.0)),),
                TypeError,
                r"^Synapse 's' \(ExpSyn\) parameter 'tau' requires a quantity compatible with ms; got 2\.0 \(no unit\)",
            ),
            (
                lambda: ((soma, Synapse("ExpSynSTD", name="s", U=0.5 * u.ms)),),
                TypeError,
                r"^Synapse 's' \(ExpSynSTD\) parameter 'U' must be dimensionless; got 0\.5 ms\. Placed on ",
            ),
            (
                lambda: ((soma, Synapse("ExpSynSTD", name="s", U=[0.5, 0.6, 0.2])),),
                ValueError,
                r"^Synapse parameter 'U' must resolve to one scalar per logical instance; got shape \(3, 3\)\.$",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", tau=-1.0 * u.ms)),),
                ValueError,
                r"^Synapse 's' \(ExpSyn\) parameter 'tau' must be > 0\. Placed on ",
            ),
            (
                lambda: ((soma, Synapse("ExpSyn", name="s", foo=1.0)),),
                TypeError,
                r"^Synapse type 'ExpSyn' has no parameters \('foo',\)\.$",
            ),
        )
        for declare, error, message in cases:
            with self.subTest(message=message):
                cell = Cell(make_tree_s(), pop_size=(3,))
                with warnings.catch_warnings():
                    # Declarations warn now and raise from braincell 0.4.
                    warnings.simplefilter("ignore", FutureWarning)
                    for locset, declaration in declare():
                        cell.place(locset, declaration)
                with self.assertRaisesRegex(error, message):
                    _SynapseStore(cell)

    def test_values_of_a_parameter_without_a_unit_prototype(self):
        # Each value is its own unit prototype, so values that pass their
        # own check can still disagree with each other.
        @braincell.mech.register_synapse("_FreeGainExpSyn")
        class _FreeGainExpSyn(braincell.synapse.ExpSyn):
            def __init__(self, size, name=None, tau=0.1 * u.ms, e=0.0 * u.mV, gain=None):
                super().__init__(size, name=name, tau=tau, e=e)
                self.gain = gain

        cases = (
            ((2.0 * u.ms, 2500.0 * u.us), None, None),
            ((2.0, 1.0 * u.ms), TypeError, r"^Synapse parameter 'gain' mixes dimensioned and dimensionless values\.$"),
            (
                (2.0 * u.ms, 1.0 * u.mV),
                ValueError,
                r"^Synapse parameter 'gain' has incompatible units across instances\.$",
            ),
        )
        try:
            for gains, error, message in cases:
                with self.subTest(gains=gains):
                    cell = Cell(make_tree_s(), pop_size=(2,))
                    for x, gain in zip((0.25, 0.75), gains):
                        cell.place(at("soma", x), braincell.mech.Synapse("_FreeGainExpSyn", gain=gain))
                    if error is None:
                        column = _SynapseStore(cell).parameter_columns["_FreeGainExpSyn"]["gain"]
                        self.assertEqual(column.unit, u.ms)
                        np.testing.assert_array_equal(column.mantissa, [2.0, 2.5, 2.0, 2.5])
                    else:
                        with self.assertRaisesRegex(error, message):
                            _SynapseStore(cell)
        finally:
            braincell.mech.get_registry().unregister("synapse", "_FreeGainExpSyn")


if __name__ == "__main__":
    unittest.main()
