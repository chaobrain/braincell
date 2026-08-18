import unittest

import brainunit as u
import brainstate
import numpy as np

import braincell
from braincell import Branch, Cell, Contact, CVPerBranch, Morphology, NetStim, Network
from braincell.filter import at


def _population(size=3):
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[10.0, 10.0] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    cell = Cell(morpho, cv_policy=CVPerBranch(), pop_size=(size,))
    exp = braincell.mech.Synapse("ExpSyn", name="exp", tau=2.0 * u.ms, e=0.0 * u.mV)
    cell.place(at("soma", 0.5), exp)
    return cell, exp


class ContactTest(unittest.TestCase):
    def test_equal_sizes_zip_and_target_cell_owns_contact(self) -> None:
        cell, exp = _population(3)
        contact = Contact(
            source=NetStim(size=3),
            target=cell.synapses[exp],
            weight=np.asarray([1.0, -2.0, 3.0]) * u.uS,
        )

        np.testing.assert_array_equal(contact.source_index, [0, 1, 2])
        np.testing.assert_array_equal(contact.target_index, [0, 1, 2])
        np.testing.assert_array_equal(contact.id, [0, 1, 2])
        self.assertEqual(cell.contacts, (contact,))

    def test_fanout_convergence_explicit_pairs_and_nonreused_ids(self) -> None:
        cell, exp = _population(3)
        fanout = Contact(source=NetStim(), target=cell.synapses[exp])
        convergence = Contact(source=NetStim(size=2), target=cell.synapses[exp][0])
        explicit = Contact(
            source=NetStim(size=2),
            target=cell.synapses[exp],
            pairs=np.asarray([[1, 2], [0, 0]]),
        )
        fanout.remove()

        np.testing.assert_array_equal(convergence.target_index, [0, 0])
        np.testing.assert_array_equal(explicit.source_index, [1, 0])
        np.testing.assert_array_equal(explicit.target_index, [2, 0])
        self.assertEqual(convergence.id.tolist(), [3, 4])
        self.assertEqual(explicit.id.tolist(), [5, 6])
        self.assertTrue(fanout.removed)
        self.assertEqual(cell.contacts, (convergence, explicit))

    def test_incompatible_shapes_require_pairs(self) -> None:
        cell, exp = _population(3)
        with self.assertRaisesRegex(ValueError, "provide pairs"):
            Contact(source=NetStim(size=2), target=cell.synapses[exp])

    def test_heterogeneous_events_scatter_to_aligned_population_targets(self) -> None:
        cell, exp = _population(3)
        Contact(
            source=NetStim(
                size=3,
                start=np.asarray([1.0, 2.0, 1.0]) * u.ms,
                interval=10.0 * u.ms,
            ),
            target=cell.synapses[exp],
            weight=np.asarray([1.0, 2.0, -3.0]) * u.uS,
        )
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        node = cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(node.pre_drive().to_decimal(u.uS)[..., 0], [1.0, 0.0, -3.0])

    def test_target_owned_contacts_run_inside_network_population(self) -> None:
        cell, exp = _population(3)
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
        )
        Contact(
            source=NetStim(size=3, start=0.5 * u.ms, interval=10.0 * u.ms),
            target=cell.synapses[exp],
            weight=np.asarray([0.1, 0.2, 0.3]) * u.uS,
        )
        network = Network(name="netstim_population")
        network.add_population("post", cell)

        result = network.run(dt=0.05 * u.ms, duration=2.0 * u.ms)

        self.assertGreater(float(np.max(result.traces["post"]["g"].to_decimal(u.uS))), 0.0)

    def test_non_grid_event_and_delay_arrive_on_nearest_step_across_runs(self) -> None:
        cell, exp = _population(1)
        cell.place(
            at("soma", 0.5),
            braincell.mech.MechanismProbe(name="g", mechanism="exp", field="g"),
        )
        Contact(
            source=NetStim(start=1.03 * u.ms, interval=10.0 * u.ms),
            target=cell.synapses[exp],
            weight=0.2 * u.uS,
            delay=0.10 * u.ms,
        )

        first = cell.run(dt=0.05 * u.ms, duration=1.0 * u.ms)
        second = cell.run(dt=0.05 * u.ms, duration=0.5 * u.ms)
        first_g = first.traces["g"].to_decimal(u.uS)[..., 0]
        second_g = second.traces["g"].to_decimal(u.uS)[..., 0]

        np.testing.assert_allclose(first_g, 0.0)
        first_nonzero = int(np.flatnonzero(second_g > 0.0)[0])
        self.assertAlmostEqual(float(second.time[first_nonzero].to_decimal(u.ms)), 1.15)

    def test_non_grid_event_uses_nearest_not_ceil_step(self) -> None:
        source = NetStim(start=1.021 * u.ms, interval=10.0 * u.ms)

        at_nearest = source.event_count(
            np.asarray([0]),
            t=1.0 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )
        at_ceil = source.event_count(
            np.asarray([0]),
            t=1.05 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )

        np.testing.assert_array_equal(at_nearest, [1])
        np.testing.assert_array_equal(at_ceil, [0])

    def test_exact_half_step_tie_uses_later_boundary(self) -> None:
        source = NetStim(start=1.025 * u.ms, interval=10.0 * u.ms)

        earlier = source.event_count(
            np.asarray([0]),
            t=1.0 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )
        later = source.event_count(
            np.asarray([0]),
            t=1.05 * u.ms,
            delay=np.asarray([0.0]) * u.ms,
            dt=0.05 * u.ms,
        )

        np.testing.assert_array_equal(earlier, [0])
        np.testing.assert_array_equal(later, [1])

    def test_multiple_contacts_add_on_one_target(self) -> None:
        cell, exp = _population(1)
        target = cell.synapses[exp]
        Contact(source=NetStim(start=1.0 * u.ms), target=target, weight=0.2 * u.uS)
        Contact(source=NetStim(start=1.0 * u.ms), target=target, weight=-0.05 * u.uS)
        cell.init_state()
        layout = next(item for item in cell.runtime.layouts if item.kind == "synapse:ExpSyn")
        node = cell.runtime.get_runtime_node(layout.id)

        with brainstate.environ.context(t=1.0 * u.ms, dt=0.05 * u.ms):
            cell._prepare_runtime_synapse_inputs(cell._cv_to_point(cell.V.value))

        np.testing.assert_allclose(node.pre_drive().to_decimal(u.uS), [[0.15]])


if __name__ == "__main__":
    unittest.main()
