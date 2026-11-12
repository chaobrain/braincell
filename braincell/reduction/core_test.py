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

"""Tests for the reduction contract types: construction checks, read-only views, lifecycle defaults and views."""

import dataclasses
import gc
import unittest

import brainunit as u
import numpy as np

import braincell
from braincell.filter import at


class _Owner:
    """Stand-in for the owning Cell: a context reads ``pop_size`` and keeps a weak reference."""

    def __init__(self, pop_size) -> None:
        self.pop_size = pop_size


def _synapse(**parameters) -> braincell.ReductionSynapse:
    return braincell.ReductionSynapse(
        id=0,
        synapse_index=0,
        population_index=0,
        placement_id=0,
        point_id=0,
        cv_id=0,
        branch_id=0,
        branch_x=0.5,
        name="syn",
        synapse_type="ExpSyn",
        **parameters,
    )


def _schema(layout_id=0, *, synapse_id=(0, 1), synapse_index=(0, 1), population_index=(0, 0)):
    return braincell.ReductionInputGroupSchema(
        layout_id=layout_id,
        synapse_type="ExpSyn",
        event_input=braincell.mech.ScalarEventInput(u.uS),
        synapse_id=synapse_id,
        synapse_index=synapse_index,
        population_index=population_index,
    )


def _context(owner) -> braincell.ReductionContext:
    return braincell.ReductionContext.with_cell(owner, synapses=(), input_groups=(), fingerprint="core-test")


def _cell(pop_size=3) -> braincell.Cell:
    soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
    cell = braincell.Cell(
        braincell.Morphology.from_root(soma, name="soma"),
        pop_size=pop_size,
        cv_policy=braincell.CVPerBranch(1),
    )
    cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="syn"))
    return cell


class ReductionSynapseTest(unittest.TestCase):
    def test_parameters_are_a_read_only_snapshot(self) -> None:
        source = {"tau": 2.0 * u.ms}
        synapse = _synapse(parameters=source)
        source["tau"] = 5.0 * u.ms
        source["e"] = 0.0 * u.mV

        self.assertEqual(list(synapse.parameters), ["tau"])
        self.assertEqual(synapse.parameters["tau"].to_decimal(u.ms), 2.0)
        with self.assertRaises(TypeError):
            synapse.parameters["tau"] = 1.0 * u.ms
        with self.assertRaises(dataclasses.FrozenInstanceError):
            synapse.parameters = {}

    def test_default_parameters_are_empty(self) -> None:
        self.assertEqual(dict(_synapse().parameters), {})


class ReductionInputGroupSchemaTest(unittest.TestCase):
    def test_index_arrays_are_int64_one_dimensional_and_read_only(self) -> None:
        schema = _schema(synapse_id=[0, 1], synapse_index=[0, 1], population_index=[0, 0])

        for name in ("synapse_id", "synapse_index", "population_index"):
            with self.subTest(name=name):
                array = getattr(schema, name)
                self.assertEqual(array.dtype, np.int64)
                self.assertEqual(array.ndim, 1)
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array[0] = 7
        np.testing.assert_array_equal(schema.synapse_id, [0, 1])
        np.testing.assert_array_equal(schema.population_index, [0, 0])
        self.assertEqual(schema.size, 2)

    def test_index_arrays_are_flattened_copies(self) -> None:
        source = np.asarray([[4], [5]], dtype=np.int64)
        schema = _schema(synapse_id=source)
        source[0, 0] = 9

        np.testing.assert_array_equal(schema.synapse_id, [4, 5])
        self.assertTrue(source.flags.writeable)
        self.assertEqual(schema.size, 2)


class ReductionInputsTest(unittest.TestCase):
    def test_empty_inputs(self) -> None:
        inputs = braincell.ReductionInputs()

        self.assertEqual(len(inputs), 0)
        self.assertEqual(list(inputs), [])

    def test_iteration_returns_groups_in_order(self) -> None:
        first = braincell.ReductionInputGroup(_schema(0), payload=[1.0, 0.0] * u.uS)
        second = braincell.ReductionInputGroup(_schema(1), payload=[0.0, 2.0] * u.uS)
        inputs = braincell.ReductionInputs((first, second))

        self.assertEqual(len(inputs), 2)
        groups = list(inputs)
        self.assertIs(groups[0], first)
        self.assertIs(groups[1], second)


class ReductionOutputTest(unittest.TestCase):
    def test_values_are_a_read_only_copy_in_declaration_order(self) -> None:
        source = {"voltage": np.zeros(2), "latent": np.ones(2)}
        output = braincell.ReductionOutput(values=source, event=np.zeros(2, dtype=np.int32))
        source["extra"] = np.zeros(2)

        self.assertEqual(list(output.values), ["voltage", "latent"])
        with self.assertRaises(TypeError):
            output.values["voltage"] = np.ones(2)

    def test_names_must_be_non_empty_strings(self) -> None:
        for name in ("", 1):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "non-empty strings"):
                braincell.ReductionOutput(values={name: np.zeros(1)}, event=np.zeros(1, dtype=np.int32))


class ReductionContextTest(unittest.TestCase):
    def test_population_size_flattens_pop_size(self) -> None:
        for pop_size, expected in (((2,), 2), ((), 1), ((2, 3), 6)):
            with self.subTest(pop_size=pop_size):
                owner = _Owner(pop_size)
                context = _context(owner)
                self.assertEqual(context.pop_size, pop_size)
                self.assertEqual(context.population_size, expected)
                self.assertIs(context.cell, owner)

    def test_pop_size_is_stored_as_a_tuple(self) -> None:
        owner = _Owner([4])

        self.assertEqual(_context(owner).pop_size, (4,))

    def test_cell_raises_once_the_owner_is_collected(self) -> None:
        owner = _Owner((2,))
        context = _context(owner)
        del owner
        gc.collect()

        with self.assertRaisesRegex(RuntimeError, "no longer exists"):
            _ = context.cell
        self.assertEqual(context.population_size, 2)


class _InitOnly(braincell.ReductionModel):
    def init_state(self, context, batch_size=None):
        del context, batch_size


class _Minimal(braincell.ReductionModel):
    def init_state(self, context, batch_size=None):
        del context, batch_size

    def update(self, inputs):
        del inputs

    def reset_state(self, batch_size=None):
        del batch_size


class ReductionModelTest(unittest.TestCase):
    def test_lifecycle_methods_are_abstract(self) -> None:
        with self.assertRaises(TypeError):
            _InitOnly()

    def test_optional_methods_have_defaults(self) -> None:
        model = _Minimal()
        message = r"^_Minimal does not expose population parameters\.$"

        self.assertIsNone(model.reset())
        with self.assertRaisesRegex(NotImplementedError, message):
            model.get("x", (0,))
        with self.assertRaisesRegex(NotImplementedError, message):
            model.set((0,), x=1)


class ReductionViewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = _cell(3)
        self.model = braincell.EventAccumulatorReduction()
        self.view = self.cell.add_reduction("acc", self.model)

    def test_view_describes_its_registration_and_selection(self) -> None:
        self.assertEqual(self.view.name, "acc")
        self.assertIs(self.view.model, self.model)
        self.assertEqual(self.view.population_indices, (0, 1, 2))
        self.assertEqual(
            repr(self.view),
            "ReductionView(name='acc', model=EventAccumulatorReduction, population_indices=(0, 1, 2), selected=False)",
        )

        self.cell.use_model("acc")
        self.assertTrue(self.view.is_selected)
        self.assertTrue(repr(self.view).endswith("selected=True)"))
        self.cell.use_model()
        self.assertFalse(self.view.is_selected)

    def test_sub_view_sets_only_its_members(self) -> None:
        sub = self.cell[1:].reductions["acc"]

        self.assertEqual(sub.population_indices, (1, 2))
        self.assertIs(sub.set(alpha=0.5), sub)
        np.testing.assert_allclose(self.cell.reductions["acc"].get("alpha"), [0.0, 0.5, 0.5])

    def test_set_without_parameters_returns_the_view(self) -> None:
        self.assertIs(self.view.set(), self.view)
        np.testing.assert_allclose(self.view.get("alpha"), [0.0, 0.0, 0.0])

    def test_get_requires_a_non_empty_name(self) -> None:
        for field in ("", None):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "non-empty string"):
                self.view.get(field)

    def test_collection_lists_registered_models(self) -> None:
        reductions = self.cell.reductions

        self.assertEqual(len(reductions), 1)
        self.assertEqual(list(reductions), ["acc"])
        self.assertIn("acc", reductions)
        self.assertNotIn("nope", reductions)
        self.assertEqual(repr(reductions), "ReductionViewCollection(names=('acc',), population_indices=(0, 1, 2))")
        self.assertEqual(
            repr(self.cell[1:].reductions), "ReductionViewCollection(names=('acc',), population_indices=(1, 2))"
        )
        with self.assertRaisesRegex(KeyError, r"available models: \('acc',\)"):
            _ = reductions["nope"]

    def test_set_after_init_state_asks_for_reset_and_reset_recovers(self) -> None:
        self.cell.use_model("acc")
        self.cell.init_state()

        with self.assertRaisesRegex(RuntimeError, r"call reset\(\) first"):
            self.view.set(alpha=0.1)
        np.testing.assert_allclose(self.view.get("alpha"), [0.0, 0.0, 0.0])

        self.cell.reset()
        self.view.set(alpha=0.1)
        np.testing.assert_allclose(self.view.get("alpha"), [0.1, 0.1, 0.1])
        self.cell.init_state()
        self.assertEqual(self.cell.outputs["value"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
