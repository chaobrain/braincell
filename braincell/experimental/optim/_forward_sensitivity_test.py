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

"""Tests for the experimental exact forward-sensitivity core."""

import os
import unittest
from unittest.mock import patch

import braincell
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute._testing import _build_tree
from braincell.filter import AllRegion, RootLocation

from braincell.experimental.optim._forward_sensitivity import (
    ActiveStateSelection,
    _align_state_values,
    bptt_reference_loss,
    build_active_state_projection,
    build_stateful_step,
    compact_forward_sensitivity_rollout,
    compact_forward_sensitivity_step,
    forward_sensitivity_rollout,
    forward_sensitivity_step,
    initialize_forward_sensitivity,
    inspect_state_sensitivity,
    initialized_bptt_reference_loss,
    seed_scalar_parameter_directions,
    select_parameter_derivatives,
)
from braincell.experimental.optim.gradients import build_rollout_value_and_grad


def _leak_cell():
    cell = braincell.Cell(_build_tree(), pop_size=(1,))
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "IL",
            name="leak",
            g_max=0.3 * u.mS / u.cm**2,
            E=-54.3 * u.mV,
        ),
    )
    cell.init_state()
    cell.channels["leak"].trainable(g_max=braincell.trainable.scale(group_by="all", name="leak.factor"))
    cell.reset_state()
    return cell


def _hh_cell():
    cell = braincell.Cell(_build_tree(), pop_size=(1,))
    cell.paint(
        AllRegion(), braincell.mech.Ion("SodiumFixed", name="na"), braincell.mech.Ion("PotassiumFixed", name="k")
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "Na_HH1952",
            name="na",
            g_max=12.0 * u.mS / u.cm**2,
        ),
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "K_HH1952",
            name="k",
            g_max=3.6 * u.mS / u.cm**2,
        ),
    )
    cell.paint(
        AllRegion(),
        braincell.mech.Channel(
            "IL",
            name="leak",
            g_max=0.3 * u.mS / u.cm**2,
            E=-54.3 * u.mV,
        ),
    )
    cell.init_state()
    cell.channels["na"].trainable(g_max=braincell.trainable.scale(group_by="all", name="na.factor"))
    cell.channels["k"].trainable(g_max=braincell.trainable.scale(group_by="all", name="k.factor"))
    cell.reset_state()
    return cell


def _functional_voltage_loss(cell, *, example_target):
    def local_loss(target_mv):
        cell.trainables.materialize()
        cell.update()
        error = cell.V.value.to_decimal(u.mV) - target_mv
        return jnp.mean(error * error)

    return build_stateful_step(
        local_loss,
        example_target,
        cell.trainables.parameters().states(),
    )


class ForwardSensitivityCoreTest(unittest.TestCase):
    def test_state_sensitivity_inventory_marks_parameter_and_float0_leaves(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _leak_cell()
            target = jnp.asarray(-60.0)
            step = _functional_voltage_loss(cell, example_target=target)
            values = step.state_values()
            tangents = seed_scalar_parameter_directions(step, values)
            entries = inspect_state_sensitivity(step, values, tangents)
        self.assertEqual(len(entries), len(values))
        self.assertTrue(all(entry.tangent_bytes >= 0 for entry in entries))
        self.assertTrue(all(entries[index].is_parameter for index in step.parameter_indices))
        self.assertIn(False, {entry.has_float0_tangent for entry in entries})

    def test_leak_prefix_gradients_match_reverse_mode_and_finite_difference(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _leak_cell()
            target = jnp.asarray(-60.0, dtype=jnp.float64)

            def weighted_local_loss(data):
                target_mv, loss_weight = data
                cell.trainables.materialize()
                cell.update()
                error = cell.V.value.to_decimal(u.mV) - target_mv
                return loss_weight * jnp.mean(error * error)

            step = build_stateful_step(
                weighted_local_loss,
                (target, jnp.asarray(1.0, dtype=jnp.float64)),
                cell.trainables.parameters().states(),
            )
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            targets = jnp.full((8,), target)
            step_data = (targets, jnp.ones((8,), dtype=jnp.float64))

            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, step_data)
            reverse_state_gradient = jax.grad(
                lambda values: bptt_reference_loss(step, values, step_data), allow_int=True
            )(initial_values)
            reverse = select_parameter_derivatives(step, reverse_state_gradient)["leak.factor"]

            def reverse_prefix(loss_weights):
                derivative = jax.grad(
                    lambda values: bptt_reference_loss(step, values, (targets, loss_weights)), allow_int=True
                )(initial_values)
                return select_parameter_derivatives(step, derivative)["leak.factor"]

            reverse_prefixes = jax.vmap(reverse_prefix)(jnp.tril(jnp.ones((8, 8), dtype=jnp.float64)))

            epsilon = 1e-4
            parameter_index = step.parameter_indices[0]

            def shifted_loss(delta):
                shifted = list(initial_values)
                shifted[parameter_index] = shifted[parameter_index] + delta
                return bptt_reference_loss(step, tuple(shifted), step_data)

            finite_difference = (shifted_loss(epsilon) - shifted_loss(-epsilon)) / (2.0 * epsilon)

        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward.prefix_gradients[:, 0], reverse_prefixes, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forward.final_gradient[0], finite_difference, rtol=1e-6, atol=1e-8)
        self.assertEqual(forward.final_gradient.shape, (1,))
        self.assertEqual(forward.prefix_gradients.shape, (8, 1))
        self.assertEqual(forward.local_gradients.shape, (8, 1))
        for tangent, value in zip(
            jax.tree.leaves(forward.final_state_tangents),
            jax.tree.leaves(forward.final_state_values),
        ):
            self.assertEqual(np.shape(tangent), (1,) + np.shape(value))
        np.testing.assert_allclose(
            forward.prefix_gradients,
            np.cumsum(np.asarray(forward.local_gradients), axis=0),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_hh_multiple_parameter_directions_match_reverse_mode(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = _hh_cell()
            target = jnp.asarray(-62.0, dtype=jnp.float64)
            step = _functional_voltage_loss(cell, example_target=target)
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            targets = jnp.full((4,), target)

            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, targets)
            reverse_state_gradient = jax.grad(
                lambda values: bptt_reference_loss(step, values, targets), allow_int=True
            )(initial_values)
            selected = select_parameter_derivatives(step, reverse_state_gradient)
            reverse = jnp.stack([selected[name] for name in step.parameter_names])

        np.testing.assert_allclose(forward.final_gradient, reverse, rtol=1e-9, atol=1e-10)
        self.assertEqual(forward.final_gradient.shape, (2,))
        self.assertEqual(forward.prefix_gradients.shape, (4, 2))

    def test_parameter_dependent_reset_is_included_in_initial_sensitivity(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(AllRegion(), braincell.mech.Ion("SodiumFixed", name="na"))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel("Na_HH1952", name="na", g_max=12.0 * u.mS / u.cm**2),
            )
            cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            cell.init_state()
            cell.channels["na"].trainable(V_sh=braincell.trainable.scale(group_by="all", name="na.vsh.factor"))
            cell.reset_state()
            parameters = cell.trainables.parameters().states()

            def reset_and_zero(_):
                cell.reset_state()
                return jnp.asarray(0.0, dtype=jnp.float64)

            initializer = build_stateful_step(reset_and_zero, None, parameters)
            target = jnp.asarray(-62.0, dtype=jnp.float64)
            step = _functional_voltage_loss(cell, example_target=target)
            initial_values, initial_tangents = initialize_forward_sensitivity(initializer, step, None)
            targets = jnp.full((3,), target)
            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, targets)

            initializer_values = initializer.state_values()
            reverse_initializer_gradient = jax.grad(
                lambda values: initialized_bptt_reference_loss(
                    initializer,
                    step,
                    values,
                    None,
                    targets,
                ),
                allow_int=True,
            )(initializer_values)
            reverse = select_parameter_derivatives(initializer, reverse_initializer_gradient)["na.vsh.factor"]

            naive_values = step.state_values()
            naive_tangents = seed_scalar_parameter_directions(step, naive_values)
            naive = forward_sensitivity_rollout(step, naive_values, naive_tangents, targets)

        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-9, atol=1e-10)
        self.assertGreater(float(jnp.abs(forward.final_gradient[0] - naive.final_gradient[0])), 1e-6)

    def test_fixed_delayed_external_clamps_remain_exact_scan_inputs(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            cell.place(
                RootLocation(x=0.0),
                braincell.mech.CurrentClamp(
                    delay=0.025 * u.ms,
                    durations=0.05 * u.ms,
                    amplitudes=0.2 * u.nA,
                ),
            )
            cell.place(
                RootLocation(x=1.0),
                braincell.mech.CurrentClamp(
                    delay=0.075 * u.ms,
                    durations=0.05 * u.ms,
                    amplitudes=0.1 * u.nA,
                ),
            )
            cell.init_state()
            cell.channels["leak"].trainable(g_max=braincell.trainable.scale(group_by="all", name="leak.factor"))
            cell.reset_state()
            target = jnp.asarray(-60.0, dtype=jnp.float64)

            def local_loss(data):
                time_ms, target_mv = data
                with brainstate.environ.context(t=time_ms * u.ms):
                    cell.trainables.materialize()
                    cell.update()
                error = cell.V.value.to_decimal(u.mV) - target_mv
                return jnp.mean(error * error)

            example_data = (jnp.asarray(0.0, dtype=jnp.float64), target)
            step = build_stateful_step(
                local_loss,
                example_data,
                cell.trainables.parameters().states(),
            )
            initial_values = step.state_values()
            initial_tangents = seed_scalar_parameter_directions(step, initial_values)
            step_data = (
                jnp.arange(6, dtype=jnp.float64) * 0.025,
                jnp.full((6,), target),
            )
            forward = forward_sensitivity_rollout(step, initial_values, initial_tangents, step_data)
            reverse_state_gradient = jax.grad(
                lambda values: bptt_reference_loss(step, values, step_data), allow_int=True
            )(initial_values)
            reverse = select_parameter_derivatives(step, reverse_state_gradient)["leak.factor"]

            clamp_delays = sorted(
                float(delay)
                for layout in cell.layouts
                if layout.kind == "CurrentClamp"
                for delay in np.asarray(cell.get_state(layout.id, "delay").to_decimal(u.ms)).reshape(-1)
            )

        np.testing.assert_allclose(clamp_delays, [0.025, 0.075], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(forward.final_gradient[0], reverse, rtol=1e-9, atol=1e-10)
        self.assertTrue(bool(jnp.all(jnp.isfinite(forward.local_gradients))))

    def test_requires_scalar_local_loss_and_scalar_coordinate_roots(self) -> None:
        with brainstate.environ.context(dt=0.025 * u.ms):
            cell = _leak_cell()
            parameter_states = cell.trainables.parameters().states()

            def vector_loss(_):
                cell.trainables.materialize()
                return jnp.ones((2,))

            with self.assertRaisesRegex(ValueError, "scalar local loss"):
                build_stateful_step(vector_loss, None, parameter_states)

            vector_cell = braincell.Cell(_build_tree(), pop_size=(2,))
            vector_cell.paint(
                AllRegion(),
                braincell.mech.Channel(
                    "IL",
                    name="leak",
                    g_max=0.3 * u.mS / u.cm**2,
                    E=-54.3 * u.mV,
                ),
            )
            vector_cell.init_state()
            vector_cell.channels["leak"].trainable(
                g_max=braincell.trainable.scale(group_by="population", name="leak.population.factor")
            )
            vector_cell.reset_state()
            vector_step = _functional_voltage_loss(vector_cell, example_target=jnp.asarray(-60.0))
            with self.assertRaisesRegex(ValueError, "not scalar"):
                seed_scalar_parameter_directions(vector_step, vector_step.state_values())

    def test_operator_input_steps_match_steps_that_build_their_own(self) -> None:
        # OPT-14: an engine's step takes the axial operator as an input; a
        # rollout builds it once (with its tangents) and passes it to every
        # step. Each way of calling must give the transition of a step that
        # builds the operator itself, in both JVP modes.
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell = braincell.Cell(_build_tree(), pop_size=(1,))
            cell.paint(
                AllRegion(),
                braincell.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
            )
            cell.place(
                RootLocation(0.5),
                braincell.mech.CurrentClamp(delay=0.0 * u.ms, durations=1.0 * u.ms, amplitudes=0.05 * u.nA),
            )
            cell.init_state()
            cell.geometry.Ra.trainable(braincell.trainable.scale(group_by="all", name="Ra"))
            cell.reset_state()

            def step(target_mv):
                cell.update()
                return jnp.mean((cell.V.value.to_decimal(u.mV) - target_mv) ** 2)

            engine = build_rollout_value_and_grad(cell, step=step, method="rtrl").prepare(jnp.asarray(-60.0))
            self.assertEqual(engine.axial_operator_mode, "rollout")
            hoisted = engine._functional_step
            plain = _functional_voltage_loss(cell, example_target=jnp.asarray(-60.0))
            self.assertIsNone(plain.build_operator(plain.state_values()))
            self.assertEqual(plain.build_operator_with_tangents(plain.state_values(), None), (None, None))
            values, tangents = engine._initial_full_carry(tuple(s.value for s in engine.parameter_states.values()))
            data = jnp.asarray(-61.0)

            expected = hoisted.call(values, data)
            operator = hoisted.build_operator(values)
            for actual in (
                hoisted.call_with_operator(values, data, operator),
                plain.call_with_operator(_align_state_values(hoisted, plain, values), data, None),
            ):
                np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12)

            operator, operator_tangents = hoisted.build_operator_with_tangents(values, tangents)
            self.assertGreater(max(float(np.max(np.abs(leaf))) for leaf in jax.tree.leaves(operator_tangents)), 0.0)
            for mode in ("linearize", "direct"):
                with self.subTest(mode=mode), patch.dict(os.environ, {"BRAINCELL_RTRL_JVP_MODE": mode}):
                    built = forward_sensitivity_step(hoisted, values, tangents, data)
                    given = forward_sensitivity_step(hoisted, values, tangents, data, operator, operator_tangents)
                    self.assertNotEqual(float(given[3][0]), 0.0)
                    np.testing.assert_allclose(given[2], built[2], rtol=1e-12)
                    np.testing.assert_allclose(given[3], built[3], rtol=1e-12)


_COMPACT_STEPS = 32


def _compact_hh_cell(field):
    """Two-branch HH cable, ions declared, a 0.1 nA step at the root; one ``scale`` root on ``field``."""
    cell = braincell.Cell(_build_tree(), pop_size=(2,), V_init=-65.0 * u.mV)
    cell.paint(
        AllRegion(),
        braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
        braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
        braincell.mech.Channel("Na_HH1952", name="na", g_max=120.0 * u.mS / u.cm**2),
        braincell.mech.Channel("K_HH1952", name="k", g_max=36.0 * u.mS / u.cm**2),
        braincell.mech.Channel("IL", name="leak", g_max=0.3 * u.mS / u.cm**2, E=-54.3 * u.mV),
    )
    cell.place(
        RootLocation(0.5),
        braincell.mech.CurrentClamp(delay=0.0 * u.ms, durations=50.0 * u.ms, amplitudes=0.1 * u.nA),
    )
    cell.init_state()
    source = braincell.trainable.scale(group_by="all", name=field)
    if field == "Ra":
        cell.geometry.Ra.trainable(source)
    else:
        cell.channels["na"].trainable(g_max=source)
    cell.reset_state()
    return cell


def _compact_case(field, *, materialize_in_step=False):
    """The cell, its BPTT and RTRL rollout engines and the V, m, h, n selections."""
    cell = _compact_hh_cell(field)

    def step(target_mv):
        if materialize_in_step:
            cell.trainables.materialize()
        cell.update()
        return jnp.mean((cell.V.value.to_decimal(u.mV) - target_mv) ** 2)

    example = jnp.asarray(-64.0)
    bptt = build_rollout_value_and_grad(cell, step=step, method="bptt").prepare(example)
    rtrl = build_rollout_value_and_grad(cell, step=step, method="rtrl").prepare(example)

    def node(kind):
        layout = next(layout for layout in cell.layouts if layout.kind == kind)
        return cell.runtime.get_runtime_node(layout.id)

    sodium, potassium = node("channel:Na_HH1952"), node("channel:K_HH1952")
    selections = [
        _select_all("V", cell.V),
        _select_all("m", sodium.p),
        _select_all("h", sodium.q),
        _select_all("n", potassium.p),
    ]
    return cell, bptt, rtrl, sodium, selections


def _select_all(name, state):
    return ActiveStateSelection(name, state, tuple(range(int(np.size(u.get_mantissa(state.value))))))


def _compact_gradient(engine, projection, roots, data):
    values, tangents = engine._initial_full_carry(roots)
    result = compact_forward_sensitivity_rollout(
        engine._functional_step,
        projection,
        engine._parameter_coordinates,
        values,
        projection.extract_tangents(tangents),
        data,
    )
    return result.final_gradient[0]


def _central_difference(engine, roots, data, h=1e-5):
    loss = jax.jit(lambda parameters: engine._bptt(parameters, data).loss)
    return (float(loss((roots[0] + h,))) - float(loss((roots[0] - h,)))) / (2 * h)


def _state_fingerprint(cell, functional_step):
    """Every traced state's value, the step's fields and the cell's attributes, for a refused call."""
    values = tuple(np.asarray(u.get_mantissa(leaf)) for leaf in jax.tree.leaves(functional_step.state_values()))
    return values, tuple(functional_step), dict(vars(cell))


class CompactActiveStateTest(unittest.TestCase):
    """OPT-16: a compact carry must keep the states a trainable parameter reaches the step through."""

    def _assert_fingerprint_unchanged(self, before, after):
        for old, new in zip(before[0], after[0]):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(len(before[0]), len(after[0]))
        self.assertEqual(len(before[1]), len(after[1]))
        for old, new in zip(before[1], after[1]):
            self.assertIs(old, new)
        self.assertEqual(before[2].keys(), after[2].keys())
        for key, value in before[2].items():
            self.assertIs(after[2][key], value, key)

    def test_rollout_materialized_g_max_outside_the_active_states_is_refused(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell, _bptt, rtrl, sodium, selections = _compact_case("g_max")
            step = rtrl._functional_step
            self.assertEqual(rtrl.materialization_mode, "rollout")
            g_max = vars(sodium)["g_max"]
            before = _state_fingerprint(cell, step)
            with self.assertRaises(ValueError) as caught:
                build_active_state_projection(step, selections)
            self._assert_fingerprint_unchanged(before, _state_fingerprint(cell, step))
            message = str(caught.exception)
            self.assertIn("channel:Na_HH1952.g_max", message)
            index = next(i for i, state in enumerate(step.state_trace.states) if state is g_max)
            self.assertIn(f"state_trace.states[{index}]", message)
            self.assertIn("'g_max'", message)
            self.assertEqual(step.materialized_states, ((index, "channel:Na_HH1952.g_max"),))

            # A second call is refused the same way.
            with self.assertRaisesRegex(ValueError, "channel:Na_HH1952.g_max"):
                build_active_state_projection(step, selections)

    def test_each_compact_entry_refuses_a_projection_built_without_the_check(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            cell, _bptt, rtrl, _sodium, selections = _compact_case("g_max")
            step = rtrl._functional_step
            # A projection built against a step that records no materialized
            # state is refused when used with the engine's step.
            projection = build_active_state_projection(step._replace(materialized_states=()), selections)
            roots = tuple(state.value for state in rtrl.parameter_states.values())
            values, tangents = rtrl._initial_full_carry(roots)
            active = projection.extract_tangents(tangents)
            data = jnp.linspace(-64.0, -60.0, 4)
            before = _state_fingerprint(cell, step)
            with self.assertRaises(ValueError) as rollout:
                compact_forward_sensitivity_rollout(step, projection, rtrl._parameter_coordinates, values, active, data)
            with self.assertRaises(ValueError) as one_step:
                compact_forward_sensitivity_step(step, projection, rtrl._parameter_coordinates, values, active, data[0])
            self._assert_fingerprint_unchanged(before, _state_fingerprint(cell, step))
            for caught in (rollout, one_step):
                self.assertIn("channel:Na_HH1952.g_max", str(caught.exception))

    def test_completed_active_states_match_the_central_difference(self) -> None:
        with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
            data = jnp.linspace(-64.0, -60.0, _COMPACT_STEPS)
            cases = {
                # g_max materialized before the rollout, its State selected.
                "g_max rollout": ("g_max", False, True),
                # The step materializes: it reads the root, nothing recorded.
                "g_max step": ("g_max", True, False),
                # Ra is materialized every step on a cable cell.
                "Ra step": ("Ra", False, False),
            }
            for label, (field, materialize_in_step, add_g_max) in cases.items():
                with self.subTest(case=label):
                    _cell, bptt, rtrl, sodium, selections = _compact_case(
                        field, materialize_in_step=materialize_in_step
                    )
                    self.assertEqual(rtrl.materialization_mode, "rollout" if add_g_max else "step")
                    self.assertEqual(bool(rtrl._functional_step.materialized_states), add_g_max)
                    if add_g_max:
                        selections = selections + [_select_all("g_max", vars(sodium)["g_max"])]
                    projection = build_active_state_projection(rtrl._functional_step, selections)
                    roots = tuple(state.value for state in rtrl.parameter_states.values())
                    gradient = float(jax.jit(lambda r: _compact_gradient(rtrl, projection, r, data))(roots))
                    expected = _central_difference(bptt, roots, data)
                    self.assertTrue(np.isfinite(gradient))
                    self.assertNotEqual(expected, 0.0)
                    np.testing.assert_allclose(gradient, expected, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
