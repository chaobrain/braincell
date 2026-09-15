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

import numpy as np
import brainunit as u
import brainstate
import jax
import jax.numpy as jnp

import braincell
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology
from braincell._discretization.policy import CVPerBranch
from braincell._compute._testing import _build_tree
from braincell.experimental.optim.gradients import build_trajectory_value_and_grad
from braincell.trainable import scale
from validation.optim.nonlinear_pattern_separation.runners.braincell.geometry_checks import compare_finite_difference, recover_one_parameter


def _cell():
    branch = Branch.from_lengths(
        lengths=[20.0, 20.0] * u.um,
        radii=[5.0, 4.0, 3.0] * u.um,
        type="soma",
    )
    cell = braincell.Cell(Morphology.from_root(branch, name="soma"), cv_policy=CVPerBranch(), V_init=-65.0 * u.mV)
    cell.init_state()
    return cell


def test_cable_gradient_matches_central_finite_difference():
    cell = _cell()
    analytic, finite = compare_finite_difference(
        cell.runtime.reference_cable,
        cell.runtime.reference_ra,
        np.asarray([1.1, 0.9, 1.2, 1.3]),
        step=1e-3,
    )
    np.testing.assert_allclose(analytic, finite, rtol=5e-3, atol=2e-6)


def test_runtime_geometry_changes_values_without_changing_topology():
    cell = _cell()
    n_cv = cell.n_cv
    topology = cell.node_tree
    area = cell.runtime.cable.area
    cell.geometry.length.set(cell.geometry.length.get() * 1.2)
    cell.geometry.radius_scale.set(0.8)
    assert cell.n_cv == n_cv
    assert cell.node_tree is topology
    assert not np.allclose(
        np.asarray(cell.runtime.cable.area.to_decimal(u.cm**2)),
        np.asarray(area.to_decimal(u.cm**2)),
    )


def test_four_geometry_groupings_register():
    for group in ("row", "cv", "population", "all"):
        cell = _cell()
        cell.geometry.length.trainable(scale(group_by=group))
        assert cell.trainables.bindings()[0].group_by == group


def _rollout_engine(method):
    cell = braincell.Cell(_build_tree(), V_init=-65.0 * u.mV, solver="staggered")
    cell.init_state()
    cell.geometry.length.trainable(scale(group_by="all", name="geometry.length"))

    def step(data):
        time_ms, _target = data
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return cell.V.value.to_decimal(u.mV)

    def loss(observations, data):
        _times, target = data
        return jnp.mean((observations - target) ** 2)

    return cell, build_trajectory_value_and_grad(cell, step=step, loss=loss, method=method)


def test_geometry_rollout_bptt_and_rtrl_align_with_finite_difference():
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        times = jnp.arange(4, dtype=jnp.float64) * 0.025
        target = jnp.full((4, 1, 1), -60.0, dtype=jnp.float64)
        data = (times, target)
        bptt_cell, bptt = _rollout_engine("bptt")
        rtrl_cell, rtrl = _rollout_engine("rtrl")
        bptt_result = bptt(data)
        rtrl_result = rtrl(data)
        name = next(iter(bptt_result.gradients))
        np.testing.assert_allclose(rtrl_result.loss, bptt_result.loss, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(rtrl_result.gradients[name], bptt_result.gradients[name], rtol=1e-7, atol=1e-9)

        rtrl.prepare((times[0], target[0]))
        root = tuple(state.value for state in rtrl.parameter_states.values())
        epsilon = 1e-4

        def loss_at(value):
            observations = rtrl._observation_rollout((value,), data)
            return rtrl.loss(observations, data)

        finite = (loss_at(root[0] + epsilon) - loss_at(root[0] - epsilon)) / (2.0 * epsilon)
        np.testing.assert_allclose(rtrl_result.gradients[name], finite, rtol=2e-5, atol=2e-7)
        parameters = bptt_cell.trainables.parameters()
        before = bptt_result.loss
        current = parameters.states()[name].value
        parameters.set_physical_values({name: current - 0.01 * bptt_result.gradients[name]})
        after = bptt(data).loss
        # A very short rollout can be locally flat in the geometry direction;
        # the finite-difference and BPTT/RTRL checks above are the correctness
        # criterion, while an optimizer step must not increase this loss.
        assert float(after) <= float(before) + 1e-10
        assert bptt_cell.n_cv == rtrl_cell.n_cv


def test_synthetic_recovery_covers_all_four_geometry_parameters():
    cell = _cell()
    true_values = np.asarray([1.25, 0.75, 1.3, 1.4], dtype=float)
    initial_values = (0.8, 1.2, 0.7, 0.7)
    for index, initial in enumerate(initial_values):
        recovered, final_loss = recover_one_parameter(
            cell.runtime.reference_cable,
            cell.runtime.reference_ra,
            index=index,
            initial=initial,
            target=true_values,
            steps=120,
            learning_rate=0.01 if index == 1 else 0.15,
        )
        assert float(final_loss) < 1e-8
        np.testing.assert_allclose(recovered[index], true_values[index], rtol=3e-3, atol=3e-3)


def _voltage_cell():
    return braincell.Cell(
        _build_tree(),
        V_init=jnp.asarray([[-65.0, -50.0]], dtype=jnp.float64) * u.mV,
        solver="euler",
    )


def _synthetic_voltage_data(cell, steps=4):
    values = []
    for index in range(steps):
        with brainstate.environ.context(t=index * 0.025 * u.ms):
            cell.update()
        values.append(cell.V.value.to_decimal(u.mV))
    return jnp.arange(steps, dtype=jnp.float64) * 0.025, jnp.stack(values)


def _voltage_recovery_engine(cell, method):
    def step(data):
        time_ms, _target = data
        with brainstate.environ.context(t=time_ms * u.ms):
            cell.update()
        return cell.V.value.to_decimal(u.mV)

    def loss(observations, data):
        return jnp.mean((observations - data[1]) ** 2)

    return build_trajectory_value_and_grad(cell, step=step, loss=loss, method=method)


def test_synthetic_voltage_recovery_lowers_loss_for_all_four_parameters():
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        true_values = (1.15, 0.85, 1.2, 1.3)
        initial_values = (0.85, 1.15, 0.8, 0.8)
        for field, true_value, initial_value in zip(
            ("length", "radius_scale", "Ra", "cm"), true_values, initial_values
        ):
            true_cell = _voltage_cell()
            true_cell.init_state()
            true_cell.geometry.__getattr__(field).set(
                true_value if field == "radius_scale" else true_value *
                (u.um if field == "length" else u.ohm * u.cm if field == "Ra" else u.uF / u.cm**2)
            )
            data = _synthetic_voltage_data(true_cell)

            learner = _voltage_cell()
            learner.init_state()
            unit = (u.UNITLESS if field == "radius_scale" else
                    u.um if field == "length" else u.ohm * u.cm if field == "Ra" else u.uF / u.cm**2)
            learner.geometry.__getattr__(field).set(initial_value if unit is u.UNITLESS else initial_value * unit)
            learner.geometry.__getattr__(field).trainable(scale(group_by="all", name=f"synthetic.{field}"))
            engine = _voltage_recovery_engine(learner, "bptt")
            before = engine(data)
            parameters = learner.trainables.parameters()
            name = next(iter(before.gradients))
            current = parameters.states()[name].value
            parameters.set_physical_values({name: current - 1e-3 * before.gradients[name]})
            after = engine(data)
            assert float(after.loss) < float(before.loss), field


def test_multistep_synthetic_voltage_recovery_converges_and_plots(tmp_path):
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        true_values = (1.15, 0.85, 1.2, 1.3)
        initial_values = (0.85, 1.15, 0.8, 0.8)
        histories = {}
        for field, true_value, initial_value in zip(
            ("length", "radius_scale", "Ra", "cm"), true_values, initial_values
        ):
            true_cell = _voltage_cell()
            true_cell.init_state()
            field_view = true_cell.geometry.__getattr__(field)
            if field == "radius_scale":
                field_view.set(true_value)
            elif field == "length":
                field_view.set(true_value * u.um)
            elif field == "Ra":
                field_view.set(true_value * u.ohm * u.cm)
            else:
                field_view.set(true_value * u.uF / u.cm**2)
            data = _synthetic_voltage_data(true_cell, steps=6)

            learner = _voltage_cell()
            learner.init_state()
            field_view = learner.geometry.__getattr__(field)
            if field == "radius_scale":
                field_view.set(initial_value)
            elif field == "length":
                field_view.set(initial_value * u.um)
            elif field == "Ra":
                field_view.set(initial_value * u.ohm * u.cm)
            else:
                field_view.set(initial_value * u.uF / u.cm**2)
            field_view.trainable(scale(group_by="all", name=f"convergence.{field}"))
            engine = _voltage_recovery_engine(learner, "bptt")
            parameters = learner.trainables.parameters()
            losses = []
            for _ in range(12):
                result = engine(data)
                losses.append(float(result.loss))
                name = next(iter(result.gradients))
                current = parameters.states()[name].value
                parameters.set_physical_values({name: current - 1e-3 * result.gradients[name]})
            histories[field] = losses
            assert losses[-1] < losses[0], field

        figure, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        for axis, (field, losses) in zip(axes.flat, histories.items()):
            axis.plot(np.arange(1, len(losses) + 1), losses, marker="o")
            axis.set_title(field)
            axis.set_xlabel("gradient step")
            axis.set_ylabel("voltage MSE")
            axis.set_yscale("log")
            axis.grid(True, alpha=0.3)
        output_dir = os.environ.get("BRAINCELL_GEOMETRY_ARTIFACT_DIR")
        if output_dir:
            output = __import__("pathlib").Path(output_dir) / "geometry_loss_convergence.png"
            output.parent.mkdir(parents=True, exist_ok=True)
        else:
            output = tmp_path / "geometry_loss_convergence.png"
        figure.savefig(output, dpi=140)
        plt.close(figure)
        assert output.exists() and output.stat().st_size > 0
