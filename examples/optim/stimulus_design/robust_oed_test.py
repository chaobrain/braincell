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

from __future__ import annotations

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.optim.stimulus_design.dataset import (
    N_STEPS,
    PARAMETER_NAMES,
    build_cell,
    simulate_voltage,
)
from examples.optim.stimulus_design.robust_oed import (
    ObservationInformationEngine,
    prior_scales,
    robust_greedy_order,
    scales_to_raw_roots,
)


def test_prior_scales_are_reproducible_target_led_and_strictly_bounded() -> None:
    first = prior_scales()
    second = prior_scales()

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], np.ones((6,)))
    assert first.shape == (17, 6)
    assert np.all((first[1:] > 0.5) & (first[1:] < 1.5))


def test_robust_order_is_stable_and_excludes_invalid_protocol() -> None:
    information = np.zeros((2, 4, 6, 6))
    information[:, 0] = np.diag([4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    information[:, 1] = np.diag([0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
    information[:, 2] = np.eye(6)
    information[:, 3] = np.nan

    ordering, metrics = robust_greedy_order(information, ("a", "b", "c", "invalid"))

    assert ordering.tolist() == [2, 1, 0]
    assert metrics["invalid_indices"].tolist() == [3]
    assert metrics["worst_rank"][-1] == 6
    assert np.isfinite(metrics["minimum_eigenvalue"][-1])


def test_online_information_matches_log_scale_directional_finite_difference() -> None:
    current = np.zeros((1, N_STEPS, 3))
    current[:, 1:3, 0] = 0.02
    direction = np.asarray((0.3, -0.2, 0.1, -0.4, 0.25, -0.15))
    epsilon = 1e-4
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        engine = ObservationInformationEngine(current, num_steps=3)
        information = engine.information(np.ones((1, 6)))[0, 0]
        cell = build_cell(current, trainable=True)
        parameters = cell.trainables.parameters()

        def voltage(log_step):
            scales = np.exp(log_step * direction)
            parameters.set_physical_values({name: scales[index] for index, name in enumerate(PARAMETER_NAMES)})
            return np.asarray(simulate_voltage(cell, num_steps=3))

        finite_difference = (voltage(epsilon) - voltage(-epsilon)) / (2.0 * epsilon)

    expected = np.mean(finite_difference * finite_difference)
    actual = float(direction @ information @ direction)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-8)


def _assembly_scatters(closed, n_point) -> dict[str, int]:
    """``(n_point, n_point)`` scatter-adds of the axial operator's assembly, in and outside loops."""
    counts = {"loop": 0, "outside": 0}

    def walk(jaxpr, looped):
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            if name == "scatter-add" and tuple(eqn.outvars[0].aval.shape[-2:]) == (n_point, n_point):
                counts["loop" if looped else "outside"] += 1
            for value in eqn.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(item, jax.extend.core.ClosedJaxpr):
                        walk(item.jaxpr, looped or name in ("scan", "while"))
                    elif isinstance(item, jax.extend.core.Jaxpr):
                        walk(item, looped or name in ("scan", "while"))

    walk(closed.jaxpr, False)
    return counts


def test_information_scan_builds_the_axial_operator_once() -> None:
    # OPT-14: the staggered cable's DHS coefficients are built before the
    # scan, not at every step of it.
    current = np.zeros((1, N_STEPS, 3))
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        engine = ObservationInformationEngine(current, num_steps=3)
        scales = jax.numpy.ones((6,))
        closed = jax.make_jaxpr(engine._one_reference)(scales_to_raw_roots(scales), scales)
    counts = _assembly_scatters(closed, engine.cell._runtime.n_point)
    assert counts["loop"] == 0, counts
    assert counts["outside"] > 0, counts
    assert engine.engine.axial_operator_mode == "rollout"

