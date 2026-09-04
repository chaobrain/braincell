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

import brainstate
import brainunit as u
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.models import (
    CONDUCTANCE_UNIT,
    hh_1cv_classic_bounded_direct,
)


def test_parameter_space_round_trips_physical_normalized_and_z_coordinates() -> None:
    space = hh_1cv_classic_bounded_direct().parameter_space
    physical = np.asarray([[0.2, 90.0, 45.0], [0.4, 150.0, 27.0]])

    with jax.enable_x64(True):
        restored_from_u = space.normalized_to_physical(space.physical_to_normalized(physical))
        restored_from_z = space.z_to_physical(space.physical_to_z(physical))
        quantity_mapping = space.physical_quantities(physical)

    np.testing.assert_allclose(restored_from_u, physical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(restored_from_z, physical, rtol=1e-12, atol=1e-12)
    assert quantity_mapping["na.g_max"].unit == CONDUCTANCE_UNIT


def test_one_cv_cell_owns_three_bounded_direct_parameters_without_scale_roots() -> None:
    model = hh_1cv_classic_bounded_direct()
    with jax.enable_x64(True), brainstate.environ.context(dt=0.025 * u.ms, precision=64):
        cell = model.build_cell(np.zeros((2, 3)), trainable=True)
        parameters = cell.trainables.parameters()
        assert cell.n_cv == 1
        assert tuple(parameters.states()) == model.parameter_space.names
        assert all("scale" not in name for name in parameters.states())
        np.testing.assert_allclose(list(parameters.optimizer_values().values()), 0.0, atol=1e-14)
        for value, target in zip(parameters.physical_values().values(), model.parameter_space.target):
            np.testing.assert_allclose(value.to_decimal(CONDUCTANCE_UNIT), target)
