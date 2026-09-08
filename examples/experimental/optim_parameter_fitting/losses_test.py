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

import jax.numpy as jnp
import jax
import numpy as np

from examples.experimental.optim_parameter_fitting.losses import (
    protocol_balanced_voltage_mse,
    protocol_balanced_huber,
    raw_voltage_mse,
)


def test_raw_voltage_mse_has_identical_training_and_split_reduction() -> None:
    loss = raw_voltage_mse()
    target = jnp.zeros((2, 3, 1))
    prediction = jnp.asarray([target + 1.0, target + 2.0])

    total, raw, protocol = loss.evaluate(prediction, target)
    local_sum = sum(loss.local(prediction[0, :, step], target[:, step], num_steps=3) for step in range(3))

    np.testing.assert_allclose(total, [1.0, 4.0])
    np.testing.assert_allclose(raw, total)
    np.testing.assert_allclose(protocol, [[1.0, 1.0], [4.0, 4.0]])
    np.testing.assert_allclose(local_sum, total[0])


def test_protocol_balanced_weights_are_mean_one_and_use_std_floor() -> None:
    loss = protocol_balanced_voltage_mse(std_floor_mv=5.0)
    target = jnp.asarray([[[0.0], [20.0]], [[0.0], [2.0]]])

    weights = loss.prepare(target)

    np.testing.assert_allclose(np.mean(weights), 1.0)
    assert weights[1] > weights[0]


def test_mse_normalized_huber_matches_mse_inliers_and_is_continuous() -> None:
    loss = protocol_balanced_huber(delta_mv=5.0, std_floor_mv=5.0)

    def penalty(error):
        prediction = jnp.asarray([[[[error]]]])
        target = jnp.zeros((1, 1, 1))
        return loss.evaluate(prediction, target)[0][0]

    np.testing.assert_allclose(penalty(2.0), 4.0)
    np.testing.assert_allclose(jax.grad(penalty)(2.0), 4.0)
    np.testing.assert_allclose(penalty(5.0), 25.0)
    np.testing.assert_allclose(penalty(5.0 + 1e-5), 25.0 + 1e-4, rtol=1e-5)
    np.testing.assert_allclose(jax.grad(penalty)(6.0), 10.0)
