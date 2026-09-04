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
import jax.numpy as jnp
import numpy as np

from examples.experimental.optim_parameter_fitting.optimizers import adam, optax_rprop, rprop, sgd


def test_adam_stage_builds_fresh_optimizer_state_for_each_stage() -> None:
    stage = adam(epochs=2, checkpoint_every=1)
    first_state = {"theta": brainstate.ParamState(jnp.zeros((2,)))}
    second_state = {"theta": brainstate.ParamState(jnp.zeros((2,)))}

    first = stage.build_optimizer(first_state)
    second = stage.build_optimizer(second_state)

    assert first is not second
    assert stage.describe()["gradient_method"] == "rtrl"
    assert stage.describe()["gradient_clipping"] is None


def test_rprop_stage_exposes_resilient_step_configuration() -> None:
    stage = rprop(epochs=2, checkpoint_every=1)
    states = {"theta": brainstate.ParamState(jnp.zeros((2,)))}

    optimizer = stage.build_optimizer(states)

    assert optimizer.etas == (0.5, 1.2)
    assert optimizer.step_sizes == (1e-6, 50.0)
    assert stage.describe()["optimizer"] == "Rprop"


def test_optax_rprop_applies_learning_rate_exactly_once() -> None:
    stage = optax_rprop(epochs=2, checkpoint_every=1, learning_rate=1e-4)
    state = brainstate.ParamState(jnp.asarray([0.0]))
    optimizer = stage.build_optimizer({"theta": state})

    optimizer.update({"theta": jnp.asarray([1.0])})
    first = float(state.value[0])
    optimizer.update({"theta": jnp.asarray([1.0])})
    second = float(state.value[0])

    assert first == 0.0
    assert abs(second + 1e-4) < 1e-10
    assert stage.describe()["lr_application"] == "single_in_custom_tx"


def test_validation_every_is_clear_alias_for_legacy_checkpoint_name() -> None:
    stage = optax_rprop(epochs=20, validation_every=5)

    assert stage.validation_every == 5
    assert stage.checkpoint_every == 5

    try:
        optax_rprop(epochs=20, validation_every=5, checkpoint_every=10)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Conflicting validation cadence aliases must fail.")


def test_vanilla_sgd_applies_gradient_once_without_momentum() -> None:
    stage = sgd(epochs=2, validation_every=1, learning_rate=1e-4)
    state = brainstate.ParamState(jnp.asarray([1.0, -2.0]))
    optimizer = stage.build_optimizer({"theta": state})

    optimizer.update({"theta": jnp.asarray([3.0, -4.0])})

    np.testing.assert_allclose(state.value, [1.0 - 3e-4, -2.0 + 4e-4], rtol=0.0, atol=1e-7)
    assert stage.describe()["momentum"] == 0.0
    assert stage.describe()["nesterov"] is False
    assert stage.describe()["weight_decay"] == 0.0


def test_momentum_and_nesterov_match_optax_trace_updates() -> None:
    gradient = {"theta": jnp.asarray([2.0])}
    outputs = {}
    for name, nesterov in (("momentum", False), ("nesterov", True)):
        stage = sgd(
            epochs=2,
            validation_every=1,
            learning_rate=1e-4,
            momentum=0.9,
            nesterov=nesterov,
        )
        state = brainstate.ParamState(jnp.asarray([0.0]))
        optimizer = stage.build_optimizer({"theta": state})
        optimizer.update(gradient)
        first = float(state.value[0])
        optimizer.update(gradient)
        outputs[name] = (first, float(state.value[0]))

    np.testing.assert_allclose(outputs["momentum"], [-2e-4, -5.8e-4], atol=1e-10)
    np.testing.assert_allclose(outputs["nesterov"], [-3.8e-4, -9.22e-4], atol=1e-10)
