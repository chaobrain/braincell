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

"""Small pure-cable gradient alignment problem."""

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._compute.cable import CableArrays


def cable_loss(reference: CableArrays, reference_ra, values):
    """Return a scalar loss through all four geometry controls."""
    length, radius, ra, cm = values
    ra_ratio = ra / reference_ra.to_decimal(u.ohm * u.cm)
    cable = CableArrays(
        area=reference.area * (length * radius),
        cm=u.Quantity(cm, u.uF / u.cm**2),
        resistance_prox=reference.resistance_prox * (length * ra_ratio / radius**2),
        resistance_dist=reference.resistance_dist * (length * ra_ratio / radius**2),
    )
    area = jnp.sum(cable.area.to_decimal(u.cm**2))
    capacitance = jnp.sum((cable.area * cable.cm).to_decimal(u.uF))
    resistance = jnp.sum(cable.resistance_prox.to_decimal(u.ohm))
    return area + 0.01 * capacitance + 1e-6 * resistance


def compare_finite_difference(reference, reference_ra, values, *, step=1e-5):
    values = np.asarray(values, dtype=float)
    analytic = np.asarray(jax.grad(lambda x: cable_loss(reference, reference_ra, x))(jnp.asarray(values)))
    finite = np.zeros_like(values)
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        finite[index] = (
            float(cable_loss(reference, reference_ra, plus))
            - float(cable_loss(reference, reference_ra, minus))
        ) / (2.0 * step)
    return analytic, finite


def cable_observation(reference: CableArrays, reference_ra, values):
    """Synthetic measurements used for the four-parameter recovery test."""
    length, radius, ra, cm = values
    ra_ratio = ra / reference_ra.to_decimal(u.ohm * u.cm)
    area = reference.area.to_decimal(u.cm**2) * length * radius
    capacitance = area * cm
    resistance = reference.resistance_prox.to_decimal(u.ohm) * length * ra_ratio / radius**2
    return jnp.concatenate((area, capacitance, resistance))


def cable_observation_scales(reference: CableArrays, reference_ra, scales):
    """Synthetic observation using dimensionless scales for stable optimization."""
    length, radius, ra_scale, cm_scale = scales
    ra = reference_ra.to_decimal(u.ohm * u.cm) * ra_scale
    cm = reference.cm.to_decimal(u.uF / u.cm**2) * cm_scale
    observation = cable_observation(reference, reference_ra, (length, radius, ra, cm))
    area_size = reference.area.to_decimal(u.cm**2).size
    reference_observation = cable_observation(
        reference,
        reference_ra,
        (jnp.asarray(1.0), jnp.asarray(1.0), reference_ra.to_decimal(u.ohm * u.cm), reference.cm.to_decimal(u.uF / u.cm**2)),
    )
    # The three concatenated blocks have different physical magnitudes. Scale
    # them to order-one synthetic measurements before gradient descent.
    del area_size
    return observation / (jnp.abs(reference_observation) + 1e-12)


def recover_one_parameter(reference, reference_ra, *, index, initial, target, steps=80, learning_rate=0.2):
    """Recover one synthetic geometry control while holding the others fixed."""
    true_values = jnp.asarray(target)
    values = true_values.at[index].set(initial)
    target_observation = cable_observation_scales(reference, reference_ra, true_values)

    def loss(parameter_values):
        residual = cable_observation_scales(reference, reference_ra, parameter_values) - target_observation
        return jnp.mean(residual * residual)

    gradient = jax.grad(loss)
    for _ in range(steps):
        update = gradient(values).at[jnp.arange(values.size) != index].set(0.0)
        values = values - learning_rate * update
    return values, loss(values)
