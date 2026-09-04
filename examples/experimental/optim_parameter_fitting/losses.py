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

"""Loss definitions shared by training and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class LossDefinition:
    """Compute equal-weight raw voltage mean squared error."""

    name: str = "raw_voltage_mse_v1"

    def prepare(self, target_mv):
        """Return fixed protocol weights for this target split."""
        return jnp.ones((jnp.asarray(target_mv).shape[0],))

    def local(self, prediction_mv, target_mv, *, num_steps: int, protocol_weights=None):
        """Return one time step's contribution to total raw MSE."""
        per_protocol = jnp.mean((prediction_mv - target_mv) ** 2, axis=-1)
        weights = jnp.ones_like(per_protocol) if protocol_weights is None else protocol_weights
        return jnp.mean(weights * per_protocol) / num_steps

    def evaluate(self, prediction_mv, target_mv):
        """Return per-candidate aggregate and per-protocol MSE."""
        prediction = jnp.asarray(prediction_mv)
        target = jnp.asarray(target_mv)
        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("Expected prediction (candidate,protocol,time,CV) and target (protocol,time,CV).")
        if prediction.shape[1:] != target.shape:
            raise ValueError(f"Prediction and target shapes differ: {prediction.shape!r} vs {target.shape!r}.")
        per_protocol = jnp.mean((prediction - target[None]) ** 2, axis=(2, 3))
        raw = jnp.mean(per_protocol, axis=1)
        weights = self.prepare(target)
        objective = jnp.mean(per_protocol * weights[None, :], axis=1)
        return objective, raw, per_protocol

    def describe(self) -> dict[str, object]:
        """Return serializable loss metadata."""
        return {
            "name": self.name,
            "quantity": "voltage",
            "reduction": "equal mean over protocol, time, and CV",
            "unit": "mV^2",
            "trajectory_mode": "local_sum",
        }


def raw_voltage_mse() -> LossDefinition:
    """Return the zero-trick raw voltage MSE definition."""
    return LossDefinition()


@dataclass(frozen=True)
class ProtocolBalancedMSE(LossDefinition):
    """Weight protocol MSE inversely by floored target voltage standard deviation."""

    name: str = "protocol_balanced_voltage_mse_v1"
    std_floor_mv: float = 5.0

    def prepare(self, target_mv):
        target = jnp.asarray(target_mv)
        standard_deviation = jnp.std(target, axis=(1, 2))
        inverse = 1.0 / jnp.maximum(standard_deviation, self.std_floor_mv)
        return inverse / jnp.mean(inverse)

    def describe(self) -> dict[str, object]:
        result = super().describe()
        result.update(
            {
                "name": self.name,
                "protocol_weighting": "inverse_target_voltage_std",
                "std_floor_mv": self.std_floor_mv,
            }
        )
        return result


def protocol_balanced_voltage_mse(*, std_floor_mv: float = 5.0) -> ProtocolBalancedMSE:
    """Return target-amplitude-balanced protocol MSE."""
    if std_floor_mv <= 0.0:
        raise ValueError("std_floor_mv must be positive.")
    return ProtocolBalancedMSE(std_floor_mv=std_floor_mv)


@dataclass(frozen=True)
class ProtocolBalancedHuber(ProtocolBalancedMSE):
    """Use balanced MSE inliers and linear tails for large voltage residuals."""

    name: str = "protocol_balanced_huber_v1"
    delta_mv: float = 5.0

    def local(self, prediction_mv, target_mv, *, num_steps: int, protocol_weights=None):
        error = prediction_mv - target_mv
        absolute = jnp.abs(error)
        pointwise = jnp.where(
            absolute <= self.delta_mv,
            error**2,
            2.0 * self.delta_mv * absolute - self.delta_mv**2,
        )
        per_protocol = jnp.mean(pointwise, axis=-1)
        weights = jnp.ones_like(per_protocol) if protocol_weights is None else protocol_weights
        return jnp.mean(weights * per_protocol) / num_steps

    def evaluate(self, prediction_mv, target_mv):
        prediction = jnp.asarray(prediction_mv)
        target = jnp.asarray(target_mv)
        error = prediction - target[None]
        raw_protocol_mse = jnp.mean(error**2, axis=(2, 3))
        raw_total_mse = jnp.mean(raw_protocol_mse, axis=1)
        absolute = jnp.abs(error)
        pointwise = jnp.where(
            absolute <= self.delta_mv,
            error**2,
            2.0 * self.delta_mv * absolute - self.delta_mv**2,
        )
        huber_protocol = jnp.mean(pointwise, axis=(2, 3))
        weights = self.prepare(target)
        objective = jnp.mean(huber_protocol * weights[None, :], axis=1)
        return objective, raw_total_mse, raw_protocol_mse

    def describe(self) -> dict[str, object]:
        result = super().describe()
        result.update(
            {
                "name": self.name,
                "pointwise_penalty": "mse_normalized_huber",
                "delta_mv": self.delta_mv,
            }
        )
        return result


def protocol_balanced_huber(
    *,
    delta_mv: float = 5.0,
    std_floor_mv: float = 5.0,
) -> ProtocolBalancedHuber:
    """Return protocol-balanced MSE-normalized Huber loss."""
    if delta_mv <= 0.0 or std_floor_mv <= 0.0:
        raise ValueError("delta_mv and std_floor_mv must be positive.")
    return ProtocolBalancedHuber(std_floor_mv=std_floor_mv, delta_mv=delta_mv)
