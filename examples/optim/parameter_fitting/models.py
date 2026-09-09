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

"""Model and bounded physical-parameter definitions for fitting experiments."""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell
from braincell.filter import AllRegion

CONDUCTANCE_UNIT = u.mS / u.cm**2
PARAMETER_NAMES = ("leak.g_max", "na.g_max", "k.g_max")


@dataclass(frozen=True)
class ParameterSpace:
    """Convert bounded conductances between physical, unit, and optimizer coordinates."""

    names: tuple[str, ...]
    target: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    unit: object = CONDUCTANCE_UNIT

    def __post_init__(self) -> None:
        count = len(self.names)
        if count < 1 or any(len(values) != count for values in (self.target, self.lower, self.upper)):
            raise ValueError("Parameter-space names, target, lower, and upper must have the same non-zero length.")
        target = np.asarray(self.target)
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        if not np.all(np.isfinite(target)) or not np.all(lower < target) or not np.all(target < upper):
            raise ValueError("Parameter targets must be finite and strictly inside their bounds.")

    @property
    def size(self) -> int:
        """Return the number of physical degrees of freedom."""
        return len(self.names)

    def normalized_to_physical(self, normalized) -> object:
        """Map bounded unit coordinates to numeric conductances in ``self.unit``."""
        values = jnp.asarray(normalized)
        self._require_final_axis(values)
        lower = jnp.asarray(self.lower, dtype=values.dtype)
        upper = jnp.asarray(self.upper, dtype=values.dtype)
        return lower + (upper - lower) * values

    def physical_to_normalized(self, physical) -> object:
        """Map numeric conductances to bounded unit coordinates."""
        values = self._numeric_physical(physical)
        lower = jnp.asarray(self.lower, dtype=values.dtype)
        upper = jnp.asarray(self.upper, dtype=values.dtype)
        return (values - lower) / (upper - lower)

    def z_to_physical(self, z) -> object:
        """Map unconstrained optimizer coordinates to numeric conductances."""
        return self.normalized_to_physical(jax.nn.sigmoid(jnp.asarray(z)))

    def physical_to_z(self, physical) -> object:
        """Invert bounded conductances to unconstrained optimizer coordinates."""
        normalized = self.physical_to_normalized(physical)
        eps = jnp.finfo(normalized.dtype).eps
        normalized = jnp.clip(normalized, eps, 1.0 - eps)
        return jnp.log(normalized) - jnp.log1p(-normalized)

    def z_roots(self, physical) -> tuple[object, ...]:
        """Return parameter-leading optimizer roots for an external start ``vmap``."""
        z = self.physical_to_z(physical)
        return tuple(z[..., index] for index in range(self.size))

    def physical_quantities(self, physical) -> dict[str, object]:
        """Return a stable mapping of numeric conductances with units attached."""
        values = self._numeric_physical(physical)
        return {name: values[..., index] * self.unit for index, name in enumerate(self.names)}

    def describe(self) -> dict[str, object]:
        """Return serializable parameter-space metadata."""
        return {
            "names": list(self.names),
            "unit": str(self.unit),
            "target": list(self.target),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "transform": "bounded_sigmoid",
            "optimizer_coordinate": "z",
            "normalized_coordinate": "u",
            "runtime_binding": "direct_parameter",
        }

    def _numeric_physical(self, physical) -> object:
        values = physical.to_decimal(self.unit) if isinstance(physical, u.Quantity) else physical
        values = jnp.asarray(values)
        self._require_final_axis(values)
        return values

    def _require_final_axis(self, values) -> None:
        if values.ndim < 1 or values.shape[-1] != self.size:
            raise ValueError(f"Expected final parameter axis of size {self.size}, got {values.shape!r}.")


@dataclass(frozen=True)
class ModelDefinition:
    """Build the one-CV classical HH target or bounded-direct candidate cell."""

    name: str
    parameter_space: ParameterSpace
    soma_length_um: float = 25.0
    soma_radius_um: float = 12.5

    def build_cell(self, current_na, *, trainable: bool) -> braincell.Cell:
        """Build a protocol-population one-CV cell with static current playback."""
        current = np.asarray(current_na, dtype=np.float64)
        if current.ndim != 2 or current.shape[0] < 1 or not np.all(np.isfinite(current)):
            raise ValueError(f"current_na must have finite shape (protocol,time), got {current.shape!r}.")
        soma = braincell.Branch.from_lengths(
            lengths=[self.soma_length_um] * u.um,
            radii=[self.soma_radius_um, self.soma_radius_um] * u.um,
            type="soma",
        )
        cell = braincell.Cell(
            braincell.Morphology.from_root(soma, name="soma"),
            cv_policy=braincell.CVPerBranch(),
            pop_size=(current.shape[0],),
            V_init=-65.0 * u.mV,
            solver="staggered",
        )
        target = self.parameter_space.target
        cell.paint(
            AllRegion(),
            braincell.mech.CableProperty(
                resting_potential=-65.0 * u.mV,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            ),
            braincell.mech.Ion("SodiumFixed", E=50.0 * u.mV),
            braincell.mech.Ion("PotassiumFixed", E=-77.0 * u.mV),
            braincell.mech.Channel("IL", name="leak", g_max=target[0] * CONDUCTANCE_UNIT, E=-54.387 * u.mV),
            braincell.mech.Channel("Na_HH1952", name="na", g_max=target[1] * CONDUCTANCE_UNIT),
            braincell.mech.Channel("K_HH1952", name="k", g_max=target[2] * CONDUCTANCE_UNIT),
        )
        if trainable:
            for index, channel_name in enumerate(("leak", "na", "k")):
                transform = brainstate.nn.SigmoidT(
                    self.parameter_space.lower[index] * CONDUCTANCE_UNIT,
                    self.parameter_space.upper[index] * CONDUCTANCE_UNIT,
                )
                cell.channels[channel_name].trainable(
                    g_max=braincell.trainable.parameter(
                        group_by="all",
                        transform=transform,
                        name=self.parameter_space.names[index],
                    )
                )
        cell.init_state()
        if cell.n_cv != 1:
            raise RuntimeError(f"Expected one CV, got {cell.n_cv}.")
        target_point = int(cell.node_tree.cv_to_mid_node_id[0])
        area = cell.runtime.point_area.to_decimal(u.cm**2)[target_point]
        current_values = jnp.asarray(current)

        def playback(_point_voltage):
            time_ms = cell._resolve_t().to_decimal(u.ms)
            dt_ms = brainstate.environ.get("dt").to_decimal(u.ms)
            index = jnp.clip(jnp.rint(time_ms / dt_ms).astype(jnp.int32), 0, current_values.shape[1] - 1)
            density = jnp.zeros((current_values.shape[0], cell.n_point), dtype=current_values.dtype)
            density = density.at[:, target_point].set(current_values[:, index] / area)
            return density * u.nA / u.cm**2

        cell.add_current_input("parameter_experiment_current", playback)
        return cell

    def simulate(self, current_na, *, dt_ms: float, num_steps: int | None = None) -> object:
        """Simulate target voltage with shape ``(protocol,time,1)``."""
        current = np.asarray(current_na)
        steps = current.shape[1] if num_steps is None else num_steps
        if steps < 1 or steps > current.shape[1]:
            raise ValueError("num_steps must lie within the current time axis.")
        cell = self.build_cell(current, trainable=False)
        cell.reset_state()

        def step(time_ms):
            voltage = cell.V.value.to_decimal(u.mV)
            with brainstate.environ.context(t=time_ms * u.ms):
                cell.update()
            return voltage

        times = jnp.arange(steps, dtype=jnp.float64) * dt_ms
        with jax.enable_x64(True), brainstate.environ.context(dt=dt_ms * u.ms, precision=64):
            return jnp.moveaxis(brainstate.transform.for_loop(step, times), 0, 1)

    def describe(self) -> dict[str, object]:
        """Return serializable model metadata."""
        return {
            "name": self.name,
            "morphology": {"soma_length_um": self.soma_length_um, "soma_radius_um": self.soma_radius_um},
            "n_cv": 1,
            "state_names": ["V", "m", "h", "n"],
            "channels": ["IL", "Na_HH1952", "K_HH1952"],
            "parameter_space": self.parameter_space.describe(),
        }


def hh_1cv_classic_bounded_direct(
    *,
    bound_multipliers: tuple[float, float] = (0.5, 1.5),
) -> ModelDefinition:
    """Return the fixed one-CV classical-conductance baseline model."""
    target = (0.3, 120.0, 36.0)
    lower_multiplier, upper_multiplier = bound_multipliers
    if not 0.0 < lower_multiplier < 1.0 < upper_multiplier:
        raise ValueError("bound_multipliers must strictly contain the target multiplier 1.0.")
    return ModelDefinition(
        name="hh_1cv_classic_bounded_direct_v1",
        parameter_space=ParameterSpace(
            names=PARAMETER_NAMES,
            target=target,
            lower=tuple(lower_multiplier * value for value in target),
            upper=tuple(upper_multiplier * value for value in target),
        ),
    )
