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

"""Declaration-time views over concrete synapse instances."""

from dataclasses import dataclass

import brainunit as u
import numpy as np

from braincell.mech import Synapse

__all__ = ["SynapseInstance", "SynapseInstanceTable", "SynapseInstanceView"]


@dataclass(frozen=True)
class SynapseInstance:
    """One logical synapse instance owned by a cell population member."""

    id: int
    placement_id: int
    population_index: int
    point_id: int
    cv_id: int
    branch_id: int
    branch_x: float
    mechanism: Synapse


class SynapseInstanceView:
    """A stable ordered selection of logical synapse instances."""

    __slots__ = ("_cell", "_instances")

    def __init__(self, cell, instances: tuple[SynapseInstance, ...]) -> None:
        self._cell = cell
        self._instances = instances

    @property
    def cell(self):
        """Owning :class:`~braincell.Cell`."""
        return self._cell

    @property
    def instances(self) -> tuple[SynapseInstance, ...]:
        """Selected instances in stable declaration order."""
        return self._instances

    @property
    def id(self) -> np.ndarray:
        """Logical instance identifiers."""
        return np.asarray([item.id for item in self._instances], dtype=np.int64)

    @property
    def placement_id(self) -> np.ndarray:
        """Underlying point-placement identifiers."""
        return np.asarray([item.placement_id for item in self._instances], dtype=np.int64)

    @property
    def population_index(self) -> np.ndarray:
        """Owning population-member identifiers."""
        return np.asarray([item.population_index for item in self._instances], dtype=np.int64)

    @property
    def point_id(self) -> np.ndarray:
        """Resolved electrical point identifiers."""
        return np.asarray([item.point_id for item in self._instances], dtype=np.int64)

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, index: object) -> "SynapseInstanceView":
        selected = np.asarray(self._instances, dtype=object)[index]
        if not isinstance(selected, np.ndarray):
            values = (selected,)
        else:
            values = tuple(selected.reshape(-1).tolist())
        return SynapseInstanceView(self._cell, values)

    def set(self, **parameters: object) -> "SynapseInstanceView":
        """Set per-instance declaration parameters before initialization.

        Each value may be scalar or have one value per selected instance.
        Quantity inputs must be compatible with the unit declared by the
        selected synapse specification.
        """
        self._cell._raise_if_initialized("set synapse parameters")
        if not parameters:
            return self
        if len(self._instances) == 0:
            raise ValueError("Cannot set parameters on an empty synapse view.")

        for parameter, value in parameters.items():
            normalized = _normalize_parameter_values(
                self._instances,
                parameter=str(parameter),
                value=value,
            )
            for instance, item in zip(self._instances, normalized):
                self._cell._synapse_parameter_overrides[
                    (instance.placement_id, instance.population_index, str(parameter))
                ] = item
        return self


class SynapseInstanceTable(SynapseInstanceView):
    """All logical point-synapse instances currently declared on a cell."""

    def __init__(self, cell) -> None:
        super().__init__(cell, _expand_instances(cell))

    def __getitem__(self, selector: object) -> SynapseInstanceView:
        if isinstance(selector, Synapse):
            return SynapseInstanceView(
                self._cell,
                tuple(item for item in self._instances if item.mechanism is selector),
            )
        if isinstance(selector, str):
            return SynapseInstanceView(
                self._cell,
                tuple(item for item in self._instances if item.mechanism.instance_name == selector),
            )
        return super().__getitem__(selector)


def _expand_instances(cell) -> tuple[SynapseInstance, ...]:
    if len(cell.pop_size) > 1:
        raise ValueError(
            f"Synapse instance views currently require scalar or one-dimensional pop_size; got {cell.pop_size!r}."
        )
    population_size = 1 if len(cell.pop_size) == 0 else int(cell.pop_size[0])
    result: list[SynapseInstance] = []
    for placement in cell.point_placements:
        if not isinstance(placement.mechanism, Synapse):
            continue
        population_indices = (
            range(population_size) if placement.population_index is None else (int(placement.population_index),)
        )
        for population_index in population_indices:
            result.append(
                SynapseInstance(
                    id=len(result),
                    placement_id=int(placement.id),
                    population_index=int(population_index),
                    point_id=int(placement.point_id),
                    cv_id=int(placement.cv_id),
                    branch_id=int(placement.branch_id),
                    branch_x=float(placement.branch_x),
                    mechanism=placement.mechanism,
                )
            )
    return tuple(result)


def _normalize_parameter_values(
    instances: tuple[SynapseInstance, ...],
    *,
    parameter: str,
    value: object,
) -> tuple[object, ...]:
    defaults: list[object] = []
    for instance in instances:
        if parameter not in instance.mechanism.params:
            raise KeyError(f"Synapse {instance.mechanism.instance_name!r} does not declare parameter {parameter!r}.")
        defaults.append(instance.mechanism.params[parameter])

    first_default = defaults[0]
    count = len(instances)
    if isinstance(first_default, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Synapse parameter {parameter!r} requires a quantity.")
        try:
            decimal = np.asarray(value.to_decimal(first_default.unit))
        except Exception as exc:
            raise ValueError(f"Synapse parameter {parameter!r} has an incompatible unit.") from exc
        values = _broadcast_vector(decimal, count=count, parameter=parameter)
        return tuple(u.Quantity(item, first_default.unit) for item in values)

    if isinstance(value, u.Quantity):
        raise TypeError(f"Synapse parameter {parameter!r} is dimensionless.")
    return tuple(_broadcast_vector(np.asarray(value), count=count, parameter=parameter).tolist())


def _broadcast_vector(value: np.ndarray, *, count: int, parameter: str) -> np.ndarray:
    if value.ndim == 0:
        return np.broadcast_to(value, (count,))
    if value.shape == (count,):
        return value
    raise ValueError(f"Synapse parameter {parameter!r} must be scalar or have shape {(count,)!r}, got {value.shape!r}.")
