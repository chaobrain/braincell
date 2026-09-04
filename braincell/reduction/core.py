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

"""Public contracts for interchangeable Cell reduction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import weakref

import numpy as np

__all__ = [
    "ReductionContext",
    "ReductionInputGroup",
    "ReductionInputGroupSchema",
    "ReductionInputs",
    "ReductionModel",
    "ReductionOutput",
    "ReductionSynapse",
    "ReductionView",
    "ReductionViewCollection",
]


@dataclass(frozen=True)
class ReductionSynapse:
    """Describe one logical synapse exposed to a reduction model."""

    id: int
    synapse_index: int
    population_index: int
    placement_id: int
    point_id: int
    cv_id: int
    branch_id: int
    branch_x: float
    name: str
    synapse_type: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class ReductionInputGroupSchema:
    """Describe one packed homogeneous synapse-payload group."""

    layout_id: int
    synapse_type: str
    event_input: object
    synapse_id: np.ndarray
    synapse_index: np.ndarray
    population_index: np.ndarray

    def __post_init__(self) -> None:
        for name in ("synapse_id", "synapse_index", "population_index"):
            value = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def size(self) -> int:
        """Return the number of packed synapse rows in this group."""
        return int(self.synapse_id.size)


@dataclass(frozen=True)
class ReductionInputGroup:
    """Pair one static input schema with its current event payload."""

    schema: ReductionInputGroupSchema
    payload: object


@dataclass(frozen=True)
class ReductionInputs:
    """Hold all synapse payload groups consumed by one reduced update."""

    groups: tuple[ReductionInputGroup, ...] = ()

    def __iter__(self):
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class ReductionOutput:
    """Return named raw values and one canonical event output."""

    values: Mapping[str, object]
    event: object

    def __post_init__(self) -> None:
        prepared = dict(self.values)
        for name in prepared:
            if not isinstance(name, str) or not name:
                raise ValueError("Reduction output names must be non-empty strings.")
        object.__setattr__(self, "values", MappingProxyType(prepared))


@dataclass(frozen=True)
class ReductionContext:
    """Provide a reduction model with the current static Cell input schema."""

    pop_size: tuple[int, ...]
    synapses: tuple[ReductionSynapse, ...]
    input_groups: tuple[ReductionInputGroupSchema, ...]
    fingerprint: str
    _cell_ref: object = field(repr=False, compare=False)

    @classmethod
    def with_cell(
        cls,
        cell,
        *,
        synapses: tuple[ReductionSynapse, ...],
        input_groups: tuple[ReductionInputGroupSchema, ...],
        fingerprint: str,
    ) -> "ReductionContext":
        """Create a context carrying a weak reference to its owner Cell."""
        return cls(
            pop_size=tuple(cell.pop_size),
            synapses=synapses,
            input_groups=input_groups,
            fingerprint=fingerprint,
            _cell_ref=weakref.ref(cell),
        )

    @property
    def cell(self):
        """Return the owning Cell while it remains alive."""
        cell = self._cell_ref()
        if cell is None:
            raise RuntimeError("The Cell owning this ReductionContext no longer exists.")
        return cell

    @property
    def population_size(self) -> int:
        """Return the flattened number of Cell population members."""
        return int(np.prod(self.pop_size, dtype=np.int64)) if self.pop_size else 1


class ReductionModel(ABC):
    """Define the minimal lifecycle implemented by a Cell reduction model."""

    @abstractmethod
    def init_state(self, context: ReductionContext, batch_size=None) -> ReductionOutput:
        """Initialize model state for one current Cell context."""

    @abstractmethod
    def update(self, inputs: ReductionInputs) -> ReductionOutput:
        """Consume current synapse payloads and advance the model once."""

    @abstractmethod
    def reset_state(self, batch_size=None) -> ReductionOutput:
        """Reset dynamic state and return the model's initial output."""

    def reset(self) -> None:
        """Drop optional initialized caches while retaining model parameters."""

    def get(self, field: str, population_indices: tuple[int, ...]):
        """Return one optional population parameter for selected members."""
        raise NotImplementedError(f"{type(self).__name__} does not expose population parameters.")

    def set(self, population_indices: tuple[int, ...], **parameters) -> None:
        """Set optional population parameters for selected members."""
        raise NotImplementedError(f"{type(self).__name__} does not expose population parameters.")


class ReductionView:
    """Expose one named reduction model over selected population members."""

    __slots__ = ("_cell", "_name", "_population_indices")

    def __init__(self, cell, name: str, population_indices: tuple[int, ...]) -> None:
        self._cell = cell
        self._name = name
        self._population_indices = tuple(int(item) for item in population_indices)

    @property
    def name(self) -> str:
        """Return the Cell-local registered model name."""
        return self._name

    @property
    def model(self) -> ReductionModel:
        """Return the registered model object."""
        return self._cell._reduction_models[self._name]

    @property
    def population_indices(self) -> tuple[int, ...]:
        """Return selected root-population indices."""
        return self._population_indices

    @property
    def is_selected(self) -> bool:
        """Return whether this is the root Cell's selected execution model."""
        return self._cell._selected_model_name == self._name

    def get(self, field: str):
        """Read a model-defined population parameter over this view."""
        if not isinstance(field, str) or not field:
            raise ValueError("Reduction parameter name must be a non-empty string.")
        return self.model.get(field, self._population_indices)

    def set(self, **parameters) -> "ReductionView":
        """Set model-defined population parameters over this view."""
        self._cell._raise_if_initialized("set reduction parameters")
        if not parameters:
            return self
        self.model.set(self._population_indices, **parameters)
        self._cell._run_loop_cache.clear()
        return self

    def __repr__(self) -> str:
        return (
            f"ReductionView(name={self.name!r}, model={type(self.model).__name__}, "
            f"population_indices={self.population_indices!r}, selected={self.is_selected!r})"
        )


class ReductionViewCollection(Mapping[str, ReductionView]):
    """Provide Mapping-style access to a Cell's registered reductions."""

    __slots__ = ("_cell", "_population_indices")

    def __init__(self, cell, population_indices: tuple[int, ...]) -> None:
        self._cell = cell
        self._population_indices = tuple(int(item) for item in population_indices)

    def __getitem__(self, name: str) -> ReductionView:
        if name not in self._cell._reduction_models:
            raise KeyError(
                f"Unknown reduction model {name!r}; available models: {tuple(self._cell._reduction_models)!r}."
            )
        return ReductionView(self._cell, name, self._population_indices)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cell._reduction_models)

    def __len__(self) -> int:
        return len(self._cell._reduction_models)

    def __repr__(self) -> str:
        return f"ReductionViewCollection(names={tuple(self)!r}, population_indices={self._population_indices!r})"
