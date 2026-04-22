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

from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

import numpy as np

from braincell.mech import (
    CurrentProbe,
    Density,
    MechanismProbe,
    Point,
    ProbeMechanism,
    StateProbe,
    Synapse,
)
from .runtime import CellRuntimeState

__all__ = [
    "MechanismObjectCell",
    "MechanismObjectTable",
    "mechanism_cell_key",
]


# These tables are inspection helpers layered on top of compiled runtime state.
# They do not participate in rebuild or solver execution; they only provide
# convenient matrix-like views for debugging and notebooks.


@dataclass(frozen=True)
class MechanismObjectCell:
    """Per-column runtime mechanism view in a merged mechanism table.

    One ``MechanismObjectCell`` represents the intersection of:

    - one logical mechanism row
    - one point/CV column in a table view
    - one lowered runtime layout id

    It acts as a lightweight facade over both declaration-level and runtime
    state. Callers can inspect the original declaration, the installed runtime
    node, or one parameter value resolved at the specific point/CV column.

    This object is mainly consumed by notebook/debug tooling through the table
    APIs exposed on :class:`Cell`.
    """

    runtime: CellRuntimeState
    layout_id: int
    class_name: str
    instance_name: str
    column_id: int
    domain: Literal["cv", "point"]
    cv_id: int | None = None
    point_id: int | None = None

    @property
    def row_key(self) -> tuple[str, str]:
        return (self.class_name, self.instance_name)

    @property
    def row_label(self) -> str:
        if self.instance_name == self.class_name:
            return self.class_name
        return f"{self.instance_name}:{self.class_name}"

    @property
    def node(self) -> object:
        return self.runtime.runtime_nodes.get(self.layout_id, self.declaration)

    @property
    def mechanism(self) -> object:
        return self.node

    @property
    def declaration(self) -> object:
        return self.runtime.get_layout_mechanism(self.layout_id)

    def get_param(self, name: str) -> object:
        key = str(name)
        if self.point_id is not None and self.runtime.has_layout_value(self.layout_id, key):
            return self.runtime.get_layout_value(self.layout_id, point_id=self.point_id, var_name=key)
        declaration = self.declaration
        if isinstance(declaration, Density):
            if key in declaration.params:
                return declaration.params[key]
        node = self.node
        if hasattr(node, key):
            return getattr(node, key)
        raise AttributeError(f"Mechanism cell {self.row_label!r} has no parameter {key!r}.")

    def __getattr__(self, name: str) -> object:
        if name.startswith("_") or name in {
            "runtime", "layout_id", "class_name", "instance_name",
            "column_id", "domain", "cv_id", "point_id",
        }:
            raise AttributeError(name)
        declaration = self.declaration
        node = self.node
        candidates: set[str] = set()
        if hasattr(declaration, "params"):
            candidates.update(declaration.params.keys())
        if is_dataclass(declaration):
            candidates.update(f.name for f in fields(declaration))
        if node is not None and node is not declaration:
            candidates.update(
                attr_name for attr_name in dir(node) if not attr_name.startswith("_")
            )
        if name in candidates:
            try:
                return self.get_param(name)
            except AttributeError:
                pass
        raise AttributeError(
            f"Mechanism cell {self.row_label!r} has no attribute {name!r}. "
            f"Known params: {sorted(candidates)!r}."
        )


@dataclass(frozen=True)
class MechanismObjectTable:
    """Merged mechanism-object matrix for point-domain runtime views.

    The table groups runtime layouts by logical mechanism identity so debugging
    code can ask questions in a matrix form:

    - rows correspond to mechanism class/instance identities
    - columns correspond to point ids in the current lowered point view
    - entries are :class:`MechanismObjectCell` facades or ``None``

    ``Cell`` builds these tables on demand as inspection helpers; they are not
    part of rebuild or solver execution.
    """

    domain: Literal["point"]
    row_keys: tuple[tuple[str, str], ...]
    row_labels: tuple[str, ...]
    column_ids: tuple[int, ...]
    values: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(dim) for dim in self.values.shape)

    def get(self, row_key: tuple[str, str], column_id: int) -> MechanismObjectCell | None:
        try:
            row_index = self.row_keys.index((str(row_key[0]), str(row_key[1])))
        except ValueError as exc:
            raise KeyError(f"Unknown mechanism row key {row_key!r}.") from exc
        return self._get_at(row_index=row_index, column_id=column_id)

    def get_by_label(self, row_label: str, column_id: int) -> MechanismObjectCell | None:
        try:
            row_index = self.row_labels.index(str(row_label))
        except ValueError as exc:
            raise KeyError(f"Unknown mechanism row label {row_label!r}.") from exc
        return self._get_at(row_index=row_index, column_id=column_id)

    def _get_at(self, *, row_index: int, column_id: int) -> MechanismObjectCell | None:
        try:
            column_index = self.column_ids.index(int(column_id))
        except ValueError as exc:
            raise KeyError(f"Unknown {self.domain} column id {column_id!r}.") from exc
        return self.values[row_index, column_index]


def mechanism_cell_key(mechanism: object) -> tuple[str, str]:
    """Return a ``(class_name, instance_name)`` key for table indexing.

    Every mechanism type is mapped to a stable tuple that table views
    use as a row identifier. :class:`Density` mechanisms use
    ``(class_name, instance_name)`` (which collapses to
    ``(class_name, class_name)`` when no explicit name override was
    provided). :class:`Point` mechanisms fall back to their Python
    class name on both axes unless they carry their own identity.
    """
    if isinstance(mechanism, Density):
        return (mechanism.class_name, mechanism.instance_name)
    if isinstance(mechanism, Synapse):
        return (mechanism.synapse_type, mechanism.instance_name)
    if isinstance(mechanism, ProbeMechanism):
        class_name = "ProbeMechanism"
        instance_name = (
            mechanism.variable
            if mechanism.target is None
            else f"{mechanism.variable}@{mechanism.target}"
        )
        return (class_name, instance_name)
    if isinstance(mechanism, StateProbe):
        return ("StateProbe", mechanism.name if mechanism.name is not None else mechanism.field)
    if isinstance(mechanism, MechanismProbe):
        return (
            "MechanismProbe",
            mechanism.name if mechanism.name is not None else f"{mechanism.mechanism}_{mechanism.field}",
        )
    if isinstance(mechanism, CurrentProbe):
        return (
            "CurrentProbe",
            mechanism.name if mechanism.name is not None else (
                f"{mechanism.mechanism}_current" if mechanism.mechanism is not None else f"{mechanism.ion}_current"
            ),
        )
    if isinstance(mechanism, Point):
        class_name = type(mechanism).__name__
        return (class_name, class_name)
    class_name = type(mechanism).__name__
    return (class_name, class_name)
