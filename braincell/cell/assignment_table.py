

from dataclasses import dataclass
from typing import Literal

import numpy as np

from braincell.mech.density import DensityMechanism
from braincell.mech.point import (
    CurrentClamp,
    FunctionClamp,
    GapJunctionMechanism,
    ProbeMechanism,
    SineClamp,
    SynapseMechanism,
)
from braincell.mech.spec import MechanismSpec, density_class_name, density_instance_name, is_density_mechanism

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

    runtime: "CellRuntimeState"
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
        if is_density_mechanism(declaration):
            params = dict(_mechanism_params(declaration))
            if key in params:
                return params[key]
        node = self.node
        if hasattr(node, key):
            return getattr(node, key)
        raise AttributeError(f"Mechanism cell {self.row_label!r} has no parameter {key!r}.")

    def __getattr__(self, name: str) -> object:
        try:
            return self.get_param(name)
        except AttributeError as exc:
            raise AttributeError(name) from exc


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
    if is_density_mechanism(mechanism):
        return density_class_name(mechanism)[1], density_instance_name(mechanism)
    if isinstance(mechanism, SynapseMechanism):
        return mechanism.synapse_type, mechanism.synapse_type
    if isinstance(mechanism, ProbeMechanism):
        class_name = "ProbeMechanism"
        instance_name = mechanism.variable if mechanism.target is None else f"{mechanism.variable}@{mechanism.target}"
        return class_name, instance_name
    if isinstance(mechanism, (GapJunctionMechanism, CurrentClamp, SineClamp, FunctionClamp)):
        class_name = type(mechanism).__name__
        return class_name, class_name
    class_name = type(mechanism).__name__
    return class_name, class_name


def _mechanism_params(mechanism: object) -> tuple[tuple[str, object], ...]:
    if isinstance(mechanism, DensityMechanism):
        return tuple(mechanism.params)
    if isinstance(mechanism, MechanismSpec):
        return tuple(mechanism.params)
    return ()
