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

"""Explicit event-source to synapse contacts."""

import brainunit as u
import numpy as np

from braincell._multi_compartment.synapses import SynapseInstanceView
from braincell.event import NetStim
from braincell.mech import get_registry

__all__ = ["Contact"]


class Contact:
    """Connect event sources to concrete synapse instances.

    Contacts are strongly owned by the target cell until :meth:`remove` is
    called. Source and target populations zip when equal-sized, fan out when
    the source has size one, and converge when the target has size one.
    Other shapes require an explicit ``pairs`` matrix.
    """

    __slots__ = (
        "source",
        "target",
        "source_index",
        "target_index",
        "weight",
        "delay",
        "id",
        "_removed",
    )

    def __init__(
        self,
        *,
        source: NetStim,
        target: SynapseInstanceView,
        weight=1.0 * u.uS,
        delay=0.0 * u.ms,
        pairs=None,
    ) -> None:
        if not isinstance(source, NetStim):
            raise TypeError(f"Contact.source must be NetStim, got {type(source).__name__!s}.")
        if not isinstance(target, SynapseInstanceView):
            raise TypeError(f"Contact.target must be a cell.synapses[...] view, got {type(target).__name__!s}.")
        if len(target) == 0:
            raise ValueError("Contact.target cannot be empty.")
        source_index, target_index = _contact_indices(source.size, len(target), pairs=pairs)
        count = len(source_index)
        normalized_weight = _normalize_weight(target, weight, count=count)
        normalized_delay = _quantity_vector(delay, unit=u.ms, count=count, name="Contact.delay")
        if np.any(np.asarray(normalized_delay.to_decimal(u.ms)) < 0.0):
            raise ValueError("Contact.delay must be >= 0 ms.")

        cell = target.cell
        cell._raise_if_initialized("add Contact")
        contact_ids = np.arange(cell._next_contact_id, cell._next_contact_id + count, dtype=np.int64)
        cell._next_contact_id += count

        self.source = source
        self.target = target
        self.source_index = source_index
        self.target_index = target_index
        self.weight = normalized_weight
        self.delay = normalized_delay
        self.id = contact_ids
        self._removed = False
        cell._contacts.append(self)

    def __len__(self) -> int:
        return len(self.source_index)

    @property
    def removed(self) -> bool:
        """Whether this contact group has been explicitly removed."""
        return self._removed

    def remove(self) -> None:
        """Remove this contact group while leaving its identifiers unused."""
        if self._removed:
            return
        self.target.cell._raise_if_initialized("remove Contact")
        self.target.cell._contacts.remove(self)
        self._removed = True


def _contact_indices(source_size: int, target_size: int, *, pairs):
    if pairs is not None:
        values = np.asarray(pairs)
        if values.ndim != 2 or values.shape[1] != 2 or values.dtype.kind not in "iu":
            raise TypeError("Contact.pairs must be an integer array with shape (n_contact, 2).")
        source_index = np.asarray(values[:, 0], dtype=np.int32)
        target_index = np.asarray(values[:, 1], dtype=np.int32)
    elif source_size == target_size:
        source_index = np.arange(source_size, dtype=np.int32)
        target_index = np.arange(target_size, dtype=np.int32)
    elif source_size == 1:
        source_index = np.zeros(target_size, dtype=np.int32)
        target_index = np.arange(target_size, dtype=np.int32)
    elif target_size == 1:
        source_index = np.arange(source_size, dtype=np.int32)
        target_index = np.zeros(source_size, dtype=np.int32)
    else:
        raise ValueError(
            "Contact source and target sizes must match, or one side must have size 1; "
            "provide pairs=(n_contact, 2) for other mappings."
        )
    if np.any(source_index < 0) or np.any(source_index >= source_size):
        raise IndexError("Contact source index is out of range.")
    if np.any(target_index < 0) or np.any(target_index >= target_size):
        raise IndexError("Contact target index is out of range.")
    return source_index, target_index


def _quantity_vector(value, *, unit, count: int, name: str) -> u.Quantity:
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be a quantity.")
    try:
        decimal = np.asarray(value.to_decimal(unit), dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{name} has an incompatible unit.") from exc
    if decimal.ndim == 0:
        decimal = np.broadcast_to(decimal, (count,))
    elif decimal.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape {(count,)!r}, got {decimal.shape!r}.")
    if np.any(~np.isfinite(decimal)):
        raise ValueError(f"{name} must contain finite values.")
    return u.Quantity(np.array(decimal, copy=True), unit)


def _normalize_weight(target: SynapseInstanceView, value, *, count: int):
    units = []
    for instance in target.instances:
        runtime_cls = get_registry().get("synapse", instance.mechanism.synapse_type)
        units.append(getattr(runtime_cls, "event_weight_unit", None))
    first = units[0]
    if any(unit != first for unit in units[1:]):
        raise ValueError("Contact.target mixes synapse models with different event weight units.")
    if first is not None:
        return _quantity_vector(value, unit=first, count=count, name="Contact.weight")
    if isinstance(value, u.Quantity):
        if not value.dim.is_dimensionless:
            raise ValueError("Contact.weight must be dimensionless for the target synapse model.")
        value = value.to_decimal()
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.broadcast_to(values, (count,))
    elif values.shape != (count,):
        raise ValueError(f"Contact.weight must be scalar or have shape {(count,)!r}, got {values.shape!r}.")
    if np.any(~np.isfinite(values)):
        raise ValueError("Contact.weight must contain finite values.")
    return np.array(values, copy=True)
