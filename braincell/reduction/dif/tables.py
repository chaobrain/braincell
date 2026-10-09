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

"""Define DIF input identities, measured responses and calibrated tables."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import brainstate
import brainunit as u
import numpy as np

_DIMENSIONS = ("m", "kg", "s", "A", "K", "mol", "cd")


def payload_key(value):
    """Normalize one physical event payload without quantizing its strength."""
    if value is None:
        return 1.0, ()
    if not isinstance(value, u.Quantity):
        raise TypeError("Synapse event weights must carry physical units.")
    magnitude = np.asarray(value.mantissa)
    if magnitude.shape != () or not np.isfinite(magnitude):
        raise ValueError("Each connection weight must be a finite scalar.")
    dimensions = tuple(float(value.unit.dim.get_dimension(name)) for name in _DIMENSIONS)
    return float(magnitude) * value.unit.magnitude, dimensions


@dataclass(frozen=True, order=True)
class ResponseSlot:
    """One physical synapse identity and one calibrated input payload."""

    signature: str
    instance: int
    magnitude: float
    dimensions: tuple[float, ...]

    @classmethod
    def from_input(cls, identity, weight):
        return cls(*identity, *payload_key(weight))

    def payload(self):
        if not self.dimensions:
            return None
        return u.Quantity(self.magnitude, u.Unit(u.get_or_create_dimension(self.dimensions)))


@dataclass(eq=False, frozen=True)
class DIFTable:
    """Host-owned conductance banks, input identities and interpolation coordinates."""

    path: Path | None
    slots: tuple[ResponseSlot, ...]
    calibration_date: str
    dt_ms: float
    leak_per_ms: float
    abort_voltage_mV: float
    pair_source_ptr: np.ndarray
    pair_sources: np.ndarray
    pair_targets: np.ndarray
    pair_reverse: np.ndarray
    rest_single: np.ndarray
    rest_single_ptr: np.ndarray
    rest_pair: np.ndarray
    rest_pair_ptr: np.ndarray
    rest_grid: np.ndarray
    post_state_grid: np.ndarray
    post_single: np.ndarray
    post_single_ptr: np.ndarray
    post_pair: np.ndarray
    post_pair_ptr: np.ndarray
    rebase_single: np.ndarray
    rebase_single_ptr: np.ndarray
    rebase_pair: np.ndarray
    rebase_pair_ptr: np.ndarray
    rest_support: np.ndarray
    post_support: np.ndarray
    rebase_support: np.ndarray
    history_support: np.ndarray
    time_steps: np.ndarray
    time_lower: np.ndarray
    time_upper: np.ndarray
    time_ratio: np.ndarray
    time_inverse_width: np.ndarray
    rebase_single_initial: np.ndarray
    rebase_single_initial_ptr: np.ndarray
    rebase_pair_initial: np.ndarray
    rebase_pair_initial_ptr: np.ndarray
    reversal: np.ndarray
    query_support: np.ndarray
    response_steps: int
    eta: np.ndarray
    rest_voltage: float
    initial_voltage: float
    spike_threshold: float
    confirmation_voltage: float
    rebase_age: int
    candidate_refractory_age: int
    trough_age: int


def load_table(path):
    """Load a DIF table without starting a simulation.

    Parameters
    ----------
    path : str or pathlib.Path
        Calibration archive outside the source checkout.

    Returns
    -------
    DIFTable
        Calibrated conductance banks and host metadata for one cell class.

    Notes
    -----
    Historical archive keys containing ``voltage`` store the fitted conductance
    banks. That file-format mapping is confined to this loader. Existing tables
    remain usable without recalibration.
    """
    from braincell.reduction.dif.tables_utils import build_table

    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        return build_table(archive, path=path)


@brainstate.util.dataclass(eq=False)
class ResponseMeasurements:
    """Detailed response measurements used internally during DIF calibration.

    Curve banks are flat arrays with a zero tail covering the response horizon.
    REST pair curves are state-major, with pointers spanning one interpolation
    state; conductance fitting preserves this row layout.
    ``calibration_date`` records the generating machine's local calendar date
    as YYYY-MM-DD. It is informational, not a compatibility check.
    """

    baseline_voltage_mV: object
    single_voltage_mV: object
    single_voltage_ptr: object
    pair_voltage_mV: object
    pair_voltage_ptr: object
    rest_single_support_steps: object
    post_single_support_steps: object
    rebase_single_support_steps: object
    history_support_steps: object
    tau_steps: object
    event_age_steps: object
    release_age_steps: object
    rest_voltage_grid_mV: object
    V_init_mV: object
    V_rest_mV: object
    V_th_mV: object
    population_location_ptr: object
    population_pair_ptr: object
    pair_slots: object
    dt_ms: object
    voltage_interval_mV: object
    single_response_tolerance_mV: object
    single_response_tail_ms: object
    single_response_horizon_bound_mV: object
    single_probe_ms: object
    clamp_duration_ms: object
    clamp_series_resistance_MOhm: object
    teacher_spike_threshold_mV: object
    abort_voltage_mV: object
    rebase_age_steps: object
    trough_age_steps: object
    eta_recovery_age_steps: object
    eta_ptr: object
    eta_voltage_mV: object
    post_state_ptr: object
    post_state_voltage_mV: object
    post_rebase_single_voltage_mV: object
    post_rebase_single_voltage_ptr: object
    post_single_voltage_mV: object
    post_single_voltage_ptr: object
    post_pair_voltage_mV: object
    post_pair_voltage_ptr: object
    post_rebase_pair_voltage_mV: object
    post_rebase_pair_voltage_ptr: object
    calibration_date: str = brainstate.util.field(pytree_node=False)
    # Host metadata used once to bind physical connections to packed curve rows.
    slots: tuple[ResponseSlot, ...] = brainstate.util.field(pytree_node=False)

    def as_arrays(self):
        """Expose measurements and slot metadata without an intermediate archive."""
        payload = {name: np.asarray(getattr(self, name)) for name in _TABLE_ARRAY_FIELDS}
        payload.update(
            calibration_date=np.asarray(self.calibration_date),
            slot_instance=np.asarray([slot.instance for slot in self.slots], dtype=np.int64),
            slot_signature=np.asarray([slot.signature for slot in self.slots], dtype=np.str_),
            slot_magnitude=np.asarray([slot.magnitude for slot in self.slots], dtype=np.float64),
            slot_is_trigger=np.asarray([not slot.dimensions for slot in self.slots], dtype=np.bool_),
            slot_dimensions=np.asarray([slot.dimensions or (0.0,) * 7 for slot in self.slots], dtype=np.float64),
        )
        return payload


_TABLE_ARRAY_FIELDS = tuple(
    item.name for item in fields(ResponseMeasurements) if item.name not in ("calibration_date", "slots")
)
