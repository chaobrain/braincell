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

"""Stable DBNN channels derived from BrainCell logical synapses."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Literal

import brainunit as u
import numpy as np

from braincell.mech import Synapse, get_registry


POLARITY_THRESHOLD_MV = -50.0


@dataclass(frozen=True)
class ChannelLayout:
    """Store resolved BrainCell specs and aligned logical-synapse columns."""

    specs: tuple[Synapse, ...]
    synapse_ids: tuple[int, ...]
    placement_ids: tuple[int, ...]
    point_ids: tuple[int, ...]
    spec_indices: tuple[int, ...]
    reference_weights_us: tuple[float, ...] = ()
    training_ranges: tuple[tuple[float, float], ...] = ()
    branch_ids: tuple[int, ...] = ()
    branch_xs: tuple[float, ...] = ()

    def __post_init__(self):
        specs = tuple(_resolve_spec(spec) for spec in self.specs)
        if not specs:
            raise ValueError("A channel layout requires at least one synapse spec.")
        identities = tuple((spec.name, spec.synapse_type) for spec in specs)
        if len(set(identities)) != len(identities):
            raise ValueError("Synapse spec name/type identities must be unique.")

        raw_identity_columns = tuple(
            tuple(column) for column in (self.synapse_ids, self.placement_ids, self.point_ids, self.spec_indices)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for column in raw_identity_columns
            for value in column
        ):
            raise TypeError("DBNN channel identifier columns must contain only integers.")
        synapse_ids, placement_ids, point_ids, spec_indices = (
            tuple(int(value) for value in column) for column in raw_identity_columns
        )
        channel_count = len(synapse_ids)
        if channel_count == 0:
            raise ValueError("A channel layout requires at least one configured input channel.")
        if any(len(column) != channel_count for column in (placement_ids, point_ids, spec_indices)):
            raise ValueError("DBNN logical-synapse channel columns must have equal lengths.")
        if min((*synapse_ids, *placement_ids, *point_ids, *spec_indices)) < 0:
            raise ValueError("DBNN channel identifiers must be non-negative.")
        if len(set(synapse_ids)) != channel_count:
            raise ValueError("DBNN channel logical synapse IDs must be unique.")
        if len(set(placement_ids)) != channel_count:
            raise ValueError("DBNN template placement IDs must be unique.")
        if any(index >= len(specs) for index in spec_indices):
            raise ValueError("DBNN channel spec indices must reference configured specs.")

        try:
            raw_branch_ids = tuple(self.branch_ids)
        except TypeError as exc:
            raise TypeError("DBNN channel branch IDs must be an iterable of integers.") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw_branch_ids
        ):
            raise TypeError("DBNN channel branch IDs must contain only integers.")
        branch_ids = tuple(int(value) for value in raw_branch_ids)
        branch_xs = tuple(float(value) for value in self.branch_xs)
        if bool(branch_ids) != bool(branch_xs) or branch_ids and (
            len(branch_ids) != channel_count or len(branch_xs) != channel_count
        ):
            raise ValueError("DBNN channel branch IDs and coordinates must both be empty or complete.")
        if any(value < 0 for value in branch_ids):
            raise ValueError("DBNN channel branch IDs must be non-negative.")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in branch_xs):
            raise ValueError("DBNN channel branch coordinates must be finite and lie in [0, 1].")

        reference_weights = self.reference_weights_us or (1.0,) * len(specs)
        reference_weights = tuple(float(value) for value in reference_weights)
        training_ranges = self.training_ranges or ((0.0, 1.0),) * len(specs)
        training_ranges = tuple((float(low), float(high)) for low, high in training_ranges)
        if len(reference_weights) != len(specs) or len(training_ranges) != len(specs):
            raise ValueError("Reference weights and training ranges must align with synapse specs.")
        if not np.isfinite(reference_weights).all() or any(value <= 0 for value in reference_weights):
            raise ValueError("Reference weights must be finite and positive.")
        if any(not np.isfinite((low, high)).all() or low < 0 or high < low for low, high in training_ranges):
            raise ValueError("Training ranges must be finite, non-negative, and ordered.")

        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "synapse_ids", synapse_ids)
        object.__setattr__(self, "placement_ids", placement_ids)
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "spec_indices", spec_indices)
        object.__setattr__(self, "reference_weights_us", reference_weights)
        object.__setattr__(self, "training_ranges", training_ranges)
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "branch_xs", branch_xs)

    @property
    def n_channels(self) -> int:
        """Return the number of fixed input channels."""
        return len(self.synapse_ids)

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 digest of the complete layout."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def channel_id(self, synapse_id: int) -> int:
        """Return the channel ID for one template logical synapse ID."""
        try:
            return self.synapse_ids.index(int(synapse_id))
        except ValueError as exc:
            raise ValueError(f"Detailed Cell has no DBNN input channel for logical synapse {synapse_id}.") from exc

    def synapse_id(self, channel_id: int) -> int:
        """Return the template logical synapse ID represented by a channel."""
        self._validate_channel_id(channel_id)
        return self.synapse_ids[channel_id]

    def placement_id(self, channel_id: int) -> int:
        """Return the template placement ID represented by a channel."""
        self._validate_channel_id(channel_id)
        return self.placement_ids[channel_id]

    def point_id(self, channel_id: int) -> int:
        """Return the electrical point ID represented by a channel."""
        self._validate_channel_id(channel_id)
        return self.point_ids[channel_id]

    def spec_index(self, channel_id: int) -> int:
        """Return the synapse-spec index represented by a channel."""
        self._validate_channel_id(channel_id)
        return self.spec_indices[channel_id]

    def spec(self, channel_id: int) -> Synapse:
        """Return the resolved BrainCell synapse spec represented by a channel."""
        return self.specs[self.spec_index(channel_id)]

    def reversal_potential_mv(self, channel_id: int) -> float:
        """Return a channel's canonical reversal potential in millivolts."""
        value = self.spec(channel_id).params["e"]
        return float(np.asarray(value.to_decimal(u.mV)))

    def polarity(self, channel_id: int) -> str:
        """Derive the channel E/I class from its reversal potential."""
        return "E" if self.reversal_potential_mv(channel_id) >= POLARITY_THRESHOLD_MV else "I"

    def reference_weight_us(self, channel_id: int) -> float:
        """Return the reference input weight represented by a channel."""
        return self.reference_weights_us[self.spec_index(channel_id)]

    def training_range(self, channel_id: int) -> tuple[float, float]:
        """Return the training input range represented by a channel."""
        return self.training_ranges[self.spec_index(channel_id)]

    def to_dict(self) -> dict[str, Any]:
        """Convert the layout to a JSON-compatible dictionary."""
        return {
            "specs": [_spec_to_dict(spec) for spec in self.specs],
            "synapse_ids": list(self.synapse_ids),
            "placement_ids": list(self.placement_ids),
            "point_ids": list(self.point_ids),
            "spec_indices": list(self.spec_indices),
            "reference_weights_us": list(self.reference_weights_us),
            "training_ranges": [list(value) for value in self.training_ranges],
            "branch_ids": list(self.branch_ids),
            "branch_xs": list(self.branch_xs),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ChannelLayout":
        """Reconstruct a layout from its JSON-compatible dictionary."""
        return cls(
            specs=tuple(_spec_from_dict(item) for item in value["specs"]),
            synapse_ids=tuple(value["synapse_ids"]),
            placement_ids=tuple(value["placement_ids"]),
            point_ids=tuple(value["point_ids"]),
            spec_indices=tuple(value["spec_indices"]),
            reference_weights_us=tuple(value["reference_weights_us"]),
            training_ranges=tuple(tuple(item) for item in value["training_ranges"]),
            branch_ids=tuple(value["branch_ids"]),
            branch_xs=tuple(value["branch_xs"]),
        )

    def _validate_channel_id(self, channel_id: int) -> None:
        if not 0 <= channel_id < self.n_channels:
            raise ValueError(f"Invalid channel_id {channel_id}.")


@dataclass(frozen=True)
class ChannelAlignment:
    """Record how candidate channels map to checkpoint input channels.

    ``permutation[i]`` is the candidate channel supplying checkpoint channel
    ``i``.

    Parameters
    ----------
    method : {"identity", "branch_position", "identifier_distance"}
        Matching strategy used to construct the permutation.
    permutation : tuple of int
        Complete candidate-channel permutation in checkpoint order.
    """

    method: Literal["identity", "branch_position", "identifier_distance"]
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.method not in {"identity", "branch_position", "identifier_distance"}:
            raise ValueError("Invalid channel alignment method.")
        try:
            raw_permutation = tuple(self.permutation)
        except TypeError as exc:
            raise TypeError("permutation must be an iterable of integers.") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in raw_permutation
        ):
            raise TypeError("permutation must contain only integers.")
        permutation = tuple(int(value) for value in raw_permutation)
        if sorted(permutation) != list(range(len(permutation))):
            raise ValueError("permutation must contain every channel index exactly once.")
        object.__setattr__(self, "permutation", permutation)


def build_channel_layout(
    template_cell: Any,
    specs: tuple[Synapse, ...],
    *,
    reference_weights_us: tuple[float, ...] = (),
    training_ranges: tuple[tuple[float, float], ...] = (),
) -> ChannelLayout:
    """Build DBNN channels from BrainCell's resolved logical synapse view."""
    if not hasattr(template_cell, "synapses"):
        raise TypeError("template_cell must expose a BrainCell SynapseView through cell.synapses.")
    pop_size = tuple(template_cell.pop_size)
    if pop_size not in {(), (1,)}:
        raise ValueError(f"DBNN channel templates require exactly one Cell population member, got {pop_size!r}.")
    specs = tuple(_resolve_spec(spec) for spec in specs)
    identity_to_spec = {(spec.name, spec.synapse_type): spec_index for spec_index, spec in enumerate(specs)}
    if len(identity_to_spec) != len(specs):
        raise ValueError("DBNN synapse spec name/type identities must be unique.")

    synapses = template_cell.synapses
    synapse_ids = []
    placement_ids = []
    point_ids = []
    spec_indices = []
    branch_ids = []
    branch_xs = []
    unmatched = set()
    for row in range(len(synapses)):
        view = synapses[row]
        identity = (str(view.name[0]), str(view.synapse_type[0]))
        spec_index = identity_to_spec.get(identity)
        if spec_index is None:
            unmatched.add(identity)
            continue
        _validate_view_parameters(specs[spec_index], view)
        synapse_ids.append(int(view.id[0]))
        placement_ids.append(int(view.placement_id[0]))
        point_ids.append(int(view.point_id[0]))
        spec_indices.append(spec_index)
        branch_ids.append(int(view.branch_id[0]))
        branch_xs.append(float(view.branch_x[0]))
    if unmatched:
        raise ValueError(f"Detailed Cell synapses have no matching DBNN specs: {sorted(unmatched)!r}.")
    return ChannelLayout(
        specs,
        tuple(synapse_ids),
        tuple(placement_ids),
        tuple(point_ids),
        tuple(spec_indices),
        reference_weights_us,
        training_ranges,
        tuple(branch_ids),
        tuple(branch_xs),
    )


def align_channels(
    checkpoint_layout: ChannelLayout,
    candidate_layout: ChannelLayout,
    max_branch_x_distance: float = 0.05,
) -> ChannelAlignment:
    """Align candidate channels into checkpoint/model input order.

    Parameters
    ----------
    checkpoint_layout : ChannelLayout
        Channel layout stored with the checkpoint.
    candidate_layout : ChannelLayout
        Candidate runtime channel layout.
    max_branch_x_distance : float, optional
        Largest accepted normalized branch-coordinate distance.

    Returns
    -------
    ChannelAlignment
        Matching method and complete candidate-index permutation.

    Raises
    ------
    TypeError
        If either layout has the wrong type or the threshold is not numeric.
    ValueError
        If channel counts differ or the threshold is not finite and positive.
    """
    if not isinstance(checkpoint_layout, ChannelLayout):
        raise TypeError("checkpoint_layout must be a ChannelLayout.")
    if not isinstance(candidate_layout, ChannelLayout):
        raise TypeError("candidate_layout must be a ChannelLayout.")
    try:
        threshold = float(max_branch_x_distance)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_branch_x_distance must be a number.") from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("max_branch_x_distance must be finite and positive.")
    if checkpoint_layout.n_channels != candidate_layout.n_channels:
        raise ValueError(
            "Channel alignment requires equal channel counts, "
            f"got checkpoint={checkpoint_layout.n_channels} and candidate={candidate_layout.n_channels}."
        )
    channel_count = checkpoint_layout.n_channels
    if channel_count == 0:  # pragma: no cover - ChannelLayout enforces this invariant
        raise ValueError("Channel alignment requires at least one channel.")

    if checkpoint_layout.fingerprint == candidate_layout.fingerprint:
        return ChannelAlignment("identity", tuple(range(channel_count)))

    if checkpoint_layout.branch_ids and candidate_layout.branch_ids:
        checkpoint_groups = _location_groups(checkpoint_layout)
        candidate_groups = _location_groups(candidate_layout)
        if {key: len(value) for key, value in checkpoint_groups.items()} == {
            key: len(value) for key, value in candidate_groups.items()
        }:
            permutation = [-1] * channel_count
            within_threshold = True
            for key in sorted(checkpoint_groups):
                checkpoint_order = sorted(
                    checkpoint_groups[key],
                    key=lambda index: (
                        checkpoint_layout.branch_xs[index],
                        checkpoint_layout.synapse_ids[index],
                        index,
                    ),
                )
                candidate_order = sorted(
                    candidate_groups[key],
                    key=lambda index: (
                        candidate_layout.branch_xs[index],
                        candidate_layout.synapse_ids[index],
                        index,
                    ),
                )
                for checkpoint_index, candidate_index in zip(checkpoint_order, candidate_order):
                    if (
                        abs(
                            checkpoint_layout.branch_xs[checkpoint_index]
                            - candidate_layout.branch_xs[candidate_index]
                        )
                        > threshold
                        + 8.0
                        * np.finfo(np.float64).eps
                        * max(
                            1.0,
                            abs(checkpoint_layout.branch_xs[checkpoint_index]),
                            abs(candidate_layout.branch_xs[candidate_index]),
                            threshold,
                        )
                    ):
                        within_threshold = False
                    permutation[checkpoint_index] = candidate_index
            if within_threshold:
                return ChannelAlignment("branch_position", tuple(permutation))

    permutation = [-1] * channel_count
    checkpoint_order = sorted(
        range(channel_count), key=lambda index: (checkpoint_layout.synapse_ids[index], index)
    )
    candidate_order = sorted(
        range(channel_count), key=lambda index: (candidate_layout.synapse_ids[index], index)
    )
    for checkpoint_index, candidate_index in zip(checkpoint_order, candidate_order):
        permutation[checkpoint_index] = candidate_index
    return ChannelAlignment("identifier_distance", tuple(permutation))


def _location_groups(layout: ChannelLayout) -> dict[tuple[int, str], list[int]]:
    groups: dict[tuple[int, str], list[int]] = {}
    for channel_index, branch_id in enumerate(layout.branch_ids):
        spec = layout.spec(channel_index)
        key = (branch_id, json.dumps(_spec_to_dict(spec), sort_keys=True, separators=(",", ":")))
        groups.setdefault(key, []).append(channel_index)
    return groups


def _resolve_spec(spec: Synapse) -> Synapse:
    if not isinstance(spec, Synapse):
        raise TypeError(f"spec must be a BrainCell Synapse, got {type(spec).__name__!r}.")
    runtime_cls = get_registry().get("synapse", spec.synapse_type)
    unknown = set(spec.params).difference(runtime_cls.parameters)
    if unknown:
        raise TypeError(f"Synapse type {spec.synapse_type!r} has no parameters {tuple(sorted(unknown))!r}.")
    effective = {
        name: spec.params[name] if name in spec.params else parameter.default
        for name, parameter in runtime_cls.parameters.items()
    }
    for name, value in effective.items():
        runtime_cls.parameters[name].validate(value, name)
    runtime_cls.validate_parameter_values(effective)
    if "e" not in effective:
        raise ValueError("A DBNN synapse spec requires a reversal-potential parameter named 'e'.")
    canonical = {}
    for name, value in sorted(effective.items()):
        default = runtime_cls.parameters[name].default
        scalar = _canonical_parameter_scalar(value, default, name=name)
        canonical[name] = u.Quantity(scalar, default.unit) if isinstance(default, u.Quantity) else scalar
    return Synapse(spec.synapse_type, name=spec.name, **canonical)


def _spec_to_dict(spec: Synapse) -> dict[str, Any]:
    runtime_cls = get_registry().get("synapse", spec.synapse_type)
    return {
        "synapse_type": spec.synapse_type,
        "name": spec.name,
        "parameters": [
            [name, _canonical_parameter_scalar(value, runtime_cls.parameters[name].default, name=name)]
            for name, value in sorted(spec.params.items())
        ],
    }


def _spec_from_dict(value: dict[str, Any]) -> Synapse:
    synapse_type = str(value["synapse_type"])
    runtime_cls = get_registry().get("synapse", synapse_type)
    params = {}
    for name, scalar in value["parameters"]:
        default = runtime_cls.parameters[str(name)].default
        params[str(name)] = (
            u.Quantity(float(scalar), default.unit) if isinstance(default, u.Quantity) else float(scalar)
        )
    return Synapse(synapse_type, name=value.get("name"), **params)


def _validate_view_parameters(spec: Synapse, synapse_view: Any) -> None:
    runtime_cls = get_registry().get("synapse", spec.synapse_type)
    for name, expected in spec.params.items():
        default = runtime_cls.parameters[name].default
        expected_value = _canonical_parameter_scalar(expected, default, name=name)
        actual = synapse_view.get(name)
        if isinstance(default, u.Quantity):
            actual = actual.to_decimal(default.unit)
        values = np.asarray(actual, dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, expected_value):
            raise ValueError(f"Cell synapse {spec.instance_name!r} parameter {name!r} does not match its DBNN spec.")


def _canonical_parameter_scalar(value: Any, default: Any, *, name: str) -> float:
    if isinstance(default, u.Quantity):
        if not isinstance(value, u.Quantity):
            raise TypeError(f"Synapse parameter {name!r} must carry units compatible with {default.unit}.")
        value = value.to_decimal(default.unit)
    array = np.asarray(value, dtype=float)
    if array.shape != () or not np.isfinite(array).all():
        raise ValueError(f"DBNN synapse spec parameter {name!r} must be one finite scalar.")
    return float(array)


def packed_pair_index(i: int, j: int, n_channels: int) -> int:
    """Return the strict-upper packed index for channel pair ``(i, j)``."""
    if not 0 <= i < j < n_channels:
        raise ValueError(f"Expected 0 <= i < j < {n_channels}, got ({i}, {j}).")
    return i * (2 * n_channels - i - 1) // 2 + j - i - 1


def validate_channel_coverage(layout: ChannelLayout, spec_index: int, synapse_view: Any) -> np.ndarray:
    """Validate teacher replication and return channel IDs in view order."""
    if not 0 <= spec_index < len(layout.specs):
        raise ValueError(f"Invalid spec index {spec_index}.")
    spec = layout.specs[spec_index]
    expected_channels = {
        placement_id: (channel_id, point_id)
        for channel_id, (placement_id, point_id, channel_spec_index) in enumerate(
            zip(layout.placement_ids, layout.point_ids, layout.spec_indices)
        )
        if channel_spec_index == spec_index
    }
    pop_size = tuple(synapse_view.cell.pop_size)
    if len(pop_size) != 1:
        raise ValueError(f"DBNN channel coverage requires one-dimensional Cell pop_size, got {pop_size!r}.")
    populations = set(range(int(pop_size[0])))
    actual_pairs = list(
        zip(
            np.asarray(synapse_view.population_index, dtype=int).reshape(-1),
            np.asarray(synapse_view.placement_id, dtype=int).reshape(-1),
        )
    )
    expected_pairs = {(population, placement_id) for population in populations for placement_id in expected_channels}
    actual_pair_set = set(actual_pairs)
    if len(actual_pair_set) != len(actual_pairs) or actual_pair_set != expected_pairs:
        raise ValueError(
            "Placed population/placement rows do not match layout; "
            f"missing={sorted(expected_pairs - actual_pair_set)}, extra={sorted(actual_pair_set - expected_pairs)}."
        )
    if len(synapse_view):
        identities = set(zip(synapse_view.name.tolist(), synapse_view.synapse_type.tolist()))
        if identities != {(spec.name, spec.synapse_type)}:
            raise ValueError(f"Teacher synapse identities {identities!r} do not match spec {(spec.name, spec.synapse_type)!r}.")
        _validate_view_parameters(spec, synapse_view)
    channel_ids = []
    for placement_id, point_id in zip(synapse_view.placement_id.tolist(), synapse_view.point_id.tolist()):
        channel_id, expected_point_id = expected_channels[int(placement_id)]
        if int(point_id) != expected_point_id:
            raise ValueError(
                f"Teacher synapse placement {placement_id} resolved to point {point_id}, expected {expected_point_id}."
            )
        channel_ids.append(channel_id)
    return np.asarray(channel_ids, dtype=np.int64)


__all__ = [
    "ChannelAlignment",
    "ChannelLayout",
    "POLARITY_THRESHOLD_MV",
    "align_channels",
    "build_channel_layout",
    "packed_pair_index",
    "validate_channel_coverage",
]
