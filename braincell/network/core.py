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

"""Core network declarations shared by topology and runtime layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import brainunit as u
import numpy as np

from braincell._misc import concat_values as _concat_values, same_time_quantity as _same_time


class Population:
    """Resolved one-dimensional Network model population.

    A Population may own a :class:`Cell`, :class:`NetStim`, or
    :class:`EventSequence`. Custom metadata shares the attribute namespace
    with reserved Population attributes and is stored with a leading
    population axis.

    Parameters
    ----------
    name : str
        Population name.
    model : object
        Resolved model object.
    **metadata
        Scalar or population-aligned custom metadata.
    """

    _RESERVED_NAMES = frozenset(
        {
            "name",
            "model",
            "cell",
            "kind",
            "size",
            "ids",
            "event_outputs",
            "synapses",
            "connections",
            "metadata",
            "register_event_output",
            "set",
        }
    )

    def __init__(self, name: str, model, **metadata) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Population name must be a non-empty string.")
        size, kind = _model_size_and_kind(model)
        overlap = self._RESERVED_NAMES.intersection(metadata)
        if overlap:
            raise ValueError(f"Population metadata conflicts with reserved names: {sorted(overlap)!r}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "ids", _readonly_array(np.arange(size, dtype=np.int64)))
        object.__setattr__(self, "_metadata", {})
        object.__setattr__(self, "_event_outputs", {})
        self.set(**metadata)

    @property
    def cell(self):
        """Return the owned Cell for compatibility with cell-only code."""
        if self.kind != "cell":
            raise TypeError(f"Population {self.name!r} owns {self.kind}, not a Cell.")
        return self.model

    @property
    def metadata(self):
        """Return the read-only custom metadata mapping."""
        return MappingProxyType(self._metadata)

    @property
    def event_outputs(self):
        """Return the read-only named event-output ports."""
        if self.kind == "cell":
            outputs = self.model.event_outputs
            result = {name: outputs[name] for name in outputs}
            result.update(self._event_outputs)
            return MappingProxyType(result)
        port = "spike" if self.kind == "netstim" else "event"
        return MappingProxyType({port: self.model.view})

    @property
    def synapses(self):
        """Return logical synapses owned by a Cell population."""
        if self.kind != "cell":
            raise TypeError(f"Population {self.name!r} owns {self.kind}, not a Cell with synapses.")
        return self.model.synapses

    @property
    def connections(self):
        """Return routing rows targeting a Cell population."""
        if self.kind != "cell":
            raise TypeError(f"Population {self.name!r} owns {self.kind}, not a Cell connection target.")
        return self.model.connections

    def set(self, **metadata) -> "Population":
        """Set custom population metadata after shape validation.

        Parameters
        ----------
        **metadata
            Scalars or arrays whose leading dimension equals ``size``.

        Returns
        -------
        Population
            This population.
        """
        overlap = self._RESERVED_NAMES.intersection(metadata)
        if overlap:
            raise ValueError(f"Population metadata conflicts with reserved names: {sorted(overlap)!r}.")
        prepared = {name: _population_metadata(value, self.size, name=name) for name, value in metadata.items()}
        self._metadata.update(prepared)
        return self

    def register_event_output(self, source, *, name: str | None = None) -> object:
        """Register an additional live event output.

        Parameters
        ----------
        source : EventSource or EventSourceView
            Live source driven by this population's Cell.
        name : str, optional
            Port name unique within this population. Defaults to the source's
            semantic name.

        Returns
        -------
        EventSourceView
            Full registered source-owner view.
        """
        port, view, pending = self._prepare_event_output_registration(source, name=name)
        if pending:
            self._commit_event_output_registration(port, view)
        return view

    def _prepare_event_output_registration(self, source, *, name: str | None = None):
        """Validate one event-output registration without mutating the Population."""
        from .event import EventSource, EventSourceView

        if self.kind != "cell":
            raise TypeError("Additional event outputs are supported only for Cell populations.")
        if isinstance(source, EventSourceView):
            owner = source.owner
        elif isinstance(source, EventSource):
            owner = source
        else:
            raise TypeError("Population.register_event_output(...) expects an EventSource or EventSourceView.")
        if owner.execution_owner is not self.model:
            raise ValueError("A Cell population event output must be driven by that population's Cell.")
        if getattr(self.model, "_initialized", False):
            raise RuntimeError("Population event outputs must be registered before Cell initialization.")

        for existing_name, existing_view in self.event_outputs.items():
            if existing_view.owner is not owner:
                continue
            if name is not None and name != existing_name:
                raise ValueError(
                    f"EventSource is already registered as {existing_name!r} on Population {self.name!r}; "
                    f"cannot also register it as {name!r}."
                )
            return existing_name, existing_view, False

        port = owner.source_name if name is None else name
        if not isinstance(port, str) or not port:
            raise ValueError(
                "Additional EventSource used by a Network must have a non-empty source name; "
                "set source.name or call population.register_event_output(source, name=...)."
            )
        if port in self.event_outputs:
            raise ValueError(f"Population {self.name!r} already has a different event output named {port!r}.")
        return port, owner.view, True

    def _commit_event_output_registration(self, name: str, view) -> None:
        """Commit an event-output registration validated by the prepare step."""
        self._event_outputs[name] = view

    def __getitem__(self, name: str):
        if name in self._RESERVED_NAMES:
            return getattr(self, name)
        try:
            return self._metadata[name]
        except KeyError as exc:
            raise KeyError(f"Population {self.name!r} has no metadata named {name!r}.") from exc

    def __getattr__(self, name: str):
        metadata = self.__dict__.get("_metadata", {})
        if name in metadata:
            return metadata[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._RESERVED_NAMES:
            raise AttributeError(f"Population attribute {name!r} is reserved and read-only.")
        self.set(**{name: value})

    def __repr__(self) -> str:
        """Return a compact population summary."""
        return (
            f"Population(name={self.name!r}, kind={self.kind!r}, size={self.size}, "
            f"model={type(self.model).__name__}, metadata={tuple(self._metadata)!r}, "
            f"initialized={bool(getattr(self.model, '_initialized', False))})"
        )

    __str__ = __repr__


def _model_size_and_kind(model) -> tuple[int, str]:
    from .event import EventSequence, NetStim

    if isinstance(model, (NetStim, EventSequence)):
        return int(model.size), "netstim" if isinstance(model, NetStim) else "event_sequence"
    if not hasattr(model, "pop_size"):
        raise TypeError(f"Population model must be a Cell, NetStim, or EventSequence; got {type(model).__name__!s}.")
    pop_size = tuple(model.pop_size)
    if not pop_size:
        raise ValueError("Network v1 requires one-dimensional cell.pop_size; got scalar pop_size ().")
    if len(pop_size) != 1:
        raise ValueError(
            "Network v1 requires one-dimensional cell.pop_size; "
            "Cell supports multidimensional pop_size, but network indexing "
            "does not yet; "
            f"got {pop_size!r}."
        )
    size = int(pop_size[0])
    if size <= 0:
        raise ValueError(f"Population size must be > 0, got {pop_size!r}.")
    return size, "cell"


def _population_metadata(value, size: int, *, name: str):
    shape = tuple(getattr(value, "shape", ()))
    if shape == ():
        if isinstance(value, u.Quantity):
            return u.math.broadcast_to(value, (size,))
        return _readonly_array(np.full((size,), value))
    if shape[0] != size:
        raise ValueError(
            f"Population metadata {name!r} must be scalar or have leading dimension {size}; got {shape!r}."
        )
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    return value


def _readonly_array(value) -> np.ndarray:
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class NetworkResult:
    """Immutable result of :meth:`braincell.network.Network.run`.

    Attributes
    ----------
    time : brainunit.Quantity
        Step times spanning ``[start_t, start_t + duration)``.
    traces : dict
        ``population_name -> {probe_name: trace}`` mapping.
    events : dict
        ``population_name -> {port_name: EventSeries}`` mapping.
    """

    time: object
    traces: dict
    samples: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    start_time: object | None = None
    stop_time: object | None = None
    dt: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", _nested_mapping_proxy(self.traces))
        object.__setattr__(self, "samples", _nested_mapping_proxy(self.samples))
        object.__setattr__(self, "events", _nested_mapping_proxy(self.events))

    @classmethod
    def concat(cls, parts) -> "NetworkResult":
        """Concatenate contiguous results with matching recording schemas.

        Parameters
        ----------
        parts : iterable of NetworkResult
            Ordered contiguous segments.

        Returns
        -------
        NetworkResult
            Concatenated immutable result.
        """
        from braincell._multi_compartment.run import RunResult

        parts = tuple(parts)
        if not parts:
            raise ValueError("NetworkResult.concat(...) requires at least one part.")
        first = parts[0]
        if first.dt is None:
            raise ValueError("NetworkResult.concat(...) requires results carrying dt metadata.")
        for previous, current in zip(parts, parts[1:]):
            if not _same_time(previous.dt, current.dt):
                raise ValueError("NetworkResult.concat(...) requires identical dt values.")
            if not _same_time(previous.stop_time, current.start_time):
                raise ValueError("NetworkResult.concat(...) requires contiguous segments.")
            if tuple(previous.samples) != tuple(current.samples):
                raise ValueError("NetworkResult.concat(...) requires identical sample populations.")
            if tuple(previous.events) != tuple(current.events):
                raise ValueError("NetworkResult.concat(...) requires identical event populations.")
            for population_name in previous.samples:
                if tuple(previous.samples[population_name]) != tuple(current.samples[population_name]):
                    raise ValueError(f"Recording names changed for population {population_name!r}.")
            for population_name in previous.events:
                if tuple(previous.events[population_name]) != tuple(current.events[population_name]):
                    raise ValueError(f"Event ports changed for population {population_name!r}.")
        samples = {}
        for population_name in first.samples:
            samples[population_name] = {}
            for recording_name in first.samples[population_name]:
                pseudo = tuple(
                    RunResult(
                        time=part.time,
                        traces={recording_name: part.samples[population_name][recording_name].values},
                        samples={recording_name: part.samples[population_name][recording_name]},
                        start_time=part.start_time,
                        stop_time=part.stop_time,
                        dt=part.dt,
                    )
                    for part in parts
                )
                samples[population_name][recording_name] = RunResult.concat(pseudo).samples[recording_name]
        traces = {
            population_name: {
                name: _concat_values(tuple(part.traces[population_name][name] for part in parts))
                for name in first.traces.get(population_name, {})
            }
            for population_name in first.traces
        }
        events = {
            population_name: {
                port: _concat_event_series(tuple(part.events[population_name][port] for part in parts))
                for port in first.events[population_name]
            }
            for population_name in first.events
        }
        return cls(
            time=_concat_values(tuple(part.time for part in parts)),
            traces=traces,
            samples=samples,
            events=events,
            start_time=first.start_time,
            stop_time=parts[-1].stop_time,
            dt=first.dt,
        )


# Transitional public name retained for existing callers.
NetworkRunResult = NetworkResult


def _nested_mapping_proxy(value):
    return MappingProxyType(
        {key: MappingProxyType(dict(item)) if isinstance(item, dict) else item for key, item in dict(value).items()}
    )


def _concat_event_series(series):
    from .recording import EventSeries

    unit = series[0].time.unit
    return EventSeries(
        time=u.Quantity(
            u.math.concatenate(tuple(item.time.to_decimal(unit) for item in series), axis=0),
            unit,
        ),
        source_id=np.concatenate(tuple(item.source_id for item in series)),
        count=np.concatenate(tuple(item.count for item in series)),
        metadata=series[0].metadata,
    )
