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
import jax
import numpy as np


class Population:
    """Resolved one-dimensional Network model population.

    A Population may own a :class:`Cell`, :class:`NetStim`, or
    :class:`EventSequence`. Custom fields share the attribute namespace with
    formal fields and are stored with a leading population axis.

    Parameters
    ----------
    name : str
        Population name.
    model : object
        Resolved model object.
    **fields
        Scalar or population-aligned custom metadata.
    """

    _FORMAL_NAMES = frozenset(
        {
            "name",
            "model",
            "cell",
            "kind",
            "size",
            "ids",
            "sources",
            "synapses",
            "connections",
            "fields",
            "event_outputs",
            "add_source",
            "set",
        }
    )

    def __init__(self, name: str, model, **fields) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Population name must be a non-empty string.")
        size, kind = _model_size_and_kind(model)
        overlap = self._FORMAL_NAMES.intersection(fields)
        if overlap:
            raise ValueError(f"Population custom fields conflict with formal names: {sorted(overlap)!r}.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "ids", _readonly_array(np.arange(size, dtype=np.int64)))
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_sources", {})
        self.set(**fields)

    @property
    def cell(self):
        """Return the owned Cell for compatibility with cell-only code."""
        if self.kind != "cell":
            raise TypeError(f"Population {self.name!r} owns {self.kind}, not a Cell.")
        return self.model

    @property
    def fields(self):
        """Read-only custom-field mapping."""
        return MappingProxyType(self._fields)

    @property
    def sources(self):
        """Read-only canonical event-output ports."""
        if self.kind == "cell":
            outputs = self.model.event_outputs
            result = {name: outputs[name] for name in outputs}
            result.update(self._sources)
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

    @property
    def event_outputs(self):
        """Compatibility alias for :attr:`sources`."""
        return self.sources

    def set(self, **fields) -> "Population":
        """Set custom population fields after shape validation.

        Parameters
        ----------
        **fields
            Scalars or arrays whose leading dimension equals ``size``.

        Returns
        -------
        Population
            This population.
        """
        overlap = self._FORMAL_NAMES.intersection(fields)
        if overlap:
            raise ValueError(f"Population custom fields conflict with formal names: {sorted(overlap)!r}.")
        prepared = {name: _population_field(value, self.size, name=name) for name, value in fields.items()}
        self._fields.update(prepared)
        return self

    def add_source(self, name: str, source) -> object:
        """Register an additional named live event output.

        Parameters
        ----------
        name : str
            Port name unique within this population.
        source : EventSource or EventSourceView
            Live source driven by this population's Cell.

        Returns
        -------
        EventSourceView
            Registered source view.
        """
        from braincell.event import EventSource, EventSourceView

        if self.kind != "cell":
            raise TypeError("Additional source ports are supported only for Cell populations.")
        if not isinstance(name, str) or not name:
            raise ValueError("Population source name must be a non-empty string.")
        if name in self.sources:
            raise ValueError(f"Population {self.name!r} already has a source port named {name!r}.")
        view = (
            source if isinstance(source, EventSourceView) else source.view if isinstance(source, EventSource) else None
        )
        if view is None:
            raise TypeError("Population.add_source(...) expects an EventSource or EventSourceView.")
        if view.owner.execution_owner is not self.model:
            raise ValueError("A Cell population source must be driven by that population's Cell.")
        if getattr(self.model, "_initialized", False):
            raise RuntimeError("Population source ports must be registered before Cell initialization.")
        self._sources[name] = view
        return view

    def __getitem__(self, name: str):
        if name in self._FORMAL_NAMES:
            return getattr(self, name)
        try:
            return self._fields[name]
        except KeyError as exc:
            raise KeyError(f"Population {self.name!r} has no field {name!r}.") from exc

    def __getattr__(self, name: str):
        fields = self.__dict__.get("_fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._FORMAL_NAMES:
            raise AttributeError(f"Population formal field {name!r} is read-only.")
        self.set(**{name: value})

    def __repr__(self) -> str:
        """Return a compact population summary."""
        return (
            f"Population(name={self.name!r}, kind={self.kind!r}, size={self.size}, "
            f"model={type(self.model).__name__}, fields={tuple(self._fields)!r}, "
            f"initialized={bool(getattr(self.model, '_initialized', False))})"
        )

    __str__ = __repr__


def _model_size_and_kind(model) -> tuple[int, str]:
    from braincell.event import EventSequence, NetStim

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


def _population_field(value, size: int, *, name: str):
    shape = tuple(getattr(value, "shape", ()))
    if shape == ():
        if isinstance(value, u.Quantity):
            return u.math.broadcast_to(value, (size,))
        return _readonly_array(np.full((size,), value))
    if shape[0] != size:
        raise ValueError(f"Population field {name!r} must be scalar or have leading dimension {size}; got {shape!r}.")
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
    spikes : dict
        ``population_name -> spike_trace`` mapping.
    """

    time: object
    traces: dict
    spikes: dict
    samples: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    start_time: object | None = None
    stop_time: object | None = None
    dt: object | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", _nested_mapping_proxy(self.traces))
        object.__setattr__(self, "spikes", MappingProxyType(dict(self.spikes)))
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
        spikes = {name: _concat_values(tuple(part.spikes[name] for part in parts)) for name in first.spikes}
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
            spikes=spikes,
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


def _concat_values(values):
    first = values[0]
    if isinstance(first, u.Quantity):
        unit = first.unit
        return u.Quantity(u.math.concatenate(tuple(value.to_decimal(unit) for value in values), axis=0), unit)
    return u.math.concatenate(values, axis=0)


def _concat_event_series(series):
    from braincell.recording import EventSeries

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


def _same_time(left, right) -> bool:
    if left is None or right is None:
        return left is right
    return bool(
        np.allclose(
            np.asarray(left.to_decimal(u.ms), dtype=float),
            np.asarray(right.to_decimal(u.ms), dtype=float),
            rtol=1e-7,
            atol=1e-9,
        )
    )
