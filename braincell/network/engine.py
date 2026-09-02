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

"""Network runtime loop."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import numpy as np

from braincell._misc import freeze_array, normalize_loop_outputs, scalar_decimal, validate_time_quantity
from braincell._multi_compartment import probes as _probes
from braincell._multi_compartment.run import _duration_steps, _recording_time_mask
from braincell._multi_compartment.synapses import SynapseView
from .connection import NetworkConnections, PairingSpec, _UNSET, _connect_with_pairing_seed
from .recording import EventSeries, SampleBlock

from .core import NetworkResult, Population
from .delivery import (
    DeliveryBlock,
    advance_delivery_state,
    apply_immediate_events,
    create_delivery_state,
    delivery_blocks as build_delivery_blocks,
    enqueue_future_events,
    make_delivery_op,
    normalize_event_backend,
    resolve_event_backend,
    write_arrivals,
)
from .lowering import lower_direct_connections
from braincell.mech import Synapse


@dataclass(frozen=True)
class _RunSetup:
    """Reusable topology/backend data for one ``Network.run`` configuration."""

    delivery_blocks: tuple[DeliveryBlock, ...]
    delivery_ops: tuple
    ordered_population_names: tuple[str, ...]
    probe_names: dict[str, tuple[str, ...]]
    compiled_recordings: dict[str, tuple]
    event_sources: dict[str, tuple]
    n_trace: int
    n_recording: int
    n_event: int


@dataclass(frozen=True)
class _CachedRunLoop:
    """Compiled network scan plus its reusable delivery buffers."""

    runner: object
    delivery_state: object


class Network:
    """Named population network using existing ``Cell`` runtimes."""

    def __init__(self, name: str | None = None, *, seed: int = 0) -> None:
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Network name must be a non-empty string or None.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("Network seed must be an integer.")
        self.name = name
        self.seed = int(seed)
        self.populations: dict[str, Population] = {}
        self._topology_version = 0
        self._run_setup_cache: dict[tuple, _RunSetup] = {}
        self._network_run_loop_cache: dict[tuple, _CachedRunLoop] = {}
        self._delivery_state_cache: dict[tuple, object] = {}
        self._runtime_config: tuple | None = None
        self._scheduled_dt_ms: float | None = None
        self._cell_lifecycle_active = False
        self._initialized = False
        self._source_current_time = 0.0 * u.ms

    def _raise_if_initialized(self, action: str) -> None:
        if self._initialized:
            raise RuntimeError(f"Cannot {action} after Network initialization.")

    @contextlib.contextmanager
    def _cell_lifecycle(self):
        """Mark the network as driving its cells' init/reset transitions.

        ``Cell`` consults ``_cell_lifecycle_active`` to tell a network-driven
        transition from a user calling ``cell.init_state()`` directly
        (``_multi_compartment/cell.py:862``, ``:888``). Both
        :meth:`init_state` and :meth:`reset_state` need the flag cleared even
        when a cell raises, so the ``try``/``finally`` lives here once.
        """
        self._cell_lifecycle_active = True
        try:
            yield
        finally:
            self._cell_lifecycle_active = False

    def _mark_topology_changed(self) -> None:
        self._topology_version += 1
        self._run_setup_cache.clear()
        self._network_run_loop_cache.clear()
        self._delivery_state_cache.clear()

    def __repr__(self) -> str:
        """Return a compact network summary."""
        return (
            f"Network(name={self.name!r}, populations={len(self.populations)}, "
            f"connections={self.connections.n_connections}, rows={self.connections.n_rows})"
        )

    @property
    def connections(self) -> NetworkConnections:
        """Return target-population-scoped direct connections."""
        return NetworkConnections(self)

    def __str__(self) -> str:
        """Return a readable multi-line network summary."""
        lines = [repr(self)]
        lines.append("  populations:")
        if self.populations:
            for name, population in self.populations.items():
                initialized = bool(getattr(population.model, "_initialized", False))
                lines.append(
                    f"    {name}: size={population.size}, kind={population.kind}, "
                    f"model={type(population.model).__name__}, initialized={initialized}"
                )
        else:
            lines.append("    <none>")

        lines.append("  connections:")
        if len(self.connections):
            for target in self.connections.target_names:
                view = self.connections[target]
                lines.append(f"    {target}: connections={len(view.connect_names)}, rows={len(view)}")
                for name in view.connect_names:
                    lines.append(f"      {self.connections.describe(target, name)}")
        else:
            lines.append("    <none>")
        return "\n".join(lines)

    def add_population(self, name: str, model, **metadata) -> Population:
        """Resolve and add one named model population.

        Parameters
        ----------
        name : str
            Unique non-empty population name.
        model : Cell, NetStim, EventSequence, or callable
            Model owner or zero-argument provider returning one.
        **metadata
            Scalar or population-aligned custom metadata.

        Returns
        -------
        Population
            Resolved Network-owned population.
        """
        self._raise_if_initialized("add a population")
        from .event import EventSource

        if callable(model) and not isinstance(model, EventSource) and not hasattr(model, "pop_size"):
            model = model()
        population = Population(name, model, **metadata)
        if population.name in self.populations:
            raise ValueError(f"Network already has a population named {population.name!r}.")
        if any(existing.model is model for existing in self.populations.values()):
            raise ValueError("The same model object cannot be registered as more than one Network population.")
        if population.kind == "cell":
            model._bind_network_owner(self)
        elif population.kind == "netstim":
            model._bind_network_seed(self.seed, population.name)
        self.populations[population.name] = population
        self._mark_topology_changed()
        return population

    def connect(
        self,
        name: str,
        *,
        source,
        synapse,
        target=None,
        locations=None,
        pairing: PairingSpec | None = None,
        weight=_UNSET,
        delay=0.0 * u.ms,
    ):
        """Connect a registered source to existing or newly placed synapses.

        Parameters
        ----------
        name : str
            Connection name, unique within the target Cell.
        source : Population, EventSource, or EventSourceView
            Registered event source or a selected source view.
        synapse : SynapseView or Synapse
            Existing logical synapses, or one declaration to place before
            connecting.
        target : Population or CellView, optional
            Registered Cell target. Required with a ``Synapse`` and
            forbidden with an existing ``SynapseView``.
        locations : LocsetExpr, LocsetMask, LocsetBatch, or sequence, optional
            Locations forwarded to ``target.place`` for a ``Synapse``.
        pairing : PairingSpec, optional
            Endpoint sampling declaration. Supported only when ``synapse`` is
            an existing ``SynapseView``.
        weight : Quantity, optional
            Scalar or row-aligned event payload.
        delay : Quantity, optional
            Scalar or row-aligned non-negative delivery delay.

        Returns
        -------
        ConnectionView
            Concrete target-owned routing rows.
        """
        self._raise_if_initialized("add a connection")
        source_population = self._require_registered_source(source)
        pending_event_output = self._prepare_source_event_output_registration(source_population, source)

        if isinstance(synapse, SynapseView):
            if target is not None or locations is not None:
                raise TypeError("Existing SynapseView connections do not accept target or locations.")
            target_population = self._require_registered_cell(synapse.cell, role="synapse target")
            result = _connect_with_pairing_seed(
                name,
                source=source,
                synapse=synapse,
                pairing=pairing,
                weight=weight,
                delay=delay,
                pairing_seed_root=self.seed,
                pairing_seed_path=(source_population.name, target_population.name, name),
            )
            self._commit_source_event_output_registration(source_population, pending_event_output)
            self._mark_topology_changed()
            return result

        if not isinstance(synapse, Synapse):
            raise TypeError("Network.connect synapse must be a SynapseView or Synapse.")
        if pairing is not None:
            raise TypeError("Network.connect pairing requires an existing SynapseView.")
        if target is None or locations is None:
            raise TypeError("Synapse connections require both target and locations.")
        target_scope, target_cell = self._resolve_registered_target(target)
        previous_rules = target_cell._place_rules
        previous_origins = dict(target_cell._synapse_origins)
        previous_ids = set(target_cell.synapses.id.tolist())
        try:
            target_scope.place(locations, synapse)
            new_ids = np.asarray(
                [item for item in target_cell.synapses.id.tolist() if item not in previous_ids],
                dtype=np.int64,
            )
            if new_ids.size == 0:
                raise ValueError("Network.connect locations produced no synapses.")
            result = _connect_with_pairing_seed(
                name,
                source=source,
                synapse=SynapseView(target_cell, new_ids),
                pairing=None,
                weight=weight,
                delay=delay,
                pairing_seed_root=self.seed,
                pairing_seed_path=(),
            )
        except Exception:
            target_cell._place_rules = previous_rules
            target_cell._synapse_origins = previous_origins
            target_cell._invalidate_discretization_cache()
            raise
        self._commit_source_event_output_registration(source_population, pending_event_output)
        self._mark_topology_changed()
        return result

    def init_state(self, batch_size=None) -> "Network":
        """Initialize all population cell runtime states.

        Parameters
        ----------
        batch_size : int, optional
            Optional batch size forwarded to uninitialized cell
            ``init_state`` calls.

        Returns
        -------
        Network
            This network, for fluent setup code.

        Notes
        -----
        This method is idempotent at the network level: already initialized
        cells are left unchanged because ``Cell.init_state`` itself is a
        one-shot declaration-to-runtime transition.
        """
        _reject_batch_size(batch_size)
        if self._initialized:
            return self
        self._validate_direct_source_ownership()
        with self._cell_lifecycle():
            for population in self._cell_populations().values():
                if not getattr(population.cell, "_initialized", False):
                    population.cell.init_state()
        self._initialized = True
        return self

    def reset_state(self, batch_size=None) -> "Network":
        """Reset all population cell dynamic states.

        Parameters
        ----------
        batch_size : int, optional
            Optional batch size forwarded to cell ``init_state`` and
            ``reset_state`` calls.

        Returns
        -------
        Network
            This network, for fluent setup code.

        Notes
        -----
        ``Network.reset_state`` resets runtime state in place. It does not call
        ``Cell.reset()``, which would tear down runtime objects and return the
        cell to the declaration phase.
        """
        _reject_batch_size(batch_size)
        uninitialized = [
            name
            for name, population in self._cell_populations().items()
            if not getattr(population.cell, "_initialized", False)
        ]
        if uninitialized:
            raise RuntimeError(
                "Network.reset_state() requires initialized population Cells; "
                f"call Network.init_state() or Network.run() first (uninitialized={uninitialized!r})."
            )
        with self._cell_lifecycle():
            for population in self._cell_populations().values():
                population.cell.reset_state()
        for state in self._delivery_state_cache.values():
            _reset_delivery_state(state)
        self._source_current_time = 0.0 * u.ms
        return self

    def run(
        self,
        *,
        dt,
        duration,
        delay_quantization: str = "nearest",
        event_backend: str = "auto",
        brainevent_backend: str | None = "jax_raw",
    ) -> NetworkResult:
        """Run the network for ``duration`` at fixed step ``dt``."""
        validate_time_quantity(dt, name="dt", prefix="Network.run(...)")
        validate_time_quantity(duration, name="duration", prefix="Network.run(...)")
        event_backend = normalize_event_backend(event_backend)
        if not self.populations:
            raise ValueError("Network.run(...) requires at least one population.")
        self.init_state()
        if not self._cell_populations():
            return self._run_scheduled_sources_only(dt=dt, duration=duration)
        setup_key = self._run_setup_cache_key(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        runtime_config = setup_key[2:]
        if self._runtime_config is None:
            self._runtime_config = runtime_config
        elif self._runtime_config != runtime_config:
            raise RuntimeError(
                "Network runtime configuration is fixed after the first run; "
                "dt, delay quantization, and event backend must remain unchanged."
            )
        setup = self._run_setup(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        ordered_population_names = setup.ordered_population_names
        start_t = self._common_start_time(ordered_population_names)
        n_steps = _duration_steps(duration, dt)
        relative_times = u.math.arange(n_steps) * dt
        probe_names = setup.probe_names
        n_trace = setup.n_trace
        n_recording = setup.n_recording
        n_event = setup.n_event
        cached_loop = self._network_run_loop(
            setup=setup,
            setup_key=setup_key,
            dt=dt,
            n_steps=int(relative_times.shape[0]),
        )
        times, samples_over_time = cached_loop.runner(start_t, relative_times)

        samples_tuple = normalize_loop_outputs(
            samples_over_time,
            count=n_recording + n_trace + n_event,
            prefix="Network.run(...)",
            noun="scan outputs",
        )
        recording_values = samples_tuple[:n_recording]
        trace_values = samples_tuple[n_recording : n_recording + n_trace]
        event_values = samples_tuple[n_recording + n_trace :]

        samples = {}
        recording_index = 0
        stop_t = start_t + n_steps * dt
        for population_name in ordered_population_names:
            samples[population_name] = {}
            for compiled in setup.compiled_recordings[population_name]:
                values = recording_values[recording_index]
                recording_index += 1
                mask = _recording_time_mask(times, compiled.schema)
                selected = values[mask]
                first_time = None if not np.any(mask) else times[int(np.flatnonzero(mask)[0])]
                samples[population_name][compiled.spec.name] = SampleBlock(
                    values=selected,
                    schema=compiled.schema,
                    segment_start=start_t,
                    segment_stop=stop_t,
                    first_time=first_time,
                )
        traces = {}
        index = 0
        for name in ordered_population_names:
            traces[name] = {}
            for probe_name in probe_names[name]:
                traces[name][probe_name] = trace_values[index]
                index += 1
            traces[name].update({recording_name: block.values for recording_name, block in samples[name].items()})
        events = self._result_events(
            times=times,
            start_t=start_t,
            stop_t=stop_t,
            event_sources=setup.event_sources,
            event_values=event_values,
        )
        self._source_current_time = stop_t
        return NetworkResult(
            time=times,
            traces=traces,
            samples=samples,
            events=events,
            start_time=start_t,
            stop_time=stop_t,
            dt=dt,
        )

    def _run_setup(
        self,
        *,
        dt,
        delay_quantization: str,
        event_backend: str,
        brainevent_backend: str | None,
    ) -> _RunSetup:
        cache_key = self._run_setup_cache_key(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        setup = self._run_setup_cache.get(cache_key)
        if setup is not None:
            return setup

        blocks = lower_direct_connections(
            self.populations,
            dt=dt,
            delay_quantization=delay_quantization,
        )
        delivery_backend = resolve_event_backend(event_backend)
        grouped_by_delay = delivery_backend == "brainevent"
        delivery_blocks = build_delivery_blocks(blocks, group_by_delay=grouped_by_delay)
        # A delivery op is only ever indexed for a block with a scalar delay,
        # which is exactly what grouping by delay produces. Without grouping
        # every block carries a per-contact delay vector and the vector path
        # in enqueue_future_events/apply_immediate_events runs instead -- so
        # building the ops there would materialize two device index arrays per
        # block, and close over the block, for something never called. The two
        # cannot disagree: both read grouped_by_delay.
        delivery_ops = (
            tuple(
                make_delivery_op(
                    block,
                    pre_size=self.populations[block.source.pre_population].size,
                    backend=delivery_backend,
                    brainevent_backend=brainevent_backend,
                )
                for block in delivery_blocks
            )
            if grouped_by_delay
            else (None,) * len(delivery_blocks)
        )
        ordered_population_names = tuple(self._cell_populations())
        # Read the probe keys without evaluating the probes: sample_probes()
        # runs real gathers outside jit just to discard everything but the
        # names. probe_names() reads the same ordering off the layouts.
        probe_names = {
            name: tuple(sorted(_probes.probe_names(population.cell)))
            for name, population in self._cell_populations().items()
        }
        compiled_recordings = {
            name: population.cell._compiled_recordings(dt) for name, population in self._cell_populations().items()
        }
        event_sources = {
            name: tuple(population.event_outputs.items()) for name, population in self._cell_populations().items()
        }
        setup = _RunSetup(
            delivery_blocks=delivery_blocks,
            delivery_ops=delivery_ops,
            ordered_population_names=ordered_population_names,
            probe_names=probe_names,
            compiled_recordings=compiled_recordings,
            event_sources=event_sources,
            n_trace=sum(len(names) for names in probe_names.values()),
            n_recording=sum(len(items) for items in compiled_recordings.values()),
            n_event=sum(len(items) for items in event_sources.values()),
        )
        self._run_setup_cache[cache_key] = setup
        return setup

    def _run_setup_cache_key(
        self,
        *,
        dt,
        delay_quantization: str,
        event_backend: str,
        brainevent_backend: str | None,
    ) -> tuple:
        dt_ms = scalar_decimal(dt, u.ms)
        runtime_ids = tuple(
            (name, id(population.cell.runtime), population.size)
            for name, population in self._cell_populations().items()
        )
        return (
            self._topology_version,
            runtime_ids,
            dt_ms,
            delay_quantization,
            event_backend,
            brainevent_backend,
        )

    def _network_run_loop(
        self,
        *,
        setup: _RunSetup,
        setup_key: tuple,
        dt,
        n_steps: int,
    ) -> _CachedRunLoop:
        key = (
            setup_key,
            int(n_steps),
            int(brainstate.environ.get_precision()),
        )
        cached = self._network_run_loop_cache.get(key)
        if cached is None:
            delivery_state = self._delivery_state_cache.get(setup_key)
            if delivery_state is None:
                delivery_state = create_delivery_state(
                    setup.delivery_blocks,
                    delivery_ops=setup.delivery_ops,
                )
                self._delivery_state_cache[setup_key] = delivery_state
            cached = _CachedRunLoop(
                runner=self._make_network_run_loop(
                    setup=setup,
                    dt=dt,
                    delivery_state=delivery_state,
                ),
                delivery_state=delivery_state,
            )
            self._network_run_loop_cache[key] = cached
        return cached

    def _make_network_run_loop(
        self,
        *,
        setup: _RunSetup,
        dt,
        delivery_state,
    ):
        """Create a persistent jitted scan for one network run shape."""
        delivery_blocks = setup.delivery_blocks
        ordered_population_names = setup.ordered_population_names
        probe_names = setup.probe_names
        compiled_recordings = setup.compiled_recordings
        event_sources = setup.event_sources

        def _run_loop(start_t, relative_times):
            times = start_t + relative_times
            with brainstate.environ.context(dt=dt):

                def _step(t):
                    with brainstate.environ.context(t=t):
                        with jax.named_scope("braincell:network_run:sample_recordings"):
                            recording_snapshots = tuple(
                                compiled.sample()
                                for name in ordered_population_names
                                for compiled in compiled_recordings[name]
                            )
                        with jax.named_scope("braincell:network_run:write_arrivals"):
                            write_arrivals(delivery_state, populations=self.populations)
                        with jax.named_scope("braincell:network_run:prepare_inputs"):
                            for name in ordered_population_names:
                                self.populations[name].cell._prepare_next_synapse_inputs()
                        with jax.named_scope("braincell:network_run:begin_cells"):
                            for name in ordered_population_names:
                                self.populations[name].cell._begin_step()
                        with jax.named_scope("braincell:network_run:update_cells"):
                            for name in ordered_population_names:
                                cell = self.populations[name].cell
                                cell._update_dynamics()
                        with jax.named_scope("braincell:network_run:apply_zero_delay_events"):
                            apply_immediate_events(
                                delivery_blocks,
                                delivery_state,
                                populations=self.populations,
                            )
                        with jax.named_scope("braincell:network_run:sample_probes"):
                            snapshots = {
                                name: self.populations[name].cell.sample_probes() for name in ordered_population_names
                            }
                        with jax.named_scope("braincell:network_run:record_events"):
                            events = tuple(
                                view.owner.current_event_count(view.source_id)
                                for name in ordered_population_names
                                for _, view in event_sources[name]
                            )
                        with jax.named_scope("braincell:network_run:enqueue_events"):
                            enqueue_future_events(delivery_blocks, delivery_state)
                        with jax.named_scope("braincell:network_run:advance_delivery"):
                            advance_delivery_state(delivery_state)
                        with jax.named_scope("braincell:network_run:pack_traces"):
                            traces = tuple(
                                snapshots[name][probe_name]
                                for name in ordered_population_names
                                for probe_name in probe_names[name]
                            )
                        return recording_snapshots + traces + events

                samples_over_time = brainstate.transform.for_loop(_step, times)

            end_t = start_t + int(times.shape[0]) * dt
            for population in self._cell_populations().values():
                population.cell._set_current_time(end_t)
            return times, samples_over_time

        return brainstate.transform.jit(_run_loop)

    def _common_start_time(self, names: tuple[str, ...]):
        first = self.populations[names[0]].cell.current_time
        for name in names[1:]:
            current = self.populations[name].cell.current_time
            if not np.allclose(
                np.asarray(current.to_decimal(u.ms), dtype=float),
                np.asarray(first.to_decimal(u.ms), dtype=float),
            ):
                raise ValueError("Network populations must have the same current_time.")
        return first

    def _cell_populations(self) -> dict[str, Population]:
        return {name: population for name, population in self.populations.items() if population.kind == "cell"}

    def _require_registered_cell(self, cell, *, role: str) -> Population:
        for population in self.populations.values():
            if population.kind == "cell" and population.cell is cell:
                return population
        raise RuntimeError(f"Network.connect {role} is not registered in this Network.")

    def _require_registered_source(self, source) -> Population:
        from .event import EventSource, EventSourceView

        if isinstance(source, Population):
            registered = self.populations.get(source.name)
            if registered is not source:
                raise RuntimeError("Network.connect source Population is not registered in this Network.")
            return source
        if isinstance(source, EventSourceView):
            source = source.owner
        if not isinstance(source, EventSource):
            raise TypeError("Network.connect source must be a Population, EventSource, or EventSourceView.")
        owner = source if source.is_scheduled else source.execution_owner
        for population in self.populations.values():
            if population.model is owner:
                return population
        raise RuntimeError("Network.connect source owner is not registered in this Network.")

    @staticmethod
    def _prepare_source_event_output_registration(source_population: Population, source):
        """Validate automatic publication of a live Cell event source."""
        from .event import EventSource, EventSourceView

        if isinstance(source, Population) or source_population.kind != "cell":
            return None
        if isinstance(source, EventSourceView):
            source = source.owner
        if not isinstance(source, EventSource):
            return None
        prepared = source_population._prepare_event_output_registration(source)
        return prepared if prepared[2] else None

    @staticmethod
    def _commit_source_event_output_registration(source_population: Population, prepared) -> None:
        """Publish a prevalidated source after its Connection succeeds."""
        if prepared is None:
            return
        name, view, _ = prepared
        source_population._commit_event_output_registration(name, view)

    def _resolve_registered_target(self, target):
        if isinstance(target, Population):
            registered = self.populations.get(target.name)
            if registered is not target:
                raise RuntimeError("Network.connect target Population is not registered in this Network.")
            if target.kind != "cell":
                raise TypeError(f"Network.connect target population owns {target.kind}, not a Cell.")
            return target.cell, target.cell
        target_cell = getattr(target, "root", None)
        if target_cell is None or not hasattr(target, "place"):
            raise TypeError("Network.connect target must be a Cell Population or CellView.")
        self._require_registered_cell(target_cell, role="target")
        return target, target_cell

    def _validate_direct_source_ownership(self) -> None:
        registered_models = {id(population.model) for population in self.populations.values()}
        for post in self._cell_populations().values():
            for connection in post.cell.connections._call_views():
                source = connection.source
                owner = source if source.is_scheduled else source.execution_owner
                if id(owner) not in registered_models:
                    source_name = source.source_name or source.source_type
                    raise RuntimeError(
                        f"EventSource {source_name!r} is outside this Network execution scope; "
                        "register it with Network.add_population()."
                    )

    def _result_events(
        self,
        *,
        times,
        start_t,
        stop_t,
        event_sources,
        event_values,
    ) -> dict:
        events = {}
        start_ms = scalar_decimal(start_t, u.ms)
        stop_ms = scalar_decimal(stop_t, u.ms)
        event_index = 0
        for name, population in self.populations.items():
            if population.kind != "cell":
                table = population.model.events
                event_ms = np.asarray(table.time.to_decimal(u.ms), dtype=float)
                selected = (event_ms >= start_ms) & (event_ms < stop_ms)
                selected_rows = np.flatnonzero(selected)
                if selected_rows.size:
                    order = np.lexsort((table.source_index[selected_rows], event_ms[selected_rows]))
                    selected_rows = selected_rows[order]
                port = "spike" if population.kind == "netstim" else "event"
                count = np.ones(int(selected_rows.size), dtype=np.int64)
                events[name] = {
                    port: EventSeries(
                        time=table.time[selected_rows],
                        source_id=table.source_index[selected_rows],
                        count=count,
                        metadata={"population": name, "port": port},
                    )
                }
                continue

            events[name] = {}
            for port, view in event_sources[name]:
                values = np.asarray(event_values[event_index])
                event_index += 1
                time_index, source_id = np.nonzero(values)
                events[name][port] = EventSeries(
                    time=times[time_index],
                    source_id=source_id,
                    count=np.asarray(values[time_index, source_id], dtype=np.int64),
                    metadata=_event_source_metadata(name, port, view),
                )
        return events

    def _run_scheduled_sources_only(self, *, dt, duration) -> NetworkResult:
        dt_ms = scalar_decimal(dt, u.ms)
        if self._scheduled_dt_ms is None:
            self._scheduled_dt_ms = dt_ms
        elif not np.isclose(self._scheduled_dt_ms, dt_ms):
            raise RuntimeError("Network runtime dt is fixed after the first run.")
        n_steps = _duration_steps(duration, dt)
        start_t = self._source_current_time
        stop_t = start_t + n_steps * dt
        times = start_t + u.math.arange(n_steps) * dt
        events = self._result_events(
            times=times,
            start_t=start_t,
            stop_t=stop_t,
            event_sources={},
            event_values=(),
        )
        self._source_current_time = stop_t
        return NetworkResult(
            time=times,
            traces={},
            samples={},
            events=events,
            start_time=start_t,
            stop_time=stop_t,
            dt=dt,
        )


def _reject_batch_size(batch_size) -> None:
    """Reject the batch argument both lifecycle entry points still accept."""
    if batch_size is not None:
        raise NotImplementedError(
            "Network batch execution is not implemented yet; use Cell batch execution for same-topology batches."
        )


def _event_source_metadata(population: str, port: str, view) -> dict:
    """Build immutable source-row metadata for one published event port."""
    metadata = {"population": population, "port": port}
    source_ids = view.source_id
    owner = view.owner
    for name in ("population_index", "location_index", "cv_id"):
        value = getattr(owner, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.broadcast_to(array, (owner.size,))
        if array.shape[0] != owner.size:
            continue
        metadata[name] = freeze_array(array[source_ids])
    return metadata


def _reset_delivery_state(state) -> None:
    """Clear persistent event queues at an explicit network reset boundary."""
    for index in range(len(state.ring_buffers)):
        state.ring_buffers[index].value = u.math.zeros_like(state.ring_buffers[index].value)
        state.ring_cursors[index].value = np.asarray(0, dtype=np.int32)
