"""Network runtime loop."""

from __future__ import annotations

from dataclasses import dataclass

import brainstate
import brainunit as u
import numpy as np

from braincell._multi_compartment.run import _validate_time_quantity

from .core import Connection, NetworkRunResult, Population
from .delivery import (
    DeliveryBlock,
    advance_delivery_state,
    create_delivery_state,
    delivery_blocks as build_delivery_blocks,
    enqueue_future_events,
    make_delivery_op,
    normalize_event_backend,
    population_spike,
    resolve_event_backend,
    write_arrivals,
)
from .lowering import lower_connections, resolve_synapse_layout
from .projections import Projection


@dataclass(frozen=True)
class _RunSetup:
    """Reusable topology/backend data for one ``Network.run`` configuration."""

    delivery_blocks: tuple[DeliveryBlock, ...]
    delivery_backend: str
    delivery_ops: tuple
    ordered_population_names: tuple[str, ...]
    probe_names: dict[str, tuple[str, ...]]
    n_trace: int


class Network:
    """Named population network using existing ``Cell`` runtimes."""

    def __init__(self, name: str | None = None) -> None:
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Network name must be a non-empty string or None.")
        self.name = name
        self.populations: dict[str, Population] = {}
        self.connections: list[Connection] = []
        self.edge_sets = {}
        self.proj: dict[str, Projection] = {}
        self.projections = self.proj
        self._topology_version = 0
        self._run_setup_cache: dict[tuple, _RunSetup] = {}

    def _mark_topology_changed(self) -> None:
        self._topology_version += 1
        self._run_setup_cache.clear()

    def __repr__(self) -> str:
        """Return a compact network summary."""
        return (
            f"Network(name={self.name!r}, populations={len(self.populations)}, "
            f"edge_sets={len(self.edge_sets)}, projections={len(self.projections)}, "
            f"connections={len(self.connections)})"
        )

    def __str__(self) -> str:
        """Return a readable multi-line network summary."""
        lines = [repr(self)]
        lines.append("  populations:")
        if self.populations:
            for name, population in self.populations.items():
                initialized = bool(getattr(population.cell, "_initialized", False))
                lines.append(
                    f"    {name}: size={population.size}, "
                    f"cell={type(population.cell).__name__}, initialized={initialized}"
                )
        else:
            lines.append("    <none>")

        lines.append("  edge_sets:")
        if self.edge_sets:
            for name, edge_set in self.edge_sets.items():
                lines.append(
                    f"    {name}: {edge_set.pre_population} -> "
                    f"{edge_set.post_population}, n_edge={edge_set.n_edge}"
                )
        else:
            lines.append("    <none>")

        lines.append("  projections:")
        if self.projections:
            for name, projection in self.projections.items():
                lines.append(
                    f"    {name}: edges={projection.edge_set_name!r}, "
                    f"synapse={projection.synapse!r}"
                )
        else:
            lines.append("    <none>")

        lines.append("  direct connections:")
        if self.connections:
            for index, connection in enumerate(self.connections):
                lines.append(
                    f"    [{index}]: {connection.pre_population} -> "
                    f"{connection.post_population}, synapse={connection.synapse!r}, "
                    f"n_contact={connection.n_contact}"
                )
        else:
            lines.append("    <none>")
        return "\n".join(lines)

    def add_population(self, name: str, cell) -> Population:
        """Add a named population and return its declaration."""
        population = Population(name=name, cell=cell)
        if population.name in self.populations:
            raise ValueError(f"Network already has a population named {population.name!r}.")
        self.populations[population.name] = population
        self._mark_topology_changed()
        return population

    def add_connection(self, connection: Connection) -> Connection:
        """Add a sparse connection declaration and return it."""
        if not isinstance(connection, Connection):
            raise TypeError(
                f"Network.add_connection(...) expects Connection, got {type(connection).__name__!s}."
            )
        self.connections.append(connection)
        self._mark_topology_changed()
        return connection

    def add_edge_set(self, edge_set):
        """Add a reusable cell-level edge set and return it."""
        if edge_set.name in self.edge_sets:
            raise ValueError(f"Network already has an edge set named {edge_set.name!r}.")
        self.edge_sets[edge_set.name] = edge_set
        self._mark_topology_changed()
        return edge_set

    def add_edges(self, *, name: str, pre: str, post: str, method):
        """Create and store a reusable cell-level edge set."""
        from . import edges as edge_builders

        if pre not in self.populations:
            raise KeyError(f"Unknown pre population {pre!r}.")
        if post not in self.populations:
            raise KeyError(f"Unknown post population {post!r}.")
        edge_set = edge_builders.build(
            name=name,
            pre=pre,
            post=post,
            method=method,
            n_pre=self.populations[pre].size,
            n_post=self.populations[post].size,
        )
        return self.add_edge_set(edge_set)

    def add_projection(
        self,
        projection: Projection | None = None,
        *,
        name: str | None = None,
        edges=None,
        synapse=None,
        **kwargs,
    ) -> Projection:
        """Add a named synaptic projection and return it."""
        if projection is None:
            if name is None:
                raise ValueError("Network.add_projection(...) requires name.")
            if edges is None:
                raise ValueError("Network.add_projection(...) requires edges.")
            if synapse is None:
                raise ValueError("Network.add_projection(...) requires synapse.")
            projection = Projection(name=name, edges=edges, synapse=synapse, **kwargs)
        elif any(value is not None for value in (name, edges, synapse)) or kwargs:
            raise TypeError(
                "Network.add_projection(...) accepts either a Projection object "
                "or direct projection keyword arguments, not both."
            )
        if not isinstance(projection, Projection):
            raise TypeError(
                f"Network.add_projection(...) expects Projection, got {type(projection).__name__!s}."
            )
        if projection.name in self.proj:
            raise ValueError(f"Network already has a projection named {projection.name!r}.")
        self.proj[projection.name] = projection
        self._mark_topology_changed()
        return projection

    def make_edges(self, *, name: str, pre: str, post: str, connector: str, **kwargs):
        """Create and store a reusable cell-level edge set.

        Prefer :meth:`add_edges` with ``braincell.network.all_pairs(...)``,
        ``braincell.network.probability(...)``, ``braincell.network.dense(...)``,
        ``braincell.network.pairs(...)``, or a custom callable method.
        """
        from . import edges as edge_builders

        extra = dict(kwargs)
        if connector == "pairs":
            method = edge_builders.pairs(extra.pop("pairs"))
        elif connector == "all_pairs":
            method = edge_builders.all_pairs(**extra)
            extra.clear()
        elif connector == "dense":
            method = edge_builders.dense(extra.pop("adjacency"))
            extra.clear()
        elif connector == "probability":
            method = edge_builders.probability(**extra)
            extra.clear()
        else:
            raise ValueError(f"Unknown edge connector {connector!r}.")
        if extra:
            raise TypeError(f"Unexpected connector arguments: {tuple(extra)!r}.")
        return self.add_edges(name=name, pre=pre, post=post, method=method)

    def project(self, *, name: str, edges, synapse, **kwargs):
        """Create and store a projection from an edge set to synapse targets."""
        return self.add_projection(name=name, edges=edges, synapse=synapse, **kwargs)

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
        for population in self.populations.values():
            if not getattr(population.cell, "_initialized", False):
                population.cell.init_state(batch_size=batch_size)
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
        for population in self.populations.values():
            if not getattr(population.cell, "_initialized", False):
                population.cell.init_state(batch_size=batch_size)
            population.cell.reset_state(batch_size=batch_size)
        return self

    def run(
        self,
        *,
        dt,
        duration,
        delay_quantization: str = "ceil",
        event_backend: str = "auto",
        brainevent_backend: str | None = "jax_raw",
        spike_recording: str = "full",
    ) -> NetworkRunResult:
        """Run the network for ``duration`` at fixed step ``dt``."""
        _validate_time_quantity(dt, name="dt")
        _validate_time_quantity(duration, name="duration")
        event_backend = normalize_event_backend(event_backend)
        spike_recording = _normalize_spike_recording(spike_recording)
        if not self.populations:
            raise ValueError("Network.run(...) requires at least one population.")

        self.init_state()

        setup = self._run_setup(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        delivery_blocks = setup.delivery_blocks
        delivery_state = create_delivery_state(
            delivery_blocks,
            populations=self.populations,
            delivery_ops=setup.delivery_ops,
        )

        ordered_population_names = setup.ordered_population_names
        start_t = self._common_start_time(ordered_population_names)
        relative_times = u.math.arange(0.0 * u.ms, duration, dt)
        if int(relative_times.shape[0]) == 0:
            raise ValueError(
                "Network.run(...) produced no timesteps; ensure duration > 0 and dt > 0."
            )
        times = start_t + relative_times
        probe_names = setup.probe_names
        n_trace = setup.n_trace
        n_spike = 0 if spike_recording == "none" else len(ordered_population_names)

        with brainstate.environ.context(dt=dt):
            def _step(t):
                with brainstate.environ.context(t=t):
                    write_arrivals(
                        delivery_blocks,
                        delivery_state,
                        populations=self.populations,
                    )
                    for name in ordered_population_names:
                        self.populations[name].cell._prepare_next_synapse_inputs()
                    for name in ordered_population_names:
                        self.populations[name].cell._begin_step()
                    for name in ordered_population_names:
                        cell = self.populations[name].cell
                        cell._update_dynamics()
                    snapshots = {
                        name: self.populations[name].cell.sample_probes()
                        for name in ordered_population_names
                    }
                    if spike_recording == "full":
                        spikes = tuple(
                            self.populations[name].cell.spike.value
                            for name in ordered_population_names
                        )
                    elif spike_recording == "population":
                        spikes = tuple(
                            population_spike(self.populations[name].cell.spike.value)
                            for name in ordered_population_names
                        )
                    else:
                        spikes = ()
                    enqueue_future_events(
                        delivery_blocks,
                        delivery_state,
                        populations=self.populations,
                    )
                    advance_delivery_state(delivery_state)
                    traces = tuple(
                        snapshots[name][probe_name]
                        for name in ordered_population_names
                        for probe_name in probe_names[name]
                    )
                    return traces + spikes

            samples_over_time = brainstate.transform.for_loop(_step, times)

        end_t = start_t + int(times.shape[0]) * dt
        for population in self.populations.values():
            population.cell._set_current_time(end_t)

        samples_tuple = _normalize_scan_samples(
            samples_over_time,
            n_samples=n_trace + n_spike,
        )
        trace_values = samples_tuple[:n_trace]
        spike_values = samples_tuple[n_trace:]

        traces = {}
        index = 0
        for name in ordered_population_names:
            traces[name] = {}
            for probe_name in probe_names[name]:
                traces[name][probe_name] = trace_values[index]
                index += 1
        spikes = {}
        if spike_recording != "none":
            spikes = {
                name: spike_values[index]
                for index, name in enumerate(ordered_population_names)
            }
        return NetworkRunResult(time=times, traces=traces, spikes=spikes)

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

        blocks = lower_connections(
            self.populations,
            tuple(self.connections) + self._projection_connections(),
            dt=dt,
            delay_quantization=delay_quantization,
        )
        delivery_blocks = build_delivery_blocks(blocks)
        delivery_backend = resolve_event_backend(event_backend)
        delivery_ops = tuple(
            make_delivery_op(
                block,
                pre_size=self.populations[block.source.pre_population].size,
                post_size=self.populations[block.source.post_population].size,
                backend=delivery_backend,
                brainevent_backend=brainevent_backend,
            )
            for block in delivery_blocks
        )
        ordered_population_names = tuple(self.populations)
        probe_names = {
            name: tuple(sorted(population.cell.sample_probes()))
            for name, population in self.populations.items()
        }
        setup = _RunSetup(
            delivery_blocks=delivery_blocks,
            delivery_backend=delivery_backend,
            delivery_ops=delivery_ops,
            ordered_population_names=ordered_population_names,
            probe_names=probe_names,
            n_trace=sum(len(names) for names in probe_names.values()),
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
        dt_ms = float(np.asarray(dt.to_decimal(u.ms), dtype=float).reshape(()))
        runtime_ids = tuple(
            (name, id(population.cell.runtime), population.size)
            for name, population in self.populations.items()
        )
        return (
            self._topology_version,
            runtime_ids,
            dt_ms,
            delay_quantization,
            event_backend,
            brainevent_backend,
        )

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

    def _projection_connections(self) -> tuple[Connection, ...]:
        connections = []
        for projection in self.proj.values():
            edge_name = projection.edge_set_name
            if edge_name not in self.edge_sets:
                raise KeyError(
                    f"Projection {projection.name!r} references unknown EdgeSet {edge_name!r}."
                )
            edge_set = self.edge_sets[edge_name]
            pool_size = None
            pool_name = projection.synapse
            if pool_name is not None:
                post = self.populations[edge_set.post_population]
                _, pool_size, _ = resolve_synapse_layout(post, pool_name)
            connections.extend(
                projection.to_connections(
                    edge_set,
                    pre_size=self.populations[edge_set.pre_population].size,
                    post_size=self.populations[edge_set.post_population].size,
                    pool_size=pool_size,
                )
            )
        return tuple(connections)


def _normalize_spike_recording(value: str) -> str:
    if value not in ("full", "population", "none"):
        raise ValueError(
            "Network spike_recording must be 'full', 'population', or 'none', "
            f"got {value!r}."
        )
    return value


def _normalize_scan_samples(values, *, n_samples: int) -> tuple:
    if n_samples == 0:
        return ()
    if n_samples == 1:
        return values if isinstance(values, tuple) else (values,)
    if not isinstance(values, tuple):
        raise TypeError(
            f"Network.run(...) expected {n_samples} scan outputs, "
            f"got {type(values).__name__!s}."
        )
    if len(values) != n_samples:
        raise ValueError(
            f"Network.run(...) expected {n_samples} scan outputs, got {len(values)!r}."
        )
    return values
