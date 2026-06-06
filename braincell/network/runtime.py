"""Network runtime loop."""

from __future__ import annotations

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell._multi_compartment.run import _validate_time_quantity

from .connection import Connection
from .lowering import ConnectionBlock, lower_connections, resolve_synapse_layout
from .population import Population
from .projection import Projection
from .result import NetworkRunResult


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

    def add_population(self, name: str, cell) -> Population:
        """Add a named population and return its declaration."""
        population = Population(name=name, cell=cell)
        if population.name in self.populations:
            raise ValueError(f"Network already has a population named {population.name!r}.")
        self.populations[population.name] = population
        return population

    def add_connection(self, connection: Connection) -> Connection:
        """Add a sparse connection declaration and return it."""
        if not isinstance(connection, Connection):
            raise TypeError(
                f"Network.add_connection(...) expects Connection, got {type(connection).__name__!s}."
            )
        self.connections.append(connection)
        return connection

    def add_edge_set(self, edge_set):
        """Add a reusable cell-level edge set and return it."""
        if edge_set.name in self.edge_sets:
            raise ValueError(f"Network already has an edge set named {edge_set.name!r}.")
        self.edge_sets[edge_set.name] = edge_set
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
        synapse_pool=None,
        synapse=None,
        target_group=None,
        **kwargs,
    ) -> Projection:
        """Add a named synaptic projection and return it."""
        if projection is None:
            target = synapse_pool
            if target is None:
                target = synapse if synapse is not None else target_group
            if name is None:
                raise ValueError("Network.add_projection(...) requires name.")
            if edges is None:
                raise ValueError("Network.add_projection(...) requires edges.")
            if target is None:
                raise ValueError("Network.add_projection(...) requires synapse_pool.")
            projection = Projection(name=name, edges=edges, synapse_pool=target, **kwargs)
        elif any(value is not None for value in (name, edges, synapse_pool, synapse, target_group)) or kwargs:
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
        return projection

    def make_edges(self, *, name: str, pre: str, post: str, connector: str, **kwargs):
        """Create and store a reusable cell-level edge set.

        This compatibility wrapper accepts the earlier connector-string API.
        Prefer :meth:`add_edges` with ``braincell.network.all_to_all()``,
        ``braincell.network.probability(...)``, or ``braincell.network.pairs(...)``.
        """
        from . import edges as edge_builders

        extra = dict(kwargs)
        if connector == "pairs":
            method = edge_builders.pairs(extra.pop("pairs"))
        elif connector == "all_to_all":
            method = edge_builders.all_to_all(**extra)
            extra.clear()
        elif connector == "probability":
            method = edge_builders.probability(**extra)
            extra.clear()
        else:
            raise ValueError(f"Unknown edge connector {connector!r}.")
        if extra:
            raise TypeError(f"Unexpected connector arguments: {tuple(extra)!r}.")
        return self.add_edges(name=name, pre=pre, post=post, method=method)

    def project(self, *, name: str, edges, synapse=None, target_group=None, **kwargs):
        """Create and store a projection from an edge set to synapse targets."""
        target = kwargs.pop("synapse_pool", None)
        if target is None:
            target = synapse if synapse is not None else target_group
        return self.add_projection(name=name, edges=edges, synapse_pool=target, **kwargs)

    def run(self, *, dt, duration) -> NetworkRunResult:
        """Run the network for ``duration`` at fixed step ``dt``."""
        _validate_time_quantity(dt, name="dt")
        _validate_time_quantity(duration, name="duration")
        if not self.populations:
            raise ValueError("Network.run(...) requires at least one population.")

        for population in self.populations.values():
            if not getattr(population.cell, "_initialized", False):
                population.cell.init_state()

        blocks = lower_connections(
            self.populations,
            tuple(self.connections) + self._projection_connections(),
            dt=dt,
        )
        _raise_if_unsupported_scan_delay(blocks)
        pending_arrivals = tuple(
            brainstate.ShortTermState(
                _zero_arrival(block, post_size=self.populations[block.post_population].size)
            )
            for block in blocks
        )

        ordered_population_names = tuple(self.populations)
        start_t = self._common_start_time(ordered_population_names)
        relative_times = u.math.arange(0.0 * u.ms, duration, dt)
        if int(relative_times.shape[0]) == 0:
            raise ValueError(
                "Network.run(...) produced no timesteps; ensure duration > 0 and dt > 0."
            )
        times = start_t + relative_times
        probe_names = {
            name: tuple(sorted(population.cell.sample_probes()))
            for name, population in self.populations.items()
        }
        n_trace = sum(len(names) for names in probe_names.values())
        n_spike = len(ordered_population_names)

        with brainstate.environ.context(dt=dt):
            def _step(t):
                with brainstate.environ.context(t=t):
                    self._write_arrivals(blocks, pending_arrivals)
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
                    spikes = tuple(
                        self.populations[name].cell.spike.value
                        for name in ordered_population_names
                    )
                    self._enqueue_future_events(blocks, pending_arrivals)
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
        spikes = {
            name: spike_values[index]
            for index, name in enumerate(ordered_population_names)
        }
        return NetworkRunResult(time=times, traces=traces, spikes=spikes)

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

    def _write_arrivals(self, blocks, pending_arrivals) -> None:
        for index, block in enumerate(blocks):
            arrival = pending_arrivals[index].value
            cell = self.populations[block.post_population].cell
            cell.runtime.state_buffers[(int(block.layout_id), "pre_spike")] = arrival

    def _enqueue_future_events(self, blocks, pending_arrivals) -> None:
        for index, block in enumerate(blocks):
            pre_cell = self.populations[block.pre_population].cell
            pre_spike = _population_spike(pre_cell.spike.value)
            pre_values = pre_spike[block.pre_index]
            edge_event = pre_values * block.weight
            post_size = self.populations[block.post_population].size
            pending_arrivals[index].value = _scatter_post_events(
                edge_event,
                block.post_index,
                block.synapse_index,
                post_size=post_size,
                n_active=block.n_active,
            )

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
            pool_name = projection.scalar_synapse_pool
            if pool_name is not None:
                post = self.populations[edge_set.post_population]
                _, pool_size, _ = resolve_synapse_layout(post, pool_name)
            connections.extend(projection.to_connections(edge_set, pool_size=pool_size))
        return tuple(connections)


def _population_spike(spike) -> object:
    if spike.ndim == 2 and spike.shape[-1] == 1:
        return spike[..., 0]
    if spike.ndim > 1:
        return jnp.any(spike, axis=tuple(range(1, spike.ndim)))
    return spike


def _scatter_post_events(edge_event, post_index: np.ndarray, synapse_index: np.ndarray, *, post_size: int, n_active: int):
    out = _zeros_like_events(edge_event, post_size=post_size, n_active=n_active)
    return out.at[jnp.asarray(post_index), jnp.asarray(synapse_index)].add(edge_event)


def _zero_arrival(block: ConnectionBlock, *, post_size: int):
    return _zeros_like_events(block.weight, post_size=post_size, n_active=block.n_active)


def _zeros_like_events(value, *, post_size: int, n_active: int):
    if isinstance(value, u.Quantity):
        return u.math.zeros_like(value, shape=(post_size, n_active))
    return jnp.zeros((post_size, n_active), dtype=jnp.asarray(value).dtype)


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


def _raise_if_unsupported_scan_delay(blocks: tuple[ConnectionBlock, ...]) -> None:
    for block in blocks:
        if np.any(np.asarray(block.delay_steps) != 1):
            raise NotImplementedError(
                "Network.run(...) scan runtime currently supports only zero-delay "
                "next-step delivery. Multi-step delays need a scan-compatible ring buffer."
            )
