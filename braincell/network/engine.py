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
import warnings
from dataclasses import dataclass

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._misc import (
    freeze_array,
    is_traced_value,
    normalize_loop_outputs,
    scalar_decimal,
    validate_time_quantity,
)
from braincell._multi_compartment.run import (
    _duration_steps,
    _recording_time_mask,
    check_step_limit,
    host_clock_start,
)
from braincell._multi_compartment.synapses import SynapseView
from braincell.network.connection import NetworkConnections, PairingSpec, _UNSET, _connect_with_pairing_seed
from braincell.network.recording import EventSeries, SampleBlock

from braincell.network.core import NetworkResult, Population
from braincell.network.delivery import (
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
from braincell.network.lowering import lower_direct_connections
from braincell.network.scheduled import (
    ScheduledPlans,
    loop_schedule_key,
    normalize_scheduled_delivery,
    scheduled_inputs,
)
from braincell.mech import Synapse


@dataclass(frozen=True)
class _RunSetup:
    """Reusable topology/backend data for one ``Network.run`` configuration."""

    delivery_blocks: tuple[DeliveryBlock, ...]
    delivery_ops: tuple
    ordered_population_names: tuple[str, ...]
    compiled_recordings: dict[str, tuple]
    event_sources: dict[str, tuple]
    n_recording: int
    n_event: int


@dataclass(frozen=True)
class _CachedRunLoop:
    """Compiled network scan plus its reusable delivery buffers."""

    runner: object
    delivery_state: object


class Network:
    """Named population network using existing ``Cell`` runtimes.

    Parameters
    ----------
    name : str, optional
        Non-empty network name.
    seed : int or None, optional
        Root of the network's derived random streams. A registered
        ``NetStim`` without its own ``seed`` and a ``connect`` pairing rule
        without its own ``seed`` derive their streams from this root and
        their population and connection names, so registration order does
        not matter.

        ``None``, the default, draws the root once, here, from the
        BrainState default random stream: an enclosing
        ``brainstate.random.seed_context(s)`` (or an earlier
        ``brainstate.random.seed(s)``) then controls these streams together
        with initializers and callbacks, and each such ``Network`` advances
        the default stream by one draw. Outside any context the default
        stream is seeded randomly when BrainState is imported, so each
        process draws another root. The drawn root is stored in ``seed``;
        ``Network(seed=net.seed)`` reproduces the derived streams. An
        integer is a local root that no context changes. Inside
        ``brainstate.transform.jit``, ``for_loop`` or another
        transformation the default stream is traced and ``None`` raises
        ``TypeError``; build the network outside it or pass an integer.

    Attributes
    ----------
    name : str or None
        The network name.
    seed : int
        The root, as given or as drawn for ``seed=None``.

    Examples
    --------
    >>> import brainstate
    >>> import braincell
    >>> with brainstate.random.seed_context(42):
    ...     first = braincell.Network("demo")
    >>> with brainstate.random.seed_context(42):
    ...     again = braincell.Network("demo")
    >>> first.seed == again.seed
    True
    >>> braincell.Network("fixed", seed=7).seed
    7
    """

    def __init__(self, name: str | None = None, *, seed: int | None = None) -> None:
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Network name must be a non-empty string or None.")
        if seed is None:
            from braincell._random import draw_seed

            seed = draw_seed("Network(seed=None)")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("Network seed must be an integer or None.")
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
        # Steps run by a Network without Cell populations (its Cells keep their own clocks).
        self._source_step = 0
        from braincell.trainable._network import NetworkTrainables

        self.trainables = NetworkTrainables(self)
        self._prepared_run = None
        # Planned scheduled delivery (CELL-9): the plans, and those prepare_run made for update().
        self._scheduled_plans: ScheduledPlans | None = None
        self._prepared_plans = None

    def _raise_if_initialized(self, action: str) -> None:
        if self._initialized:
            raise RuntimeError(f"Cannot {action} after Network initialization.")

    @contextlib.contextmanager
    def _cell_lifecycle(self):
        """Mark the network as driving its cells' init/reset transitions.

        ``Cell`` consults ``_cell_lifecycle_active`` to tell a network-driven
        transition from a user calling ``cell.init_state()`` directly
        (``Cell._raise_if_network_owned``, ``Cell._raise_if_initialized``). Both
        :meth:`init_state` and :meth:`reset_state` need the flag cleared even
        when a cell raises, so the ``try``/``finally`` lives here once.
        """
        self._cell_lifecycle_active = True
        try:
            yield
        finally:
            self._cell_lifecycle_active = False

    def _drop_compiled_loops(self) -> None:
        """Drop the compiled run and update loops, keeping the delivery buffers.

        A member Cell calls this when it registers a callable current or
        delta input (brainpy projections): the loops traced its input tables
        once, so they would miss the new input. Value inputs are arguments
        of the loops (:meth:`_take_value_inputs`).
        """
        self._network_run_loop_cache.clear()
        if self._prepared_run is not None:
            setup, cached, dt = self._prepared_run
            self._prepared_run = (setup, _CachedRunLoop(runner=None, delivery_state=cached.delivery_state), dt)

    def _mark_topology_changed(self) -> None:
        self._topology_version += 1
        self._run_setup_cache.clear()
        self._network_run_loop_cache.clear()
        self._delivery_state_cache.clear()
        self._scheduled_plans = None

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
        summaries = self.connections._summaries()
        for target, rows, calls in summaries:
            lines.append(f"    {target}: connections={len(calls)}, rows={rows}")
            lines.extend(f"      {line}" for line in calls)
        if not summaries:
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
        from braincell.network.event import EventSource

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
        plasticity=None,
    ):
        """Connect a registered source to existing or newly placed synapses.

        Parameters
        ----------
        name : str
            Connection name, unique within the target Cell.
        source : Population, EventSource, or EventSourceView
            Registered event source or a selected source view.
        synapse : SynapseView or braincell.mech.Synapse
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
        plasticity : braincell.mech.ConnectionRule, optional
            A plasticity rule for every row of this call, as in
            :func:`braincell.connect`. A rule on a live source runs only in
            a Network.

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
                plasticity=plasticity,
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
                plasticity=plasticity,
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

        It initializes all of its cells or none. If a cell's ``init_state``
        raises, the cells this call initialized before it go back to
        declaring, as they were before the call, and the error propagates
        unchanged; the Network stays uninitialized. Correct the declaration
        and call ``init_state()`` again. Cells initialized before the call
        are left initialized.
        """
        _reject_batch_size(batch_size)
        if self._initialized:
            return self
        self._validate_direct_source_ownership()
        uninitialized = [
            population.cell
            for population in self._cell_populations().values()
            if not getattr(population.cell, "_initialized", False)
        ]
        # Recordings and connections declared on an older grid follow it, or
        # raise, before any Cell initializes (MORPH-15).
        for cell in uninitialized:
            cell._resolve_declarations()
        initialized = []
        with self._cell_lifecycle():
            try:
                for cell in uninitialized:
                    preview = cell._declaration_preview()
                    cell.init_state()
                    initialized.append((cell, preview))
            except BaseException:
                # All or none (CELL-37): a Cell left initialized here could
                # neither reset() nor take a correction, and the Network
                # would not run it.
                for cell, preview in reversed(initialized):
                    cell._roll_back_init_state(preview)
                raise
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
        ``Network.reset_state`` resets runtime state in place, and the step
        count to 0 (``t = 0``). It does not call ``Cell.reset()``, which
        would tear down runtime objects and return the cell to the
        declaration phase.
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
        if self._prepared_run is not None:
            # update() counts steps on the clocks prepare_run() set, without
            # resetting them itself (a compiled loop would rewrite a constant
            # on every step): set them again, at t = 0.
            dt = self._prepared_run[2]
            for population in self._cell_populations().values():
                population.cell._set_clock(0.0 * u.ms, 0, dt)
        for state in self._delivery_state_cache.values():
            _reset_delivery_state(state)
        self._source_step = 0
        return self

    def run(
        self,
        *,
        dt,
        duration,
        delay_quantization: str = "nearest",
        event_backend: str = "auto",
        brainevent_backend: str | None = "jax_raw",
        scheduled_delivery: str | None = None,
    ) -> NetworkResult:
        """Run the network for ``duration`` at fixed step ``dt``.

        Parameters
        ----------
        dt : brainunit.Quantity
            Fixed time step.
        duration : brainunit.Quantity
            Length of this segment, an integer multiple of ``dt``.
        delay_quantization : str, optional
            How connection delays round to steps: ``"nearest"`` (default),
            ``"ceil"``, ``"floor"`` or ``"strict"``.
        event_backend : str, optional
            Delivery of live events: ``"auto"`` (default, which is
            ``"scatter"``), ``"scatter"`` or ``"brainevent"``.
        brainevent_backend : str or None, optional
            Backend of ``brainevent.csrmv`` under ``"brainevent"``.
        scheduled_delivery : {None, "direct", "padded"}, optional
            How the rows of scheduled sources (``NetStim``,
            ``EventSequence``) reach their synapses. ``None``, the default,
            counts each row's arrivals on every step. ``"direct"`` and
            ``"padded"`` plan the arrivals of this run's steps on the host
            before the loop; see Notes. Rows with a plasticity rule keep
            the rule's path, and live sources the delay queues.

        Returns
        -------
        NetworkResult
            Samples and events of the half-open segment
            ``[start_time, stop_time)``.

        Raises
        ------
        ValueError
            If the run would take the step count past ``2**31 - 1`` (14.9 h
            at ``dt = 0.025 ms``); nothing has run. :meth:`reset_state`
            starts the count again. Also if ``scheduled_delivery`` is not
            one of its values, before anything is initialized.
        PaddingLimitError
            A ``ValueError``, if a ``"padded"`` plan would exceed its limits
            (Notes); nothing has run.
        RuntimeError
            If ``dt``, ``delay_quantization``, the backends or
            ``scheduled_delivery`` differ from the first run's (or
            :meth:`prepare_run`'s).

        Warns
        -----
        RuntimeWarning
            Once per edit, made after initialization, of the morphology a
            population Cell was declared with: each Cell runs on the
            snapshot taken at its ``init_state``.

        Notes
        -----
        The network counts its steps: step ``k`` of a run starts at
        ``t = (n + k)·dt``, where ``n`` is the number of steps run since
        initialization or :meth:`reset_state` by ``run`` and
        :meth:`update`. Each time is one rounding of a product, in the run
        precision, so a run split into segments, or into ``update()``
        calls, steps at the same times as one run, however long it is.

        **Scheduled delivery** (``scheduled_delivery``, CELL-9). The
        arrivals of the run's steps ``[n, n + steps)`` are planned on the
        host before the compiled loop: for each synapse layout, one record
        per step and row that has arrivals, holding their number. The
        plan's arrays are arguments of the loop, so a later window of the
        same length reuses the compiled loop. The plans of the last four
        windows are kept: a run after :meth:`reset_state`, or after a
        weight change (weights, trainable ones included, are read on every
        step), builds no plan. Each event arrives at the step, and with
        the weight, of the default path, so the results are the default
        path's (to rounding where several rows reach one synapse).

        - ``"direct"`` stores the records in step order with one offset per
          step and reads the current step's: storage grows with the
          records, the steps and the rows. It is reverse-differentiable to
          first order only: forward mode (``jax.jvp``, ``jacfwd``) through it
          raises ``TypeError`` when traced.
        - ``"padded"`` stores ``steps x width`` slots, ``width`` being the
          most records any step has, rounded up to a power of two so that
          windows of one length share a shape. It raises
          :class:`~braincell.network.scheduled.PaddingLimitError` before the
          run when the slots would take more than 256 MiB, or the steps
          times the needed width would exceed 8 times the records; it does
          not fall back to another method. It differentiates as ordinary
          JAX code, forward mode included.

        Try ``"direct"`` when many scheduled rows have few arrivals per
        step, above all when rows share synapses, or when a source's
        schedule is much longer than the run; ``"padded"`` within its
        limits. Keep the default for small models: preparing a plan costs
        more than it saves. The method is part of the runtime
        configuration, fixed after the first run as ``event_backend`` is;
        a first run that raised ``PaddingLimitError`` fixes nothing.

        Examples
        --------
        .. code-block:: python

            >>> import brainunit as u
            >>> import braincell
            >>> from braincell.filter import at
            >>> cell = braincell.Cell(
            ...     braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um), pop_size=1
            ... )
            >>> _ = cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
            >>> net = braincell.Network("demo", seed=1)
            >>> post = net.add_population("post", cell)
            >>> stim = net.add_population(
            ...     "stim", braincell.NetStim(size=1, start=1.0 * u.ms, interval=2.0 * u.ms, number=5)
            ... )
            >>> _ = net.connect("drive", source=stim, synapse=post.synapses["s"], weight=0.001 * u.uS)
            >>> first = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms, scheduled_delivery="direct")
            >>> second = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms, scheduled_delivery="direct")
            >>> second.events["stim"]["spike"].time  # the second window's events, planned before it ran
            Quantity([5. 7. 9.], "ms")
        """
        from braincell.network.event import _schedule_horizon

        validate_time_quantity(dt, name="dt", prefix="Network.run(...)")
        validate_time_quantity(duration, name="duration", prefix="Network.run(...)")
        event_backend = normalize_event_backend(event_backend)
        scheduled_delivery = normalize_scheduled_delivery(scheduled_delivery, prefix="Network.run(...)")
        if not self.populations:
            raise ValueError("Network.run(...) requires at least one population.")
        self.init_state()
        if not self._cell_populations():
            return self._run_scheduled_sources_only(dt=dt, duration=duration)
        self._warn_on_morphology_edits()
        self.trainables.materialize()
        setup_key = self._run_setup_cache_key(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        runtime_config = setup_key[2:] + (scheduled_delivery,)
        first_config = self._runtime_config is None
        if first_config:
            self._runtime_config = runtime_config
        elif self._runtime_config != runtime_config:
            raise RuntimeError(
                "Network runtime configuration is fixed after the first run; "
                "dt, delay quantization, event backend, and scheduled delivery must remain unchanged."
            )
        setup = self._run_setup(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        ordered_population_names = setup.ordered_population_names
        n_steps = _duration_steps(duration, dt)
        start_ms = self._start_clocks(ordered_population_names, dt=dt, n_steps=n_steps, prefix="Network.run(...)")
        n_recording = setup.n_recording
        n_event = setup.n_event
        # NetStim events up to one step past the end cover every arrival the
        # loop delivers; later events are not generated.
        until_ms = start_ms + (n_steps + 1) * scalar_decimal(dt, u.ms)
        schedule_key = self._prepare_netstim_schedules(until_ms=until_ms)
        plans = None
        if scheduled_delivery is not None:
            # The plans of the window [step, step + n_steps), before the loop (CELL-9).
            start_step = self._grid_start_step(ordered_population_names, dt)
            plans, schedule_key = self._scheduled_arrays(
                scheduled_delivery, dt=dt, window=(start_step, n_steps), first_config=first_config
            )
        cached_loop = self._network_run_loop(
            setup=setup,
            setup_key=setup_key,
            dt=dt,
            n_steps=n_steps,
            schedule_key=schedule_key,
            scheduled_delivery=scheduled_delivery,
        )
        values = self._take_value_inputs(ordered_population_names)
        try:
            with _schedule_horizon(until_ms):
                if plans is None:
                    times, start_t, stop_t, samples_over_time = cached_loop.runner(values)
                else:
                    times, start_t, stop_t, samples_over_time = cached_loop.runner(values, plans)
        except BaseException:
            self._restore_value_inputs(values)
            raise

        samples_tuple = normalize_loop_outputs(
            samples_over_time,
            count=n_recording + n_event,
            prefix="Network.run(...)",
            noun="scan outputs",
        )
        recording_values = samples_tuple[:n_recording]
        event_values = samples_tuple[n_recording:]

        samples = {}
        recording_index = 0
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
        traces = {
            name: {recording_name: block.values for recording_name, block in samples[name].items()}
            for name in ordered_population_names
        }
        events = self._result_events(
            times=times,
            start_t=start_t,
            stop_t=stop_t,
            event_sources=setup.event_sources,
            event_values=event_values,
        )
        return NetworkResult(
            time=times,
            traces=traces,
            samples=samples,
            events=events,
            start_time=start_t,
            stop_time=stop_t,
            dt=dt,
        )

    def _warn_on_morphology_edits(self) -> None:
        """Warn once per edit of a population Cell's declaration tree after init (MORPH-14)."""
        edited = [
            name
            for name, population in self._cell_populations().items()
            if population.cell._unannounced_declaration_edit()
        ]
        if edited:
            names = ", ".join(repr(name) for name in edited)
            warnings.warn(
                f"The Cells of populations {names} run on the snapshot of their morphology taken at "
                "init_state(); the morphology they were declared with has been edited since, and the edit "
                "does not apply to an initialized Network. Declare new Cells and a new Network to run the "
                "edited morphology.",
                RuntimeWarning,
                stacklevel=3,
            )

    def prepare_run(
        self,
        *,
        dt,
        delay_quantization="nearest",
        event_backend="auto",
        brainevent_backend="jax_raw",
        scheduled_delivery=None,
    ):
        """Prepare static routing and queues outside a differentiated rollout.

        Parameters
        ----------
        dt : brainunit.Quantity
            Fixed integration step.
        delay_quantization : str, optional
            Existing delay quantization policy; delays are never trainable.
        event_backend : str, optional
            Event delivery backend, as in :meth:`run`.
        brainevent_backend : str or None, optional
            Backend for the sparse event operator.
        scheduled_delivery : {None, "direct", "padded"}, optional
            As in :meth:`run`. The plan :meth:`update` reads covers every
            event, steps ``[0, last arrival + 1)`` of each synapse layout,
            and is built once, here: a training loop of ``update()`` calls,
            split over several ``for_loop`` calls or rerun after
            :meth:`reset_state`, uses it throughout. Steps past the last
            arrival receive nothing, as on the default path.

        Returns
        -------
        Network
            This initialized and prepared network.

        Raises
        ------
        ValueError
            If ``scheduled_delivery`` is not one of its values; or
            ``PaddingLimitError`` if a ``"padded"`` plan exceeds its limits.
        RuntimeError
            If the configuration differs from an earlier run's.

        Notes
        -----
        ``"direct"`` supports first-order reverse differentiation of the
        ``update()`` rollout (``jax.grad``, ``brainstate.transform.grad``)
        only; forward mode through it (``jax.jvp``, ``jacfwd``) raises
        ``TypeError`` when traced. ``"padded"`` differentiates as ordinary
        JAX code. Either way the gradients with respect to weights and
        synapse parameters are the default path's.

        Examples
        --------
        .. code-block:: python

            >>> import brainstate
            >>> import brainunit as u
            >>> import jax.numpy as jnp
            >>> import braincell
            >>> from braincell.filter import at
            >>> cell = braincell.Cell(
            ...     braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um), pop_size=1
            ... )
            >>> _ = cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
            >>> net = braincell.Network("train", seed=1)
            >>> post = net.add_population("post", cell)
            >>> stim = net.add_population(
            ...     "stim", braincell.NetStim(size=1, start=1.0 * u.ms, interval=2.0 * u.ms, number=5)
            ... )
            >>> _ = net.connect("drive", source=stim, synapse=post.synapses["s"], weight=0.001 * u.uS)
            >>> _ = net.prepare_run(dt=0.025 * u.ms, scheduled_delivery="direct")
            >>> spikes = brainstate.transform.for_loop(lambda _: net.update()["post"], jnp.arange(400))
            >>> spikes.shape
            (400, 1, 1)
        """
        validate_time_quantity(dt, name="dt", prefix="Network.prepare_run")
        event_backend = normalize_event_backend(event_backend)
        scheduled_delivery = normalize_scheduled_delivery(scheduled_delivery, prefix="Network.prepare_run(...)")
        self.init_state()
        if not self._cell_populations():
            raise ValueError("Network.prepare_run requires at least one Cell population.")
        key = self._run_setup_cache_key(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        runtime_config = key[2:] + (scheduled_delivery,)
        if self._runtime_config is not None and self._runtime_config != runtime_config:
            raise RuntimeError("Network runtime configuration is fixed after the first run.")
        first_config = self._runtime_config is None
        self._runtime_config = runtime_config
        setup = self._run_setup(
            dt=dt,
            delay_quantization=delay_quantization,
            event_backend=event_backend,
            brainevent_backend=brainevent_backend,
        )
        # update() runs for an unknown number of steps: each eager call checks
        # the step limit, and every event is generated.
        self._start_clocks(setup.ordered_population_names, dt=dt, n_steps=0, prefix="Network.prepare_run(...)")
        schedule_key = self._prepare_netstim_schedules(until_ms=None)
        plans = None
        if scheduled_delivery is not None:
            # One plan of every event, for every update() (CELL-9).
            plans, schedule_key = self._scheduled_arrays(
                scheduled_delivery, dt=dt, window=None, first_config=first_config
            )
        cached = self._network_run_loop(
            setup=setup,
            setup_key=key,
            dt=dt,
            n_steps=1,
            schedule_key=schedule_key,
            scheduled_delivery=scheduled_delivery,
        )
        self.trainables.materialize()
        self._prepared_run = (setup, cached, dt)
        self._prepared_plans = plans
        return self

    def update(self):
        """Advance a prepared network by one differentiable time step.

        Returns
        -------
        dict
            Floating spike arrays keyed by Cell population name. Continuous
            observations remain available on each Cell's runtime states.

        Raises
        ------
        RuntimeError
            If :meth:`prepare_run` has not been called outside tracing.
        ValueError
            If called eagerly with the step count at its limit,
            ``2**31 - 1``. Inside a compiled loop the count is not
            checked, as that would cost a host round trip per step: keep
            such a loop within the limit (14.9 h at ``dt = 0.025 ms``).

        Notes
        -----
        Compose with ``brainstate.transform.for_loop`` or ``scan``. No event
        tables or host-side recording conversions are built here.

        The step starts at ``t = n·dt`` and advances the step count ``n``
        by one, as in :meth:`run`: a loop of ``update()`` calls steps at the
        times of one ``run``.
        """
        if self._prepared_run is None:
            raise RuntimeError("Call Network.prepare_run() before update().")
        setup, cached, dt = self._prepared_run
        if cached.runner is None:
            # A Cell registered an input since the loop was traced.
            cached = _CachedRunLoop(
                runner=self._make_network_run_loop(
                    setup=setup,
                    dt=dt,
                    n_steps=1,
                    delivery_state=cached.delivery_state,
                    scheduled_delivery=self._runtime_config[-1],
                ),
                delivery_state=cached.delivery_state,
            )
            self._prepared_run = (setup, cached, dt)
        step = self.populations[setup.ordered_population_names[0]].cell._clock_step.value
        if not is_traced_value(step):
            # Inside a compiled loop the count is a tracer: checking it would
            # cost a host round trip per step.
            check_step_limit(step, 1, dt=dt, prefix="Network.update()")
        self.trainables.materialize()
        values = self._take_value_inputs(setup.ordered_population_names)
        try:
            if self._prepared_plans is None:
                cached.runner(values)
            else:
                cached.runner(values, self._prepared_plans)
        except BaseException:
            self._restore_value_inputs(values)
            raise
        return {name: self.populations[name].cell.spike.value for name in setup.ordered_population_names}

    def _take_value_inputs(self, names) -> dict:
        """Take the value inputs of the Cells named ``names`` for one call of a compiled loop.

        They are arguments of the loop, which applies them to its first
        step (``Cell._take_value_inputs``); as constants of its trace they
        would act on every step, and on every later call.

        Returns
        -------
        dict
            ``{population name: (current, delta)}`` for the Cells that have
            any.
        """
        values = {}
        for name in names:
            taken = self.populations[name].cell._take_value_inputs()
            if any(taken):
                values[name] = taken
        return values

    def _restore_value_inputs(self, values) -> None:
        """Put back the inputs :meth:`_take_value_inputs` took, for a loop that failed."""
        for name, taken in values.items():
            self.populations[name].cell._restore_value_inputs(taken)

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

        # Live plasticity rows quantize their delays and allocate their count
        # history before lowering, which then leaves their delayed rows to
        # the rule boundary.
        for population in self._cell_populations().values():
            population.cell.connections._prepare_rules(dt, delay_quantization=delay_quantization)
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
            compiled_recordings=compiled_recordings,
            event_sources=event_sources,
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
            (name, id(population.cell._event_runtime()), population.size)
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
        schedule_key: tuple = (),
        scheduled_delivery: str | None = None,
    ) -> _CachedRunLoop:
        key = (
            setup_key,
            int(n_steps),
            int(brainstate.environ.get_precision()),
            schedule_key,
        )
        if scheduled_delivery is not None:
            key = key + (scheduled_delivery,)
        cached = self._network_run_loop_cache.get(key)
        if cached is None:
            delivery_state = self._delivery_state_cache.get(setup_key)
            if delivery_state is None:
                delivery_state = create_delivery_state(
                    setup.delivery_blocks,
                    delivery_ops=setup.delivery_ops,
                    counted=self._counted_layouts(),
                )
                self._delivery_state_cache[setup_key] = delivery_state
            cached = _CachedRunLoop(
                runner=self._make_network_run_loop(
                    setup=setup,
                    dt=dt,
                    n_steps=n_steps,
                    delivery_state=delivery_state,
                    scheduled_delivery=scheduled_delivery,
                ),
                delivery_state=delivery_state,
            )
            self._network_run_loop_cache[key] = cached
        return cached

    def _counted_layouts(self) -> frozenset:
        """The ``(population, layout_id)`` pairs whose layout counts events (``"sum_count"``, CELL-12b)."""
        return frozenset(
            (name, int(layout_id))
            for name, population in self._cell_populations().items()
            if not population.cell._uses_reduction
            for layout_id in population.cell._runtime.event_counts
        )

    def _make_network_run_loop(
        self,
        *,
        setup: _RunSetup,
        dt,
        n_steps: int,
        delivery_state,
        scheduled_delivery: str | None = None,
    ):
        """Create a persistent jitted scan of ``n_steps`` steps for one network run shape.

        The loop takes the value inputs and returns ``(times, start time,
        stop time, samples)``. Step ``k`` runs at ``origin + (step + k)·dt`` of the
        Cells' clocks, which :meth:`_start_clocks` made equal; the step count
        is a state, so a later segment reuses the loop (CELL-45).

        With ``scheduled_delivery`` the loop also takes the plans' arrays
        (:meth:`_scheduled_arrays`), so a window of the same length reuses
        it (CELL-9).
        """
        from braincell.network.event import boundary_step, grid_step

        delivery_blocks = setup.delivery_blocks
        ordered_population_names = setup.ordered_population_names
        compiled_recordings = setup.compiled_recordings
        event_sources = setup.event_sources
        cells = tuple(self.populations[name].cell for name in ordered_population_names)

        def _run_loop(values):
            origin = cells[0]._clock_origin.value
            step = cells[0]._clock_step.value
            times = origin + (step + jnp.arange(n_steps, dtype=jnp.int32)) * dt
            # The steps on the dt grid from time 0, which the event arrivals
            # are compared with (CELL-46).
            grid = grid_step(origin, step, dt) + jnp.arange(n_steps, dtype=jnp.int32)
            with brainstate.environ.context(dt=dt):

                def _step(t, s):
                    with brainstate.environ.context(t=t), boundary_step(t, s):
                        with jax.named_scope("braincell:network_run:prepare_clamps"):
                            for name in ordered_population_names:
                                self.populations[name].cell._prepare_step_clamps(t=t, dt=dt)
                        with jax.named_scope("braincell:network_run:sample_recordings"):
                            flat_recordings = tuple(
                                compiled for name in ordered_population_names for compiled in compiled_recordings[name]
                            )
                            pre_recording_snapshots = {
                                index: compiled.sample()
                                for index, compiled in enumerate(flat_recordings)
                                if compiled.phase == "pre"
                            }
                        with jax.named_scope("braincell:network_run:write_arrivals"):
                            write_arrivals(delivery_state, populations=self.populations)
                        with jax.named_scope("braincell:network_run:prepare_inputs"):
                            for name in ordered_population_names:
                                self.populations[name].cell._prepare_next_synapse_inputs()
                        with jax.named_scope("braincell:network_run:begin_cells"):
                            for name in ordered_population_names:
                                self.populations[name].cell._begin_step()
                        with jax.named_scope("braincell:network_run:update_cells"):
                            with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                                for name in ordered_population_names:
                                    cell = self.populations[name].cell
                                    cell._update_dynamics()
                        with jax.named_scope("braincell:network_run:apply_zero_delay_events"):
                            apply_immediate_events(
                                delivery_blocks,
                                delivery_state,
                                populations=self.populations,
                            )
                        with jax.named_scope("braincell:network_run:sample_outputs"):
                            recording_snapshots = tuple(
                                pre_recording_snapshots[index] if compiled.phase == "pre" else compiled.sample()
                                for index, compiled in enumerate(flat_recordings)
                            )
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
                        return recording_snapshots + events

                if values:

                    def _step_with_values(index, t, s):
                        # The value inputs act on the first step only.
                        with contextlib.ExitStack() as stack:
                            for name, taken in values.items():
                                cell = self.populations[name].cell
                                stack.enter_context(cell._value_input_step(taken, first=index == 0))
                            return _step(t, s)

                    samples_over_time = brainstate.transform.for_loop(
                        _step_with_values, jnp.arange(times.shape[0]), times, grid
                    )
                else:
                    samples_over_time = brainstate.transform.for_loop(_step, times, grid)

            end_step = (step + n_steps).astype(jnp.int32)
            for cell in cells:
                cell._clock_step.value = end_step
            return times, origin + step * dt, origin + end_step * dt, samples_over_time

        if scheduled_delivery is None:
            return brainstate.transform.jit(_run_loop)
        routes = self._scheduled_plans.routes

        def _scheduled_run_loop(values, plans):
            with scheduled_inputs(scheduled_delivery, routes, plans):
                return _run_loop(values)

        return brainstate.transform.jit(_scheduled_run_loop)

    def _scheduled_arrays(self, method: str, *, dt, window, first_config: bool) -> tuple:
        """Return the plans' arrays of ``window`` (``None``: every event) and the loop's schedule token.

        A plan that cannot be built (``PaddingLimitError``) leaves the
        runtime configuration as this call found it, so a first run can
        switch to another method.
        """
        from braincell.network.event import _host_ms

        try:
            plans = self._scheduled_plans
            if plans is None or plans.method != method:
                plans = ScheduledPlans.build(method, self.populations, float(_host_ms(dt)))
                self._scheduled_plans = plans
            arrays = plans.everything() if window is None else plans.window(*window)
        except BaseException:
            if first_config:
                self._runtime_config = None
            raise
        return arrays, loop_schedule_key(self.populations, plans)

    def _grid_start_step(self, names, dt) -> int:
        """The step on the ``dt`` grid from time 0 at which a run of the Cells ``names`` starts.

        It is the loop's first ``grid_step`` (CELL-46): the clock's step
        plus its origin in steps, which is 0 unless a Cell ran at another
        ``dt`` before.
        """
        from braincell.network.event import grid_step

        cell = self.populations[names[0]].cell
        origin_ms, step, _ = cell._clock_host()
        if origin_ms == 0.0:
            return int(step)
        return int(grid_step(cell._clock_origin.value, jnp.asarray(step, dtype=jnp.int32), dt))

    def _prepare_netstim_schedules(self, *, until_ms: float | None) -> tuple:
        """Generate NetStim events up to ``until_ms``; return the loop-cache token."""
        from braincell.network.event import _prepare_netstim_schedules

        return _prepare_netstim_schedules(
            (population.model for population in self.populations.values() if population.kind == "netstim"),
            until_ms=until_ms,
        )

    def _start_clocks(self, names: tuple[str, ...], *, dt, n_steps: int, prefix: str) -> float:
        """Give the Cells named ``names`` one clock counting steps of ``dt``; return its time in ms.

        The Cells must be at the same ``current_time``. They take the clock
        of the first, counting from its time if it last ran at another
        ``dt`` (``Cell._clock_start``); the run loop reads it from the
        first and advances them all. Raises ``ValueError``, before changing
        any clock, when ``n_steps`` more steps would pass the step limit.
        The usual call, on clocks already shared and counting ``dt``, only
        reads them on the host.
        """
        cells = [self.populations[name].cell for name in names]
        clocks = [cell._clock_host() for cell in cells]
        times = [origin + step * clock_dt for origin, step, clock_dt in clocks]
        if not np.allclose(times[1:], times[0]):
            raise ValueError("Network populations must have the same current_time.")
        origin_ms, step, counts_dt = host_clock_start(clocks[0], dt)
        check_step_limit(step, n_steps, dt=dt, prefix=prefix)
        if not counts_dt:
            origin, step_value = cells[0]._clock_start(dt)
            for cell in cells:
                cell._set_clock(origin, step_value, dt)
        elif any(clock != clocks[0] for clock in clocks[1:]):
            first = cells[0]
            for cell in cells[1:]:
                cell._set_clock(first._clock_origin.value, first._clock_step.value, first._clock_dt.value)
        return origin_ms + step * scalar_decimal(dt, u.ms)

    def _cell_populations(self) -> dict[str, Population]:
        return {name: population for name, population in self.populations.items() if population.kind == "cell"}

    def _require_registered_cell(self, cell, *, role: str) -> Population:
        for population in self.populations.values():
            if population.kind == "cell" and population.cell is cell:
                return population
        raise RuntimeError(f"Network.connect {role} is not registered in this Network.")

    def _require_registered_source(self, source) -> Population:
        from braincell.network.event import EventSource, EventSourceView

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
        from braincell.network.event import EventSource, EventSourceView

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
                if population.kind == "netstim":
                    table = population.model._event_table(until_ms=stop_ms)
                else:
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
        check_step_limit(self._source_step, n_steps, dt=dt, prefix="Network.run(...)")
        step = self._source_step
        times = 0.0 * u.ms + (step + u.math.arange(n_steps)) * dt
        start_t = 0.0 * u.ms + step * dt
        stop_t = 0.0 * u.ms + (step + n_steps) * dt
        events = self._result_events(
            times=times,
            start_t=start_t,
            stop_t=stop_t,
            event_sources={},
            event_values=(),
        )
        self._source_step = step + n_steps
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
        count_ring = state.count_ring(index)
        if count_ring is not None:
            count_ring.value = jnp.zeros_like(count_ring.value)
