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
import copy
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
    MAX_STEPS,
    _arrival_origin,
    _arrival_origin_ms,
    _dt_change_handover,
    _duration_steps,
    _float64_clock,
    _keep_started_clock,
    _kept_handover,
    _sample_rows,
    axial_operator_before_steps,
    check_step_limit,
    host_clock_start,
)
from braincell._multi_compartment.synapses import SynapseView
from braincell.network.connection import NetworkConnections, PairingSpec, _UNSET, _connect_with_pairing_seed
from braincell.network.recording import EventSeries, SampleBlock, step_end_times

from braincell.network.core import NetworkResult, Population
from braincell.network.delivery import (
    DeliveryBlock,
    advance_delivery_state,
    create_delivery_state,
    delivery_blocks as build_delivery_blocks,
    make_delivery_op,
    normalize_event_backend,
    record_source_events,
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
from braincell.quad.protocol import DiffEqState


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


@dataclass(frozen=True)
class _Join:
    """What the Cells of a run take over from the runs before it (CELL-57).

    ``handovers`` holds ``(name, ArrivalHandover)`` for the Cells that ran
    at another ``dt`` before, whose arrival steps keep what those runs took;
    it is empty for Cells that only the Network ran, or ran at its ``dt``.
    A run on its own leaves no boundary prepared (CELL-55), so the
    Network's first step prepares the boundary it starts at for every Cell.
    """

    handovers: tuple = ()

    @property
    def key(self) -> tuple:
        return self.handovers


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
        # The NetStim horizon of update(): its events up to one step past the step limit (CELL-56).
        self._prepared_until_ms: float | None = None
        # The step update() may not reach: prepare_run(duration=...) generated
        # no NetStim events for it (CELL-66). None without a duration.
        self._prepared_stop_step: int | None = None
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
            setup, cached, dt, join = self._prepared_run
            self._prepared_run = (setup, _CachedRunLoop(runner=None, delivery_state=cached.delivery_state), dt, join)

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
        # A copy of the same type: the Cell's mapping stays one leaf to the graph walk (CELL-51).
        previous_origins = copy.copy(target_cell._synapse_origins)
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
            setup, cached, dt, join = self._prepared_run
            for population in self._cell_populations().values():
                population.cell._set_clock(0.0 * u.ms, 0, dt)
            if join.key:
                # From t = 0 no earlier run took any event: update() builds
                # its loop, and the plan, again without the hand-overs
                # (CELL-57).
                cached = _CachedRunLoop(runner=None, delivery_state=cached.delivery_state)
                self._prepared_run = (setup, cached, dt, _Join())
                if self._prepared_plans is not None:
                    self._prepared_plans, _ = self._scheduled_arrays(
                        self._runtime_config[-1], dt=dt, window=None, first_config=False
                    )
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
            Samples of the half-open segment ``[start_time, stop_time)``;
            the events of its steps: a Cell's, emitted in the step from
            ``t_m`` to ``t_{m+1}``, at ``t_{m+1}``, in
            ``(start_time, stop_time]`` (CELL-80), and a NetStim's or an
            EventSequence's at their own times in ``[start_time, stop_time)``.

        Raises
        ------
        ValueError
            If the run would take the step count past ``2**31 - 1`` (14.9 h
            at ``dt = 0.025 ms``); nothing has run. :meth:`reset_state`
            starts the count again. Also if ``scheduled_delivery`` is not
            one of its values, before anything is initialized. Also if the
            Cells would start off the ``dt`` grid from ``t = 0``: a Cell
            that ran on its own at another ``dt`` (3 steps of 0.025 ms reach
            0.075 ms, 1.5 steps of 0.05 ms) cannot start a Network, whose
            recordings and scheduled events assume the grid; nothing has
            changed. Call :meth:`reset_state`, or run the Cell on its own to
            a multiple of ``dt`` before adding it. Also if Cells that ran on
            their own are not at one ``current_time`` (a Network steps them
            on one clock): clocks counting the same ``dt`` from one grid
            must have counted the same step; other clocks must agree to
            within ``4·eps·max(dt, |t|)`` of the run precision, which in
            float32 passes one step of 0.025 ms from 52.4 s on. Nothing has
            changed; call :meth:`reset_state` (after :meth:`init_state` if a
            Cell is not initialized), or run the Cells on their own to the
            same time before adding them.
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

        **Cells that ran on their own** (CELL-57, CELL-55). A step of
        ``Cell.run``, like a Network step, prepares the boundary it starts
        at and consumes it within the step, so a Cell's run took the
        scheduled events up to half of its step before the time it reached.
        A Network the Cell then joins, at a time on its ``dt`` grid,
        prepares that time's boundary in its first step: each event, and
        each plasticity rule update, happens once. When the Cell ran at
        another ``dt``, the Network's steps skip the events the Cell's run
        took and deliver the others on their own grid, none before the
        join, as in a Network run from ``t = 0``. A Cell that later runs on
        its own at another ``dt`` skips what the Network delivered in the
        same way. :meth:`reset_state` forgets the join.

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
          records, the steps and the rows. A window keeps the record slots
          of the largest window of its length the Network planned, so a
          later one whose records fit reuses the compiled loop; a window
          that needs more takes twice its records plus 128 slots and
          traces the loop again. It is reverse-differentiable to
          first order only: forward mode (``jax.jvp``, ``jacfwd``) through it
          raises ``TypeError`` when traced.
        - ``"padded"`` stores ``steps x width`` slots, ``width`` being the
          most records any step has, rounded up to a power of two so that
          windows of one length share a shape. It raises
          :class:`~braincell.network.scheduled.PaddingLimitError` before the
          run when the slots would take more than 256 MiB, or the steps
          times the needed width would exceed 8 times the records; it does
          not fall back to another method. The ratio is not checked for a
          window whose slots take no more bytes than a padded plan this
          Network already built for the same synapse layout, so the
          sparse tail of a schedule runs in the storage an earlier window
          was accepted in; whether a window passes thus depends on the
          windows run before it (since initialization; :meth:`reset_state`
          keeps them). The plan :meth:`prepare_run` builds is checked on
          the steps it stores, which are only those with arrivals when
          the window form would not fit. It differentiates as ordinary
          JAX code, forward mode included.

        The default is ``None`` on every device; choose a method by
        measuring. In ``benchmarks/performance/synapse_events`` (its
        ``regression`` phase, on one CPU and one GPU, an RTX 3090) the
        better method depended on the device:

        - On the CPU, ``"direct"`` and ``"padded"`` were several times
          faster when many scheduled rows have few arrivals per step and
          share synapses, and ``"direct"`` helped when a source's schedule
          is much longer than the run.
        - On the GPU, ``"direct"`` was slower than the default for every
          model measured, and ``"padded"``, within its limits, was 1.05 to
          1.5 times faster.

        Keep the default for small models: preparing a plan costs more than
        it saves. The method is part of the runtime configuration, fixed
        after the first run as ``event_backend`` is; a first run that raised
        ``PaddingLimitError`` fixes nothing.

        **Compiled loops** (CELL-74). A run with as many steps as an
        earlier one reuses its compiled loop. On the default path
        (``scheduled_delivery=None``) the loop holds the NetStim events
        generated so far as constants. A NetStim with ``size`` times
        ``max(number)`` at most ``2**20`` generates every event at
        construction, so its events never change the loop. A larger one
        generates them on demand: a run that needs later events first
        extends the schedule to at least twice its width, then traces and
        compiles the loop again, which reads every generated event. So
        ``k`` runs of one length compile the loop about ``log2(k) + 1``
        times; after :meth:`reset_state` the schedule keeps its width and
        the runs compile nothing. The cost of an extension grows with the
        generated events: with a million sources, on one CPU, the runs that
        extended the schedule took 3.1 to 5.5 s and those that reused the
        loop 0.6 to 0.9 s. With ``scheduled_delivery`` the plans are
        arguments of the loop, so a NetStim that only planned rows read
        grows without tracing the loop again; ``"direct"`` is the path for
        long continuations from large NetStims. A NetStim that a plasticity
        rule reads, as its rows' source or its ``post`` signal, still keys
        the loop: the rule reads its events when the loop is traced.

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
        self._reject_different_times(dt, prefix="Network.run(...)")
        self._reject_off_grid_start(dt, prefix="Network.run(...)")
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
        start_ms, join = self._start_clocks(ordered_population_names, dt=dt, n_steps=n_steps, prefix="Network.run(...)")
        n_recording = setup.n_recording
        n_event = setup.n_event
        # The step of the grid from time 0 the run starts at, on which its
        # schedules sample and its scheduled events arrive (CELL-46, CELL-59a).
        start_step = self._grid_start_step(ordered_population_names, dt)
        # NetStim events up to one step past the end cover every arrival the
        # loop delivers; later events are not generated.
        until_ms = start_ms + (n_steps + 1) * scalar_decimal(dt, u.ms)
        schedule_key = self._prepare_netstim_schedules(until_ms=until_ms)
        plans = None
        if scheduled_delivery is not None:
            # The plans of the window [step, step + n_steps), before the loop (CELL-9).
            plans, schedule_key = self._scheduled_arrays(
                scheduled_delivery,
                dt=dt,
                window=(start_step, n_steps),
                first_config=first_config,
                handovers=join.handovers,
            )
        cached_loop = self._network_run_loop(
            setup=setup,
            setup_key=setup_key,
            dt=dt,
            n_steps=n_steps,
            schedule_key=schedule_key,
            scheduled_delivery=scheduled_delivery,
            join=join,
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
                rows = _sample_rows(start_step, n_steps, compiled.schema, dt)
                selected = values[rows.start : rows.stop : rows.step]
                first_time = times[rows[0]] if rows else None
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
        duration=None,
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
            event it can deliver (see Notes), and is built once, here: a
            training loop of ``update()`` calls, split over several
            ``for_loop`` calls or rerun after :meth:`reset_state`, uses it
            throughout. Each synapse layout stores steps ``[0, last
            arrival + 1)``, as a run's window does, when that fits the
            method's budget (``"direct"``: no more steps than record slots;
            ``"padded"``: the limits in :meth:`run`). Otherwise it stores
            only the steps with arrivals, and each ``update()`` finds its
            step among them by binary search, so a schedule that reaches
            the step limit takes memory in proportion to its events. Steps
            without arrivals receive nothing, as on the default path.
        duration : brainunit.Quantity, optional
            How long the :meth:`update` calls that follow run: an integer
            multiple of ``dt``, ``n`` steps from the time reached. ``NetStim``
            events are then generated only for those steps, and
            :meth:`update` refuses, or turns to NaN, the steps after them
            (see Notes). ``None`` (default) prepares every step up to the
            step limit.

        Returns
        -------
        Network
            This initialized and prepared network.

        Raises
        ------
        ValueError
            If ``scheduled_delivery`` is not one of its values; or
            ``PaddingLimitError`` if a ``"padded"`` plan exceeds its limits;
            or if the Cells would start off the ``dt`` grid, or are not at
            one ``current_time``, as in :meth:`run`, before anything has
            changed. Also if ``duration`` is not a positive integer multiple
            of ``dt``, or the steps it adds would pass the step limit,
            before anything has changed.
        TypeError
            If ``duration`` has no time unit.
        RuntimeError
            If the configuration differs from an earlier run's.

        Notes
        -----
        Without ``duration``, ``update()`` runs for as many steps as its
        caller loops, but no further than the step limit, ``2**31 - 1``
        steps. ``NetStim`` events are generated here as far as that limit,
        however large ``number`` is: about ``2**31 * dt / interval`` per
        source, 5.4 million at ``dt = 0.025 ms`` and ``interval = 10 ms``,
        about 150 bytes each while a loop is traced. A large population of
        such sources (``number = 1e9``, as NEURON writes an unlimited train)
        then runs out of memory: give ``duration``.

        With ``duration`` of ``n`` steps, prepared at step ``k`` (``k = 0``
        on a new or reset network), the events are generated up to one step
        past step ``k + n``, so memory follows the duration. Step ``k + n``
        is the *stop*: ``update()`` delivers the events of the steps before
        it, and the steps from it on have none. So

        - an eager ``update()`` at or after the stop raises ``ValueError``,
          naming the stop, before anything changes;
        - inside a compiled loop (``brainstate.transform.for_loop``,
          ``jit``) the step is not known on the host, and a step at or
          after the stop multiplies every Cell's membrane and mechanism
          states and its spikes by NaN. The states, the spikes, a loss
          computed from them and its gradient are all NaN from that step on:
          **a training loss or gradient that turns NaN after a fixed number
          of steps means the loop ran past the stop.** The steps before it
          are those of a ``prepare_run`` without ``duration``, bit for bit.

        To run further, call ``prepare_run(dt=..., duration=...)`` again:
        the new duration counts from the time reached, and extends the
        events. :meth:`reset_state` returns to ``t = 0`` and keeps the
        stop, so the same steps run again. A function the caller compiled
        (``jax.jit``, a ``for_loop`` traced once) holds the stop and events
        it was traced with: after a second ``prepare_run`` trace it again,
        or its steps past the old stop are NaN. :meth:`run` does not use
        ``duration`` and generates its own events.

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

        An unlimited Poisson background, prepared for 10 ms:

        .. code-block:: python

            >>> cell = braincell.Cell(
            ...     braincell.Morphology.from_soma(length=20.0 * u.um, radius=10.0 * u.um), pop_size=1
            ... )
            >>> _ = cell.place(at("soma", 0.5), braincell.mech.Synapse("ExpSyn", name="s", tau=2.0 * u.ms))
            >>> net = braincell.Network("background", seed=1)
            >>> post = net.add_population("post", cell)
            >>> noise = net.add_population(
            ...     "noise",
            ...     braincell.NetStim(size=100, interval=10.0 * u.ms, number=10**9, noise=1.0, seed=3),
            ... )
            >>> _ = net.connect("drive", source=noise, synapse=post.synapses["s"], weight=0.001 * u.uS)
            >>> _ = net.prepare_run(dt=0.025 * u.ms, duration=10.0 * u.ms)
            >>> spikes = brainstate.transform.for_loop(lambda _: net.update()["post"], jnp.arange(400))
            >>> bool(u.math.isfinite(post.cell.V.value).all())
            True
        """
        validate_time_quantity(dt, name="dt", prefix="Network.prepare_run")
        n_prepared = 0
        if duration is not None:
            validate_time_quantity(duration, name="duration", prefix="Network.prepare_run")
            n_prepared = _duration_steps(duration, dt)
        event_backend = normalize_event_backend(event_backend)
        scheduled_delivery = normalize_scheduled_delivery(scheduled_delivery, prefix="Network.prepare_run(...)")
        self._reject_different_times(dt, prefix="Network.prepare_run(...)")
        self._reject_off_grid_start(dt, prefix="Network.prepare_run(...)")
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
        # Without a duration, update() runs for an unknown number of steps,
        # but no further than the step limit (each eager call checks it): as
        # for a run of that many steps, NetStim events up to one step past it
        # cover every arrival update() delivers, and later events are not
        # generated (CELL-56). With one, they cover its n steps, which end at
        # the stop update() checks (CELL-66).
        start_ms, join = self._start_clocks(
            setup.ordered_population_names, dt=dt, n_steps=n_prepared, prefix="Network.prepare_run(...)"
        )
        _, step, _ = self.populations[setup.ordered_population_names[0]].cell._clock_host()
        if duration is None:
            stop_step = None
            until_ms = start_ms + (MAX_STEPS - int(step) + 1) * scalar_decimal(dt, u.ms)
        else:
            stop_step = int(step) + n_prepared
            until_ms = start_ms + (n_prepared + 1) * scalar_decimal(dt, u.ms)
        schedule_key = self._prepare_netstim_schedules(until_ms=until_ms)
        plans = None
        if scheduled_delivery is not None:
            # One plan of every event, for every update() (CELL-9).
            plans, schedule_key = self._scheduled_arrays(
                scheduled_delivery, dt=dt, window=None, first_config=first_config, handovers=join.handovers
            )
        cached = self._network_run_loop(
            setup=setup,
            setup_key=key,
            dt=dt,
            n_steps=1,
            schedule_key=schedule_key,
            scheduled_delivery=scheduled_delivery,
            join=join,
        )
        self.trainables.materialize()
        self._prepared_run = (setup, cached, dt, join)
        self._prepared_plans = plans
        self._prepared_until_ms = until_ms
        self._prepared_stop_step = stop_step
        return self

    def update(self):
        """Advance a prepared network by one differentiable time step.

        Returns
        -------
        dict
            Floating spike arrays keyed by Cell population name: this
            step's, from ``t`` to ``t + dt``, whose time is ``t + dt`` (as
            in :meth:`run`'s events). Continuous observations remain
            available on each Cell's runtime states.

        Raises
        ------
        RuntimeError
            If :meth:`prepare_run` has not been called outside tracing.
        ValueError
            If called eagerly with the step count at its limit,
            ``2**31 - 1``, or at or after the stop of
            ``prepare_run(duration=...)``, before anything has changed; the
            message names the stop. Inside a compiled loop the count is not
            checked, as that would cost a host round trip per step: keep
            such a loop within the limit (14.9 h at ``dt = 0.025 ms``) and
            the duration (see Notes).

        Notes
        -----
        Compose with ``brainstate.transform.for_loop`` or ``scan``. No event
        tables or host-side recording conversions are built here.

        The step starts at ``t = n·dt`` and advances the step count ``n``
        by one, as in :meth:`run`: a loop of ``update()`` calls steps at the
        times of one ``run``.

        After ``prepare_run(duration=...)``, a step of a compiled loop at or
        after the stop multiplies every Cell's dynamic states and spikes by
        NaN, so they, a loss computed from them and its gradient are NaN
        from that step on; the earlier steps are unchanged. If a loss or
        gradient turns NaN after a fixed number of steps, the loop ran past
        the duration: prepare a longer one. A NaN spike compares false, so
        a boolean reading (``spike > 0``) shows no spike there rather than
        NaN; check the membrane potential or the float spikes.
        """
        from braincell.network.event import _schedule_horizon

        if self._prepared_run is None:
            raise RuntimeError("Call Network.prepare_run() before update().")
        setup, cached, dt, join = self._prepared_run
        step = self.populations[setup.ordered_population_names[0]].cell._clock_step.value
        stop = self._prepared_stop_step
        if not is_traced_value(step):
            # Inside a compiled loop the count is a tracer: checking it would
            # cost a host round trip per step.
            check_step_limit(step, 1, dt=dt, prefix="Network.update()")
            if stop is not None and int(step) >= stop:
                raise ValueError(_past_duration_message(int(step), stop))
        if cached.runner is None:
            # A Cell registered an input since the loop was traced, or
            # reset_state() dropped the hand-overs it was traced with.
            cached = _CachedRunLoop(
                runner=self._make_network_run_loop(
                    setup=setup,
                    dt=dt,
                    n_steps=1,
                    delivery_state=cached.delivery_state,
                    scheduled_delivery=self._runtime_config[-1],
                    join=join,
                ),
                delivery_state=cached.delivery_state,
            )
            self._prepared_run = (setup, cached, dt, join)
        self.trainables.materialize()
        values = self._take_value_inputs(setup.ordered_population_names)
        try:
            # A trace of the step reads the NetStim events prepare_run generated.
            with _schedule_horizon(self._prepared_until_ms):
                if self._prepared_plans is None:
                    cached.runner(values)
                else:
                    cached.runner(values, self._prepared_plans)
        except BaseException:
            self._restore_value_inputs(values)
            raise
        if stop is not None and is_traced_value(step):
            self._poison_past_stop(setup.ordered_population_names, step >= stop)
        return {name: self.populations[name].cell.spike.value for name in setup.ordered_population_names}

    def _poison_past_stop(self, names, past) -> None:
        """Multiply the Cells' dynamic states and spikes by NaN on a step past the prepared ones.

        A step of a compiled ``update()`` loop at or after the stop of
        ``prepare_run(duration=...)`` lacks the NetStim events that were not
        generated. Raising there would cost a host round trip per step, and
        setting NaN with ``where`` would leave the gradient finite; a factor
        ``where(past, nan, 1)`` makes the states, the loss and its gradient
        NaN from that step on, and leaves the others unchanged bit for bit
        (CELL-66).
        """
        factor = jnp.where(past, jnp.nan, 1.0)

        def poison(leaf):
            # Integration states and spikes are floating point.
            return leaf * factor.astype(leaf.dtype)

        for name in names:
            cell = self.populations[name].cell
            states = [state for state in brainstate.graph.states(cell).values() if isinstance(state, DiffEqState)]
            states.append(cell.spike)
            for state in states:
                state.value = jax.tree.map(poison, state.value, is_leaf=lambda x: isinstance(x, u.Quantity))

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
        # in write_arrivals runs instead -- so
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
        join: _Join = _Join(),
    ) -> _CachedRunLoop:
        key = (
            setup_key,
            int(n_steps),
            int(brainstate.environ.get_precision()),
            schedule_key,
        )
        if scheduled_delivery is not None:
            key = key + (scheduled_delivery,)
        if join.key:
            key = key + (join.key,)
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
                    join=join,
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
        join: _Join = _Join(),
    ):
        """Create a persistent jitted scan of ``n_steps`` steps for one network run shape.

        The loop takes the value inputs and returns ``(times, start time,
        stop time, samples)``. Step ``k`` runs at ``origin + (step + k)·dt`` of the
        Cells' clocks, which :meth:`_start_clocks` made equal; the step count
        is a state, so a later segment reuses the loop (CELL-45).

        With ``scheduled_delivery`` the loop also takes the plans' arrays
        (:meth:`_scheduled_arrays`), so a window of the same length reuses
        it (CELL-9).

        ``join`` (:class:`_Join`) holds what the Cells take over from their
        runs before (CELL-57): the arrival steps of each Cell keep its
        hand-over. Every step prepares the boundary it starts at for every
        Cell, the first included (CELL-55), so the loop of ``update()``,
        which runs one step per call, is the loop of a one-step run.
        """
        from braincell.network.event import boundary_step, grid_step, handed_over

        delivery_blocks = setup.delivery_blocks
        ordered_population_names = setup.ordered_population_names
        compiled_recordings = setup.compiled_recordings
        event_sources = setup.event_sources
        cells = tuple(self.populations[name].cell for name in ordered_population_names)
        handovers = dict(join.handovers)

        def _prepare_inputs() -> None:
            for name in ordered_population_names:
                with handed_over(handovers.get(name)):
                    self.populations[name].cell._prepare_next_synapse_inputs()

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
                            write_arrivals(delivery_blocks, delivery_state, populations=self.populations)
                        with jax.named_scope("braincell:network_run:prepare_inputs"):
                            _prepare_inputs()
                        with jax.named_scope("braincell:network_run:begin_cells"):
                            for name in ordered_population_names:
                                self.populations[name].cell._begin_step()
                        with jax.named_scope("braincell:network_run:update_cells"):
                            with brainstate.environ.context(_braincell_step_clamps_prepared=True):
                                for name in ordered_population_names:
                                    cell = self.populations[name].cell
                                    cell._update_dynamics()
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
                        with jax.named_scope("braincell:network_run:record_source_events"):
                            record_source_events(delivery_state)
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

        loop = axial_operator_before_steps(_run_loop, cells)
        if scheduled_delivery is None:
            return brainstate.transform.jit(loop)
        routes = self._scheduled_plans.routes

        def _scheduled_run_loop(values, plans):
            with scheduled_inputs(scheduled_delivery, routes, plans):
                return loop(values)

        return brainstate.transform.jit(_scheduled_run_loop)

    def _scheduled_arrays(self, method: str, *, dt, window, first_config: bool, handovers: tuple = ()) -> tuple:
        """Return the plans' arrays of ``window`` (``None``: every event) and the loop's schedule token.

        ``handovers`` holds ``(population, ArrivalHandover)`` of the Cells
        that ran at another ``dt`` before (CELL-57). A plan that cannot be
        built (``PaddingLimitError``) leaves the runtime configuration as
        this call found it, so a first run can switch to another method.
        """
        from braincell.network.event import _host_ms

        try:
            plans = self._scheduled_plans
            if plans is None or plans.method != method:
                plans = ScheduledPlans.build(method, self.populations, float(_host_ms(dt)))
                self._scheduled_plans = plans
            plans.handovers = dict(handovers)
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

    def _reject_different_times(self, dt, *, prefix: str) -> None:
        """Raise ``ValueError``, before changing anything, if the Cells are not at one time.

        The Cells take the clock of the first (:meth:`_start_clocks`): a
        Cell that ran on its own to another time would run from the first
        Cell's, skipping or repeating the difference (CELL-60: 100000.5 ms
        ran from 100000 ms, 20 steps of 0.025 ms back, and took again an
        event it had taken). A Cell whose clock states equal the first's is
        at its time: a run or :meth:`prepare_run` gives the members one
        clock, and a member cannot run on its own, so only Cells that ran
        on their own before joining can differ. :func:`_same_time` compares
        those with the first; a Cell not yet initialized is at 0.
        """
        populations = list(self._cell_populations().items())
        if len(populations) < 2:
            return
        dt_ms = scalar_decimal(dt, u.ms)
        (first, first_population), *others = populations
        first_states, first_clock = _host_clock(first_population.cell)
        for name, population in others:
            states, clock = _host_clock(population.cell)
            # Equal states step together, whatever float64 values the Cells
            # kept beside them (each keeps the time it reached when a run
            # here changes its dt).
            if states == first_states or _same_time(first_clock, clock, dt_ms):
                continue
            times = tuple(origin + step * clock_dt for origin, step, clock_dt in (first_clock, clock))
            shown = tuple(f"{time:.9g}" for time in times)
            if shown[0] == shown[1]:
                shown = tuple(repr(time) for time in times)
            raise ValueError(
                f"{prefix} requires its Cell populations at the same current_time: population {first!r} is at "
                f"t = {shown[0]} ms and population {name!r} at t = {shown[1]} ms. The Cells step on one clock, "
                f"the first population's, so {name!r} would skip or repeat the difference. Call "
                "Network.reset_state() to start them all from t = 0 (after Network.init_state() if one is not "
                "initialized yet), or run the Cells on their own to the same time before adding them."
            )

    def _reject_off_grid_start(self, dt, *, prefix: str) -> None:
        """Raise ``ValueError``, before changing anything, if the Network would start off the ``dt`` grid from ``t = 0``.

        A Network steps at ``t = (n + k)·dt`` (CELL-45): its recordings
        sample, and its scheduled events arrive, on that grid. Its Cells
        take the clock of the first (:meth:`_start_clocks`), which counts
        from the time that Cell reached if it ran on its own at another
        ``dt``. That time can fall between two steps (3 steps of 0.025 ms
        reach 0.075 ms, 1.5 steps of 0.05 ms): such a run recorded nothing
        and delivered each scheduled event half a step early (CELL-58). A
        time on the grid, to within the run precision, continues as it
        did. A Network whose ``dt`` is fixed at another value raises its
        own ``RuntimeError`` instead.
        """
        dt_ms = scalar_decimal(dt, u.ms)
        if self._runtime_config is not None and self._runtime_config[0] != dt_ms:
            return
        first = next(iter(self._cell_populations().items()), None)
        if first is None or not getattr(first[1].cell, "_initialized", False):
            return
        name, cell = first[0], first[1].cell
        origin_ms, step, _ = host_clock_start(_float64_clock(cell, cell._clock_host()), dt)
        if _arrival_origin(origin_ms, dt) is None:
            return
        raise ValueError(
            f"{prefix} cannot start population {name!r} at t = {origin_ms + step * dt_ms:.9g} ms, which is not "
            f"a multiple of dt = {dt_ms:g} ms: its Cell ran on its own at another dt. A Network steps at "
            "multiples of dt, where it samples recordings and delivers scheduled events. Call "
            "Network.reset_state() to start from t = 0, or run the Cell on its own to a multiple of dt before "
            "adding it."
        )

    def _start_clocks(self, names: tuple[str, ...], *, dt, n_steps: int, prefix: str) -> tuple[float, _Join]:
        """Give the Cells named ``names`` one clock counting steps of ``dt``; return its time in ms and the join.

        The Cells are at one ``current_time``, to within the run precision
        (:meth:`_reject_different_times` checked it at the entry). They take
        the clock of the first, counting from its time if it last ran at another
        ``dt`` (``Cell._clock_start``); the run loop reads it from the
        first and advances them all. Raises ``ValueError``, before changing
        any clock, when ``n_steps`` more steps would pass the step limit.
        The usual call, on clocks already shared and counting ``dt``, only
        reads them on the host.

        The :class:`_Join` holds the hand-over of each Cell that ran at
        another ``dt`` (CELL-57). A run on its own leaves no boundary
        prepared (CELL-55), so the hand-over starts at the time reached. A
        Cell whose clock changes ``dt`` here keeps its float64 time, ``dt``
        and hand-over beside the new clock, as ``Cell.run`` does, for a
        later run on its own.
        """
        cells = [self.populations[name].cell for name in names]
        clocks = [cell._clock_host() for cell in cells]
        origin_ms, step, counts_dt = host_clock_start(clocks[0], dt)
        check_step_limit(step, n_steps, dt=dt, prefix=prefix)
        if not counts_dt:
            handovers = [_dt_change_handover(cell, clock, dt, prepared=False) for cell, clock in zip(cells, clocks)]
            reached = [
                origin64 + step64 * dt64
                for origin64, step64, dt64 in (_float64_clock(cell, clock) for cell, clock in zip(cells, clocks))
            ]
            origin, step_value = cells[0]._clock_start(dt)
            for cell in cells:
                cell._set_clock(origin, step_value, dt)
            dt_ms = scalar_decimal(dt, u.ms)
            for cell, time_ms, handover in zip(cells, reached, handovers):
                _keep_started_clock(cell, time_ms, dt_ms, handover)
        else:
            handovers = [_kept_handover(cell, clock) for cell, clock in zip(cells, clocks)]
            if any(clock != clocks[0] for clock in clocks[1:]):
                first = cells[0]
                for cell in cells[1:]:
                    cell._set_clock(first._clock_origin.value, first._clock_step.value, first._clock_dt.value)
        join = _Join(
            handovers=tuple((name, handover) for name, handover in zip(names, handovers) if handover is not None),
        )
        return origin_ms + step * scalar_decimal(dt, u.ms), join

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
        host_times = None
        event_index = 0
        for name, population in self.populations.items():
            if population.kind != "cell":
                if population.kind == "netstim":
                    # Only the window's events: not a table of every generated one.
                    source_id, event_ms = population.model._window_events(start_ms, stop_ms)
                else:
                    table = population.model.events
                    event_ms = np.asarray(table.time.to_decimal(u.ms), dtype=float)
                    selected_rows = np.flatnonzero((event_ms >= start_ms) & (event_ms < stop_ms))
                    source_id, event_ms = table.source_index[selected_rows], event_ms[selected_rows]
                if source_id.size:
                    order = np.lexsort((source_id, event_ms))
                    source_id, event_ms = source_id[order], event_ms[order]
                port = "spike" if population.kind == "netstim" else "event"
                count = np.ones(int(source_id.size), dtype=np.int64)
                events[name] = {
                    port: EventSeries(
                        time=u.Quantity(event_ms, u.ms),
                        source_id=source_id,
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
                if host_times is None:
                    # On the host: indexing the device times compiles a gather for each new event count.
                    # A step's events take its end time, t_{m+1} (CELL-80).
                    host_times, unit = step_end_times(times, stop_t)
                events[name][port] = EventSeries(
                    time=u.Quantity(host_times[time_index], unit),
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


def _past_duration_message(step: int, stop: int) -> str:
    """The error of an eager ``Network.update()`` at or past the stop of ``prepare_run(duration=...)``."""
    return (
        f"Network.update() at step {step}: prepare_run(duration=...) prepared the steps before step {stop} "
        "and generated no NetStim events after them; call prepare_run(..., duration=...) again with a "
        "duration that covers the steps to run, or reset_state() to start again from t = 0."
    )


def _reject_batch_size(batch_size) -> None:
    """Reject the batch argument both lifecycle entry points still accept."""
    if batch_size is not None:
        raise NotImplementedError(
            "Network batch execution is not implemented yet; use Cell batch execution for same-topology batches."
        )


def _host_clock(cell) -> tuple[tuple[float, int, float], tuple[float, int, float]]:
    """The Cell's clock on the host, ``(origin in ms, step, dt in ms)``, as its states hold it and at float64.

    The second is ``_float64_clock``'s: at float64 where the Cell kept
    the values so. Before ``init_state`` both are ``(0, 0, 0)``, the clock
    ``init_state`` sets.
    """
    if not getattr(cell, "_initialized", False):
        return (0.0, 0, 0.0), (0.0, 0, 0.0)
    states = cell._clock_host()
    return states, _float64_clock(cell, states)


def _same_time(first, other, dt_ms: float) -> bool:
    """Whether two host clocks (:func:`_host_clock`) are at one time, for a Network stepping at ``dt_ms``.

    Clocks that count the same ``dt`` (equal at float64) from the same
    origin, or from origins both on that ``dt``'s grid, are at one time
    when they have counted the same step from ``t = 0``: integers, exact
    at any time. Other clocks, whose histories differ (another ``dt``, an
    origin between steps, a run inside ``jax.jit``, which keeps no float64
    values), agree when their times differ by at most
    ``4·eps·max(dt, |t|)`` of the run precision, the basis of the ``dt``
    grid check (:meth:`Network._reject_off_grid_start`). That bound grows
    with ``t``: in float32 it passes one step of 0.025 ms from 52.4 s on,
    as the grid check does.
    """
    (origin_a, step_a, dt_a), (origin_b, step_b, dt_b) = first, other
    if dt_a == dt_b:
        if origin_a == origin_b:
            return step_a == step_b
        if dt_a > 0 and _arrival_origin_ms(origin_a, dt_a) is None and _arrival_origin_ms(origin_b, dt_b) is None:
            return int(np.rint(origin_a / dt_a)) + step_a == int(np.rint(origin_b / dt_b)) + step_b
    time_a, time_b = origin_a + step_a * dt_a, origin_b + step_b * dt_b
    eps = float(np.finfo(brainstate.environ.dftype()).eps)
    return abs(time_a - time_b) <= 4.0 * eps * max(dt_ms, abs(time_a), abs(time_b))


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
    """Clear the event histories at an explicit network reset boundary."""
    for history, cursor in zip(state.histories, state.history_cursors):
        if history is not None:
            history.value = jnp.zeros_like(history.value)
            cursor.value = np.asarray(0, dtype=np.int32)
