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

"""Connection plasticity rules at runtime: binding, state, and the rule boundary (CELL-12a).

A rule is declared on one ``connect`` call (``plasticity=``) and covers all
of its rows. :func:`bind_rule` checks the declaration there and builds the
runtime rule. The target Cell owns the rule's state: ``init_state`` sets the
dynamic weight to the rows' weight and the rest of the state to the rule's
initial values, ``reset_state`` does it again, ``Cell.reset`` drops it.

A rule runs once per event boundary ``t_n = n dt``, in the Cell's input
preparation for that boundary (:func:`advance_rules`), before the synapses
consume its events. Every read at a boundary sees the state before it:

- ``pre`` is each row's arrivals at the boundary, the events its synapse
  receives there. A scheduled source counts them as the Cell does for any
  scheduled row. A live source detected in step ``m`` with a delay of ``k``
  steps arrives at ``t_{m + max(k, 1)}``.
- A row with a scheduled source, or a live source and ``k >= 1``, delivers
  ``n * w(t_n-)`` here (and ``n`` to a ``"sum_count"`` synapse's event
  count, CELL-12b), not through the Network's weighted delay ring: a
  live call keeps a history of its sources' spike counts instead, one
  column per distinct source (fan-out rows share it). A live row with
  ``k = 0`` keeps the Network's immediate path, at the end of the detection
  step, which reads the dynamic weight; the rule has not run for that
  boundary yet, so it reads ``w(t_n-)`` too.
- ``post`` is the events of the member that owns each row's synapse:
  a live source's events of step ``m`` are at ``t_{m + 1}``, a scheduled
  source's at the boundary its events round to.

Live rule rows need the Network's step and delay quantization, so they run
in a Network only; a standalone ``Cell.run`` raises for them.
"""

from __future__ import annotations

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base_rule import ConnectionRule as RuntimeRule
from braincell._multi_compartment.synapses import SynapseView
from braincell.mech import ConnectionRule, get_registry
from braincell.network.event import EventSource, EventSourceView
from braincell.network.lowering import _expand_delay_steps

__all__ = [
    "RuleBinding",
    "advance_rules",
    "bind_rule",
]

#: Signals the runtime provides to a rule besides ``pre``, with whether a
#: missing binding has a default (``post``: the target Cell's spikes).
_PROVIDED_SIGNALS = {"post": True}


def bind_rule(declaration, *, cell, weight, n_rows: int) -> "RuleBinding":
    """Check a ``plasticity=`` declaration at ``connect`` and build its runtime rule.

    Parameters
    ----------
    declaration : braincell.mech.ConnectionRule
        The declaration passed to ``connect``.
    cell : Cell
        The Cell that owns the destination synapses.
    weight : brainunit.Quantity or None
        The rows' normalized weight; ``None`` for a trigger-only synapse.
    n_rows : int
        Number of rows the call creates.

    Returns
    -------
    RuleBinding
        The declaration, the runtime rule and the resolved ``post`` signal.

    Raises
    ------
    TypeError
        If ``declaration`` is not a ``ConnectionRule``, the synapse is
        trigger-only, the rule class is not a ``braincell.ConnectionRule``,
        or a signal name is not one the rule declares.
    KeyError
        If no rule is registered under the declared name.
    ValueError
        If a ``post`` source has another number of endpoints than the
        target Cell has members, or is a live source of another Cell; and
        whatever the rule raises for its parameters or the initial weight.
    """
    if not isinstance(declaration, ConnectionRule):
        raise TypeError(
            "connect(plasticity=...) takes a braincell.mech.ConnectionRule declaration, "
            f"got {type(declaration).__name__}."
        )
    if weight is None:
        raise TypeError(
            "A plasticity rule changes the weight of its rows, and rows to a trigger-only synapse "
            "(weight=None) have none."
        )
    cls = get_registry().get("connection_rule", declaration.rule_type)
    if not (isinstance(cls, type) and issubclass(cls, RuntimeRule)):
        raise TypeError(
            f"Connection rule {declaration.rule_type!r} resolves to {cls!r}, which is not a braincell.ConnectionRule."
        )
    name = declaration.rule_type
    declared = tuple(cls.signals)
    unknown = [signal for signal in declared if signal not in _PROVIDED_SIGNALS]
    if unknown:
        raise TypeError(
            f"Connection rule {name!r} reads signals {tuple(unknown)!r}, which braincell does not provide yet; "
            f"provided: {tuple(_PROVIDED_SIGNALS)!r}."
        )
    extra = [signal for signal in declaration.signals if signal not in declared]
    if extra:
        raise TypeError(
            f"Connection rule {name!r} has no signal {extra[0]!r} to bind; it accepts signals={declared!r}. "
            "'pre' is the arrivals of each row's own source and cannot be bound."
        )
    missing = [signal for signal in declared if signal not in declaration.signals and not _PROVIDED_SIGNALS[signal]]
    if missing:  # pragma: no cover - every signal provided so far has a default
        raise TypeError(f"Connection rule {name!r} needs signals {tuple(missing)!r}; bind them with signals={{...}}.")
    post = _post_signal(declaration.signals.get("post"), cell, name) if "post" in declared else None
    rule = cls(int(n_rows), u.get_unit(weight), **declaration.params)
    rule.check_weight(weight)
    return RuleBinding(declaration, rule, post)


def _post_signal(value, cell, name: str) -> EventSourceView:
    """The ``post`` source view of a rule on ``cell``: one endpoint per population member."""
    if value is None:
        return cell.event_outputs["spike"]
    if isinstance(value, EventSource):
        value = value.view
    if not isinstance(value, EventSourceView):
        raise TypeError(
            f"Connection rule {name!r} signal 'post' must be an EventSource or EventSourceView, "
            f"got {type(value).__name__}."
        )
    members = cell._population_size
    if len(value) != members:
        raise ValueError(
            f"Connection rule {name!r} signal 'post' has {len(value)} endpoints; it needs one per member of the "
            f"target Cell ({members})."
        )
    owner = value.owner
    if not owner.is_scheduled and owner.execution_owner is not cell:
        raise ValueError(
            f"Connection rule {name!r} signal 'post' is a live source of another model; a rule reads only the "
            "target Cell's own events (another Cell's events reach it through a connection), or a scheduled "
            "source such as EventSequence or NetStim as a teaching signal."
        )
    return value


class _RowPlan:
    """Static per-row arrays of one rule call, built once at ``init_state``."""

    __slots__ = (
        "n_rows",
        "active",
        "layout_id",
        "synapse_rows",
        "source_index",
        "delay",
        "post_index",
        "scheduled",
    )

    def __init__(self, cell, call, store, post) -> None:
        rows = store.rows(call.row_ids)
        active = np.asarray(store.active[rows], dtype=bool)
        n_rows = int(call.row_ids.size)
        source_index = np.asarray(store.source_index[rows], dtype=np.int64)
        synapse_id = np.asarray(store.synapse_id[rows], dtype=np.int64)
        synapses = cell._get_synapse_store()
        # Removed rows keep their slot (the rule's state covers the call) but
        # receive nothing and deliver nothing; their synapse ids may be stale.
        synapse_rows = np.zeros(n_rows, dtype=np.int32)
        member = np.zeros(n_rows, dtype=np.int64)
        if active.any():
            synapse_rows[active] = synapses.runtime_rows(synapse_id[active]).astype(np.int32)
            member[active] = SynapseView(cell, synapse_id[active]).population_index
        self.n_rows = n_rows
        self.active = None if active.all() else active
        self.layout_id = int(synapses.layout_id(call.synapse_type))
        self.synapse_rows = synapse_rows
        self.source_index = source_index
        self.delay = store.delay[rows]
        self.scheduled = bool(call.source.is_scheduled)
        self.post_index = None if post is None else np.asarray(post.source_id, dtype=np.int64)[member]


class _LivePlan:
    """Quantized delays and count-history layout of a live rule call, built at the Network run setup.

    A spike detected in step ``m`` is the source's current count at the
    boundary ``t_{m+1}``; with ``k`` delay steps it arrives at
    ``t_{m+max(k,1)}``, so a row reads the count written ``max(k-1, 0)``
    boundaries ago. The history keeps one column per distinct source: at
    ``1e5`` rows from ``1e3`` sources a step costs as much as without a rule,
    where one column per row cost 2.8 times as much (CELL-12a).
    """

    __slots__ = ("lag", "uniform_lag", "depth", "columns", "slot", "deliver")

    def __init__(self, steps, plan) -> None:
        lag = np.maximum(steps - 1, 0).astype(np.int32)
        if plan.active is not None:
            lag = np.where(plan.active, lag, 0).astype(np.int32)
        columns, slot = np.unique(plan.source_index, return_inverse=True)
        slot = slot.reshape(-1).astype(np.int32)
        self.lag = lag
        self.uniform_lag = int(lag[0]) if lag.size and bool(np.all(lag == lag[0])) else None
        self.depth = int(lag.max(initial=0)) + 1
        self.columns = columns
        # Rows that each read a source of their own, in source order, need no gather.
        self.slot = None if np.array_equal(slot, np.arange(columns.size)) else slot
        # Zero-delay rows stay on the Network's immediate path (lowering).
        deliver = steps >= 1
        self.deliver = deliver if plan.active is None else deliver & plan.active

    def per_row(self, counts):
        """Spread per-source ``counts`` to the rows."""
        return counts if self.slot is None else counts[jnp.asarray(self.slot)]


class RuleBinding:
    """The rule of one ``connect`` call: declaration, runtime rule, signals, and runtime state."""

    __slots__ = ("declaration", "rule", "post", "states", "runtime", "plan", "_live")

    def __init__(self, declaration, rule, post) -> None:
        self.declaration = declaration
        self.rule = rule
        self.post = post
        self.clear()

    # -- lifecycle --------------------------------------------------------

    def init(self, cell, call, store) -> dict:
        """Allocate the state at ``init_state``; return what the Cell keeps for its state tree."""
        self.plan = _RowPlan(cell, call, store, self.post)
        self.states = {}
        for name, value in self._initial_state(call.weight).items():
            self.states[name] = brainstate.ShortTermState(value)
        self.runtime = {}
        return {"state": self.states, "runtime": self.runtime}

    def reset(self, weight) -> None:
        """Return the state to its initial values for ``weight`` (``reset_state``)."""
        if self.states is None:
            return
        for name, value in self._initial_state(weight).items():
            self.states[name].value = value
        if "history" in self.runtime:
            self.runtime["history"].value = jnp.zeros_like(self.runtime["history"].value)
            self.runtime["cursor"].value = jnp.asarray(0, dtype=jnp.int32)

    def clear(self) -> None:
        """Drop the runtime state (``Cell.reset``); the declaration and rule stay."""
        self.states = None
        self.runtime = None
        self.plan = None
        self._live = None

    def _initial_state(self, weight) -> dict:
        weight = u.Quantity(jnp.asarray(u.get_mantissa(weight)), u.get_unit(weight))
        state = dict(self.rule.init_state(weight))
        if "w" not in state:
            raise TypeError(f"{type(self.rule).__name__}.init_state must return the dynamic weight as 'w'.")
        return state

    def set_weight(self, offsets, weight) -> None:
        """Write ``weight[offsets]`` into the dynamic weight (``ConnectionView.set``)."""
        if self.states is None:
            return
        state = self.states["w"]
        unit = u.get_unit(state.value)
        values = jnp.asarray(np.asarray(u.Quantity(weight).to_decimal(unit))[offsets])
        mantissa = u.get_mantissa(state.value)
        state.value = u.Quantity(mantissa.at[jnp.asarray(offsets)].set(values.astype(mantissa.dtype)), unit)

    def value(self, name: str):
        """The current value of the rule state ``name``, one entry per row of the call."""
        try:
            return self.states[name].value
        except KeyError:
            raise KeyError(
                f"Connection rule {self.declaration.rule_type!r} has no state {name!r}; "
                f"its states are {tuple(self.states)!r}."
            ) from None

    def initial_state(self, weight) -> dict:
        """The rule's initial state for ``weight``, as ``init_state`` sets it."""
        return self._initial_state(weight)

    def prepare_network(self, call, *, dt, delay_quantization: str) -> None:
        """Quantize a live call's delays and allocate its count history (Network run setup)."""
        if self.states is None or call.source.is_scheduled or self._live is not None:
            # A Network fixes dt and the delay quantization at its first run.
            return
        plan = self.plan
        steps = _expand_delay_steps(plan.delay, dt=dt, n_contact=plan.n_rows, quantization=delay_quantization)
        self._live = _LivePlan(steps, plan)
        if self._live.depth > 1:
            self.runtime["history"] = brainstate.ShortTermState(
                jnp.zeros((self._live.depth, self._live.columns.size), dtype=brainstate.environ.dftype())
            )
            self.runtime["cursor"] = brainstate.ShortTermState(jnp.asarray(0, dtype=jnp.int32))

    # -- the boundary -----------------------------------------------------

    def _require_live_plan(self):
        """The live plan a Network prepared; a standalone Cell has none, and is refused."""
        if self._live is None:
            raise NotImplementedError(
                f"Connection rule {self.declaration.rule_type!r} on a live source runs only in a braincell.Network: "
                "a standalone Cell delivers live events on another schedule. Add the Cells to a Network and "
                "run that, or connect a scheduled source (EventSequence, NetStim)."
            )
        return self._live

    def _live_arrivals(self, source):
        """Arrivals of a live call at this boundary, from its sources' count history (Network only)."""
        live = self._require_live_plan()
        current = jnp.asarray(source.current_event_count(live.columns), dtype=brainstate.environ.dftype())
        if "history" not in self.runtime:
            return live.per_row(current)
        history_state = self.runtime["history"]
        cursor_state = self.runtime["cursor"]
        cursor = cursor_state.value
        history = jax.lax.dynamic_update_index_in_dim(history_state.value, current, cursor, axis=0)
        if live.uniform_lag is not None:
            arrivals = jax.lax.dynamic_index_in_dim(
                history, (cursor - live.uniform_lag) % live.depth, axis=0, keepdims=False
            )
            arrivals = live.per_row(arrivals)
        else:
            column = jnp.arange(live.columns.size, dtype=jnp.int32) if live.slot is None else jnp.asarray(live.slot)
            arrivals = history[(cursor - jnp.asarray(live.lag)) % live.depth, column]
        history_state.value = history
        cursor_state.value = (cursor + 1) % live.depth
        return arrivals

    def _post_counts(self, *, t, dt):
        post = self.post
        index = self.plan.post_index
        dtype = brainstate.environ.dftype()
        owner = post.owner
        if owner.is_scheduled:
            counts = owner.event_count(index, t=t, delay=0.0 * u.ms, dt=dt)
        else:
            counts = owner.current_event_count(index)
        return jnp.asarray(counts, dtype=dtype)

    def advance(self, cell, call, *, t, dt) -> None:
        """Deliver this call's arrivals at boundary ``t`` with ``w(t-)``, then run the rule."""
        plan = self.plan
        dtype = brainstate.environ.dftype()
        if plan.scheduled:
            pre = jnp.asarray(call.source.event_count(plan.source_index, t=t, delay=plan.delay, dt=dt), dtype=dtype)
            deliver = plan.active
        else:
            pre = self._live_arrivals(call.source)
            deliver = self._live.deliver
        if plan.active is not None:
            pre = pre * jnp.asarray(plan.active, dtype=dtype)
        state = {name: item.value for name, item in self.states.items()}
        if deliver is None or deliver.any():
            payload = pre if deliver is None or deliver.all() else pre * jnp.asarray(deliver, dtype=dtype)
            runtime = cell._event_runtime()
            template = runtime.get_event_buffer(plan.layout_id)
            mantissa = _event_mantissa(template, state["w"])
            drive = jnp.zeros_like(_raw(template)).at[..., plan.synapse_rows].add(payload * mantissa)
            cell._set_event_buffer(plan.layout_id, template + _rewrap(template, drive))
            count_state = cell._event_count_state(plan.layout_id)
            if count_state is not None:
                # A "sum_count" synapse also receives the number of events (CELL-12b).
                counts = jnp.zeros_like(drive).at[..., plan.synapse_rows].add(payload.astype(drive.dtype))
                count_state.value = count_state.value + counts
        signals = {}
        if self.post is not None:
            signals["post"] = self._post_counts(t=t, dt=dt)
            if plan.active is not None:
                signals["post"] = signals["post"] * jnp.asarray(plan.active, dtype=dtype)
        new = self.rule.update(state, pre=pre, dt=dt, **signals)
        if set(new) != set(state):
            raise TypeError(
                f"{type(self.rule).__name__}.update returned states {tuple(new)!r}; it must return {tuple(state)!r}."
            )
        for name, value in new.items():
            self.states[name].value = value

    def weight_getter(self, offsets):
        """A reader of the dynamic weight of the rows at ``offsets``, for a lowered zero-delay block."""
        states = self.states
        if offsets.size == self.plan.n_rows and np.array_equal(offsets, np.arange(offsets.size)):
            return lambda: states["w"].value
        index = np.asarray(offsets, dtype=np.int64)
        return lambda: states["w"].value[index]


def reads_voltage_before_step(cell) -> bool:
    """Whether a rule of ``cell`` reads, at a boundary, the voltage before the step that ended there.

    A ``post`` signal that is a ``VoltageCrossingSource`` comparing the
    voltages before and after a step does, when the rule runs at the start
    of the next step (CELL-55); a step's run loop must then carry that
    voltage from step to step.
    """
    if not cell._connection_rules:
        return False
    for call in cell._get_connection_store()._calls:
        binding = call.plasticity
        if binding is None or binding.states is None or binding.post is None:
            continue
        if getattr(binding.post.owner, "_reads_previous_voltage", False):
            return True
    return False


def advance_rules(cell, *, t) -> None:
    """Run every plasticity rule of ``cell`` at the boundary ``t`` (its input preparation)."""
    dt = brainstate.environ.get_dt()
    for call in cell._get_connection_store()._calls:
        binding = call.plasticity
        if binding is not None and binding.states is not None:
            binding.advance(cell, call, t=t, dt=dt)


def _raw(template):
    return jnp.asarray(template.to_decimal(template.unit) if isinstance(template, u.Quantity) else template)


def _rewrap(template, mantissa):
    return u.Quantity(mantissa, template.unit) if isinstance(template, u.Quantity) else mantissa


def _event_mantissa(template, weight):
    """The dynamic weight in the event buffer's unit, as a mantissa."""
    if isinstance(template, u.Quantity):
        return weight.to_decimal(template.unit)
    return u.get_mantissa(weight)
