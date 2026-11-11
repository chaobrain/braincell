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

"""``dt`` convergence and stiffness of channel mechanisms, without NEURON.

**Convergence.** Every command of a voltage-clamp protocol is one population
of a channel_no_conc Cell, replayed with :mod:`validation.neuron._clamp`.
``vc_steps`` and the relaxation protocol (independent states start at 0,
-80 mV for 35 ms) run at each of :data:`DTS_MS` against ``exp_euler`` at
0.025 ms, exact while the voltage is constant over a step (HH gates, always
``ind_exp_euler`` in a Cell, are switched to ``exp_euler`` for it; a
channel without states has nothing to switch). ``vc_ramp`` has no
reference: :data:`RAMP_DTS_MS` (or finer steps, for a channel still
pre-asymptotic there) give two successive differences and their Richardson
order. ``e_I`` (normalized by the largest
reference current) and ``e_s`` are compared every :data:`SAMPLE_MS` ms.
:func:`classify` sorts the step result into :data:`EXACT`,
:data:`CONVERGENT`, :data:`ABOVE_GUIDE_LINE` or :data:`NOT_CONVERGENT`.

**Stiffness.** ``rho_max`` is the largest rate of the state equations over
:data:`STIFFNESS_GRID_MV` at 36 degC: the spectral radius of a Markov
transition matrix, or ``phi (alpha + beta)`` of an HH gate. The class is
stiff at ``dt`` when ``rho_max * dt > 2``, the stability bound of explicit
Euler and two-stage Runge-Kutta. **Singularities**: gate values, gate
derivatives and the current must be finite on :data:`SINGULARITY_GRID_MV`.
"""

from collections.abc import Iterable, Mapping
import contextlib
from dataclasses import dataclass, field
import inspect
import json
import math
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from validation.neuron import _clamp

CHANNEL_NO_CONC_ROOT = Path(__file__).resolve().parent / "channel_no_conc"

DTS_MS = (0.1, 0.05, 0.025, 0.0125, 0.00625)
"""Step sizes of the step and relaxation protocols, in ms."""
RAMP_DTS_MS = (0.05, 0.025, 0.0125)
"""Step sizes of the ramp's Richardson differences, in ms."""
REFERENCE_DT_MS = 0.025
REFERENCE_SOLVER = "exp_euler"
SAMPLE_MS = 0.1
"""Spacing of the comparison points; a multiple of every step size."""

EXACT_LIMIT = 1e-10
GUIDE_LINE = 1e-2
LIMIT_FACTOR = 1.5
"""A table limit is at least this multiple of the measured error; see :func:`rounded_limit`."""
STEP_ORDER_WINDOW = (0.8, 1.2)
RAMP_ORDER_WINDOWS = {"e_I": (0.7, 1.3), "e_s": (0.8, 1.2)}

EXACT = "exact"
CONVERGENT = "convergent"
ABOVE_GUIDE_LINE = "convergent, error above the guide line"
NOT_CONVERGENT = "not convergent"

STEPS, RAMP, RELAXATION = "vc_steps", "vc_ramp", "relaxation"
RELAXATION_SPEC = {"kind": "steps", "hold_mV": -80.0, "hold_ms": 0.0, "test_mV": [-80.0], "test_ms": 35.0, "tail_ms": 0.0}
"""Hold -80 mV for 35 ms; the run starts every independent channel state at 0."""

TEMPERATURE_CELSIUS = 36.0
STIFFNESS_DT_MS = 0.025
STIFFNESS_LIMIT = 2.0
STIFFNESS_GRID_MV = np.linspace(-100.0, 60.0, 641)
"""-100 to 60 mV in 0.25 mV steps."""
SINGULARITY_GRID_MV = np.arange(-100_000, 60_001) / 1000.0
"""-100 to 60 mV in 0.001 mV steps, each point the nearest double to its decimal value."""


# ---------------------------------------------------------------------------
# Convergence of one configuration


def _engine():
    from validation.neuron.channel_no_conc.engine import braincell_runner, voltage_clamp

    return braincell_runner, voltage_clamp


def load_case(config: str, protocol: str):
    """Return the case and the ``voltage_clamp`` block of ``protocol`` for ``config``.

    The relaxation protocol uses the ``vc_steps`` case (``v_init = -80 mV``).
    """
    _, voltage_clamp = _engine()
    template = CHANNEL_NO_CONC_ROOT / "templates" / f"{STEPS if protocol == RELAXATION else protocol}.json"
    case, _ = voltage_clamp.load_protocol(CHANNEL_NO_CONC_ROOT / "configs" / f"{config}.json", template)
    spec = RELAXATION_SPEC if protocol == RELAXATION else json.loads(template.read_text())["voltage_clamp"]
    return case, spec


def channel_class(case):
    from braincell.mech import get_registry

    return get_registry().get("channel", case.mapping_spec.braincell.class_name)


def _is_markov(cls) -> bool:
    from braincell.channel._base import Markov

    return issubclass(cls, Markov)


def _independent_state_names(channel) -> tuple[str, ...]:
    from braincell.channel._base import HH, Markov

    if isinstance(channel, Markov):
        return channel._independent_state_names()
    if isinstance(channel, HH):
        return tuple(gate.name for gate in channel._iter_gates())
    return ()


def _channel_states(cell, cls) -> list:
    """The independent states of the ``cls`` channels of ``cell``."""
    states = []
    for node in cell.nodes().values():
        channel = getattr(node, "_channel", node)
        if isinstance(channel, cls):
            states += [getattr(channel, name) for name in _independent_state_names(channel)]
    return states


def _zero_channel_states(cell, cls) -> int:
    """Set every independent state of the ``cls`` channels of ``cell`` to 0; return how many were set."""
    states = _channel_states(cell, cls)
    for state in states:
        state.value = state.value * 0.0
    return len(states)


def simulate(
    case,
    spec: Mapping[str, Any],
    dt_ms: float,
    *,
    subsolver: str | None = None,
    substeps: int | None = None,
    gate_solver: str | None = None,
    zero_states: bool = False,
) -> dict[str, Any]:
    """Replay every command of ``spec`` at ``dt_ms``, one population each.

    ``subsolver`` and ``substeps`` are the Cell's (Markov channels);
    ``gate_solver`` replaces ``ind_exp_euler`` for the HH gates.
    ``zero_states`` starts the channel's independent states at 0. A channel
    without states (a leak) ignores ``gate_solver``: there is nothing to
    integrate, and ``exp_euler`` rejects an empty state set. Returns
    ``"current"`` (mA/cm^2) and ``"states"`` (by name), each of shape
    ``(n_sample, n_command)`` at ``t = SAMPLE_MS, 2 SAMPLE_MS, ...``.
    """
    import brainunit as u
    import jax

    from braincell._multi_compartment import cell as cell_module
    from braincell.quad import get_integrator

    braincell_runner, _ = _engine()
    table = _clamp.replay_table(spec, dt_ms)
    cell = braincell_runner.build_cell(
        case,
        solver=_clamp.voltage_replay_solver(table),
        pop_size=table.shape[1],
        subsolver=subsolver,
        substeps=substeps,
    )
    if zero_states and not _zero_channel_states(cell, channel_class(case)):
        raise ValueError(f"{case.mapping_spec.braincell.class_name} has no independent state to start at 0.")
    if gate_solver is not None and not _channel_states(cell, channel_class(case)):
        gate_solver = None
    patch = (
        contextlib.nullcontext()
        if gate_solver is None
        else mock.patch.object(cell_module, "ind_exp_euler_step", get_integrator(gate_solver))
    )
    n_steps = table.shape[0] - 1
    with patch:
        result = cell.run(dt=dt_ms * u.ms, duration=n_steps * dt_ms * u.ms)
    jax.block_until_ready(jax.tree.leaves(result.traces))

    every = round(SAMPLE_MS / dt_ms)
    if abs(every * dt_ms - SAMPLE_MS) > 1e-12:
        raise ValueError(f"dt = {dt_ms!r} ms does not divide the sample spacing {SAMPLE_MS} ms.")
    name = case.mapping_spec.braincell.class_name

    def sampled(trace):
        values = trace.to_decimal(u.mA / u.cm**2) if isinstance(trace, u.Quantity) and not trace.is_unitless else u.get_magnitude(trace)
        values = np.asarray(values, dtype=float).reshape(n_steps, -1)
        # Trace row k is the state after step k, at t_{k+1}.
        return values[every - 1 :: every]

    return {
        "current": sampled(result.traces[f"soma(0.5)_{name}_current"]),
        "states": {gate: sampled(result.traces[f"soma(0.5)_{name}_{gate}"]) for gate in case.mapping_spec.braincell.gate_names},
    }


def errors(run: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Return ``e_I`` and ``e_s`` of ``run`` against ``reference``.

    ``e_I`` is ``None`` when the reference current is 0 throughout (a
    channel with ``g_max = 0``), ``e_s`` when the channel has no state. A
    NaN output gives NaN.
    """
    scale = float(np.max(np.abs(reference["current"])))
    current_diff = float(np.max(np.abs(run["current"] - reference["current"])))
    state_diffs = [np.max(np.abs(run["states"][g] - reference["states"][g])) for g in reference["states"]]
    state_diff = float(np.max(state_diffs)) if state_diffs else None
    return {"e_I": None if scale == 0.0 else current_diff / scale, "e_s": state_diff}


def rounded_limit(measured: float) -> float:
    """Return the table limit of a measured error: :data:`LIMIT_FACTOR` times it, rounded up to 1, 2 or 5 times a power of ten."""
    if not (math.isfinite(measured) and measured > 0.0):
        raise ValueError(f"a limit needs a positive, finite measurement, got {measured!r}.")
    target = LIMIT_FACTOR * measured
    exponent = math.floor(math.log10(target))
    for mantissa in (1, 2, 5, 10):
        limit = float(f"{mantissa}e{exponent}")
        if limit >= target:
            return limit
    raise AssertionError("unreachable")  # pragma: no cover


def fitted_order(dts: Iterable[float], values: Iterable[float]) -> float:
    """Least-squares slope of ``log(error)`` over ``log(dt)``; ``nan`` if an error is not positive and finite."""
    dts, values = np.asarray(list(dts), dtype=float), np.asarray(list(values), dtype=float)
    if not np.all(np.isfinite(values) & (values > 0.0)):
        return math.nan
    return float(np.polyfit(np.log(dts), np.log(values), 1)[0])


@dataclass
class ConvergenceReport:
    """Convergence of one configuration on one path; see the module docstring."""

    config: str
    class_name: str
    path: str
    protocol: str
    errors: dict[float, dict[str, float]]
    orders: dict[str, float]
    classification: str | None
    ramp_differences: list[dict[str, float]] = field(default_factory=list)
    ramp_orders: dict[str, float] = field(default_factory=dict)

    def error(self, metric: str, dt_ms: float = REFERENCE_DT_MS) -> float:
        return self.errors[dt_ms][metric]

    @property
    def ramp_ok(self) -> bool:
        """The ramp is finite and each non-zero difference converges at the window's order."""
        for metric, (low, high) in RAMP_ORDER_WINDOWS.items():
            values = [d[metric] for d in self.ramp_differences if d[metric] is not None]
            if not all(math.isfinite(v) for v in values):
                return False
            if values and min(values) > EXACT_LIMIT and not low <= self.ramp_orders[metric] <= high:
                return False
        return True


def classify(errors_by_dt: Mapping[float, Mapping[str, float]]) -> tuple[str | None, dict[str, float]]:
    """Return the classification and the fitted orders of a step or relaxation protocol.

    Non-finite at any ``dt``: not convergent. Every value within
    :data:`EXACT_LIMIT`: exact. Otherwise the order fitted over the three
    smallest ``dt`` must lie in :data:`STEP_ORDER_WINDOW`, and the errors at
    0.025 ms decide between convergent and above the :data:`GUIDE_LINE`.
    Fewer than three ``dt`` leave a finite result unclassified (``None``).
    """
    metrics = [m for m in ("e_I", "e_s") if all(e[m] is not None for e in errors_by_dt.values())]
    values = [errors_by_dt[dt][m] for dt in errors_by_dt for m in metrics]
    if not all(math.isfinite(v) for v in values):
        return NOT_CONVERGENT, {}
    if len(errors_by_dt) < 3:
        return None, {}
    if all(v <= EXACT_LIMIT for v in values):
        return EXACT, {}
    smallest = sorted(errors_by_dt)[:3]
    orders = {m: fitted_order(smallest, [errors_by_dt[dt][m] for dt in smallest]) for m in metrics}
    low, high = STEP_ORDER_WINDOW
    if not all(low <= orders[m] <= high for m in metrics):
        return NOT_CONVERGENT, orders
    at_reference = [errors_by_dt[REFERENCE_DT_MS][m] for m in metrics]
    return (CONVERGENT if max(at_reference) <= GUIDE_LINE else ABOVE_GUIDE_LINE), orders


def check_config(
    config: str,
    *,
    subsolver: str | None = None,
    substeps: int | None = None,
    protocol: str = STEPS,
    dts_ms: Iterable[float] = DTS_MS,
    ramp: bool = True,
    ramp_dts_ms: Iterable[float] = RAMP_DTS_MS,
) -> ConvergenceReport:
    """Measure the ``dt`` convergence of ``config`` on the Cell's path.

    The path is the Cell default (HH gates ``ind_exp_euler``; Markov
    channels ``backward_euler`` x 1) unless ``subsolver`` and ``substeps``
    choose the Markov subsolver. ``protocol`` is :data:`STEPS` or
    :data:`RELAXATION`; ``ramp`` adds the ``vc_ramp`` Richardson differences
    over ``ramp_dts_ms``.
    """
    if protocol not in (STEPS, RELAXATION):
        raise ValueError(f"protocol must be {STEPS!r} or {RELAXATION!r}, got {protocol!r}.")
    case, spec = load_case(config, protocol)
    cls = channel_class(case)
    markov = _is_markov(cls)
    zero = protocol == RELAXATION
    reference = (
        simulate(case, spec, REFERENCE_DT_MS, subsolver=REFERENCE_SOLVER, substeps=1, zero_states=zero)
        if markov
        else simulate(case, spec, REFERENCE_DT_MS, gate_solver=REFERENCE_SOLVER, zero_states=zero)
    )
    errors_by_dt = {}
    for dt in dts_ms:
        run = simulate(case, spec, dt, subsolver=subsolver, substeps=substeps, zero_states=zero)
        errors_by_dt[float(dt)] = errors(run, reference)
        if zero:  # relaxation compares states only: the current at -80 mV may be 0.
            errors_by_dt[float(dt)]["e_I"] = None
    classification, orders = classify(errors_by_dt)
    if markov:
        path = f"{subsolver or 'backward_euler'} x {substeps or 1}"
    else:
        path = "ind_exp_euler"
    report = ConvergenceReport(config, cls.__name__, path, protocol, errors_by_dt, orders, classification)
    if ramp:
        report.ramp_differences, report.ramp_orders = ramp_convergence(
            config, subsolver=subsolver, substeps=substeps, dts_ms=ramp_dts_ms
        )
    return report


def ramp_convergence(
    config: str,
    *,
    subsolver: str | None = None,
    substeps: int | None = None,
    dts_ms: Iterable[float] = RAMP_DTS_MS,
):
    """Return the successive differences over ``dts_ms`` (three, each half the last) and their Richardson orders."""
    dts_ms = tuple(float(dt) for dt in dts_ms)
    if len(dts_ms) != 3 or any(abs(fine * 2.0 - coarse) > 1e-15 for coarse, fine in zip(dts_ms, dts_ms[1:])):
        raise ValueError(f"the ramp needs three time steps, each half the last, got {dts_ms!r}.")
    case, spec = load_case(config, RAMP)
    runs = [simulate(case, spec, dt, subsolver=subsolver, substeps=substeps) for dt in dts_ms]
    differences = [errors(coarse, fine) for coarse, fine in zip(runs, runs[1:])]
    orders = {}
    for metric in RAMP_ORDER_WINDOWS:
        first, second = (d[metric] for d in differences)
        if first is None or second is None:
            continue
        orders[metric] = math.log2(first / second) if first > 0.0 and second > 0.0 else math.nan
    return differences, orders


# ---------------------------------------------------------------------------
# Static scans of a channel class


def channel_kwargs(cls, case=None) -> dict[str, Any]:
    """Return the constructor arguments of ``cls``: the case's parameters, or defaults at 36 degC."""
    import brainunit as u

    braincell_runner, _ = _engine()
    if case is None:
        kwargs, celsius = {}, TEMPERATURE_CELSIUS
    else:
        kwargs = braincell_runner._convert_channel_params_for_braincell(case.mapping_spec, case.channel_params)
        celsius = case.simulation.temperature_celsius
    parameters = inspect.signature(cls.__init__).parameters
    # The same rule as channel_no_conc's ``_inject_temperature_kw``, for classes outside the registry too.
    for candidate in ("temp", "T", "temperature"):
        if candidate in parameters and candidate not in kwargs:
            kwargs[candidate] = u.celsius2kelvin(float(celsius))
            break
    return kwargs


def channel_ion_infos(cls, case=None) -> tuple:
    """Return the ion infos a single-soma Cell passes to ``cls``, in call order.

    As in channel_no_conc: the ion of a case whose ``ion_state`` has
    concentrations is its ``InitNernst`` ion; a case's reversal potential
    overrides that ion's; every other ion family is the Cell's placeholder.
    Chloride has no placeholder (a Cell without a declared chloride ion
    raises ``KeyError``), so it is ``ChlorideFixed`` unless the case sets it.
    """
    import brainunit as u

    from braincell._compute.bindings import _channel_family_slots
    from braincell.ion import ChlorideFixed, build_placeholder_ions
    from braincell.mech import get_registry

    braincell_runner, _ = _engine()
    ions = {**build_placeholder_ions(size=(1,)), "cl": ChlorideFixed(size=(1,))}
    family = None
    if case is not None and case.ion_state is not None:
        ion_name = case.mapping_spec.current_source.ion_name
        family = "ca" if ion_name == "cal" else ion_name
        if case.ion_state.has_concentrations:
            declaration = braincell_runner._build_init_nernst_ion(ion_name=ion_name, case=case)
            ion = get_registry().get("ion", declaration.class_name)(size=(1,), **dict(declaration.params))
            ion.init_state(float(case.simulation.v_init_mV) * u.mV)
            ions[family] = ion
    infos = []
    for slot, _ in _channel_family_slots(cls):
        info = ions[slot].pack_info()
        if slot == family and case.ion_state.has_reversal:
            info = info._replace(E=float(case.ion_state.E_mV) * u.mV)
        infos.append(info)
    return tuple(infos)


def _magnitude(value, unit=None) -> np.ndarray:
    import brainunit as u

    if unit is not None and isinstance(value, u.Quantity) and not value.is_unitless:
        value = value.to_decimal(unit)
    return np.asarray(u.get_magnitude(value), dtype=float)


def markov_generator(channel, V_mV, ions) -> np.ndarray:
    """Return the transition matrices ``Q[v, i, j]`` (rate from state ``j`` to ``i``, 1/ms) at each voltage."""
    import brainunit as u

    V_mV = np.asarray(V_mV, dtype=float)
    names = channel._state_names()
    index = {name: i for i, name in enumerate(names)}
    Q = np.zeros((V_mV.size, len(names), len(names)))
    V = V_mV * u.mV
    for pair in channel._iter_pairs():
        for src, dst, rate in ((pair.src, pair.dst, pair.forward), (pair.dst, pair.src, pair.backward)):
            if rate is None:
                continue
            value = np.broadcast_to(_magnitude(channel._transition_rate(rate, V, *ions)).reshape(-1), V_mV.shape)
            Q[:, index[dst], index[src]] += value
            Q[:, index[src], index[src]] -= value
    return Q


def hh_gate_rates(channel, V_mV, ions) -> dict[str, np.ndarray]:
    """Return ``-d(dx/dt)/dx`` of each HH gate at each voltage (1/ms): ``phi (alpha + beta)`` or ``phi / tau``.

    Taken from :meth:`compute_derivative` at ``x = 0`` and ``x = 1`` with the
    other gates at 0.5, so overridden derivatives are covered as well.
    """
    import brainunit as u

    V_mV = np.asarray(V_mV, dtype=float)
    V = V_mV * u.mV
    gates = [gate.name for gate in channel._iter_gates()]
    rates = {}
    for gate in gates:
        derivatives = []
        for x in (0.0, 1.0):
            for other in gates:
                getattr(channel, other).value = np.full(V_mV.shape, 0.5)
            getattr(channel, gate).value = np.full(V_mV.shape, x)
            channel.compute_derivative(V, *ions)
            derivatives.append(_magnitude(getattr(channel, gate).derivative, u.ms**-1))
        rates[gate] = -(derivatives[1] - derivatives[0])
    return rates


def _channel(cls, n: int, case=None):
    import brainunit as u

    channel = cls(n, **channel_kwargs(cls, case))
    ions = channel_ion_infos(cls, case)
    channel.init_state(np.full(n, -80.0) * u.mV, *ions)
    return channel, ions


@dataclass
class StiffnessReport:
    """Largest state-equation rate of a channel class over :data:`STIFFNESS_GRID_MV`."""

    class_name: str
    kind: str
    rho: np.ndarray
    rho_max: float
    V_at_max_mV: float
    gate: str | None = None

    def stiff(self, dt_ms: float = STIFFNESS_DT_MS) -> bool:
        return self.rho_max * dt_ms > STIFFNESS_LIMIT

    @property
    def decade(self) -> int | None:
        """``floor(log10(rho_max))``; ``None`` for a channel without states."""
        return None if self.rho_max == 0.0 else math.floor(math.log10(self.rho_max))


def stiffness(cls, case=None, V_mV=STIFFNESS_GRID_MV) -> StiffnessReport:
    """Return ``rho_max`` of ``cls``: Markov spectral radius or the largest HH gate rate."""
    from braincell.channel._base import HH

    V_mV = np.asarray(V_mV, dtype=float)
    channel, ions = _channel(cls, V_mV.size, case)
    gate = None
    if _is_markov(cls):
        kind = "Markov"
        rho = np.max(np.abs(np.linalg.eigvals(markov_generator(channel, V_mV, ions))), axis=-1)
    elif isinstance(channel, HH) and channel._iter_gates():
        kind = "HH"
        rates = hh_gate_rates(channel, V_mV, ions)
        gate = max(rates, key=lambda name: np.nanmax(rates[name]))
        rho = rates[gate]
    else:
        kind = "stateless"
        rho = np.zeros_like(V_mV)
    i = int(np.nanargmax(rho)) if np.any(np.isfinite(rho)) else 0
    return StiffnessReport(cls.__name__, kind, rho, float(rho[i]), float(V_mV[i]), gate)


def singularities(cls, case=None, V_mV=SINGULARITY_GRID_MV) -> dict[str, np.ndarray]:
    """Return, for each non-finite quantity, the voltages where it is not finite.

    The channel is reset (to its steady state where it has one) at each
    voltage of the grid; the quantities are each independent state, its
    derivative and the current. An empty result means everything is finite.
    """
    import brainunit as u

    V_mV = np.asarray(V_mV, dtype=float)
    channel, ions = _channel(cls, V_mV.size, case)
    V = V_mV * u.mV
    channel.reset_state(V, *ions)
    channel.compute_derivative(V, *ions)
    quantities = {}
    for name in _independent_state_names(channel):
        state = getattr(channel, name)
        quantities[f"{name}.value"] = state.value
        quantities[f"{name}.derivative"] = state.derivative
    quantities["current"] = channel.current(V, *ions)
    found = {}
    for name, value in quantities.items():
        finite = np.broadcast_to(np.isfinite(_magnitude(value)), V_mV.shape)
        if not finite.all():
            found[name] = V_mV[~finite]
    return found
