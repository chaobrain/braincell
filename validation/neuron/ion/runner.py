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

"""Calcium-pool comparisons against NEURON: configurations, protocols and runner.

A configuration under ``configs/`` names one pool (optionally with one calcium
channel), its ``.mod`` sources under ``data/`` and the NEURON <-> BrainCell
names of its parameters and states. A template under ``templates/`` defines a
protocol on a single soma:

- ``pool_zero_current``: no channel, membrane held at ``v_init``; the pool
  relaxes from its initial (or configured) state.
- ``pool_vc``: voltage clamp. NEURON's ``SEClamp`` follows the command and
  BrainCell replays NEURON's voltage (``_clamp.voltage_replay_solver``), so the
  calcium current drives both pools from the same voltage.
- ``pool_cc``: current clamp with an ``IClamp`` pulse; both sides solve V.

Sample ``i`` of both sides is the state at ``t = i dt``. Metrics (see
``_tolerance``): ``e_Ci`` (``Ci`` over the protocol's largest NEURON ``cai``),
``e_pool`` (each other pool state over its own largest NEURON value, the
largest of these), ``e_I`` and ``e_s`` (channel current and gates, voltage
clamp) and ``e_V`` (current clamp).

All configurations' sources are compiled into one library, loaded once per
process; do not load another suite's channel libraries in the same process.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.neuron import _clamp, _tolerance
from validation.neuron._paths import DATA_ROOT

ION_ROOT = Path(__file__).resolve().parent
CONFIGS_ROOT = ION_ROOT / "configs"
TEMPLATES_ROOT = ION_ROOT / "templates"
PROTOCOLS = ("pool_zero_current", "pool_vc", "pool_cc")

UNITS = {
    "mM": lambda u: u.mM,
    "/mM/ms": lambda u: 1.0 / (u.mM * u.ms),
    "mM*um": lambda u: u.mM * u.um,
    "mol/cm2": lambda u: u.mol / u.cm**2,
    "S/cm2": lambda u: u.siemens / u.cm**2,
    "cm/s": lambda u: u.cm / u.second,
}
"""Units a configuration may give a BrainCell parameter or state (NEURON's units)."""


def _pairs(names: Mapping[str, list[str]], what: str) -> tuple[tuple[str, str], ...]:
    if "common" in names:
        return tuple((name, name) for name in names["common"])
    neuron, braincell = names["neuron"], names["braincell"]
    if len(neuron) != len(braincell):
        raise ValueError(f"{what}: 'neuron' and 'braincell' list different numbers of names.")
    return tuple(zip(neuron, braincell))


@dataclass(frozen=True)
class Parameter:
    """One parameter: NEURON attribute, BrainCell keyword, unit and value."""

    neuron: str | None
    braincell: str | None
    unit: str
    value: float

    def quantity(self):
        import brainunit as u

        return self.value * UNITS[self.unit](u)


@dataclass(frozen=True)
class Channel:
    """The calcium channel that drives the pool under ``pool_vc`` and ``pool_cc``."""

    neuron: str
    braincell: str
    gates: tuple[tuple[str, str], ...]
    params: tuple[Parameter, ...]


@dataclass(frozen=True)
class PoolCase:
    """One configuration under one protocol, fully resolved."""

    config: str
    protocol: str
    mod_files: tuple[Path, ...]
    pool: tuple[str, str]
    pool_params: tuple[Parameter, ...]
    states: tuple[tuple[str, str], ...]
    ci_state: str | None
    initial: Mapping[str, float]
    neuron_initial: Mapping[str, float]
    state_units: Mapping[str, str]
    braincell_solver: str | None
    channel: Channel | None
    simulation: Mapping[str, float]
    morphology: Mapping[str, float]
    voltage_clamp: Mapping[str, Any] | None = None
    current_clamp: Mapping[str, float] | None = None
    protocols: tuple[str, ...] = PROTOCOLS

    @property
    def n_steps(self) -> int:
        return _clamp._steps(self.simulation["duration_ms"], self.simulation["dt_ms"])


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged.get(key, {}), value) if isinstance(value, Mapping) else value
    return merged


def config_names() -> list[str]:
    """Every configuration under ``configs/``, as ``<model>/<name>``."""
    return sorted(str(p.relative_to(CONFIGS_ROOT).with_suffix("")) for p in CONFIGS_ROOT.glob("*/*.json"))


def load_config(config: str) -> dict[str, Any]:
    raw = json.loads((CONFIGS_ROOT / f"{config}.json").read_text())
    unknown = set(raw) - {"meta", "identity", "defaults", "mapping", "templates"}
    if unknown:
        raise ValueError(f"{config}: unknown top-level keys {sorted(unknown)}.")
    return raw


def load_case(config: str, protocol: str) -> PoolCase:
    """Resolve ``config`` under the template ``protocol``.

    Raises ``ValueError`` when the configuration does not list the protocol,
    when a channel protocol has no channel, or when a name or unit is unknown.
    """
    raw = load_config(config)
    protocols = tuple(Path(t).stem for t in raw["templates"])
    if not set(protocols) <= set(PROTOCOLS):
        raise ValueError(f"{config}: unknown templates {sorted(set(protocols) - set(PROTOCOLS))}.")
    if protocol not in protocols:
        raise ValueError(f"{config} does not list the {protocol!r} template.")
    template = json.loads((TEMPLATES_ROOT / f"{protocol}.json").read_text())
    defaults, mapping = raw.get("defaults", {}), raw["mapping"]
    base = _merge(template["base"], {k: defaults[k] for k in ("simulation", "morphology") if k in defaults})

    def params(kind: str, values: Mapping[str, float]) -> tuple[Parameter, ...]:
        names = mapping.get(kind, {})
        unknown = set(values) - set(names)
        if unknown:
            raise ValueError(f"{config}: {kind} has no mapping for {sorted(unknown)}.")
        resolved = []
        for key, value in values.items():
            spec = names[key]
            if spec["unit"] not in UNITS:
                raise ValueError(f"{config}: unknown unit {spec['unit']!r} of {key!r}.")
            resolved.append(Parameter(spec.get("neuron"), spec.get("braincell"), spec["unit"], float(value)))
        return tuple(resolved)

    channel = None
    if "impl_name" in mapping and protocol != "pool_zero_current":
        channel = Channel(
            neuron=mapping["impl_name"]["neuron"],
            braincell=mapping["impl_name"]["braincell"],
            gates=_pairs(mapping.get("gate_names", {"common": []}), f"{config} gate_names"),
            params=params("channel_params", defaults.get("channel_params", {})),
        )
    elif protocol != "pool_zero_current":
        raise ValueError(f"{config}: {protocol!r} needs a channel (mapping.impl_name).")
    pool_states = mapping.get("pool_states", {})
    relaxing = protocol == "pool_zero_current"
    return PoolCase(
        config=config,
        protocol=protocol,
        mod_files=tuple(DATA_ROOT / path for path in raw["identity"]["mod_files"]),
        pool=(mapping["pool_name"]["neuron"], mapping["pool_name"]["braincell"]),
        pool_params=params("pool_params", defaults.get("pool_params", {})),
        states=_pairs({k: v for k, v in pool_states.items() if k != "units"} or {"common": []}, f"{config} pool_states"),
        ci_state=mapping.get("ci_state"),
        initial=dict(defaults.get("initial_states", {})) if relaxing else {},
        neuron_initial=dict(defaults.get("neuron_initial_states", {})) if relaxing else {},
        state_units={"Ci": "mM", **pool_states.get("units", {})},
        braincell_solver=defaults.get("pool_solver"),
        channel=channel,
        simulation=base["simulation"],
        morphology=base["morphology"],
        voltage_clamp=template.get("voltage_clamp"),
        current_clamp=_merge(template["current_clamp"], defaults.get("current_clamp", {}))
        if "current_clamp" in template
        else None,
        protocols=protocols,
    )


# ----------------------------------------------------------------------------- NEURON

_LOADED: list[Path] = []


def mechanism_library() -> Path:
    """Compile every configuration's sources into one library and load it once."""
    from neuron import h

    from validation.neuron._mechanisms import compile_mechanisms
    from validation.neuron.cell._nrnmech import nrnmech_path

    files = sorted({DATA_ROOT / p for name in config_names() for p in load_config(name)["identity"]["mod_files"]})
    library = nrnmech_path(compile_mechanisms("testing", files=files) / "x86_64")
    if not _LOADED:
        h.nrn_load_dll(str(library))
        h.load_file("stdrun.hoc")
        _LOADED.append(library)
    elif _LOADED[0] != library:
        raise RuntimeError(f"another ion library is loaded in this process: {_LOADED[0]}")
    return library


def _set(h, section, mechanism: str, name: str, value: float) -> Callable[[], None]:
    """Set a RANGE or GLOBAL parameter; return a function that restores a GLOBAL."""
    target = getattr(section(0.5), mechanism)
    if hasattr(target, name):
        setattr(target, name, value)
        return lambda: None
    global_name = f"{name}_{mechanism}"
    old = getattr(h, global_name)
    setattr(h, global_name, value)
    return lambda: setattr(h, global_name, old)


def run_neuron(case: PoolCase, *, initialize=None) -> dict[str, np.ndarray]:
    """Run ``case`` in NEURON; return ``v``, ``cai``, pool states, ``ica`` and gates from ``t = 0``.

    ``ica`` is inward-positive, like BrainCell's channel current.
    """
    mechanism_library()
    from neuron import h

    section = h.Section(name="soma")
    restore = []
    try:
        section.L, section.diam, section.nseg = case.morphology["length_um"], case.morphology["diam_um"], 1
        segment = section(0.5)
        section.insert(case.pool[0])
        pool = getattr(segment, case.pool[0])
        restore += [_set(h, section, case.pool[0], p.neuron, p.value) for p in case.pool_params if p.neuron]
        observables = {"cai": lambda: segment.cai}
        observables.update({f"pool:{b}": (lambda n=n: getattr(pool, n)) for n, b in case.states})
        if case.channel is not None:
            section.insert(case.channel.neuron)
            mechanism = getattr(segment, case.channel.neuron)
            restore += [_set(h, section, case.channel.neuron, p.neuron, p.value) for p in case.channel.params]
            observables["ica"] = lambda: -segment.ica
            observables.update({f"gate:{b}": (lambda n=n: getattr(mechanism, n)) for n, b in case.channel.gates})

        def set_initial_states(h, segment, **kwargs):
            (initialize or _clamp.initialize_neuron)(h, segment, **kwargs)
            neuron_names = {b: n for n, b in case.states}
            for name, value in case.initial.items():
                if name == "Ci":
                    segment.cai = value
                    if case.ci_state:
                        setattr(pool, case.ci_state, value)
                else:
                    setattr(pool, neuron_names[name], value)
            for name, value in case.neuron_initial.items():
                setattr(pool, name, value)

        sim = case.simulation
        if case.current_clamp is None:
            spec = case.voltage_clamp
            commands = _clamp.protocol_commands(spec, sim["dt_ms"])
            if len(commands) != 1 or commands[0][0] != sim["v_init_mV"] or commands[0].size != case.n_steps:
                raise ValueError(f"{case.protocol}: one command spanning the simulation from v_init is required.")
            traces = _clamp.run_neuron_seclamp(
                h,
                segment,
                commands[0],
                dt_ms=sim["dt_ms"],
                celsius=sim["temperature_celsius"],
                observables=observables,
                initialize=set_initial_states,
            )
            traces["command_mV"] = commands[0]
        else:
            clamp = h.IClamp(segment)
            spec = case.current_clamp
            clamp.delay, clamp.dur, clamp.amp = spec["delay_ms"], spec["dur_ms"], spec["amp_nA"]
            set_initial_states(h, segment, v_init_mV=sim["v_init_mV"], dt_ms=sim["dt_ms"], celsius=sim["temperature_celsius"])
            observables = {"v": lambda: segment.v, **observables}
            records = {name: [] for name in observables}
            for step in range(case.n_steps + 1):
                if step:
                    h.fadvance()
                h.fcurrent()
                for name, read in observables.items():
                    records[name].append(read())
            traces = {name: np.asarray(values, dtype=float) for name, values in records.items()}
    finally:
        for undo in restore:
            undo()
        h.delete_section(sec=section)
    return traces


# ----------------------------------------------------------------------------- BrainCell


def _with_temperature(cls, kwargs: dict[str, Any], celsius: float) -> dict[str, Any]:
    import brainunit as u

    for name in ("temp", "T", "temperature"):
        if name in inspect.signature(cls.__init__).parameters and name not in kwargs:
            return {**kwargs, name: u.celsius2kelvin(float(celsius))}
    return kwargs


def run_braincell(case: PoolCase, voltages_mV=None, *, pool_solver=None) -> dict[str, np.ndarray]:
    """Run ``case`` in BrainCell; ``voltages_mV`` (``V_0 .. V_N``) replays a clamp.

    ``pool_solver`` (a name or a step function) replaces the configuration's
    pool solver. Returns the same keys as :func:`run_neuron`, ``N`` samples
    from ``t = 0``.
    """
    import brainunit as u

    import braincell as bc
    from braincell.filter import AllRegion, at
    from braincell.mech import CurrentClamp, get_registry

    registry = get_registry()
    sim = case.simulation
    branch = bc.Branch.from_lengths(
        lengths=[case.morphology["length_um"]] * u.um,
        radii=[case.morphology["diam_um"] / 2.0] * 2 * u.um,
        type="soma",
    )
    cell = bc.Cell(
        bc.Morphology.from_root(branch, name="soma"),
        cv_policy=bc.CVPerBranch(),
        V_init=sim["v_init_mV"] * u.mV,
        solver="staggered" if voltages_mV is None else _clamp.voltage_replay_solver(voltages_mV),
    )
    pool_cls = registry.get("ion", case.pool[1])
    kwargs = {p.braincell: p.quantity() for p in case.pool_params if p.braincell}
    accepted = inspect.signature(pool_cls.__init__).parameters
    for name, value in case.initial.items():
        value = value * UNITS[case.state_units.get(name, "mM")](u)
        if f"{name}_initializer" in accepted:
            kwargs[f"{name}_initializer"] = value
        elif "species_initializers" in accepted:
            kwargs.setdefault("species_initializers", {})[name] = value
        else:
            raise ValueError(f"{case.config}: {pool_cls.__name__} takes no initial {name!r}.")
    solver = pool_solver if pool_solver is not None else case.braincell_solver
    if solver is not None:
        kwargs.update(solver=solver, substeps=1)
    cell.paint(AllRegion(), bc.mech.Ion(case.pool[1], name="ca", **_with_temperature(pool_cls, kwargs, sim["temperature_celsius"])))
    cell.record("v", bc.observe.state("v"))
    cell.record("cai", bc.observe.ion(name="ca").state("Ci"))
    for _, name in case.states:
        cell.record(f"pool:{name}", bc.observe.ion(name="ca").state(name))
    if case.channel is not None:
        channel_cls = registry.get("channel", case.channel.braincell)
        kwargs = _with_temperature(channel_cls, {p.braincell: p.quantity() for p in case.channel.params}, sim["temperature_celsius"])
        cell.paint(AllRegion(), bc.mech.Channel(case.channel.braincell, name="chan", **kwargs))
        cell.record("ica", bc.observe.channel(name="chan").current())
        for _, name in case.channel.gates:
            cell.record(f"gate:{name}", bc.observe.channel(name="chan").state(name))
    if case.current_clamp is not None:
        spec = case.current_clamp
        cell.place(
            at("soma", 0.5),
            CurrentClamp(delay=spec["delay_ms"] * u.ms, durations=spec["dur_ms"] * u.ms, amplitudes=spec["amp_nA"] * u.nA),
        )
    cell.init_state()
    cell.reset_state()
    result = cell.run(dt=sim["dt_ms"] * u.ms, duration=case.n_steps * sim["dt_ms"] * u.ms)
    traces = {}
    for name, block in result.samples.items():
        values = block.values
        if name == "ica":
            values = values.to_decimal(u.mA / u.cm**2)
        elif name == "v":
            values = values.to_decimal(u.mV)
        elif name == "cai":
            values = values.to_decimal(u.mM)
        elif name.startswith("pool:"):
            values = values.to_decimal(UNITS[case.state_units.get(name[5:], "mM")](u))
        traces[name] = np.asarray(values, dtype=float).reshape(case.n_steps, -1)[:, 0]
    return traces


# ----------------------------------------------------------------------------- comparison


def compare_protocol(config: str, protocol: str, *, pool_solver=None, initialize=None) -> dict[str, Any]:
    """Run ``config`` under ``protocol`` on both sides and return its metrics.

    ``pool_solver`` replaces BrainCell's pool solver; ``initialize`` replaces
    NEURON's initializer (tests use both). Returns every metric of the
    protocol (``_tolerance.PROTOCOL_METRICS``), ``clamp_error_mV`` under a
    voltage command and both sides' traces.
    """
    case = load_case(config, protocol)
    neuron = run_neuron(case, initialize=initialize)
    voltages = neuron["v"] if case.current_clamp is None else None
    braincell = run_braincell(case, voltages, pool_solver=pool_solver)
    n = case.n_steps
    ref = {name: trace[:n] for name, trace in neuron.items()}
    result = {"case": case, "neuron": neuron, "braincell": braincell}
    if case.current_clamp is None:
        result["clamp_error_mV"] = float(np.max(np.abs(neuron["v"][1:] - neuron["command_mV"])))
    result["e_Ci"] = _tolerance.relative_current_error([(braincell["cai"], ref["cai"])])
    pool = [_tolerance.relative_current_error([(braincell[k], ref[k])]) for k in braincell if k.startswith("pool:")]
    result["e_pool"] = float(np.max(pool)) if pool else 0.0
    if case.protocol == "pool_vc":
        result["e_I"] = _tolerance.relative_current_error([(braincell["ica"], ref["ica"])])
        gates = [(braincell[k], ref[k]) for k in braincell if k.startswith("gate:")]
        result["e_s"] = _tolerance.max_abs_error(gates) if gates else 0.0
    if case.protocol == "pool_cc":
        result["e_V"] = _tolerance.max_abs_error([(braincell["v"], ref["v"])])
    return result


def neuron_sparse_solver(converge: float = 1e-6, max_iterations: int = 21):
    """Return a pool solver that iterates implicit Euler the way NEURON's ``sparse`` does.

    NEURON solves a ``KINETIC`` block with Newton iterations from the previous
    state, stopping once the summed absolute correction of one iteration is at
    most ``converge`` (``CONVERGE = 1e-6``) and clamping negative states to
    zero after each. BrainCell's ``backward_euler`` takes the first of these
    iterations only. Tests use this solver to show that this is the only cause
    of a kinetic pool's difference from NEURON.
    """
    import jax
    import jax.numpy as jnp

    from braincell.quad._util import apply_standard_solver_step, environ_time, linearize_flat

    def newton(f, y0, t, dt, args):
        n = y0.shape[-1]

        def body(carry):
            y, count, _, _ = carry
            dt_value, A, df, aux = linearize_flat(f, y, t, dt, args)
            residual = dt_value * df - (y - y0).reshape(-1, n)
            delta = jax.scipy.linalg.solve(jnp.eye(n, dtype=y.dtype) - dt_value * A, residual[..., None])
            delta = delta.reshape(y.shape)
            return jnp.maximum(y + delta, 0.0), count + 1, jnp.sum(jnp.abs(delta)), aux

        _, _, _, aux = linearize_flat(f, y0, t, dt, args)
        start = (y0, 0, jnp.asarray(np.inf, y0.dtype), aux)
        y, _, _, aux = jax.lax.while_loop(lambda c: (c[2] > converge) & (c[1] <= max_iterations), body, start)
        return y, aux

    def step(target, *args, excluded_paths=()):
        t, dt = environ_time(target)
        apply_standard_solver_step(newton, target, t, dt, *args, merging="stack", excluded_paths=excluded_paths)

    return step
