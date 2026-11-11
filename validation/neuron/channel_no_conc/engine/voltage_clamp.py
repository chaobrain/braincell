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

"""Voltage-clamp comparison of one channel_no_conc configuration.

NEURON clamps the single soma with ``SEClamp``; BrainCell rebuilds the same
Cell through :func:`braincell_runner.run_case` with a solver that replays
NEURON's voltage. The comparison metrics are ``e_I`` (current, normalized by
the protocol's largest NEURON current) and ``e_s`` (every compared state).
"""

from collections.abc import Callable, Collection
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.neuron import _clamp, _tolerance

from .braincell_runner import run_case as run_braincell_case
from .experiment_schema import ChannelNoConcCase, build_sweep_config, expand_cases, load_model_config, load_scan_template
from . import neuron_runner


def load_protocol(config_path: str | Path, template_path: str | Path) -> tuple[ChannelNoConcCase, list[np.ndarray]]:
    """Return the case and commands of a voltage-clamp template.

    The template need not be listed in the configuration's ``templates``. It
    defines one case (no sweep) whose ``duration_ms`` and ``v_init_mV`` match
    its commands; otherwise ``ValueError``.
    """
    template_path = Path(template_path).resolve()
    spec = json.loads(template_path.read_text()).get("voltage_clamp")
    if spec is None:
        raise ValueError(f"{template_path.name} has no voltage_clamp block.")
    cases = expand_cases(build_sweep_config(load_model_config(config_path), load_scan_template(template_path)))
    if len(cases) != 1:
        raise ValueError(f"{template_path.name}: a voltage-clamp template defines exactly one case.")
    case = ChannelNoConcCase.from_dict(cases[0])
    commands = _clamp.protocol_commands(spec, case.simulation.dt_ms)
    n_steps = int(round(case.simulation.duration_ms / case.simulation.dt_ms))
    for command in commands:
        if command.size != n_steps or command[0] != case.simulation.v_init_mV:
            raise ValueError(f"{template_path.name}: commands must span simulation.duration_ms from v_init_mV.")
    return case, commands


def run_neuron_command(case: ChannelNoConcCase, command: np.ndarray, *, initialize=None) -> dict[str, Any]:
    """Clamp NEURON's soma to ``command``; return ``v``, ``ix`` and gates from ``t = 0``."""
    mapping = case.mapping_spec
    h, section, segment = neuron_runner._build_neuron_model(case, mapping_spec=mapping)
    try:
        mechanism = getattr(segment, mapping.neuron.mechanism_name)
        current = neuron_runner._resolve_neuron_current_ref(
            segment,
            mech_obj=mechanism,
            mechanism_name=mapping.neuron.mechanism_name,
            current_var=mapping.current_source.neuron_current_var,
        )
        observables = {"ix": lambda: -current[0]}
        for gate in mapping.neuron.gate_names:
            observables[f"gate:{gate}"] = lambda gate=gate: getattr(mechanism, gate)
        concentrations = {}
        ion_name = mapping.current_source.ion_name
        if case.ion_state is not None and case.ion_state.has_concentrations:
            ci, co = neuron_runner._resolve_neuron_concentration_fields(ion_name)
            concentrations = {ci: case.ion_state.Ci_mM, co: case.ion_state.Co_mM}
            observables[ci] = lambda: getattr(segment, ci)
        traces = _clamp.run_neuron_seclamp(
            h,
            segment,
            command,
            dt_ms=case.simulation.dt_ms,
            celsius=case.simulation.temperature_celsius,
            concentrations=concentrations,
            observables=observables,
            initialize=initialize,
        )
    finally:
        h.delete_section(sec=section)
    return {
        "v": traces.pop("v"),
        "ix": traces.pop("ix"),
        "gates": {name.removeprefix("gate:"): traces.pop(name) for name in list(traces) if name.startswith("gate:")},
        "concentrations": traces,
    }


def compare_protocol(
    config_path: str | Path,
    template_path: str | Path,
    *,
    initialize=None,
    replay_voltages: Callable[[np.ndarray], np.ndarray] | None = None,
    skip_commands: Collection[int] = (),
) -> dict[str, Any]:
    """Run every command of a voltage-clamp template on both sides.

    ``initialize`` replaces the NEURON initializer and ``replay_voltages``
    maps NEURON's voltage trace to the table BrainCell replays (tests use
    both to check the initialization order and the step alignment).
    ``skip_commands`` lists command indices left out, for a reference that is
    not finite there. Returns ``e_I``, ``e_s`` (``None`` for a channel without
    gates), ``clamp_error_mV`` (largest ``|v - command|``), the ``case`` and the
    per-command traces of both sides.
    """
    case, commands = load_protocol(config_path, template_path)
    gate_map = case.mapping_spec.gate_map
    currents, states, results = [], [], []
    clamp_error = 0.0
    for index, command in enumerate(commands):
        if index in skip_commands:
            continue
        neuron = run_neuron_command(case, command, initialize=initialize)
        clamp_error = max(clamp_error, float(np.max(np.abs(neuron["v"][1:] - command))))
        voltages = neuron["v"] if replay_voltages is None else replay_voltages(neuron["v"])
        braincell = run_braincell_case(case, solver=_clamp.voltage_replay_solver(voltages))
        currents.append((braincell["current"]["ix"], neuron["ix"][1:]))
        states.extend((braincell["gates"][g.braincell], neuron["gates"][g.neuron][1:]) for g in gate_map)
        results.append({"command_mV": command, "neuron": neuron, "braincell": braincell})
    return {
        "case": case,
        "e_I": _tolerance.relative_current_error(currents),
        "e_s": _tolerance.max_abs_error(states) if states else None,
        "clamp_error_mV": clamp_error,
        "commands": results,
    }
