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

"""Run one NEURON-side case for channel_no_conc."""



from pathlib import Path
from typing import Any

import numpy as np

try:
    from .experiment_schema import ChannelNoConcCase
    from .mapping_schema import MappingSpec
    from .metrics import ensure_1d
    from .stimulus import current_at_ms
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from experiment_schema import ChannelNoConcCase  # type: ignore
    from mapping_schema import MappingSpec  # type: ignore
    from metrics import ensure_1d  # type: ignore
    from stimulus import current_at_ms  # type: ignore


_LOADED_MECHANISM_DIRS: set[str] = set()


def stimulus_samples(case: ChannelNoConcCase) -> tuple[np.ndarray, np.ndarray]:
    """Return the step start times and the clamp amplitude played at each.

    ``Vector.play`` without interpolation holds ``amp`` constant over each
    fixed step ``[t, t + dt)``. BrainCell samples every current clamp once
    per step at the step midpoint, so the amplitude for the step starting at
    ``t`` is ``I(t + dt / 2)``. Sampling at ``t`` instead shifts a sine by
    half a step and gave a 0.016 mV voltage error (MECH-15).

    Parameters
    ----------
    case : ChannelNoConcCase
        Case whose ``simulation`` sets the step grid and whose ``stimulus``
        is sampled.

    Returns
    -------
    tuple of numpy.ndarray
        ``(times_ms, amplitudes_nA)``; ``times_ms`` are the step starts
        ``0, dt, 2 dt, ...`` below ``duration_ms``.

    Raises
    ------
    ValueError
        If ``duration_ms`` yields no step.
    """
    dt_ms = float(case.simulation.dt_ms)
    times_ms = np.arange(0.0, float(case.simulation.duration_ms), dt_ms, dtype=float)
    if times_ms.size == 0:
        raise ValueError("simulation.duration_ms must produce at least one sample.")
    amplitudes = np.asarray(
        [current_at_ms(case.stimulus, float(t_ms) + 0.5 * dt_ms) for t_ms in times_ms],
        dtype=float,
    )
    return times_ms, amplitudes


def run_case(case: ChannelNoConcCase) -> dict[str, Any]:
    mapping_spec = case.mapping_spec
    h, section, segment = _build_neuron_model(case, mapping_spec=mapping_spec)

    initialize_neuron(h=h, section=section, segment=segment, case=case)
    sample_times_ms, amp_values = stimulus_samples(case)

    stim = h.IClamp(segment)
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0
    amp_time_vec = h.Vector(sample_times_ms)
    amp_value_vec = h.Vector(amp_values)
    amp_value_vec.play(stim._ref_amp, amp_time_vec)

    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(segment._ref_v)
    mech_obj = getattr(segment, mapping_spec.neuron.mechanism_name)
    current_var = _resolve_neuron_current_var(mapping_spec)
    current_vec = h.Vector().record(
        _resolve_neuron_current_ref(
            segment,
            mech_obj=mech_obj,
            mechanism_name=mapping_spec.neuron.mechanism_name,
            current_var=current_var,
        )
    )
    gate_vecs = {
        gate_name: h.Vector().record(getattr(mech_obj, f"_ref_{gate_name}"))
        for gate_name in mapping_spec.neuron.gate_names
    }
    ion_state_vecs = _build_neuron_ion_state_recorders(segment=segment, mapping_spec=mapping_spec, h=h)

    try:
        h.tstop = float(case.simulation.duration_ms)
        h.run()
    finally:
        amp_value_vec.play_remove()
        h.delete_section(sec=section)

    return {
        "time_ms": ensure_1d(t_vec, name="neuron.time_ms")[1:],
        "voltage_mV": ensure_1d(v_vec, name="neuron.voltage_mV")[1:],
        "current": {"ix": -ensure_1d(current_vec, name="neuron.current.ix")[1:]},
        "ion_state": {
            key: ensure_1d(vec, name=f"neuron.ion_state.{key}")[1:]
            for key, vec in ion_state_vecs.items()
        },
        "gates": {
            gate_name: ensure_1d(gate_vec, name=f"neuron.gates.{gate_name}")[1:]
            for gate_name, gate_vec in gate_vecs.items()
        },
    }


def initialize_neuron(*, h, section, segment, case: ChannelNoConcCase) -> None:
    """Initialize the fixed-step run of a model built by :func:`_build_neuron_model`.

    Sets ``dt``, ``celsius`` and ``v_init``, calls ``finitialize`` and then
    writes the case's ion concentrations.
    """
    h.cvode_active(0)
    h.dt = float(case.simulation.dt_ms)
    h.steps_per_ms = 1.0 / float(case.simulation.dt_ms)
    h.celsius = float(case.simulation.temperature_celsius)
    h.v_init = float(case.simulation.v_init_mV)
    h.finitialize(h.v_init)
    _initialize_neuron_ion_state(h=h, section=section, segment=segment, case=case, mapping_spec=case.mapping_spec)


def _build_neuron_model(case: ChannelNoConcCase, *, mapping_spec: MappingSpec):
    from neuron import h, load_mechanisms

    if case.leak.enabled and mapping_spec.neuron.mechanism_name == "pas":
        # NEURON has one pas per section: the channel's parameters would overwrite the leak's.
        raise ValueError("A template leak cannot be combined with the pas channel under comparison.")
    mechanism_dir = str(Path(case.mod_dir).resolve())
    if mechanism_dir not in _LOADED_MECHANISM_DIRS:
        load_mechanisms(mechanism_dir)
        _LOADED_MECHANISM_DIRS.add(mechanism_dir)
    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = float(case.morphology.length_um)
    soma.diam = float(2.0 * case.morphology.radius_um)
    soma.nseg = 1
    soma.cm = float(case.morphology.cm_uF_cm2)

    if case.leak.enabled:
        soma.insert("pas")
        for seg in soma:
            seg.pas.g = float(case.leak.g_S_cm2)
            seg.pas.e = float(case.leak.e_mV)

    soma.insert(mapping_spec.neuron.mechanism_name)
    disable_rate_table(h, mapping_spec.neuron.mechanism_name)
    if case.ion_state is not None:
        ion_name = mapping_spec.current_source.ion_name
        if ion_name is None:
            raise ValueError("ion_state requires mapping.current to resolve to ik/ina/ica.")
        if case.ion_state.has_reversal:
            setattr(
                soma,
                _resolve_neuron_erev_field(ion_name),
                float(case.ion_state.E_mV),
            )
        elif ion_name in _NERNST_STYLE_IONS and case.ion_state.has_concentrations:
            _use_nernst_reversal(h, soma, ion_name=ion_name, Ci_mM=case.ion_state.Ci_mM, Co_mM=case.ion_state.Co_mM)

    for seg in soma:
        mech_obj = getattr(seg, mapping_spec.neuron.mechanism_name)
        for key, value in case.channel_params.items():
            try:
                attr_name = mapping_spec.parameter_map[key].neuron
            except KeyError as exc:
                raise ValueError(
                    f"channel_params contains unsupported param for mapping: {key!r}."
                ) from exc
            if attr_name is None:
                continue
            global_name = f"{attr_name}_{mapping_spec.neuron.mechanism_name}"
            if not hasattr(mech_obj, attr_name) and not hasattr(soma, attr_name) and hasattr(h, global_name):
                # A GLOBAL (Nav1.1's gateCurrent) is shared by every instance and
                # keeps its value for later models in this process; a configuration
                # that varies one sets it in its defaults.
                setattr(h, global_name, float(value))
                continue
            target = _resolve_neuron_parameter_target(soma=soma, mech_obj=mech_obj, attr_name=attr_name)
            setattr(target, attr_name, float(value))

    return h, soma, soma(0.5)


def disable_rate_table(h, mechanism_name: str) -> bool:
    """Make a mechanism with a ``TABLE`` statement evaluate its rates exactly.

    NMODL ``TABLE`` interpolates the rates on a voltage grid (NEURON's ``hh``:
    200 intervals from -100 to 100 mV), which BrainCell does not; setting
    ``usetable_<mechanism> = 0`` turns the table off. Returns whether the
    mechanism has a table.
    """
    name = f"usetable_{mechanism_name}"
    if not hasattr(h, name):
        return False
    setattr(h, name, 0)
    return True


def _resolve_neuron_erev_field(ion_name: str) -> str:
    return {
        "na": "ena",
        "k": "ek",
        "ca": "eca",
        "cal": "ecal",
        "cl": "ecl",
    }[ion_name]


#: Ions whose mechanisms only READ the reversal potential: NEURON keeps it a
#: parameter (0 mV for chloride, with cli = clo = 1 mM) unless ion_style says
#: to compute it from the concentrations.
_NERNST_STYLE_IONS = frozenset({"cl"})


def _use_nernst_reversal(h, section, *, ion_name: str, Ci_mM: float, Co_mM: float) -> None:
    """Make NEURON compute the reversal potential of ``ion_name`` from fixed concentrations.

    ``ion_style(c_style=1, e_style=2, einit=1, eadvance=1, cinit=1)``: the
    concentrations are parameters, initialized at ``finitialize`` from the
    global ``<ion>i0_<ion>_ion`` / ``<ion>o0_<ion>_ion``, which are set here,
    and the Nernst potential is computed at ``finitialize`` and every step.
    """
    h.ion_style(f"{ion_name}_ion", 1, 2, 1, 1, 1, sec=section)
    setattr(h, f"{ion_name}i0_{ion_name}_ion", float(Ci_mM))
    setattr(h, f"{ion_name}o0_{ion_name}_ion", float(Co_mM))


def _initialize_neuron_ion_state(*, h, section, segment, case: ChannelNoConcCase, mapping_spec: MappingSpec) -> None:
    if case.ion_state is None:
        return
    ion_name = mapping_spec.current_source.ion_name
    if ion_name is None:
        raise ValueError("ion_state requires mapping.current to resolve to ik/ina/ica.")
    if case.ion_state.has_concentrations:
        ci_attr, co_attr = _resolve_neuron_concentration_fields(ion_name)
        setattr(segment, ci_attr, float(case.ion_state.Ci_mM))
        setattr(segment, co_attr, float(case.ion_state.Co_mM))
        h.frecord_init()


def _resolve_neuron_concentration_fields(ion_name: str) -> tuple[str, str]:
    return {
        "na": ("nai", "nao"),
        "k": ("ki", "ko"),
        "ca": ("cai", "cao"),
        "cal": ("cali", "calo"),
        "cl": ("cli", "clo"),
    }[ion_name]


def _build_neuron_ion_state_recorders(*, segment, mapping_spec: MappingSpec, h) -> dict[str, Any]:
    ion_name = mapping_spec.current_source.ion_name
    if ion_name not in {"ca", "cal"}:
        return {}
    ci_attr, co_attr = _resolve_neuron_concentration_fields(ion_name)
    e_attr = _resolve_neuron_erev_field(ion_name)
    field_names = {
        "ci_mM": ci_attr,
        "co_mM": co_attr,
        "eca_mV": e_attr,
    }
    recorders: dict[str, Any] = {}
    for key, attr_name in field_names.items():
        recorders[key] = h.Vector().record(getattr(segment, f"_ref_{attr_name}"))
    return recorders


def _resolve_neuron_current_var(mapping_spec: MappingSpec) -> str:
    current_var = mapping_spec.current_source.neuron_current_var
    return current_var


def _resolve_neuron_parameter_target(*, soma, mech_obj, attr_name: str):
    on_mechanism = hasattr(mech_obj, attr_name)
    on_section = hasattr(soma, attr_name)
    if on_mechanism and on_section:
        raise ValueError(
            f"Ambiguous NEURON parameter target for {attr_name!r}; "
            "it exists on both the mechanism and the section."
        )
    if on_mechanism:
        return mech_obj
    if on_section:
        return soma
    raise ValueError(
        f"Could not resolve NEURON parameter target for {attr_name!r}; "
        "it exists on neither the mechanism nor the section."
    )


def _resolve_neuron_current_ref(segment, *, mech_obj=None, mechanism_name: str | None = None, current_var: str):
    candidate_names = [f"_ref_{current_var}"]
    if mechanism_name is not None:
        candidate_names.append(f"_ref_{current_var}_{mechanism_name}")

    for ref_name in candidate_names:
        try:
            return getattr(segment, ref_name)
        except AttributeError:
            pass

    if mech_obj is not None:
        mech_ref_name = f"_ref_{current_var}"
        try:
            return getattr(mech_obj, mech_ref_name)
        except AttributeError:
            pass

    raise AttributeError(
        f"Could not resolve NEURON current ref for current var {current_var!r}; "
        f"tried segment refs {candidate_names!r}"
        + (" and mechanism ref " + f"'_ref_{current_var}'." if mech_obj is not None else ".")
    )
