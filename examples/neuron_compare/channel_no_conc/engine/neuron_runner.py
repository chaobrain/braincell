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


def run_case(case: ChannelNoConcCase) -> dict[str, Any]:
    mapping_spec = case.mapping_spec
    h, section, segment = _build_neuron_model(case, mapping_spec=mapping_spec)

    h.cvode_active(0)
    h.dt = float(case.simulation.dt_ms)
    h.steps_per_ms = 1.0 / float(case.simulation.dt_ms)
    h.celsius = float(case.simulation.temperature_celsius)
    h.v_init = float(case.simulation.v_init_mV)
    h.finitialize(h.v_init)
    sample_times_ms = np.arange(0.0, float(case.simulation.duration_ms), float(case.simulation.dt_ms), dtype=float)
    if sample_times_ms.size == 0:
        raise ValueError("simulation.duration_ms must produce at least one sample.")

    stim = h.IClamp(segment)
    stim.delay = 0.0
    stim.dur = 1e9
    stim.amp = 0.0
    amp_values = np.asarray(
        [current_at_ms(case.stimulus, float(t_ms)) for t_ms in sample_times_ms],
        dtype=float,
    )
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
        "gates": {
            gate_name: ensure_1d(gate_vec, name=f"neuron.gates.{gate_name}")[1:]
            for gate_name, gate_vec in gate_vecs.items()
        },
    }


def _build_neuron_model(case: ChannelNoConcCase, *, mapping_spec: MappingSpec):
    from neuron import h, load_mechanisms

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
    if case.ion_state is not None:
        ion_name = mapping_spec.current_source.ion_name
        if ion_name is None:
            raise ValueError("ion_state requires mapping.current to resolve to ik/ina/ica.")
        setattr(
            soma,
            _resolve_neuron_erev_field(ion_name),
            float(case.ion_state.E_mV),
        )

    for seg in soma:
        mech_obj = getattr(seg, mapping_spec.neuron.mechanism_name)
        for key, value in case.channel_params.items():
            try:
                attr_name = mapping_spec.parameter_map[key].neuron
            except KeyError as exc:
                raise ValueError(
                    f"channel_params contains unsupported param for mapping: {key!r}."
                ) from exc
            target = _resolve_neuron_parameter_target(soma=soma, mech_obj=mech_obj, attr_name=attr_name)
            setattr(target, attr_name, float(value))

    return h, soma, soma(0.5)


def _resolve_neuron_erev_field(ion_name: str) -> str:
    return {
        "na": "ena",
        "k": "ek",
        "ca": "eca",
    }[ion_name]


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

