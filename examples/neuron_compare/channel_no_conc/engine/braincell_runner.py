"""Run one braincell-side case for channel_no_conc."""



import inspect
from pathlib import Path
from typing import Any

import brainunit as u
import numpy as np

try:
    from .experiment_schema import ChannelNoConcCase
    from .mapping_schema import MappingSpec
    from .metrics import ensure_1d
    from .stimulus import build_braincell_stimulus
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from experiment_schema import ChannelNoConcCase  # type: ignore
    from mapping_schema import MappingSpec  # type: ignore
    from metrics import ensure_1d  # type: ignore
    from stimulus import build_braincell_stimulus  # type: ignore


def run_case(case: ChannelNoConcCase) -> dict[str, Any]:
    import braincell
    from braincell.filter import AllRegion, at
    from braincell.mech import CableProperty, Channel, CurrentProbe, Ion, MechanismProbe, StateProbe, get_registry
    from braincell.morph.branch import Branch
    from braincell.morph.morphology import Morphology

    mapping_spec = case.mapping_spec
    channel_kwargs = _convert_channel_params_for_braincell(mapping_spec, case.channel_params)
    channel_kwargs = _inject_temperature_kw(
        channel_name=mapping_spec.braincell.class_name,
        channel_kwargs=channel_kwargs,
        temperature_celsius=case.simulation.temperature_celsius,
        registry=get_registry(),
    )

    soma = Branch.from_lengths(
        lengths=[float(case.morphology.length_um)] * u.um,
        radii=[float(case.morphology.radius_um), float(case.morphology.radius_um)] * u.um,
        type="soma",
    )
    morpho = Morphology.from_root(soma, name="soma")
    cell = braincell.Cell(
        morpho,
        cv_policy=braincell.CVPerBranch(),
        V_init=float(case.simulation.v_init_mV) * u.mV,
    )

    region = AllRegion()
    cell.paint(
        region,
        CableProperty(
            resting_potential=float(case.simulation.v_init_mV) * u.mV,
            membrane_capacitance=float(case.morphology.cm_uF_cm2) * (u.uF / (u.cm ** 2)),
            axial_resistivity=100.0 * (u.ohm * u.cm),
            temperature=u.celsius2kelvin(float(case.simulation.temperature_celsius)),
        ),
    )
    if case.ion_state is not None and case.ion_state.has_concentrations:
        ion_name = mapping_spec.current_source.ion_name
        if ion_name is None:
            raise ValueError("ion_state requires mapping.current to resolve to ik/ina/ica.")
        cell.paint(
            region,
            _build_init_nernst_ion(
                ion_name=ion_name,
                case=case,
            ),
        )
    cell.paint(region, Channel(mapping_spec.braincell.class_name, **channel_kwargs))

    if case.leak.enabled:
        cell.paint(
            region,
            Channel(
                "IL",
                g_max=float(case.leak.g_S_cm2) * (u.siemens / (u.cm ** 2)),
                E=float(case.leak.e_mV) * u.mV,
            ),
        )

    probe_loc = at("soma", 0.5)
    cell.place(probe_loc, build_braincell_stimulus(case.stimulus))
    cell.place(probe_loc, StateProbe())
    cell.place(
        probe_loc,
        *(
            MechanismProbe(mechanism=mapping_spec.braincell.class_name, field=gate_name)
            for gate_name in mapping_spec.braincell.gate_names
        ),
    )
    cell.place(
        probe_loc,
        _build_current_probe(mapping_spec),
    )

    cell.init_state()
    if case.ion_state is not None:
        ion_name = mapping_spec.current_source.ion_name
        if ion_name is None:
            raise ValueError("ion_state requires mapping.current to resolve to ik/ina/ica.")
        if case.ion_state.has_reversal:
            ion = cell.get_ion(ion_name)
            ion.E = float(case.ion_state.E_mV) * u.mV
    cell.reset_state()

    dt = float(case.simulation.dt_ms) * u.ms
    result = cell.run(
        dt=dt,
        duration=float(case.simulation.duration_ms) * u.ms,
    )

    current_probe_name = _resolve_current_probe_name(mapping_spec)
    return {
        # Cell.run() samples after each integration step while exposing the
        # pre-step time grid. Shift by one dt so the exported traces align
        # with NEURON's post-step samples.
        "time_ms": ensure_1d((result.time + dt).to_decimal(u.ms), name="braincell.time_ms"),
        "voltage_mV": ensure_1d(result.traces["soma(0.5)_v"].to_decimal(u.mV), name="braincell.voltage_mV"),
        "current": {
            "ix": ensure_1d(
                result.traces[current_probe_name].to_decimal(u.mA / (u.cm ** 2)),
                name="braincell.current.ix",
            ),
        },
        "gates": {
            gate_name: ensure_1d(
                np.asarray(result.traces[f"soma(0.5)_{mapping_spec.braincell.class_name}_{gate_name}"]),
                name=f"braincell.gates.{gate_name}",
            )
            for gate_name in mapping_spec.braincell.gate_names
        },
    }


def _build_current_probe(mapping_spec: MappingSpec):
    from braincell.mech import CurrentProbe

    return CurrentProbe(mechanism=mapping_spec.braincell.class_name)


def _build_init_nernst_ion(*, ion_name: str, case: ChannelNoConcCase):
    from braincell.mech import Ion

    class_name = {
        "ca": "CalciumInitNernst",
        "na": "SodiumInitNernst",
        "k": "PotassiumInitNernst",
    }[ion_name]
    return Ion(
        class_name,
        name=ion_name,
        temp=u.celsius2kelvin(float(case.simulation.temperature_celsius)),
        Ci=float(case.ion_state.Ci_mM) * u.mM,
        Co=float(case.ion_state.Co_mM) * u.mM,
    )


def _resolve_current_probe_name(mapping_spec: MappingSpec) -> str:
    return f"soma(0.5)_{mapping_spec.braincell.class_name}_current"


def _convert_channel_params_for_braincell(
    mapping_spec: MappingSpec,
    channel_params: dict[str, Any],
) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for ir_key, value in channel_params.items():
        try:
            mapping = mapping_spec.parameter_map[ir_key]
        except KeyError as exc:
            raise ValueError(
                f"channel_params contains unsupported param for mapping: {ir_key!r}."
            ) from exc
        converted[mapping.braincell] = _convert_braincell_value(value, ir_key=ir_key)
    return converted


def _convert_braincell_value(value: Any, *, ir_key: str) -> Any:
    if ir_key.endswith("_S_cm2"):
        return float(value) * (u.siemens / (u.cm ** 2))
    if ir_key.endswith("_cm_s"):
        return float(value) * (u.cm / u.second)
    if ir_key.endswith("_mV"):
        return float(value) * u.mV
    return value


def _inject_temperature_kw(*, channel_name: str, channel_kwargs: dict[str, Any], temperature_celsius: float, registry) -> dict[str, Any]:
    resolved = dict(channel_kwargs)
    channel_cls = registry.get("channel", channel_name)
    signature = inspect.signature(channel_cls.__init__)
    temperature_kelvin = u.celsius2kelvin(float(temperature_celsius))

    for candidate_name in ("temp", "T", "temperature"):
        if candidate_name in signature.parameters and candidate_name not in resolved:
            resolved[candidate_name] = temperature_kelvin
            break
    return resolved
