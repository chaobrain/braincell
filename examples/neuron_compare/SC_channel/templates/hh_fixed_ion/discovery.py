"""Auto-discovery helpers for NEURON and braincell channel metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import inspect

import brainstate
import brainunit as u


_ION_NAMES = ("na", "k", "ca")
_CURRENT_KEYS = {"i", "ina", "ik", "ica"}
_KNOWN_PARAM_HINTS = {
    "gbar",
    "g",
    "gmax",
    "g_max",
    "e",
    "ena",
    "ek",
    "eca",
    "v12",
    "vhalf",
    "q",
    "ra",
    "rb",
    "phi",
    "temp",
    "t",
    "tref",
    "celsius",
    "v_sh",
    "vshift",
    "shift",
}


@dataclass(frozen=True)
class NeuronChannelMetadata:
    mechanism_name: str
    gate_names: tuple[str, ...]
    ion_type: str | None
    current_field: str
    current_owner: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class BraincellChannelMetadata:
    channel_name: str
    gate_names: tuple[str, ...]
    ion_type: str | None
    channel_kind: str
    constructor_param_names: tuple[str, ...]


def discover_neuron_channel_metadata(mod_dir: str | Path, mechanism_name: str) -> NeuronChannelMetadata:
    from neuron import h, load_mechanisms

    load_mechanisms(str(Path(mod_dir).resolve()))
    h.load_file("stdrun.hoc")
    sec = h.Section(name="discover_neuron")
    try:
        sec.insert(str(mechanism_name))
        density_mech = sec.psection()["density_mechs"][str(mechanism_name)]
        ions = sec.psection()["ions"]
        ion_candidates = [name for name in ions.keys() if name in _ION_NAMES]
        ion_type = ion_candidates[0] if len(ion_candidates) == 1 else None

        if "i" in density_mech:
            current_field = "i"
            current_owner = "mechanism"
        elif ion_type is not None:
            current_field = {"na": "ina", "k": "ik", "ca": "ica"}[ion_type]
            current_owner = "segment"
        else:
            raise ValueError(
                f"Could not determine current field for mechanism {mechanism_name!r}."
            )

        keys = set(density_mech.keys())
        gate_names = tuple(sorted(name for name in keys if name not in _CURRENT_KEYS and name not in _KNOWN_PARAM_HINTS))
        parameter_names = tuple(sorted(name for name in keys if name not in _CURRENT_KEYS and name not in gate_names))
        return NeuronChannelMetadata(
            mechanism_name=str(mechanism_name),
            gate_names=gate_names,
            ion_type=ion_type,
            current_field=current_field,
            current_owner=current_owner,
            parameter_names=parameter_names,
        )
    finally:
        h.delete_section(sec=sec)


def discover_braincell_channel_metadata(channel_name: str) -> BraincellChannelMetadata:
    import braincell
    from braincell.quad import DiffEqState

    if not hasattr(braincell.channel, channel_name):
        raise KeyError(f"braincell.channel has no channel named {channel_name!r}.")
    cls = getattr(braincell.channel, channel_name)

    ion_type = _infer_braincell_ion_type(getattr(cls, "root_type", None))
    gate_names = _discover_braincell_gate_names(cls, ion_type=ion_type)
    channel_kind = _infer_braincell_channel_kind(cls, gate_names=gate_names)
    signature = inspect.signature(cls.__init__)
    constructor_param_names = tuple(
        name for name in signature.parameters
        if name not in {"self", "size", "name"}
    )
    return BraincellChannelMetadata(
        channel_name=channel_name,
        gate_names=gate_names,
        ion_type=ion_type,
        channel_kind=channel_kind,
        constructor_param_names=constructor_param_names,
    )


def _infer_braincell_ion_type(root_type) -> str | None:
    import braincell

    try:
        if isinstance(root_type, type):
            if issubclass(root_type, braincell.ion.Sodium):
                return "na"
            if issubclass(root_type, braincell.ion.Potassium):
                return "k"
            if issubclass(root_type, braincell.ion.Calcium):
                return "ca"
    except TypeError:
        return None
    return None


def _infer_braincell_channel_kind(cls, *, gate_names: tuple[str, ...]) -> str:
    if getattr(cls, "gates", ()):
        return "hh"
    if getattr(cls, "pairs", ()):
        return "markov"
    if len(gate_names) > 0:
        return "hh"
    return "unknown"


def _discover_braincell_gate_names(cls, *, ion_type: str | None) -> tuple[str, ...]:
    gates = getattr(cls, "gates", ())
    if gates:
        names = []
        for gate in gates:
            if hasattr(gate, "name"):
                names.append(gate.name)
            else:
                names.append(str(gate[0]))
        return tuple(names)

    if ion_type is None:
        return ()

    import braincell
    from braincell.quad import DiffEqState

    channel = cls(size=1)
    v = u.math.asarray([-65.0]) * u.mV
    ion_group = {
        "na": braincell.ion.SodiumFixed(1),
        "k": braincell.ion.PotassiumFixed(1),
        "ca": braincell.ion.CalciumFixed(1),
    }[ion_type]
    channel.init_state(v, ion_group.pack_info())
    names = [
        name
        for name, value in vars(channel).items()
        if isinstance(value, DiffEqState)
    ]
    return tuple(sorted(names))
