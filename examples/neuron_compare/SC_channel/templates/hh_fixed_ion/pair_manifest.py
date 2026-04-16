"""Minimal pair manifest for the single-compartment HH + fixed-ion template."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairEntry:
    pair_id: str
    description: str
    neuron_mechanism_name: str
    braincell_channel_name: str
    ion_type_override: str | None = None
    gate_name_map: dict[str, str] | None = None
    notes: tuple[str, ...] = ()


PAIR_MANIFEST: dict[str, PairEntry] = {
    "kv_test": PairEntry(
        pair_id="kv_test",
        description="Minimal Kv HH-style pair used for the first template preview.",
        neuron_mechanism_name="Kv",
        braincell_channel_name="IK_Kv_test",
        notes=(
            "This pair is the first smoke-test target for HH + fixed-ion.",
        ),
    ),
}


def get_pair_entry(pair_id: str) -> PairEntry:
    try:
        return PAIR_MANIFEST[str(pair_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown pair_id {pair_id!r}.") from exc
