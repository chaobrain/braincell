"""Local morphology import helpers for multi-compartment cable comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from .case_schema import MorphologySpec
except ImportError:  # pragma: no cover
    import sys

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from case_schema import MorphologySpec  # type: ignore


def load_braincell_morphology(case_or_spec: Any):
    from braincell import Morphology

    spec = _resolve_morphology_spec(case_or_spec)
    if spec.kind == "swc":
        return Morphology.from_swc(spec.path, mode="neuron")
    if spec.kind == "asc":
        return Morphology.from_asc(spec.path)
    raise NotImplementedError("morphology.kind='neuroml2' is reserved but not implemented yet.")


def load_neuron_sections(case_or_spec: Any) -> tuple[Any, ...]:
    from neuron import h

    spec = _resolve_morphology_spec(case_or_spec)
    h.load_file("stdlib.hoc")
    h.load_file("import3d.hoc")

    existing_count = sum(1 for _ in h.allsec())
    reader = _build_neuron_reader(spec)
    reader.input(str(spec.path))
    h.Import3d_GUI(reader, 0).instantiate(None)

    sections = tuple(h.allsec())[existing_count:]
    if len(sections) == 0:
        raise RuntimeError(f"NEURON import3d instantiated no sections from {spec.path!r}.")
    return sections


def delete_neuron_sections(secs) -> None:
    from neuron import h

    for sec in secs:
        try:
            h.delete_section(sec=sec)
        except Exception:
            pass


def locate_root_neuron_soma(secs):
    from neuron import h

    soma_roots = []
    for sec in secs:
        ref = h.SectionRef(sec=sec)
        if ref.has_parent():
            continue
        if _infer_neuron_branch_type(sec.name()) == "soma":
            soma_roots.append(sec)
    if len(soma_roots) != 1:
        raise ValueError(f"Expected exactly one root soma section, found {len(soma_roots)}.")
    return soma_roots[0]


def _resolve_morphology_spec(case_or_spec: Any) -> MorphologySpec:
    if _looks_like_morphology_spec(case_or_spec):
        return case_or_spec
    spec = getattr(case_or_spec, "morphology", None)
    if _looks_like_morphology_spec(spec):
        return spec
    raise TypeError(f"Could not resolve morphology spec from {type(case_or_spec).__name__!s}.")


def _build_neuron_reader(spec: MorphologySpec):
    from neuron import h

    if spec.kind == "swc":
        return h.Import3d_SWC_read()
    if spec.kind == "asc":
        return h.Import3d_Neurolucida3()
    raise NotImplementedError("morphology.kind='neuroml2' is reserved but not implemented yet.")


def _infer_neuron_branch_type(section_name: str) -> str:
    prefix = section_name.split("[", 1)[0]
    if prefix == "soma":
        return "soma"
    if prefix == "axon":
        return "axon"
    if prefix == "dend":
        return "basal_dendrite"
    return prefix


def _looks_like_morphology_spec(value: Any) -> bool:
    return hasattr(value, "kind") and hasattr(value, "path")
