from __future__ import annotations

import warnings

import braincell


def test_legacy_morphology_symbol_still_accessible_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        morphology_cls = braincell.Morphology
    assert morphology_cls.__name__ == "Morphology"
    assert any("deprecated" in str(item.message).lower() for item in caught)


def test_legacy_morph_module_still_accessible_with_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        morph_module = braincell.morph
    assert hasattr(morph_module, "from_swc")
    if caught:
        assert any("braincell.morph" in str(item.message) for item in caught)


def test_multi_compartment_still_available() -> None:
    assert braincell.MultiCompartment.__name__ == "MultiCompartment"
