from __future__ import annotations

from .variants.one_ion_hh_ohmic import VARIANT_NAME as ONE_ION_HH_OHMIC_VARIANT
from .variants.one_ion_hh_ohmic import build_one_ion_hh_ohmic_ir
from .variants.one_ion_hh_ohmic import run as run_one_ion_hh_ohmic
from .variants.one_ion_hh_ohmic import summarize_one_ion_hh_ohmic_ir

STEP_RUNNERS = {
    ONE_ION_HH_OHMIC_VARIANT: run_one_ion_hh_ohmic,
}


def get_variants() -> list[str]:
    return sorted(STEP_RUNNERS)


def run(step1_result: dict, *, variant: str = ONE_ION_HH_OHMIC_VARIANT):
    try:
        runner = STEP_RUNNERS[variant]
    except KeyError as exc:
        raise SystemExit(f"Unknown Step 2 variant `{variant}`. Available: {', '.join(get_variants())}") from exc
    return runner(step1_result)
