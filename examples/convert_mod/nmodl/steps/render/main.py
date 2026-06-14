

from .variants.braincell_one_ion_hh_ohmic import VARIANT_NAME as BRAINCELL_ONE_ION_HH_OHMIC_VARIANT
from .variants.braincell_one_ion_hh_ohmic import run as run_braincell_one_ion_hh_ohmic

STEP_RUNNERS = {
    BRAINCELL_ONE_ION_HH_OHMIC_VARIANT: run_braincell_one_ion_hh_ohmic,
}


def get_variants() -> list[str]:
    return sorted(STEP_RUNNERS)


def run(step2_result: dict, *, variant: str = BRAINCELL_ONE_ION_HH_OHMIC_VARIANT):
    try:
        runner = STEP_RUNNERS[variant]
    except KeyError as exc:
        raise SystemExit(f"Unknown Step 3 variant `{variant}`. Available: {', '.join(get_variants())}") from exc
    return runner(step2_result)
