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
