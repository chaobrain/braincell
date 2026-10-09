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

"""Select calibration time nodes from Exp2 synapse parameters."""

import argparse
import json
from pathlib import Path

from braincell.reduction.dif.calibration import INTERPOLATION_FRACTION, select
from braincell.reduction.dif.calibration_utils import TAIL_FRACTION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--definition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    definition = json.loads(args.definition.read_text())
    groups = {}
    for site in definition['sites']:
        groups.setdefault((site['tau1_ms'], site['tau2_ms']), []).append(site['role'])
    rows = [select(rise, decay, float(definition['dt_ms']), roles) for (rise, decay), roles in sorted(groups.items())]
    result = dict(
        schema_version=1,
        rule='exp2_peak_anchored_curvature',
        curves=rows,
        terminal_peak_fraction_limit=TAIL_FRACTION,
        interpolation_peak_fraction_limit=INTERPOLATION_FRACTION,
        calibration_run=False,
        seed_results_used=False,
        scope='Synapse waveform interpolation only; no voltage or spike error guarantee.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'curves': rows}, indent=2))


if __name__ == '__main__':
    main()
