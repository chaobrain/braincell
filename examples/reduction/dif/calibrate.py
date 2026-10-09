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

"""Build a DIF table in one call from a declared detailed network.

The factory is an importable module:function called with optional JSON keyword
arguments. It returns a braincell.Network. Its workload runs automatically
for the default 30 seconds before canonical AP and REST/POST calibration.
--duration-ms optionally overrides this duration. Alternatively,
--trace reuses an NPZ with voltage_mV and arrival_steps. The generated table
belongs outside the source repository.
"""

import argparse
import importlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory', required=True, help='Importable module:function returning a declared Network.')
    parser.add_argument('--factory-kwargs', type=Path, help='Optional JSON object of factory keyword arguments.')
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument('--duration-ms', type=float, help='Override the default 30000 ms detailed workload duration.')
    workload.add_argument('--trace', type=Path, help='Optionally reuse voltage_mV and arrival_steps arrays from NPZ.')
    parser.add_argument('--dt-ms', type=float, required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=30000, help='Positive condition batch size per GPU.')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='Logical GPU indices visible to JAX.')
    parser.add_argument('--time-grid', type=Path)
    parser.add_argument('--canonical-delay-ms', type=float, default=20.0)
    parser.add_argument('--canonical-duration-ms', type=float, default=0.3)
    parser.add_argument('--canonical-amplitude-na', type=float, default=6.125)
    args = parser.parse_args()
    if args.batch < 1 or args.dt_ms <= 0 or (args.duration_ms is not None and args.duration_ms <= 0):
        parser.error('Batch size, timestep and duration must be positive.')

    import brainstate
    import brainunit as u
    import numpy as np
    from braincell.reduction.dif.calibration import CalibrationExecution, calibrate

    brainstate.environ.set(precision=64)
    module, function = args.factory.rsplit(':', 1)
    kwargs = json.loads(args.factory_kwargs.read_text()) if args.factory_kwargs else {}
    network = getattr(importlib.import_module(module), function)(**kwargs)
    workload_args = {}
    if args.trace is None:
        if args.duration_ms is not None:
            workload_args['duration'] = args.duration_ms * u.ms
    else:
        with np.load(args.trace, allow_pickle=False) as trace:
            workload_args['voltage'] = trace['voltage_mV'] * u.mV
            workload_args['arrival_steps'] = trace['arrival_steps']
    table_path = calibrate(
        network,
        population=args.population,
        **workload_args,
        dt=args.dt_ms * u.ms,
        output=args.output,
        execution=CalibrationExecution(tuple(args.devices), args.batch),
        time_grid=json.loads(args.time_grid.read_text()) if args.time_grid else None,
        canonical_current_delay=args.canonical_delay_ms * u.ms,
        canonical_current_duration=args.canonical_duration_ms * u.ms,
        canonical_current_amplitude=args.canonical_amplitude_na * u.nA,
    )
    print(json.dumps({'table': str(table_path)}, indent=2))


if __name__ == '__main__':
    main()
