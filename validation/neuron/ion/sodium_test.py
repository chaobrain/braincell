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

"""``SodiumDetailed`` against Canavier's (1999) soma in NEURON, the pump off and on (slow tier).

ModelDB 6763's ``nadifl.mod``, ``pump.mod`` and ``leak.mod`` carry no license
statement; whether they may be committed awaits a maintainers' decision. Until
then the configurations ``ca1999/sodium_detailed_pump_off`` and
``ca1999/sodium_detailed_pump`` read them from ``BRAINCELL_CA1999_MOD_DIR``
(SHA-256 checked) and these tests skip without it.
The soma: L = diam = 15 um, 35 degC, ``nai`` 4 and ``nao`` 145 mM, IClamp
-0.02 nA from t = 0, 2000 ms. BrainCell: ``SodiumDetailed(d=lambda ctx:
ctx.diam_mid / 4)`` with ``Na_Leak`` (40 uS/cm2), ``PotassiumFixed(E=-100 mV)``
and ``K_Leak`` (100 uS/cm2); the second configuration adds ``pump.mod``
(``ipumpmax = 0.012`` mA/cm2, the ``seg1.nrn`` value) and
``NaKPump_Ca1999(imax=0.012 mA/cm2)``. With no relaxation the pool update is
the same in both programs, so the limits are T0.
"""

import dataclasses
import os
import unittest
from unittest import mock

import numpy as np
import pytest

from validation.neuron._suite_loader import use_double_precision
from validation.neuron._tolerance import ToleranceTable
from validation.neuron.ion import runner
from validation.neuron.ion.tolerances import TABLE

os.environ.setdefault("JAX_PLATFORMS", "cpu")
use_double_precision()
pytestmark = pytest.mark.slow

CONFIG, PROTOCOL = "ca1999/sodium_detailed_pump_off", "pool_cc"
PUMP_CONFIG = "ca1999/sodium_detailed_pump"


def _violations(result, table: ToleranceTable = TABLE, config: str = CONFIG) -> list[str]:
    return [m for m in ("e_V", "e_Ci", "e_pool") if not result[m] <= table.limit(config, PROTOCOL, m)]


class CanavierSomaPumpOffTest(unittest.TestCase):
    def setUp(self):
        try:
            runner.load_case(CONFIG, PROTOCOL)
        except runner.SourcesUnavailable as reason:
            self.skipTest(str(reason))

    def test_sodium_accumulation_matches_neuron(self):
        result = runner.compare_protocol(CONFIG, PROTOCOL)
        nai = result["neuron"]["cai"]
        print(
            f"\n{CONFIG}: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}, "
            f"nai {nai[0]:.4f} -> {nai[-1]:.4f} mM"
        )
        self.assertEqual(nai[0], 4.0)  # written before finitialize (L23)
        self.assertGreater(nai[-1], 4.3)
        self.assertEqual(_violations(result), [])
        self.assertEqual(_violations(result, ToleranceTable([])), [])  # T0: no table entry

    def test_a_doubled_volume_factor_fails(self):
        load_case = runner.load_case

        def doubled(name, protocol):
            case = load_case(name, protocol)
            (d,) = case.pool_params
            return dataclasses.replace(case, pool_params=(dataclasses.replace(d, value=2.0 * d.value),))

        with mock.patch.object(runner, "load_case", doubled):
            result = runner.compare_protocol(CONFIG, PROTOCOL)
        print(f"\nd x 2: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}")
        self.assertEqual(sorted(_violations(result, ToleranceTable([]))), ["e_Ci", "e_V"])


class CanavierSomaPumpTest(unittest.TestCase):
    def setUp(self):
        try:
            runner.load_case(PUMP_CONFIG, PROTOCOL)
        except runner.SourcesUnavailable as reason:
            self.skipTest(str(reason))

    def test_sodium_with_the_pump_matches_neuron(self):
        result = runner.compare_protocol(PUMP_CONFIG, PROTOCOL)
        nai = result["neuron"]["cai"]
        print(
            f"\n{PUMP_CONFIG}: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}, "
            f"nai {nai[0]:.4f} -> {nai[-1]:.4f} mM"
        )
        self.assertEqual(nai[0], 4.0)  # written before finitialize (L23)
        self.assertLess(abs(nai[-1] - 3.9901), 1e-4)  # the pump outweighs the leak: nai falls
        self.assertEqual(_violations(result, config=PUMP_CONFIG), [])
        self.assertEqual(_violations(result, ToleranceTable([]), PUMP_CONFIG), [])  # T0: no table entry

    def test_km_one_percent_high_fails(self):
        load_case = runner.load_case

        def raised(name, protocol):
            case = load_case(name, protocol)
            mechanisms = []
            for kind, cls, params in case.braincell_mechanisms:
                if cls == "NaKPump_Ca1999":
                    params = tuple(dataclasses.replace(p, value=1.01 * p.value) if p.braincell == "km" else p for p in params)
                mechanisms.append((kind, cls, params))
            return dataclasses.replace(case, braincell_mechanisms=tuple(mechanisms))

        with mock.patch.object(runner, "load_case", raised):
            result = runner.compare_protocol(PUMP_CONFIG, PROTOCOL)
        print(f"\nkm x 1.01: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}")
        self.assertEqual(sorted(_violations(result, ToleranceTable([]), PUMP_CONFIG)), ["e_Ci", "e_V"])


if __name__ == "__main__":
    unittest.main()
