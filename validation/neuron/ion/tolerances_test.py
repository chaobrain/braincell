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

"""Every ion-pool configuration against NEURON, within ``tolerances.py`` (slow tier).

Each configuration runs under each of its templates. The T1 entries are then
checked from both sides: every limit is needed (the measurement exceeds T0)
and tight (a limit below the measurement fails), and patching every named
cause of a configuration brings all of its protocols to T0.
"""

from contextlib import ExitStack, contextmanager
import dataclasses
import os
from pathlib import Path
import unittest
from unittest import mock

import jax
import pytest

from validation.neuron._suite_loader import use_double_precision
from validation.neuron._tolerance import PROTOCOL_METRICS, T0_LIMITS, ToleranceTable
from validation.neuron.ion import runner
from validation.neuron.ion.tolerances import EXPLICIT_EULER, FORWARD_DIFFERENCE, GHK, NEWTON, TABLE

os.environ.setdefault("JAX_PLATFORMS", "cpu")
use_double_precision()
pytestmark = pytest.mark.slow

CASES = tuple(
    (config, Path(template).stem) for config in runner.config_names() for template in runner.load_config(config)["templates"]
)


def _violations(result, table: ToleranceTable) -> list[str]:
    """Return every metric of ``result`` above its limit in ``table``; VC also checks the clamp."""
    case = result["case"]
    failed = [
        f"{case.config} {case.protocol} {metric} = {result[metric]:.3e}"
        for metric in PROTOCOL_METRICS[case.protocol]
        if not result[metric] <= table.limit(case.config, case.protocol, metric)
    ]
    if "clamp_error_mV" in result and not result["clamp_error_mV"] <= 1e-7:
        failed.append(f"{case.config} {case.protocol} clamp error {result['clamp_error_mV']:.3e} mV")
    return failed


def _explicit_euler(original):
    def derivative(self, Ci, V, total_current=None):
        # NEURON's cnexp falls back to explicit Euler for this DERIVATIVE block:
        # with Ci frozen in the derivative, BrainCell's exponential Euler step is explicit too.
        return original(self, jax.lax.stop_gradient(Ci), V, total_current)

    return derivative


def _forward_difference(target, point_V_n, *, point_capacitance, t):
    """The staggered membrane linearization with NEURON's ``(i(v + 0.001) - i(v)) / 0.001``."""
    import brainunit as u

    from braincell._multi_compartment import currents

    def rate(V):
        return currents.total_membrane_rate_point(target, point_V=V, point_capacitance=point_capacitance, t=t)

    step = 0.001 * u.mV
    at_v = rate(point_V_n)
    return (rate(point_V_n + step) - at_v) / step, at_v


@contextmanager
def _root_causes_patched(config: str):
    """Patch every cause the table names for ``config``; yield ``compare_protocol`` keywords."""
    from braincell.ion import CdpHVA_SU2015_DCN
    from braincell.quad import _staggered

    sections = {entry.section for entry in TABLE.entries if entry.config == config}
    load_case = runner.load_case

    def frozen(name, protocol):
        case = load_case(name, protocol)
        if case.channel is None:
            return case
        return dataclasses.replace(case, channel=dataclasses.replace(case.channel, braincell=f"{case.channel.braincell}_Frozen"))

    with ExitStack() as stack:
        if EXPLICIT_EULER in sections:
            patched = _explicit_euler(CdpHVA_SU2015_DCN.derivative)
            stack.enter_context(mock.patch.object(CdpHVA_SU2015_DCN, "derivative", patched))
        if FORWARD_DIFFERENCE in sections:
            stack.enter_context(mock.patch.object(_staggered, "_point_linear_and_rate", _forward_difference))
        if GHK in sections:
            stack.enter_context(mock.patch.object(runner, "load_case", frozen))
        yield {"pool_solver": runner.neuron_sparse_solver()} if NEWTON in sections else {}


class PoolAgreementTest(unittest.TestCase):
    results: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.results = {key: runner.compare_protocol(*key) for key in CASES}
        print(f"\n{len(CASES)} pool comparisons, {len(TABLE.entries)} T1 entries")

    def test_every_protocol_is_within_the_table(self):
        self.assertEqual(len(CASES), 22)
        failed = [line for result in self.results.values() for line in _violations(result, TABLE)]
        self.assertEqual(failed, [])

    def test_t1_limits_are_needed_and_tight(self):
        external = set(runner.external_config_names())  # checked by their own test modules
        for entry in TABLE.entries:
            if entry.config in external:
                continue
            result = self.results[(entry.config, entry.protocol)]
            for metric, limit in entry.limits.items():
                measured = result[metric]
                with self.subTest(config=entry.config, protocol=entry.protocol, metric=metric):
                    self.assertGreater(measured, T0_LIMITS[metric])
                    self.assertLessEqual(limit, 4.0 * measured)
                    self.assertEqual(_violations(result, TABLE), [])
                    lower = dataclasses.replace(entry, limits={**entry.limits, metric: max(0.9 * measured, 1.01 * T0_LIMITS[metric])})
                    tightened = ToleranceTable([lower if other is entry else other for other in TABLE.entries])
                    self.assertTrue(_violations(result, tightened))

    def test_patching_the_named_causes_gives_t0(self):
        configs = sorted({entry.config for entry in TABLE.entries})
        t0 = ToleranceTable([])
        for config, protocol in CASES:
            if config not in configs:
                continue
            with self.subTest(config=config, protocol=protocol), _root_causes_patched(config) as keywords:
                self.assertEqual(_violations(runner.compare_protocol(config, protocol, **keywords), t0), [])

    def test_a_pump_rate_off_by_0p1_percent_fails_the_voltage_clamp(self):
        config, protocol = "goc_ma2020/cdpstc_ma20_goc", "pool_vc"
        load_case = runner.load_case

        def faster_pump(name, protocol):
            case = load_case(name, protocol)
            kpmp1 = runner.Parameter(None, "kpmp1", "/mM/ms", 1.001 * 3e-3)  # BrainCell only
            return dataclasses.replace(case, pool_params=case.pool_params + (kpmp1,))

        with _root_causes_patched(config) as keywords, mock.patch.object(runner, "load_case", faster_pump):
            result = runner.compare_protocol(config, protocol, **keywords)
        self.assertTrue(_violations(result, ToleranceTable([])))
        self.assertGreater(result["e_Ci"], 1e-6)


if __name__ == "__main__":
    unittest.main()
