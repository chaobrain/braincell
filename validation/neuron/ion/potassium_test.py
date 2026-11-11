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

"""``PotassiumDetailedFH`` against Botta et al.'s (2010) ``K_conc.mod`` in NEURON (slow tier).

ModelDB 127021's ``K_conc.mod`` carries no license statement; whether it may
be committed awaits a maintainers' decision. Until then the configuration
``bo2010/potassium_detailed_fh`` reads it from ``BRAINCELL_BO2010_MOD_DIR``
(SHA-256 checked) and these tests skip without it. The soma: L = diam =
15 um, 23 degC, ``ki`` 140 and ``ko`` 5 mM, the ``.mod`` file's ``d = 0.2 um``
(``d = d_o = 0.4 um`` in BrainCell: the file writes the influx as
``ik / (2 FARADAY d)``), ``beta = 0.075 /ms`` (the published simulation's),
a 1 mS/cm2 potassium leak (``kleak.mod``, ``K_Leak``), IClamp 0.05 nA from
t = 0, 200 ms. NEURON's ``derivimplicit`` is implicit Euler and BrainCell's
pool update exponential Euler, so the comparison is T1, class (c).
"""

import dataclasses
import os
import unittest
from unittest import mock

import numpy as np
import pytest

from validation.neuron._suite_loader import use_double_precision
from validation.neuron._tolerance import T0_LIMITS, ToleranceTable
from validation.neuron.ion import runner
from validation.neuron.ion.tolerances import IMPLICIT_EULER, TABLE

os.environ.setdefault("JAX_PLATFORMS", "cpu")
use_double_precision()
pytestmark = pytest.mark.slow

CONFIG, PROTOCOL = "bo2010/potassium_detailed_fh", "pool_cc"
METRICS = ("e_V", "e_Ci", "e_pool")
BETA, DT = 0.075, 0.025  # /ms, ms


def _violations(result, table: ToleranceTable = TABLE) -> list[str]:
    return [m for m in METRICS if not result[m] <= table.limit(CONFIG, PROTOCOL, m)]


def _implicit_euler(original):
    """Scale the pool's derivatives by ``log(1 + x) / x``, ``x = beta dt``.

    Under the exponential Euler step ``y + (1 - exp(-k x)) / (k beta) * k f``
    of the linear relaxation ``f = s - beta (y - y_rest)``, ``k = log(1 + x)
    / x`` gives ``y + dt f / (1 + x)``: NEURON's implicit Euler step.
    """

    def hook(self, V):
        import brainstate
        import brainunit as u

        original(self, V)
        dt = brainstate.environ.get("dt")
        for state, beta in ((self.Ci, self.beta), (self.Co, self.beta_o)):
            x = beta * dt
            state.derivative = state.derivative * (u.math.log1p(x) / x)

    return hook


def _patched():
    from braincell.ion import PotassiumDetailedFH

    hook = _implicit_euler(PotassiumDetailedFH._ion_compute_derivative_hook)
    return mock.patch.object(PotassiumDetailedFH, "_ion_compute_derivative_hook", hook)


def _with_d_o(factor):
    load_case = runner.load_case

    def scaled(name, protocol):
        case = load_case(name, protocol)
        params = tuple(dataclasses.replace(p, value=factor * p.value) if p.braincell == "d_o" else p for p in case.pool_params)
        return dataclasses.replace(case, pool_params=params)

    return mock.patch.object(runner, "load_case", scaled)


class BottaPotassiumPoolTest(unittest.TestCase):
    result: dict = {}

    @classmethod
    def setUpClass(cls):
        try:
            runner.load_case(CONFIG, PROTOCOL)
        except runner.SourcesUnavailable as reason:
            raise unittest.SkipTest(str(reason))
        cls.result = runner.compare_protocol(CONFIG, PROTOCOL)

    def test_the_pool_matches_neuron_within_its_t1_entry(self):
        result, neuron = self.result, self.result["neuron"]
        ki, ko = neuron["cai"], neuron["pool:Co"]
        print(
            f"\n{CONFIG}: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}, e_pool = {result['e_pool']:.3e}, "
            f"ko {ko[0]:.4f} -> {ko[-1]:.4f} mM"
        )
        # k_conc writes ki and ko, so finitialize resets them (L23): both start at the configured values.
        self.assertEqual((ki[0], ko[0]), (140.0, 5.0))
        self.assertEqual(result["braincell"]["pool:Co"][0], 5.0)
        self.assertGreater(ko[-1], 5.02)
        self.assertEqual(_violations(result), [])

    def test_the_t1_limits_are_needed_and_tight(self):
        (entry,) = [e for e in TABLE.entries if (e.config, e.protocol) == (CONFIG, PROTOCOL)]
        self.assertEqual((entry.deviation_class, entry.section), ("c", IMPLICIT_EULER))
        for metric, limit in entry.limits.items():
            measured = self.result[metric]
            with self.subTest(metric=metric):
                self.assertGreater(measured, T0_LIMITS[metric])
                self.assertLessEqual(limit, 4.0 * measured)
                lower = dataclasses.replace(entry, limits={**entry.limits, metric: 0.9 * measured})
                self.assertEqual(_violations(self.result, ToleranceTable([lower])), [metric])

    def test_the_first_step_differs_by_the_implicit_euler_factor(self):
        neuron, braincell = self.result["neuron"]["pool:Co"], self.result["braincell"]["pool:Co"]
        x = BETA * DT
        ratio = (braincell[1] - braincell[0]) / (neuron[1] - neuron[0])
        print(f"\nfirst-step change of ko, BrainCell / NEURON = {ratio:.15f}")
        self.assertAlmostEqual(ratio, (1.0 - np.exp(-x)) * (1.0 + x) / x, delta=1e-10)

    def test_an_implicit_euler_pool_update_gives_t0(self):
        with _patched():
            result = runner.compare_protocol(CONFIG, PROTOCOL)
        print(f"\nimplicit Euler: e_V = {result['e_V']:.3e} mV, e_Ci = {result['e_Ci']:.3e}, e_pool = {result['e_pool']:.3e}")
        self.assertEqual(_violations(result, ToleranceTable([])), [])

    def test_a_space_depth_off_by_0p1_percent_fails_on_the_t0_path(self):
        with _patched(), _with_d_o(1.001):
            result = runner.compare_protocol(CONFIG, PROTOCOL)
        print(f"\nd_o x 1.001: e_V = {result['e_V']:.3e} mV, e_pool = {result['e_pool']:.3e}")
        self.assertEqual(sorted(_violations(result, ToleranceTable([]))), ["e_V", "e_pool"])
        self.assertGreater(result["e_pool"], 1e-6)


if __name__ == "__main__":
    unittest.main()
