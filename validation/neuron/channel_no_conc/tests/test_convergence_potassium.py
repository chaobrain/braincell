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

"""``dt`` convergence of the potassium and sodium-activated potassium channels.

Every configuration of a class defined in ``braincell/channel/potassium.py``
or ``potassium_sodium.py`` runs on the Cell's default path and is checked
against its ``convergence.CONFIGS`` entry: all are HH channels on
``ind_exp_euler``, exact under voltage steps. Every ramp is finite and its
non-zero differences converge at the window's order; ``K_Kv_test``'s ramp
crosses its former 0/0 point at 25 mV. No class of the family is stiff at
0.025 ms.
"""

import ast
import json
import math
from pathlib import Path
import unittest
from unittest import mock

import brainunit as u
import pytest

import braincell.channel
from braincell.channel import K_Kv_test
from validation.neuron import _convergence as cv
from validation.neuron._suite_loader import use_double_precision
from validation.neuron.channel_no_conc import convergence as table

use_double_precision()

pytestmark = pytest.mark.slow

FAMILY_MODULES = ("potassium", "potassium_sodium")
CHANNEL_ROOT = Path(braincell.channel.__file__).resolve().parent
STIFF: set[str] = set()
KV_TESTING = "testing/kv_testing"


def family_classes() -> tuple[str, ...]:
    """The registered channel classes defined in the family's modules."""
    registered = set(table.class_names())
    names = []
    for module in FAMILY_MODULES:
        tree = ast.parse((CHANNEL_ROOT / f"{module}.py").read_text())
        names += [node.name for node in tree.body if isinstance(node, ast.ClassDef) and node.name in registered]
    return tuple(sorted(names))


def family_configs() -> tuple[str, ...]:
    """The configurations whose BrainCell class is in :func:`family_classes`."""
    classes = set(family_classes())
    found = []
    for config in table.config_names():
        payload = json.loads((table.CONFIGS_ROOT / f"{config}.json").read_text())
        if payload.get("mapping", {}).get("impl_name", {}).get("braincell") in classes:
            found.append(config)
    return tuple(found)


FAMILY_CONFIGS = family_configs()
ENTRIES = {entry.config: entry for entry in table.CONFIGS if entry.config in FAMILY_CONFIGS}


def _prefix_kv_test_n_tau(self, V, K):
    # ``K_Kv_test.f_n_tau`` as written before its analytic limit, verbatim: 0/0 at V - V_sh = v12.
    V = (V - self.V_sh).to_decimal(u.mV)
    Ra = self.Ra / (1 / u.mV / u.ms)
    Rb = self.Rb / (1 / u.mV / u.ms)
    q = self.q.to_decimal(u.mV) if hasattr(self.q, "to_decimal") else self.q
    v12 = self.v12.to_decimal(u.mV) if hasattr(self.v12, "to_decimal") else self.v12
    denom = Ra * (V - v12) / (1.0 - u.math.exp(-(V - v12) / q)) + (-Rb) * (V - v12) / (
        1.0 - u.math.exp(-(V - v12) / (-q))
    )
    return 1.0 / denom


class FamilyTableTest(unittest.TestCase):
    def test_the_family_is_in_the_tables(self):
        classes = family_classes()
        self.assertEqual((len(classes), len(FAMILY_CONFIGS)), (39, 31))
        self.assertIn("Kv1p5_MA2020_GrC", classes)  # potassium_sodium.py
        self.assertEqual(set(ENTRIES), set(FAMILY_CONFIGS))
        # A class or configuration added to the family must join the family's tuples.
        self.assertEqual(classes, tuple(sorted(table.POTASSIUM_FAMILY_CLASSES)))
        self.assertEqual(FAMILY_CONFIGS, tuple(sorted(table.POTASSIUM_FAMILY_CONFIGS)))
        stiff = {entry.name for entry in table.CLASSES if entry.name in classes and entry.stiff}
        self.assertEqual(stiff, STIFF)
        self.assertEqual({entry.classification for entry in ENTRIES.values()}, {cv.EXACT})

    def test_removing_any_family_entry_fails_the_coverage_check(self):
        for name in FAMILY_CONFIGS:
            remaining = [entry for entry in table.CONFIGS if entry.config != name]
            self.assertTrue(table.coverage_problems(configs=remaining), name)
        for name in family_classes():
            remaining = [entry for entry in table.CLASSES if entry.name != name]
            self.assertTrue(table.coverage_problems(classes=remaining), name)


class FamilyConvergenceTest(unittest.TestCase):
    """One test per configuration; see the module docstring."""

    def _check(self, config: str) -> None:
        entry = ENTRIES[config]
        report = cv.check_config(config, protocol=entry.protocol, ramp_dts_ms=entry.ramp_dts_ms)
        table.assert_matches(report, entry)
        self.assertTrue(report.ramp_ok, (report.ramp_differences, report.ramp_orders))

    def test_kv_testing_ramp_is_not_finite_with_the_source_expression(self):
        # Counter-example: the ramp samples 25 mV exactly, where the pre-fix rate is 0/0.
        with mock.patch.object(K_Kv_test, "f_n_tau", _prefix_kv_test_n_tau):
            differences, _ = cv.ramp_convergence(KV_TESTING)
        self.assertTrue(any(v is not None and not math.isfinite(v) for d in differences for v in d.values()), differences)


for _config in FAMILY_CONFIGS:
    setattr(
        FamilyConvergenceTest,
        "test_" + _config.replace("/", "__"),
        lambda self, config=_config: self._check(config),
    )


if __name__ == "__main__":
    unittest.main()
