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

"""``dt`` convergence of the calcium and calcium-activated potassium channels.

Every configuration of a class defined in ``braincell/channel/calcium.py`` or
``potassium_calcium.py`` runs on the Cell's default path and is checked
against its ``convergence.CONFIGS`` entry: HH channels are exact under
voltage steps; the ``Kca1p1_*`` Markov channels converge at first order
(and ``exp_euler`` is exact); ``Kca2p2_*`` and ``SK_SU2015_DCN``, gated by
calcium alone, use the relaxation protocol. Every ramp is finite and its
non-zero differences converge at the window's order.
"""

import ast
import dataclasses
import json
from pathlib import Path
import unittest

import pytest

import braincell.channel
from validation.neuron import _convergence as cv
from validation.neuron._suite_loader import use_double_precision
from validation.neuron.channel_no_conc import convergence as table

use_double_precision()

pytestmark = pytest.mark.slow

FAMILY_MODULES = ("calcium", "potassium_calcium")
CHANNEL_ROOT = Path(braincell.channel.__file__).resolve().parent
STIFF = {
    "Cav1p2_MA2020_GoC",
    "Cav1p2_MA2025_BC",
    "Cav1p3_MA2020_GoC",
    "Cav1p3_MA2025_BC",
    "Kca1p1_MA2020_GoC",
    "Kca1p1_MA2020_GrC",
    "Kca1p1_MA2024_PC",
    "Kca1p1_MA2025_BC",
    "Kca1p1_RI2021_SC",
}


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


class FamilyTableTest(unittest.TestCase):
    def test_the_family_is_in_the_tables_and_the_coverage_scope(self):
        classes = family_classes()
        self.assertEqual((len(classes), len(FAMILY_CONFIGS)), (50, 45))
        self.assertEqual(set(ENTRIES), set(FAMILY_CONFIGS))
        # A class or configuration added to the family must join the coverage scope.
        self.assertEqual(classes, tuple(sorted(table.CALCIUM_FAMILY_CLASSES)))
        self.assertEqual(FAMILY_CONFIGS, tuple(sorted(table.CALCIUM_FAMILY_CONFIGS)))
        self.assertLessEqual(set(classes), set(table.COVERAGE_SCOPE["classes"]))
        self.assertLessEqual(set(FAMILY_CONFIGS), set(table.COVERAGE_SCOPE["configs"]))
        stiff = {entry.name for entry in table.CLASSES if entry.name in classes and entry.stiff}
        self.assertEqual(stiff, STIFF)

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
        report = cv.check_config(config, protocol=entry.protocol)
        table.assert_matches(report, entry)
        self.assertTrue(report.ramp_ok, (report.ramp_differences, report.ramp_orders))
        for metric, limit in entry.limits.items():
            # Each limit fails once tightened below the measurement.
            tightened = dataclasses.replace(entry, limits={**entry.limits, metric: 0.9 * report.error(metric)})
            with self.assertRaises(AssertionError):
                table.assert_matches(report, tightened)
        if report.class_name.startswith("Kca1p1_"):
            exp_euler = cv.check_config(config, subsolver="exp_euler", substeps=1, ramp=False)
            table.assert_matches(exp_euler, table.ConfigEntry(config, cv.EXACT))


for _config in FAMILY_CONFIGS:
    setattr(
        FamilyConvergenceTest,
        "test_" + _config.replace("/", "__"),
        lambda self, config=_config: self._check(config),
    )


if __name__ == "__main__":
    unittest.main()
