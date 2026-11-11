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

"""``dt`` convergence of the sodium, HCN and leak channels.

Every configuration of a class defined in ``braincell/channel/sodium.py``,
``hyperpolarization_activated.py`` or ``leaky.py`` runs on the Cell's default
path and is checked against its ``convergence.CONFIGS`` entry: HH channels
are exact under voltage steps, ``IL`` has no state (``Na_Leak``, also
stateless, has no configuration); the eight sodium Markov
channels converge at first order on ``backward_euler`` x 1 with errors above
the guide line, and ``exp_euler`` is exact. Every ramp is finite and its
non-zero differences converge at the window's order (``Na_HH1952`` on finer
steps, see its entry). Twelve classes of the family are stiff at 0.025 ms.
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

FAMILY_MODULES = ("sodium", "hyperpolarization_activated", "leaky")
CHANNEL_ROOT = Path(braincell.channel.__file__).resolve().parent
MARKOV = {
    "NaFHF_MA2020_GrC",
    "Nav1p1_MA2025_BC",
    "Nav1p1_RI2021_SC",
    "Nav1p6_MA2020_GoC",
    "Nav1p6_MA2024_PC",
    "Nav1p6_MA2025_BC",
    "Nav1p6_RI2021_SC",
    "Nav_MA2020_GrC",
}
STIFF = MARKOV | {"HCN1_MA2024_PC", "HCN1_MA2025_BC", "HCN1_RI2021_SC", "Na_ZH2019_IO"}


def family_classes() -> tuple[str, ...]:
    """The registered channel classes defined in the family's modules."""
    registered = set(table.class_names())
    names = []
    for module in FAMILY_MODULES:
        tree = ast.parse((CHANNEL_ROOT / f"{module}.py").read_text())
        names += [node.name for node in tree.body if isinstance(node, ast.ClassDef) and node.name in registered]
    return tuple(sorted(names))


def _braincell_class(config: str) -> str | None:
    payload = json.loads((table.CONFIGS_ROOT / f"{config}.json").read_text())
    return payload.get("mapping", {}).get("impl_name", {}).get("braincell")


def family_configs() -> tuple[str, ...]:
    """The configurations whose BrainCell class is in :func:`family_classes`."""
    classes = set(family_classes())
    return tuple(config for config in table.config_names() if _braincell_class(config) in classes)


FAMILY_CONFIGS = family_configs()
ENTRIES = {entry.config: entry for entry in table.CONFIGS if entry.config in FAMILY_CONFIGS}


class FamilyTableTest(unittest.TestCase):
    def test_the_family_is_in_the_tables(self):
        classes = family_classes()
        self.assertEqual((len(classes), len(FAMILY_CONFIGS)), (24, 20))
        self.assertIn("IL", classes)  # leaky.py
        self.assertEqual(set(ENTRIES), set(FAMILY_CONFIGS))
        # A class or configuration added to the family must join the family's tuples.
        self.assertEqual(classes, tuple(sorted(table.SODIUM_HCN_LEAK_FAMILY_CLASSES)))
        self.assertEqual(FAMILY_CONFIGS, tuple(sorted(table.SODIUM_HCN_LEAK_FAMILY_CONFIGS)))
        stiff = {entry.name for entry in table.CLASSES if entry.name in classes and entry.stiff}
        self.assertEqual(stiff, STIFF)
        for config, entry in ENTRIES.items():
            expected = cv.ABOVE_GUIDE_LINE if _braincell_class(config) in MARKOV else cv.EXACT
            self.assertEqual(entry.classification, expected, config)
        self.assertEqual(sum(_braincell_class(config) in MARKOV for config in ENTRIES), 8)

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
        for metric, limit in entry.limits.items():
            # Each limit fails once tightened below the measurement.
            tightened = dataclasses.replace(entry, limits={**entry.limits, metric: 0.9 * report.error(metric)})
            with self.assertRaises(AssertionError):
                table.assert_matches(report, tightened)
        if report.class_name in MARKOV:
            exp_euler = cv.check_config(config, subsolver="exp_euler", substeps=1, ramp=False)
            table.assert_matches(exp_euler, table.ConfigEntry(config, cv.EXACT))
        if report.class_name == "IL":
            # No state to compare; the current is compared and does not depend on dt.
            self.assertTrue(all(e["e_s"] is None and e["e_I"] is not None for e in report.errors.values()))

    def test_finer_ramp_steps_are_needed(self):
        # Each entry with its own ramp steps is out of the window on RAMP_DTS_MS.
        finer = [entry for entry in ENTRIES.values() if entry.ramp_dts_ms != cv.RAMP_DTS_MS]
        self.assertEqual([entry.config for entry in finer], ["builtin/na_hh1952_hh"])
        for entry in finer:
            _, orders = cv.ramp_convergence(entry.config)
            low, high = cv.RAMP_ORDER_WINDOWS["e_I"]
            self.assertFalse(low <= orders["e_I"] <= high, orders)


for _config in FAMILY_CONFIGS:
    setattr(
        FamilyConvergenceTest,
        "test_" + _config.replace("/", "__"),
        lambda self, config=_config: self._check(config),
    )


if __name__ == "__main__":
    unittest.main()
