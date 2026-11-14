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

"""Fast checks of the ion-pool configurations, templates and tolerance table (no NEURON run)."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from validation.neuron import _clamp
from validation.neuron._paths import DATA_ROOT
from validation.neuron._tolerance import POOL_PROTOCOL_METRICS
from validation.neuron.ion import runner
from validation.neuron.ion.tolerances import KNOWN_DIFFERENCES, TABLE

NOTEBOOK_POOLS = {
    "CdpStC_MA2020_GoC",
    "CdpCAM_MA2024_PC",
    "CdpHVA_SU2015_DCN",
    "ToyCaBindingKinetic_SU2015_DCN",
    "ToyCaBindingSourceKinetic_SU2015_DCN",
    "ToyCaPumpFactorKinetic_SU2015_DCN",
    "ToyDiamFactorKinetic_SU2015_DCN",
}
"""The pools of the ion notebooks these configurations replaced."""


def _cases():
    for config in runner.config_names():
        for template in runner.load_config(config)["templates"]:
            yield runner.load_case(config, Path(template).stem)


class PoolConfigurationTest(unittest.TestCase):
    def test_every_configuration_resolves_under_its_templates(self):
        from braincell.mech import get_registry

        registry, cases = get_registry(), list(_cases())
        self.assertEqual(len(runner.config_names()), 11)
        self.assertEqual(len(cases), 22)
        for case in cases:
            with self.subTest(config=case.config, protocol=case.protocol):
                self.assertTrue(all(path.is_file() for path in case.mod_files))
                registry.get("ion", case.pool[1])
                self.assertEqual(case.channel is None, case.protocol == "pool_zero_current")
                if case.channel is not None:
                    registry.get("channel", case.channel.braincell)
                    self.assertTrue(case.channel.gates)
                    # A channel with zero conductance compares nothing (L32).
                    self.assertTrue(case.channel.params and all(p.value > 0 for p in case.channel.params))
                self.assertEqual(case.n_steps * case.simulation["dt_ms"], case.simulation["duration_ms"])

    def test_the_notebook_pools_are_covered_with_a_relaxation(self):
        relaxing = {case.pool[1] for case in _cases() if case.protocol == "pool_zero_current"}
        self.assertLessEqual(NOTEBOOK_POOLS, relaxing)

    def test_unknown_names_and_protocols_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not list"):
            runner.load_case("goc_ma2020/cdpstc_ma20_goc_cav3p1", "pool_zero_current")
        raw = runner.load_config("goc_ma2020/cdpstc_ma20_goc")
        mapping = raw["mapping"]
        broken = {
            "needs a channel": {**raw, "mapping": {k: v for k, v in mapping.items() if k != "impl_name"}},
            "unknown unit": {
                **raw,
                "mapping": {**mapping, "channel_params": {"g_max_S_cm2": {**mapping["channel_params"]["g_max_S_cm2"], "unit": "mS"}}},
            },
            "no mapping": {**raw, "defaults": {"channel_params": {"gbar": 1.0}}},
        }
        for message, config in broken.items():
            with self.subTest(message), mock.patch.object(runner, "load_config", lambda name, config=config: config):
                with self.assertRaisesRegex(ValueError, message):
                    runner.load_case("goc_ma2020/cdpstc_ma20_goc", "pool_vc")

    def test_protocol_commands(self):
        vc = runner.load_case("goc_ma2020/cdpstc_ma20_goc", "pool_vc")
        (command,) = _clamp.protocol_commands(vc.voltage_clamp, vc.simulation["dt_ms"])
        self.assertEqual(command.size, 2000)
        np.testing.assert_array_equal(command[[0, 199, 200, 999, 1000, 1999]], [-70, -70, -10, -10, -70, -70])
        zero = runner.load_case("goc_ma2020/cdpstc_ma20_goc", "pool_zero_current")
        (held,) = _clamp.protocol_commands(zero.voltage_clamp, zero.simulation["dt_ms"])
        np.testing.assert_array_equal(held, np.full(1600, -70.0))
        cc = runner.load_case("goc_ma2020/cdpstc_ma20_goc_cav2p1_ri21_sc", "pool_cc")
        self.assertEqual(dict(cc.current_clamp), {"delay_ms": 5.0, "dur_ms": 20.0, "amp_nA": 0.5})


class PinnedSourcesTest(unittest.TestCase):
    """Configurations whose ``.mod`` files are pinned third-party sources under ``data/modeldb``."""

    def test_pinned_configurations_resolve_from_data(self):
        from braincell.mech import get_registry

        # Derived from the configurations, not listed: every one is pinned or not, never both.
        pinned, internal = runner.pinned_config_names(), runner.config_names()
        print(f"\n{len(pinned)} pinned and {len(internal)} other configurations")
        self.assertTrue(pinned and internal)
        self.assertFalse(set(pinned) & set(internal))
        self.assertEqual(sorted(pinned + internal), runner._all_config_names())
        registry = get_registry()
        for config in pinned:
            spec = runner.load_config(config)["identity"]["pinned_mod_files"]
            with self.subTest(config=config):
                self.assertEqual(set(spec), {"dir", "source", "sha256"})
                self.assertTrue(spec["dir"].startswith("modeldb/"))
                case = runner.load_case(config, "pool_cc")
                folder = DATA_ROOT / spec["dir"]
                self.assertEqual(sorted(p.name for p in case.mod_files if p.parent == folder), sorted(spec["sha256"]))
                self.assertTrue(all(p.is_file() for p in case.mod_files))
                # The hashes are the ones in the folder's provenance README.
                readme = (folder / "README.md").read_text(encoding="utf-8")
                for name, digest in spec["sha256"].items():
                    self.assertIn(f"| `{name}` | `{digest}` |", readme)
                registry.get("ion", case.pool[1])
                registry.get("channel", case.channel.braincell)
                for kind, name, params in case.braincell_mechanisms:
                    registry.get(kind, name)
                # The BrainCell conductances conduct (L32).
                conductances = [p for p in case.channel.params if p.braincell == "g_max"]
                self.assertTrue(conductances and all(p.value > 0 for p in conductances))
                self.assertEqual(case.n_steps * case.simulation["dt_ms"], case.simulation["duration_ms"])

    def test_a_changed_or_missing_pinned_file_is_refused(self):
        import shutil
        import tempfile

        for config in runner.pinned_config_names():
            spec = runner.load_config(config)["identity"]["pinned_mod_files"]
            name = sorted(spec["sha256"])[0]
            with tempfile.TemporaryDirectory() as directory, self.subTest(config=config):
                root = Path(directory)
                shutil.copytree(DATA_ROOT / spec["dir"], root / spec["dir"])
                with mock.patch.object(runner, "DATA_ROOT", root):
                    self.assertEqual(len(runner.pinned_mod_files(config)), len(spec["sha256"]))  # the copy passes
                    path = root / spec["dir"] / name
                    data = bytearray(path.read_bytes())
                    data[len(data) // 2] ^= 0x01
                    path.write_bytes(bytes(data))
                    with self.assertRaisesRegex(ValueError, rf"{config} .*{name} has SHA-256"):
                        runner.load_case(config, "pool_cc")
                    path.unlink()
                    with self.assertRaisesRegex(FileNotFoundError, f"{name} is missing"):
                        runner.load_case(config, "pool_cc")

    def test_a_written_extracellular_concentration_is_a_pool_state(self):
        import brainunit as u

        case = runner.load_case("bo2010/potassium_detailed_fh", "pool_cc")
        self.assertIn(DATA_ROOT / "modeldb" / "127021" / "K_conc.mod", case.mod_files)
        self.assertEqual((case.ion, case.states, dict(case.concentrations)), ("k", (("ko", "Co"),), {"Ci": 140.0, "Co": 5.0}))
        braincell = {p.braincell: p.quantity() for p in case.pool_params if p.braincell}
        neuron = {p.neuron: p.value for p in case.pool_params if p.neuron}
        # K_conc.mod's 2 FARADAY d: its d = 0.2 um is d = d_o = 0.4 um at z = 1.
        self.assertEqual((neuron["d"], braincell["d"], braincell["d_o"]), (0.2, 0.4 * u.um, 0.4 * u.um))
        self.assertEqual((neuron["beta"], braincell["beta"], braincell["beta_o"]), (0.075, 0.075 / u.ms, 0.075 / u.ms))
        self.assertEqual((neuron["ki0"], neuron["ko0"]), (140.0, 5.0))
        self.assertEqual((braincell["C_rest"], braincell["Co_rest"]), (140.0 * u.mM, 5.0 * u.mM))

    def test_the_pump_has_the_same_parameters_on_both_sides(self):
        case = runner.load_case("ca1999/sodium_detailed_pump", "pool_cc")
        self.assertIn(DATA_ROOT / "modeldb" / "6763" / "pump.mod", case.mod_files)
        ((suffix, neuron),) = case.neuron_mechanisms
        neuron = {p.neuron: p.value for p in neuron}
        (braincell,) = [params for _, cls, params in case.braincell_mechanisms if cls == "NaKPump_Ca1999"]
        braincell = {p.braincell: (p.value, p.unit) for p in braincell}
        self.assertEqual(suffix, "pump")
        self.assertEqual(neuron, {"ipumpmax": 0.012, "km": 10.0, "n": 1.5})
        self.assertEqual(braincell, {"imax": (0.012, "mA/cm2"), "km": (10.0, "mM")})

    def test_a_diameter_fraction_is_a_spatial_callable(self):
        from types import SimpleNamespace

        import brainunit as u

        d = runner.Parameter(None, "d", runner.DIAM_FRACTION, 0.25).quantity()
        self.assertEqual(d(SimpleNamespace(diam_mid=15.0 * u.um)), 3.75 * u.um)


class RateTableTest(unittest.TestCase):
    def test_a_table_is_switched_off_and_restored(self):
        h = SimpleNamespace(usetable_hh=1.0)
        undo = runner._rate_table_off(h, "hh")
        self.assertEqual(h.usetable_hh, 0)
        undo()
        self.assertEqual(h.usetable_hh, 1.0)

    def test_a_mechanism_without_a_table_is_left_alone(self):
        h = SimpleNamespace()
        runner._rate_table_off(h, "pas")()
        self.assertEqual(vars(h), {})


class PoolToleranceTableTest(unittest.TestCase):
    def test_entries_name_listed_protocols_and_documented_causes(self):
        listed = {(case.config, case.protocol) for case in _cases()}
        listed |= {(c, Path(t).stem) for c in runner.pinned_config_names() for t in runner.load_config(c)["templates"]}
        for entry in TABLE.entries:
            with self.subTest(config=entry.config, protocol=entry.protocol):
                self.assertIn((entry.config, entry.protocol), listed)
                self.assertIn(entry.protocol, POOL_PROTOCOL_METRICS)
        self.assertEqual(TABLE.missing_sections(KNOWN_DIFFERENCES.read_text()), [])


if __name__ == "__main__":
    unittest.main()
