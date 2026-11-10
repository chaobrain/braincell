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

"""Catalogue comparison with NEURON: the calcium and calcium-activated potassium channels.

Every configuration of a class in ``braincell.channel.calcium`` or
``braincell.channel.potassium_calcium`` runs the three current-clamp templates
it lists and both voltage-clamp templates; the limits come from
``tolerances.py`` (T0 unless an entry raises them), leaving out the commands
listed in ``NON_FINITE_REFERENCE``. The ``VARIANTS`` of these
classes run the same five protocols with their constructor overrides. Two
tests re-run the root-cause patches of this family's voltage-clamp T1 entries,
which must bring the configuration back to T0.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import unittest
from unittest import mock

import brainunit as u
import numpy as np
import pytest

import braincell.channel.calcium as calcium
import braincell.channel.potassium_calcium as potassium_calcium
from braincell.mech import get_registry
from validation.neuron import _clamp
from validation.neuron._tolerance import CURRENT_CLAMP_PROTOCOLS, VOLTAGE_CLAMP_PROTOCOLS
from validation.neuron.channel_no_conc.tolerances import NON_FINITE_REFERENCE, TABLE, UNCOVERED, VARIANTS

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, load_module

os.environ.setdefault("JAX_PLATFORMS", "cpu")

pytestmark = pytest.mark.slow

experiment_schema = load_module(TEMPLATES_ROOT / "experiment_schema.py", "channel_no_conc_schema_for_calcium")
compare_module = load_module(TEMPLATES_ROOT / "compare.py", "channel_no_conc_compare_for_calcium")
voltage_clamp = load_module(TEMPLATES_ROOT / "voltage_clamp.py", "channel_no_conc_voltage_clamp_for_calcium")
braincell_runner = load_module(TEMPLATES_ROOT / "braincell_runner.py", "channel_no_conc_runner_for_calcium")

FAMILY = frozenset(calcium.__all__) | frozenset(potassium_calcium.__all__)
"""Registry names of the classes in the two modules."""


def _config_path(config: str) -> Path:
    return CHANNEL_NO_CONC_ROOT / "configs" / f"{config}.json"


def _template_path(protocol: str) -> Path:
    return CHANNEL_NO_CONC_ROOT / "templates" / f"{protocol}.json"


def _class_name(config: str) -> str:
    return experiment_schema.load_model_config(_config_path(config)).mapping_spec.braincell.class_name


CONFIGS = tuple(
    sorted(
        str(path.relative_to(CHANNEL_NO_CONC_ROOT / "configs").with_suffix(""))
        for path in (CHANNEL_NO_CONC_ROOT / "configs").rglob("*.json")
        if _class_name(str(path.relative_to(CHANNEL_NO_CONC_ROOT / "configs").with_suffix(""))) in FAMILY
    )
)
FAMILY_VARIANTS = {name: spec for name, spec in VARIANTS.items() if spec[0] in CONFIGS}


@contextmanager
def _overrides(config: str, overrides: dict):
    """Pass ``overrides`` to every construction of the configuration's BrainCell class."""
    cls = get_registry().get("channel", _class_name(config))
    original = cls.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **{**kwargs, **overrides})

    with mock.patch.object(cls, "__init__", init):
        yield


def _current_clamp_error(config: str, protocol: str) -> float:
    sweep = experiment_schema.load_sweep_config(_config_path(config), _template_path(protocol))
    errors = [
        compare_module.compare_case(experiment_schema.ChannelNoConcCase.from_dict(payload))["metrics"]["voltage"]["max_abs"]
        for payload in experiment_schema.expand_cases(sweep)
    ]
    return float(np.max(errors))  # NaN propagates and fails the assertion


def _skip_unless_compiled(test: unittest.TestCase) -> None:
    missing = sorted(
        {
            str(Path(experiment_schema.load_model_config(_config_path(config)).mod_dir))
            for config in CONFIGS
            if not (Path(experiment_schema.load_model_config(_config_path(config)).mod_dir) / "x86_64" / "libnrnmech.so").exists()
        }
    )
    if missing:
        test.skipTest(f"NEURON mechanisms are not compiled: {missing}.")


class CalciumCatalogueTest(unittest.TestCase):
    def test_family_is_configured_or_listed(self):
        configured = {_class_name(config) for config in CONFIGS}
        uncovered = FAMILY & set(UNCOVERED)
        self.assertEqual(len(FAMILY), 50)
        self.assertEqual(len(CONFIGS), 45)
        self.assertEqual(configured | uncovered, FAMILY)
        self.assertEqual(
            uncovered,
            {"AHP_De1994", "CaHT_HM1992", "CaHT_Re1993", "CaL_IS2008", "CaN_IS2008", "CaT_HM1992", "CaT_HP1992"},
        )

    def _skipped_commands(self, config: str, protocol: str) -> tuple[int, ...]:
        """Indices of the commands with a non-finite NEURON run; checks they are still not finite."""
        if (config, protocol) not in NON_FINITE_REFERENCE:
            return ()
        test_mV, _ = NON_FINITE_REFERENCE[(config, protocol)]
        case, commands = voltage_clamp.load_protocol(_config_path(config), _template_path(protocol))
        index = json.loads(_template_path(protocol).read_text())["voltage_clamp"]["test_mV"].index(test_mV)
        neuron = voltage_clamp.run_neuron_command(case, commands[index])
        self.assertFalse(np.all(np.isfinite(neuron["ix"])), (config, protocol, test_mV))
        return (index,)

    def _assert_protocols(self, name: str, config: str) -> dict[str, float]:
        templates = experiment_schema.load_model_config(_config_path(config)).template_paths
        self.assertEqual({Path(path).stem for path in templates}, set(CURRENT_CLAMP_PROTOCOLS), config)
        measured = {}
        for protocol in CURRENT_CLAMP_PROTOCOLS:
            with self.subTest(config=name, protocol=protocol):
                measured[protocol] = _current_clamp_error(config, protocol)
                self.assertLessEqual(measured[protocol], TABLE.limit(name, protocol, "e_V"), f"{measured[protocol]:.3e} mV")
        for protocol in VOLTAGE_CLAMP_PROTOCOLS:
            with self.subTest(config=name, protocol=protocol):
                skip = self._skipped_commands(config, protocol)
                result = voltage_clamp.compare_protocol(_config_path(config), _template_path(protocol), skip_commands=skip)
                # The clamp follows its command; calva_su15_dcn's current holds it 3.3e-6 mV off.
                # BrainCell replays NEURON's voltage, so this does not enter e_I or e_s.
                self.assertLessEqual(result["clamp_error_mV"], 1e-5)
                for metric in ("e_I", "e_s"):
                    self.assertLessEqual(result[metric], TABLE.limit(name, protocol, metric), f"{metric}: {result[metric]:.3e}")
        return measured

    def test_every_configuration_stays_within_its_limits(self):
        _skip_unless_compiled(self)
        for config in CONFIGS:
            self._assert_protocols(config, config)

    def test_default_off_options_stay_within_their_limits(self):
        _skip_unless_compiled(self)
        self.assertEqual(set(FAMILY_VARIANTS), {"zh19_io/ca_zh19_io:freeze_m_inf=False"})
        for name, (config, overrides) in FAMILY_VARIANTS.items():
            with _overrides(config, overrides):
                measured = self._assert_protocols(name, config)
            # The override reaches the channel: each T1 protocol leaves T0.
            for protocol in CURRENT_CLAMP_PROTOCOLS:
                if TABLE.tier(name, protocol) == "T1":
                    self.assertGreater(measured[protocol], TABLE.limit(config, protocol, "e_V"), (name, protocol))


class CalciumRootCauseTest(unittest.TestCase):
    """Patching only the named cause brings a T1 configuration back to T0."""

    def test_cav3p3_mod_ghk_formula_explains_the_ramp(self):
        config = "ma24_pc/cav3p3_ma24_pc"
        _skip_unless_compiled(self)

        def mod_ghk(V, ci, co, z, temp):
            # The .mod's w/(exp(w) - 1) without BrainCell's small-w branch.
            faraday, gas = calcium._CAV3P3_NMODL_FARADAY, calcium._CAV3P3_NMODL_GAS_CONSTANT
            w = (z * faraday * V) / (gas * (temp + calcium._CAV3P3_NMODL_TEMP_OFFSET))
            return -z * faraday * (co - ci * u.math.exp(w)) * w / (u.math.exp(w) - 1)

        self.assertEqual(TABLE.tier(config, "vc_ramp"), "T1")
        with mock.patch.object(calcium.Cav3p3_RI2021_SC, "ghk", staticmethod(mod_ghk)):
            result = voltage_clamp.compare_protocol(_config_path(config), _template_path("vc_ramp"))
        self.assertLessEqual(result["e_I"], 1e-10)

    def test_kca2p2_exact_initial_state_explains_the_steps(self):
        config = "ma20_goc/kca2p2_ma20_goc"
        _skip_unless_compiled(self)
        case, commands = voltage_clamp.load_protocol(_config_path(config), _template_path("vc_steps"))
        # BrainCell starts at the exact steady state; a constant clamp keeps it there.
        held = braincell_runner.run_case(case, solver=_clamp.voltage_replay_solver(np.full(commands[0].size + 1, commands[0][0])))
        steady = {gate.neuron: float(held["gates"][gate.braincell][0]) for gate in case.mapping_spec.gate_map}

        def initialize(h, segment, **kwargs):
            _clamp.initialize_neuron(h, segment, **kwargs)
            mechanism = getattr(segment, case.mapping_spec.neuron.mechanism_name)
            for state, value in {**steady, "c1": 1.0 - sum(steady.values())}.items():
                setattr(mechanism, state, value)

        self.assertEqual(TABLE.tier(config, "vc_steps"), "T1")
        result = voltage_clamp.compare_protocol(_config_path(config), _template_path("vc_steps"), initialize=initialize)
        self.assertLessEqual(result["e_I"], 1e-10)
        self.assertLessEqual(result["e_s"], 1e-10)


if __name__ == "__main__":
    unittest.main()
