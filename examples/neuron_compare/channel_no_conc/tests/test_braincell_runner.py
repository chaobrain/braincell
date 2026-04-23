

import json
import os
import unittest

import numpy as np

from ._helpers import CHANNEL_NO_CONC_ROOT, MOD_VALIDATE_MOD_DIR, TEMPLATES_ROOT, build_case_payload, build_mapping_payload, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_braincell_test",
)
braincell_runner = load_module(
    TEMPLATES_ROOT / "braincell_runner.py",
    "channel_no_conc_braincell_runner_test",
)


class BraincellRunnerTest(unittest.TestCase):
    def _build_payload(self, *, stimulus: dict | None = None) -> dict:
        return build_case_payload(
            case_id="kv_braincell",
            config_name="kv_test",
            template_name="braincell",
            stimulus=stimulus or {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.2},
        )

    def test_run_case_returns_time_voltage_current_and_gates(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        result = braincell_runner.run_case(case)

        self.assertEqual(sorted(result.keys()), ["current", "gates", "time_ms", "voltage_mV"])
        self.assertEqual(sorted(result["gates"].keys()), ["n"])
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertEqual(result["gates"]["n"].shape, result["time_ms"].shape)

    def test_leak_enabled_does_not_break_output(self) -> None:
        payload = self._build_payload()
        payload["leak"] = {"enabled": True, "g_S_cm2": 1e-4, "e_mV": -65.0}
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = braincell_runner.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)

    def test_sine_stimulus_runs(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(
            self._build_payload(
                stimulus={
                    "kind": "sine",
                    "start_ms": 0.0,
                    "duration_ms": 2.0,
                    "amplitude_nA": 0.2,
                    "frequency_hz": 250.0,
                    "phase_rad": 0.0,
                    "offset_nA": 0.0,
                }
            )
        )
        result = braincell_runner.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_dc_stimulus_changes_voltage_relative_to_zero_amp_case(self) -> None:
        driven_case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        quiet_case = experiment_schema.ChannelNoConcCase.from_dict(
            self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        )
        driven = braincell_runner.run_case(driven_case)
        quiet = braincell_runner.run_case(quiet_case)
        self.assertGreater(np.max(np.abs(driven["voltage_mV"] - quiet["voltage_mV"])), 1e-6)

    def test_reset_state_initializes_gate_and_current_for_nondefault_v_init(self) -> None:
        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.02})
        payload["simulation"]["v_init_mV"] = 0.0
        payload["simulation"]["temperature_celsius"] = 10.0
        payload["channel_params"]["g_max_S_cm2"] = 0.001
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = braincell_runner.run_case(case)
        self.assertGreater(result["gates"]["n"][0], 0.01)
        self.assertGreater(abs(result["current"]["ix"][0]), 1e-3)

    def test_temperature_is_forwarded_to_temp_aware_channel(self) -> None:
        base = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        base["mapping"] = build_mapping_payload(
            impl_name={"common": "Kir2p3_MA2024_PC"},
            gate_names={"common": ["d"]},
        )
        base["channel_params"]["g_max_S_cm2"] = 0.0009
        base["simulation"]["v_init_mV"] = -50.0
        base["simulation"]["duration_ms"] = 10.0
        base["stimulus"]["dur_ms"] = 10.0

        cold_payload = json.loads(json.dumps(base))
        warm_payload = json.loads(json.dumps(base))
        cold_payload["simulation"]["temperature_celsius"] = 10.0
        warm_payload["simulation"]["temperature_celsius"] = 30.0

        cold_case = experiment_schema.ChannelNoConcCase.from_dict(cold_payload)
        warm_case = experiment_schema.ChannelNoConcCase.from_dict(warm_payload)

        cold = braincell_runner.run_case(cold_case)
        warm = braincell_runner.run_case(warm_case)

        self.assertGreater(
            float(np.max(np.abs(cold["gates"]["d"] - warm["gates"]["d"]))),
            1e-4,
        )
        self.assertGreater(
            float(np.max(np.abs(cold["voltage_mV"] - warm["voltage_mV"]))),
            1e-3,
        )

    def test_pure_channel_current_uses_mechanism_probe_without_ion_state(self) -> None:
        payload = build_case_payload(
            case_id="ih_braincell",
            config_name="ih_test",
            template_name="braincell",
            identity={"mod_dir": MOD_VALIDATE_MOD_DIR},
            mapping=build_mapping_payload(
                current="ih",
                impl_name={"common": "HCN_HM1992"},
                gate_names={"common": ["p"]},
                channel_params={
                    "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
                    "E_mV": {"neuron": "eh", "braincell": "E"},
                },
            ),
            stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
        )
        payload.pop("ion_state", None)
        payload["channel_params"]["g_max_S_cm2"] = 0.001
        payload["channel_params"]["E_mV"] = -43.0
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)

        result = braincell_runner.run_case(case)

        self.assertEqual(sorted(result["gates"].keys()), ["p"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc" / "hcn1_ma24_pc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_bc_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc" / "hcn1_ma25_bc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_goc_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "hcn1_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["o_fast", "o_slow"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_dcn_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["m"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_sc_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "hcn1_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_dcn_smoke_case_runs(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = braincell_runner.run_case(case)

        self.assertIsNone(case.ion_state)
        self.assertEqual(sorted(result["gates"].keys()), ["m"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
