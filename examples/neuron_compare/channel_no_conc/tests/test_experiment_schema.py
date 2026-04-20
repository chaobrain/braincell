from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from ._helpers import (
    CHANNEL_NO_CONC_ROOT,
    TEMPLATES_ROOT,
    build_case_payload,
    build_main_config_payload,
    build_mapping_payload,
    build_scan_template_payload,
    load_module,
    write_json,
)


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_test",
)


class ExperimentSchemaTest(unittest.TestCase):
    def test_builds_minimal_case(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(build_case_payload())
        self.assertAlmostEqual(case.morphology.radius_um, 50.0 / 3.141592653589793, places=12)
        self.assertEqual(case.mapping_spec.neuron.mechanism_name, "Kv")
        self.assertEqual(case.mapping_spec.current_source.ion_name, "k")
        self.assertEqual(case.ion_state.E_mV, -80.0)
        self.assertEqual(case.channel_params["g_max_S_cm2"], 0.0)

    def test_rejects_legacy_pair_id_shape(self) -> None:
        payload = build_case_payload()
        payload.pop("identity")
        payload.pop("mapping")
        payload["pair_id"] = "kv_test"
        payload["mod_dir"] = "/tmp/mods"
        with self.assertRaisesRegex(ValueError, "Legacy channel_no_conc case schema"):
            experiment_schema.ChannelNoConcCase.from_dict(payload)

    def test_rejects_non_integer_number_of_dt_steps(self) -> None:
        payload = build_case_payload()
        payload["simulation"]["duration_ms"] = 0.11
        payload["stimulus"]["dur_ms"] = 0.11
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            experiment_schema.ChannelNoConcCase.from_dict(payload)

    def test_rejects_unknown_channel_override(self) -> None:
        payload = build_case_payload()
        payload["channel_params"]["bad_param"] = 1.0
        with self.assertRaisesRegex(ValueError, "mapping.channel_params"):
            experiment_schema.ChannelNoConcCase.from_dict(payload)

    def test_load_model_config_resolves_relative_template_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json"]),
            )

            model_config = experiment_schema.load_model_config(config_path)

        self.assertEqual(model_config.config_name, "kv_test")
        self.assertEqual(model_config.template_paths, (template_path.resolve(),))
        self.assertEqual(model_config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gbar")
        self.assertEqual(model_config.defaults, {})

    def test_load_model_config_accepts_defaults_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(
                    ["templates/smoke.json"],
                    defaults={"channel_params": {"g_max_S_cm2": 0.002}},
                ),
            )

            model_config = experiment_schema.load_model_config(config_path)

        self.assertEqual(model_config.defaults, {"channel_params": {"g_max_S_cm2": 0.002}})

    def test_load_scan_template_builds_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_path = write_json(
                Path(tmpdir) / "vinit_scan.json",
                build_scan_template_payload(
                    group_id="v_init_scan",
                    sweep_axes={"simulation.v_init_mV": [-70.0, -50.0, -30.0]},
                ),
            )
            template = experiment_schema.load_scan_template(template_path)

        self.assertEqual(template.template_name, "vinit_scan")
        self.assertEqual(template.group_id, "v_init_scan")
        self.assertEqual(template.raw_sweep_axes["simulation.v_init_mV"], (-70.0, -50.0, -30.0))

    def test_load_sweep_config_builds_run_id_from_config_and_template_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(template_dir / "smoke.json", build_scan_template_payload())
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/smoke.json"]),
            )

            config = experiment_schema.load_sweep_config(config_path, "templates/smoke.json")

        self.assertEqual(config.config_id, "kv_test__smoke")
        self.assertEqual(config.group.group_id, "smoke")
        self.assertEqual(config.base_case["identity"]["mod_dir"], "/home/swl/braincell/examples/convert_mod/mod_validate/mods")

    def test_load_sweep_config_rejects_template_not_declared_in_main_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            write_json(template_dir / "declared.json", build_scan_template_payload())
            undeclared_path = write_json(template_dir / "other.json", build_scan_template_payload())
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/declared.json"]),
            )

            with self.assertRaisesRegex(ValueError, "not declared in config.templates"):
                experiment_schema.load_sweep_config(config_path, undeclared_path)

    def test_expand_cases_generates_group_cartesian_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "grid.json",
                build_scan_template_payload(
                    group_id="grid",
                    sweep_axes={
                        "simulation.v_init_mV": [-65.0, -50.0],
                        "simulation.temperature_celsius": [20.0, 25.0],
                    },
                ),
            )
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/grid.json"]),
            )
            config = experiment_schema.load_sweep_config(config_path, template_path)

        expanded = experiment_schema.expand_cases(config)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(expanded[0]["case_id"], "grid__000")
        self.assertEqual(expanded[-1]["case_id"], "grid__003")

    def test_config_defaults_merge_into_template_base(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "grid.json",
                build_scan_template_payload(
                    base={
                        "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
                        "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
                        "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
                        "ion_state": {"E_mV": -80.0},
                    },
                ),
            )
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(
                    ["templates/grid.json"],
                    defaults={"channel_params": {"g_max_S_cm2": 0.002}},
                ),
            )
            config = experiment_schema.load_sweep_config(config_path, template_path)

        expanded = experiment_schema.expand_cases(config)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.002)

    def test_template_base_overrides_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "grid.json",
                build_scan_template_payload(
                    base={
                        "morphology": {"length_um": 10.0, "diam_um": 100.0 / 3.141592653589793, "cm_uF_cm2": 1.0},
                        "simulation": {"dt_ms": 0.025, "duration_ms": 2.0, "v_init_mV": -65.0, "temperature_celsius": 25.0},
                        "stimulus": {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0},
                        "ion_state": {"E_mV": -80.0},
                        "channel_params": {"g_max_S_cm2": 0.0},
                    },
                ),
            )
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(
                    ["templates/grid.json"],
                    defaults={"channel_params": {"g_max_S_cm2": 0.002}},
                ),
            )
            config = experiment_schema.load_sweep_config(config_path, template_path)

        expanded = experiment_schema.expand_cases(config)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.0)

    def test_repo_vinit_celsius_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "kv_test.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(template.group_id, "vinit_celsius")
        self.assertEqual(len(expanded), 9)
        self.assertEqual(expanded[0]["stimulus"], {"kind": "dc", "delay_ms": 0.0, "dur_ms": 10.0, "amp_nA": 0.0})
        self.assertEqual(expanded[0]["simulation"]["v_init_mV"], -80.0)
        self.assertEqual(expanded[0]["simulation"]["temperature_celsius"], 22.0)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)
        self.assertEqual(expanded[-1]["simulation"]["v_init_mV"], -50.0)
        self.assertEqual(expanded[-1]["simulation"]["temperature_celsius"], 37.0)

    def test_repo_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "kv_test.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(template.group_id, "dc")
        self.assertEqual(len(expanded), 3)
        self.assertEqual([case["stimulus"]["amp_nA"] for case in expanded], [-0.05, 0.0, 0.05])
        self.assertEqual(expanded[0]["simulation"]["v_init_mV"], -65.0)
        self.assertEqual(expanded[0]["simulation"]["temperature_celsius"], 25.0)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)

    def test_repo_ac_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "kv_test.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "ac.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(template.group_id, "ac")
        self.assertEqual(len(expanded), 3)
        self.assertEqual([case["stimulus"]["frequency_hz"] for case in expanded], [50.0, 100.0, 250.0])
        self.assertEqual(expanded[0]["stimulus"]["kind"], "sine")
        self.assertEqual(expanded[0]["stimulus"]["amplitude_nA"], 0.02)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)

    def test_repo_hcn_vinit_template_expands_pure_channel_case(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc" / "hcn1_ma24_pc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(len(expanded), 9)
        self.assertNotIn("ion_state", expanded[0])
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -34.4)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.0001)
        self.assertEqual(expanded[0]["simulation"]["v_init_mV"], -80.0)
        self.assertEqual(expanded[0]["simulation"]["temperature_celsius"], 22.0)

    def test_repo_kir2p3_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc" / "kir2p3_ma24_pc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(template.group_id, "dc")
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gkbar")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("d",))
        self.assertEqual(len(expanded), 3)
        self.assertEqual([case["stimulus"]["amp_nA"] for case in expanded], [-0.05, 0.0, 0.05])
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)

    def test_repo_hcn_bc_vinit_template_expands_pure_channel_case(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc" / "hcn1_ma25_bc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(len(expanded), 9)
        self.assertNotIn("ion_state", expanded[0])
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -34.4)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.0001)

    def test_repo_kv1p1_bc_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc" / "kv1p1_ma25_bc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.ion_name, "k")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("n",))
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gbar")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.004)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)

    def test_repo_hcn1_goc_vinit_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "hcn1_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertIsNone(config.mapping_spec.current_source.ion_name)
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("o_fast", "o_slow"))
        self.assertEqual(len(expanded), 9)
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -20.0)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00005)

    def test_repo_cahva_goc_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "cahva_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.ion_name, "ca")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("s", "u"))
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gcabar")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00046)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], 120.0)

    def test_repo_hcn_dcn_vinit_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertIsNone(config.mapping_spec.current_source.ion_name)
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("m",))
        self.assertEqual(len(expanded), 9)
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -45.0)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00001)

    def test_repo_naf_dcn_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "naf_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.ion_name, "na")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("m", "h"))
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gbar")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00001)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], 50.0)

    def test_repo_hcn_sc_vinit_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "hcn1_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertIsNone(config.mapping_spec.current_source.ion_name)
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("h",))
        self.assertEqual(len(expanded), 9)
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -34.4)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.0001)

    def test_repo_km_sc_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "km_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.ion_name, "k")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("n",))
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gkbar")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00025)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], -80.0)

    def test_repo_hcn_dcn_vinit_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertIsNone(config.mapping_spec.current_source.ion_name)
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("m",))
        self.assertEqual(len(expanded), 9)
        self.assertEqual(expanded[0]["channel_params"]["E_mV"], -45.0)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00001)

    def test_repo_naf_dcn_dc_template_expands_expected_cases(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "naf_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "dc.json"

        model_config = experiment_schema.load_model_config(config_path)
        template = experiment_schema.load_scan_template(template_path)
        config = experiment_schema.build_sweep_config(model_config, template)
        expanded = experiment_schema.expand_cases(config)

        self.assertEqual(config.mapping_spec.current_source.ion_name, "na")
        self.assertEqual(config.mapping_spec.neuron.gate_names, ("m", "h"))
        self.assertEqual(config.mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gbar")
        self.assertEqual(len(expanded), 3)
        self.assertEqual(expanded[0]["channel_params"]["g_max_S_cm2"], 0.00001)
        self.assertEqual(expanded[0]["ion_state"]["E_mV"], 50.0)

    def test_load_sweep_config_rejects_unsupported_sweep_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            template_dir = root / "templates"
            template_dir.mkdir()
            template_path = write_json(
                template_dir / "bad.json",
                build_scan_template_payload(
                    sweep_axes={"stimulus.frequency_hz": [100.0]},
                ),
            )
            config_path = write_json(
                root / "kv_test.json",
                build_main_config_payload(["templates/bad.json"]),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported sweep path"):
                experiment_schema.load_sweep_config(config_path, template_path)

    def test_load_model_config_rejects_legacy_sweep_shape(self) -> None:
        payload = {
            "config_id": "legacy",
            "base_case": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "legacy.json"
            config_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Legacy channel_no_conc config schema"):
                experiment_schema.load_model_config(config_path)

    def test_mapping_specific_param_must_be_declared_in_mapping(self) -> None:
        mapping = build_mapping_payload(channel_params={"g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"}})
        payload = build_case_payload(mapping=mapping)
        payload["channel_params"]["alpha_shift_mV"] = 1.0
        with self.assertRaisesRegex(ValueError, "mapping.channel_params"):
            experiment_schema.ChannelNoConcCase.from_dict(payload)

    def test_ion_state_defaults_when_missing_for_known_ion(self) -> None:
        payload = build_case_payload()
        payload.pop("ion_state", None)

        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        normalized = experiment_schema.case_to_payload(case)

        self.assertEqual(case.ion_state.E_mV, -80.0)
        self.assertEqual(normalized["ion_state"]["E_mV"], -80.0)

    def test_pure_channel_case_does_not_require_ion_state(self) -> None:
        mapping = build_mapping_payload(
            current="ih",
            impl_name={"common": "Ih_HM1992"},
            gate_names={"common": ["p"]},
            channel_params={
                "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
                "E_mV": {"neuron": "eh", "braincell": "E"},
            },
        )
        payload = build_case_payload(mapping=mapping)
        payload.pop("ion_state", None)
        payload["channel_params"]["E_mV"] = -43.0

        case = experiment_schema.ChannelNoConcCase.from_dict(payload)

        self.assertIsNone(case.ion_state)
        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(case.channel_params["E_mV"], -43.0)

    def test_pure_channel_case_allows_empty_channel_params(self) -> None:
        mapping = build_mapping_payload(
            current="ih",
            impl_name={"common": "Ih_HM1992"},
            gate_names={"common": ["p"]},
            channel_params={
                "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
                "E_mV": {"neuron": "eh", "braincell": "E"},
            },
        )
        payload = build_case_payload(mapping=mapping)
        payload["channel_params"] = {}

        case = experiment_schema.ChannelNoConcCase.from_dict(payload)

        self.assertIsNone(case.ion_state)
        self.assertEqual(case.channel_params, {})

    def test_pure_channel_case_rejects_ion_state(self) -> None:
        mapping = build_mapping_payload(
            current="ih",
            impl_name={"common": "Ih_HM1992"},
            gate_names={"common": ["p"]},
            channel_params={
                "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
                "E_mV": {"neuron": "eh", "braincell": "E"},
            },
        )
        payload = build_case_payload(mapping=mapping)
        payload["ion_state"] = {"E_mV": -43.0}

        with self.assertRaisesRegex(ValueError, "ion_state is only valid"):
            experiment_schema.ChannelNoConcCase.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
