

import os
import unittest

import numpy as np

from ._helpers import CHANNEL_NO_CONC_ROOT, TEMPLATES_ROOT, build_case_payload, build_mapping_payload, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


experiment_schema = load_module(
    TEMPLATES_ROOT / "experiment_schema.py",
    "channel_no_conc_experiment_schema_for_neuron_test",
)
neuron_runner = load_module(
    TEMPLATES_ROOT / "neuron_runner.py",
    "channel_no_conc_neuron_runner_test",
)


class NeuronRunnerTest(unittest.TestCase):
    _GOC_LIBNRNMECH = CHANNEL_NO_CONC_ROOT / ".." / "Cerebellum_mod" / "GoC" / "channel" / "x86_64" / "libnrnmech.so"
    _BC_LIBNRNMECH = CHANNEL_NO_CONC_ROOT / ".." / "Cerebellum_mod" / "BC" / "channel" / "x86_64" / "libnrnmech.so"

    def _build_payload(self, *, stimulus: dict | None = None) -> dict:
        return build_case_payload(
            case_id="kv_smoke",
            config_name="kv_test",
            template_name="neuron",
            stimulus=stimulus or {"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.2},
        )

    def test_run_case_returns_time_voltage_current_and_gates(self) -> None:
        case = experiment_schema.ChannelNoConcCase.from_dict(self._build_payload())
        result = neuron_runner.run_case(case)

        self.assertEqual(sorted(result.keys()), ["current", "gates", "ion_state", "time_ms", "voltage_mV"])
        self.assertEqual(sorted(result["gates"].keys()), ["n"])
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertEqual(result["gates"]["n"].shape, result["time_ms"].shape)
        self.assertEqual(result["ion_state"], {})
        self.assertEqual(len(result["time_ms"]), 80)
        self.assertAlmostEqual(result["time_ms"][0], 0.025, places=12)
        self.assertAlmostEqual(result["time_ms"][-1], 2.0, places=12)

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
        result = neuron_runner.run_case(case)
        self.assertEqual(result["time_ms"].shape, result["voltage_mV"].shape)
        self.assertTrue(np.isfinite(result["voltage_mV"]).all())

    def test_dc_stimulus_changes_voltage_relative_to_zero_amp_case(self) -> None:
        driven = neuron_runner.run_case(experiment_schema.ChannelNoConcCase.from_dict(self._build_payload()))
        quiet = neuron_runner.run_case(
            experiment_schema.ChannelNoConcCase.from_dict(
                self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
            )
        )
        self.assertGreater(np.max(np.abs(driven["voltage_mV"] - quiet["voltage_mV"])), 1e-6)

    def test_run_case_respects_nondefault_v_init_under_h_run(self) -> None:
        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        payload["simulation"]["v_init_mV"] = 0.0
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = neuron_runner.run_case(case)
        self.assertGreater(result["voltage_mV"][0], -1.0)
        self.assertLess(result["voltage_mV"][0], 1.0)

    def test_current_ix_uses_braincell_sign_convention(self) -> None:
        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.02})
        payload["simulation"]["v_init_mV"] = 0.0
        payload["simulation"]["temperature_celsius"] = 10.0
        payload["channel_params"]["g_max_S_cm2"] = 0.001
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = neuron_runner.run_case(case)
        self.assertLess(result["current"]["ix"][0], 0.0)

    def test_resolve_neuron_current_ref_prefers_standard_ion_currents(self) -> None:
        class SegmentStub:
            _ref_ik = object()

        ref = neuron_runner._resolve_neuron_current_ref(SegmentStub(), current_var="ik")
        self.assertIs(ref, SegmentStub._ref_ik)

    def test_resolve_neuron_current_ref_missing_field_raises(self) -> None:
        class SegmentStub:
            pass

        with self.assertRaisesRegex(AttributeError, "_ref_ik"):
            neuron_runner._resolve_neuron_current_ref(SegmentStub(), current_var="ik")

    def test_resolve_neuron_current_ref_supports_custom_current_var(self) -> None:
        class SegmentStub:
            _ref_ih = object()

        ref = neuron_runner._resolve_neuron_current_ref(SegmentStub(), current_var="ih")
        self.assertIs(ref, SegmentStub._ref_ih)

    def test_resolve_neuron_current_ref_supports_segment_mechanism_suffixed_ref(self) -> None:
        class SegmentStub:
            _ref_ih_HCN1_MA2020_GoC = object()

        ref = neuron_runner._resolve_neuron_current_ref(
            SegmentStub(),
            mechanism_name="HCN1_MA2020_GoC",
            current_var="ih",
        )
        self.assertIs(ref, SegmentStub._ref_ih_HCN1_MA2020_GoC)

    def test_resolve_neuron_current_ref_supports_mechanism_local_ref(self) -> None:
        class SegmentStub:
            pass

        class MechanismStub:
            _ref_ih = object()

        ref = neuron_runner._resolve_neuron_current_ref(
            SegmentStub(),
            mech_obj=MechanismStub(),
            mechanism_name="HCN2_MA2020_GoC",
            current_var="ih",
        )
        self.assertIs(ref, MechanismStub._ref_ih)

    def test_initialize_neuron_ion_state_sets_concentrations_and_calls_frecord_init(self) -> None:
        class HStub:
            def __init__(self):
                self.called = 0

            def frecord_init(self):
                self.called += 1

        class SegmentStub:
            pass

        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        payload["mapping"] = build_mapping_payload(current_kind="ca")
        payload["ion_state"] = {"Ci_mM": 2.4e-4, "Co_mM": 2.0}
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        h = HStub()
        seg = SegmentStub()

        neuron_runner._initialize_neuron_ion_state(
            h=h,
            section=None,
            segment=seg,
            case=case,
            mapping_spec=case.mapping_spec,
        )

        self.assertEqual(seg.cai, 2.4e-4)
        self.assertEqual(seg.cao, 2.0)
        self.assertEqual(h.called, 1)

    def test_initialize_neuron_ion_state_supports_cal_concentration_fields(self) -> None:
        class HStub:
            def __init__(self):
                self.called = 0

            def frecord_init(self):
                self.called += 1

        class SegmentStub:
            pass

        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 2.0, "amp_nA": 0.0})
        payload["mapping"] = build_mapping_payload(current_kind="cal")
        payload["ion_state"] = {"Ci_mM": 2.4e-4, "Co_mM": 2.0}
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        h = HStub()
        seg = SegmentStub()

        neuron_runner._initialize_neuron_ion_state(
            h=h,
            section=None,
            segment=seg,
            case=case,
            mapping_spec=case.mapping_spec,
        )

        self.assertEqual(seg.cali, 2.4e-4)
        self.assertEqual(seg.calo, 2.0)
        self.assertEqual(h.called, 1)

    def test_resolve_neuron_parameter_target_prefers_mechanism_when_only_mechanism_has_field(self) -> None:
        class SomaStub:
            pass

        class MechanismStub:
            gbar = 1.0

        target = neuron_runner._resolve_neuron_parameter_target(
            soma=SomaStub(),
            mech_obj=MechanismStub(),
            attr_name="gbar",
        )
        self.assertIsInstance(target, MechanismStub)

    def test_resolve_neuron_parameter_target_uses_section_when_only_section_has_field(self) -> None:
        class SomaStub:
            eh = -34.4

        class MechanismStub:
            pass

        target = neuron_runner._resolve_neuron_parameter_target(
            soma=SomaStub(),
            mech_obj=MechanismStub(),
            attr_name="eh",
        )
        self.assertIsInstance(target, SomaStub)

    def test_resolve_neuron_parameter_target_rejects_ambiguous_field(self) -> None:
        class SomaStub:
            eh = -34.4

        class MechanismStub:
            eh = -34.4

        with self.assertRaisesRegex(ValueError, "Ambiguous NEURON parameter target"):
            neuron_runner._resolve_neuron_parameter_target(
                soma=SomaStub(),
                mech_obj=MechanismStub(),
                attr_name="eh",
            )

    def test_resolve_neuron_parameter_target_rejects_missing_field(self) -> None:
        class SomaStub:
            pass

        class MechanismStub:
            pass

        with self.assertRaisesRegex(ValueError, "Could not resolve NEURON parameter target"):
            neuron_runner._resolve_neuron_parameter_target(
                soma=SomaStub(),
                mech_obj=MechanismStub(),
                attr_name="missing",
            )

    def test_run_case_respects_nondefault_dt_under_h_run(self) -> None:
        payload = self._build_payload(stimulus={"kind": "dc", "delay_ms": 0.0, "dur_ms": 1.0, "amp_nA": 0.0})
        payload["simulation"]["dt_ms"] = 0.1
        payload["simulation"]["duration_ms"] = 1.0
        case = experiment_schema.ChannelNoConcCase.from_dict(payload)
        result = neuron_runner.run_case(case)
        self.assertTrue(np.allclose(np.diff(result["time_ms"]), 0.1))
        self.assertAlmostEqual(result["time_ms"][0], 0.1, places=12)
        self.assertAlmostEqual(result["time_ms"][-1], 1.0, places=12)

    def test_repo_hcn_smoke_case_runs_with_ih_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma24_pc" / "hcn1_ma24_pc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_bc_smoke_case_runs_with_ih_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc" / "hcn1_ma25_bc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_goc_smoke_case_runs_with_ih_current(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "hcn1_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["o_fast", "o_slow"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_kca3p1_goc_smoke_case_runs_with_ik_current(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "kca3p1_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ik")
        self.assertEqual(sorted(result["gates"].keys()), ["Y"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_kca2p2_goc_smoke_case_runs_with_ik_current(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "kca2p2_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ik")
        self.assertEqual(sorted(result["gates"].keys()), ["c2", "c3", "c4", "o1", "o2"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())

    def test_repo_kca1p1_goc_smoke_case_runs_with_ik_current(self) -> None:
        if not self._GOC_LIBNRNMECH.resolve().exists():
            self.skipTest("GoC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma20_goc" / "kca1p1_ma20_goc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[-1]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ik")
        self.assertEqual(
            sorted(result["gates"].keys()),
            ["C1", "C2", "C3", "C4", "O0", "O1", "O2", "O3", "O4"],
        )
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())

    def test_repo_nav1p6_bc_smoke_case_runs_with_ina_current(self) -> None:
        if not self._BC_LIBNRNMECH.resolve().exists():
            self.skipTest("BC NEURON mechanisms are not compiled into libnrnmech.so in the current environment.")
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ma25_bc" / "nav1p6_ma25_bc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ina")
        self.assertEqual(
            tuple(result["gates"]),
            ("C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B"),
        )
        self.assertTrue(all(trace.shape == result["time_ms"].shape for trace in result["gates"].values()))
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())
        self.assertGreaterEqual(float(np.max(np.abs(result["current"]["ix"]))), 0.0)

    def test_repo_hcn_dcn_smoke_case_runs_with_ih_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["m"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_hcn_sc_smoke_case_runs_with_ih_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "hcn1_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["h"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)

    def test_repo_cav2p1_sc_smoke_case_runs_with_ica_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "cav2p1_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ica")
        self.assertEqual(case.ion_state.Ci_mM, 2.4e-4)
        self.assertEqual(case.ion_state.Co_mM, 2.0)
        self.assertEqual(sorted(result["gates"].keys()), ["m"])
        self.assertEqual(sorted(result["ion_state"].keys()), ["ci_mM", "co_mM", "eca_mV"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        for trace in result["ion_state"].values():
            self.assertEqual(trace.shape, result["time_ms"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())

    def test_repo_cav3p2_sc_smoke_case_runs_with_ica_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "cav3p2_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ica")
        self.assertEqual(case.ion_state.Ci_mM, 2.4e-4)
        self.assertEqual(case.ion_state.Co_mM, 2.0)
        self.assertEqual(sorted(result["gates"].keys()), ["h", "m"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())

    def test_repo_cav3p3_sc_smoke_case_runs_with_ica_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "ri21_sc" / "cav3p3_ri21_sc.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ica")
        self.assertEqual(case.ion_state.Ci_mM, 2.4e-4)
        self.assertEqual(case.ion_state.Co_mM, 2.0)
        self.assertEqual(sorted(result["gates"].keys()), ["l", "n"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertTrue(np.isfinite(result["current"]["ix"]).all())

    def test_repo_hcn_dcn_smoke_case_runs_with_ih_current(self) -> None:
        config_path = CHANNEL_NO_CONC_ROOT / "configs" / "su15_dcn" / "hcn_su15_dcn.json"
        template_path = CHANNEL_NO_CONC_ROOT / "templates" / "vinit_celsius.json"
        config = experiment_schema.load_sweep_config(config_path, template_path)
        case_payload = experiment_schema.expand_cases(config)[0]
        case = experiment_schema.ChannelNoConcCase.from_dict(case_payload)

        result = neuron_runner.run_case(case)

        self.assertEqual(case.mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(sorted(result["gates"].keys()), ["m"])
        self.assertEqual(result["time_ms"].shape, result["current"]["ix"].shape)
        self.assertGreater(float(np.max(np.abs(result["current"]["ix"]))), 1e-6)


if __name__ == "__main__":
    unittest.main()
