from __future__ import annotations

import os
import unittest

from ._helpers import TEMPLATES_ROOT, build_mapping_payload, load_module


os.environ.setdefault("JAX_PLATFORMS", "cpu")


mapping_schema = load_module(
    TEMPLATES_ROOT / "mapping_schema.py",
    "channel_no_conc_mapping_schema_test",
)


class MappingSchemaTest(unittest.TestCase):
    def test_loads_mapping_with_current_string_and_infers_ion(self) -> None:
        mapping_spec = mapping_schema.MappingSpec.from_mapping(build_mapping_payload())
        self.assertEqual(mapping_spec.current_source.ion_name, "k")
        self.assertEqual(mapping_spec.current_source.neuron_current_var, "ik")
        self.assertEqual(mapping_spec.neuron.mechanism_name, "Kv")
        self.assertEqual(mapping_spec.braincell.class_name, "IK_Kv_test")
        self.assertEqual(mapping_spec.gate_map[0].canonical_name, "n")
        self.assertEqual(mapping_spec.parameter_map["g_max_S_cm2"].braincell, "g_max")
        self.assertEqual(mapping_spec.parameter_map["g_max_S_cm2"].neuron, "gbar")

    def test_loads_mapping_with_common_impl_name_and_gate_names(self) -> None:
        payload = build_mapping_payload(
            impl_name={"common": "Kv1p1_MA24_PC"},
            gate_names={"common": ["n"]},
        )
        mapping_spec = mapping_schema.MappingSpec.from_mapping(payload)
        self.assertEqual(mapping_spec.neuron.mechanism_name, "Kv1p1_MA24_PC")
        self.assertEqual(mapping_spec.braincell.class_name, "Kv1p1_MA24_PC")
        self.assertEqual(mapping_spec.neuron.gate_names, ("n",))
        normalized = mapping_schema.normalize_mapping_payload(payload)
        self.assertEqual(normalized["impl_name"], {"common": "Kv1p1_MA24_PC"})
        self.assertEqual(normalized["current"], "ik")

    def test_rejects_mixed_common_and_side_specific_impl_names(self) -> None:
        payload = build_mapping_payload(
            impl_name={"common": "Kv", "neuron": "Kv", "braincell": "IK_Kv_test"},
        )
        with self.assertRaisesRegex(ValueError, "cannot mix 'common'"):
            mapping_schema.MappingSpec.from_mapping(payload)

    def test_rejects_mismatched_gate_name_lengths(self) -> None:
        payload = build_mapping_payload(
            gate_names={
                "canonical": ["n", "m"],
                "neuron": ["n", "m"],
                "braincell": ["n"],
            }
        )
        with self.assertRaisesRegex(ValueError, "same length"):
            mapping_schema.MappingSpec.from_mapping(payload)

    def test_loads_pure_channel_current_without_inferred_ion(self) -> None:
        payload = build_mapping_payload(
            current="ih",
            impl_name={"common": "Ih_HM1992"},
            gate_names={"common": ["p"]},
            channel_params={
                "g_max_S_cm2": {"neuron": "gbar", "braincell": "g_max"},
                "E_mV": {"neuron": "eh", "braincell": "E"},
            },
        )
        mapping_spec = mapping_schema.MappingSpec.from_mapping(payload)

        self.assertIsNone(mapping_spec.current_source.ion_name)
        self.assertEqual(mapping_spec.current_source.neuron_current_var, "ih")
        self.assertEqual(mapping_spec.parameter_map["E_mV"].neuron, "eh")

    def test_loads_neuron_parameter_names_as_plain_strings(self) -> None:
        payload = build_mapping_payload(
            channel_params={
                "g_max_S_cm2": {
                    "neuron": "gbar",
                    "braincell": "g_max",
                },
                "E_mV": {
                    "neuron": "ek",
                    "braincell": "E",
                },
            },
        )
        mapping_spec = mapping_schema.MappingSpec.from_mapping(payload)

        self.assertEqual(mapping_spec.parameter_map["E_mV"].neuron, "ek")
        normalized = mapping_schema.normalize_mapping_payload(payload)
        self.assertEqual(normalized["channel_params"]["E_mV"], {"neuron": "ek", "braincell": "E"})

    def test_rejects_legacy_current_source_shape(self) -> None:
        payload = build_mapping_payload()
        payload.pop("current")
        payload["current_source"] = {"owner": "channel", "neuron_current_var": "ik"}

        with self.assertRaisesRegex(ValueError, "Legacy mapping current schema"):
            mapping_schema.MappingSpec.from_mapping(payload)

    def test_rejects_legacy_neuron_parameter_dict_shape(self) -> None:
        payload = build_mapping_payload(
            channel_params={
                "g_max_S_cm2": {
                    "neuron": {"name": "gbar", "owner": "mechanism"},
                    "braincell": "g_max",
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "must be a string parameter name"):
            mapping_schema.MappingSpec.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
