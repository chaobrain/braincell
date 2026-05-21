# -*- coding: utf-8 -*-
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

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell._base import IonInfo
from braincell.channel._base import HH, Markov
from braincell.channel.potassium_calcium import (
    AHP_De1994,
    Kca1p1_MA2020_GoC,
    Kca1p1_MA2020_GrC,
    Kca1p1_MA2024_PC,
    Kca1p1_MA2025_BC,
    Kca1p1_RI2021_SC,
    Kca2p2_MA2020_GoC,
    Kca2p2_MA2020_GrC,
    Kca2p2_MA2024_PC,
    Kca2p2_MA2025_BC,
    Kca2p2_RI2021_SC,
    Kca3p1_MA2020_GoC,
    Kca3p1_MA2024_PC,
    Kca3p1_MA2025_BC,
    SK_SU2015_DCN,
)
from braincell.ion import Calcium, Potassium
from braincell.mech import get_registry


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 140.0) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _ca_info(size: int = 1, C: float = 1e-4) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), C) * u.mM,
        Co=jnp.full((size,), 2.0) * u.mM,
        E=jnp.full((size,), 120.0) * u.mV,
        valence=2,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class _MixedPotassiumCalciumTemplateTest:
    CLS = None
    TEMPLATE = None

    def test_root_type_is_joint_potassium_calcium(self) -> None:
        root = self.CLS.root_type
        self.assertIsInstance(root, brainstate.mixin._JointGenericAlias)
        self.assertIn(Potassium, root.__args__)
        self.assertIn(Calcium, root.__args__)

    def test_current_owner_type_is_potassium(self) -> None:
        self.assertIs(self.CLS.current_owner_type, Potassium)

    def test_inherits_expected_template(self) -> None:
        self.assertIs(self.CLS.compute_derivative, self.TEMPLATE.compute_derivative)


class IAHPDe1994Test(_MixedPotassiumCalciumTemplateTest, unittest.TestCase):
    CLS = AHP_De1994
    TEMPLATE = HH

    def test_reset_state_matches_steady_state(self) -> None:
        ch = AHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)

        C2 = ch.alpha * (ca.Ci / u.mM) ** ch.n
        expected = C2 / (C2 + ch.beta)
        self.assertTrue(u.math.allclose(ch.p.value, expected, atol=1e-6))

    def test_current_matches_p_squared_form(self) -> None:
        ch = AHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        i = ch.current(V, k, ca)
        expected = ch.g_max * ch.p.value ** 2 * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_uses_first_order_kinetics(self) -> None:
        ch = AHP_De1994(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.p.value = jnp.array([0.2])
        ch.compute_derivative(V, k, ca)

        C2 = ch.alpha * (ca.Ci / u.mM) ** ch.n
        expected = ch.phi * (C2 * (1.0 - 0.2) - ch.beta * 0.2) / u.ms
        self.assertTrue(u.math.allclose(ch.p.derivative, expected, atol=1e-6 * u.Hz))


class SK_SU2015_DCNTest(_MixedPotassiumCalciumTemplateTest, unittest.TestCase):
    CLS = SK_SU2015_DCN
    TEMPLATE = HH

    def test_gate_definition_is_single_z_gate(self) -> None:
        ch = SK_SU2015_DCN(size=1)
        gates = ch._iter_gates()
        self.assertEqual(tuple(gate.name for gate in gates), ("z",))
        self.assertEqual(tuple(gate.power for gate in gates), (1,))

    def test_default_parameters_are_stored(self) -> None:
        ch = SK_SU2015_DCN(size=1)
        self.assertTrue(u.math.allclose(ch.g_max, 0.01 * (u.mS / u.cm ** 2), atol=1e-12 * (u.mS / u.cm ** 2)))
        self.assertTrue(u.math.allclose(ch.qdeltat, jnp.ones(1), atol=1e-12))

    def test_reset_state_sets_z_to_zinf(self) -> None:
        ch = SK_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=3e-4)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        self.assertTrue(u.math.allclose(ch.z.value, ch.f_z_inf(V, k, ca), atol=1e-6))

    def test_z_inf_matches_mod_table_interpolation(self) -> None:
        ch = SK_SU2015_DCN(size=3)
        V = _V([-60.0, -60.0, -60.0])
        k = _k_info(3)
        ca = IonInfo(
            Ci=jnp.asarray([5e-5, 3e-4, 1e-3]) * u.mM,
            Co=jnp.full((3,), 2.0) * u.mM,
            E=jnp.full((3,), 120.0) * u.mV,
            valence=2,
        )

        expected = jnp.asarray(
            [
                0.0012925563241940586,
                0.5000000000000003,
                0.9919661920981747,
            ]
        )
        self.assertTrue(u.math.allclose(ch.f_z_inf(V, k, ca), expected, atol=1e-7))

    def test_z_tau_matches_mod_table_interpolation_and_qdeltat(self) -> None:
        ch = SK_SU2015_DCN(size=3, qdeltat=2.0)
        V = _V([-60.0, -60.0, -60.0])
        k = _k_info(3)
        ca = IonInfo(
            Ci=jnp.asarray([5e-5, 0.004, 0.006]) * u.mM,
            Co=jnp.full((3,), 2.0) * u.mM,
            E=jnp.full((3,), 120.0) * u.mV,
            valence=2,
        )

        expected = jnp.asarray(
            [
                0.49533325,
                0.12666000000000002,
                0.03335,
            ]
        )
        self.assertTrue(u.math.allclose(ch.f_z_tau(V, k, ca), expected, atol=1e-7))

    def test_z_helpers_match_mod_formulas_at_table_nodes(self) -> None:
        ch = SK_SU2015_DCN(size=1)
        cai = jnp.asarray([0.0003, 0.006])
        z_expected = cai ** 4 / (cai ** 4 + 8.1e-15)
        tau_expected = jnp.where(cai < 0.005, 1.0 - 186.67 * cai, 0.0667)
        self.assertTrue(u.math.allclose(ch._z_inf_formula(cai), z_expected, atol=1e-12))
        self.assertTrue(u.math.allclose(ch._z_tau_formula(cai), tau_expected, atol=1e-12))

    def test_compute_derivative_matches_inf_tau_form(self) -> None:
        ch = SK_SU2015_DCN(size=1, qdeltat=2.0)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=3e-4)
        ch.init_state(V, k, ca)
        ch.z.value = jnp.array([0.2])
        ch.compute_derivative(V, k, ca)
        expected = (ch.f_z_inf(V, k, ca) - ch.z.value) / ch.f_z_tau(V, k, ca) / u.ms
        self.assertTrue(u.math.allclose(ch.z.derivative, expected, atol=1e-6 * u.Hz))

    def test_current_matches_mod_formula_with_braincell_sign(self) -> None:
        ch = SK_SU2015_DCN(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=3e-4)
        ch.init_state(V, k, ca)
        ch.z.value = jnp.array([0.5])
        current = ch.current(V, k, ca)
        expected = ch.g_max * ch.z.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )


class Kca3p1_MA2020_GoCTest(_MixedPotassiumCalciumTemplateTest, unittest.TestCase):
    CLS = Kca3p1_MA2020_GoC
    TEMPLATE = HH

    def test_reset_state_matches_p_inf(self) -> None:
        ch = Kca3p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        self.assertTrue(u.math.allclose(ch.p.value, ch.p_inf(V, ca), atol=1e-6))

    def test_current_matches_g_times_p_times_drive(self) -> None:
        ch = Kca3p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        i = ch.current(V, k, ca)
        expected = ch.g_max * ch.p.value * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_runs(self) -> None:
        ch = Kca3p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        ch.p.value = jnp.array([0.3])
        ch.compute_derivative(V, k, ca)
        self.assertEqual(ch.p.derivative.shape, (1,))

    def test_compute_derivative_matches_reference_without_temperature_scaling(self) -> None:
        cold = Kca3p1_MA2020_GoC(size=1, temp=u.celsius2kelvin(22.0))
        warm = Kca3p1_MA2020_GoC(size=1, temp=u.celsius2kelvin(37.0))
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info(C=1e-3)

        for ch in (cold, warm):
            ch.init_state(V, k, ca)
            ch.p.value = jnp.array([0.3])
            ch.compute_derivative(V, k, ca)

        self.assertTrue(u.math.allclose(cold.p.derivative, warm.p.derivative, atol=1e-9 * u.Hz))

    def test_rejects_legacy_temperature_keywords(self) -> None:
        with self.assertRaises(TypeError):
            Kca3p1_MA2020_GoC(size=1, T=u.celsius2kelvin(22.0))
        with self.assertRaises(TypeError):
            Kca3p1_MA2020_GoC(size=1, T_base=3.0)


class Kca3p1InheritedCellVariantTest(unittest.TestCase):
    def test_cell_variants_inherit_goc_kinetics(self) -> None:
        for cls in (Kca3p1_MA2025_BC, Kca3p1_MA2024_PC):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, Kca3p1_MA2020_GoC))
                self.assertIs(cls.root_type, Kca3p1_MA2020_GoC.root_type)
                self.assertIs(cls.current_owner_type, Potassium)

    def test_registry_resolves_cell_variants(self) -> None:
        registry = get_registry()
        self.assertIs(registry.get("channel", "Kca3p1_MA2025_BC"), Kca3p1_MA2025_BC)
        self.assertIs(registry.get("channel", "Kca3p1_MA2024_PC"), Kca3p1_MA2024_PC)

    def test_cell_variants_match_goc_state_derivative_and_current(self) -> None:
        V = _V([-55.0])
        k = _k_info()
        ca = _ca_info(C=8e-4)
        goc = Kca3p1_MA2020_GoC(size=1)

        for cls in (Kca3p1_MA2025_BC, Kca3p1_MA2024_PC):
            with self.subTest(cls=cls.__name__):
                variant = cls(size=1)
                goc.init_state(V, k, ca)
                variant.init_state(V, k, ca)
                goc.reset_state(V, k, ca)
                variant.reset_state(V, k, ca)
                self.assertTrue(u.math.allclose(variant.p.value, goc.p.value, atol=1e-6))

                goc.p.value = jnp.array([0.4])
                variant.p.value = jnp.array([0.4])
                goc.compute_derivative(V, k, ca)
                variant.compute_derivative(V, k, ca)
                self.assertTrue(
                    u.math.allclose(
                        variant.p.derivative,
                        goc.p.derivative,
                        atol=1e-6 * u.Hz,
                    )
                )

                i_goc = goc.current(V, k, ca)
                i_variant = variant.current(V, k, ca)
                self.assertTrue(
                    u.math.allclose(
                        i_variant.to_decimal(_DENSITY_UNIT),
                        i_goc.to_decimal(_DENSITY_UNIT),
                        atol=1e-6,
                    )
                )


class Kca2p2_MA2020_GoCTest(_MixedPotassiumCalciumTemplateTest, unittest.TestCase):
    CLS = Kca2p2_MA2020_GoC
    TEMPLATE = Markov

    def test_init_state_creates_independent_microstates(self) -> None:
        ch = Kca2p2_MA2020_GoC(size=2)
        V = _V([-60.0, -50.0])
        k = _k_info(2)
        ca = _ca_info(2)
        ch.init_state(V, k, ca)
        self.assertEqual(ch.state_names, ("C2", "C3", "C4", "O1", "O2"))
        self.assertEqual(ch.redundant_state, "C1")
        for name in ch.state_names:
            self.assertEqual(getattr(ch, name).value.shape, (2,))

    def test_reset_state_solves_steady_state_with_total_probability_one(self) -> None:
        ch = Kca2p2_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        states = ch.state_values()
        total = sum(states[name] for name in ("C1", "C2", "C3", "C4", "O1", "O2"))
        self.assertTrue(u.math.allclose(total, jnp.ones(1), atol=1e-6))

    def test_current_matches_open_states_times_drive(self) -> None:
        ch = Kca2p2_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.C2.value = jnp.zeros(1)
        ch.C3.value = jnp.zeros(1)
        ch.C4.value = jnp.zeros(1)
        ch.O1.value = jnp.ones(1)
        ch.O2.value = jnp.zeros(1)
        states = ch.state_values()
        i = ch.current(V, k, ca)
        expected = ch.g_max * (states["O1"] + states["O2"]) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_runs(self) -> None:
        ch = Kca2p2_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        ch.compute_derivative(V, k, ca)
        for name in ch.state_names:
            self.assertEqual(getattr(ch, name).derivative.shape, (1,))

    def test_rejects_legacy_temperature_keywords(self) -> None:
        with self.assertRaises(TypeError):
            Kca2p2_MA2020_GoC(size=1, T=u.celsius2kelvin(22.0))
        with self.assertRaises(TypeError):
            Kca2p2_MA2020_GoC(size=1, T_base=3.0)


class Kca1p1_MA2020_GoCTest(_MixedPotassiumCalciumTemplateTest, unittest.TestCase):
    CLS = Kca1p1_MA2020_GoC
    TEMPLATE = Markov

    def test_init_state_creates_independent_closed_and_open_states(self) -> None:
        ch = Kca1p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        self.assertEqual(ch.redundant_state, "C0")
        self.assertEqual(ch.state_names, ("C1", "C2", "C3", "C4", "O0", "O1", "O2", "O3", "O4"))
        for name in ch.state_names:
            self.assertEqual(getattr(ch, name).value.shape, (1,))

    def test_reset_state_solves_steady_state_with_total_probability_one(self) -> None:
        ch = Kca1p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        states = ch.state_values()
        total = sum(states[f"C{i}"] for i in range(5)) + sum(states[f"O{i}"] for i in range(5))
        self.assertTrue(u.math.allclose(total, jnp.ones(1), atol=1e-6))

    def test_current_sums_all_open_states(self) -> None:
        ch = Kca1p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        for i in range(1, 5):
            getattr(ch, f"C{i}").value = jnp.zeros(1)
        for i in range(5):
            getattr(ch, f"O{i}").value = jnp.full((1,), 0.2)
        states = ch.state_values()
        i_val = ch.current(V, k, ca)
        expected = ch.g_max * (
            states["O0"] + states["O1"] + states["O2"] + states["O3"] + states["O4"]
        ) * (k.E - V)
        self.assertTrue(
            u.math.allclose(
                i_val.to_decimal(_DENSITY_UNIT),
                expected.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_compute_derivative_runs(self) -> None:
        ch = Kca1p1_MA2020_GoC(size=1)
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        ch.init_state(V, k, ca)
        ch.reset_state(V, k, ca)
        ch.compute_derivative(V, k, ca)
        for name in ch.state_names:
            self.assertEqual(getattr(ch, name).derivative.shape, (1,))

    def test_rejects_legacy_temperature_keywords(self) -> None:
        with self.assertRaises(TypeError):
            Kca1p1_MA2020_GoC(size=1, T=u.celsius2kelvin(22.0))
        with self.assertRaises(TypeError):
            Kca1p1_MA2020_GoC(size=1, T_base=3.0)


class KcaInheritedCellVariantTest(unittest.TestCase):
    KCA2P2_VARIANTS = (
        ("Kca2p2_MA2025_BC", Kca2p2_MA2025_BC, Kca2p2_MA2020_GoC),
        ("Kca2p2_MA2020_GrC", Kca2p2_MA2020_GrC, Kca2p2_MA2020_GoC),
        ("Kca2p2_MA2024_PC", Kca2p2_MA2024_PC, Kca2p2_MA2020_GoC),
        ("Kca2p2_RI2021_SC", Kca2p2_RI2021_SC, Kca2p2_MA2020_GoC),
    )
    KCA1P1_VARIANTS = (
        ("Kca1p1_MA2025_BC", Kca1p1_MA2025_BC, Kca1p1_MA2020_GoC),
        ("Kca1p1_MA2020_GrC", Kca1p1_MA2020_GrC, Kca1p1_MA2020_GoC),
        ("Kca1p1_MA2024_PC", Kca1p1_MA2024_PC, Kca1p1_MA2020_GoC),
        ("Kca1p1_RI2021_SC", Kca1p1_RI2021_SC, Kca1p1_MA2020_GoC),
    )

    def test_cell_specific_variants_are_registered_subclasses(self) -> None:
        registry = get_registry()
        for name, cls, base in self.KCA2P2_VARIANTS + self.KCA1P1_VARIANTS:
            with self.subTest(name=name):
                self.assertTrue(issubclass(cls, base))
                self.assertIs(registry.get("channel", name), cls)
                self.assertEqual(cls.__module__, "braincell.channel")

    def test_kca2p2_variants_inherit_markov_behavior(self) -> None:
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        for name, cls, base in self.KCA2P2_VARIANTS:
            with self.subTest(name=name):
                ch = cls(size=1)
                self.assertIs(cls.compute_derivative, base.compute_derivative)
                ch.init_state(V, k, ca)
                self.assertEqual(ch.state_names, ("C2", "C3", "C4", "O1", "O2"))
                self.assertEqual(ch.redundant_state, "C1")
                ch.O1.value = jnp.ones(1)
                ch.O2.value = jnp.zeros(1)
                current = ch.current(V, k, ca)
                expected = ch.g_max * (ch.state_values()["O1"] + ch.state_values()["O2"]) * (k.E - V)
                self.assertTrue(
                    u.math.allclose(
                        current.to_decimal(_DENSITY_UNIT),
                        expected.to_decimal(_DENSITY_UNIT),
                        atol=1e-6,
                    )
                )

    def test_kca1p1_variants_inherit_markov_behavior(self) -> None:
        V = _V([-60.0])
        k = _k_info()
        ca = _ca_info()
        for name, cls, base in self.KCA1P1_VARIANTS:
            with self.subTest(name=name):
                ch = cls(size=1)
                self.assertIs(cls.compute_derivative, base.compute_derivative)
                ch.init_state(V, k, ca)
                self.assertEqual(ch.redundant_state, "C0")
                self.assertEqual(ch.state_names, ("C1", "C2", "C3", "C4", "O0", "O1", "O2", "O3", "O4"))
                for i in range(1, 5):
                    getattr(ch, f"C{i}").value = jnp.zeros(1)
                for i in range(5):
                    getattr(ch, f"O{i}").value = jnp.full((1,), 0.2)
                current = ch.current(V, k, ca)
                states = ch.state_values()
                expected = ch.g_max * (
                    states["O0"] + states["O1"] + states["O2"] + states["O3"] + states["O4"]
                ) * (k.E - V)
                self.assertTrue(
                    u.math.allclose(
                        current.to_decimal(_DENSITY_UNIT),
                        expected.to_decimal(_DENSITY_UNIT),
                        atol=1e-6,
                    )
                )


if __name__ == "__main__":
    unittest.main()
