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

import brainunit as u
import jax.numpy as jnp

import braincell.channel as channel
from braincell._base import IonInfo
from braincell.channel.potassium import Kv1p5_MA2024_PC
from braincell.channel.potassium_sodium import Kv1p5_MA2020_GrC
from braincell.ion import NonSpecific, Potassium, Sodium


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _na_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 10.0) * u.mM,
        Co=jnp.full((size,), 140.0) * u.mM,
        E=jnp.full((size,), 50.0) * u.mV,
        valence=1,
    )


def _no_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 1.0) * u.mM,
        Co=jnp.full((size,), 1.0) * u.mM,
        E=jnp.full((size,), 0.0) * u.mV,
        valence=1,
    )


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


_DENSITY_UNIT = u.mS / u.cm ** 2 * u.mV


class Kv1p5MA20GrCTest(unittest.TestCase):
    def test_package_level_export_is_preserved(self) -> None:
        self.assertIs(channel.Kv1p5_MA2020_GrC, Kv1p5_MA2020_GrC)

    def test_declares_multi_ion_dependencies_and_two_current_owners(self) -> None:
        self.assertEqual(Kv1p5_MA2020_GrC.root_type.__args__, (Potassium, Sodium, NonSpecific))
        self.assertEqual(Kv1p5_MA2020_GrC.current_owner_types, {"k": Potassium, "no": NonSpecific})

    def test_uses_g_max_without_legacy_gKur_alias(self) -> None:
        ch = Kv1p5_MA2020_GrC(size=1)
        self.assertTrue(hasattr(ch, "g_max"))
        self.assertFalse(hasattr(ch, "gKur"))

    def test_zero_gnonspec_matches_pc_variant_for_potassium_component(self) -> None:
        temp = u.celsius2kelvin(36.0)
        pc = Kv1p5_MA2024_PC(size=1, temp=temp)
        grc = Kv1p5_MA2020_GrC(size=1, temp=temp)
        V = _V([-35.0])
        k = _k_info()
        na = _na_info()
        no = _no_info()

        pc.init_state(V, k)
        grc.init_state(V, k, na, no)
        pc.reset_state(V, k)
        grc.reset_state(V, k, na, no)
        self.assertTrue(u.math.allclose(grc.m.value, pc.m.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.n.value, pc.n.value, atol=1e-6))
        self.assertTrue(u.math.allclose(grc.u.value, pc.u.value, atol=1e-6))

        pc.compute_derivative(V, k)
        grc.compute_derivative(V, k, na, no)
        self.assertTrue(u.math.allclose(grc.m.derivative, pc.m.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.n.derivative, pc.n.derivative, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(grc.u.derivative, pc.u.derivative, atol=1e-6 * u.Hz))

        i_pc = pc.current(V, k)
        components = grc.current_components(V, k, na, no)
        i_grc = grc.current(V, k, na, no)
        self.assertTrue(u.math.allclose(components["no"].to_decimal(_DENSITY_UNIT), jnp.zeros((1,)), atol=1e-12))
        self.assertTrue(
            u.math.allclose(
                components["k"].to_decimal(_DENSITY_UNIT),
                i_pc.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )
        self.assertTrue(u.math.allclose(i_grc.to_decimal(_DENSITY_UNIT), i_pc.to_decimal(_DENSITY_UNIT), atol=1e-6))

    def test_nonzero_gnonspec_contributes_nonspecific_component(self) -> None:
        temp = u.celsius2kelvin(25.0)
        ch = Kv1p5_MA2020_GrC(size=1, temp=temp, gnonspec=0.2e-3 * (u.siemens / u.cm ** 2))
        V = _V([-20.0])
        k = _k_info()
        na = _na_info()
        no = _no_info()
        ch.init_state(V, k, na, no)
        ch.m.value = jnp.array([0.2])
        ch.n.value = jnp.array([0.3])
        ch.u.value = jnp.array([0.4])

        conductance_factor = (0.2 ** 3) * 0.3 * 0.4
        voltage_factor = ch._voltage_factor(V)
        eno = (
            u.gas_constant
            * ch.temp
            / u.faraday_constant
            * u.math.log((na.Co + k.Co) / (na.Ci + k.Ci))
        )
        expected_no = ch.gnonspec * voltage_factor * conductance_factor * (eno - V)
        components = ch.current_components(V, k, na, no)
        total = ch.current(V, k, na, no)

        self.assertTrue(
            u.math.allclose(
                components["no"].to_decimal(_DENSITY_UNIT),
                expected_no.to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )
        self.assertFalse(u.math.allclose(components["no"].to_decimal(_DENSITY_UNIT), jnp.zeros((1,)), atol=1e-12))
        self.assertTrue(
            u.math.allclose(
                total.to_decimal(_DENSITY_UNIT),
                (components["k"] + components["no"]).to_decimal(_DENSITY_UNIT),
                atol=1e-6,
            )
        )
