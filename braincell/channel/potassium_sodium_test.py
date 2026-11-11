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


import math
import unittest

import brainunit as u
import jax.numpy as jnp
import numpy as np

import braincell.channel as channel
from braincell._base_channel import IonInfo
from braincell.channel.potassium import Kv1p5_MA2024_PC
from braincell.channel.potassium_sodium import Kv1p5_MA2020_GrC, NaKPump_Ca1999
from braincell.ion import NonSpecific, Potassium, Sodium
from braincell.ion._testing import run_float64
from braincell.channel._testing import (
    DENSITY_UNIT,
    k_info,
    na_info,
    nonspecific_info,
    voltage,
)


def _na_info(size: int = 1) -> IonInfo:
    """Sodium at the 10 mM intracellular concentration this pair uses."""
    return na_info(size, Ci=10.0)


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
        V = voltage([-35.0])
        k = k_info()
        na = _na_info()
        no = nonspecific_info()

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
        self.assertTrue(u.math.allclose(components["no"].to_decimal(DENSITY_UNIT), jnp.zeros((1,)), atol=1e-12))
        self.assertTrue(
            u.math.allclose(
                components["k"].to_decimal(DENSITY_UNIT),
                i_pc.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )
        self.assertTrue(u.math.allclose(i_grc.to_decimal(DENSITY_UNIT), i_pc.to_decimal(DENSITY_UNIT), atol=1e-6))

    def test_nonzero_gnonspec_contributes_nonspecific_component(self) -> None:
        temp = u.celsius2kelvin(25.0)
        ch = Kv1p5_MA2020_GrC(size=1, temp=temp, gnonspec=0.2e-3 * (u.siemens / u.cm**2))
        V = voltage([-20.0])
        k = k_info()
        na = _na_info()
        no = nonspecific_info()
        ch.init_state(V, k, na, no)
        ch.m.value = jnp.array([0.2])
        ch.n.value = jnp.array([0.3])
        ch.u.value = jnp.array([0.4])

        conductance_factor = (0.2**3) * 0.3 * 0.4
        voltage_factor = ch._voltage_factor(V)
        # The mod's ``z = R*(celsius + 273.15)/F`` is a number in volts that
        # its ``v - z*log(...)`` uses as millivolts.
        z = (u.gas_constant * ch.temp / u.faraday_constant).to_decimal(u.volt)
        eno = z * u.math.log((na.Co + k.Co) / (na.Ci + k.Ci)) * u.mV
        expected_no = ch.gnonspec * voltage_factor * conductance_factor * (eno - V)
        components = ch.current_components(V, k, na, no)
        total = ch.current(V, k, na, no)

        self.assertTrue(
            u.math.allclose(
                components["no"].to_decimal(DENSITY_UNIT),
                expected_no.to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )
        self.assertFalse(u.math.allclose(components["no"].to_decimal(DENSITY_UNIT), jnp.zeros((1,)), atol=1e-12))
        self.assertTrue(
            u.math.allclose(
                total.to_decimal(DENSITY_UNIT),
                (components["k"] + components["no"]).to_decimal(DENSITY_UNIT),
                atol=1e-6,
            )
        )

    def test_nonspecific_current_matches_mod_arithmetic(self) -> None:
        # Counter-example: the mod's reversal ``z*log((nao+ko)/(nai+ki))`` with
        # ``z = R*(celsius+273.15)/F`` in NEURON units (joule/degC, coulombs)
        # is about 1/1000 of the Nernst-like value in mV; reading it as a
        # united voltage gave a reversal 1000 times too far from 0 mV.
        celsius, v = 30.0, -20.0
        ki, ko, nai, nao = 140.0, 5.0, 10.0, 140.0
        m, n, uu = 0.2, 0.3, 0.4
        gnonspec = 1e-4
        ch = Kv1p5_MA2020_GrC(size=1, temp=u.celsius2kelvin(celsius), gnonspec=gnonspec * (u.siemens / u.cm**2))
        V = voltage([v])
        k = k_info(Ci=ki, Co=ko)
        na = na_info(1, Ci=nai, Co=nao)
        no = nonspecific_info()
        ch.init_state(V, k, na, no)
        ch.m.value = jnp.array([m])
        ch.n.value = jnp.array([n])
        ch.u.value = jnp.array([uu])

        z = 8.31446261815324 * (celsius + 273.15) / 96485.33212331001
        ino = (
            gnonspec
            * (0.1 + 1 / (1 + math.exp(-(v - 15) / 13)))
            * m**3
            * n
            * uu
            * (v - z * math.log((nao + ko) / (nai + ki)))
        )
        got = ch.current_components(V, k, na, no)["no"].to_decimal(u.mA / u.cm**2)
        self.assertTrue(jnp.allclose(got, jnp.array([-ino]), rtol=1e-5, atol=0.0), msg=f"{got} vs {-ino}")


_IMAX = 0.012 * u.mA / u.cm**2
_DENSITY = u.mA / u.cm**2


def _x64_pump_values() -> dict:
    """Float64 pump currents (mA/cm2) at ``Ci_Na`` 4 mM and at ``Ci_Na = km``."""
    import brainstate

    brainstate.environ.set(precision=64)
    pump = NaKPump_Ca1999(1, imax=_IMAX)
    V, k = voltage([-60.0]), k_info()
    parts = pump.current_components(V, na_info(1, Ci=4.0), k)
    total = pump.current(V, na_info(1, Ci=4.0), k)
    for value in (*parts.values(), total):
        assert u.get_unit(value).dim == _DENSITY.dim, u.get_unit(value)
    half = pump.pump_current(na_info(1, Ci=10.0))
    return {
        "keys": sorted(parts),
        **{key: float(value.to_decimal(_DENSITY)[0]) for key, value in parts.items()},
        "total": float(total.to_decimal(_DENSITY)[0]),
        "half": float(half.to_decimal(_DENSITY)[0]),
    }


def _x64_one_step_cell_report() -> dict:
    """One 0.025 ms step of a one-CV Cell with only the pump: ``[dNai, dKi, dV, I]`` per variant.

    The variants: dynamic ions (``SodiumDetailed``, ``PotassiumDetailed``,
    ``d`` = 1 um, ``nai`` 4 mM), fixed ions, both on a lumped Cell (``dV``
    only: a lumped Cell records the voltage alone), and three wrong pumps
    registered here (a child process): the stoichiometry swapped (``na -2``,
    ``k +3``), a ``current`` that returns the sodium part only, and no
    potassium component. ``I`` is the recorded channel current at t = 0.
    """
    import brainstate

    import braincell as bc
    from braincell.mech import CableProperty, Channel, Ion, register_channel

    brainstate.environ.set(precision=64)

    @register_channel("NaKPump_Ca1999_Swapped")
    class Swapped(NaKPump_Ca1999):
        def current_components(self, V, Na, K):
            i_p = self.pump_current(Na)
            return {"na": -2.0 * i_p, "k": 3.0 * i_p}

    @register_channel("NaKPump_Ca1999_SodiumOnly")
    class SodiumOnly(NaKPump_Ca1999):
        def current(self, V, Na, K):
            return self.current_components(V, Na, K)["na"]

    @register_channel("NaKPump_Ca1999_NoPotassium")
    class NoPotassium(NaKPump_Ca1999):
        def current_components(self, V, Na, K):
            i_p = self.pump_current(Na)
            return {"na": -3.0 * i_p, "k": 0.0 * i_p}

    dynamic = (
        Ion("SodiumDetailed", name="na", d=1.0 * u.um, Ci_initializer=4.0 * u.mM),
        Ion("PotassiumDetailed", name="k", d=1.0 * u.um),
    )
    fixed = (Ion("SodiumFixed", name="na", Ci=4.0 * u.mM), Ion("PotassiumFixed", name="k"))

    def step(ions, pump="NaKPump_Ca1999", lumped=False):
        V_init = -60.0 * u.mV
        if lumped:
            cell = bc.Cell.point(1, length=15.0 * u.um, radius=7.5 * u.um, V_init=V_init)
        else:
            branch = bc.Branch.from_lengths(lengths=[15.0] * u.um, radii=[7.5, 7.5] * u.um)
            cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=V_init)
        cell.paint(
            CableProperty(
                resting_potential=V_init,
                membrane_capacitance=1.0 * u.uF / u.cm**2,
                axial_resistivity=100.0 * u.ohm * u.cm,
            )
        )
        for ion in ions:
            cell.paint(ion)
        cell.paint(Channel(pump, name="pump", imax=_IMAX))
        # A lumped Cell records the voltage only.
        recorded = (("v", u.mV),) if lumped else (("nai", u.mM), ("ki", u.mM), ("v", u.mV))
        if not lumped:
            cell.record("nai", bc.observe.ion(name="na").state("Ci"))
            cell.record("ki", bc.observe.ion(name="k").state("Ci"))
            cell.record("pump", bc.observe.channel(name="pump").current())
        cell.record("v", bc.observe.state("v"))
        result = cell.run(dt=0.025 * u.ms, duration=0.05 * u.ms)
        report = [None, None] if lumped else []
        for key, unit in recorded:
            values = np.asarray(result.samples[key].values.to_decimal(unit))
            values = values.reshape(values.shape[0], -1)[:, 0]
            report.append(float(values[1] - values[0]))
        if not lumped:
            # The recorded channel current at t = 0, mA/cm2.
            report.append(float(np.reshape(result.samples["pump"].values.to_decimal(_DENSITY), -1)[0]))
        return report

    return {
        "dynamic": step(dynamic),
        "fixed": step(fixed),
        "lumped_fixed": step(fixed, lumped=True),
        "lumped_dynamic": step(dynamic, lumped=True),
        "swapped": step(dynamic, "NaKPump_Ca1999_Swapped"),
        "sodium_only": step(dynamic, "NaKPump_Ca1999_SodiumOnly"),
        "no_potassium": step(dynamic, "NaKPump_Ca1999_NoPotassium"),
    }


# One step of the dynamic-ion Cell, by hand (F = 96485.33212331001 C/mol,
# I_p = 0.012 / (1 + 2.5 ** 1.5) mA/cm2): dNai = -3 I_p dt / (F d),
# dKi = +2 I_p dt / (F d), dV = -I_p dt / cm.
_ONE_STEP = {"nai": -1.8833294e-5, "ki": 1.2555530e-5, "v": -0.06057122}


def _one_step_failures(deltas) -> list:
    """The names among ``nai``, ``ki``, ``v`` whose one-step change misses :data:`_ONE_STEP` (rtol 1e-6)."""
    return [name for name, got in zip(_ONE_STEP, deltas) if not abs(got / _ONE_STEP[name] - 1.0) <= 1e-6]


class NaKPumpCa1999Test(unittest.TestCase):
    def test_package_level_export_and_owners(self) -> None:
        self.assertIs(channel.NaKPump_Ca1999, NaKPump_Ca1999)
        self.assertEqual(NaKPump_Ca1999.root_type.__args__, (Sodium, Potassium))
        self.assertEqual(NaKPump_Ca1999.current_owner_types, {"na": Sodium, "k": Potassium})

    def test_defaults_are_the_mod_parameters(self) -> None:
        pump = NaKPump_Ca1999(1)
        self.assertTrue(u.math.allclose(pump.imax, 0.04 * _DENSITY))
        self.assertTrue(u.math.allclose(pump.km, 10.0 * u.mM))
        self.assertTrue(np.allclose(pump.n, 1.5))
        self.assertEqual(pump.states, {})

    def test_components_at_4_mM_and_half_activation_at_km(self) -> None:
        values = run_float64(
            "import json; from braincell.channel.potassium_sodium_test import _x64_pump_values; "
            "print(json.dumps(_x64_pump_values()))"
        )
        self.assertEqual(values["keys"], ["k", "na"])
        # 0.012 / (1 + 2.5 ** 1.5) = 2.4228489e-3 mA/cm2 outward.
        np.testing.assert_allclose(
            [values["na"], values["k"], values["total"]], [-7.2685466e-3, 4.8456978e-3, -2.4228489e-3], rtol=1e-7
        )
        np.testing.assert_allclose(values["half"], 0.006, rtol=1e-12)

    def test_does_not_depend_on_voltage_or_potassium(self) -> None:
        pump = NaKPump_Ca1999(2, imax=_IMAX)
        na = na_info(2, Ci=4.0)
        a = pump.current(voltage([-80.0, 20.0]), na, k_info(2, Ci=140.0, Co=5.0))
        b = pump.current(voltage([0.0, -40.0]), na, k_info(2, Ci=54.4, Co=2.5))
        np.testing.assert_array_equal(a.to_decimal(_DENSITY), b.to_decimal(_DENSITY))
        pump.compute_derivative(voltage([0.0, 0.0]), na, k_info(2))  # stateless: a no-op


class NaKPumpCa1999CellTest(unittest.TestCase):
    """One float64 Cell step per variant, all run in one child process."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cell = run_float64(
            "import json; from braincell.channel.potassium_sodium_test import _x64_one_step_cell_report; "
            "print(json.dumps(_x64_one_step_cell_report()))"
        )

    def test_one_step_in_a_cell_with_dynamic_sodium_and_potassium(self) -> None:
        self.assertEqual(_one_step_failures(self.cell["dynamic"]), [], self.cell["dynamic"])

    def test_one_step_with_fixed_ions_and_on_a_lumped_cell(self) -> None:
        nai, ki, dv, _ = self.cell["fixed"]
        self.assertEqual((nai, ki), (0.0, 0.0))
        self.assertEqual(_one_step_failures([_ONE_STEP["nai"], _ONE_STEP["ki"], dv]), [], dv)
        self.assertEqual(self.cell["fixed"][3], self.cell["dynamic"][3])
        # With fixed ions the pump current is constant, so the lumped Cell's
        # ralston2 step is exact; with dynamic ions its second stage sees the
        # lowered nai (2.8e-6 relative).
        dv = self.cell["lumped_fixed"][2]
        self.assertLess(abs(dv / _ONE_STEP["v"] - 1.0), 1e-6, dv)
        dv = self.cell["lumped_dynamic"][2]
        self.assertLess(abs(dv / _ONE_STEP["v"] - 1.0), 1e-5, dv)

    def test_recorded_channel_current_is_the_net_pump_current(self) -> None:
        np.testing.assert_allclose(self.cell["dynamic"][3], -2.4228489e-3, rtol=1e-7)

    def test_wrong_stoichiometry_fails_the_one_step_check(self) -> None:
        # Counter-example of the one-step check above.
        self.assertLessEqual({"nai", "ki"}, set(_one_step_failures(self.cell["swapped"])), self.cell["swapped"])

    def test_the_voltage_follows_the_components_and_the_record_follows_current(self) -> None:
        # The Cell's membrane current is what the owner ions receive, the sum of
        # the components: a component left out moves V, a wrong ``current`` does
        # not; ``current`` is what observe.channel(...).current() records.
        self.assertEqual(_one_step_failures(self.cell["no_potassium"]), ["ki", "v"], self.cell["no_potassium"])
        sodium_only = self.cell["sodium_only"]
        self.assertEqual(_one_step_failures(sodium_only[:3]), [], sodium_only)
        np.testing.assert_allclose(sodium_only[3], -7.2685466e-3, rtol=1e-7)
