# -*- coding: utf-8 -*-

import unittest

import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import Channel
from braincell._base import IonInfo
from braincell.channel._template import Gate
from braincell.channel._template import GateChannelTemplate
from braincell.channel._template import q10_scale
from braincell.channel._template import shifted_voltage
from braincell.ion import Potassium


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        C=jnp.full((size,), 0.04) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
    )


class _ExampleHHOhmic(GateChannelTemplate):
    root_type = Potassium
    gate_defs = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(22.0)),
        Gate("h", power=1, phi=2.0),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.5 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(5.0 * u.mV, self.varshape, allow_none=False)
        self.temp = u.celsius2kelvin(32.0)

    def drive(self, V, K: IonInfo):
        return K.E - V

    def f_m_inf(self, V, K: IonInfo):
        _ = K
        V = shifted_voltage(V, self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 40.0) / 5.0))

    def f_m_tau(self, V, K: IonInfo):
        _ = K
        _ = shifted_voltage(V, self.V_sh).to_decimal(u.mV)
        return 2.0

    def f_h_inf(self, V, K: IonInfo):
        _ = K
        V = shifted_voltage(V, self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 55.0) / 7.0))

    def f_h_tau(self, V, K: IonInfo):
        _ = K
        V = shifted_voltage(V, self.V_sh).to_decimal(u.mV)
        return 0.5 + 4.0 / (1.0 + u.math.exp(-(V + 40.0) / 5.0))


class GateTemplateTest(unittest.TestCase):
    def test_gate_channel_template_is_channel_subclass(self) -> None:
        self.assertTrue(issubclass(GateChannelTemplate, Channel))

    def test_shifted_voltage_helper(self) -> None:
        V = jnp.array([-55.0]) * u.mV
        got = shifted_voltage(V, 5.0 * u.mV).to_decimal(u.mV)
        self.assertTrue(u.math.allclose(got, jnp.array([-60.0]), atol=1e-6))

    def test_q10_scale_helper(self) -> None:
        got = q10_scale(u.celsius2kelvin(32.0), u.celsius2kelvin(22.0), 3.0)
        self.assertTrue(u.math.allclose(got, jnp.array(3.0), atol=1e-6))

    def test_gate_rejects_invalid_temperature_configuration(self) -> None:
        with self.assertRaises(ValueError):
            Gate("m", q10=3.0)
        with self.assertRaises(ValueError):
            Gate("m", temp_ref=u.celsius2kelvin(22.0))
        with self.assertRaises(ValueError):
            Gate("m", phi=2.0, q10=3.0, temp_ref=u.celsius2kelvin(22.0))

    def test_gate_f_phi_supports_default_phi_and_q10(self) -> None:
        ch = _ExampleHHOhmic(size=1)
        self.assertTrue(u.math.allclose(ch.gate_defs[0].f_phi(ch), jnp.array(3.0), atol=1e-6))
        self.assertEqual(ch.gate_defs[1].f_phi(ch), 2.0)
        self.assertEqual(Gate("n").f_phi(ch), 1.0)

    def test_example_channel_init_reset_derivative_and_current(self) -> None:
        ch = _ExampleHHOhmic(size=1)
        V = jnp.array([-60.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        self.assertEqual(ch.m.value.shape, (1,))
        self.assertEqual(ch.h.value.shape, (1,))

        ch.reset_state(V, K)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, K), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, K), atol=1e-6))

        ch.compute_derivative(V, K)
        expected_m = ch.gate_defs[0].f_phi(ch) * (ch.f_m_inf(V, K) - ch.m.value) / ch.f_m_tau(V, K) / u.ms
        expected_h = ch.gate_defs[1].f_phi(ch) * (ch.f_h_inf(V, K) - ch.h.value) / ch.f_h_tau(V, K) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_h, atol=1e-6 * u.Hz))

        current = ch.current(V, K)
        expected = ch.g_max * ch.m.value ** 3 * ch.h.value * (K.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(u.math.allclose(current.to_decimal(unit), expected.to_decimal(unit), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
