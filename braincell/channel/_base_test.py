# -*- coding: utf-8 -*-

import unittest

import braintools
import brainunit as u
import jax.numpy as jnp

from braincell._base import Channel
from braincell._base import IonInfo
from braincell.channel._base import Gate
from braincell.channel._base import HH
from braincell.channel._base import Markov
from braincell.channel._base import Transition
from braincell.channel._base import ghk_flux
from braincell.ion import Calcium
from braincell.ion import Potassium


def _k_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 0.04) * u.mM,
        Co=jnp.full((size,), 2.5) * u.mM,
        E=jnp.full((size,), -90.0) * u.mV,
        valence=1,
    )


def _ca_info(size: int = 1) -> IonInfo:
    return IonInfo(
        Ci=jnp.full((size,), 2.0e-4) * u.mM,
        Co=jnp.full((size,), 2.0) * u.mM,
        E=jnp.full((size,), 120.0) * u.mV,
        valence=2,
    )


class _ExampleHHInfTau(HH):
    root_type = Potassium
    gates = (
        Gate("m", power=3, q10=3.0, temp_ref=u.celsius2kelvin(22.0)),
        Gate("h", power=1, phi=2.0),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.5 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)
        self.V_sh = braintools.init.param(5.0 * u.mV, self.varshape, allow_none=False)
        self.temp = u.celsius2kelvin(32.0)

    def f_m_inf(self, V, K: IonInfo):
        _ = K
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp(-(V + 40.0) / 5.0))

    def f_m_tau(self, V, K: IonInfo):
        _ = (V, K)
        return 2.0

    def f_h_inf(self, V, K: IonInfo):
        _ = K
        V = (V - self.V_sh).to_decimal(u.mV)
        return 1.0 / (1.0 + u.math.exp((V + 55.0) / 7.0))

    def f_h_tau(self, V, K: IonInfo):
        _ = K
        V = (V - self.V_sh).to_decimal(u.mV)
        return 0.5 + 4.0 / (1.0 + u.math.exp(-(V + 40.0) / 5.0))

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)


class _ExampleHHAlphaBeta(HH):
    root_type = Potassium
    gates = (Gate("n", power=4, q10=2.0, temp_ref=u.celsius2kelvin(25.0)),)

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)
        self.temp = u.celsius2kelvin(35.0)

    def f_n_alpha(self, V, K: IonInfo):
        _ = (V, K)
        return 0.4

    def f_n_beta(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)


class _ExampleHHDefaultPhi(HH):
    root_type = Potassium
    gates = (Gate("p"),)

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)

    def f_p_inf(self, V, K: IonInfo):
        _ = (V, K)
        return 0.25

    def f_p_tau(self, V, K: IonInfo):
        _ = (V, K)
        return 2.0

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)


class _ExampleGHK(HH):
    root_type = Calcium
    gates = (Gate("p", power=2, phi=1.5), Gate("q", power=1))

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.p_max = braintools.init.param(0.01 * (u.cm / u.second), self.varshape, allow_none=False)
        self.Co = 2.0 * u.mM
        self.valence = 2
        self.temp = u.celsius2kelvin(36.0)

    def f_p_inf(self, V, Ca: IonInfo):
        _ = (V, Ca)
        return 0.25

    def f_p_tau(self, V, Ca: IonInfo):
        _ = (V, Ca)
        return 2.0

    def f_q_inf(self, V, Ca: IonInfo):
        _ = (V, Ca)
        return 0.5

    def f_q_tau(self, V, Ca: IonInfo):
        _ = (V, Ca)
        return 4.0

    def current(self, V, Ca: IonInfo):
        return self.p_max * self.conductance_factor(V, Ca) * ghk_flux(
            V=V,
            ci=Ca.Ci,
            co=self.Co,
            z=self.valence,
            temp=self.temp,
        )


class _ExampleMarkov(Markov):
    root_type = Potassium
    pairs = (
        Transition("C", "O", "open_rate", "close_rate"),
        ("O", "I", "inactivate_rate", None),
    )
    conserve = 1.0
    dependent_state = "C"

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.3 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)

    def open_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.2

    def close_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def inactivate_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.05

    def current(self, V, K: IonInfo):
        return self.g_max * (self.O.value + 0.5 * self.I.value) * (K.E - V)


class _ExampleMarkovImplicitDependent(Markov):
    root_type = Potassium
    pairs = (
        Transition("C", "O", "open_rate", "close_rate"),
        ("O", "I", "inactivate_rate", None),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.3 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)

    def open_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.2

    def close_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def inactivate_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.05

    def current(self, V, K: IonInfo):
        states = self.state_values()
        return self.g_max * states["O"] * (K.E - V)


class _ExampleMarkovTwoOpenStates(Markov):
    root_type = Potassium
    pairs = (
        Transition("C", "O1", "open1_rate", "close1_rate"),
        ("O1", "O2", "open2_rate", "close2_rate"),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)

    def open1_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.2

    def close1_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def open2_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.05

    def close2_rate(self, V, K: IonInfo):
        _ = (V, K)
        return 0.02

    def current(self, V, K: IonInfo):
        states = self.state_values()
        return self.g_max * (states["O1"] + states["O2"]) * (K.E - V)


class _ExampleMarkovVoltageOnlyRates(Markov):
    root_type = Potassium
    pairs = (
        Transition("C", "O", "open_rate", "close_rate"),
    )
    dependent_state = "C"

    def __init__(self, size=1):
        super().__init__(size=size, name=None)

    def open_rate(self, V):
        _ = V
        return 0.2

    def close_rate(self, V):
        _ = V
        return 0.1

    def current(self, V, K: IonInfo):
        return self.O.value * (K.E - V)


class _ExampleHHMixed(HH):
    root_type = Potassium
    gates = (
        Gate("m", power=3),
        Gate("h", power=1),
    )

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.25 * (u.mS / u.cm ** 2), self.varshape, allow_none=False)

    def f_m_inf(self, V, K: IonInfo):
        _ = (V, K)
        return 0.2

    def f_m_tau(self, V, K: IonInfo):
        _ = (V, K)
        return 2.0

    def f_h_alpha(self, V, K: IonInfo):
        _ = (V, K)
        return 0.3

    def f_h_beta(self, V, K: IonInfo):
        _ = (V, K)
        return 0.2

    def current(self, V, K: IonInfo):
        return self.g_max * self.conductance_factor(V, K) * (K.E - V)


class _ExampleHHConflict(HH):
    root_type = Potassium
    gates = (Gate("x"),)

    def __init__(self):
        super().__init__(size=1, name=None)

    def f_x_inf(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def f_x_tau(self, V, K: IonInfo):
        _ = (V, K)
        return 1.0

    def f_x_alpha(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1

    def f_x_beta(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1


class _ExampleHHMissing(HH):
    root_type = Potassium
    gates = (Gate("x"),)

    def __init__(self):
        super().__init__(size=1, name=None)

    def f_x_alpha(self, V, K: IonInfo):
        _ = (V, K)
        return 0.1


class ChannelTemplateTest(unittest.TestCase):
    def test_gate_validation(self) -> None:
        with self.assertRaises(ValueError):
            Gate("m", q10=3.0)
        with self.assertRaises(ValueError):
            Gate("m", temp_ref=u.celsius2kelvin(22.0))
        with self.assertRaises(ValueError):
            Gate("m", phi=2.0, q10=3.0, temp_ref=u.celsius2kelvin(22.0))

    def test_gate_phi_defaults_to_one(self) -> None:
        ch = _ExampleHHDefaultPhi(size=1)
        self.assertEqual(ch.gate_phi(type(ch).gates[0]), 1.0)

    def test_gate_phi_prefers_explicit_phi(self) -> None:
        ch = _ExampleHHInfTau(size=1)
        self.assertEqual(ch.gate_phi(type(ch).gates[1]), 2.0)

    def test_gate_phi_uses_q10_and_temp_ref(self) -> None:
        ch = _ExampleHHInfTau(size=1)
        expected = 3.0 ** (((ch.temp - u.celsius2kelvin(22.0)) / u.kelvin) / 10.0)
        self.assertTrue(u.math.allclose(ch.gate_phi(type(ch).gates[0]), expected, atol=1e-6))

    def test_hh_inf_tau_channel(self) -> None:
        ch = _ExampleHHInfTau(size=1)
        V = jnp.array([-60.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.reset_state(V, K)
        self.assertTrue(u.math.allclose(ch.m.value, ch.f_m_inf(V, K), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, ch.f_h_inf(V, K), atol=1e-6))

        ch.compute_derivative(V, K)
        expected_m = ch.gate_phi(type(ch).gates[0]) * (ch.f_m_inf(V, K) - ch.m.value) / ch.f_m_tau(V, K) / u.ms
        expected_h = ch.gate_phi(type(ch).gates[1]) * (ch.f_h_inf(V, K) - ch.h.value) / ch.f_h_tau(V, K) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_m, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_h, atol=1e-6 * u.Hz))

    def test_hh_alpha_beta_channel(self) -> None:
        ch = _ExampleHHAlphaBeta(size=1)
        V = jnp.array([-55.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.reset_state(V, K)
        expected_n = 0.4 / (0.4 + 0.1)
        self.assertTrue(u.math.allclose(ch.n.value, expected_n, atol=1e-6))

        ch.compute_derivative(V, K)
        expected_dn = ch.gate_phi(type(ch).gates[0]) * (0.4 * (1.0 - ch.n.value) - 0.1 * ch.n.value) / u.ms
        self.assertTrue(u.math.allclose(ch.n.derivative, expected_dn, atol=1e-6 * u.Hz))

    def test_hh_mixed_channel_supports_both_forms(self) -> None:
        ch = _ExampleHHMixed(size=1)
        V = jnp.array([-55.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.reset_state(V, K)
        expected_h = 0.3 / (0.3 + 0.2)
        self.assertTrue(u.math.allclose(ch.m.value, jnp.array([0.2]), atol=1e-6))
        self.assertTrue(u.math.allclose(ch.h.value, expected_h, atol=1e-6))

        ch.compute_derivative(V, K)
        expected_dm = (0.2 - ch.m.value) / 2.0 / u.ms
        expected_dh = (0.3 * (1.0 - ch.h.value) - 0.2 * ch.h.value) / u.ms
        self.assertTrue(u.math.allclose(ch.m.derivative, expected_dm, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.h.derivative, expected_dh, atol=1e-6 * u.Hz))

    def test_hh_rejects_gate_with_both_forms(self) -> None:
        ch = _ExampleHHConflict()
        with self.assertRaises(ValueError):
            ch.reset_state(jnp.array([-60.0]) * u.mV, _k_info())

    def test_hh_rejects_gate_with_incomplete_form(self) -> None:
        ch = _ExampleHHMissing()
        with self.assertRaises(ValueError):
            ch.reset_state(jnp.array([-60.0]) * u.mV, _k_info())

    def test_ghk_channel_uses_p_max(self) -> None:
        ch = _ExampleGHK(size=1)
        V = jnp.array([-50.0]) * u.mV
        Ca = _ca_info()

        ch.init_state(V, Ca)
        ch.reset_state(V, Ca)
        current = ch.current(V, Ca)

        expected = ch.p_max * ch.p.value ** 2 * ch.q.value * ghk_flux(
            V=V,
            ci=Ca.Ci,
            co=ch.Co,
            z=ch.valence,
            temp=ch.temp,
        )
        unit = expected.unit
        self.assertTrue(u.math.allclose(current.to_decimal(unit), expected.to_decimal(unit), atol=1e-6))

    def test_markov_collects_states_and_builds_dependent_state(self) -> None:
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        self.assertEqual(ch.state_names, ("O", "I"))
        self.assertEqual(ch.redundant_state, "C")
        self.assertEqual(
            ch.state_pairs,
            (
                ("C", "O", "open_rate", "close_rate"),
                ("O", "I", "inactivate_rate", None),
            ),
        )
        self.assertTrue(hasattr(ch, "O"))
        self.assertTrue(hasattr(ch, "I"))
        self.assertFalse(hasattr(ch, "C"))

        ch.reset_state(V, K)
        states = ch.state_values()
        self.assertTrue(u.math.allclose(states["C"], 1.0, atol=1e-6))
        self.assertTrue(u.math.allclose(states["O"], 0.0, atol=1e-6))
        self.assertTrue(u.math.allclose(states["I"], 0.0, atol=1e-6))

        ch.O.value = jnp.array([0.2])
        ch.I.value = jnp.array([0.1])
        states = ch.state_values()
        self.assertTrue(u.math.allclose(states["C"], jnp.array([0.7]), atol=1e-6))

        ch.compute_derivative(V, K)
        expected_dO = (states["C"] * 0.2 - states["O"] * 0.1 - states["O"] * 0.05) / u.ms
        expected_dI = (states["O"] * 0.05) / u.ms
        self.assertTrue(u.math.allclose(ch.O.derivative, expected_dO, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.I.derivative, expected_dI, atol=1e-6 * u.Hz))

        current = ch.current(V, K)
        expected_current = ch.g_max * (states["O"] + 0.5 * states["I"]) * (K.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected_current.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_markov_defaults_dependent_state_to_last_discovered_name(self) -> None:
        ch = _ExampleMarkovImplicitDependent(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        self.assertEqual(ch.redundant_state, "I")
        self.assertEqual(ch.state_names, ("C", "O"))
        self.assertTrue(hasattr(ch, "C"))
        self.assertTrue(hasattr(ch, "O"))
        self.assertFalse(hasattr(ch, "I"))

        ch.C.value = jnp.array([0.3])
        ch.O.value = jnp.array([0.2])
        states = ch.state_values()
        self.assertTrue(u.math.allclose(states["I"], jnp.array([0.5]), atol=1e-6))

    def test_markov_pre_and_post_integral_are_no_ops(self) -> None:
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        self.assertIsNone(ch.pre_integral(V, K))
        self.assertIsNone(ch.post_integral(V, K))

    def test_markov_reset_steady_state_solves_stationary_distribution(self) -> None:
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.reset_steady_state(V, K)
        states = ch.state_values()

        total = states["C"] + states["O"] + states["I"]
        self.assertTrue(u.math.allclose(total, jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(states["I"], jnp.array([1.0]), atol=1e-6))

        ch.compute_derivative(V, K)
        self.assertTrue(u.math.allclose(ch.O.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.I.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))

    def test_markov_reset_steady_state_supports_implicit_dependent_state(self) -> None:
        ch = _ExampleMarkovImplicitDependent(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.reset_steady_state(V, K)
        states = ch.state_values()

        total = states["C"] + states["O"] + states["I"]
        self.assertTrue(u.math.allclose(total, jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(states["I"], jnp.array([1.0]), atol=1e-6))

        ch.compute_derivative(V, K)
        self.assertTrue(u.math.allclose(ch.C.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.O.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))

    def test_markov_kinetic_states_clip_independent_states_for_dynamics(self) -> None:
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.O.value = jnp.array([1.2])
        ch.I.value = jnp.array([-0.1])

        raw_states = ch.state_values()
        kinetic_states = ch._kinetic_state_values()
        self.assertTrue(u.math.allclose(raw_states["O"], jnp.array([1.2]), atol=1e-6))
        self.assertTrue(u.math.allclose(raw_states["I"], jnp.array([-0.1]), atol=1e-6))
        self.assertTrue(u.math.allclose(raw_states["C"], jnp.array([-0.1]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["O"], jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["I"], jnp.array([0.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["C"], jnp.array([-0.1]), atol=1e-6))

    def test_markov_kinetic_states_can_project_independent_states_only_for_dynamics(self) -> None:
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.O.value = jnp.array([1.2])
        ch.I.value = jnp.array([-0.1])

        raw_states = ch.state_values()
        kinetic_states = ch._kinetic_state_values()
        self.assertTrue(u.math.allclose(raw_states["O"], jnp.array([1.2]), atol=1e-6))
        self.assertTrue(u.math.allclose(raw_states["I"], jnp.array([-0.1]), atol=1e-6))
        self.assertTrue(u.math.allclose(raw_states["C"], jnp.array([-0.1]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["O"], jnp.array([1.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["I"], jnp.array([0.0]), atol=1e-6))
        self.assertTrue(u.math.allclose(kinetic_states["C"], jnp.array([-0.1]), atol=1e-6))

        ch.compute_derivative(V, K)
        expected_dO = (kinetic_states["C"] * 0.2 - kinetic_states["O"] * 0.1 - kinetic_states["O"] * 0.05) / u.ms
        expected_dI = (kinetic_states["O"] * 0.05) / u.ms
        self.assertTrue(u.math.allclose(ch.O.derivative, expected_dO, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.I.derivative, expected_dI, atol=1e-6 * u.Hz))

    def test_markov_current_can_sum_multiple_open_states_manually(self) -> None:
        ch = _ExampleMarkovTwoOpenStates(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.C.value = jnp.array([0.3])
        ch.O1.value = jnp.array([0.2])
        states = ch.state_values()
        self.assertTrue(u.math.allclose(states["O2"], jnp.array([0.5]), atol=1e-6))

        current = ch.current(V, K)
        expected_current = ch.g_max * (states["O1"] + states["O2"]) * (K.E - V)
        unit = u.mS / u.cm ** 2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected_current.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_markov_rate_dispatch_respects_declared_signature(self) -> None:
        ch = _ExampleMarkovVoltageOnlyRates(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        ch.init_state(V, K)
        ch.O.value = jnp.array([0.25])
        ch.compute_derivative(V, K)

        states = ch.state_values()
        expected_dO = (states["C"] * 0.2 - states["O"] * 0.1) / u.ms
        self.assertTrue(u.math.allclose(ch.O.derivative, expected_dO, atol=1e-6 * u.Hz))

        ch.reset_steady_state(V, K)
        states = ch.state_values()
        self.assertTrue(u.math.allclose(states["O"], jnp.array([2.0 / 3.0]), atol=1e-6))

    def test_ghk_flux_small_voltage_is_finite(self) -> None:
        value = ghk_flux(
            V=jnp.array([1e-9]) * u.mV,
            ci=jnp.array([2.0e-4]) * u.mM,
            co=2.0 * u.mM,
            z=2,
            temp=u.celsius2kelvin(36.0),
        )
        self.assertEqual(value.shape, (1,))

    def test_ghk_flux_rejects_legacy_T_keyword(self) -> None:
        with self.assertRaises(TypeError):
            ghk_flux(
                V=jnp.array([-40.0]) * u.mV,
                ci=jnp.array([2.0e-4]) * u.mM,
                co=2.0 * u.mM,
                z=2,
                T=u.celsius2kelvin(36.0),
            )


if __name__ == "__main__":
    unittest.main()
