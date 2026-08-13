# -*- coding: utf-8 -*-

import unittest
import warnings

import braintools
import brainunit as u
import jax
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
from braincell.quad import get_integrator
from braincell.quad.protocol import DiffEqState


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
        self.g_max = braintools.init.param(0.5 * (u.mS / u.cm**2), self.varshape, allow_none=False)
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
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm**2), self.varshape, allow_none=False)
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
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm**2), self.varshape, allow_none=False)

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
        return (
            self.p_max
            * self.conductance_factor(V, Ca)
            * ghk_flux(
                V=V,
                ci=Ca.Ci,
                co=self.Co,
                z=self.valence,
                temp=self.temp,
            )
        )


class _ExampleMarkov(Markov):
    root_type = Potassium
    pairs = (
        Transition("C", "O", "open_rate", "close_rate"),
        ("O", "I", "inactivate_rate", None),
    )
    conserve = 1.0
    dependent_state = "C"

    def __init__(self, size=1, solver=None, substeps=None):
        super().__init__(size=size, name=None, solver=solver, substeps=substeps)
        self.g_max = braintools.init.param(0.3 * (u.mS / u.cm**2), self.varshape, allow_none=False)

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
        self.g_max = braintools.init.param(0.3 * (u.mS / u.cm**2), self.varshape, allow_none=False)

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
    dependent_state = "O2"  # what the implicit scan resolved to; pinned so reordering cannot move it

    def __init__(self, size=1):
        super().__init__(size=size, name=None)
        self.g_max = braintools.init.param(0.2 * (u.mS / u.cm**2), self.varshape, allow_none=False)

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
    pairs = (Transition("C", "O", "open_rate", "close_rate"),)
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
        self.g_max = braintools.init.param(0.25 * (u.mS / u.cm**2), self.varshape, allow_none=False)

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


def _make_hh(name: str, namespace: dict):
    """Build an ``HH`` subclass at call time.

    Definition-time validation lives in ``HH.__init_subclass__``, so an
    invalid template cannot be written as a module-level ``class`` statement
    in this file — it would abort collection. Tests that assert on rejection
    therefore create the class inside the ``assertRaises`` block.
    """
    return type(name, (HH,), {"root_type": Potassium, **namespace})


def _make_markov(name: str, namespace: dict):
    """Build a ``Markov`` subclass at call time. See :func:`_make_hh`."""
    return type(name, (Markov,), {"root_type": Potassium, **namespace})


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
        with self.assertRaisesRegex(ValueError, "both inf/tau and alpha/beta"):
            _make_hh(
                "_Conflict",
                {
                    "gates": (Gate("x"),),
                    "f_x_inf": lambda self, V, K: 0.1,
                    "f_x_tau": lambda self, V, K: 1.0,
                    "f_x_alpha": lambda self, V, K: 0.1,
                    "f_x_beta": lambda self, V, K: 0.1,
                },
            )

    def test_hh_rejects_gate_with_incomplete_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "must define either"):
            _make_hh("_Missing", {"gates": (Gate("x"),), "f_x_alpha": lambda self, V, K: 0.1})

    def test_hh_rejects_misspelled_gate_method_at_definition_time(self) -> None:
        # ``Gate("m")`` with methods written for ``n`` used to survive class
        # creation and ``init_state``, failing only at ``reset_state``.
        with self.assertRaisesRegex(ValueError, r"gate 'm' must define either"):
            _make_hh(
                "_Typo",
                {
                    "gates": (Gate("m", power=3),),
                    "f_n_inf": lambda self, V, K: 0.5,
                    "f_n_tau": lambda self, V, K: 1.0,
                },
            )

    def test_hh_rejects_duplicate_gate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared more than once"):
            _make_hh(
                "_Dup",
                {
                    "gates": (Gate("m"), Gate("m", power=2)),
                    "f_m_inf": lambda self, V, K: 0.5,
                    "f_m_tau": lambda self, V, K: 1.0,
                },
            )

    def test_hh_rejects_non_identifier_gate_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a valid Python identifier"):
            _make_hh("_BadName", {"gates": (Gate("m gate"),)})

    def test_hh_allows_gateless_abstract_subclass(self) -> None:
        # Abstract intermediates declare no gates and must stay definable.
        cls = _make_hh("_Abstract", {})
        self.assertEqual(cls._resolved_gates, ())
        self.assertEqual(cls._gate_forms, {})

    def test_hh_gates_are_resolved_once_into_gate_objects(self) -> None:
        cls = _make_hh(
            "_TupleGates",
            {
                "gates": (("m", 3), Gate("h")),
                "f_m_inf": lambda self, V, K: 0.5,
                "f_m_tau": lambda self, V, K: 1.0,
                "f_h_alpha": lambda self, V, K: 0.2,
                "f_h_beta": lambda self, V, K: 0.3,
            },
        )
        self.assertEqual([g.name for g in cls._resolved_gates], ["m", "h"])
        self.assertEqual(cls._resolved_gates[0].power, 3)
        self.assertTrue(all(isinstance(g, Gate) for g in cls._resolved_gates))
        self.assertEqual(cls._gate_forms, {"m": "inf_tau", "h": "alpha_beta"})

    def test_init_state_rejects_gate_colliding_with_parameter(self) -> None:
        # A gate named after a constructor parameter used to silently replace
        # that parameter with a DiffEqState.
        cls = _make_hh(
            "_Collide",
            {
                "gates": (Gate("g_max", power=2),),
                "f_g_max_inf": lambda self, V, K: 0.5,
                "f_g_max_tau": lambda self, V, K: 1.0,
            },
        )
        ch = cls(1)
        ch.g_max = braintools.init.param(1.0 * (u.mS / u.cm**2), ch.varshape, allow_none=False)
        with self.assertRaisesRegex(ValueError, r"gate 'g_max' collides"):
            ch.init_state(jnp.array([-60.0]) * u.mV, _k_info())

    def test_init_state_is_idempotent(self) -> None:
        ch = _ExampleHHInfTau(size=1)
        V = jnp.array([-60.0]) * u.mV
        K = _k_info()
        ch.init_state(V, K)
        ch.init_state(V, K)  # re-initialisation replaces the existing DiffEqState
        self.assertIsInstance(ch.m, DiffEqState)

    def test_markov_rejects_unknown_dependent_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not one of the declared states"):
            _make_markov(
                "_BadDependent",
                {
                    "pairs": (Transition("A", "B", "fwd", "bwd"),),
                    "dependent_state": "Z",
                    "fwd": lambda self, V: 0.1,
                    "bwd": lambda self, V: 0.2,
                },
            )

    def test_markov_rejects_missing_rate_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "rate method 'bwd', which is not defined"):
            _make_markov(
                "_MissingRate",
                {
                    "pairs": (Transition("A", "B", "fwd", "bwd"),),
                    "fwd": lambda self, V: 0.1,
                },
            )

    def test_markov_rejects_single_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two states"):
            _make_markov(
                "_OneState",
                {"pairs": (Transition("A", "A", "fwd"),), "fwd": lambda self, V: 0.1},
            )

    # ------------------------------------------------------------------
    # Gate rate units
    # ------------------------------------------------------------------

    def _deriv_of(self, cls):
        """Init, reset and differentiate a one-gate channel; return dm/dt."""
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        K = _k_info()
        ch.init_state(V, K)
        ch.reset_state(V, K)
        ch.compute_derivative(V, K)
        return ch.m.derivative

    def _inf_tau_cls(self, name, tau_value, **gate_kwargs):
        return _make_hh(
            name,
            {
                "gates": (Gate("m", **gate_kwargs),),
                "f_m_inf": lambda self, V, K: 0.7,
                "f_m_tau": lambda self, V, K: tau_value,
            },
        )

    def _alpha_beta_cls(self, name, alpha, beta, **gate_kwargs):
        return _make_hh(
            name,
            {
                "gates": (Gate("m", **gate_kwargs),),
                "f_m_alpha": lambda self, V, K: alpha,
                "f_m_beta": lambda self, V, K: beta,
            },
        )

    def test_bare_tau_is_read_as_milliseconds(self) -> None:
        deriv = self._deriv_of(self._inf_tau_cls("_TauBare", 5.0))
        self.assertTrue(u.math.allclose(deriv, (0.7 - 0.7) / 5.0 / u.ms, atol=1e-12 * u.Hz))
        # dimension must be inverse time regardless of how tau was written
        self.assertEqual(u.get_dim(deriv), u.get_dim(1 / u.ms))

    def test_united_tau_matches_bare_tau(self) -> None:
        bare = self._deriv_of(self._inf_tau_cls("_TauBare2", 5.0))
        ms = self._deriv_of(self._inf_tau_cls("_TauMs", 5.0 * u.ms))
        sec = self._deriv_of(self._inf_tau_cls("_TauSec", 0.005 * u.second))
        self.assertTrue(u.math.allclose(bare, ms, atol=1e-12 * u.Hz))
        self.assertTrue(u.math.allclose(bare, sec, atol=1e-12 * u.Hz))

    def test_united_alpha_beta_matches_bare(self) -> None:
        bare = self._deriv_of(self._alpha_beta_cls("_ABBare", 0.4, 0.1))
        united = self._deriv_of(self._alpha_beta_cls("_ABUnited", 0.4 / u.ms, 0.1 / u.ms))
        self.assertTrue(u.math.allclose(bare, united, atol=1e-12 * u.Hz))

    def test_gate_time_unit_reinterprets_bare_values(self) -> None:
        ms_gate = self._deriv_of(self._inf_tau_cls("_TauUnitMs", 5.0))
        s_gate = self._deriv_of(self._inf_tau_cls("_TauUnitS", 5.0, time_unit=u.second))
        # same bare tau, 1000x slower when read as seconds
        self.assertTrue(u.math.allclose(ms_gate, s_gate * 1000.0, atol=1e-12 * u.Hz))

    def test_gate_rejects_non_time_time_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_unit must have a time dimension"):
            Gate("m", time_unit=u.mV)

    def test_tau_with_wrong_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Gate 'm': tau must be dimensionless"):
            self._deriv_of(self._inf_tau_cls("_TauBadDim", 5.0 * u.mV))

    def test_alpha_with_wrong_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Gate 'm': alpha must be dimensionless"):
            self._deriv_of(self._alpha_beta_cls("_AlphaBadDim", 0.4 * u.mV, 0.1))

    # ------------------------------------------------------------------
    # Gate ion-argument arity
    # ------------------------------------------------------------------

    def test_gate_method_receives_only_the_ions_it_declares(self) -> None:
        seen = {}

        def f_m_inf(self, V, K):
            seen["inf"] = 1
            return 0.5

        def f_m_tau(self, V, K, Ca):
            seen["tau"] = 2
            return 1.0

        cls = _make_hh("_Arity", {"gates": (Gate("m"),), "f_m_inf": f_m_inf, "f_m_tau": f_m_tau})
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        ch.init_state(V, _k_info(), _ca_info())
        ch.compute_derivative(V, _k_info(), _ca_info())
        self.assertEqual(seen, {"inf": 1, "tau": 2})

    def test_gate_method_with_varargs_receives_every_ion(self) -> None:
        seen = {}

        def f_m_inf(self, V, *ions):
            seen["n"] = len(ions)
            return 0.5

        cls = _make_hh(
            "_ArityVar",
            {"gates": (Gate("m"),), "f_m_inf": f_m_inf, "f_m_tau": lambda self, V, *ions: 1.0},
        )
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        ch.init_state(V, _k_info(), _ca_info())
        ch.compute_derivative(V, _k_info(), _ca_info())
        self.assertEqual(seen["n"], 2)

    def test_gate_method_demanding_more_ions_than_supplied_raises(self) -> None:
        cls = _make_hh(
            "_ArityTooMany",
            {
                "gates": (Gate("m"),),
                "f_m_inf": lambda self, V, K, Ca: 0.5,
                "f_m_tau": lambda self, V, K, Ca: 1.0,
            },
        )
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        ch.init_state(V, _k_info())
        with self.assertRaisesRegex(TypeError, "expects 2 ion argument"):
            ch.compute_derivative(V, _k_info())

    # ------------------------------------------------------------------
    # Gate / state clipping
    # ------------------------------------------------------------------

    def _one_gate(self, name, **gate_kwargs):
        return _make_hh(
            name,
            {
                "gates": (Gate("m", **gate_kwargs),),
                "f_m_inf": lambda self, V, K: 0.5,
                "f_m_tau": lambda self, V, K: 1.0,
            },
        )

    def _factor_at(self, cls, gate_value):
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        ch.init_state(V, _k_info())
        ch.m.value = jnp.array([gate_value])
        return ch.conductance_factor(V, _k_info())

    def test_gates_are_not_clipped_by_default(self) -> None:
        cls = self._one_gate("_NoClip", power=3)
        self.assertTrue(u.math.allclose(self._factor_at(cls, 1.5), jnp.array([1.5**3])))
        # odd power keeps the sign of an undershooting gate
        self.assertTrue(u.math.allclose(self._factor_at(cls, -0.3), jnp.array([(-0.3) ** 3])))

    def test_even_power_rectifies_a_negative_gate_when_unclipped(self) -> None:
        cls = self._one_gate("_NoClipEven", power=2)
        self.assertTrue(u.math.allclose(self._factor_at(cls, -0.3), jnp.array([0.09])))

    def test_gate_clip_projects_into_unit_interval(self) -> None:
        cls = self._one_gate("_Clip", power=3, clip=True)
        self.assertTrue(u.math.allclose(self._factor_at(cls, 1.5), jnp.array([1.0])))
        self.assertTrue(u.math.allclose(self._factor_at(cls, -0.3), jnp.array([0.0])))
        # in-range values are untouched
        self.assertTrue(u.math.allclose(self._factor_at(cls, 0.5), jnp.array([0.125])))

    def test_gate_clip_does_not_rewrite_stored_state(self) -> None:
        cls = self._one_gate("_ClipStore", power=1, clip=True)
        ch = cls(1)
        V = jnp.array([-60.0]) * u.mV
        ch.init_state(V, _k_info())
        ch.m.value = jnp.array([1.5])
        ch.conductance_factor(V, _k_info())
        self.assertTrue(u.math.allclose(ch.m.value, jnp.array([1.5])))

    def test_markov_clip_states_defaults_on_and_can_be_disabled(self) -> None:
        self.assertTrue(Markov.clip_states)
        ch = _ExampleMarkov(size=1)
        V = jnp.array([-60.0]) * u.mV
        K = _k_info()
        ch.init_state(V, K)
        ch.O.value = jnp.array([1.4])
        self.assertTrue(u.math.allclose(ch._kinetic_state_values()["O"], jnp.array([1.0])))

        type(ch).clip_states = False
        try:
            self.assertTrue(u.math.allclose(ch._kinetic_state_values()["O"], jnp.array([1.4])))
        finally:
            type(ch).clip_states = True

    # ------------------------------------------------------------------
    # Gate metadata binding
    # ------------------------------------------------------------------

    def _phi_of(self, name, **gate_kwargs):
        cls = _make_hh(
            name,
            {
                "gates": (Gate("m", **gate_kwargs),),
                "f_m_inf": lambda self, V, K: 0.5,
                "f_m_tau": lambda self, V, K: 1.0,
                "__init__": lambda self, size=1: (
                    HH.__init__(self, size=size, name=None),
                    setattr(self, "q10", 3.0),
                    setattr(self, "temp_ref", u.celsius2kelvin(22.0)),
                    setattr(self, "temp", u.celsius2kelvin(32.0)),
                    setattr(self, "phi_value", 2.5),
                )[0],
            },
        )
        return cls(1).gate_phi(cls._resolved_gates[0])

    def test_string_reference_matches_lambda_reference(self) -> None:
        by_string = self._phi_of("_PhiStr", q10="q10", temp_ref="temp_ref")
        by_lambda = self._phi_of(
            "_PhiLambda",
            q10=lambda self: self.q10,
            temp_ref=lambda self: self.temp_ref,
        )
        self.assertTrue(u.math.allclose(by_string, by_lambda, atol=1e-12))

    def test_string_reference_resolves_explicit_phi(self) -> None:
        self.assertEqual(self._phi_of("_PhiDirect", phi="phi_value"), 2.5)

    def test_string_reference_to_missing_attribute_names_it(self) -> None:
        with self.assertRaisesRegex(AttributeError, "references attribute 'nope'"):
            self._phi_of("_PhiMissing", phi="nope")

    # ------------------------------------------------------------------
    # Markov dependent state
    # ------------------------------------------------------------------

    def _three_state_markov(self, name, order, **kwargs):
        rates = {f"r{i}": (lambda self, V: 0.1) for i in range(1, 5)}
        return _make_markov(name, {"pairs": order, **rates, **kwargs})

    def test_explicit_dependent_state_survives_pair_reordering(self) -> None:
        forward = (Transition("A", "B", "r1", "r2"), Transition("B", "C", "r3", "r4"))
        reordered = (Transition("B", "C", "r3", "r4"), Transition("A", "B", "r1", "r2"))

        a = self._three_state_markov("_OrderA", forward, dependent_state="A")
        b = self._three_state_markov("_OrderB", reordered, dependent_state="A")
        self.assertEqual(a(1)._dependent_state_name(), "A")
        self.assertEqual(b(1)._dependent_state_name(), "A")
        # ... whereas the implicit fallback would have moved with the order
        self.assertEqual(a._resolved_state_names[-1], "C")
        self.assertEqual(b._resolved_state_names[-1], "A")

    def test_implicit_dependent_state_warns(self) -> None:
        cls = self._three_state_markov(
            "_Implicit",
            (Transition("A", "B", "r1", "r2"), Transition("B", "C", "r3", "r4")),
        )
        with self.assertWarnsRegex(DeprecationWarning, "does not declare `dependent_state`"):
            self.assertEqual(cls(1)._dependent_state_name(), "C")

    def test_shipped_markov_channels_declare_dependent_state(self) -> None:
        # Regression lock: reordering `pairs` in any shipped channel must not
        # be able to silently change which state is eliminated.
        import braincell.channel as channel_pkg

        implicit = [
            name
            for name in dir(channel_pkg)
            if isinstance(cls := getattr(channel_pkg, name, None), type)
            and issubclass(cls, Markov)
            and cls is not Markov
            and cls.pairs
            and cls.dependent_state is None
        ]
        self.assertEqual(implicit, [])

    def test_markov_pairs_are_resolved_once(self) -> None:
        cls = _make_markov(
            "_TuplePairs",
            {
                "pairs": (("A", "B", "fwd", "bwd"),),
                "fwd": lambda self, V: 0.1,
                "bwd": lambda self, V: 0.2,
            },
        )
        self.assertTrue(all(isinstance(p, Transition) for p in cls._resolved_pairs))
        self.assertEqual(cls._resolved_state_names, ("A", "B"))

    def test_ghk_channel_uses_p_max(self) -> None:
        ch = _ExampleGHK(size=1)
        V = jnp.array([-50.0]) * u.mV
        Ca = _ca_info()

        ch.init_state(V, Ca)
        ch.reset_state(V, Ca)
        current = ch.current(V, Ca)

        expected = (
            ch.p_max
            * ch.p.value**2
            * ch.q.value
            * ghk_flux(
                V=V,
                ci=Ca.Ci,
                co=ch.Co,
                z=ch.valence,
                temp=ch.temp,
            )
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
        unit = u.mS / u.cm**2 * u.mV
        self.assertTrue(
            u.math.allclose(
                current.to_decimal(unit),
                expected_current.to_decimal(unit),
                atol=1e-6,
            )
        )

    def test_markov_uses_central_default_integration_schedule(self) -> None:
        ch = _ExampleMarkov(size=1)
        self.assertIs(ch.solver, get_integrator("backward_euler"))
        self.assertEqual(ch.substeps, 1)

        override = _ExampleMarkov(size=1, solver="euler", substeps=2)
        self.assertIs(override.solver, get_integrator("euler"))
        self.assertEqual(override.substeps, 2)

    def test_markov_defaults_dependent_state_to_last_discovered_name(self) -> None:
        ch = _ExampleMarkovImplicitDependent(size=1)
        V = jnp.array([-65.0]) * u.mV
        K = _k_info()

        with self.assertWarnsRegex(DeprecationWarning, "does not declare `dependent_state`"):
            ch.init_state(V, K)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ch.init_state(V, K)
            ch.reset_steady_state(V, K)
            states = ch.state_values()

            total = states["C"] + states["O"] + states["I"]
            self.assertTrue(u.math.allclose(total, jnp.array([1.0]), atol=1e-6))
            self.assertTrue(u.math.allclose(states["I"], jnp.array([1.0]), atol=1e-6))

            ch.compute_derivative(V, K)
        self.assertTrue(u.math.allclose(ch.C.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))
        self.assertTrue(u.math.allclose(ch.O.derivative, jnp.array([0.0]) / u.ms, atol=1e-6 * u.Hz))

    def test_markov_host_steady_state_matches_jax_solver(self) -> None:
        ch = _ExampleMarkovTwoOpenStates(size=3)
        V = jnp.full((3,), -65.0) * u.mV
        K = _k_info(size=3)

        ch.init_state(V, K)
        host_states = ch._solve_steady_state_host(V, K)
        jax_states = ch._solve_steady_state_jax(V, K)

        self.assertEqual(host_states.keys(), jax_states.keys())
        for name in host_states:
            self.assertTrue(u.math.allclose(host_states[name], jax_states[name], atol=1e-6))

    def test_markov_steady_state_uses_jax_fallback_when_traced(self) -> None:
        ch = _ExampleMarkovTwoOpenStates(size=3)
        V = jnp.full((3,), -65.0) * u.mV
        K = _k_info(size=3)
        ch.init_state(V, K)

        solve_open = jax.jit(lambda voltage: ch._solve_steady_state(voltage, K)["O1"])
        actual = solve_open(V)
        expected = ch._solve_steady_state_jax(V, K)["O1"]
        self.assertTrue(u.math.allclose(actual, expected, atol=1e-6))

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
        unit = u.mS / u.cm**2 * u.mV
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
