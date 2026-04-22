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

"""Comprehensive unit tests for :class:`braincell.SingleCompartment`.

The tests are organised by surface area:

* :class:`SingleCompartmentDefaultsTest` - construction & default parameters
* :class:`SingleCompartmentPropertiesTest` - ``area`` / ``pop_size`` / ``n_compartment``
* :class:`SingleCompartmentNormalizeCurrentTest` - ``_normalize_external_current``
* :class:`SingleCompartmentLifecycleTest` - ``init_state`` / ``reset_state``
* :class:`SingleCompartmentPreIntegralTest` - ``pre_integral`` forwarding
* :class:`SingleCompartmentComputeDerivativeTest` - core ``compute_derivative`` path
* :class:`SingleCompartmentPostIntegralTest` - ``post_integral`` forwarding
* :class:`SingleCompartmentSpikeTest` - ``get_spike`` / ``soma_spike``
* :class:`SingleCompartmentUpdateTest` - the full ``update`` path through the solver
* :class:`SingleCompartmentSolverResolutionTest` - string / callable solver kwarg
"""


import types
import unittest

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

import braincell
from braincell import Channel, HHTypedNeuron, SingleCompartment
from braincell.channel import (
    IK_HH1952,
    IL,
    INa_HH1952,
)
from braincell.ion import PotassiumFixed, SodiumFixed
from braincell import DiffEqState
from braincell.quad import rk2_step, rk4_step


def _V(values, unit=u.mV):
    return jnp.asarray(values) * unit


class _BadChannel(Channel):
    """A minimal :class:`Channel` whose ``current`` always raises.

    Used to exercise the exception-wrapping branch of
    :meth:`SingleCompartment.compute_derivative` without depending on the
    internal state of any real channel implementation.
    """

    root_type = HHTypedNeuron

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)

    def current(self, V):
        raise RuntimeError("intentional bad current")

    def init_state(self, V, batch_size=None):  # pragma: no cover - no state
        pass

    def reset_state(self, V, batch_size=None):  # pragma: no cover - no state
        pass

    def compute_derivative(self, V):  # pragma: no cover - never reached
        pass


# ---------------------------------------------------------------------------
# Defaults and construction
# ---------------------------------------------------------------------------


class SingleCompartmentDefaultsTest(unittest.TestCase):
    """Construction with defaults and with explicit keyword overrides."""

    def test_default_scalar_parameters(self) -> None:
        sc = SingleCompartment(size=1)
        self.assertTrue(u.math.allclose(sc.length, 10.0 * u.um, atol=1e-9 * u.um))
        self.assertTrue(u.math.allclose(sc.radius, 5.0 * u.um, atol=1e-9 * u.um))
        self.assertTrue(
            u.math.allclose(sc.C, 1.0 * u.uF / u.cm ** 2, atol=1e-12 * u.uF / u.cm ** 2)
        )
        self.assertTrue(u.math.allclose(sc.V_th, 0.0 * u.mV, atol=1e-9 * u.mV))

    def test_default_solver_is_rk2(self) -> None:
        sc = SingleCompartment(size=1)
        self.assertIs(sc.solver, rk2_step)

    def test_default_spk_fun_is_callable(self) -> None:
        sc = SingleCompartment(size=1)
        self.assertTrue(callable(sc.spk_fun))

    def test_default_v_initializer_is_uniform(self) -> None:
        sc = SingleCompartment(size=1)
        self.assertIsInstance(sc.V_initializer, braintools.init.Uniform)

    def test_varshape_matches_size(self) -> None:
        self.assertEqual(SingleCompartment(size=1).varshape, (1,))
        self.assertEqual(SingleCompartment(size=5).varshape, (5,))
        self.assertEqual(SingleCompartment(size=(3,)).varshape, (3,))

    def test_custom_scalar_parameters_are_honoured(self) -> None:
        sc = SingleCompartment(
            size=2,
            length=20.0 * u.um,
            radius=3.0 * u.um,
            C=0.5 * u.uF / u.cm ** 2,
            V_th=-30.0 * u.mV,
        )
        self.assertTrue(u.math.allclose(sc.length, 20.0 * u.um, atol=1e-9 * u.um))
        self.assertTrue(u.math.allclose(sc.radius, 3.0 * u.um, atol=1e-9 * u.um))
        self.assertTrue(
            u.math.allclose(sc.C, 0.5 * u.uF / u.cm ** 2, atol=1e-12 * u.uF / u.cm ** 2)
        )
        self.assertTrue(u.math.allclose(sc.V_th, -30.0 * u.mV, atol=1e-9 * u.mV))

    def test_callable_parameters_broadcast_across_size(self) -> None:
        sc = SingleCompartment(
            size=3,
            length=lambda shape: jnp.array([5.0, 10.0, 15.0]) * u.um,
            radius=lambda shape: jnp.array([1.0, 2.0, 3.0]) * u.um,
        )
        self.assertEqual(sc.length.shape, (3,))
        self.assertEqual(sc.radius.shape, (3,))
        self.assertTrue(
            u.math.allclose(
                sc.length,
                jnp.array([5.0, 10.0, 15.0]) * u.um,
                atol=1e-9 * u.um,
            )
        )

    def test_ion_channels_are_attached(self) -> None:
        sc = SingleCompartment(size=1, IL=IL(size=1))
        self.assertIn("IL", sc.ion_channels)
        self.assertIsInstance(sc.ion_channels["IL"], IL)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class SingleCompartmentPropertiesTest(unittest.TestCase):
    def test_n_compartment_is_one(self) -> None:
        self.assertEqual(SingleCompartment(size=1).n_compartment, 1)
        self.assertEqual(SingleCompartment(size=7).n_compartment, 1)

    def test_pop_size_matches_varshape(self) -> None:
        self.assertEqual(SingleCompartment(size=1).pop_size, (1,))
        self.assertEqual(SingleCompartment(size=4).pop_size, (4,))

    def test_area_is_cylinder_lateral_surface(self) -> None:
        sc = SingleCompartment(size=1, length=10.0 * u.um, radius=5.0 * u.um)
        expected = 2.0 * u.math.pi * 5.0 * u.um * 10.0 * u.um
        self.assertTrue(
            u.math.allclose(
                sc.area.to_decimal(u.um ** 2),
                expected.to_decimal(u.um ** 2),
                atol=1e-9,
            )
        )

    def test_area_broadcasts_with_callable_geometry(self) -> None:
        sc = SingleCompartment(
            size=2,
            length=lambda shape: jnp.array([10.0, 20.0]) * u.um,
            radius=lambda shape: jnp.array([1.0, 2.0]) * u.um,
        )
        expected = 2.0 * u.math.pi * jnp.array([1.0, 2.0]) * u.um * jnp.array([10.0, 20.0]) * u.um
        self.assertTrue(
            u.math.allclose(
                sc.area.to_decimal(u.um ** 2),
                expected.to_decimal(u.um ** 2),
                atol=1e-9,
            )
        )


# ---------------------------------------------------------------------------
# init_state / reset_state
# ---------------------------------------------------------------------------


class SingleCompartmentLifecycleTest(unittest.TestCase):
    def test_init_state_creates_diffeq_state_V(self) -> None:
        sc = SingleCompartment(size=2)
        sc.init_state()
        self.assertIsInstance(sc.V, DiffEqState)
        self.assertEqual(sc.V.value.shape, (2,))
        self.assertIsInstance(sc.spike, brainstate.ShortTermState)
        self.assertEqual(sc.spike.value.shape, (2,))

    def test_init_state_with_constant_initializer_matches_value(self) -> None:
        sc = SingleCompartment(
            size=3,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state()
        self.assertTrue(
            u.math.allclose(
                sc.V.value,
                jnp.full((3,), -65.0) * u.mV,
                atol=1e-9 * u.mV,
            )
        )

    def test_init_state_samples_in_uniform_range(self) -> None:
        sc = SingleCompartment(size=16)  # default Uniform(-70, -60) mV
        sc.init_state()
        # every sample is inside the default [-70, -60] mV range
        self.assertTrue(
            bool(jnp.all(sc.V.value.to_decimal(u.mV) >= -70.0))
        )
        self.assertTrue(
            bool(jnp.all(sc.V.value.to_decimal(u.mV) <= -60.0))
        )

    def test_init_state_with_batch_dim(self) -> None:
        sc = SingleCompartment(
            size=3,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state(batch_size=4)
        self.assertEqual(sc.V.value.shape, (4, 3))

    def test_init_state_forwards_to_child_channel_ion(self) -> None:
        sc = SingleCompartment(size=1)
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.init_state()
        inchannel = sc.na.channels["INa"]
        self.assertEqual(inchannel.p.value.shape, (1,))
        self.assertEqual(inchannel.q.value.shape, (1,))

    def test_reset_state_restores_constant_V(self) -> None:
        sc = SingleCompartment(
            size=2,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state()
        sc.V.value = jnp.array([10.0, 20.0]) * u.mV
        sc.reset_state()
        self.assertTrue(
            u.math.allclose(
                sc.V.value,
                jnp.full((2,), -65.0) * u.mV,
                atol=1e-9 * u.mV,
            )
        )

    def test_reset_state_clears_spike_buffer(self) -> None:
        sc = SingleCompartment(
            size=2,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            V_th=0.0 * u.mV,
        )
        sc.init_state()
        sc.spike.value = jnp.array([1.0, 1.0])
        sc.reset_state()
        # With V = -65 mV and V_th = 0 mV, get_spike(V, V) must be zero.
        self.assertTrue(
            u.math.allclose(sc.spike.value, jnp.zeros((2,)), atol=1e-9)
        )

    def test_reset_state_forwards_to_child_channel_ion(self) -> None:
        # Pin V to a constant so reset_state() re-samples the same V each time;
        # otherwise the default Uniform(-70, -60) mV initializer would randomise
        # V on every call and the INa steady states would not match.
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.init_state()
        sc.reset_state()

        first_p = sc.na.channels["INa"].p.value
        sc.na.channels["INa"].p.value = jnp.array([0.99])
        sc.reset_state()
        self.assertTrue(
            u.math.allclose(sc.na.channels["INa"].p.value, first_p, atol=1e-9)
        )


# ---------------------------------------------------------------------------
# pre_integral
# ---------------------------------------------------------------------------


class SingleCompartmentPreIntegralTest(unittest.TestCase):
    def test_pre_integral_without_channels_is_a_no_op(self) -> None:
        sc = SingleCompartment(size=1)
        sc.init_state()
        # Should not raise even with no ion channels and no I_ext argument.
        sc.pre_integral()
        sc.pre_integral(1.0 * u.nA / u.cm ** 2)

    def test_pre_integral_forwards_to_ion_channel(self) -> None:
        sc = SingleCompartment(size=1)
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.init_state()
        sc.reset_state()

        called = []
        original = sc.na.pre_integral

        def spy(V):
            called.append(V)
            return original(V)

        sc.na.pre_integral = types.MethodType(lambda self, V: spy(V), sc.na)
        sc.pre_integral()
        self.assertEqual(len(called), 1)


# ---------------------------------------------------------------------------
# compute_derivative
# ---------------------------------------------------------------------------


class SingleCompartmentComputeDerivativeTest(unittest.TestCase):
    def test_zero_current_no_channels_yields_zero_derivative(self) -> None:
        sc = SingleCompartment(
            size=2,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        sc.init_state()
        sc.compute_derivative(0.0 * u.nA / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                sc.V.derivative.to_decimal(u.mV / u.ms),
                jnp.zeros((2,)),
                atol=1e-9,
            )
        )

    def test_constant_current_no_channels_matches_I_over_C(self) -> None:
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
            C=1.0 * u.uF / u.cm ** 2,
        )
        sc.init_state()
        I = 1.0 * u.uA / u.cm ** 2
        sc.compute_derivative(I)
        # dV/dt = (1 uA/cm²) / (1 uF/cm²) = 1 V/s = 1 mV/ms
        self.assertTrue(
            u.math.allclose(
                sc.V.derivative.to_decimal(u.mV / u.ms),
                jnp.asarray([1.0]),
                atol=1e-9,
            )
        )

    def test_default_I_ext_argument_is_zero(self) -> None:
        # compute_derivative with no I_ext argument must default to 0 nA/cm².
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-60.0 * u.mV),
        )
        sc.init_state()
        sc.compute_derivative()
        self.assertTrue(
            u.math.allclose(
                sc.V.derivative.to_decimal(u.mV / u.ms),
                jnp.zeros((1,)),
                atol=1e-9,
            )
        )

    def test_leak_channel_contributes_current(self) -> None:
        # With V held at E_L the leak current must be zero; away from E_L it
        # must push dV/dt in the correct direction.
        V_hold = -70.0 * u.mV
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(V_hold),
            IL=IL(size=1, g_max=0.1 * u.mS / u.cm ** 2, E=V_hold),
        )
        sc.init_state()
        sc.compute_derivative(0.0 * u.nA / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                sc.V.derivative.to_decimal(u.mV / u.ms),
                jnp.zeros((1,)),
                atol=1e-9,
            )
        )

        # Now drive the cell away from E_L – leak current should pull V back.
        sc.V.value = jnp.array([-60.0]) * u.mV
        sc.compute_derivative(0.0 * u.nA / u.cm ** 2)
        # dV/dt should be negative (pulling back toward -70 mV)
        self.assertLess(float(sc.V.derivative.to_decimal(u.mV / u.ms)[0]), 0.0)

    def test_leak_channel_derivative_matches_analytical_formula(self) -> None:
        # With a leak channel and a known V, C, g_L, E_L we can compute the
        # expected dV/dt = (I_ext + g_L * (E_L - V)) / C analytically.
        g = 0.3 * u.mS / u.cm ** 2
        EL = -54.0 * u.mV
        V0 = -65.0 * u.mV
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(V0),
            C=1.0 * u.uF / u.cm ** 2,
            IL=IL(size=1, g_max=g, E=EL),
        )
        sc.init_state()
        I_ext = 2.0 * u.uA / u.cm ** 2
        sc.compute_derivative(I_ext)

        expected = (I_ext + g * (EL - V0)) / (1.0 * u.uF / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                sc.V.derivative.to_decimal(u.mV / u.ms),
                expected.to_decimal(u.mV / u.ms),
                atol=1e-6,
            )
        )

    def test_ion_channel_derivatives_are_populated(self) -> None:
        sc = SingleCompartment(size=1)
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.k = PotassiumFixed(size=1, E=-77.0 * u.mV)
        sc.k.add(IK=IK_HH1952(size=1))
        sc.init_state()
        sc.reset_state()
        sc.compute_derivative(0.0 * u.nA / u.cm ** 2)

        self.assertEqual(sc.na.channels["INa"].p.derivative.shape, (1,))
        self.assertEqual(sc.na.channels["INa"].q.derivative.shape, (1,))
        self.assertEqual(sc.k.channels["IK"].p.derivative.shape, (1,))
        self.assertEqual(sc.V.derivative.shape, (1,))

    def test_bad_channel_current_is_wrapped_in_value_error(self) -> None:
        sc = SingleCompartment(size=1, bad=_BadChannel(size=1))
        sc.init_state()
        with self.assertRaises(ValueError) as ctx:
            sc.compute_derivative(0.0 * u.nA / u.cm ** 2)
        self.assertIn("bad", str(ctx.exception))
        self.assertIn("intentional bad current", str(ctx.exception))
        # Regression: HIGH-01 — preserve the original exception as __cause__.
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("intentional bad current", str(ctx.exception.__cause__))

    def test_non_numeric_exception_passes_through_unwrapped(self) -> None:
        """MED-04: AttributeError from channel.current must not be rewritten as ValueError."""

        class _AttrErrorChannel(Channel):
            root_type = HHTypedNeuron

            def __init__(self, size, name=None):
                super().__init__(size=size, name=name)

            def current(self, V):
                raise AttributeError("lookup failed")

            def init_state(self, V, batch_size=None):  # pragma: no cover
                pass

            def reset_state(self, V, batch_size=None):  # pragma: no cover
                pass

        sc = SingleCompartment(size=1, attr=_AttrErrorChannel(size=1))
        sc.init_state()
        with self.assertRaises(AttributeError):
            sc.compute_derivative(0.0 * u.nA / u.cm ** 2)


# ---------------------------------------------------------------------------
# post_integral
# ---------------------------------------------------------------------------


class SingleCompartmentPostIntegralTest(unittest.TestCase):
    def test_post_integral_without_delta_inputs_preserves_V(self) -> None:
        sc = SingleCompartment(
            size=2,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state()
        before = sc.V.value
        sc.post_integral()
        self.assertTrue(u.math.allclose(sc.V.value, before, atol=1e-9 * u.mV))

    def test_post_integral_forwards_to_ion_channel(self) -> None:
        sc = SingleCompartment(size=1)
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.init_state()
        sc.reset_state()

        called = []

        def spy(self, V):
            called.append(V)

        sc.na.post_integral = types.MethodType(spy, sc.na)
        sc.post_integral()
        self.assertEqual(len(called), 1)


# ---------------------------------------------------------------------------
# get_spike / soma_spike
# ---------------------------------------------------------------------------


class SingleCompartmentSpikeTest(unittest.TestCase):
    def test_get_spike_rising_crossing_is_nonzero(self) -> None:
        sc = SingleCompartment(size=1, V_th=0.0 * u.mV)
        sc.init_state()
        spk = sc.get_spike(jnp.array([-10.0]) * u.mV, jnp.array([10.0]) * u.mV)
        self.assertGreater(float(spk[0]), 0.0)

    def test_get_spike_falling_crossing_is_zero(self) -> None:
        sc = SingleCompartment(size=1, V_th=0.0 * u.mV)
        sc.init_state()
        spk = sc.get_spike(jnp.array([10.0]) * u.mV, jnp.array([-10.0]) * u.mV)
        self.assertTrue(
            u.math.allclose(spk, jnp.zeros((1,)), atol=1e-9)
        )

    def test_get_spike_both_below_threshold_is_zero(self) -> None:
        sc = SingleCompartment(size=1, V_th=0.0 * u.mV)
        sc.init_state()
        spk = sc.get_spike(jnp.array([-30.0]) * u.mV, jnp.array([-20.0]) * u.mV)
        self.assertTrue(
            u.math.allclose(spk, jnp.zeros((1,)), atol=1e-9)
        )

    def test_get_spike_both_above_threshold_is_zero(self) -> None:
        sc = SingleCompartment(size=1, V_th=0.0 * u.mV)
        sc.init_state()
        spk = sc.get_spike(jnp.array([10.0]) * u.mV, jnp.array([20.0]) * u.mV)
        self.assertTrue(
            u.math.allclose(spk, jnp.zeros((1,)), atol=1e-9)
        )

    def test_soma_spike_below_threshold_is_zero(self) -> None:
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            V_th=0.0 * u.mV,
        )
        sc.init_state()
        out = sc.soma_spike()
        self.assertTrue(
            u.math.allclose(out, jnp.zeros((1,)), atol=1e-9)
        )

    def test_soma_spike_above_threshold_is_positive(self) -> None:
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(20.0 * u.mV),
            V_th=0.0 * u.mV,
        )
        sc.init_state()
        out = sc.soma_spike()
        self.assertGreater(float(out[0]), 0.0)


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class SingleCompartmentUpdateTest(unittest.TestCase):
    def test_update_zero_current_leaves_V_unchanged(self) -> None:
        sc = SingleCompartment(
            size=2,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            spk = sc.update(0.0 * u.nA / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(
                sc.V.value,
                jnp.full((2,), -65.0) * u.mV,
                atol=1e-9 * u.mV,
            )
        )
        self.assertEqual(spk.shape, (2,))
        self.assertTrue(u.math.allclose(spk, jnp.zeros((2,)), atol=1e-9))

    def test_update_nonzero_current_advances_V(self) -> None:
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
            C=1.0 * u.uF / u.cm ** 2,
        )
        sc.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=1.0 * u.ms):
            sc.update(1.0 * u.uA / u.cm ** 2)
        # dV/dt = 1 uA/cm² / 1 uF/cm² = 1 mV/ms ⇒ ΔV = 1 mV.
        self.assertTrue(
            u.math.allclose(
                sc.V.value.to_decimal(u.mV),
                jnp.asarray([-64.0]),
                atol=1e-5,
            )
        )

    def test_update_writes_spike_buffer(self) -> None:
        sc = SingleCompartment(
            size=1,
            V_initializer=braintools.init.Constant(-65.0 * u.mV),
        )
        sc.init_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            spk = sc.update(0.0 * u.nA / u.cm ** 2)
        self.assertTrue(
            u.math.allclose(sc.spike.value, spk, atol=1e-9)
        )

    def test_update_with_hh_stack_runs(self) -> None:
        # A full HH setup with Na + K + IL should go through one update step
        # without raising and produce a finite V.
        sc = SingleCompartment(size=1, solver="exp_euler")
        sc.na = SodiumFixed(size=1, E=50.0 * u.mV)
        sc.na.add(INa=INa_HH1952(size=1))
        sc.k = PotassiumFixed(size=1, E=-77.0 * u.mV)
        sc.k.add(IK=IK_HH1952(size=1))
        sc.IL = IL(size=1, E=-54.387 * u.mV, g_max=0.03 * u.mS / u.cm ** 2)
        sc.init_state()
        sc.reset_state()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            sc.update(0.0 * u.nA / u.cm ** 2)
        v = sc.V.value.to_decimal(u.mV)
        self.assertTrue(bool(jnp.all(jnp.isfinite(v))))


# ---------------------------------------------------------------------------
# solver kwarg resolution
# ---------------------------------------------------------------------------


class SingleCompartmentSolverResolutionTest(unittest.TestCase):
    def test_string_solver_resolves_via_registry(self) -> None:
        sc = SingleCompartment(size=1, solver="rk4")
        self.assertIs(sc.solver, rk4_step)

    def test_callable_solver_is_stored_directly(self) -> None:
        marker = object()

        def _stub(target, *args):  # pragma: no cover - never called
            return marker

        sc = SingleCompartment(size=1, solver=_stub)
        self.assertIs(sc.solver, _stub)

    def test_unknown_solver_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            SingleCompartment(size=1, solver="not-a-real-integrator")


if __name__ == "__main__":
    unittest.main()
