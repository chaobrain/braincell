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

"""Tests for :mod:`braincell.quad._protocol`.

These tests cover the public mixin/state classes that every BrainCell
model uses to declare itself integrable: :class:`DiffEqState`,
:class:`DiffEqGroupState`, :class:`DiffEqModule`, and
:class:`IndependentIntegration`, plus the host-scoped state factory that
picks between the grouped and non-grouped classes.
"""

import threading
import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braincell.quad import (
    get_integrator,
)
from braincell.quad.protocol import (
    DiffEqGroupState,
    DiffEqModule,
    DiffEqState,
    DiffEqSingleState,
    IndependentIntegration,
    state,
    state_grouping,
    hidden_state,
)

_FLOAT_DTYPE = jnp.asarray(0.0).dtype


class DiffEqStateMixinTest(unittest.TestCase):
    """``DiffEqState`` is the integrability marker, not a state class."""

    def test_the_marker_is_not_instantiable(self):
        with self.assertRaises(TypeError):
            DiffEqState(jnp.zeros(3) * u.mV)

    def test_the_marker_is_a_mixin_and_not_a_state(self):
        self.assertTrue(issubclass(DiffEqState, brainstate.mixin.Mixin))
        self.assertFalse(issubclass(DiffEqState, brainstate.State))

    def test_both_concrete_classes_are_states_carrying_the_marker(self):
        for cls in (DiffEqSingleState, DiffEqGroupState):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, DiffEqState))
                self.assertTrue(issubclass(cls, brainstate.State))

    def test_the_concrete_classes_are_siblings_not_a_chain(self):
        # The old hierarchy had DiffEqGroupState inherit DiffEqState, which
        # placed the ungrouped class ahead of HiddenGroupState in the MRO
        # so any method added to it would silently shadow the grouped one.
        self.assertFalse(issubclass(DiffEqGroupState, DiffEqSingleState))
        self.assertFalse(issubclass(DiffEqSingleState, DiffEqGroupState))

    def test_the_marker_precedes_the_storage_class_in_both_mros(self):
        for cls, storage in (
            (DiffEqSingleState, "HiddenState"),
            (DiffEqGroupState, "HiddenGroupState"),
        ):
            with self.subTest(cls=cls.__name__):
                names = [c.__name__ for c in cls.__mro__]
                self.assertLess(names.index("DiffEqState"), names.index(storage))

    def test_derivative_defaults_do_not_leak_between_instances(self):
        # ``_derivative``/``_diffusion`` are class attributes now that the
        # mixin has no ``__init__``, so the setters must shadow them per
        # instance rather than mutate the shared class attribute.
        a = DiffEqSingleState(jnp.zeros(3) * u.mV)
        b = DiffEqSingleState(jnp.zeros(3) * u.mV)
        a.derivative = jnp.ones(3) * (u.mV / u.ms)
        a.diffusion = jnp.ones(3) * (u.mV / u.ms)
        self.assertIsNone(b.derivative)
        self.assertIsNone(b.diffusion)
        self.assertIsNone(DiffEqState._derivative)
        self.assertIsNone(DiffEqState._diffusion)

    def test_setters_still_record_a_state_write(self):
        # The exponential-Euler and Runge-Kutta drivers discover which
        # states participate by watching this trace.
        st = DiffEqSingleState(jnp.zeros(3) * u.mV)
        with brainstate.StateTraceStack() as trace:
            st.derivative = jnp.ones(3) * (u.mV / u.ms)
        self.assertTrue(any(s is st for s in trace.get_write_states()))

    def test_repr_hides_derivative_until_it_is_written(self):
        st = DiffEqSingleState(jnp.zeros(3) * u.mV)
        self.assertNotIn("derivative", repr(st))
        st.derivative = jnp.ones(3) * (u.mV / u.ms)
        self.assertIn("derivative", repr(st))


class DiffEqSingleStateTest(unittest.TestCase):
    def test_initial_derivative_and_diffusion_are_none(self):
        st = DiffEqSingleState(jnp.zeros(3))
        self.assertIsNone(st.derivative)
        self.assertIsNone(st.diffusion)

    def test_set_derivative_and_diffusion(self):
        st = DiffEqSingleState(jnp.zeros(3))
        d = jnp.ones(3)
        st.derivative = d
        st.diffusion = 2 * d
        self.assertIs(st.derivative, d)
        np.testing.assert_array_equal(st.diffusion, 2 * d)

    def test_state_value_roundtrip(self):
        v = jnp.arange(4, dtype=_FLOAT_DTYPE) * u.mV
        st = DiffEqSingleState(v)
        np.testing.assert_array_equal(st.value.to_decimal(u.mV), np.arange(4))


class DiffEqGroupStateTest(unittest.TestCase):
    """``Cell`` hidden states group their trailing (compartment) axis."""

    def test_is_both_a_diffeq_state_and_a_brainstate_group_state(self):
        st = DiffEqGroupState(jnp.zeros((1, 4)) * u.mV)
        # Solvers select integrable states with ``isinstance(_, DiffEqState)``
        # (see ``quad/_util.py``), so this is what keeps them working.
        self.assertIsInstance(st, DiffEqState)
        self.assertIsInstance(st, brainstate.HiddenState)
        self.assertIsInstance(st, brainstate.HiddenGroupState)

    def test_mro_reaches_group_state_before_plain_hidden_state(self):
        names = [cls.__name__ for cls in DiffEqGroupState.__mro__]
        self.assertLess(names.index("HiddenGroupState"), names.index("HiddenState"))

    def test_varshape_and_num_state_split_the_trailing_axis(self):
        st = DiffEqGroupState(jnp.zeros((2, 3, 7)) * u.mV)
        self.assertEqual(st.varshape, (2, 3))
        self.assertEqual(st.num_state, 7)

    def test_rank_one_value_is_rejected_by_the_inherited_guard(self):
        # braincell deliberately does not relax this guard; instead ``Cell``
        # makes its population axis mandatory so values are always rank >= 2.
        with self.assertRaisesRegex(ValueError, "2 dimensions"):
            DiffEqGroupState(jnp.zeros(4) * u.mV)

    def test_get_value_and_set_value_address_the_group_axis(self):
        st = DiffEqGroupState(jnp.zeros((1, 3)) * u.mV)
        self.assertEqual(st.name2index, {"0": 0, "1": 1, "2": 2})
        st.set_value({1: jnp.ones(1) * (5.0 * u.mV)})
        np.testing.assert_allclose(st.get_value(1).to_decimal(u.mV), [5.0])
        np.testing.assert_allclose(st.get_value("1").to_decimal(u.mV), [5.0])
        np.testing.assert_allclose(st.get_value(0).to_decimal(u.mV), [0.0])

    def test_derivative_and_diffusion_slots_still_work(self):
        st = DiffEqGroupState(jnp.zeros((1, 3)) * u.mV)
        self.assertIsNone(st.derivative)
        self.assertIsNone(st.diffusion)
        st.derivative = jnp.ones((1, 3)) * (u.mV / u.ms)
        st.diffusion = jnp.full((1, 3), 2.0) * (u.mV / u.ms)
        np.testing.assert_allclose(st.derivative.to_decimal(u.mV / u.ms), [[1.0] * 3])
        np.testing.assert_allclose(st.diffusion.to_decimal(u.mV / u.ms), [[2.0] * 3])

    def test_survives_a_graph_split_merge_roundtrip(self):
        class Holder(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.y = DiffEqGroupState(jnp.zeros((1, 3)) * u.mV)

        graphdef, treefy = brainstate.graph.treefy_split(Holder())
        restored = brainstate.graph.treefy_merge(graphdef, treefy)
        self.assertIsInstance(restored.y, DiffEqGroupState)
        self.assertEqual(restored.y.name2index, {"0": 0, "1": 1, "2": 2})

    def test_survives_a_for_loop_transform(self):
        class Holder(brainstate.nn.Module):
            def __init__(self):
                super().__init__()
                self.y = DiffEqGroupState(jnp.zeros((1, 3)) * u.mV)

        holder = Holder()

        def step(_):
            holder.y.value = holder.y.value + 1.0 * u.mV
            return holder.y.value

        trace = brainstate.transform.for_loop(step, jnp.arange(4))
        self.assertEqual(trace.shape, (4, 1, 3))
        np.testing.assert_allclose(holder.y.value.to_decimal(u.mV), [[4.0] * 3])
        self.assertIsInstance(holder.y, DiffEqGroupState)


class StateFactoryTest(unittest.TestCase):
    """``state_grouping`` scopes which hidden-state class gets allocated."""

    def test_default_scope_is_not_grouped(self):
        self.assertIsInstance(state(jnp.zeros(3) * u.mV), DiffEqSingleState)
        self.assertNotIsInstance(hidden_state(jnp.zeros(3) * u.mV), brainstate.HiddenGroupState)

    def test_grouped_scope_selects_the_group_classes(self):
        with state_grouping(True):
            self.assertIsInstance(state(jnp.zeros((1, 3)) * u.mV), DiffEqGroupState)
            # The algebraic counterpart uses the stock brainstate class, not a
            # braincell subclass, so nothing shadows the upstream name.
            algebraic = hidden_state(jnp.zeros((1, 3)) * u.mV)
            self.assertIs(type(algebraic), brainstate.HiddenGroupState)

    def test_scopes_nest_and_restore(self):
        with state_grouping(True):
            self.assertIsInstance(state(jnp.zeros((1, 3)) * u.mV), DiffEqGroupState)
            with state_grouping(False):
                self.assertIsInstance(state(jnp.zeros(3) * u.mV), DiffEqSingleState)
            self.assertIsInstance(state(jnp.zeros((1, 3)) * u.mV), DiffEqGroupState)
        self.assertIsInstance(state(jnp.zeros(3) * u.mV), DiffEqSingleState)

    def test_scope_is_restored_after_an_exception(self):
        with self.assertRaises(RuntimeError):
            with state_grouping(True):
                raise RuntimeError("boom")
        self.assertIsInstance(state(jnp.zeros(3) * u.mV), DiffEqSingleState)

    def test_scope_does_not_leak_across_threads(self):
        # A contextvars default is per-thread, unlike a bare module global.
        observed: list = []

        def worker():
            observed.append(isinstance(state(jnp.zeros(3) * u.mV), DiffEqGroupState))

        with state_grouping(True):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

        self.assertEqual(observed, [False])

    def test_kwargs_are_forwarded_to_the_state_constructor(self):
        with state_grouping(True):
            st = state(jnp.zeros((1, 3)) * u.mV, name="grouped_v")
        self.assertEqual(st.name, "grouped_v")


class DiffEqModuleTest(unittest.TestCase):
    def test_compute_derivative_must_be_overridden(self):
        class Bare(brainstate.nn.Module, DiffEqModule):
            pass

        with self.assertRaises(NotImplementedError):
            Bare().compute_derivative()

    def test_default_pre_and_post_integral_are_noops(self):
        class Bare(brainstate.nn.Module, DiffEqModule):
            def compute_derivative(self):
                pass

        b = Bare()
        # Both methods exist on the mixin and accept arbitrary args.
        self.assertIsNone(b.pre_integral(1, 2, k=3))
        self.assertIsNone(b.post_integral(1, 2, k=3))


class IndependentIntegrationTest(unittest.TestCase):
    def _make_sub(self, solver):
        class Sub(brainstate.nn.Module, DiffEqModule, IndependentIntegration):
            def __init__(self):
                IndependentIntegration.__init__(self, solver)
                brainstate.nn.Module.__init__(self)
                self.y = DiffEqSingleState(jnp.ones(2, dtype=_FLOAT_DTYPE) * u.mV)

            def compute_derivative(self, *args, **kwargs):
                self.y.derivative = -self.y.value / (5.0 * u.ms)

        return Sub()

    def test_constructor_resolves_solver_string(self):
        sub = self._make_sub("euler")
        self.assertIs(sub.solver, get_integrator("euler"))

    def test_constructor_accepts_callable(self):
        def my_solver(target, *args):
            return "sentinel"

        sub = self._make_sub(my_solver)
        self.assertIs(sub.solver, my_solver)

    def test_constructor_resolves_alias_string(self):
        # Aliases should work the same as canonical names.
        sub = self._make_sub("explicit")
        self.assertIs(sub.solver, get_integrator("euler"))

    def test_make_integration_invokes_solver(self):
        observed = []

        def my_solver(target, *args):
            observed.append((target, args))

        sub = self._make_sub(my_solver)
        sub.make_integration("extra-arg")
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0][0], sub)
        self.assertEqual(observed[0][1], ("extra-arg",))


class IndependentIntegrationForwardsKwargsTest(unittest.TestCase):
    """ARCH-09: ``__init__`` must cooperate with sibling mixins in the MRO."""

    def test_kwargs_reach_sibling_mixin(self) -> None:
        captured: dict = {}

        class _CaptureMixin:
            def __init__(self, *, marker, **kwargs):
                captured["marker"] = marker
                super().__init__(**kwargs)

        class _Composed(IndependentIntegration, _CaptureMixin):
            pass

        _Composed(solver="exp_euler", marker="hit")

        self.assertEqual(captured["marker"], "hit")


if __name__ == "__main__":
    unittest.main()
