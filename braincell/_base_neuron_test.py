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

"""Unit tests for :mod:`braincell._base_neuron`."""

import unittest

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braincell._base_neuron import HHTypedNeuron


class BaseNeuronExportTest(unittest.TestCase):
    def test_public_namespace_reexports_this_module(self) -> None:
        import braincell
        import braincell._base_neuron as neuron_mod

        self.assertIs(braincell.HHTypedNeuron, neuron_mod.HHTypedNeuron)

    def test_module_does_not_import_base_ion(self) -> None:
        """``_base_neuron`` must stay below ``_base_ion`` in the import order.

        ``_base_ion`` names ``HHTypedNeuron`` as ``root_type`` with a plain
        top-level import; an edge back the other way would reintroduce the
        cycle that motivated splitting this module out.
        """
        import ast
        import pathlib

        import braincell._base_neuron as neuron_mod

        source = pathlib.Path(neuron_mod.__file__).read_text()
        imported = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertNotIn("_base_ion", imported)
        self.assertNotIn("braincell._base_ion", imported)


class HHTypedNeuronGetSpikeTest(unittest.TestCase):
    """ARCH-04: get_spike lives on the shared base, not on each subclass."""

    def test_arrival_boundary_does_not_repeat_at_threshold(self):
        from braincell._single_compartment.point import PointNeuron

        sc = PointNeuron(size=1, V_th=0.0 * u.mV)
        old, new = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0])
        actual = sc.get_spike(jnp.asarray(old) * u.mV, jnp.asarray(new) * u.mV)
        np.testing.assert_array_equal(actual, (old < 0) & (new >= 0))

    def test_get_spike_is_method_on_base(self) -> None:
        self.assertTrue(hasattr(HHTypedNeuron, "get_spike"))
        self.assertTrue(callable(HHTypedNeuron.get_spike))

    def test_point_neuron_inherits_get_spike(self) -> None:
        from braincell._multi_compartment.lumped import LumpedRuntime
        from braincell._single_compartment.point import PointNeuron

        for cls in (PointNeuron, LumpedRuntime):
            self.assertIs(
                cls.get_spike,
                HHTypedNeuron.get_spike,
                msg=f"{cls.__name__} must not redefine get_spike after ARCH-04",
            )

        sc = PointNeuron(size=1, V_th=0.0 * u.mV)
        spk = sc.get_spike(jnp.array([-10.0]) * u.mV, jnp.array([10.0]) * u.mV)
        self.assertGreater(float(spk[0]), 0.0)

    def test_default_crossing_surrogate_gradient_and_finite_support(self) -> None:
        from braincell._single_compartment.point import PointNeuron

        sc = PointNeuron(size=1, V_th=0.0 * u.mV)

        def crossing(last_v_mantissa, next_v_mantissa):
            last_v = jnp.asarray([last_v_mantissa]) * u.mV
            next_v = jnp.asarray([next_v_mantissa]) * u.mV
            return sc.get_spike(last_v, next_v)[0]

        self.assertEqual(float(crossing(-10.0, 10.0)), 1.0)
        grad_last, grad_next = jax.grad(crossing, argnums=(0, 1))(-10.0, 10.0)
        np.testing.assert_allclose(grad_last, -0.0075, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(grad_next, 0.0075, rtol=1e-6, atol=1e-8)

        far_grad_last, far_grad_next = jax.grad(crossing, argnums=(0, 1))(-40.0, 40.0)
        np.testing.assert_allclose(far_grad_last, 0.0, atol=0.0)
        np.testing.assert_allclose(far_grad_next, 0.0, atol=0.0)


class ValueInputStepTest(unittest.TestCase):
    """CELL-36: the step that consumes value inputs, shared by the point neuron and Cell."""

    def _neuron(self):
        from braincell._single_compartment.point import PointNeuron

        return PointNeuron(size=1)

    def test_step_keeps_values_for_every_read_and_removes_them_at_its_end(self) -> None:
        neuron = self._neuron()
        neuron.add_current_input("i", 1.0 * u.nA)
        neuron.add_delta_input("d", 2.0 * u.mV)
        neuron.add_delta_input("f", lambda: 3.0 * u.mV)
        with neuron._value_input_step():
            self.assertTrue(neuron._in_value_input_step())
            with neuron._value_input_step():  # nested: the outer block is the step
                pass
            self.assertIn("i", neuron.current_inputs)
            for _ in range(2):
                self.assertEqual(neuron.sum_current_inputs(0.0 * u.nA).to_decimal(u.nA), 1.0)
                self.assertEqual(neuron.sum_delta_inputs(0.0 * u.mV).to_decimal(u.mV), 5.0)
            # Registered anew under a taken key during the step: it is the next step's.
            neuron.current_inputs["i"] = 4.0 * u.nA
        self.assertFalse(neuron._in_value_input_step())
        self.assertEqual(dict(neuron.current_inputs), {"i": 4.0 * u.nA})
        self.assertEqual(tuple(neuron.delta_inputs), ("f",))

    def test_take_restore_and_first_step_values(self) -> None:
        neuron = self._neuron()
        self.assertEqual(neuron._take_value_inputs(), ({}, {}))
        self.assertIsNone(neuron.current_inputs)
        neuron.add_current_input("i", 1.0 * u.nA)
        neuron.add_current_input("g", lambda: 1.0 * u.nA)
        taken = neuron._take_value_inputs()
        self.assertEqual(taken, ({"i": 1.0 * u.nA}, {}))
        self.assertEqual(tuple(neuron.current_inputs), ("g",))
        neuron._restore_value_inputs(taken)
        self.assertEqual(tuple(neuron.current_inputs), ("g", "i"))

        def summed(first):
            with neuron._value_input_step(({"j": jnp.ones(2) * u.nA}, {"k": 2.0 * u.mV}), first=first):
                return neuron.sum_current_inputs(0.0 * u.nA, pop=True), neuron.sum_delta_inputs(0.0 * u.mV)

        del neuron.current_inputs["g"], neuron.current_inputs["i"]
        for first, scale in ((jnp.asarray(True), 1.0), (jnp.asarray(False), 0.0)):
            current, delta = jax.jit(summed)(first)
            np.testing.assert_array_equal(current.to_decimal(u.nA), [scale, scale])
            self.assertEqual(float(delta.to_decimal(u.mV)), 2.0 * scale)
            self.assertEqual(neuron.current_inputs, {})
            self.assertEqual(neuron.delta_inputs, {})


if __name__ == "__main__":
    unittest.main()
