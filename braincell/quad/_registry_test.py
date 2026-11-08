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

"""Tests for name resolution in :mod:`braincell.quad._registry`.

braincell has no registry of its own; these tests pin how braincell's names
map onto the :mod:`braintools.quad` registry.
"""

import copy
import importlib
import unittest

import braintools.quad as btq
import brainunit as u
from braintools.quad import _registry as bt_registry

import braincell
import braincell.quad as quad
from braincell.quad import get_integrator, integrator_info, list_integrators
from braincell.quad._registry import register_module_step
from braincell.quad._testing import HH

#: Every name braincell ships, mapped to ``(kind, canonical name, category, order)``.
EXPECTED = {
    "euler": ("ode", "euler", "explicit", 1),
    "midpoint": ("ode", "midpoint", "explicit", 2),
    "heun2": ("ode", "rk2", "explicit", 2),
    "ralston2": ("ode", "ralston2", "explicit", 2),
    "rk3": ("ode", "rk3", "explicit", 3),
    "heun3": ("ode", "heun", "explicit", 3),
    "ssprk3": ("ode", "ssprk33", "explicit", 3),
    "ralston3": ("ode", "ralston3", "explicit", 3),
    "rk4": ("ode", "rk4", "explicit", 4),
    "ralston4": ("ode", "ralston4", "explicit", 4),
    "exp_euler": ("module", "exp_euler", "exponential", 1),
    "ind_exp_euler": ("module", "ind_exp_euler", "exponential", 1),
    "backward_euler": ("module", "backward_euler", "implicit", 1),
    "crank_nicolson": ("module", "crank_nicolson", "implicit", 2),
    "staggered": ("module", "staggered", "staggered", None),
    "stagger": ("module", "staggered", "staggered", None),
    "dhs_voltage": ("module", "dhs_voltage", "voltage", None),
}


class _SavedRegistry:
    """Restore the braintools registry tables after a test."""

    def __enter__(self):
        self.saved = {
            name: copy.deepcopy({k: dict(v) for k, v in getattr(bt_registry, name).items()})
            for name in ("_REGISTRY", "_META", "_ALIASES")
        }
        return self

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            getattr(bt_registry, name).clear()
            getattr(bt_registry, name).update(value)
        return False


class ResolutionTest(unittest.TestCase):
    def test_every_shipped_name_resolves_with_metadata(self):
        for name, (kind, canonical, category, order) in EXPECTED.items():
            with self.subTest(name=name):
                info = integrator_info(name)
                self.assertEqual((info.kind, info.name, info.category, info.order), (kind, canonical, category, order))
                self.assertTrue(callable(get_integrator(name)))

    def test_module_steps_are_braincells_functions(self):
        self.assertIs(get_integrator("staggered"), quad.staggered_step)
        self.assertIs(get_integrator("stagger"), quad.staggered_step)
        self.assertIs(get_integrator("exp_euler"), quad.exp_euler_step)
        self.assertIs(get_integrator("backward_euler"), btq.get_method("backward_euler", "module"))

    def test_ode_methods_are_wrapped(self):
        self.assertIs(get_integrator("rk4"), quad.rk4_step)
        self.assertIs(integrator_info("rk4").fn, btq.ode_rk4_step)
        self.assertIs(get_integrator("ode_rk4_step"), quad.rk4_step)

    def test_callable_passes_through(self):
        def step(target, *args):
            return None

        self.assertIs(get_integrator(step), step)

    def test_list_integrators(self):
        names = list_integrators()
        self.assertEqual(names, sorted(names))
        for name in ("staggered", "dhs_voltage", "rk4", "ralston4", "dopri5", "heun2", "crank_nicolson"):
            self.assertIn(name, names)
        for name in ("rk2", "explicit", "expeuler", "implicit_euler", "stagger", "heun3"):
            self.assertNotIn(name, names)
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(callable(get_integrator(name)))


class RemovedNamesTest(unittest.TestCase):
    def test_rk2_raises_with_replacements(self):
        for call in (lambda: get_integrator("rk2"), lambda: integrator_info("rk2")):
            with self.assertRaises(ValueError) as ctx:
                call()
            self.assertIn("ralston2", str(ctx.exception))
            self.assertIn("heun2", str(ctx.exception))

    def test_rk2_fails_at_model_construction(self):
        with self.assertRaisesRegex(ValueError, "ralston2"):
            HH(1, solver="rk2")

    def test_explicit_and_expeuler(self):
        with self.assertRaisesRegex(ValueError, "'euler'"):
            get_integrator("explicit")
        with self.assertRaisesRegex(ValueError, "ind_exp_euler"):
            get_integrator("expeuler")
        with self.assertRaisesRegex(ValueError, "ind_exp_euler"):
            get_integrator("ode_expeuler_step")

    def test_removed_attributes(self):
        for name, hint in (
            ("rk2_step", "ralston2_step"),
            ("get_registry", "integrator_info"),
            ("all_integrators", "list_integrators"),
            ("register_integrator", "register_method"),
            ("IntegratorRegistry", "integrator_info"),
            ("IntegratorEntry", "MethodInfo"),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(AttributeError, hint):
                    getattr(quad, name)
        with self.assertRaisesRegex(AttributeError, "no attribute"):
            quad.not_a_name

    def test_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Did you mean 'euler'"):
            get_integrator("eler")
        with self.assertRaisesRegex(ValueError, "Unknown integrator"):
            integrator_info("qqqqqq")

    def test_bad_types(self):
        with self.assertRaises(TypeError):
            get_integrator(3)
        with self.assertRaises(TypeError):
            integrator_info(None)


class _Other:
    """A foreign function with a braincell step name."""

    @staticmethod
    def staggered_step(target, *args):
        return None


class IdempotentRegistrationTest(unittest.TestCase):
    def test_reload_does_not_raise(self):
        import braincell.quad._backward_euler as module

        with _SavedRegistry():
            reloaded = importlib.reload(module)
            self.assertIs(get_integrator("backward_euler"), reloaded.backward_euler_step)
            self.assertIsNot(reloaded.backward_euler_step, quad.backward_euler_step)
        self.assertIs(get_integrator("backward_euler"), quad.backward_euler_step)

    def test_reregistering_the_same_function_is_a_no_op(self):
        with _SavedRegistry():
            register_module_step("staggered", aliases=("stagger",), category="staggered")(quad.staggered_step)
            self.assertIs(get_integrator("stagger"), quad.staggered_step)

    def test_foreign_function_with_a_braincell_name_raises(self):
        with _SavedRegistry():
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_module_step("staggered")(_Other.staggered_step)

    def test_braincell_name_taken_by_another_package_raises(self):
        def staggered_step(target, *args):
            return None

        with _SavedRegistry():
            btq.register_method("dhs_voltage", staggered_step, kind="module", overwrite=True)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_module_step("dhs_voltage", category="voltage")(quad.dhs_voltage_step)


class CellSolverNameTest(unittest.TestCase):
    def test_alias_is_recorded_as_canonical_name(self):
        soma = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[10.0, 10.0] * u.um, type="soma")
        tree = braincell.Morphology.from_root(soma, name="soma")
        cell = braincell.Cell(tree, solver="stagger")
        self.assertEqual(cell.solver_name, "staggered")
        self.assertIs(cell.solver, quad.staggered_step)
        cell = braincell.Cell(tree, solver="heun2")
        self.assertEqual(cell.solver_name, "heun2")


class SingleCompartmentDefaultTest(unittest.TestCase):
    def test_default_solver_is_ralston2(self):
        class Leak(braincell.SingleCompartment):
            def __init__(self):
                super().__init__(1)

        self.assertIs(Leak().solver, get_integrator("ralston2"))


if __name__ == "__main__":
    unittest.main()
