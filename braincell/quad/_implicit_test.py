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

"""Tests for :mod:`braincell.quad._implicit`.

The module exposes exactly one step function, ``implicit_euler_step``,
which runs on any :class:`DiffEqModule` target. The six cell-only
splitting variants that used to live here (``splitting_step``,
``cn_rk4_step``, ``cn_exp_euler_step``, ``implicit_rk4_step``,
``implicit_exp_euler_step``, ``exp_exp_euler_step``) were removed: they
depended on a ``Cell.conductance_matrix()`` / ``Cell.Gl`` interface that
no longer exists and on a stale ``brainstate.transform.vmap2(...,
in_states=...)`` signature, so none of them could execute against a real
:class:`braincell.Cell`.
"""

import unittest

import brainstate
import brainunit as u

from braincell.quad import get_registry, implicit_euler_step
from braincell.quad._testing import LinearDecay


class ImplicitEulerLinearTest(unittest.TestCase):
    """``implicit_euler_step`` solves a Crank-Nicolson residual by Newton."""

    def test_one_step_lies_in_cn_bracket(self):
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            implicit_euler_step(m)
        v = float(m.x.value.to_decimal(u.mV)[0])
        # The 1-step Crank-Nicolson value lies between the implicit-Euler
        # value 1/(1 + dt/tau) ≈ 0.99010 and the exact decay exp(-dt/tau)
        # ≈ 0.99005.
        self.assertGreater(v, 0.989)
        self.assertLess(v, 0.991)

    def test_explicit_time_arguments_override_the_environ(self):
        # ``t``/``dt`` are keyword-only overrides; passing dt explicitly must
        # win over the surrounding context.
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=5.0 * u.ms):
            implicit_euler_step(m, dt=0.1 * u.ms)
        v = float(m.x.value.to_decimal(u.mV)[0])
        self.assertGreater(v, 0.989)
        self.assertLess(v, 0.991)

    def test_selectable_by_name_through_the_host_convention(self):
        # The step now matches ``self.solver(self)``, so a host can select it
        # by name. This is what ``requires_time_args`` used to document as
        # impossible.
        step = get_registry()["implicit_euler"]
        m = LinearDecay()
        with brainstate.environ.context(t=0.0 * u.ms, dt=0.1 * u.ms):
            step(m)
        self.assertLess(float(m.x.value.to_decimal(u.mV)[0]), 1.0)


class ImplicitMethodRegistrationTest(unittest.TestCase):
    """Registry metadata for the one implicit step this module owns."""

    #: Names deleted in the bit-rot cleanup. Re-registering any of them
    #: without a working implementation would resurrect the rot.
    REMOVED = (
        "splitting",
        "implicit_rk4",
        "implicit_exp_euler",
        "cn_rk4",
        "cn_exp_euler",
        "exp_exp_euler",
    )

    def test_implicit_euler_registered(self):
        registry = get_registry()
        self.assertIn("implicit_euler", registry)
        self.assertIs(registry["implicit_euler"], implicit_euler_step)

    def test_implicit_euler_metadata(self):
        entry = get_registry().entry("implicit_euler")
        self.assertEqual(entry.category, "implicit")
        self.assertEqual(entry.order, 1)
        self.assertEqual(entry.module, "braincell.quad")

    def test_removed_cell_only_steps_are_gone(self):
        registry = get_registry()
        import braincell.quad as quad

        for name in self.REMOVED:
            with self.subTest(name=name):
                self.assertNotIn(name, registry)
                self.assertFalse(hasattr(quad, f"{name}_step"))


if __name__ == "__main__":
    unittest.main()
