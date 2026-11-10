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

import functools
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import braincell as bc
from braincell.ion._base import KineticIon
from braincell.ion._testing import make_shell_ion
from braincell.ion.chloride import Chloride, ChlorideFixed, ChlorideInitNernst, Cldif_JE2011, _FRAT, _VRAT

_E_35C = -87.22198260005996  # mV, Ci 5 / Co 133.5 mM, z = -1, R and F as in brainunit
_E_36C = -87.50503300603127
_DT = 0.025  # ms
_V = jnp.array([-70.0]) * u.mV


def _nernst_mv(temp):
    ion = ChlorideInitNernst(size=1, temp=temp)
    ion.init_state(_V)
    return float(np.asarray(ion.E.to_decimal(u.mV)).reshape(-1)[0])


def _x64_report() -> dict:
    """Float64 measurements, run in a fresh process by ``_float64_report``.

    After a float32 ``Cell.run``, switching to x64 in the same process can feed
    a float32 device copy of brainunit's numpy constants to a float64 program
    (``RuntimeProgramInputMismatch``).
    """
    brainstate.environ.set(precision=64, dt=_DT * u.ms)
    steps = 40000
    cell = _cable_cell("Cldif_JE2011", species_initializers={"Ci": 8.0 * u.mM})
    result = cell.run(dt=_DT * u.ms, duration=(steps + 1) * _DT * u.ms)
    tau, h = 3000.0, 3.0
    report = {
        "E35": _nernst_mv(u.celsius2kelvin(35.0)),
        "E36": _nernst_mv(u.celsius2kelvin(36.0)),
        "cell_ci": np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0].tolist(),
        "explicit_4000": float(_standalone_ci(Cldif_JE2011, 4000)),
        "implicit_4000": float(_standalone_ci(_ImplicitPump, 4000)),
    }
    for cls in (Cldif_JE2011, _StopGradientPump):
        grad = float(jax.grad(lambda t: _standalone_ci(cls, 400, t))(tau))
        fd = float((_standalone_ci(cls, 400, tau + h) - _standalone_ci(cls, 400, tau - h)) / (2 * h))
        report[cls.__name__] = (grad, fd)
    return report


@functools.cache
def _float64_report() -> dict:
    code = "import json; from braincell.ion.chloride_test import _x64_report; print(json.dumps(_x64_report()))"
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


class ChlorideReversalTest(unittest.TestCase):
    """Valence -1 Nernst potentials of the fixed-concentration models."""

    def test_init_nernst_at_35_and_36_celsius(self) -> None:
        report = _float64_report()
        np.testing.assert_allclose([report["E35"], report["E36"]], [_E_35C, _E_36C], rtol=1e-12)
        np.testing.assert_allclose(_nernst_mv(u.celsius2kelvin(35.0)), _E_35C, rtol=1e-6)
        np.testing.assert_allclose(_nernst_mv(u.celsius2kelvin(36.0)), _E_36C, rtol=1e-6)

    def test_fixed_default_and_valence(self) -> None:
        np.testing.assert_allclose(np.asarray(ChlorideFixed(size=1).E.to_decimal(u.mV)), -87.5)
        for ion in (ChlorideFixed(size=1), ChlorideInitNernst(size=1), Cldif_JE2011(size=1)):
            self.assertEqual(int(np.asarray(ion.valence).reshape(-1)[0]), -1)
        self.assertEqual(Chloride.ion_symbol.lower(), "cl")

    def test_positive_valence_counter_example_fails(self) -> None:
        with mock.patch.object(Chloride, "default_valence", 1):
            e = _nernst_mv(u.celsius2kelvin(35.0))
        np.testing.assert_allclose(e, -_E_35C, rtol=1e-6)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(e, _E_35C, rtol=1e-6)


def _numpy_relaxation(steps, *, diam=1.0, ci0=8.0, rest=5.0, D=2.0, tau=3000.0):
    """Backward-Euler reference of the four shells with no current; returns ``Ci`` per step."""
    vol = diam**2 * np.asarray(_VRAT)
    k = D * np.asarray(_FRAT[1:])
    K = np.diag(k, 1) + np.diag(k, -1) - np.diag(np.r_[k, 0.0] + np.r_[0.0, k])
    A = np.diag(vol) - _DT * K
    c = np.array([ci0, rest, rest, rest])
    out = [c[0]]
    for _ in range(steps):
        src = np.zeros(4)
        src[0] = vol[0] * (rest - c[0]) / tau
        c = np.linalg.solve(A, vol * c + _DT * src)
        out.append(c[0])
    return np.asarray(out)


def _cable_cell(ion, **ion_kwargs):
    branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[0.5, 0.5] * u.um)
    cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=-70.0 * u.mV)
    cell.paint(bc.mech.Ion(ion, name="cl", **ion_kwargs))
    cell.record("ci", bc.observe.ion(name="cl").state("Ci"))
    cell.record("E", bc.observe.ion(name="cl").state("E"))
    return cell


def _standalone_ci(cls, steps, tau_ms=3000.0):
    ion = cls(size=1, tau=tau_ms * u.ms, species_initializers={"Ci": 8.0 * u.mM})
    ion.diam_arc_mean = jnp.array([1.0]) * u.um
    ion.init_state(_V)
    brainstate.transform.for_loop(lambda i: ion.make_integration(_V), jnp.arange(steps))
    return ion.Ci.value.to_decimal(u.mM)[0]


class _ImplicitPump(Cldif_JE2011):
    """Counter-example: the pump reads the solver's current ``Ci`` (implicit)."""

    def _step_solver(self, V, recursive_child=True):
        KineticIon._step_solver(self, V, recursive_child)


class _StopGradientPump(Cldif_JE2011):
    """Counter-example: the step-start ``Ci`` is cut from the gradient."""

    def _step_solver(self, V, recursive_child=True):
        self._ci_step_start = jax.lax.stop_gradient(self.Ci.value)
        try:
            KineticIon._step_solver(self, V, recursive_child)
        finally:
            del self._ci_step_start


class CldifJE2011PumpTest(unittest.TestCase):
    """The pump is explicit, as NMODL's ``<<`` flux, and differentiable."""

    def test_shell_factors_match_cldif_mod(self) -> None:
        np.testing.assert_allclose(_VRAT, [0.239982772, 0.349065850, 0.174532925, 0.021816616], atol=1e-9)
        np.testing.assert_allclose(_FRAT[1:], [15.70796327, 9.424777961, 3.141592654], atol=1e-8)

    def test_cell_relaxation_matches_explicit_reference(self) -> None:
        steps = 40000
        reference = _numpy_relaxation(steps)
        ci = np.asarray(_float64_report()["cell_ci"])
        self.assertEqual(ci.shape, reference.shape)
        self.assertLessEqual(np.max(np.abs(ci - reference)) / np.max(np.abs(reference)), 1e-10)
        np.testing.assert_allclose(ci[[1, 40, 400, 40000]], [6.3324028, 5.9165652, 5.9157254, 5.8278925], atol=5e-8)

    def test_implicit_pump_counter_example_departs_from_reference(self) -> None:
        reference = _numpy_relaxation(4000)[-1]
        report = _float64_report()
        self.assertLessEqual(abs(report["explicit_4000"] - reference) / reference, 1e-10)
        self.assertGreater(abs(report["implicit_4000"] - reference) / reference, 1e-9)

    def test_gradient_with_respect_to_tau_matches_finite_difference(self) -> None:
        report = _float64_report()
        for name, matches in (("Cldif_JE2011", True), ("_StopGradientPump", False)):
            with self.subTest(cls=name):
                grad, fd = report[name]
                self.assertEqual(abs(grad - fd) <= 1e-5 * abs(fd), matches)

    def test_inward_current_lowers_ci_at_the_hand_computed_rate(self) -> None:
        for diameter, rate in ((20.0, -6.7839e-5), (1.0, -1.35678e-3)):
            with self.subTest(diameter=diameter):
                ion = make_shell_ion(Cldif_JE2011, diameter=diameter * u.um)
                ion.init_state(_V)
                ion.compute_derivative(_V)
                baseline = ion.Ci.derivative
                ion.register_external_current("inward", lambda V, info: jnp.ones(1) * u.uA / u.cm**2)
                ion.compute_derivative(_V)
                shift = (ion.Ci.derivative - baseline).to_decimal(u.mM / u.ms)
                np.testing.assert_allclose(np.asarray(shift), rate, rtol=1e-4)

    def test_compute_derivative_outside_a_step_reads_current_ci(self) -> None:
        ion = Cldif_JE2011(size=1, species_initializers={"Ci": 8.0 * u.mM})
        ion.diam_arc_mean = jnp.array([1.0]) * u.um
        ion.init_state(_V)
        ion.compute_derivative(_V)
        diffusion = 2.0 * _FRAT[1] * (5.0 - 8.0) / _VRAT[0]
        np.testing.assert_allclose(
            np.asarray(ion.Ci.derivative.to_decimal(u.mM / u.ms)), diffusion + (5.0 - 8.0) / 3000.0, rtol=1e-5
        )


class CldifJE2011CellTest(unittest.TestCase):
    """The pool runs on cable and lumped Cells and its errors name the class."""

    def test_cable_cell_without_current_stays_at_rest(self) -> None:
        result = _cable_cell("Cldif_JE2011").run(dt=_DT * u.ms, duration=100.0 * u.ms)
        ci = np.asarray(result.samples["ci"].values.to_decimal(u.mM))
        self.assertEqual(ci.shape, (4000, 1))
        np.testing.assert_allclose(ci, 5.0, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(result.samples["E"].values.to_decimal(u.mV)), _E_35C, rtol=1e-6)

    def test_lumped_cell_without_current_stays_at_rest(self) -> None:
        # A lumped Cell records only voltage and clamp currents, so read the ion after the run.
        cell = bc.Cell.point(1, length=20.0 * u.um, radius=0.5 * u.um, V_init=-70.0 * u.mV, solver="exp_euler")
        cell.paint(bc.mech.Ion("Cldif_JE2011", name="cl"))
        cell.run(dt=_DT * u.ms, duration=100.0 * u.ms)
        ion = cell._lumped_runtime.ion_channels["cl"]
        self.assertIsInstance(ion, Cldif_JE2011)
        shells = [np.asarray(getattr(ion, name).value.to_decimal(u.mM)) for name in ("Ci", "cl1", "cl2", "cl3")]
        np.testing.assert_allclose(shells, 5.0, rtol=1e-12)

    def test_fixed_and_nernst_models_run_on_a_cell(self) -> None:
        for cls in ("ChlorideFixed", "ChlorideInitNernst"):
            with self.subTest(ion=cls):
                result = _cable_cell(cls).run(dt=_DT * u.ms, duration=0.1 * u.ms)
                np.testing.assert_allclose(np.asarray(result.samples["ci"].values.to_decimal(u.mM)), 5.0)
                e = np.asarray(result.samples["E"].values.to_decimal(u.mV))
                np.testing.assert_allclose(e, -87.5 if cls == "ChlorideFixed" else _E_36C, rtol=1e-6)

    def test_missing_diameter_and_bad_substeps_raise(self) -> None:
        with self.assertRaisesRegex(AttributeError, "Cldif_JE2011"):
            Cldif_JE2011(size=1).init_state(_V)
        with self.assertRaisesRegex(ValueError, "substeps"):
            Cldif_JE2011(size=1, substeps=0)
        with self.assertRaisesRegex(ValueError, "Ca"):
            Cldif_JE2011(size=1, species_initializers={"Ca": 1.0 * u.mM})


if __name__ == "__main__":
    unittest.main()
