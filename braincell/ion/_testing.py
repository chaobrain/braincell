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

"""Shared fixtures for the ``braincell.ion`` test modules.

Private by name so pytest does not collect it as a test module, following
the same convention as ``braincell/mech/_testing.py`` and
``braincell/filter/_testing.py``.

Two kinds of helper live here:

- constructors (:func:`V`, :func:`make_shell_ion`) that every ion test
  file otherwise retypes, and
- contract mixins (:class:`FixedIonContractTests`,
  :class:`KineticPumpContractTests`) carrying test methods that are
  identical across sibling ion species and were previously copy-pasted
  once per species.

A contract mixin is parameterized by class attributes on the concrete
``TestCase`` that mixes it in, so a new ion species inherits the shared
suite by declaring those attributes rather than by copying the methods.
"""

import json
import os
import subprocess
import sys

import brainunit as u
import jax.numpy as jnp
import numpy as np

__all__ = [
    "V",
    "make_shell_ion",
    "run_float64",
    "FixedIonContractTests",
    "KineticPumpContractTests",
    "DynamicShellIonContractTests",
]


def V(values, unit=u.mV):
    """Return a membrane-potential array for a test.

    Parameters
    ----------
    values : array-like
        Membrane potentials, in ``unit``.
    unit : brainunit.Unit, optional
        Unit the values are expressed in. Defaults to ``u.mV``.

    Returns
    -------
    brainunit.Quantity
        The values as a united array.
    """
    return jnp.asarray(values) * unit


def make_shell_ion(cls, size=1, diameter=20.0 * u.um, **kwargs):
    """Build a radial-shell ion with its compartment geometry attached.

    ``diam_arc_mean`` and ``diam_mid`` are written onto a ``cdp*`` ion by
    the compartment layer, not by its constructor, so a bare instance
    raises from ``_require_diam_arc_mean`` the moment any species is
    materialized. Every kinetic-calcium test therefore has to stand in
    for that layer.

    Parameters
    ----------
    cls : type
        The ion class to instantiate.
    size : brainstate.typing.Size, optional
        Varshape of the ion. Defaults to ``1``.
    diameter : brainunit.Quantity, optional
        Value written to both ``diam_arc_mean`` and ``diam_mid``.
        Defaults to ``20 um``.
    **kwargs
        Forwarded unchanged to ``cls``.

    Returns
    -------
    braincell.ion._base.KineticIon
        The constructed ion, ready for ``init_state``.
    """
    ion = cls(size=size, **kwargs)
    geometry = jnp.broadcast_to(u.get_mantissa(diameter), ion.varshape) * u.get_unit(diameter)
    ion.diam_mid = geometry
    ion.diam_arc_mean = geometry
    return ion


class FixedIonContractTests:
    """Behaviour every ``FixedIon`` species shares, parameterized by class.

    Mix into a :class:`unittest.TestCase` and declare:

    ``ION_CLASS``
        The concrete ``*Fixed`` class under test.
    ``FAMILY_CLASS``
        Its abstract family base (``Potassium``, ``Sodium``, ...).
    ``DEFAULT_E``, ``DEFAULT_CI``, ``DEFAULT_CO``, ``DEFAULT_VALENCE``
        The documented resting defaults, as quantities.

    The four ``*Fixed`` species differ only in those values, so the
    assertions below were previously four near-identical copies.
    """

    ION_CLASS = None
    FAMILY_CLASS = None
    DEFAULT_E = None
    DEFAULT_CI = None
    DEFAULT_CO = None
    DEFAULT_VALENCE = None

    def test_family_module_is_the_public_namespace(self) -> None:
        self.assertEqual(self.FAMILY_CLASS.__module__, "braincell.ion")
        self.assertEqual(self.ION_CLASS.__module__, "braincell.ion")

    def test_is_subclass_of_its_family(self) -> None:
        self.assertTrue(issubclass(self.ION_CLASS, self.FAMILY_CLASS))

    def test_documented_defaults_are_what_the_constructor_produces(self) -> None:
        ion = self.ION_CLASS(size=1)
        self.assertTrue(u.math.allclose(ion.E, self.DEFAULT_E, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(ion.Ci, self.DEFAULT_CI, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(ion.Co, self.DEFAULT_CO, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(ion.valence, jnp.full((1,), self.DEFAULT_VALENCE), atol=1e-9))

    def test_varshape_follows_size(self) -> None:
        self.assertEqual(self.ION_CLASS(size=1).varshape, (1,))
        self.assertEqual(self.ION_CLASS(size=5).varshape, (5,))
        self.assertEqual(self.ION_CLASS(size=(2, 3)).varshape, (2, 3))

    def test_explicit_none_reversal_potential_is_rejected(self) -> None:
        # ``E`` is the one field with no class-level default: an ion whose
        # reversal potential is unset would silently produce a zero current.
        with self.assertRaises(ValueError):
            self.ION_CLASS(size=1, E=None)

    def test_callable_parameters_broadcast_across_size(self) -> None:
        ion = self.ION_CLASS(
            size=3,
            E=lambda shape: jnp.array([-90.0, -95.0, -100.0]) * u.mV,
            Ci=lambda shape: jnp.array([0.04, 0.05, 0.06]) * u.mM,
            Co=lambda shape: jnp.array([2.5, 2.6, 2.7]) * u.mM,
        )
        self.assertEqual(ion.E.shape, (3,))
        self.assertEqual(ion.Ci.shape, (3,))
        self.assertEqual(ion.Co.shape, (3,))
        self.assertTrue(u.math.allclose(ion.E, jnp.array([-90.0, -95.0, -100.0]) * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(ion.Ci, jnp.array([0.04, 0.05, 0.06]) * u.mM, atol=1e-9 * u.mM))

    def test_pack_info_reports_the_stored_values(self) -> None:
        from braincell._base_channel import IonInfo

        ion = self.ION_CLASS(size=1, E=-85.0 * u.mV, Ci=0.02 * u.mM, Co=2.8 * u.mM, valence=1)
        info = ion.pack_info()
        self.assertIsInstance(info, IonInfo)
        self.assertTrue(u.math.allclose(info.E, -85.0 * u.mV, atol=1e-9 * u.mV))
        self.assertTrue(u.math.allclose(info.Ci, 0.02 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(info.Co, 2.8 * u.mM, atol=1e-9 * u.mM))
        self.assertTrue(u.math.allclose(info.valence, jnp.ones((1,)), atol=1e-9))

    def test_empty_container_reports_no_channels_and_no_current(self) -> None:
        ion = self.ION_CLASS(size=1)
        self.assertEqual(ion.channels, {})
        self.assertEqual(ion.external_currents, {})
        self.assertIsNone(ion.current(V([-60.0])))

    def test_external_current_keys_must_be_unique(self) -> None:
        ion = self.ION_CLASS(size=1)

        def external(V_local, info):
            return jnp.array([1.0]) * u.uA / u.cm**2

        ion.register_external_current("ext", external)
        self.assertIn("ext", ion.external_currents)
        with self.assertRaises(ValueError):
            ion.register_external_current("ext", external)


def run_float64(code: str):
    """Run ``code`` in a fresh float64 process; return the JSON its last line prints.

    After a float32 ``Cell.run``, switching to x64 in the same process can
    feed a float32 device copy of brainunit's constants to a float64
    program, so float64 checks run in a child process.
    """
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def shell_ion_rates(name: str) -> list:
    """``dCi/dt`` (mM/ms) of the registered ion ``name`` at its defaults.

    For +1 uA/cm2 given in uA/cm2, A/m2 and mA/cm2, then for no current.
    """
    from braincell.mech import get_registry

    ion = get_registry().get("ion", name)(size=1)
    ion.init_state(V([-65.0]))
    current = jnp.ones(1) * u.uA / u.cm**2
    rates = []
    for value in (current, current.to(u.amp / u.meter**2), current.in_unit(u.mA / u.cm**2), None):
        rate = ion.derivative(ion.Ci.value, V([-65.0]), total_current=value)
        assert u.get_unit(rate).dim == (u.mM / u.ms).dim, u.get_unit(rate)
        rates.append(float(np.asarray(rate.to_decimal(u.mM / u.ms))[0]))
    return rates


class DynamicShellIonContractTests:
    """Shared tests of a ``dCi/dt = I/(zFd) + beta (C_rest - Ci)`` ion.

    The concrete ``TestCase`` sets ``ION_CLASS``, ``LEAK`` (the registry
    name of a leak channel of the same family) and ``INWARD_V_INIT``, a
    membrane potential below the family's reversal potential, so that the
    leak current is inward.
    """

    ION_CLASS = None
    LEAK = None
    INWARD_V_INIT = None

    def _cable_cell(self, V_init, **ion_kwargs):
        """A one-CV cable Cell, L = diam = 15 um, with this ion."""
        import braincell as bc

        branch = bc.Branch.from_lengths(lengths=[15.0] * u.um, radii=[7.5, 7.5] * u.um)
        cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1), V_init=V_init)
        cell.paint(bc.mech.Ion(self.ION_CLASS.__name__, name="ion", **ion_kwargs))
        return cell

    def _cell(self, *channels, V_init):
        import braincell as bc

        cell = self._cable_cell(V_init)
        for name, kwargs in channels:
            cell.paint(bc.mech.Channel(name, **kwargs))
        cell.record("ci", bc.observe.ion(name="ion").state("Ci"))
        cell.record("E", bc.observe.ion(name="ion").state("E"))
        result = cell.run(dt=0.025 * u.ms, duration=10.0 * u.ms)
        return (
            np.asarray(result.samples["ci"].values.to_decimal(u.mM))[:, 0],
            np.asarray(result.samples["E"].values.to_decimal(u.mV))[:, 0],
        )

    def test_rate_is_one_over_zFd_in_any_current_unit(self):
        rates = run_float64(
            "import json; from braincell.ion._testing import shell_ion_rates; "
            f"print(json.dumps(shell_ion_rates({self.ION_CLASS.__name__!r})))"
        )
        # 1 uA/cm2 / (F * 1 um), z = 1.
        np.testing.assert_allclose(rates[:3], [1.036427e-4] * 3, rtol=1e-6)
        self.assertEqual(rates[3], 0.0)

    def test_inward_current_raises_ci_and_lowers_e_in_a_cell(self):
        # IL at V_init holds V between V_init and E, so the leak current stays inward.
        g_max = 1.0 * u.mS / u.cm**2
        ci, E = self._cell(
            (self.LEAK, {"g_max": g_max}), ("IL", {"g_max": g_max, "E": self.INWARD_V_INIT}), V_init=self.INWARD_V_INIT
        )
        self.assertTrue(np.all(np.isfinite(ci)))
        # Monotonic; a float32 step of E can round to zero.
        self.assertTrue(np.all(np.diff(ci) >= 0.0) and ci[-1] > ci[0])
        self.assertTrue(np.all(np.diff(E) <= 0.0) and E[-1] < E[0])

    def test_ci_is_constant_without_a_channel_of_its_family(self):
        ci, _ = self._cell(("IL", {}), V_init=self.INWARD_V_INIT)
        np.testing.assert_array_equal(ci, np.full_like(ci, ci[0]))

    def test_lumped_cell_runs_with_the_default_initializer(self):
        import braincell as bc

        cell = bc.Cell.point(1, length=15.0 * u.um, radius=7.5 * u.um, V_init=-65.0 * u.mV)
        cell.paint(bc.mech.Ion(self.ION_CLASS.__name__, name="ion"))
        cell.paint(bc.mech.Channel(self.LEAK))
        cell.record("v", bc.observe.state("V"))
        result = cell.run(dt=0.025 * u.ms, duration=10.0 * u.ms)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.samples["v"].values.to_decimal(u.mV)))))

    def test_parameter_without_or_with_wrong_unit_fails_at_init_state(self):
        import warnings

        for d, error, message in ((1.0, TypeError, "no unit"), (1.0 * u.mV, ValueError, "incompatible with um")):
            with self.subTest(d=d):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)  # warned at declaration in 0.3
                    cell = self._cable_cell(-65.0 * u.mV, d=d)
                with self.assertRaisesRegex(error, f"parameter 'd' .*{message}"):
                    cell.init_state()


class KineticPumpContractTests:
    """Pump behaviour every ``cdp*`` calcium pool shares.

    Mix into a :class:`unittest.TestCase` and declare ``ION_CLASS``. The
    four pools that carry a surface pump previously held byte-identical
    copies of both methods below.
    """

    ION_CLASS = None

    def make_ion(self, **kwargs):
        """Return the ion under test with its geometry attached."""
        return make_shell_ion(self.ION_CLASS, **kwargs)

    def test_positive_inward_current_produces_positive_ci_source_flux(self) -> None:
        ion = self.make_ion()
        ion.init_state(V([-60.0]))
        flux = ion.sources[0].flux(
            ion,
            V([-60.0]),
            ion.species_values(),
            total_current=jnp.array([0.01]) * u.mA / (u.cm**2),
        )
        self.assertGreater(float(flux[0].to_decimal(u.mM * u.um**2 / u.ms)), 0.0)

    def test_conserve_keeps_pump_plus_pumpca_equal_total_scaled_pool(self) -> None:
        ion = self.make_ion()
        ion.init_state(V([-60.0]))
        values = ion.species_values()
        total = ion.TotalPump * ion.parea
        combined = ion.pump.value * ion.parea + values["pumpca"] * ion.parea
        self.assertTrue(
            u.math.allclose(
                combined.to_decimal(total.unit),
                total.to_decimal(total.unit),
                atol=1e-12,
            )
        )
