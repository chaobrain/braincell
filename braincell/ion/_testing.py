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

import brainunit as u
import jax.numpy as jnp

__all__ = [
    "V",
    "make_shell_ion",
    "FixedIonContractTests",
    "KineticPumpContractTests",
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
