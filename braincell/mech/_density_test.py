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

import unittest
import warnings

import braintools
import brainunit as u
import numpy as np

import braincell
from braincell.mech import Channel, Density, Ion, Params


class DensityConstructionTest(unittest.TestCase):
    def test_channel_sets_category(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        self.assertEqual(spec.category, "channel")
        self.assertEqual(spec.class_name, "IL")

    def test_ion_sets_category(self) -> None:
        spec = Ion("SodiumFixed", E=50 * u.mV)
        self.assertEqual(spec.category, "ion")
        self.assertEqual(spec.class_name, "SodiumFixed")

    def test_channel_is_density_subclass(self) -> None:
        spec = Channel("IL")
        self.assertIsInstance(spec, Density)

    def test_ion_is_density_subclass(self) -> None:
        spec = Ion("SodiumFixed")
        self.assertIsInstance(spec, Density)

    def test_params_are_Params(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        self.assertIsInstance(spec.params, Params)
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm**2))

    def test_default_params_is_empty(self) -> None:
        spec = Channel("IL")
        self.assertEqual(len(spec.params), 0)

    def test_default_coverage_is_one(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        self.assertEqual(spec.coverage_area_fraction, 1.0)

    def test_channel_can_store_single_ion_selector(self) -> None:
        spec = Channel("Na_HH1952", ion_name="na_main")
        self.assertEqual(spec.ion_name, "na_main")
        self.assertIsNone(spec.ion_names)

    def test_channel_can_store_multi_ion_selectors(self) -> None:
        spec = Channel("Kca1p1_MA2020_GoC", ion_names={"k": "k_main", "ca": "ca_local"})
        self.assertIsNone(spec.ion_name)
        self.assertEqual(spec.ion_names, (("ca", "ca_local"), ("k", "k_main")))

    def test_integration_override_is_metadata_not_runtime_params(self) -> None:
        spec = Channel("IL", solver="rk4", substeps=3, g_max=0.1 * u.mS / u.cm**2)
        self.assertEqual(spec.solver, "rk4")
        self.assertEqual(spec.substeps, 3)
        self.assertNotIn("solver", spec.params)
        self.assertNotIn("substeps", spec.params)


class DensityClassArgumentTest(unittest.TestCase):
    def test_channel_accepts_type_argument(self) -> None:
        spec = Channel(braincell.channel.IL, g_max=0.1 * (u.mS / u.cm**2))
        self.assertEqual(spec.class_name, "IL")
        self.assertEqual(spec.category, "channel")

    def test_ion_accepts_type_argument(self) -> None:
        spec = Ion(braincell.ion.PotassiumFixed)
        self.assertEqual(spec.class_name, "PotassiumFixed")
        self.assertEqual(spec.category, "ion")

    def test_type_and_string_produce_equal_specs(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        b = Channel(braincell.channel.IL, g_max=0.1 * (u.mS / u.cm**2))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class DensityValidationTest(unittest.TestCase):
    def test_abstract_density_rejects_instantiation(self) -> None:
        with self.assertRaises(TypeError):
            Density("IL")

    def test_empty_class_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Channel("")

    def test_invalid_class_name_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel(42)  # type: ignore[arg-type]

    def test_non_string_name_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel("IL", name=42)  # type: ignore[arg-type]

    def test_out_of_range_coverage_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Channel("IL", coverage_area_fraction=2.0)
        with self.assertRaises(ValueError):
            Channel("IL", coverage_area_fraction=-0.1)

    def test_channel_cannot_define_both_ion_name_and_ion_names(self) -> None:
        with self.assertRaises(ValueError):
            Channel("Kca1p1_MA2020_GoC", ion_name="ca_main", ion_names={"ca": "ca_main"})

    def test_invalid_ion_name_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel("Na_HH1952", ion_name=42)  # type: ignore[arg-type]

    def test_invalid_ion_names_mapping_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Channel("Kca1p1_MA2020_GoC", ion_names="ca_main")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Channel("Kca1p1_MA2020_GoC", ion_names={"": "ca_main"})
        with self.assertRaises(TypeError):
            Channel("Kca1p1_MA2020_GoC", ion_names={"ca": 1})  # type: ignore[arg-type]

    def test_integration_override_must_be_an_atomic_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            Channel("IL", solver="rk4")
        with self.assertRaisesRegex(ValueError, "provided together"):
            Ion("SodiumFixed", substeps=2)

    def test_integration_override_validates_substeps(self) -> None:
        with self.assertRaises(TypeError):
            Channel("IL", solver="rk4", substeps=True)
        with self.assertRaises(ValueError):
            Channel("IL", solver="rk4", substeps=0)


class DensityIdentityTest(unittest.TestCase):
    def test_default_instance_name_is_class_name(self) -> None:
        spec = Channel("IL")
        self.assertEqual(spec.instance_name, "IL")

    def test_override_instance_name(self) -> None:
        spec = Channel("Na_HH1952", name="na_main")
        self.assertEqual(spec.instance_name, "na_main")


class DensityEqualityTest(unittest.TestCase):
    def test_keyword_order_insensitive_equality(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        b = Channel("IL", E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm**2))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_different_coverage_are_unequal(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        b = a.with_coverage(0.5)
        self.assertNotEqual(a, b)

    def test_different_names_are_unequal(self) -> None:
        a = Channel("Na_HH1952", name="na_main")
        b = Channel("Na_HH1952", name="na_alt")
        self.assertNotEqual(a, b)

    def test_channel_and_ion_are_unequal_even_with_same_fields(self) -> None:
        c = Channel("SodiumFixed")
        i = Ion("SodiumFixed")
        self.assertNotEqual(c, i)

    def test_channel_ion_selectors_participate_in_equality(self) -> None:
        a = Channel("Na_HH1952", ion_name="na_a")
        b = Channel("Na_HH1952", ion_name="na_b")
        self.assertNotEqual(a, b)

    def test_integration_override_participates_in_equality_and_hash(self) -> None:
        a = Channel("IL", solver="rk4", substeps=2)
        b = Channel("IL", solver="rk4", substeps=2)
        inherited = Channel("IL")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, inherited)

    def test_can_use_as_dict_key(self) -> None:
        a = Channel("IL", g_max=0.1 * (u.mS / u.cm**2), E=-70 * u.mV)
        b = Channel("IL", E=-70 * u.mV, g_max=0.1 * (u.mS / u.cm**2))
        bucket = {a: "one"}
        self.assertEqual(bucket[b], "one")


class DensityUpdatesTest(unittest.TestCase):
    def test_with_coverage_non_mutating(self) -> None:
        original = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        updated = original.with_coverage(0.5)
        self.assertEqual(original.coverage_area_fraction, 1.0)
        self.assertEqual(updated.coverage_area_fraction, 0.5)
        self.assertIsInstance(updated, Channel)

    def test_updates_preserve_ion_subclass(self) -> None:
        original = Ion("SodiumFixed", Ci=12.0 * u.mM)
        updated = original.with_coverage(0.25)
        self.assertIsInstance(updated, Ion)
        self.assertEqual(updated.params["Ci"], 12.0 * u.mM)
        self.assertEqual(updated.coverage_area_fraction, 0.25)

    def test_copy_preserves_subclass_fields(self) -> None:
        # ``_replace`` walks ``_FIELDS``, so a subclass slot survives a
        # copy without the subclass overriding ``_replace``.
        original = Channel("Kca1p1_MA2020_GoC", ion_names={"ca": "ca_local"})
        self.assertEqual(original.with_coverage(0.5).ion_names, (("ca", "ca_local"),))

    def test_with_coverage_rejects_out_of_range_fraction(self) -> None:
        # Regression: ``with_coverage`` routed around the ``[0, 1]`` check
        # that ``__init__`` advertises, so a copy could hold a value the
        # constructor would have rejected.
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        for bad in (7.0, -0.5):
            with self.subTest(fraction=bad):
                with self.assertRaisesRegex(ValueError, r"coverage_area_fraction must lie in \[0, 1\]"):
                    spec.with_coverage(bad)


class DensityParamsKeywordTest(unittest.TestCase):
    """``params={...}`` is rejected rather than silently mis-stored.

    ``Density.__init__`` captures ``**params``, so ``Channel("IL",
    params={"g_max": ...})`` used to create a parameter literally named
    ``"params"`` and leave ``g_max`` unset -- with no error.
    """

    def test_channel_rejects_params_mapping(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be passed as keyword arguments"):
            Channel("IL", params={"g_max": 0.1 * (u.mS / u.cm**2)})

    def test_ion_rejects_params_mapping(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be passed as keyword arguments"):
            Ion("SodiumFixed", params={"Ci": 12.0 * u.mM})

    def test_rejected_even_when_mixed_with_real_parameters(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be passed as keyword arguments"):
            Channel("IL", E=-70 * u.mV, params={"g_max": 0.1 * (u.mS / u.cm**2)})

    def test_keyword_form_is_the_supported_spelling(self) -> None:
        spec = Channel("IL", g_max=0.1 * (u.mS / u.cm**2))
        self.assertEqual(spec.params["g_max"], 0.1 * (u.mS / u.cm**2))
        self.assertNotIn("params", spec.params)


class DensityImmutabilityTest(unittest.TestCase):
    def test_cannot_set_attributes(self) -> None:
        spec = Channel("IL")
        with self.assertRaises(AttributeError):
            spec.class_name = "K_HH1952"  # type: ignore[misc]

    def test_cannot_delete_attributes(self) -> None:
        spec = Channel("IL")
        with self.assertRaises(AttributeError):
            del spec.class_name  # type: ignore[misc]


class _UnregisteredLeak(braincell.Channel):
    """An ohmic leak no registry knows, as SingleCompartment users write them."""

    root_type = braincell.HHTypedNeuron

    def __init__(self, size, g_max=0.1 * (u.mS / u.cm**2), E=-70.0 * u.mV, name=None):
        super().__init__(size, name=name)
        self.g_max = braintools.init.param(g_max, self.varshape)
        self.E = braintools.init.param(E, self.varshape)

    def init_state(self, V, batch_size=None):
        pass

    def reset_state(self, V, batch_size=None):
        pass

    def compute_derivative(self, V, *args, **kwargs):
        pass

    def current(self, V, *args, **kwargs):
        return self.g_max * (self.E - V)


class DensityUnregisteredClassTest(unittest.TestCase):
    """CELL-25: unregistered classes are kept and resolved by identity."""

    def test_unregistered_channel_resolves_to_the_class(self) -> None:
        spec = Channel(_UnregisteredLeak, g_max=0.3 * (u.mS / u.cm**2))
        self.assertEqual(spec.class_name, "_UnregisteredLeak")
        self.assertIs(braincell.mech.get_registry().get("channel", spec.class_name), _UnregisteredLeak)
        self.assertEqual(spec, Channel(_UnregisteredLeak, g_max=0.3 * (u.mS / u.cm**2)))

    def test_unregistered_classes_with_one_name_give_distinct_specs(self) -> None:
        def make():
            class Leak(_UnregisteredLeak):
                pass

            return Leak

        first, second = make(), make()
        a, b = Channel(first), Channel(second)
        self.assertNotEqual(a, b)
        registry = braincell.mech.get_registry()
        self.assertIs(registry.get("channel", a.class_name), first)
        self.assertIs(registry.get("channel", b.class_name), second)

    def test_unregistered_channel_runs_on_cable_and_lumped_cells(self) -> None:
        from braincell.filter import AllRegion

        params = dict(g_max=0.3 * (u.mS / u.cm**2), E=-54.3 * u.mV)
        for kind, make in (
            ("lumped", lambda: braincell.Cell.point(2, V_init=-65.0 * u.mV)),
            ("cable", lambda: braincell.Cell(braincell.Morphology.from_soma(), pop_size=2, V_init=-65.0 * u.mV)),
        ):
            with self.subTest(cell=kind):
                traces = []
                for channel in (_UnregisteredLeak, "IL"):
                    cell = make()
                    cell.paint(AllRegion(), Channel(channel, **params))
                    cell.record("v", braincell.observe.state("v"))
                    cell.init_state()
                    result = cell.run(dt=0.025 * u.ms, duration=2.0 * u.ms)
                    traces.append(result.samples["v"].values.to_decimal(u.mV))
                self.assertGreater(float(traces[0][-1, 0]), -61.0)
                np.testing.assert_array_equal(traces[0], traces[1])

    def test_instances_and_foreign_classes_are_rejected_at_construction(self) -> None:
        cases = (
            ("instance", lambda: Channel(braincell.channel.IL(1)), "instance of 'IL'"),
            ("ion class as channel", lambda: Channel(braincell.ion.SodiumFixed), "and not of Ion or Synapse"),
            ("synapse class as channel", lambda: Channel(braincell.synapse.ExpSyn), "and not of Ion or Synapse"),
            ("plain class as channel", lambda: Channel(int), "subclass of braincell.IonChannel"),
            ("channel class as ion", lambda: Ion(braincell.channel.IL), "subclass of braincell.Ion"),
        )
        for label, build, message in cases:
            with self.subTest(case=label):
                with self.assertRaisesRegex(TypeError, message):
                    build()


class DeclarationParameterWarningTest(unittest.TestCase):
    """MECH-26: invalid literal parameters warn where the declaration is written."""

    def _init_state_on_a_cell(self, declaration) -> None:
        from braincell.filter import AllRegion

        cell = braincell.Cell(braincell.Morphology.from_soma())
        cell.paint(AllRegion(), declaration)
        cell.init_state()

    def test_invalid_literals_warn_at_this_line_and_still_raise_at_init_state(self) -> None:
        mS_cm2 = u.mS / u.cm**2
        cases = (
            (lambda: Channel("IL", name="leak", g_max=0.1), TypeError, ("Channel 'leak' (IL)", "'g_max'", "mS / cm^2")),
            (
                lambda: Channel("IL", name="leak", g_max=0.1 * u.mV),
                ValueError,
                ("Channel 'leak' (IL)", "'g_max'", "mS / cm^2", "0.1 mV"),
            ),
            (
                lambda: Channel("IL", name="leak", gmax=0.1 * mS_cm2),
                TypeError,
                ("Channel 'leak' (IL)", "'gmax'", "Did you mean 'g_max'?"),
            ),
            (lambda: Ion("SodiumFixed", E=50.0), TypeError, ("Ion 'SodiumFixed'", "'E'", "mV", "(no unit)")),
            (lambda: Ion("SodiumFixed", E=50.0 * u.mM), ValueError, ("Ion 'SodiumFixed'", "'E'", "mV", "50.0 mM")),
            (lambda: Ion("SodiumFixed", Cii=10.0 * u.mM), ValueError, ("Ion 'SodiumFixed'", "'Cii'", "'Ci'")),
        )
        for build, error, fragments in cases:
            with self.subTest(fragments=fragments):
                with self.assertWarns(FutureWarning) as caught:
                    declaration = build()
                self.assertEqual(caught.filename, __file__)
                message = str(caught.warning)
                for fragment in fragments:
                    self.assertIn(fragment, message)
                self.assertTrue(
                    message.endswith(f"This will raise {error.__name__} at declaration from braincell 0.4.")
                )
                with self.assertRaises(error):
                    self._init_state_on_a_cell(declaration)

    def test_one_warning_per_invalid_parameter(self) -> None:
        with self.assertWarns(FutureWarning) as caught:
            Channel("IL", g_max=0.1, E=-70.0)
        self.assertEqual({w.filename for w in caught.warnings}, {__file__})
        self.assertEqual([str(w.message).split()[3] for w in caught.warnings], ["'g_max'", "'E'"])

    def test_values_checked_only_at_init_state_do_not_warn(self) -> None:
        mS_cm2 = u.mS / u.cm**2
        builds = (
            lambda: Channel("IL", g_max=0.1 * mS_cm2, E=-70.0 * u.mV),
            lambda: Channel("IL", g_max=np.full(3, 0.1) * mS_cm2),
            lambda: Channel("IL", g_max=lambda ctx: 0.1),
            lambda: Channel("IL", g_max=braintools.init.Constant(0.1 * mS_cm2)),
            lambda: Channel("IL", g_max=None),
            lambda: Channel("NotARealChannel", g_max=0.1, foo=1.0),
            lambda: Ion("NotARealIon", E=50.0),
            lambda: Ion("SodiumFixed", Ci=braintools.init.Constant(10.0 * u.mM), valence=1),
            lambda: Channel(_UnregisteredLeak, g_max=0.3 * mS_cm2),
        )
        for build in builds:
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                build()

    def test_copies_are_not_checked_again(self) -> None:
        with self.assertWarns(FutureWarning):
            spec = Channel("IL", g_max=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            self.assertEqual(spec.with_coverage(0.5).params, spec.params)

    def test_unregistered_class_is_checked_against_its_own_signature(self) -> None:
        with self.assertWarns(FutureWarning) as caught:
            Channel(_UnregisteredLeak, g_max=0.3 * u.mV)
        self.assertIn("Channel '_UnregisteredLeak' parameter 'g_max' has units incompatible", str(caught.warning))

    def test_traced_values_are_checked_for_units_only(self) -> None:
        # ``Params`` cannot hash a traced scalar, so declarations inside
        # ``jit`` carry traced arrays; their entries are unknown until run.
        import jax

        mS_cm2 = u.mS / u.cm**2

        def declare(g, unit):
            Channel("IL", g_max=u.Quantity(g, unit))
            return g

        nan = np.full(3, np.nan, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            jax.jit(declare, static_argnums=1)(nan, mS_cm2)
        with self.assertWarnsRegex(FutureWarning, "has units incompatible with mS / cm\\^2; got an array of shape"):
            jax.jit(declare, static_argnums=1)(nan, u.mV)


if __name__ == "__main__":
    unittest.main()
