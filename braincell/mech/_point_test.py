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

import brainunit as u
import numpy as np

from braincell.mech import (
    CurrentProbe,
    CurrentClamp,
    FunctionClamp,
    Junction,
    MechanismProbe,
    Mechanism,
    Point,
    ProbeMechanism,
    SineClamp,
    StateProbe,
    Synapse,
)


class PointBaseTest(unittest.TestCase):
    def test_every_concrete_subclass_is_Point(self) -> None:
        cc = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        sine = SineClamp(amplitude=0.1 * u.nA, frequency=10 * u.Hz)
        fc = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        state_probe = StateProbe(name="v_soma")
        mechanism_probe = MechanismProbe(name="na_p", mechanism="na_soma", field="p")
        current_probe = CurrentProbe(ion="na", mechanism="na_soma")
        probe = ProbeMechanism(variable="v")
        syn = Synapse("ExpSyn")
        gap = Junction()

        for mech in (cc, sine, fc, state_probe, mechanism_probe, current_probe, probe, syn, gap):
            self.assertIsInstance(mech, Point)
            self.assertIsInstance(mech, Mechanism)


class CurrentClampTest(unittest.TestCase):
    def test_canonical_multisegment_form(self) -> None:
        cc = CurrentClamp(
            delay=1.0 * u.ms,
            durations=(2.0 * u.ms, 3.0 * u.ms),
            amplitudes=(0.0 * u.nA, 0.3 * u.nA),
        )
        self.assertEqual(cc.durations.shape, (2,))
        self.assertEqual(cc.amplitudes.shape, (2,))
        self.assertEqual(cc.delay.to_decimal(u.ms), 1.0)

    def test_multisegment_population_values_keep_step_axis_last(self) -> None:
        cc = CurrentClamp(
            delay=1.0 * u.ms,
            durations=(2.0 * u.ms, 3.0 * u.ms),
            amplitudes=(
                u.Quantity(np.asarray([0.1, 0.2]), u.nA),
                u.Quantity(np.asarray([0.3, 0.4]), u.nA),
            ),
        )
        self.assertEqual(cc.amplitudes.shape, (2, 2))
        np.testing.assert_allclose(
            cc.amplitudes.to_decimal(u.nA),
            np.asarray([[0.1, 0.3], [0.2, 0.4]]),
        )

    def test_scalar_single_segment_form(self) -> None:
        cc = CurrentClamp(delay=10.0 * u.ms, durations=50.0 * u.ms, amplitudes=0.2 * u.nA)
        self.assertEqual(cc.delay.to_decimal(u.ms), 10.0)
        self.assertEqual(cc.durations.to_decimal(u.ms), 50.0)
        self.assertEqual(cc.amplitudes.to_decimal(u.nA), 0.2)

    def test_default_delay_is_zero(self) -> None:
        cc = CurrentClamp(durations=10.0 * u.ms, amplitudes=0.1 * u.nA)
        self.assertEqual(cc.delay.to_decimal(u.ms), 0.0)

    def test_empty_durations_raise(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(delay=0.0 * u.ms, durations=(), amplitudes=())

    def test_zero_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(0.0 * u.ms,),
                amplitudes=(0.1 * u.nA,),
            )

    def test_negative_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                delay=0.0 * u.ms,
                durations=(-1.0 * u.ms,),
                amplitudes=(0.1 * u.nA,),
            )

    def test_is_frozen_hashable(self) -> None:
        a = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        b = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        # Structural equality via dataclass auto __eq__
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_cannot_mutate(self) -> None:
        cc = CurrentClamp(durations=1.0 * u.ms, amplitudes=0.1 * u.nA)
        with self.assertRaises(Exception):
            cc.delay = 5.0 * u.ms  # type: ignore[misc]


class SineClampTest(unittest.TestCase):
    def test_basic_construction(self) -> None:
        sc = SineClamp(
            amplitude=0.4 * u.nA,
            frequency=50.0 * u.Hz,
            delay=1.0 * u.ms,
            duration=5.0 * u.ms,
        )
        self.assertEqual(sc.frequency.to_decimal(u.Hz), 50.0)
        self.assertEqual(sc.delay.to_decimal(u.ms), 1.0)
        self.assertEqual(sc.phase, 0.0)

    def test_hashable_and_equal(self) -> None:
        a = SineClamp(amplitude=0.4 * u.nA, frequency=50.0 * u.Hz)
        b = SineClamp(amplitude=0.4 * u.nA, frequency=50.0 * u.Hz)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class SineClampValidatesInputsTest(unittest.TestCase):
    """MED-02: SineClamp rejects non-positive frequency / duration."""

    def test_zero_frequency_raises(self) -> None:
        with self.assertRaises(ValueError):
            SineClamp(amplitude=1.0 * u.nA, frequency=0.0 * u.Hz)

    def test_negative_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            SineClamp(
                amplitude=1.0 * u.nA,
                frequency=50.0 * u.Hz,
                duration=-1.0 * u.ms,
            )

    def test_phase_must_be_real_number(self) -> None:
        with self.assertRaises(TypeError):
            SineClamp(
                amplitude=1.0 * u.nA,
                frequency=50.0 * u.Hz,
                phase="hi",
            )


class FunctionClampTest(unittest.TestCase):
    def test_callable_stored(self) -> None:
        fn = lambda t: 0.1 * u.nA
        fc = FunctionClamp(fn=fn)
        self.assertIs(fc.fn, fn)

    def test_different_lambdas_are_distinct(self) -> None:
        a = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        b = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        # Identity-based equality since lambdas compare by is
        self.assertNotEqual(a, b)


class FunctionClampValidatesInputsTest(unittest.TestCase):
    """MED-02: FunctionClamp rejects non-callable fn."""

    def test_fn_must_be_callable(self) -> None:
        with self.assertRaises(TypeError):
            FunctionClamp(fn=None)


class StateProbeTest(unittest.TestCase):
    def test_basic_construction(self) -> None:
        probe = StateProbe()
        self.assertIsNone(probe.name)
        self.assertEqual(probe.field, "v")

    def test_only_v_is_supported(self) -> None:
        with self.assertRaises(ValueError):
            StateProbe(name="bad", field="cm")


class MechanismProbeTest(unittest.TestCase):
    def test_basic_construction(self) -> None:
        probe = MechanismProbe(mechanism="na_soma", field="p")
        self.assertIsNone(probe.name)
        self.assertEqual(probe.mechanism, "na_soma")
        self.assertEqual(probe.field, "p")

    def test_fields_must_be_non_empty(self) -> None:
        with self.assertRaises(ValueError):
            MechanismProbe(name="x", mechanism="", field="p")
        with self.assertRaises(ValueError):
            MechanismProbe(name="x", mechanism="na", field="")
        with self.assertRaises(ValueError):
            MechanismProbe(name="", mechanism="na", field="p")


class CurrentProbeTest(unittest.TestCase):
    def test_basic_construction_with_mechanism(self) -> None:
        probe = CurrentProbe(ion="k", mechanism="K_Kv_test")
        self.assertEqual(probe.ion, "k")
        self.assertEqual(probe.mechanism, "K_Kv_test")
        self.assertIsNone(probe.name)

    def test_basic_construction_with_mechanism_only(self) -> None:
        probe = CurrentProbe(mechanism="HCN_HM1992")
        self.assertIsNone(probe.ion)
        self.assertEqual(probe.mechanism, "HCN_HM1992")

    def test_basic_construction_for_total_ion_current(self) -> None:
        probe = CurrentProbe(ion="k")
        self.assertEqual(probe.ion, "k")
        self.assertIsNone(probe.mechanism)

    def test_invalid_fields_raise(self) -> None:
        with self.assertRaises(ValueError):
            CurrentProbe(ion="")
        with self.assertRaises(ValueError):
            CurrentProbe(ion="k", mechanism="")
        with self.assertRaises(ValueError):
            CurrentProbe()
        with self.assertRaises(ValueError):
            CurrentProbe(ion="k", name="")


class ProbeMechanismTest(unittest.TestCase):
    def test_basic_construction(self) -> None:
        p = ProbeMechanism(variable="v", target="soma")
        self.assertEqual(p.variable, "v")
        self.assertEqual(p.target, "soma")

    def test_default_target_is_none(self) -> None:
        p = ProbeMechanism(variable="v")
        self.assertIsNone(p.target)

    def test_hashable(self) -> None:
        a = ProbeMechanism(variable="v")
        b = ProbeMechanism(variable="v")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class ProbeMechanismValidatesInputsTest(unittest.TestCase):
    """MED-02: ProbeMechanism rejects empty variable name."""

    def test_empty_variable_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            ProbeMechanism(variable="")


class SynapseTest(unittest.TestCase):
    def test_direct_construction(self) -> None:
        with self.assertWarnsRegex(FutureWarning, "Synapse 'ExpSyn' parameter 'tau' requires a quantity"):
            syn = Synapse("ExpSyn", tau=5.0)
        self.assertEqual(syn.synapse_type, "ExpSyn")
        self.assertEqual(syn.params["tau"], 5.0)

    def test_default_instance_name(self) -> None:
        syn = Synapse("ExpSyn")
        self.assertEqual(syn.instance_name, "ExpSyn")

    def test_override_instance_name(self) -> None:
        syn = Synapse("ExpSyn", name="fast")
        self.assertEqual(syn.instance_name, "fast")

    def test_equality_and_hash(self) -> None:
        # Bare numbers are invalid but still compared as data (MECH-26).
        with self.assertWarns(FutureWarning):
            a = Synapse("ExpSyn", tau=5.0, e=0.0)
        with self.assertWarns(FutureWarning):
            b = Synapse("ExpSyn", e=0.0, tau=5.0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_empty_synapse_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Synapse("")

    def test_keyword_synapse_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Synapse(synapse_type="ExpSyn")  # type: ignore[call-arg]

    def test_params_mapping_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "keyword arguments"):
            Synapse("ExpSyn", params={"tau": 5.0})

    def test_synapse_is_immutable(self) -> None:
        syn = Synapse("ExpSyn")
        with self.assertRaises(AttributeError):
            syn.synapse_type = "Exp2Syn"

    def test_synapse_type_is_not_resolved_at_declaration_time(self) -> None:
        # Name resolution is deferred to Cell build, matching Channel/Ion.
        # ``_compute.state_test`` covers the registry error that arrives there.
        self.assertEqual(Synapse("NotARealSynapse").synapse_type, "NotARealSynapse")


class SynapseClassArgumentTest(unittest.TestCase):
    """CELL-25: ``Synapse`` accepts classes, like ``Channel`` and ``Ion``."""

    def test_registered_class_equals_its_name(self) -> None:
        import braincell

        by_class = Synapse(braincell.synapse.ExpSyn, name="fast", tau=2.0 * u.ms)
        by_name = Synapse("ExpSyn", name="fast", tau=2.0 * u.ms)
        self.assertEqual(by_class, by_name)
        self.assertEqual(hash(by_class), hash(by_name))
        self.assertEqual(by_class.synapse_type, "ExpSyn")

    def test_unregistered_class_runs_on_a_cell(self) -> None:
        import braincell
        from braincell.filter import AllRegion, RootLocation

        class SlowExpSyn(braincell.synapse.ExpSyn):
            pass

        declaration = Synapse(SlowExpSyn, name="slow", tau=5.0 * u.ms)
        self.assertIs(braincell.mech.get_registry().get("synapse", declaration.synapse_type), SlowExpSyn)
        cell = braincell.Cell(braincell.Morphology.from_soma(), pop_size=2)
        cell.paint(AllRegion(), braincell.mech.Channel("IL"))
        cell.place(RootLocation(0.5), declaration)
        cell.init_state()
        cell.run(dt=0.025 * u.ms, duration=0.25 * u.ms)
        runtime_types = {type(node) for node in cell.nodes().values()}
        self.assertIn(SlowExpSyn, runtime_types)

    def test_instances_and_foreign_classes_are_rejected(self) -> None:
        import braincell

        with self.assertRaisesRegex(TypeError, "instance of 'int'"):
            Synapse(3)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "synapse declaration needs a subclass"):
            Synapse(braincell.channel.IL)


class SynapseDeclarationWarningTest(unittest.TestCase):
    """MECH-26: invalid literal parameters warn where the declaration is written."""

    def test_invalid_literals_warn_at_this_line_and_still_raise_at_init_state(self) -> None:
        import braincell
        from braincell.filter import AllRegion, RootLocation

        cases = (
            (lambda: Synapse("ExpSyn", name="fast", tau=5.0), TypeError, ("Synapse 'fast' (ExpSyn)", "'tau'", "ms")),
            (
                lambda: Synapse("ExpSyn", name="fast", tau=5.0 * u.mV),
                ValueError,
                ("Synapse 'fast' (ExpSyn)", "'tau'", "ms", "5.0 mV"),
            ),
            (
                lambda: Synapse("ExpSyn", name="fast", taus=5.0 * u.ms),
                TypeError,
                ("Synapse 'fast' (ExpSyn)", "'taus'", "Did you mean 'tau'?"),
            ),
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
                cell = braincell.Cell(braincell.Morphology.from_soma())
                cell.paint(AllRegion(), braincell.mech.Channel("IL"))
                cell.place(RootLocation(0.5), declaration)
                with self.assertRaises(error):
                    cell.init_state()

    def test_valid_and_unresolvable_declarations_do_not_warn(self) -> None:
        import warnings

        import braincell

        for build in (
            lambda: Synapse("ExpSyn", tau=2.0 * u.ms, e=0.0 * u.mV),
            lambda: Synapse(braincell.synapse.ExpSyn, tau=np.array([2.0, 3.0]) * u.ms),
            lambda: Synapse("NotARealSynapse", tau=5.0),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                build()


if __name__ == "__main__":
    unittest.main()
