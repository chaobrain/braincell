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
    Params,
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
        syn = Synapse("AMPA")
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
            CurrentClamp(
                delay=0.0 * u.ms, durations=(), amplitudes=()
            )

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

    def test_target_index_is_normalized(self) -> None:
        cc = CurrentClamp(
            durations=1.0 * u.ms,
            amplitudes=0.1 * u.nA,
            target_index=[2, 0],
        )
        self.assertEqual(cc.target_index, (2, 0))

    def test_bad_target_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                durations=1.0 * u.ms,
                amplitudes=0.1 * u.nA,
                target_index=[[0]],
            )
        with self.assertRaises(TypeError):
            CurrentClamp(
                durations=1.0 * u.ms,
                amplitudes=0.1 * u.nA,
                target_index=[0.5],
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
                amplitude=1.0 * u.nA, frequency=50.0 * u.Hz,
                duration=-1.0 * u.ms,
            )

    def test_phase_must_be_real_number(self) -> None:
        with self.assertRaises(TypeError):
            SineClamp(
                amplitude=1.0 * u.nA, frequency=50.0 * u.Hz, phase="hi",
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
        syn = Synapse("AMPA", tau=5.0)
        self.assertEqual(syn.synapse_type, "AMPA")
        self.assertEqual(syn.params["tau"], 5.0)

    def test_default_instance_name(self) -> None:
        syn = Synapse("AMPA")
        self.assertEqual(syn.instance_name, "AMPA")
        self.assertEqual(syn.identity, ("AMPA", "AMPA"))

    def test_override_instance_name(self) -> None:
        syn = Synapse("AMPA", name="ampa_main")
        self.assertEqual(syn.instance_name, "ampa_main")
        self.assertEqual(syn.identity, ("ampa_main", "AMPA"))

    def test_equality_and_hash(self) -> None:
        a = Synapse("AMPA", tau=5.0, e=0.0)
        b = Synapse("AMPA", e=0.0, tau=5.0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_empty_synapse_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Synapse("")

    def test_keyword_synapse_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Synapse(synapse_type="AMPA")  # type: ignore[call-arg]

    def test_params_mapping_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "keyword arguments"):
            Synapse("AMPA", params={"tau": 5.0})

    def test_synapse_is_immutable(self) -> None:
        syn = Synapse("AMPA")
        with self.assertRaises(AttributeError):
            syn.synapse_type = "NMDA"


if __name__ == "__main__":
    unittest.main()
