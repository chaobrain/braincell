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

from braincell.mech import (
    CurrentClamp,
    FunctionClamp,
    GapJunctionMechanism,
    Params,
    PointMechanism,
    ProbeMechanism,
    SineClamp,
    Synapse,
    SynapseMechanism,
)


class PointMechanismBaseTest(unittest.TestCase):
    def test_every_concrete_subclass_is_PointMechanism(self) -> None:
        cc = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        sine = SineClamp(amplitude=0.1 * u.nA, frequency=10 * u.Hz)
        fc = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        probe = ProbeMechanism(variable="v")
        syn = SynapseMechanism(synapse_type="AMPA")
        gap = GapJunctionMechanism()

        for mech in (cc, sine, fc, probe, syn, gap):
            self.assertIsInstance(mech, PointMechanism)


class CurrentClampTest(unittest.TestCase):
    def test_canonical_multisegment_form(self) -> None:
        cc = CurrentClamp(
            start=1.0 * u.ms,
            durations=(2.0 * u.ms, 3.0 * u.ms),
            amplitudes=(0.0 * u.nA, 0.3 * u.nA),
        )
        self.assertEqual(len(cc.durations), 2)
        self.assertEqual(len(cc.amplitudes), 2)
        self.assertEqual(cc.start.to_decimal(u.ms), 1.0)

    def test_step_classmethod_round_trip(self) -> None:
        cc = CurrentClamp.step(0.2 * u.nA, 50.0 * u.ms, delay=10.0 * u.ms)
        self.assertEqual(cc.start.to_decimal(u.ms), 10.0)
        self.assertEqual(len(cc.durations), 1)
        self.assertEqual(cc.durations[0].to_decimal(u.ms), 50.0)
        self.assertEqual(cc.amplitudes[0].to_decimal(u.nA), 0.2)

    def test_step_default_delay_is_zero(self) -> None:
        cc = CurrentClamp.step(0.1 * u.nA, 10.0 * u.ms)
        self.assertEqual(cc.start.to_decimal(u.ms), 0.0)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                start=0.0 * u.ms,
                durations=(1.0 * u.ms,),
                amplitudes=(0.0 * u.nA, 0.3 * u.nA),
            )

    def test_empty_durations_raise(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                start=0.0 * u.ms, durations=(), amplitudes=()
            )

    def test_zero_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                start=0.0 * u.ms,
                durations=(0.0 * u.ms,),
                amplitudes=(0.1 * u.nA,),
            )

    def test_negative_duration_raises(self) -> None:
        with self.assertRaises(ValueError):
            CurrentClamp(
                start=0.0 * u.ms,
                durations=(-1.0 * u.ms,),
                amplitudes=(0.1 * u.nA,),
            )

    def test_is_frozen_hashable(self) -> None:
        a = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        b = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        # Structural equality via dataclass auto __eq__
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_cannot_mutate(self) -> None:
        cc = CurrentClamp.step(0.1 * u.nA, 1.0 * u.ms)
        with self.assertRaises(Exception):
            cc.start = 5.0 * u.ms  # type: ignore[misc]


class SineClampTest(unittest.TestCase):
    def test_basic_construction(self) -> None:
        sc = SineClamp(
            amplitude=0.4 * u.nA,
            frequency=50.0 * u.Hz,
            duration=5.0 * u.ms,
        )
        self.assertEqual(sc.frequency.to_decimal(u.Hz), 50.0)
        self.assertEqual(sc.phase, 0.0)

    def test_hashable_and_equal(self) -> None:
        a = SineClamp(amplitude=0.4 * u.nA, frequency=50.0 * u.Hz)
        b = SineClamp(amplitude=0.4 * u.nA, frequency=50.0 * u.Hz)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class FunctionClampTest(unittest.TestCase):
    def test_callable_stored(self) -> None:
        fn = lambda t: 0.1 * u.nA
        fc = FunctionClamp(fn=fn, duration=4.0 * u.ms)
        self.assertIs(fc.fn, fn)

    def test_different_lambdas_are_distinct(self) -> None:
        a = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        b = FunctionClamp(fn=lambda t: 0.1 * u.nA)
        # Identity-based equality since lambdas compare by is
        self.assertNotEqual(a, b)


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


class SynapseMechanismTest(unittest.TestCase):
    def test_direct_construction(self) -> None:
        syn = SynapseMechanism(
            synapse_type="AMPA",
            params=Params(tau=5.0),
        )
        self.assertEqual(syn.synapse_type, "AMPA")
        self.assertEqual(syn.params["tau"], 5.0)

    def test_factory_builds_synapse(self) -> None:
        syn = Synapse("AMPA", tau_rise=0.5, tau_decay=5.0)
        self.assertIsInstance(syn, SynapseMechanism)
        self.assertEqual(syn.synapse_type, "AMPA")
        self.assertEqual(syn.params["tau_rise"], 0.5)

    def test_default_instance_name(self) -> None:
        syn = Synapse("AMPA")
        self.assertEqual(syn.instance_name, "AMPA")
        self.assertEqual(syn.identity, ("AMPA", "AMPA"))

    def test_override_instance_name(self) -> None:
        syn = Synapse("AMPA", name="ampa_main")
        self.assertEqual(syn.instance_name, "ampa_main")
        self.assertEqual(syn.identity, ("ampa_main", "AMPA"))

    def test_keyword_order_insensitive_equality(self) -> None:
        a = Synapse("AMPA", tau_rise=0.5, tau_decay=5.0)
        b = Synapse("AMPA", tau_decay=5.0, tau_rise=0.5)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_empty_synapse_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SynapseMechanism(synapse_type="")


class GapJunctionMechanismTest(unittest.TestCase):
    def test_default_empty_params(self) -> None:
        gap = GapJunctionMechanism()
        self.assertEqual(len(gap.params), 0)

    def test_with_params(self) -> None:
        gap = GapJunctionMechanism(params=Params(g=1.0))
        self.assertEqual(gap.params["g"], 1.0)


if __name__ == "__main__":
    unittest.main()
