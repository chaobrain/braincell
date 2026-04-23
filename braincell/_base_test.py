# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell import Channel, HHTypedNeuron, IonChannel, IonInfo, MixIons
from braincell.ion import CalciumFixed, PotassiumFixed, SodiumFixed


class _RecordingKCaChannel(Channel):
    """Records every call to ``update`` so the test can assert dispatch."""

    root_type = brainstate.mixin.JointTypes[PotassiumFixed, CalciumFixed]

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self.calls = []

    def update(self, V, K: IonInfo, Ca: IonInfo):
        self.calls.append((V, K, Ca))

    def init_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def reset_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def compute_derivative(self, V, K, Ca):  # pragma: no cover
        pass

    def current(self, V, K, Ca):  # pragma: no cover
        return 0.0 * u.nA / u.cm ** 2


class MixIonsUpdateReceiverTest(unittest.TestCase):
    """Regression for CRIT-01: MixIons.update iterated the wrong graph."""

    def test_update_reaches_child_channel(self) -> None:
        k = PotassiumFixed(size=1)
        ca = CalciumFixed(size=1)
        mix = MixIons(k, ca)
        rec = _RecordingKCaChannel(size=1)
        mix.add(kca=rec)

        V = jnp.zeros((1,)) * u.mV
        mix.update(V)

        self.assertEqual(len(rec.calls), 1, "child channel must see exactly one update")
        seen_V, seen_K, seen_Ca = rec.calls[0]
        self.assertIsInstance(seen_K, IonInfo)
        self.assertIsInstance(seen_Ca, IonInfo)


class IonCurrentExternalOnlyTest(unittest.TestCase):
    """Regression for CRIT-02: Ion.current crashed with empty nodes."""

    def test_external_only_returns_sum_without_crashing(self) -> None:
        na = SodiumFixed(size=1)

        expected = 1.5 * u.nA / u.cm ** 2
        na.register_external_current(
            "probe",
            lambda V, ion_info: u.math.broadcast_to(expected, V.shape),
        )

        V = jnp.zeros((1,)) * u.mV
        out = na.current(V, include_external=True)

        self.assertTrue(
            u.math.allclose(
                out.to_decimal(u.nA / u.cm ** 2),
                expected.to_decimal(u.nA / u.cm ** 2),
                atol=1e-9,
            )
        )

    def test_external_only_without_request_returns_none(self) -> None:
        na = SodiumFixed(size=1)
        na.register_external_current(
            "probe",
            lambda V, ion_info: 1.0 * u.nA / u.cm ** 2,
        )
        V = jnp.zeros((1,)) * u.mV
        self.assertIsNone(na.current(V, include_external=False))


class _FakeChannelLike(brainstate.nn.Module):
    """Not a Channel subclass, but a graph Node with a compatible root_type.

    Satisfies ``check_hierarchies`` (Node + ``issubclass(Ion_subclass, IonChannel)``
    holds) yet is not a ``Channel`` — so ``_format_elements(Channel, ...)`` must
    reject it.
    """

    root_type = IonChannel

    def __init__(self, size=1):
        super().__init__()
        self.in_size = (size,)


class IonAddChannelValidationTest(unittest.TestCase):
    """Regression for HIGH-02: Ion.add must reject non-Channel objects."""

    def test_add_rejects_non_channel_object_even_with_root_type(self) -> None:
        na = SodiumFixed(size=1)
        with self.assertRaises(TypeError):
            na.add(fake=_FakeChannelLike())


class MixIonsFactoryArityTest(unittest.TestCase):
    """LOW-04: mix_ions must flag single-ion calls with its own message."""

    def test_single_ion_raises_with_mix_ions_message(self) -> None:
        from braincell import mix_ions
        with self.assertRaises(AssertionError) as ctx:
            mix_ions(SodiumFixed(size=1))
        self.assertIn("mix_ions", str(ctx.exception))


class HHTypedNeuronGetSpikeTest(unittest.TestCase):
    """ARCH-04: get_spike and _cast_like live on the shared base."""

    def test_get_spike_is_method_on_base(self) -> None:
        from braincell._base import HHTypedNeuron
        self.assertTrue(hasattr(HHTypedNeuron, "get_spike"))
        self.assertTrue(callable(HHTypedNeuron.get_spike))

    def test_cast_like_is_importable_from_base(self) -> None:
        from braincell._base import _cast_like
        self.assertTrue(callable(_cast_like))

    def test_single_compartment_inherits_get_spike(self) -> None:
        from braincell._base import HHTypedNeuron
        from braincell._single_compartment.base import SingleCompartment

        self.assertIs(
            SingleCompartment.get_spike,
            HHTypedNeuron.get_spike,
            msg="SingleCompartment must not redefine get_spike after ARCH-04",
        )

        sc = SingleCompartment(size=1, V_th=0.0 * u.mV)
        spk = sc.get_spike(jnp.array([-10.0]) * u.mV, jnp.array([10.0]) * u.mV)
        self.assertGreater(float(spk[0]), 0.0)
