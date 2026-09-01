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

"""Unit tests for :mod:`braincell._base_ion`."""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp

from braincell._base_channel import Channel, IonChannel, IonInfo
from braincell._base_ion import Ion, MixIons, mix_ions
from braincell.ion import CalciumFixed, PotassiumFixed, SodiumFixed


class BaseIonExportTest(unittest.TestCase):
    def test_public_namespace_reexports_this_module(self) -> None:
        import braincell
        import braincell._base_ion as ion_mod

        self.assertIs(braincell.Ion, ion_mod.Ion)
        self.assertIs(braincell.MixIons, ion_mod.MixIons)
        self.assertIs(braincell.mix_ions, ion_mod.mix_ions)

    def test_ion_inherits_from_ion_channel(self) -> None:
        self.assertTrue(issubclass(Ion, IonChannel))
        self.assertTrue(issubclass(MixIons, IonChannel))


class _RecordingKCaChannel(Channel):
    """Records every call to ``ind_update`` so the test can assert dispatch."""

    root_type = brainstate.mixin.JointTypes[PotassiumFixed, CalciumFixed]

    def __init__(self, size, name=None):
        super().__init__(size=size, name=name)
        self.calls = []

    def ind_update(self, V, K: IonInfo, Ca: IonInfo):
        self.calls.append((V, K, Ca))

    def init_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def reset_state(self, V, K, Ca, batch_size=None):  # pragma: no cover
        pass

    def compute_derivative(self, V, K, Ca):  # pragma: no cover
        pass

    def current(self, V, K, Ca):  # pragma: no cover
        return 0.0 * u.nA / u.cm**2


class MixIonsIndependentUpdateReceiverTest(unittest.TestCase):
    """Regression for CRIT-01: MixIons.ind_update iterated the wrong graph."""

    def test_ind_update_reaches_child_channel(self) -> None:
        k = PotassiumFixed(size=1)
        ca = CalciumFixed(size=1)
        mix = MixIons(k, ca)
        rec = _RecordingKCaChannel(size=1)
        mix.add(kca=rec)

        V = jnp.zeros((1,)) * u.mV
        mix.ind_update(V)

        self.assertEqual(len(rec.calls), 1, "child channel must see exactly one update")
        _, seen_K, seen_Ca = rec.calls[0]
        self.assertIsInstance(seen_K, IonInfo)
        self.assertIsInstance(seen_Ca, IonInfo)


class IonCurrentExternalOnlyTest(unittest.TestCase):
    """Regression for CRIT-02: Ion.current crashed with empty nodes."""

    def test_external_only_returns_sum_without_crashing(self) -> None:
        na = SodiumFixed(size=1)

        expected = 1.5 * u.nA / u.cm**2
        na.register_external_current(
            "probe",
            lambda V, ion_info: u.math.broadcast_to(expected, V.shape),
        )

        V = jnp.zeros((1,)) * u.mV
        out = na.current(V, include_external=True)

        self.assertTrue(
            u.math.allclose(
                out.to_decimal(u.nA / u.cm**2),
                expected.to_decimal(u.nA / u.cm**2),
                atol=1e-9,
            )
        )

    def test_external_only_without_request_returns_none(self) -> None:
        na = SodiumFixed(size=1)
        na.register_external_current(
            "probe",
            lambda V, ion_info: 1.0 * u.nA / u.cm**2,
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
        with self.assertRaises(AssertionError) as ctx:
            mix_ions(SodiumFixed(size=1))
        self.assertIn("mix_ions", str(ctx.exception))


class IonIndependentIntegrationDispatchTest(unittest.TestCase):
    def test_ind_update_dispatches_to_make_integration_for_independent_ions(self) -> None:
        from braincell.ion import Calcium
        from braincell.quad.protocol import IndependentIntegration

        class _IndependentChildChannel(Channel, IndependentIntegration):
            root_type = Calcium

            def __init__(self, size=1):
                Channel.__init__(self, size=size, name=None)
                IndependentIntegration.__init__(self, solver="euler")
                self.calls = []

            def make_integration(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def init_state(self, V, ion, batch_size=None):  # pragma: no cover
                pass

            def reset_state(self, V, ion, batch_size=None):  # pragma: no cover
                pass

            def compute_derivative(self, V, ion):  # pragma: no cover
                pass

            def current(self, V, ion):  # pragma: no cover
                return 0.0 * u.nA / u.cm**2

        class _IndependentIon(Calcium, IndependentIntegration):
            def __init__(self):
                Calcium.__init__(self, size=1, name=None, child=_IndependentChildChannel())
                IndependentIntegration.__init__(self, solver="euler")
                self.Ci = 0.1 * u.mM
                self.Co = 2.0 * u.mM
                self.temp = u.celsius2kelvin(36.0)
                self.valence = 2
                self.calls = []

            @property
            def E(self):
                return 120.0 * u.mV

            def make_integration(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        ion = _IndependentIon()
        ion.ind_update(jnp.array([-65.0]) * u.mV)

        self.assertEqual(len(ion.calls), 1)
        self.assertEqual(len(ion.channels["child"].calls), 1)
        child_args, child_kwargs = ion.channels["child"].calls[0]
        self.assertEqual(len(child_args), 2)
        self.assertEqual(child_kwargs, {})
        self.assertIsInstance(child_args[1], IonInfo)

    def test_ind_update_skips_dependent_child_channel_under_independent_ion(self) -> None:
        from braincell.ion import Calcium
        from braincell.quad.protocol import DiffEqSingleState
        from braincell.quad.protocol import IndependentIntegration

        class _DependentChildChannel(Channel):
            root_type = Calcium

            def __init__(self, size=1):
                super().__init__(size=size, name=None)
                self.x = DiffEqSingleState(jnp.asarray([1.0]))

            def init_state(self, V, ion, batch_size=None):  # pragma: no cover
                pass

            def reset_state(self, V, ion, batch_size=None):  # pragma: no cover
                pass

            def compute_derivative(self, V, ion):  # pragma: no cover
                pass

            def current(self, V, ion):  # pragma: no cover
                return 0.0 * u.nA / u.cm**2

        class _IndependentIon(Calcium, IndependentIntegration):
            def __init__(self):
                Calcium.__init__(self, size=1, name=None, child=_DependentChildChannel())
                IndependentIntegration.__init__(self, solver="euler")
                self.Ci = 0.1 * u.mM
                self.Co = 2.0 * u.mM
                self.temp = u.celsius2kelvin(36.0)
                self.valence = 2
                self.calls = []

            @property
            def E(self):
                return 120.0 * u.mV

            def make_integration(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        ion = _IndependentIon()
        ion.ind_update(jnp.array([-65.0]) * u.mV)

        self.assertEqual(len(ion.calls), 1)
        self.assertEqual(float(ion.channels["child"].x.value[0]), 1.0)

    def test_recursive_child_false_skips_child_lifecycle_methods(self) -> None:
        from braincell.ion import Calcium

        class _ChildChannel(Channel):
            root_type = Calcium

            def __init__(self):
                super().__init__(size=1, name=None)
                self.calls = []

            def pre_integral(self, V, ion):
                self.calls.append(("pre", ion))

            def compute_derivative(self, V, ion):
                self.calls.append(("compute", ion))

            def post_integral(self, V, ion):
                self.calls.append(("post", ion))

            def current(self, V, ion):  # pragma: no cover
                return 0.0 * u.nA / u.cm**2

        class _Ion(Calcium):
            def __init__(self):
                Calcium.__init__(self, size=1, name=None, child=_ChildChannel())
                self.Ci = 0.1 * u.mM
                self.Co = 2.0 * u.mM
                self.temp = u.celsius2kelvin(36.0)
                self.valence = 2
                self.calls = []

            @property
            def E(self):
                return 120.0 * u.mV

            def _ion_pre_integral_hook(self, V):
                self.calls.append("pre")

            def _ion_compute_derivative_hook(self, V):
                self.calls.append("compute")

            def _ion_post_integral_hook(self, V):
                self.calls.append("post")

        ion = _Ion()
        V = jnp.array([-65.0]) * u.mV

        ion.pre_integral(V, recursive_child=False)
        ion.compute_derivative(V, recursive_child=False)
        ion.post_integral(V, recursive_child=False)

        self.assertEqual(ion.calls, ["pre", "compute", "post"])
        self.assertEqual(ion.channels["child"].calls, [])

    def test_ind_update_recursive_child_false_skips_independent_child_channel(self) -> None:
        from braincell.ion import Calcium
        from braincell.quad.protocol import IndependentIntegration

        class _IndependentChildChannel(Channel, IndependentIntegration):
            root_type = Calcium

            def __init__(self):
                Channel.__init__(self, size=1, name=None)
                IndependentIntegration.__init__(self, solver="euler")
                self.calls = []

            def make_integration(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def current(self, V, ion):  # pragma: no cover
                return 0.0 * u.nA / u.cm**2

        class _IndependentIon(Calcium, IndependentIntegration):
            def __init__(self):
                Calcium.__init__(self, size=1, name=None, child=_IndependentChildChannel())
                IndependentIntegration.__init__(self, solver="euler")
                self.Ci = 0.1 * u.mM
                self.Co = 2.0 * u.mM
                self.temp = u.celsius2kelvin(36.0)
                self.valence = 2
                self.calls = []

            @property
            def E(self):
                return 120.0 * u.mV

            def make_integration(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        ion = _IndependentIon()
        ion.ind_update(jnp.array([-65.0]) * u.mV, recursive_child=False)

        self.assertEqual(len(ion.calls), 1)
        args, kwargs = ion.calls[0]
        self.assertEqual(len(args), 1)
        self.assertEqual(kwargs, {"recursive_child": False})
        self.assertEqual(ion.channels["child"].calls, [])


if __name__ == "__main__":
    unittest.main()
