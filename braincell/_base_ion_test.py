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

import gc
import unittest
import weakref

import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp

import braincell
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


class MixIonsConstructorChannelTest(unittest.TestCase):
    """``MixIons(*ions, **channels)`` must match ``MixIons(*ions).add(...)``.

    The constructor used to update ``self.channels`` directly, skipping the
    hierarchy check *and* the ``register_external_current`` calls that
    ``add`` makes. A channel passed to the constructor therefore reached
    no owner ion, and ``ion.current(V, include_external=True)`` returned
    ``None`` where the ``add`` spelling returned a current.
    """

    @staticmethod
    def _build(via_constructor: bool):
        k = PotassiumFixed(size=1)
        ca = CalciumFixed(size=1)
        channel = _RecordingKCaChannel(size=1)
        if via_constructor:
            mix = MixIons(k, ca, kca=channel)
        else:
            mix = MixIons(k, ca)
            mix.add(kca=channel)
        return k, mix

    def test_constructor_registers_external_currents_like_add(self) -> None:
        via_init, _ = self._build(True)
        via_add, _ = self._build(False)
        self.assertEqual(len(via_init.external_currents), len(via_add.external_currents))
        self.assertEqual(len(via_init.external_currents), 1)

    def test_constructor_channel_reaches_the_owner_ion_current(self) -> None:
        V = jnp.zeros((1,)) * u.mV
        via_init, _ = self._build(True)
        via_add, _ = self._build(False)
        self.assertIsNotNone(via_init.current(V, include_external=True))
        u.math.allclose(
            via_init.current(V, include_external=True),
            via_add.current(V, include_external=True),
        )

    def test_constructor_rejects_a_channel_whose_root_type_does_not_match(self) -> None:
        class _SodiumOnly(Channel):
            root_type = SodiumFixed

            def init_state(self, V, Na, batch_size=None):  # pragma: no cover
                pass

            def current(self, V, Na):  # pragma: no cover
                return 0.0 * u.nA / u.cm**2

        with self.assertRaises(TypeError):
            MixIons(PotassiumFixed(size=1), CalciumFixed(size=1), bad=_SodiumOnly(size=1))


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


class EmptyPoolCurrentTest(unittest.TestCase):
    """A pool with no channels of its own says so with ``None``.

    ``Ion.current`` already returned ``None``; ``MixIons.current`` returned
    a bare, unitless ``0.0``. The two disagreed, and the second is not a
    valid addend for a current density -- summing over the pools raised
    ``UnitMismatchError`` instead of producing a current.
    """

    def _kca_cell(self):
        cell = braincell.SingleCompartment(1, V_initializer=braintools.init.Constant(-65.0 * u.mV))
        cell.k = braincell.ion.PotassiumFixed(1, E=-90.0 * u.mV)
        cell.ca = braincell.ion.CalciumDetailed(1, C_rest=2.4e-4 * u.mM, tau=10.0 * u.ms, d=0.5 * u.um)
        cell.kca = braincell.mix_ions(cell.k, cell.ca)
        cell.kca.add(ahp=braincell.channel.AHP_De1994(1, g_max=10.0 * u.mS / u.cm**2))
        brainstate.nn.init_all_states(cell)
        return cell

    def test_both_pool_types_return_none_when_empty(self) -> None:
        cell = self._kca_cell()
        V = cell.V.value
        # The AHP channel belongs to the mixed pool, so neither ion has a
        # channel of its own.
        self.assertIsNone(cell.k.current(V))
        self.assertIsNone(cell.ca.current(V))
        self.assertIsNone(braincell.mix_ions(cell.k, cell.ca).current(V))

    def test_a_kca_only_model_integrates(self) -> None:
        # The whole point: this model is the ordinary way to write a
        # calcium-dependent potassium current, and it used to crash in
        # ``compute_derivative`` before a single step ran.
        cell = self._kca_cell()
        for i in range(3):
            with brainstate.environ.context(dt=0.01 * u.ms, t=i * 0.01 * u.ms):
                cell.update(0.0 * u.nA / u.cm**2)
        self.assertTrue(u.math.isfinite(cell.V.value).all())
        self.assertTrue(u.math.isfinite(cell.ca.Ci.value).all())

    def test_a_dynamic_ion_accepts_an_absent_drive_current(self) -> None:
        cell = self._kca_cell()
        zero = cell.ca._drive_current(None)
        self.assertEqual(u.get_unit(zero).dim, (u.mA / u.cm**2).dim)
        self.assertEqual(float(u.math.sum(zero.to_decimal(u.mA / u.cm**2))), 0.0)
        supplied = 1.0 * u.mA / u.cm**2
        self.assertIs(cell.ca._drive_current(supplied), supplied)


class MixIonsPackingTest(unittest.TestCase):
    """Each mixed ion is packed once per lifecycle call, not once per channel."""

    def _pool(self, n_channels: int):
        cell = braincell.SingleCompartment(4, V_initializer=braintools.init.Constant(-65.0 * u.mV))
        cell.k = braincell.ion.PotassiumFixed(4, E=-90.0 * u.mV)
        cell.ca = braincell.ion.CalciumDetailed(4, C_rest=2.4e-4 * u.mM, tau=10.0 * u.ms, d=0.5 * u.um)
        cell.kca = braincell.mix_ions(cell.k, cell.ca)
        cell.kca.add(
            **{f"ahp{i}": braincell.channel.AHP_De1994(4, g_max=1.0 * u.mS / u.cm**2) for i in range(n_channels)}
        )
        brainstate.nn.init_all_states(cell)
        return cell

    def test_pack_info_runs_once_per_ion_regardless_of_channel_count(self) -> None:
        cell = self._pool(5)
        calls = []
        for ion in cell.kca.ions:
            original = ion.pack_info

            def counted(_original=original, _ion=ion):
                calls.append(type(_ion).__name__)
                return _original()

            ion.pack_info = counted

        cell.kca.current(cell.V.value)
        # Two ions, five channels. Packing per (channel, root) would be ten.
        self.assertEqual(len(calls), 2)
        self.assertEqual(sorted(calls), sorted({type(ion).__name__ for ion in cell.kca.ions}))

    def test_every_channel_still_receives_its_own_roots_in_order(self) -> None:
        cell = self._pool(2)
        infos = cell.kca._pack_ion_infos()
        channel = cell.kca.channels["ahp0"]
        selected = cell.kca._infos_for(channel, infos)
        self.assertEqual(len(selected), len(channel.root_type.__args__))
        for info, root in zip(selected, channel.root_type.__args__):
            self.assertIs(info, infos[id(cell.kca._get_ion(root))])

    def test_the_external_current_callback_does_not_retain_the_mixed_pool(self) -> None:
        # The callback lives in ``Ion._external_currents`` for the life of
        # the model. Capturing ``self`` pinned the whole ``MixIons``, and
        # through it every child channel, behind a reference cycle.
        cell = self._pool(1)
        callback = next(iter(cell.k.external_currents.values()))
        captured = [cell_ref.cell_contents for cell_ref in (callback.__closure__ or ())]
        self.assertFalse(any(isinstance(obj, braincell.MixIons) for obj in captured))

    def test_the_mixed_pool_is_freed_without_the_cyclic_collector(self) -> None:
        cell = self._pool(1)
        pool = weakref.ref(cell.kca)
        channel = weakref.ref(cell.kca.channels["ahp0"])
        gc.disable()
        try:
            del cell
            self.assertIsNone(pool(), "MixIons survived refcounting alone")
            self.assertIsNone(channel(), "child channel survived refcounting alone")
        finally:
            gc.enable()


if __name__ == "__main__":
    unittest.main()
