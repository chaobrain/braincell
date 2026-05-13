"""Unit tests for :mod:`braincell._base_ion`.

ARCH-03: verify Ion / MixIons / mix_ions live in their own module and
remain re-exported through :mod:`braincell._base` for back-compat.
"""

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp


class BaseIonSplitTest(unittest.TestCase):
    def test_ion_lives_in_base_ion(self) -> None:
        import braincell._base as base
        import braincell._base_ion as ion_mod

        self.assertIs(base.Ion, ion_mod.Ion)
        self.assertIs(base.MixIons, ion_mod.MixIons)
        self.assertIs(base.mix_ions, ion_mod.mix_ions)

    def test_direct_import_still_works(self) -> None:
        from braincell._base import Ion, MixIons, mix_ions
        self.assertTrue(callable(mix_ions))

    def test_ion_inherits_from_ion_channel(self) -> None:
        from braincell._base_channel import IonChannel
        from braincell._base_ion import Ion, MixIons
        self.assertTrue(issubclass(Ion, IonChannel))
        self.assertTrue(issubclass(MixIons, IonChannel))


class IonIndependentIntegrationDispatchTest(unittest.TestCase):
    def test_update_dispatches_to_make_integration_for_independent_ions(self) -> None:
        from braincell.ion import Calcium
        from braincell._base import Channel, IonInfo
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
                return 0.0 * u.nA / u.cm ** 2

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
        ion.update(jnp.array([-65.0]) * u.mV)

        self.assertEqual(len(ion.calls), 1)
        self.assertEqual(len(ion.channels["child"].calls), 1)
        child_args, child_kwargs = ion.channels["child"].calls[0]
        self.assertEqual(len(child_args), 2)
        self.assertEqual(child_kwargs, {})
        self.assertIsInstance(child_args[1], IonInfo)


if __name__ == "__main__":
    unittest.main()
