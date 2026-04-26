"""Unit tests for :mod:`braincell._base_ion`.

ARCH-03: verify Ion / MixIons / mix_ions live in their own module and
remain re-exported through :mod:`braincell._base` for back-compat.
"""

import unittest


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


if __name__ == "__main__":
    unittest.main()
