"""Unit tests for :mod:`braincell._base_channel`.

ARCH-03: verify ion-channel family classes live in their own module and
remain re-exported through :mod:`braincell._base` for back-compat.
"""

import unittest


class BaseChannelSplitTest(unittest.TestCase):
    def test_ion_channel_lives_in_base_channel(self) -> None:
        import braincell._base as base
        import braincell._base_channel as channel_mod

        self.assertIs(base.IonChannel, channel_mod.IonChannel)
        self.assertIs(base.Channel, channel_mod.Channel)
        self.assertIs(base.Synapse, channel_mod.Synapse)
        self.assertIs(base.IonInfo, channel_mod.IonInfo)

    def test_direct_import_still_works(self) -> None:
        from braincell._base import IonChannel, Channel, Synapse, IonInfo
        self.assertTrue(
            all(cls is not None for cls in (IonChannel, Channel, Synapse, IonInfo)),
        )


if __name__ == "__main__":
    unittest.main()
