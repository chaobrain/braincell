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
