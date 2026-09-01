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

The ion-channel family classes are defined here; ``braincell`` re-exports
them, and that public path is the one users import.
"""

import unittest


class BaseChannelExportTest(unittest.TestCase):
    def test_public_namespace_reexports_this_module(self) -> None:
        import braincell
        import braincell._base_channel as channel_mod

        self.assertIs(braincell.IonChannel, channel_mod.IonChannel)
        self.assertIs(braincell.Channel, channel_mod.Channel)
        self.assertIs(braincell.Synapse, channel_mod.Synapse)
        self.assertIs(braincell.IonInfo, channel_mod.IonInfo)

    def test_channel_and_synapse_derive_from_ion_channel(self) -> None:
        from braincell._base_channel import Channel, IonChannel, Synapse

        self.assertTrue(issubclass(Channel, IonChannel))
        self.assertTrue(issubclass(Synapse, IonChannel))


if __name__ == "__main__":
    unittest.main()
