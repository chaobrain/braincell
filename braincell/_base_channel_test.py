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
    def test_synapse_legacy_schema_requires_migration(self):
        from braincell import Synapse
        from braincell.mech import ParameterSpec

        class Legacy(Synapse):
            parameters = {"gain": ParameterSpec(1.0)}

        with self.assertRaisesRegex(TypeError, "explicit __init__"):
            Legacy(1)

    def test_synapse_signatures_forward_inheritance_and_required_fields(self):
        import brainunit as u
        import numpy as np
        from braincell import Synapse
        from braincell.synapse import ExpSyn

        class Extended(ExpSyn):
            def __init__(self, size, gain=1.0, **kwargs):
                super().__init__(size, **kwargs)
                self._init_parameters(gain=gain)

        self.assertEqual(set(Extended.parameter_info()), {"tau", "e", "gain"})
        self.assertEqual(float(Extended(1).gain), 1.0)

        class Required(Synapse):
            def __init__(self, size, reversal):
                super().__init__(size)
                self._init_parameters(reversal=reversal)

        node = Required(2, -70.0 * u.mV)
        np.testing.assert_allclose(node.reversal.to_decimal(u.mV), [-70.0, -70.0])
        node = ExpSyn(2, tau=lambda shape: np.full(shape, 3.0) * u.ms)
        np.testing.assert_allclose(node.tau.to_decimal(u.ms), [3.0, 3.0])

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
