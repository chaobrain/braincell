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

from braincell.mech import Channel, DensityMechanism
from braincell.mech.spec import density_class_name, density_identity, density_instance_name


class MechanismSpecIdentityTest(unittest.TestCase):
    def test_channel_uses_class_name_as_default_instance_name(self) -> None:
        mechanism = Channel("IL")

        self.assertEqual(mechanism.class_name, "IL")
        self.assertEqual(mechanism.name, "IL")
        self.assertEqual(density_class_name(mechanism), ("channel", "IL"))
        self.assertEqual(density_instance_name(mechanism), "IL")
        self.assertEqual(density_identity(mechanism), ("IL", "IL"))

    def test_channel_can_override_instance_name(self) -> None:
        mechanism = Channel("INa_HH1952", name="na_main")

        self.assertEqual(mechanism.class_name, "INa_HH1952")
        self.assertEqual(mechanism.name, "na_main")
        self.assertEqual(density_identity(mechanism), ("na_main", "INa_HH1952"))

    def test_legacy_density_mechanism_maps_to_default_identity(self) -> None:
        mechanism = DensityMechanism(channel_type="leaky")

        self.assertEqual(density_class_name(mechanism), ("channel", "leaky"))
        self.assertEqual(density_instance_name(mechanism), "leaky")
        self.assertEqual(density_identity(mechanism), ("leaky", "leaky"))

