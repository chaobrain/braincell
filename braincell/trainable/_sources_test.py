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

"""Tests for trainable source declarations."""

import unittest

import brainstate

from braincell.trainable import parameter, parameterized, scale


class TrainableSourceTest(unittest.TestCase):
    def test_group_and_name_validation(self) -> None:
        with self.assertRaises(ValueError):
            parameter(group_by="branch")
        with self.assertRaises(ValueError):
            parameter(name="")

    def test_existing_scale_parameter_owns_transform(self) -> None:
        root = brainstate.nn.Param(1.0)
        with self.assertRaises(ValueError):
            scale(root, transform=brainstate.nn.IdentityT())

    def test_parameterized_requires_callable(self) -> None:
        with self.assertRaises(TypeError):
            parameterized(1)
