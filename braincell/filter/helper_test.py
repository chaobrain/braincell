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

"""Tests for :mod:`braincell.filter.helper`.

The interval algebra in this module is currently reached almost entirely
through :class:`RegionSetOp` in ``region_test.py``; only the export surface
is asserted here directly.
"""

import unittest

from braincell.filter import helper as helper_mod


class HelperModuleAllTest(unittest.TestCase):
    def test_helper_module_declares_all(self) -> None:
        self.assertIn("branch_slice_intervals", helper_mod.__all__)
        self.assertIn("union_region_intervals", helper_mod.__all__)
        self.assertIn("complement_region_intervals", helper_mod.__all__)


if __name__ == "__main__":
    unittest.main()
