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

"""Package-level guards for :mod:`braincell.synapse`.

``__all__`` is this package's contract. A name that drops out of it is a
silent break for ``import *`` users; a name that stays in it after its
definition moves is an ``AttributeError`` the next time someone runs
``from braincell.synapse import *``. Neither shows up in a normal test run,
which is why this guard exists.
"""

import unittest

import braincell.synapse
from braincell._testing import ReExportTests


class SynapseReExportTest(ReExportTests, unittest.TestCase):
    """``braincell.synapse.__all__`` resolves, has no duplicates, and stays sorted."""

    package = braincell.synapse
    require_sorted_all = True


if __name__ == "__main__":
    unittest.main()
