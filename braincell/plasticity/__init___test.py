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

"""Package-level guards for :mod:`braincell.plasticity`."""

import pathlib
import unittest

import braincell.plasticity
from braincell._testing import DocstringConformanceTests, ReExportTests, UnitSafetyGuardTests
from braincell.plasticity import stdp

_COVERED_MODULES = (stdp,)


class PlasticityReExportTest(ReExportTests, unittest.TestCase):
    """``braincell.plasticity.__all__`` re-exports every rule module and stays sorted."""

    package = braincell.plasticity
    reexport_sources = _COVERED_MODULES
    require_sorted_all = True


class PlasticityDocstringTest(DocstringConformanceTests, unittest.TestCase):
    covered_modules = _COVERED_MODULES


class PlasticityUnitSafetyTest(UnitSafetyGuardTests, unittest.TestCase):
    """No bare-number reads or ``TypeError`` fallbacks in the rules."""

    package_dir = pathlib.Path(braincell.plasticity.__file__).parent


if __name__ == "__main__":
    unittest.main()
