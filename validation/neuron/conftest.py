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

import brainstate
import pytest


@pytest.fixture(autouse=True, scope="package")
def _double_precision():
    """Run every test under validation/neuron in float64, and no other test.

    NEURON integrates in double precision (see ``use_double_precision``). The
    suites still set float64 when imported, but the root ``conftest.py``
    restores the session's precision after collection, so that a braincell
    test collected in the same session keeps its own. Package scope covers
    ``setUpClass`` and module fixtures, which run the simulations.
    """
    with brainstate.environ.context(precision=64):
        yield
