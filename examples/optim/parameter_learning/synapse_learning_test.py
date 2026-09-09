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

"""The notebook's one-parameter demonstrations must converge."""

import unittest

import brainstate

from examples.optim.parameter_learning.synapse_learning import DT, fit_one


class SynapseLearningTest(unittest.TestCase):
    def test_single_parameter_spiking_fits(self):
        with brainstate.environ.context(dt=DT):
            for field in ("tau", "weight", "threshold"):
                with self.subTest(field=field):
                    result = fit_one(field)
                    self.assertNotEqual(result["initial_gradient"], 0.0)
                    self.assertGreaterEqual(result["spikes"], 1)
                    limit = 1.0 if field == "threshold" else 0.1
                    self.assertLess(result["final_mse"], limit * result["initial_mse"])
