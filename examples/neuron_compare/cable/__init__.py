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

# Package marker for the cable neuron-compare suite.
#
# This exists so that `cable/tests/` imports as `cable.tests` rather than as a
# bare top-level `tests` package. `channel_no_conc/tests/` is also a package
# named `tests`; without a distinguishing parent the two collide in
# `sys.modules` and pytest aborts collection with "import file mismatch" for
# the basenames they share (`_helpers.py`, `test_dispatch.py`,
# `test_experiment_schema.py`, `test_workflow_api.py`).
