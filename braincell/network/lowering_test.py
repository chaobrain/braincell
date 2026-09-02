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

import brainunit as u

from braincell.network.lowering import lower_direct_connections


class TimeQuantityValidationTest(unittest.TestCase):
    """Lowering rejects a ``dt`` the shared validator refuses.

    ``dt`` and ``delay`` used to run through a single local helper that
    decided what to check by comparing the parameter *name* against
    ``"dt"``, so anything not called ``dt`` silently skipped every real
    check. The validator's own rules are pinned in ``_misc_test.py``; what
    is pinned here is that this entry point applies them.
    """

    def test_network_dt_must_be_a_positive_scalar_time_quantity(self) -> None:
        with self.assertRaisesRegex(TypeError, "Network dt must be a time quantity"):
            lower_direct_connections({}, dt=0.1)
        with self.assertRaisesRegex(ValueError, "Network dt must be > 0"):
            lower_direct_connections({}, dt=0.0 * u.ms)


if __name__ == "__main__":
    unittest.main()
