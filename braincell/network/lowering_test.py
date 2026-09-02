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

from braincell._misc import validate_time_quantity
from braincell.network.lowering import lower_direct_connections


class TimeQuantityValidationTest(unittest.TestCase):
    """``dt`` and ``delay`` are validated under different, explicit rules.

    Both used to run through a single local helper that decided what to
    check by comparing the parameter *name* against ``"dt"``, so anything
    not called ``dt`` silently skipped every real check. The rules are now
    named arguments; these tests pin both halves of that split.
    """

    def test_network_dt_must_be_a_positive_scalar_time_quantity(self) -> None:
        with self.assertRaisesRegex(TypeError, "Network dt must be a time quantity"):
            lower_direct_connections({}, dt=0.1)
        with self.assertRaisesRegex(ValueError, "Network dt must be > 0"):
            lower_direct_connections({}, dt=0.0 * u.ms)

    def test_delay_may_be_a_zero_or_vector_quantity(self) -> None:
        # A delay is legitimately per-contact and zero means immediate
        # delivery, so neither rule applies to it.
        validate_time_quantity(
            [0.0, 1.0] * u.ms,
            name="delay",
            prefix="Network",
            require_scalar=False,
            require_positive=False,
        )

    def test_prefix_names_the_calling_layer(self) -> None:
        # The network layer used to report single-cell wording because it
        # borrowed Cell.run's validator wholesale.
        with self.assertRaisesRegex(TypeError, r"Network\.run\(\.\.\.\) duration"):
            validate_time_quantity(1.0, name="duration", prefix="Network.run(...)")
        with self.assertRaisesRegex(TypeError, r"Cell\.run\(\.\.\.\) duration"):
            validate_time_quantity(1.0, name="duration", prefix="Cell.run(...)")


if __name__ == "__main__":
    unittest.main()
