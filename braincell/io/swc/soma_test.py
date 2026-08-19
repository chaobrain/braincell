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

"""Tests for :mod:`braincell.io.swc.soma`.

Only the tests that call the soma predicates directly live here. The
soma *geometry* tests assert what ``SwcReader`` produces rather than what
these predicates return, so they stay in ``reader_test.py``.
"""

import unittest

from braincell.io.swc.soma import is_contour_soma, is_special_three_point_soma
from braincell.io.swc.types import _SwcRow


class SomaRuleTest(unittest.TestCase):
    def _soma_row(
        self,
        node_id: int,
        x: float,
        y: float,
        z: float = 0.0,
        radius: float = 5.0,
        parent_id: int = -1,
    ) -> _SwcRow:
        return _SwcRow(
            line_number=node_id,
            fields=tuple(),
            node_id=node_id,
            type_code=1,
            x=x,
            y=y,
            z=z,
            radius=radius,
            parent_id=parent_id,
        )

    def test_contour_rule_is_disabled(self) -> None:
        acute_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 5.0, 1.0),
            self._soma_row(3, 0.0, 2.0),
            self._soma_row(4, -5.0, 1.0),
        )
        right_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 5.0, 5.0),
            self._soma_row(3, 8.0, 1.0),
            self._soma_row(4, 10.0, 0.0),
        )
        obtuse_rows = (
            self._soma_row(1, 0.0, 0.0),
            self._soma_row(2, 3.0, 4.0),
            self._soma_row(3, 7.0, 4.0),
            self._soma_row(4, 10.0, 0.0),
        )

        self.assertFalse(is_contour_soma(acute_rows))
        self.assertFalse(is_contour_soma(right_rows))
        self.assertFalse(is_contour_soma(obtuse_rows))

    def test_special_three_point_rule_uses_center_first_topology(self) -> None:
        special_rows = (
            self._soma_row(1, 0.0, 0.0, radius=5.0, parent_id=-1),
            self._soma_row(2, -5.0, 0.0, radius=5.0, parent_id=1),
            self._soma_row(3, 0.0, 5.0, radius=5.0, parent_id=1),
        )
        chain_rows = (
            self._soma_row(1, -5.0, 0.0, radius=5.0, parent_id=-1),
            self._soma_row(2, 0.0, 0.0, radius=5.0, parent_id=1),
            self._soma_row(3, 5.0, 0.0, radius=5.0, parent_id=2),
        )
        empty_children = {1: [], 2: [], 3: []}

        is_special, ordered = is_special_three_point_soma(special_rows, empty_children)
        self.assertTrue(is_special)
        self.assertIsNotNone(ordered)
        self.assertEqual(ordered[0].node_id, 1)
        self.assertFalse(is_special_three_point_soma(chain_rows, empty_children)[0])

    def test_special_three_point_rule_requires_leaf_side_points(self) -> None:
        rows = (
            self._soma_row(1, 0.0, 0.0, radius=5.0, parent_id=-1),
            self._soma_row(2, -5.0, 0.0, radius=5.0, parent_id=1),
            self._soma_row(3, 0.0, 5.0, radius=5.0, parent_id=1),
        )

        self.assertFalse(is_special_three_point_soma(rows, {1: [2, 3], 2: [4], 3: []})[0])
        self.assertFalse(is_special_three_point_soma(rows, {1: [2, 3], 2: [], 3: [4]})[0])


if __name__ == "__main__":
    unittest.main()
