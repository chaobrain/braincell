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

from types import SimpleNamespace
import unittest

import brainunit as u
import numpy as np

from braincell.filter import metric
from braincell.mech import CVContext


class SpatialMetricTest(unittest.TestCase):
    def test_sampling_and_synapse_shaped_contexts_share_metric_surface(self) -> None:
        context = SimpleNamespace(
            branch_x=np.asarray([0.2, 0.7]),
            radius=np.asarray([1.0, 2.0]) * u.um,
            path_distance_from_soma=np.asarray([10.0, 30.0]) * u.um,
            position=np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]) * u.um,
        )

        self.assertIs(metric.branch_x(context), context.branch_x)
        self.assertIs(metric.radius(context), context.radius)
        self.assertIs(metric.path_distance_from_soma(context), context.path_distance_from_soma)
        self.assertIs(metric.position(context), context.position)

    def test_cv_context_maps_midpoint_and_midpoint_radius(self) -> None:
        context = CVContext(
            cv_id=0,
            branch_id=1,
            branch_name="dend",
            branch_type="dendrite",
            prox=0.25,
            dist=0.75,
            midpoint=0.5,
            length=10.0 * u.um,
            area=20.0 * u.um**2,
            radius_prox=1.2 * u.um,
            radius_mid=1.0 * u.um,
            radius_dist=0.8 * u.um,
            diam_mid=2.0 * u.um,
            diam_arc_mean=2.0 * u.um,
            path_distance_to_root=30.0 * u.um,
            path_distance_from_soma=20.0 * u.um,
            _local_position=np.asarray([1.0, 2.0, 3.0]) * u.um,
        )

        self.assertEqual(metric.branch_x(context), 0.5)
        self.assertEqual(metric.radius(context), 1.0 * u.um)
        self.assertEqual(metric.path_distance_from_soma(context), 20.0 * u.um)
        np.testing.assert_array_equal(metric.position(context).to_decimal(u.um), [1.0, 2.0, 3.0])

    def test_position_requires_full_point_geometry(self) -> None:
        context = SimpleNamespace()
        with self.assertRaisesRegex(TypeError, "does not expose a 3-D position"):
            metric.position(context)


if __name__ == "__main__":
    unittest.main()
