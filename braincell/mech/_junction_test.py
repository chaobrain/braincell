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
import numpy as np

import braincell
from braincell.filter import AllRegion, RootLocation
from braincell.mech import Junction, Mechanism, Params, Point


class JunctionTest(unittest.TestCase):
    def test_default_empty_params(self) -> None:
        gap = Junction()
        self.assertEqual(len(gap.params), 0)

    def test_with_params(self) -> None:
        gap = Junction(params=Params(g=1.0))
        self.assertEqual(gap.params["g"], 1.0)

    def test_coerces_dict_params(self) -> None:
        gap = Junction(params={"g": 1.0})
        self.assertIsInstance(gap.params, Params)
        self.assertEqual(gap.params["g"], 1.0)

    def test_is_point_and_mechanism_subclass(self) -> None:
        gap = Junction()
        self.assertIsInstance(gap, Point)
        self.assertIsInstance(gap, Mechanism)

    def test_an_empty_junction_initializes_and_runs(self) -> None:
        # On main init_state raised KeyError: "Mechanism Junction has no
        # attribute 'conductance'." -- the layout asked for a parameter the
        # empty mapping never had.
        for policy in (braincell.CVPerBranch(), braincell.Lumped()):
            for junction in (Junction(), Junction(params={"g": 1.0})):
                with self.subTest(policy=type(policy).__name__, params=dict(junction.params)):
                    cell = braincell.Cell(braincell.Morphology.from_soma(), cv_policy=policy, V_init=-65.0 * u.mV)
                    cell.paint(AllRegion(), braincell.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
                    cell.place(RootLocation(0.5), junction)
                    cell.record("v", braincell.observe.state("V"))
                    cell.init_state()
                    v = cell.run(dt=0.025 * u.ms, duration=0.1 * u.ms).samples["v"].values.to_decimal(u.mV)
                    np.testing.assert_allclose(np.asarray(v), -65.0, atol=1e-5)

    def test_hashable_and_equal(self) -> None:
        a = Junction(params=Params(g=1.0))
        b = Junction(params=Params(g=1.0))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


if __name__ == "__main__":
    unittest.main()
