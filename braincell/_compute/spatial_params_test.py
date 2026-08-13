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

from braincell import Cell
from braincell._discretization import CVPerBranch
from braincell.filter import AllRegion
from braincell.mech import CVContext, Channel, Ion
from braincell.morph.branch import Branch
from braincell.morph.morphology import Morphology


def _cell(*, pop_size=1) -> Cell:
    soma = Branch.from_lengths(
        lengths=[20.0] * u.um,
        radii=[5.0, 5.0] * u.um,
        type="soma",
    )
    dend = Branch.from_lengths(
        lengths=[100.0] * u.um,
        radii=[2.0, 1.0] * u.um,
        type="basal_dendrite",
    )
    tree = Morphology.from_root(soma, name="soma")
    tree.soma.attach(dend, name="dend", parent_x=0.5)
    return Cell(
        tree,
        pop_size=pop_size,
        cv_policy=CVPerBranch(cv_per_branch=2),
    )


class SpatialDensityParameterTest(unittest.TestCase):
    def test_channel_callable_resolves_once_per_cv_and_broadcasts_population(self) -> None:
        seen: list[CVContext] = []

        def g_max(context: CVContext):
            seen.append(context)
            distance = context.path_distance_from_soma.to_decimal(u.um)
            return (0.02 + 0.00008 * distance) * (u.mS / u.cm**2)

        cell = _cell(pop_size=(2,))
        cell.paint(
            AllRegion(),
            Channel(
                "IL",
                name="distance_leak",
                g_max=g_max,
                E=-70.0 * u.mV,
            ),
        )
        cell.init_state()

        layouts = [layout for layout in cell.layouts if layout.kind == "channel:IL"]
        self.assertEqual(len(layouts), 1)
        self.assertEqual(len(seen), cell.n_cv)
        self.assertTrue(all(isinstance(context, CVContext) for context in seen))

        state = cell.get_state(layouts[0].id, "g_max")
        midpoint_ids = cell.node_tree.cv_to_mid_node_id
        actual = np.asarray(state[..., midpoint_ids].to_decimal(u.mS / u.cm**2))
        expected = np.asarray([0.02, 0.02, 0.022, 0.026])
        np.testing.assert_allclose(actual, np.broadcast_to(expected, (2, 4)))
        self.assertEqual(cell.expected_state_shape(layouts[0].id, "g_max"), (2, 7))

        _ = cell.get_state(layouts[0].id, "g_max")
        self.assertEqual(len(seen), cell.n_cv)

    def test_ion_callable_resolves_with_units(self) -> None:
        def reversal(context: CVContext):
            distance = context.path_distance_from_soma.to_decimal(u.um)
            return (50.0 - 0.1 * distance) * u.mV

        cell = _cell()
        cell.paint(
            AllRegion(),
            Ion("SodiumFixed", name="na_distance", E=reversal),
        )
        cell.init_state()

        layout = next(layout for layout in cell.layouts if layout.kind == "ion:SodiumFixed")
        state = cell.get_state(layout.id, "E")
        actual = state[..., cell.node_tree.cv_to_mid_node_id].to_decimal(u.mV)
        np.testing.assert_allclose(actual, [[50.0, 50.0, 47.5, 42.5]])

    def test_ion_callable_accepts_unitless_scalar(self) -> None:
        cell = _cell()
        cell.paint(
            AllRegion(),
            Ion("SodiumFixed", name="na", valence=lambda context: 1.0),
        )
        cell.init_state()

        layout = next(layout for layout in cell.layouts if layout.kind == "ion:SodiumFixed")
        state = cell.get_state(layout.id, "valence")
        np.testing.assert_allclose(
            state[..., cell.node_tree.cv_to_mid_node_id],
            np.ones(cell.pop_size + (cell.n_cv,)),
        )

    def test_callable_rejects_non_scalar_result_with_cv_details(self) -> None:
        cell = _cell()
        cell.paint(
            AllRegion(),
            Channel(
                "IL",
                g_max=lambda context: [0.1, 0.2] * (u.mS / u.cm**2),
                E=-70.0 * u.mV,
            ),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*CV 0.*must return a scalar",
        ):
            cell.init_state()

    def test_callable_rejects_mixed_unitful_and_unitless_results(self) -> None:
        def mixed(context: CVContext):
            if context.branch_type == "soma":
                return 0.02 * (u.mS / u.cm**2)
            return 0.03

        cell = _cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=mixed, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*expected a Quantity consistently",
        ):
            cell.init_state()

    def test_callable_rejects_incompatible_quantity_units(self) -> None:
        def incompatible(context: CVContext):
            if context.branch_type == "soma":
                return 0.02 * (u.mS / u.cm**2)
            return -70.0 * u.mV

        cell = _cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=incompatible, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            TypeError,
            r"parameter 'g_max'.*compatible with",
        ):
            cell.init_state()

    def test_callable_wraps_user_error_with_cv_details(self) -> None:
        def broken(context: CVContext):
            raise RuntimeError("bad spatial rule")

        cell = _cell()
        cell.paint(
            AllRegion(),
            Channel("IL", g_max=broken, E=-70.0 * u.mV),
        )
        with self.assertRaisesRegex(
            ValueError,
            r"parameter 'g_max'.*CV 0.*bad spatial rule",
        ):
            cell.init_state()


if __name__ == "__main__":
    unittest.main()
